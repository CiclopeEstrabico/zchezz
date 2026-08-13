"""
train/labeling/normalize_columns.py — force every parquet dataset onto the
canonical column vocabulary of CLAUDE.md rule 10.

WHY
───
Rule 10 defines exactly three names, and `wdl` is a PURE FUNCTION of `cp`:

    result  real game outcome, white-relative   0.0 / 0.5 / 1.0
    cp      evaluation in centipawns, white-relative
    wdl     sigmoid(cp / 320)                   <- derivable, never independent

Storing `cp` and `wdl` side by side is therefore redundant, and redundancy
rots. It measurably HAD rotted:

  * `extraquiet_cp_sf5k_res_filter` collapsed duplicate FENs by averaging
    `cp` and `wdl` INDEPENDENTLY. Because mean(sigmoid(x)) != sigmoid(mean(x)),
    13.8% of its rows carry a `wdl` that cannot be reproduced from its own
    `cp`. The stored `wdl` is a mean-of-sigmoids; nothing downstream can know
    that.
  * The same averaging turned 0.3% of its `result` values into empirical win
    RATES (2/3, 5/6, 3/8, ...) rather than outcomes. That is a fourth
    quantity the vocabulary does not have.
  * `lichess_cp_sfdb_filter` holds `wdl` with NO `cp` at all — the one case
    where `wdl` is not redundant but is still off-convention.

THE RULE THIS SCRIPT ENFORCES
─────────────────────────────
Every dataset ends up with `cp` and no `wdl`. `cp` is the primitive; the
trainer derives `wdl` from it at load time (train_nnue.py already prefers
`cp` and recomputes `wdl` whenever `cp` is present, so this changes nothing
about what a training run sees — it only removes the rotted copy).

  cp + wdl   -> drop `wdl`                        (redundant by definition)
  wdl only   -> cp = 320 * logit(wdl), drop `wdl` (exact inverse; the only
                lossy part is saturation, see CP_CLAMP below)
  cp only    -> untouched
  result     -> snapped to {0.0, 0.5, 1.0}        (it is an OUTCOME, not a rate)
  id         -> dropped when the column is empty/null in EVERY row of the
                dataset; kept verbatim when it actually carries something

RESHARDING
──────────
A second, independent defect this script also repairs: a dataset stored as
ONE enormous parquet cannot be sampled. train_nnue.py's `pct_mode='shards'`
drops whole FILES, and its "never starve a source" guard keeps the single
file no matter how small `pct` is — so a 40.1M-row one-file dataset at
pct=0.02 silently trains on 40.1M rows per epoch instead of 800k, 50x its
intended weight in the mix. The per-file encode cache also tries to build
that whole file into one tensor (~5-8 GB) and torch.save it.

So any input file longer than MAX_ROWS_PER_FILE is split into parts of
ROWS_PER_SHARD. Files already below the limit keep their exact identity —
one input file, one output file, same name — because their count IS the
sampling granularity and changing it would silently change every dataset's
effective proportion in the mix.

SAFETY
──────
DRY_RUN is the default and reports every planned change without writing.
When writing, each file is built as `<name>.tmp` and only then swapped over
the original, so an interrupted run cannot leave a half-written parquet in
place of good data.

Usage:
    python train/labeling/normalize_columns.py     # dry run: report only
    (set DRY_RUN = False, re-run to apply)
"""
from __future__ import annotations

import glob
import math
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ═══════════════════════════════ CONFIGURATION ═══════════════════════════════
DRY_RUN = False             # True = analyse and report only, write nothing
DRY_RUN_MAX_FILES = 8       # in a dry run, actually read at most this many files
                            # per dataset. A full dry pass over 173M rows costs
                            # ~45 min and buys nothing: the PLAN (which columns
                            # change) comes from the schema, and the per-row
                            # statistics converge long before file 8. Set to 0
                            # to force a full, exhaustive dry pass.
DATA_DIR = "data"           # root holding one folder per dataset

ONLY = ['extraquiet_cp_sf60k_filter', 'extraquiet_cp_sfd12_endgames_filter', 'extraquiet_cp_sfd14_endgames_filter', 'lichess_cp_sf400k_filter', 'lichess_cp_sf_filter', 'lichess_cp_sfdb_filter', 'selfplay-lichess_cp_sf500k_res_filter_endgames', 'selfplay_cp_sf100k_res_endgames_filter_data20260414', 'selfplay_cp_sf50k_res_filter_data20260404', 'selfplay_cp_sf50k_res_filter_data20260410', 'selfplay_cp_zchezz_res_filter_data20260401', 'selfplay_cp_zchezz_res_filter_data20260404', 'selfplay_cp_zchezz_res_filter_data20260410', 'selfplay_raw_res_data20260401', 'selfplay_raw_res_data20260404', 'selfplay_raw_res_data20260410', 'selfplay_raw_res_endgames_data20260414', 'synthetic_endgame_cp_sf_filter_data20260413', 'synthetic_endgame_cp_sf_filter_data20260414', 'viriformat_cp_virichess_filter_data20260112', 'viriformat_cp_virichess_filter_data20260312']                   # [] = every dataset folder; else a list of folder
                            # names to restrict the run to

CP_TO_WDL_T = 320.0         # sigmoid temperature. MUST match nnue.c's
                            # `_nnL3B * 320.0f` output scale and
                            # train_nnue.py's CP_TO_WDL_T. Changing one
                            # without the others silently rescales everything.

CP_CLAMP = 20000.0          # |cp| ceiling when inverting wdl -> cp. wdl
                            # saturates: sigmoid(x) underflows to 0.0 in
                            # float64 around x = -745, so a stored wdl of
                            # exactly 0.0 or 1.0 carries no finite cp. Those
                            # rows are mate scores; +/-20000 is the mate
                            # magnitude the engine itself uses.

SNAP_RESULT = True          # snap `result` to {0.0, 0.5, 1.0}. Ties round to
                            # 0.5 (the conservative, least-committal outcome).
DROP_EMPTY_ID = True        # drop an `id` column that is empty in every row

MAX_ROWS_PER_FILE = 4_000_000   # any input file longer than this is split
ROWS_PER_SHARD    = 2_000_000   # ... into parts of this many rows
READ_BATCH        = 200_000     # streaming read size, bounds peak memory
COMPRESSION       = "snappy"
# ═════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════ COMMAND LINE ════════════════════════════════
# The CONFIGURATION block above is the primary interface: a bare
# `python train/labeling/normalize_columns.py` does exactly what it says.
# Every constant is ALSO a flag that overrides it for a one-off run, and
# `--show-config` prints the resolved settings and exits without touching a
# file. See utils/cliconf.py and CLAUDE.md rule 8.
# ═════════════════════════════════════════════════════════════════════════════
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "utils"))
from cliconf import force_utf8_stdio, override_from_cli  # noqa: E402

force_utf8_stdio()

CLI = [
    ("DRY_RUN",           "--dry-run",        bool, "analyse and report only, write nothing"),
    ("DRY_RUN_MAX_FILES", "--dry-run-files",  int,  "files per dataset read in a dry run (0 = all)"),
    ("DATA_DIR",          "--data-dir",       str,  "root holding one folder per dataset"),
    ("ONLY",              "--only",           list, "restrict to this dataset folder (repeatable; empty = all)"),
    ("CP_TO_WDL_T",       "--cp-to-wdl-t",    float,"sigmoid temperature; must match nnue.c and train_nnue.py"),
    ("CP_CLAMP",          "--cp-clamp",       float,"|cp| ceiling when inverting a saturated wdl"),
    ("SNAP_RESULT",       "--snap-result",    bool, "snap `result` to {0.0, 0.5, 1.0}"),
    ("DROP_EMPTY_ID",     "--drop-empty-id",  bool, "drop an `id` column that is empty in every row"),
    ("MAX_ROWS_PER_FILE", "--max-rows-per-file", int, "input files longer than this are split"),
    ("ROWS_PER_SHARD",    "--rows-per-shard", int,  "rows per output shard when splitting"),
    ("READ_BATCH",        "--read-batch",     int,  "streaming read size (bounds peak memory)"),
    ("COMPRESSION",       "--compression",    str,  "parquet compression codec"),
]

CANON = ("fen", "cp", "wdl", "result")


def wdl_to_cp(wdl: np.ndarray) -> np.ndarray:
    """Exact inverse of sigmoid(cp/T), clamped where wdl has saturated.

    logit() blows up to +/-inf at 0 and 1, which is not a bug in the data —
    it is what a mate score looks like after the sigmoid. Clamping to
    CP_CLAMP keeps those rows usable instead of poisoning the column
    with inf/nan."""
    w = np.clip(wdl.astype(np.float64), 1e-15, 1.0 - 1e-15)
    cp = CP_TO_WDL_T * np.log(w / (1.0 - w))
    return np.clip(cp, -CP_CLAMP, CP_CLAMP)


# Legacy PGN result strings. CLAUDE.md rule 10 says `result` is written as a
# NUMBER in new datasets and the PGN strings are only ACCEPTED when reading —
# `miscelaneous_cp_sf1M_res_filter` still stores them as text, so this is
# where that legacy form is converted once and for all.
_RESULT_STR = {"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0,
               "1": 1.0, "1/2": 0.5, "0": 0.0,
               "0.5": 0.5, "1.0": 1.0, "0.0": 0.0,
               "*": np.nan, "": np.nan}


def result_to_float(vals: list, stats: dict) -> np.ndarray:
    """Coerce a `result` column to float, accepting the legacy PGN strings.

    An unrecognised value becomes NaN and is COUNTED rather than guessed at —
    silently mapping an unknown token to 0.5 would invent a draw."""
    out = np.empty(len(vals), dtype=np.float64)
    for i, v in enumerate(vals):
        if v is None:
            out[i] = np.nan
        elif isinstance(v, str):
            s = v.strip()
            if s in _RESULT_STR:
                out[i] = _RESULT_STR[s]
                stats["result_str"] += 1
            else:
                out[i] = np.nan
                stats["result_bad"] += 1
        else:
            out[i] = float(v)
    return out


def snap_result(r: np.ndarray) -> tuple[np.ndarray, int]:
    """Snap to the only three legal outcomes. Returns (snapped, n_changed).

    Averaging duplicate FENs produced values like 2/3 and 5/6 — empirical
    win rates. A rate is not an outcome, and rule 10 has no name for it."""
    out = r.astype(np.float64, copy=True)
    finite = np.isfinite(out)
    v = out[finite]
    # nearest of {0, 0.5, 1}; a tie (0.25, 0.75) resolves to 0.5 because the
    # comparison chain below tests the 0.5 band inclusively on both sides.
    snapped = np.where(v < 0.25, 0.0, np.where(v > 0.75, 1.0, 0.5))
    changed = int((snapped != v).sum())
    out[finite] = snapped
    return out, changed


def col_is_empty(tbl_col) -> bool:
    """True when every value is null or the empty string."""
    vals = tbl_col.to_pylist()
    return all(v is None or v == "" for v in vals)


def scan_dataset(folder: str) -> dict:
    """Decide, from the data itself, what this folder needs. Nothing here is
    inferred from the folder NAME — names have been wrong before."""
    files = sorted(glob.glob(os.path.join(folder, "**", "*.parquet"), recursive=True))
    if not files:
        return {}
    cols = pq.ParquetFile(files[0]).schema_arrow.names
    plan = {
        "files": files,
        "cols": cols,
        "drop_wdl": "wdl" in cols and "cp" in cols,
        "invert_wdl": "wdl" in cols and "cp" not in cols,
        "snap": SNAP_RESULT and "result" in cols,
        "drop_id": False,
    }
    if DROP_EMPTY_ID and "id" in cols:
        # Sample the first row group of the first AND last file — a column
        # that is populated only late (or only early) must not be dropped.
        empty = True
        for f in (files[0], files[-1]):
            t = pq.ParquetFile(f).read_row_group(0, columns=["id"])
            if not col_is_empty(t.column("id")):
                empty = False
                break
        plan["drop_id"] = empty
    plan["big"] = [f for f in files
                   if pq.ParquetFile(f).metadata.num_rows > MAX_ROWS_PER_FILE]
    return plan


def transform_batch(b: pa.RecordBatch, plan: dict, stats: dict) -> dict:
    """Apply the column rules to one read batch, returning plain lists."""
    d = {n: b.column(n).to_pylist() for n in b.schema.names}

    if plan["invert_wdl"]:
        w = np.asarray(d.pop("wdl"), dtype=np.float64)
        d["cp"] = wdl_to_cp(w).tolist()
        stats["inverted"] += len(w)
    elif plan["drop_wdl"]:
        # Before discarding, measure how far the stored wdl had drifted from
        # its own cp. A nonzero count here is the rot this script exists for,
        # and it belongs in the report rather than being silently deleted.
        w = np.asarray(d.pop("wdl"), dtype=np.float64)
        cp = np.asarray(d["cp"], dtype=np.float64)
        m = np.isfinite(w) & np.isfinite(cp)
        if m.any():
            err = np.abs(w[m] - 1.0 / (1.0 + np.exp(-cp[m] / CP_TO_WDL_T)))
            stats["wdl_checked"] += int(m.sum())
            stats["wdl_rotted"] += int((err > 1e-9).sum())
            stats["wdl_maxerr"] = max(stats["wdl_maxerr"], float(err.max()))

    if plan["snap"]:
        r = result_to_float(d["result"], stats)
        r, n = snap_result(r)
        stats["snapped"] += n
        d["result"] = r.tolist()

    if plan["drop_id"] and "id" in d:
        d.pop("id")

    return d


def out_schema_names(plan: dict) -> list[str]:
    cols = [c for c in plan["cols"] if c != "wdl"]
    if plan["invert_wdl"] and "cp" not in cols:
        cols.insert(1, "cp")
    if plan["drop_id"] and "id" in cols:
        cols.remove("id")
    return cols


def process_file(src: str, plan: dict, stats: dict) -> int:
    """Rewrite one file. Returns the number of output files produced.

    Split only when the input is oversized: for every normal file this is a
    1:1 rewrite keeping the original name, so the dataset's file COUNT — and
    therefore its shard-sampling granularity — is unchanged."""
    pf = pq.ParquetFile(src)
    n_rows = pf.metadata.num_rows
    names = out_schema_names(plan)
    split = n_rows > MAX_ROWS_PER_FILE
    base = src[:-len(".parquet")]

    buf: dict[str, list] = {c: [] for c in names}
    buf_rows = 0
    part = 0
    written: list[str] = []

    def write_out(dst: str) -> None:
        """Write the buffer to `dst` via a .tmp + atomic rename, so an
        interrupted run can never leave a truncated parquet where good data
        used to be."""
        nonlocal buf, buf_rows, part
        if not DRY_RUN:
            tmp = dst + ".tmp"
            pq.write_table(pa.table({c: buf[c] for c in names}), tmp,
                           compression=COMPRESSION)
            os.replace(tmp, dst)
        written.append(dst)
        part += 1
        buf = {c: [] for c in names}
        buf_rows = 0

    try:
        for b in pf.iter_batches(batch_size=READ_BATCH):
            d = transform_batch(b, plan, stats)
            for c in names:
                buf[c].extend(d[c])
            buf_rows += b.num_rows
            # Only the SPLIT path may write mid-stream: its parts have new
            # names, so writing them while `src` is still being read is safe.
            # The 1:1 path writes over `src` itself and therefore must wait
            # until the whole file has been read AND the reader closed —
            # flushing it per batch would leave `src` holding just the last
            # batch, and on Windows os.replace() over a file that is still
            # open fails outright.
            if split and buf_rows >= ROWS_PER_SHARD:
                write_out(f"{base}_part{part:04d}.parquet")
    finally:
        pf.close()          # release the handle before touching `src` on disk

    if buf_rows:
        write_out(f"{base}_part{part:04d}.parquet" if split else src)

    if split and not DRY_RUN and os.path.exists(src):
        os.remove(src)      # fully superseded by its _partNNNN pieces
    stats["rows"] += n_rows
    return len(written)


def run_folder(name: str) -> None:
    folder = os.path.join(DATA_DIR, name)
    plan = scan_dataset(folder)
    if not plan:
        return

    actions = []
    if plan["invert_wdl"]:
        actions.append("wdl -> cp (inverse sigmoid)")
    if plan["drop_wdl"]:
        actions.append("drop wdl")
    if plan["snap"]:
        actions.append("snap result")
    if plan["drop_id"]:
        actions.append("drop empty id")
    if plan["big"]:
        actions.append(f"reshard {len(plan['big'])} oversized file(s)")
    if not actions:
        print(f"  {name:56s} already canonical ({plan['cols']})")
        return

    print(f"  {name:56s} {', '.join(actions)}")
    stats = {"rows": 0, "snapped": 0, "inverted": 0, "result_str": 0,
             "result_bad": 0, "wdl_checked": 0, "wdl_rotted": 0, "wdl_maxerr": 0.0}
    n_out = 0
    todo = plan["files"]
    if DRY_RUN and DRY_RUN_MAX_FILES and len(todo) > DRY_RUN_MAX_FILES:
        todo = todo[:DRY_RUN_MAX_FILES]
        print(f"      (dry run: sampling {len(todo)} of {len(plan['files'])} files)")
    for i, f in enumerate(todo):
        n_out += process_file(f, plan, stats)
        if (i + 1) % 200 == 0:
            print(f"      {i+1}/{len(plan['files'])} files, {stats['rows']:,} rows")

    print(f"      {stats['rows']:,} rows -> {n_out} file(s)"
          f" (was {len(plan['files'])})")
    if stats["wdl_checked"]:
        pct = 100.0 * stats["wdl_rotted"] / stats["wdl_checked"]
        print(f"      dropped wdl: {stats['wdl_rotted']:,}/{stats['wdl_checked']:,}"
              f" ({pct:.2f}%) disagreed with cp, max err {stats['wdl_maxerr']:.2e}")
    if stats["inverted"]:
        print(f"      inverted {stats['inverted']:,} wdl values to cp")
    if stats["result_str"]:
        print(f"      converted {stats['result_str']:,} legacy PGN result strings to numbers")
    if stats["result_bad"]:
        print(f"      WARNING: {stats['result_bad']:,} unrecognised result values -> NaN")
    if stats["snapped"]:
        print(f"      snapped {stats['snapped']:,} result values to {{0, 0.5, 1}}")


def main() -> int:
    print("DRY RUN — nothing will be written\n" if DRY_RUN else "WRITING\n")
    names = ONLY or sorted(d for d in os.listdir(DATA_DIR)
                           if os.path.isdir(os.path.join(DATA_DIR, d)))
    for n in names:
        run_folder(n)
    if DRY_RUN:
        print("\nSet DRY_RUN = False to apply.")
    return 0


if __name__ == "__main__":
    override_from_cli(globals(), CLI, description=__doc__, prog="normalize_columns.py")
    sys.exit(main())
