"""
train/labeling/merge_datasets.py — join two extractions of the SAME positions

WHY
───
`data/` holds pairs of datasets that are the same position set extracted
twice, each extraction keeping the half the other discarded. The measured
case:

    extra_quiet_raw_wdl   47,495,908 rows   fen, result          (no eval)
    extra-quiet-n5k_sf    47,495,907 rows   fen, cp, wdl         (no result)

Looking 3,000 FENs from the first up in the ENTIRE second finds 3000/3000 =
100.0%. They are the same human positions. Joining them on `fen` yields
`cp` + `result` together — which is what `lambda` needs to do anything at
all (it blends the two, so it is inert when a dataset has only one).

This nearly doubles the lambda-active corpus, 50.7M -> 98.2M rows, without
running an engine over anything.

HOW THE JOIN IS DONE, and why not a dict
────────────────────────────────────────
A Python dict of 47.5M FEN strings costs several GB and is slow to build.
Instead the key side is reduced to two numpy arrays — a 64-bit hash of each
FEN and its value — sorted once, then probed with `np.searchsorted`. That is
~570 MB and vectorised.

Hash collisions would silently attach the wrong label, so:
  * a 64-bit hash over 47.5M keys has an expected collision count of
    n^2 / 2^65 ~= 6e-5, i.e. essentially zero, AND
  * VERIFY_SAMPLE rows are re-checked by comparing the actual FEN strings
    after the join. Any mismatch aborts the run.

Never trust the hash alone. The verification is what makes this safe.

OUTPUT
──────
One folder holding the canonical shape `fen, cp, result` — `wdl` is NOT
written, because it is exactly sigmoid(cp/320) and the trainer derives it
(CLAUDE.md rule 10: a stored wdl next to a cp is redundant and can rot).

DRY RUN IS THE DEFAULT.

Usage:
    python train/labeling/merge_datasets.py       # dry run: verify + report
    (set DRY_RUN = False, re-run to write)
"""
from __future__ import annotations

import glob
import hashlib
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ═══════════════════════════════ CONFIGURATION ═══════════════════════════════
DRY_RUN = True            # True = verify and report only, write nothing
DATA_DIR = "data"

# One merge job per entry.
#   key    : dataset holding the column(s) to be attached (read fully into RAM
#            as hash + value arrays)
#   key_col: which column to carry over from `key`
#   base   : dataset that is streamed; supplies everything else
#   out    : output folder name
#   keep   : columns to write, in order. `wdl` is deliberately excluded — it
#            is derivable from cp and the trainer recomputes it.
JOBS = [
    {
        "name":    "extra_quiet",
        "key":     "extra_quiet_raw_wdl",     # fen, result
        "key_col": "result",
        "base":    "extra-quiet-n5k_sf",      # fen, cp, wdl
        "out":     "extraquiet_cp_sf5k_res_filter",      # -> fen, cp, result
        "keep":    ["fen", "cp", "result"],
    },
]

ROWS_PER_OUT_FILE = 2_000_000   # rows per output parquet shard
READ_BATCH        = 200_000     # streaming batch size for the base side
VERIFY_SAMPLE     = 20_000      # rows whose FEN is re-compared after the join
COMPRESSION       = "snappy"

# A single job can also be described entirely from the command line, without
# editing JOBS — leave these empty to use the JOBS list above.
JOB_KEY     = ""    # dataset folder holding the column to attach
JOB_KEY_COL = ""    # which column to carry over from JOB_KEY
JOB_BASE    = ""    # dataset folder that is streamed
JOB_OUT     = ""    # output folder name (under DATA_DIR)
JOB_KEEP    = ["fen", "cp", "result"]   # columns to write, in order
# ═════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════ COMMAND LINE ════════════════════════════════
# The block above is the interface: `python train/labeling/merge_datasets.py`
# runs exactly what it says (a DRY RUN, by design). Every constant is also a
# flag that overrides it — `--no-dry-run` to actually write, `--key/--base/
# --out` to describe a one-off job without editing JOBS, `--show-config` to
# print the resolved settings and stop. See utils/cliconf.py, CLAUDE.md rule 8.
# ═════════════════════════════════════════════════════════════════════════════
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "utils"))
from cliconf import force_utf8_stdio, override_from_cli  # noqa: E402

force_utf8_stdio()

CLI = [
    ("DRY_RUN",           "--dry-run",      bool, "verify and report only, write nothing"),
    ("DATA_DIR",          "--data-dir",     str,  "root holding one folder per dataset"),
    ("ROWS_PER_OUT_FILE", "--rows-per-file", int, "rows per output parquet shard"),
    ("READ_BATCH",        "--read-batch",   int,  "streaming batch size for the base side"),
    ("VERIFY_SAMPLE",     "--verify-sample", int, "rows whose FEN is re-compared after the join"),
    ("COMPRESSION",       "--compression",  str,  "parquet compression codec"),
    ("JOB_KEY",           "--key",          str,  "one-off job: dataset holding the column to attach"),
    ("JOB_KEY_COL",       "--key-col",      str,  "one-off job: column carried over from --key"),
    ("JOB_BASE",          "--base",         str,  "one-off job: dataset that is streamed"),
    ("JOB_OUT",           "--out",          str,  "one-off job: output folder name"),
    ("JOB_KEEP",          "--keep",         list, "one-off job: column to write (repeatable, in order)"),
]


def hash_fens(fens: list[str]) -> np.ndarray:
    """Stable 64-bit hash per FEN.

    Uses blake2b, NOT Python's builtin hash(): builtin str hashing is salted
    per process (PYTHONHASHSEED), so a hash built in one run would not match
    one built in another. This has to be reproducible."""
    out = np.empty(len(fens), dtype=np.uint64)
    for i, f in enumerate(fens):
        out[i] = int.from_bytes(
            hashlib.blake2b(f.encode("utf-8"), digest_size=8).digest(), "little")
    return out


def load_key_side(folder: str, key_col: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read the key dataset into (sorted hashes, values, fens-by-sorted-order)."""
    files = sorted(glob.glob(os.path.join(folder, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet in {folder}")
    hashes, values, fens = [], [], []
    for i, f in enumerate(files):
        for b in pq.ParquetFile(f).iter_batches(batch_size=READ_BATCH,
                                                columns=["fen", key_col]):
            fl = b.column("fen").to_pylist()
            hashes.append(hash_fens(fl))
            values.append(np.asarray(b.column(key_col).to_pylist(), dtype=np.float32))
            fens.extend(fl)
        if (i + 1) % 5 == 0 or i + 1 == len(files):
            print(f"   key side: {i+1}/{len(files)} files, {sum(len(h) for h in hashes):,} rows")
    h = np.concatenate(hashes)
    v = np.concatenate(values)
    order = np.argsort(h, kind="stable")
    h, v = h[order], v[order]
    fens_sorted = [fens[i] for i in order]
    dup = int((np.diff(h) == 0).sum())
    if dup:
        print(f"   NOTE: {dup:,} duplicate hashes on the key side "
              f"(duplicate positions, or — far less likely — collisions). "
              f"searchsorted takes the first; the FEN verification below "
              f"is what catches a genuine collision.")
    return h, v, fens_sorted


def run_job(job: dict) -> None:
    key_dir = os.path.join(DATA_DIR, job["key"])
    base_dir = os.path.join(DATA_DIR, job["base"])
    out_dir = os.path.join(DATA_DIR, job["out"])
    print(f"\n== {job['name']}: {job['base']} + {job['key']}.{job['key_col']} -> {job['out']}")

    print("   loading key side into memory ...")
    kh, kv, kfens = load_key_side(key_dir, job["key_col"])
    print(f"   key side ready: {len(kh):,} rows")

    files = sorted(glob.glob(os.path.join(base_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet in {base_dir}")

    if not DRY_RUN:
        os.makedirs(out_dir, exist_ok=True)

    matched = total = 0
    verified = mismatched = 0
    buf: dict[str, list] = {c: [] for c in job["keep"]}
    buf_rows = shard = 0

    def flush():
        nonlocal buf, buf_rows, shard
        if not buf_rows:
            return
        if not DRY_RUN:
            tbl = pa.table({c: buf[c] for c in job["keep"]})
            pq.write_table(tbl, os.path.join(out_dir, f"merged_{shard:05d}.parquet"),
                           compression=COMPRESSION)
        shard += 1
        buf = {c: [] for c in job["keep"]}
        buf_rows = 0

    for fi, f in enumerate(files):
        for b in pq.ParquetFile(f).iter_batches(batch_size=READ_BATCH):
            fl = b.column("fen").to_pylist()
            total += len(fl)
            bh = hash_fens(fl)
            idx = np.searchsorted(kh, bh)
            idx = np.clip(idx, 0, len(kh) - 1)
            hit = kh[idx] == bh
            matched += int(hit.sum())

            # Verify by comparing the REAL FEN strings, not just the hash.
            if verified < VERIFY_SAMPLE:
                for j in np.flatnonzero(hit)[:VERIFY_SAMPLE - verified]:
                    verified += 1
                    if kfens[idx[j]] != fl[j]:
                        mismatched += 1
                        if mismatched <= 3:
                            print(f"   HASH COLLISION: {fl[j]!r} != {kfens[idx[j]]!r}")

            sel = np.flatnonzero(hit)
            if sel.size:
                cols = {c: b.column(c).to_pylist() for c in b.schema.names if c in job["keep"]}
                for c in job["keep"]:
                    if c == job["key_col"]:
                        buf[c].extend(kv[idx[sel]].tolist())
                    else:
                        src = cols.get(c)
                        buf[c].extend([src[k] for k in sel] if src else [None] * sel.size)
                buf_rows += int(sel.size)
                if buf_rows >= ROWS_PER_OUT_FILE:
                    flush()
        if (fi + 1) % 100 == 0 or fi + 1 == len(files):
            print(f"   base side: {fi+1}/{len(files)} files, matched {matched:,}/{total:,}")
    flush()

    if mismatched:
        raise SystemExit(f"ABORT: {mismatched} hash collisions found in {verified} "
                         f"verified rows — the join would attach wrong labels.")
    print(f"   matched {matched:,}/{total:,} ({100*matched/max(1,total):.2f}%)")
    print(f"   FEN-verified {verified:,} joined rows, 0 mismatches")
    print(f"   {'would write' if DRY_RUN else 'wrote'} {shard} shard(s) to {out_dir}")


def main() -> int:
    print("DRY RUN — nothing will be written" if DRY_RUN else "WRITING")
    # A fully specified --key/--key-col/--base/--out replaces the JOBS list
    # for this run; anything less falls back to it (a half-described job is a
    # typo, not a request, so it is refused rather than silently ignored).
    cli_bits = [JOB_KEY, JOB_KEY_COL, JOB_BASE, JOB_OUT]
    jobs = JOBS
    if any(cli_bits):
        if not all(cli_bits):
            raise SystemExit("ABORT: --key, --key-col, --base and --out must be given together "
                             "(or none of them, to use the JOBS block).")
        jobs = [{"name": JOB_OUT, "key": JOB_KEY, "key_col": JOB_KEY_COL,
                 "base": JOB_BASE, "out": JOB_OUT, "keep": list(JOB_KEEP)}]
    for job in jobs:
        run_job(job)
    if DRY_RUN:
        print("\nSet DRY_RUN = False to write the merged dataset.")
    return 0


if __name__ == "__main__":
    override_from_cli(globals(), CLI, description=__doc__, prog="merge_datasets.py")
    sys.exit(main())
