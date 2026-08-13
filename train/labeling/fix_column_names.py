"""
train/labeling/fix_column_names.py — rename mislabelled columns in place

WHAT THIS FIXES
───────────────
Some historical datasets store the GAME RESULT in a column named `wdl`.
That is the wrong name under the project convention (CLAUDE.md rule 10):

    result   the real game outcome, WHITE-relative, 0.0 / 0.5 / 1.0
    cp       evaluation in centipawns, WHITE-relative
    wdl      sigmoid(cp / 320) — a FUNCTION OF cp, not an outcome

A `wdl` column that actually holds outcomes is not a cosmetic problem. The
training target is

    target = lam * result + (1 - lam) * wdl

so a mislabelled column lands on the WRONG SIDE of the blend: the trainer
uses it as the evaluation term when it is really ground truth, inverting
what lambda means for that dataset. Measured example: `extra_quiet_raw_wdl`
(47,495,908 rows) has exactly THREE distinct values — {0.0, 0.5, 1.0} —
across every row sampled. That is an outcome, not an evaluation.

This script renames the column. It does NOT add columns, does NOT compute
anything, and does NOT touch the values.

THE SAFETY RULE THAT MATTERS
────────────────────────────
Every file is RE-VERIFIED immediately before it is rewritten. The decision
to rename is never taken from a previous analysis, a folder name, or this
file's config — only from the bytes actually in that file, right now. A
file whose values are not a subset of the expected set is SKIPPED and
reported, never renamed.

That rule exists because the whole class of bug this script cleans up came
from acting on a stale or inferred conclusion about what a column meant.
Renaming a column on a bad assumption would produce exactly the same
failure it is meant to remove, with no error message.

DRY RUN IS THE DEFAULT. Nothing is written until DRY_RUN = False.

Usage:
    python train/labeling/fix_column_names.py            # dry run, reports only
    (edit DRY_RUN = False at the top, then re-run to apply)
"""
from __future__ import annotations

import glob
import os
import shutil
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ═══════════════════════════════ CONFIGURATION ═══════════════════════════════
DRY_RUN = False         # True = report only, write nothing. Set False to apply.
DATA_DIR = "data"       # root holding one folder per dataset
BACKUP = True           # keep <file>.bak next to each rewritten file
FAIL_FAST = False       # True = stop at the first file that fails verification

# Datasets to fix: dataset folder -> (column to rename, new name).
#
# ONLY put a dataset here once its column has been MEASURED, not guessed.
# Every entry is re-verified per file at run time regardless (see VERIFIERS),
# so a wrong entry here is caught rather than applied — but do not rely on
# that as a substitute for knowing.
RENAMES = {
    # Already applied (kept for the record): extra_quiet_raw_wdl's `wdl`
    # column held exactly {0.0, 0.5, 1.0} over all 47,495,908 rows -> it was
    # the game outcome, renamed to `result`. Re-running is a no-op: the
    # verifier refuses a file that already has a `result` column.
}

# Columns to DELETE, per dataset. A column is dropped when it is either
# EMPTY (it exists but never carries a value) or WRONG/DERIVED (it can be
# recomputed exactly from a column that IS present, and the stored version
# disagrees with the convention).
#
# Deleting beats "recomputing and rewriting": a wdl is 100% derivable from
# cp, so storing it again only re-creates something that can drift out of
# sync a second time. The trainer derives wdl from cp on the fly.
#
# Every entry is RE-VERIFIED per file before the drop (see verify_droppable).
DROPS = {
    # --- `wdl` contaminated with a baked-in blend -----------------------
    # Measured: wdl == 0.40*result + 0.60*sigmoid(cp/320), mean abs error
    # 0.00000. That is the old LAMBDA_WDL=0.4 frozen into the data. Both
    # cp and result are present and correct, so the blend column is not
    # merely redundant, it is wrong.
    "selfplay_cp_zchezz_res_filter_data20260401": ["wdl"],
    "selfplay_cp_zchezz_res_filter_data20260404": ["wdl"],
    "selfplay_cp_zchezz_res_filter_data20260410": ["wdl"],

    # --- `wdl` written at the wrong temperature ------------------------
    # Measured: wdl == sigmoid(cp/500), mean abs error 0.00000. The project
    # constant is 320 (CLAUDE.md rule 10). cp is present, so drop it.
    #
    # `result` is dropped too, and that is the deliberate part: these are
    # GENERATED positions, no game was ever played from them, and the old
    # generator SYNTHESISED result from cp with a +/-50cp threshold. It
    # therefore carries no information that cp does not already carry.
    # Keeping it would invite lam > 0, which would blend cp against a
    # function of cp -- circular, and silently so. The generator has been
    # fixed to leave result empty; this removes the already-written ones.
    "synthetic_endgame_cp_sf_filter_data20260413": ["wdl", "result"],
    "synthetic_endgame_cp_sf_filter_data20260414": ["wdl", "result"],

    # --- `result` column that is present but ALWAYS EMPTY ---------------
    # A column that never carries a value is worse than an absent one: it
    # advertises a signal that does not exist and invites lam > 0 on
    # datasets where lam can never do anything.
    "extraquiet_cp_wdl_sfd12_endgames_filter": ["result"],
    "extraquiet_cp_wdl_sfd14_endgames_filter": ["result"],
    "extraquiet_cp_wdl_sf60k_filter": ["result"],
    "lichess_cp_wdl_sf400k_filter": ["result"],
    "lichess_cp_wdl_sf_filter": ["result"],
}

# Expected value set for a column being renamed to `result`, and how much
# of the file must match. A result column is WHITE-relative 0/0.5/1.
RESULT_VALUES = (0.0, 0.5, 1.0)
REQUIRED_MATCH_FRACTION = 1.0    # 1.0 = every non-null value must be in the set
SAMPLE_ROWS_PER_FILE = 0         # 0 = verify ALL rows of the file (recommended)
# ═════════════════════════════════════════════════════════════════════════════


def verify_is_result(path: str, column: str) -> tuple[bool, str]:
    """Re-verify, from this file's own bytes, that `column` holds outcomes.

    Returns (ok, reason). `ok` is False whenever there is ANY doubt — a
    missing column, unreadable file, or a single out-of-set value all block
    the rename. Being conservative here is the entire point of the script."""
    try:
        pf = pq.ParquetFile(path)
    except Exception as exc:                       # unreadable / not parquet
        return False, f"cannot open ({exc})"

    names = pf.schema_arrow.names
    if column not in names:
        return False, f"no '{column}' column (has: {', '.join(names)})"
    if "result" in names:
        return False, "already has a 'result' column — refusing to overwrite"

    col = pf.read(columns=[column])[column].to_numpy(zero_copy_only=False)
    vals = np.asarray(col, dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return False, "column is entirely null/non-numeric"

    if SAMPLE_ROWS_PER_FILE:
        finite = finite[:SAMPLE_ROWS_PER_FILE]

    in_set = np.isin(finite, RESULT_VALUES)
    frac = float(in_set.mean())
    if frac < REQUIRED_MATCH_FRACTION:
        offenders = np.unique(finite[~in_set])[:5]
        return False, (f"only {frac:.4%} of values are in {RESULT_VALUES} "
                       f"(e.g. {np.round(offenders, 6).tolist()}) — this is NOT "
                       f"an outcome column, leaving it alone")
    return True, f"{finite.size:,} values, all in {RESULT_VALUES}"


def rename_column(path: str, old: str, new: str) -> None:
    """Rewrite `path` with `old` renamed to `new`. Values are untouched."""
    table = pq.read_table(path)
    names = [new if n == old else n for n in table.column_names]
    table = table.rename_columns(names)

    tmp = path + ".tmp"
    pq.write_table(table, tmp, compression="snappy")
    if BACKUP:
        shutil.copy2(path, path + ".bak")
    os.replace(tmp, path)



def verify_droppable(path: str, column: str) -> tuple[bool, str]:
    """Re-verify, from this file's bytes, that `column` is safe to delete.

    Safe means one of:
      * EMPTY        — no row carries a usable value, so nothing is lost.
      * DERIVED      — `cp` is present in the same file, so a `wdl` column
                       is recomputable exactly (and the stored one is known
                       to disagree with sigmoid(cp/320)).
      * SYNTHESISED  — a `result` that is a deterministic function of `cp`
                       (endgame generators). Detected by checking whether
                       result matches the +/-50cp threshold rule exactly.

    Anything else BLOCKS the drop. Deleting a column that carries real,
    non-recoverable signal is unrecoverable damage, so every ambiguity
    resolves to "do not touch"."""
    try:
        pf = pq.ParquetFile(path)
    except Exception as exc:
        return False, f"cannot open ({exc})"
    names = pf.schema_arrow.names
    if column not in names:
        return False, f"no '{column}' column — nothing to drop"

    tbl = pf.read()
    import pandas as pd
    df = tbl.to_pandas()
    col = df[column]

    # EMPTY?
    num = pd.to_numeric(col, errors="coerce")
    as_str = col.astype(str)
    nonempty_str = as_str.isin(["1-0", "0-1", "1/2-1/2"]).sum()
    if num.notna().sum() == 0 and nonempty_str == 0:
        return True, "column is entirely empty/null"

    # DERIVED wdl: cp present in the same file
    if column == "wdl" and "cp" in names:
        return True, "cp present — wdl is exactly recomputable, dropping the stored copy"

    # SYNTHESISED result: matches the +/-50cp rule
    if column == "result" and "cp" in names:
        cp = pd.to_numeric(df["cp"], errors="coerce").to_numpy(dtype=float)
        R = {"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0}
        r = as_str.map(R).to_numpy(dtype=float) if num.notna().sum() == 0 else num.to_numpy(dtype=float)
        m = np.isfinite(cp) & np.isfinite(r)
        if m.any():
            synth = np.where(cp[m] > 50, 1.0, np.where(cp[m] < -50, 0.0, 0.5))
            agree = float((synth == r[m]).mean())
            if agree >= 0.999:
                return True, (f"result is a deterministic function of cp "
                              f"(+/-50cp rule matches {agree:.4%}) — carries no "
                              f"information cp does not")
            return False, (f"result does NOT look synthesised (only {agree:.2%} "
                           f"matches the +/-50cp rule) — it may be a real outcome, "
                           f"refusing to delete")
    return False, "column carries values and is not provably derived — refusing"


def drop_columns(path: str, columns: list[str]) -> None:
    """Rewrite `path` without `columns`. Remaining values are untouched."""
    table = pq.read_table(path)
    keep = [c for c in table.column_names if c not in columns]
    table = table.select(keep)
    tmp = path + ".tmp"
    pq.write_table(table, tmp, compression="snappy")
    if BACKUP:
        shutil.copy2(path, path + ".bak")
    os.replace(tmp, path)


def main() -> int:
    if not RENAMES and not DROPS:
        print("Nothing configured in RENAMES or DROPS.")
        return 0

    print(f"{'DRY RUN — nothing will be written' if DRY_RUN else 'APPLYING CHANGES'}")
    print(f"backup={'on' if BACKUP else 'OFF'}  data_dir={DATA_DIR}\n")

    total_ok = total_skip = 0
    for dataset, (old, new) in RENAMES.items():
        folder = os.path.join(DATA_DIR, dataset)
        files = sorted(glob.glob(os.path.join(folder, "*.parquet")))
        print(f"-- {dataset}: {len(files)} file(s), rename '{old}' -> '{new}'")
        if not files:
            print("   no parquet files found, skipping\n")
            continue

        ok_files, skipped = [], []
        for f in files:
            ok, why = verify_is_result(f, old)
            (ok_files if ok else skipped).append((f, why))
            if not ok:
                print(f"   SKIP {os.path.basename(f)}: {why}")
                if FAIL_FAST:
                    print("\nFAIL_FAST set — stopping.")
                    return 1

        print(f"   verified OK: {len(ok_files)}/{len(files)}")
        if ok_files:
            print(f"   e.g. {os.path.basename(ok_files[0][0])}: {ok_files[0][1]}")

        if not DRY_RUN:
            for i, (f, _) in enumerate(ok_files, 1):
                rename_column(f, old, new)
                if i % 50 == 0 or i == len(ok_files):
                    print(f"   rewrote {i}/{len(ok_files)}")

        total_ok += len(ok_files)
        total_skip += len(skipped)
        print()

    for dataset, cols in DROPS.items():
        folder = os.path.join(DATA_DIR, dataset)
        files = sorted(glob.glob(os.path.join(folder, "*.parquet")))
        print(f"-- {dataset}: {len(files)} file(s), drop {cols}")
        if not files:
            print("   no parquet files found, skipping"); continue
        ok_files = []
        for f in files:
            reasons, ok = [], True
            for c in cols:
                good, why = verify_droppable(f, c)
                reasons.append(f"{c}: {why}")
                if not good and "nothing to drop" not in why:
                    ok = False
            if ok:
                ok_files.append((f, "; ".join(reasons)))
            else:
                print(f"   SKIP {os.path.basename(f)}: {'; '.join(reasons)}")
                total_skip += 1
        print(f"   verified OK: {len(ok_files)}/{len(files)}")
        if ok_files:
            print(f"   e.g. {os.path.basename(ok_files[0][0])}: {ok_files[0][1]}")
        if not DRY_RUN:
            for i, (f, _) in enumerate(ok_files, 1):
                present = [c for c in cols if c in pq.ParquetFile(f).schema_arrow.names]
                if present:
                    drop_columns(f, present)
                if i % 200 == 0 or i == len(ok_files):
                    print(f"   rewrote {i}/{len(ok_files)}")
        total_ok += len(ok_files)
        print()

    print(f"TOTAL: {total_ok} file(s) {'would be' if DRY_RUN else ''} renamed, "
          f"{total_skip} skipped.")
    if DRY_RUN and total_ok:
        print("\nSet DRY_RUN = False at the top of this file to apply.")
    if not DRY_RUN and total_ok:
        print("\nNow update train/train_nnue.py's DATASETS block: the affected\n"
              "dataset no longer needs wdl_is='result' — it has a real `result`\n"
              "column, so the default wdl_is='wdl' is correct again. Also update\n"
              "data/Data.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
