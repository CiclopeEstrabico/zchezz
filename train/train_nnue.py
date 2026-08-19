"""
train/train_nnue.py — trains the v4.00 HalfKP-4Bucket NNUE

WHAT IT DOES
────────────
Reads positions from one or more datasets, encodes them as HalfKP-4Bucket
features (encoding.py), trains model.py's network with BCE against a 0..1
target, and writes one .pt checkpoint per epoch into CKPT_DIR. Turn a
checkpoint into the engine's weight file with train/export_nnu4.py.

HOW TO RUN
──────────
    python train/train_nnue.py                # uses the CONFIGURATION block below
    python train/train_nnue.py --epochs 30    # every constant is also a flag

Everything adjustable lives in the CONFIGURATION block and the DATASETS
block below. A bare run does exactly what those blocks say; the flags only
override them.

WHAT TO SET, IN THE ORDER THAT MATTERS
──────────────────────────────────────
  LR           learning rate. ~1e-3 from random weights; ~1e-5 to refine an
               already-trained network.
  EPOCHS       matters more than it looks: the schedule is
               CosineAnnealingLR(T_max=EPOCHS), so this number also sets how
               fast the LR decays to eta_min.
  DATASETS     one entry per dataset: `pct` (fraction used per epoch), `mode`
               ('sample-rows' or 'sample-files'), `lam`
               (how much game RESULT is blended into the target).
  BATCH_SIZE / WORKERS / DEVICE   throughput, not quality.

DATA SOURCES
────────────
  'parquet'  columns fen + cp and/or result. Most of data/ is this.
  'bin'      the native selfplay format (engine/c/tools/sample.h, read via
             memmap by dataset.py), labelled with wl_target(k).
One run can mix both: each source is an entry in DATASETS / BIN_DATASETS,
or a --source on the command line.

TRAIN / VALIDATION SPLIT
────────────────────────
Every source is split by row index (deterministic, seeded by SEED) before
any epoch sampling, and validation loss is reported every VAL_EVERY epochs
in eval() mode with no gradient. With RESAMPLE_EACH_EPOCH on and `pct < 1.0`,
the TRAINING subsample is redrawn each epoch (seed = SEED + epoch); the
train/val split itself never moves.

HOW TO READ THE LOSS
────────────────────
The target is continuous (0..1), not 0/1, so the BCE floor is NOT 0.693 —
it is the mean entropy of the target itself, typically ~0.62 for the warmup
mix. Compare val_loss against that entropy, not against 0.693, and read
val_mae as the direct error in wdl units.

CHECKPOINT
──────────
A dict saved with torch.save holding the state_dict, the optimizer, the
epoch, the metrics and an 'arch' sub-dict:

    'arch': {'input': 2560, 'h1': 512, 'concat': 1024, 'h2': 32,
              'encoding': 'halfkp_4bucket'}

A run resumes only when --dataset-name matches the tag stored in the
checkpoint; with a different tag the weights are transferred instead and
the learning rate becomes TRANSFER_LR.

Training runs only under `if __name__ == "__main__"` — Windows
multiprocessing uses 'spawn' and re-imports this module in every worker.

════════════════════════════════════════════════════════════════════════
 LABEL CONVENTION (CLAUDE.md rule 10) — result / cp / wdl
════════════════════════════════════════════════════════════════════════

 | Name     | Meaning                                | Range / frame          |
 |----------|----------------------------------------|------------------------|
 | result   | the REAL game outcome                  | 0.0 / 0.5 / 1.0, WHITE |
 |          |                                        | 0 = Black won, 1 = White|
 | cp       | evaluation in centipawns               | int, WHITE-relative    |
 | wdl      | sigmoid(cp / 320) — a FUNCTION OF cp,  | 0..1, WHITE-relative   |
 |          | NOT an outcome                         |                        |
 | target   | lam*result + (1-lam)*wdl               | 0..1, computed at      |
 |          |                                        | TRAINING time only     |

 * All three are WHITE-relative, so ONE flip (x -> 1-x) converts the whole
   set to STM-relative downstream.
 * result and wdl share the SAME 0..1 scale because the target is a CONVEX
   combination. A result on a -1..1 scale would leave the sigmoid's range
   at lam=1 and break the BCE loss.
 * `wdl` is DERIVABLE from `cp`, so the two can rot apart. Whenever `cp`
   exists it wins and wdl is recomputed from it; a stored wdl is used only
   when cp is missing, and a disagreement is reported.
 * NEVER bake the blend into a dataset. lam is per-dataset, set in the
   DATASETS block at the top of train/train_nnue.py, and is precisely the
   knob you anneal across bootstrap generations.
 * 320 is the same constant as nnue.c's `_nnL3B * 320.0f` output scale.
   Changing one without the others silently rescales everything.
════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import time
import zlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from encoding import encode_positions, encode_mailbox_batch, LEGAL_MAX_ACTIVE_FEATURES
from model import NNUE, clamp_weights_, QA, QB
import dataset


# ════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — every CLI default below is one of these constants,
#  never a literal (CLAUDE.md rule 8). Edit here to change the defaults
#  used when a flag is not passed on the command line.
# ════════════════════════════════════════════════════════════════════════
CKPT_DIR = "checkpoints/v402"     # where the per-epoch .pt checkpoints are written
DATASET_NAME = "halfkp4b_v402_ft"  # tag stored in the checkpoint; resume only continues if this matches
EPOCHS = 100                      # number of training epochs. This also sets the LR
                                   # schedule: CosineAnnealingLR(T_max=EPOCHS), so a small
                                   # EPOCHS anneals the learning rate to eta_min quickly.
BATCH_SIZE = 65536                 # minibatch size (positions per optimizer step)
MAX_POSITIONS_CHUNK = 1_100_000   # positions buffered before a chunk is handed to the DataLoader
LR = 1e-3                         # fresh-start learning rate. ~1e-3 suits random weights;
                                   # ~1e-5 suits refining an already-trained net.
TRANSFER_LR = 3e-5                # learning rate when resuming onto a different --dataset-name
                                   # (weight transfer / refinement of an already-trained net)
WEIGHT_DECAY = 1e-4                # Adam weight decay
WORKERS = os.cpu_count() or 4     # multiprocessing.Pool size for FEN -> HalfKP encoding
DEVICE = "auto"                   # "auto" | "cuda" | "cpu"
SEED = 1234                       # seed for the deterministic train/val split and shard sampling
VAL_EVERY = 1                     # run a validation pass every N epochs
PARQUET_CHUNK_ROWS = 200_000      # row batch size when streaming a parquet source
RESAMPLE_EACH_EPOCH = True        # draw a fresh row subsample every epoch (sources with pct < 1.0)
ENCODE_CACHE = True               # cache each parquet file's encoded tensors next to it as
                                   # "*_encoded.pt", so later runs skip re-encoding the same FENs
HEARTBEAT_EVERY_BATCHES = 200     # print a mid-epoch loss/MAE/ETA line every N training batches
# ════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
#  Data source specification + CLI parsing
# ════════════════════════════════════════════════════════════════════════

# Previous value names for `mode`, accepted so an existing config keeps working.
# 'lines'/'shards' described the IMPLEMENTATION (read line by line vs skip whole
# shards); the current names say what `pct` actually selects.
_PCT_MODE_ALIASES = {"lines": "sample-rows", "shards": "sample-files"}


@dataclass
class SourceSpec:
    kind: Literal["parquet", "bin"]
    path: str                      # parquet: dir/glob of *.parquet ; bin: glob of *.bin
    target_col: str = "cp"         # parquet only: legacy hint, see the DATASETS block
    k: float = 1.0                 # bin only: wl_target() blend weight
    lam: float = 0.0               # parquet: lambda in target = lam*result + (1-lam)*sigmoid(cp/320)
    train_pct: float = 1.0         # fraction of this source's positions used per epoch
    val_frac: float = 0.02         # fraction of this source held out for validation
    name: str = ""                 # label for logging; defaults to basename(path)
    pct_mode: Literal["sample-rows", "sample-files"] = "sample-rows"
    # WHAT `pct` SELECTS — rows, or whole files:
    #   'sample-rows'   open every file, keep a random `pct` fraction OF ROWS.
    #                   Best mixing, but pays the full read cost even at
    #                   pct=0.02: at 2% you still read 100% of the bytes.
    #   'sample-files'  keep a random `pct` fraction OF FILES and never open
    #                   the rest. Far cheaper on a 1460-file dataset, but
    #                   coarser — rows inside one file stay together, so they
    #                   are correlated within an epoch.
    # Rule of thumb: 'sample-files' for a dataset split into many shards,
    # 'sample-rows' for one made of a few huge files.
    #
    # WARNING: 'sample-files' cannot subsample a dataset stored as ONE file.
    # Selection is per file, and a guard keeps at least one file so a source is
    # never starved to nothing — so a single-file dataset is read IN FULL
    # whatever `pct` says. Shard such a dataset (~2M rows per file) or use
    # 'sample-rows'.
    suffix: str = ".parquet"       # only files ending with this are used

    def __post_init__(self) -> None:
        if not self.name:
            self.name = os.path.basename(self.path.rstrip("/\\"))
        # Accept the previous value names so an old config keeps working.
        self.pct_mode = _PCT_MODE_ALIASES.get(self.pct_mode, self.pct_mode)
        if self.pct_mode not in ("sample-rows", "sample-files"):
            raise SystemExit(
                f"{self.name}: mode={self.pct_mode!r} is not valid. "
                f"Use 'sample-rows' or 'sample-files'.")


# ════════════════════════════════════════════════════════════════════════
#  DATASETS — the training mix, editable HERE (CLAUDE.md rule 8)
#
#  HOW IT WORKS
#    * Set `pct` to the fraction of that dataset used per epoch.
#      pct = 0.0  DISABLES the dataset entirely (it is not even opened).
#    * `mode` is 'sample-rows' or 'sample-files' — see SourceSpec.pct_mode.
#    * `col`  is a LEGACY hint, kept only for logging and for the case of a
#      dataset with neither `cp` nor `result`. Since
#      train/labeling/normalize_columns.py made `cp` the single stored
#      primitive (CLAUDE.md rule 10 — `wdl` is sigmoid(cp/320), so storing
#      both lets them rot apart), every entry below is 'cp'. blend_target()
#      keys off the columns ACTUALLY present, not off this field.
#    * `lam`  is the TRAINING-TARGET BLEND for this dataset:
#           target = lam * real_game_result + (1 - lam) * sigmoid(cp/320)
#      lam = 0.0  trust the labelling engine's evaluation (dense, low noise,
#                 but capped by what that engine knows).
#      lam = 1.0  trust only the real game outcome (ground truth, but one
#                 noisy bit smeared over every position of the game).
#      DEFAULTS ARE 0.00 because that is what was ACTUALLY in use: the old
#      `wdl` column these runs trained on is sigmoid(cp/320), i.e. pure
#      evaluation. Raise lam deliberately, per dataset — it only has an
#      effect where the dataset actually carries a `result` column (the
#      *_sf / selfplay_* / endgame_* sets do; extra-quiet-n5k_sf,
#      viriformat_* and the lichess-*wdl sets do NOT, and are reported).
#    * `k`    is the wl_target blend for kind='bin' sources only.
#    * Passing --source on the CLI OVERRIDES this whole block (so scripted
#      runs and sweeps still work); with no --source, this block is used.

# ════════════════════════════════════════════════════════════════════════
DATA_DIR = "data"        # root holding one folder per dataset

#    * `pct`  fraction of that dataset used per epoch. 0.0 DISABLES it
#             entirely (it is not even opened), which is how you take a
#             dataset out of the mix without deleting the line.
#    * `mode` 'sample-rows' (keeps a fraction of ROWS, reads every file) or
#             'sample-files' (keeps a fraction of FILES, never opens the
#             rest — far cheaper on a many-shard dataset, slightly coarser).
#    * `col`  legacy hint, used only for logging and for the case of a dataset
#             with neither `cp` nor `result`. `cp` is the single stored
#             primitive (CLAUDE.md rule 10), so every entry is 'cp'.
#             blend_target() keys off the columns ACTUALLY present.
#    * `lam`  target = lam*result + (1-lam)*sigmoid(cp/320). INERT unless the
#             dataset has BOTH columns — see data/Data.md for which do.
#
# Viriformat (viriformat_*, 42.5M rows) is deliberately absent: data judged
# low quality. Add it back only with a measured reason.
DATASETS = [
    # name                                                            pct   mode      col    lam
    {"name": "extraquiet_cp_sf5k_res_filter",                     "pct": 0.00, "mode": "sample-files", "col": "cp", "lam": 0.00},   # 40.1M  humans / SF 5k nodes
    {"name": "lichess_cp_sfdb_filter",                            "pct": 0.00, "mode": "sample-files", "col": "cp", "lam": 0.00},   # 18.6M  lichess / SF evals from the lichess DB
    {"name": "selfplay_cp_sf50k_res_filter_data20260410",         "pct": 0.20, "mode": "sample-files", "col": "cp", "lam": 0.00},   # 16.1M  selfplay / SF 50k
    {"name": "lichess_cp_sf_filter",                              "pct": 0.00, "mode": "sample-files", "col": "cp", "lam": 0.00},   # 15.2M  lichess / SF
    {"name": "selfplay_cp_zchezz_res_filter_data20260401",        "pct": 0.50, "mode": "sample-files", "col": "cp", "lam": 0.00},   #  9.1M  selfplay / Zchezz itself
    {"name": "selfplay_cp_zchezz_res_filter_data20260404",        "pct": 0.50, "mode": "sample-files", "col": "cp", "lam": 0.00},   #  8.8M  selfplay / Zchezz itself
    {"name": "selfplay_cp_sf50k_res_filter_data20260404",         "pct": 0.00, "mode": "sample-files", "col": "cp", "lam": 0.00},   #  7.5M  selfplay / SF 50k
    {"name": "selfplay_cp_sf100k_res_endgames_filter_data20260414","pct": 0.50, "mode": "sample-files", "col": "cp", "lam": 0.00},  #  5.7M  selfplay endgames / SF 100k
    {"name": "extraquiet_cp_sfd14_endgames_filter",               "pct": 1.00, "mode": "sample-files", "col": "cp", "lam": 0.00},   #  2.8M  human endgames / SF depth 14
    {"name": "selfplay-lichess_cp_sf500k_res_filter_endgames",    "pct": 1.00, "mode": "sample-rows",  "col": "cp", "lam": 0.00},   #  2.7M  selfplay endgames / SF 500k
    {"name": "extraquiet_cp_sf60k_filter",                        "pct": 0.10, "mode": "sample-files", "col": "cp", "lam": 0.00},   #  1.5M  humans / SF 60k
    {"name": "synthetic_endgame_cp_sf_filter_data20260414",       "pct": 1.00, "mode": "sample-files", "col": "cp", "lam": 0.00},   #  1.2M  generated endgames / SF
    {"name": "selfplay_cp_zchezz_res_filter_data20260410",        "pct": 1.00, "mode": "sample-files", "col": "cp", "lam": 0.00},   #  850k  selfplay / Zchezz itself
    {"name": "synthetic_endgame_cp_sf_filter_data20260413",       "pct": 1.00, "mode": "sample-rows",  "col": "cp", "lam": 0.00},   #  521k  generated endgames / SF
    {"name": "extraquiet_cp_sfd12_endgames_filter",               "pct": 0.50, "mode": "sample-files", "col": "cp", "lam": 0.00},   #  262k  human endgames / SF depth 12
    {"name": "lichess_cp_sf400k_filter",                          "pct": 0.00, "mode": "sample-files", "col": "cp", "lam": 0.00},   #  150k  lichess / SF 400k
    {"name": "miscelaneous_cp_sf1M_res_filter",                   "pct": 1.00, "mode": "sample-rows",  "col": "cp", "lam": 0.00},   #   34k  mixed / SF 1M
]

# Packed .bin selfplay sources (Appendix F.2/F.3). Same rules; `k` is the
# wl_target blend (1.0 = pure game result, see dataset.py).
BIN_DATASETS = [
    # {"path": "C:/nnue_checkpoints/selfplay/gen1/*.bin", "pct": 1.00, "k": 1.0, "name": "gen1"},
]


def sources_from_config() -> list[SourceSpec]:
    """Build the --source list from the DATASETS/BIN_DATASETS blocks above.

    Datasets with pct == 0.0 are skipped entirely — they are not opened,
    so a disabled 145M-row dataset costs nothing. Missing directories are
    reported and skipped rather than raising, so one stale entry in the
    block does not block a run."""
    out: list[SourceSpec] = []
    for d in DATASETS:
        if d.get("pct", 0.0) <= 0.0:
            continue
        path = os.path.join(DATA_DIR, d["name"])
        if not os.path.isdir(path):
            print(f"  [config] WARNING: dataset dir not found, skipping: {path}")
            continue
        out.append(SourceSpec(kind="parquet", path=path, target_col=d.get("col", "wdl"),
                              train_pct=float(d["pct"]), pct_mode=d.get("mode", "sample-rows"),
                              lam=float(d.get("lam", 0.0)),
                              suffix=d.get("suffix", ".parquet"), name=d["name"]))
    for d in BIN_DATASETS:
        if d.get("pct", 0.0) <= 0.0:
            continue
        out.append(SourceSpec(kind="bin", path=d["path"], k=float(d.get("k", 1.0)),
                              train_pct=float(d["pct"]), name=d.get("name", "")))
    return out


def _parse_kv_spec(spec: str) -> dict[str, str]:
    """Parse a 'key=val,key=val,...' CLI token into a dict of strings."""
    out: dict[str, str] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise argparse.ArgumentTypeError(f"bad --source token (expected key=val): {part!r}")
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _source_from_string(spec: str) -> SourceSpec:
    kv = _parse_kv_spec(spec)
    kind = kv.get("kind")
    if kind not in ("parquet", "bin"):
        raise argparse.ArgumentTypeError(
            f"--source kind must be 'parquet' or 'bin', got {kind!r} in {spec!r}"
        )
    path = kv.get("path") or kv.get("glob")
    if not path:
        raise argparse.ArgumentTypeError(f"--source missing path=/glob=: {spec!r}")

    return SourceSpec(
        kind=kind,
        path=path,
        target_col=kv.get("target_col", "wdl"),
        k=float(kv.get("k", 1.0)),
        train_pct=float(kv.get("train_pct", 1.0)),
        val_frac=float(kv.get("val_frac", 0.02)),
        name=kv.get("name", ""),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Zchezz v4.00 HalfKP-4Bucket NNUE trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--source", action="append", type=_source_from_string, dest="sources", default=[],
        metavar="kind=parquet|bin,path=...,[target_col=wdl|cp],[k=1.0],[train_pct=1.0],[val_frac=0.02],[name=...]",
        help="One data source. Repeat --source for each dataset to mix. "
             "Example: --source kind=parquet,path=C:/nnue_checkpoints/data/lichess-quiet,target_col=wdl,train_pct=0.5 "
             "--source kind=bin,path=C:/nnue_checkpoints/selfplay/gen1/*.bin,k=1.0",
    )
    p.add_argument("--ckpt-dir", default=CKPT_DIR,
                    help="Where to write nnue_v400_epochNN_*.pt checkpoints.")
    p.add_argument("--dataset-name", default=DATASET_NAME,
                    help="Tag stored in the checkpoint; resume logic only resumes "
                         "if this matches the latest checkpoint's tag.")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--max-positions-chunk", type=int, default=MAX_POSITIONS_CHUNK,
                    help="Max positions held in one streamed chunk before it is "
                         "handed to the DataLoader (mirrors v3.14's MAX_POSITIONS).")
    p.add_argument("--lr", type=float, default=LR, help="Fresh-start learning rate.")
    p.add_argument("--transfer-lr", type=float, default=TRANSFER_LR,
                    help="LR used when resuming onto a different --dataset-name "
                         "(weight transfer instead of a same-dataset resume).")
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--workers", type=int, default=WORKERS,
                    help="multiprocessing.Pool size used for FEN -> HalfKP encoding.")
    p.add_argument("--device", default=DEVICE, choices=["auto", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=SEED,
                    help="Seed for the deterministic train/val split and shard sampling.")
    p.add_argument("--val-every", type=int, default=VAL_EVERY,
                    help="Run a validation pass every N epochs.")
    p.add_argument("--parquet-chunk-rows", type=int, default=PARQUET_CHUNK_ROWS,
                    help="Row batch size when streaming a parquet source.")
    p.add_argument("--resample-each-epoch", action=argparse.BooleanOptionalAction, default=RESAMPLE_EACH_EPOCH,
                    help="For sources with train_pct < 1.0, draw a FRESH random "
                         "row subsample every epoch (seed = --seed + epoch number) "
                         "instead of reusing the same subsample forever. Matches "
                         "v3.14 mixtrain.py's RESAMPLE_EACH_EPOCH=True default. "
                         "The train/val split itself is always stable across "
                         "epochs regardless of this flag — only the in-train "
                         "subsample is affected. Use --no-resample-each-epoch to "
                         "pin training to the same row subset every epoch.")
    p.add_argument("--encode-cache", action=argparse.BooleanOptionalAction, default=ENCODE_CACHE,
                    help="Cache each parquet file's encoded HalfKP tensors to a sibling "
                         "'*_encoded.pt' file, reused on later runs instead of re-running "
                         "encode_positions() on the same FENs. Use --no-encode-cache to "
                         "always re-encode (e.g. while iterating on encoding.py itself).")
    p.add_argument("--show-config", action="store_true",
                    help="print the resolved configuration (including the DATASETS "
                         "mix) and exit without training")
    p.add_argument("--heartbeat-every-batches", type=int, default=HEARTBEAT_EVERY_BATCHES,
                    help="Print a running loss/MAE/ETA heartbeat every N training batches.")
    return p


# ════════════════════════════════════════════════════════════════════════
#  Multiprocessing-based FEN -> HalfKP encoding
# ════════════════════════════════════════════════════════════════════════

def _encode_fen_chunk(fens: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Worker-process entry point: must be a module-level function (not a
    lambda/closure) so it is picklable for `multiprocessing.Pool` under
    Windows' 'spawn' start method."""
    return encode_positions(fens)


def encode_fens_parallel(fens: list[str], pool: "mp.pool.Pool", n_workers: int
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split `fens` into n_workers contiguous chunks, encode each chunk in
    its own process via `pool.map`, then re-merge the per-chunk sparse
    (indices, offsets) pairs into one batch-wide pair.

    This is the direct real-multiprocessing replacement for v3.14's
    `build_tensors_parallel` (which used ThreadPoolExecutor to no real
    benefit — see module docstring point 4).
    """
    if not fens:
        empty_i = np.zeros(0, dtype=np.int32)
        empty_o = np.zeros(0, dtype=np.int64)
        return empty_i, empty_o, empty_i, empty_o

    n_workers = max(1, min(n_workers, len(fens)))
    chunk_size = max(1, (len(fens) + n_workers - 1) // n_workers)
    chunks = [fens[i:i + chunk_size] for i in range(0, len(fens), chunk_size)]

    results = pool.map(_encode_fen_chunk, chunks)

    stm_idx_parts, stm_off_parts, opp_idx_parts, opp_off_parts = [], [], [], []
    stm_running, opp_running = 0, 0
    for stm_idx, stm_off, opp_idx, opp_off in results:
        stm_idx_parts.append(stm_idx)
        stm_off_parts.append(stm_off + stm_running)
        stm_running += len(stm_idx)

        opp_idx_parts.append(opp_idx)
        opp_off_parts.append(opp_off + opp_running)
        opp_running += len(opp_idx)

    return (
        np.concatenate(stm_idx_parts), np.concatenate(stm_off_parts),
        np.concatenate(opp_idx_parts), np.concatenate(opp_off_parts),
    )


# ════════════════════════════════════════════════════════════════════════
#  Parquet source: streaming read + train/val split
# ════════════════════════════════════════════════════════════════════════

def to_wdl(values: np.ndarray, col: str) -> np.ndarray:
    """Same convention as v3.14 and APPENDIX D.2: 'wdl' columns pass
    through, 'cp' columns are squashed via sigmoid(cp/320)."""
    if col == "wdl":
        return values.astype(np.float32)
    return (1.0 / (1.0 + np.exp(-values.astype(np.float32) / 320.0))).astype(np.float32)


# ── Training target: the lambda blend ───────────────────────────────
#
#   target = lam * result_prob + (1 - lam) * sigmoid(cp / CP_TO_WDL_T)
#
# Both terms are WHITE-RELATIVE here; the caller flips to STM-relative.
#
#   lam = 1.0  -> pure real game outcome (TD(1)). Ground truth, but one
#                 noisy bit spread over every position of the game.
#   lam = 0.0  -> pure engine evaluation. Dense and low-noise, but it can
#                 only ever teach what the labelling engine already knows.
#   in between -> the F.3.0 blend.
#
# WHY THIS IS COMPUTED HERE AND NOT IN THE DATASET: baking the blend into
# the parquet freezes lambda at generation time, and lambda is exactly the
# knob you want to anneal across generations. It also went wrong in
# practice: the historical `wdl` column is NOT the game outcome, it is
# sigmoid(cp/320) — a deterministic transform of `cp` (verified: cp=298 ->
# wdl=0.7173 -> T=320.0). Anything that blended `wdl` with `cp` was
# computing lam*f(cp) + (1-lam)*f(cp) = f(cp), with lambda doing nothing.
# NAMING CONVENTION (project-wide):
#     result  the real game outcome:  '1-0' | '1/2-1/2' | '0-1'
#     cp      evaluation in centipawns, WHITE-relative
#     wdl     sigmoid(cp / CP_TO_WDL_T)  -- a pure function of cp, NOT an
#             outcome. It is stored for convenience, never as ground truth.
#     target  lam * result_prob + (1 - lam) * wdl        <- computed HERE
#
# Because `wdl` is DERIVABLE from `cp`, the two can silently disagree if
# anything ever writes an inconsistent pair. So: whenever `cp` is present
# it WINS -- wdl is recomputed from it rather than trusted -- and a stored
# `wdl` that disagrees is reported once per dataset. The stored column is
# used as the eval term only when `cp` is absent.
#
# MISSING COLUMNS: a source may have only one of the two. Rather than
# dropping those rows, the available term is used alone and the fallback
# is COUNTED and reported, so "this dataset silently trained at lam=0"
# is visible instead of invisible.
CP_TO_WDL_T = 320.0    # must match nnue.c's `_nnL3B * 320.0f` output scale

_BLEND_STATS = {"result_only": 0, "cp_only": 0, "both": 0, "warned": set()}

# `result` is WHITE-RELATIVE and lives on the SAME SCALE as `wdl`: 0.0 =
# Black won, 0.5 = draw, 1.0 = White won. Same scale matters — the blend
# lam*result + (1-lam)*wdl is a convex combination, so a result in, say,
# [-1,1] against a wdl in [0,1] would not just be inconsistent, it would
# push the target outside the sigmoid's range and break the BCE loss at
# lam=1. Same frame matters too: cp, wdl and result are all white-relative,
# so the single STM flip (x -> 1-x) applied downstream is correct for all
# three at once.
#
# New datasets should write `result` as a NUMBER. The legacy string forms
# are still accepted so the existing parquet corpus keeps working.
_RESULT_TO_PROB = {"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0,
                   "1": 1.0, "0": 0.0, "1/2": 0.5, "": np.nan}


def _result_column_to_prob(col) -> np.ndarray:
    """Map a `result` column to white-relative probability in [0,1].

    Accepts numeric (already 0.0/0.5/1.0) or the legacy PGN strings."""
    if pd.api.types.is_numeric_dtype(col):
        v = pd.to_numeric(col, errors="coerce").to_numpy(dtype=np.float64)
        # Guard against a -1..1 column sneaking in: it would silently skew
        # every target. Detect and convert rather than train on nonsense.
        finite = v[np.isfinite(v)]
        if finite.size and finite.min() < -0.001:
            v = (v + 1.0) / 2.0
        return v
    return col.map(_RESULT_TO_PROB).to_numpy(dtype=np.float64)


def blend_target(df, source: SourceSpec) -> np.ndarray:
    """White-relative training target for one parquet row batch."""
    n = len(df)
    has_cp = "cp" in df.columns
    has_res = "result" in df.columns

    cp_prob = None
    res_prob = None
    if has_res:
        res_prob = _result_column_to_prob(df["result"])
    if has_cp:
        cp = pd.to_numeric(df["cp"], errors="coerce").to_numpy(dtype=np.float64)
        cp_prob = 1.0 / (1.0 + np.exp(-cp / CP_TO_WDL_T))
        cp_prob = np.where(np.isfinite(cp_prob), cp_prob, np.nan)
        # Consistency check against a stored `wdl`, if there is one. They
        # must agree by definition; if they do not, the dataset was written
        # by something that used a different temperature or baked a blend
        # into the column. Report once and keep the cp-derived value.
        if "wdl" in df.columns:
            stored = pd.to_numeric(df["wdl"], errors="coerce").to_numpy(dtype=np.float64)
            both = np.isfinite(stored) & np.isfinite(cp_prob)
            if both.any():
                bad = both & (np.abs(stored - cp_prob) > 0.02)
                if bad.any():
                    key = (source.name, "wdl_mismatch")
                    if key not in _BLEND_STATS["warned"]:
                        _BLEND_STATS["warned"].add(key)
                        frac = bad.sum() / max(1, both.sum())
                        print(f"  [blend] {source.name}: stored 'wdl' disagrees with "
                              f"sigmoid(cp/{CP_TO_WDL_T:.0f}) on {frac:.1%} of rows "
                              f"(max diff {np.abs(stored[bad] - cp_prob[bad]).max():.3f}). "
                              f"Using the cp-derived value; check how this dataset was written.")

    # Fall back column-by-column so a dataset with a few missing cells still
    # contributes those rows through whichever term it does have.
    if cp_prob is None and res_prob is None:
        raise SystemExit(
            f"{source.name}: no 'cp' and no 'result' column — there is no target to "
            f"train on. Columns present: {list(df.columns)}. Every dataset must carry "
            f"`cp`, `result`, or both (CLAUDE.md rule 10).")

    if res_prob is None:
        _BLEND_STATS["cp_only"] += n
        return cp_prob
    if cp_prob is None:
        _BLEND_STATS["result_only"] += n
        return res_prob

    lam = float(source.lam)
    out = lam * res_prob + (1.0 - lam) * cp_prob
    # Per-cell fallback where one of the two is missing.
    only_cp = np.isnan(res_prob) & ~np.isnan(cp_prob)
    only_res = np.isnan(cp_prob) & ~np.isnan(res_prob)
    out = np.where(only_cp, cp_prob, out)
    out = np.where(only_res, res_prob, out)
    _BLEND_STATS["cp_only"] += int(only_cp.sum())
    _BLEND_STATS["result_only"] += int(only_res.sum())
    _BLEND_STATS["both"] += int((~np.isnan(cp_prob) & ~np.isnan(res_prob)).sum())
    return np.nan_to_num(out, nan=0.5)


def report_blend_stats() -> None:
    s = _BLEND_STATS
    tot = s["both"] + s["cp_only"] + s["result_only"]
    if tot:
        print(f"  [blend] rows: both cp+result={s['both']:,}  cp only={s['cp_only']:,}  "
              f"result only={s['result_only']:,}")


# ── Encoded-chunk .pt cache (restores v3.14 mixtrain.py's pt_name()/
#    _encoded.pt caching) ──────────────────────────────────────────────
#
# Each parquet FILE gets one sibling cache file holding the ALREADY-
# ENCODED HalfKP sparse tensors for every legal row in that file, plus
# the raw label columns (cp / result-as-probability / wdl) and an
# is_black flag per row — everything iter_parquet_rows needs to rebuild
# a train/val split and the lam-blended target WITHOUT re-parsing a
# single FEN through python-chess.
#
# What is cached vs recomputed on load, and why:
#   * stm/opp sparse bags   — cached. This is the expensive part
#     (encode_positions()'s per-position python-chess Board+piece_map
#     walk) and it never changes for a given FEN, so it is exactly what
#     the cache exists to avoid recomputing.
#   * raw cp / result-prob / wdl columns — cached RAW, not the final
#     blended target. `target = lam*result + (1-lam)*wdl` depends on
#     `source.lam`, which is exactly the knob later runs may want to
#     change (see module docstring, LABEL CONVENTION section); baking a
#     frozen target into the cache would silently pin lam at
#     cache-build time. blend_target() is re-run on every load instead.
#   * is_black — cached (derived once from the FEN's side-to-move
#     field) so the STM-relative flip never needs the FEN text again.
#   * illegal positions (see _fen_is_trainable) are dropped once, at
#     cache-BUILD time, and simply never enter the cache — the dropped
#     count is still reported every time the cache is consulted (hit or
#     miss) so per-run stats stay accurate.
#
# Corrupt/incompatible caches (schema changed, truncated write, etc.)
# fall back to a full re-encode + rewrite, exactly like v3.14's
# try/except around torch.load — never a hard crash.
def _encoded_cache_path(fpath: str, suffix: str) -> str:
    if suffix and fpath.endswith(suffix):
        return fpath[: -len(suffix)] + "_encoded.pt"
    return fpath + "_encoded.pt"


_ENCODED_CACHE_KEYS = ("stm_idx", "stm_off", "opp_idx", "opp_off", "is_black",
                       "cp", "result_prob", "wdl", "n_rows", "n_dropped")

# Cache-invalidation stamp. The cache path is derived from the parquet's NAME
# alone, so without this a rewritten parquet would silently keep serving the
# labels encoded from its previous contents — the run would look perfectly
# healthy and train on data that no longer exists on disk. This is not
# hypothetical: train/labeling/normalize_columns.py rewrites every dataset in
# place (dropping `wdl`, snapping `result`), keeping the same filenames.
#
# (size, mtime_ns) is enough here — these files are only ever rewritten
# wholesale by a labeling script, never edited in a way that preserves both.
_CACHE_STAMP_KEY = "src_stamp"


def _source_stamp(fpath: str) -> tuple[int, int]:
    st = os.stat(fpath)
    return (st.st_size, st.st_mtime_ns)


def _build_parquet_file_cache(fpath: str, source: SourceSpec, chunk_rows: int,
                               pool: "mp.pool.Pool", n_workers: int) -> dict:
    """Read one whole parquet file, drop illegal positions, encode the
    survivors, and return the cache dict described above (also written
    to disk by the caller when --encode-cache is on)."""
    import pyarrow.parquet as pq

    fens: list[str] = []
    cp_parts, result_parts, wdl_parts = [], [], []
    have_cp = have_result = have_wdl = False

    pf = pq.ParquetFile(fpath)
    for batch in pf.iter_batches(batch_size=chunk_rows):
        df = batch.to_pandas()
        fens.extend(df["fen"].tolist())
        if "cp" in df.columns:
            have_cp = True
            cp_parts.append(pd.to_numeric(df["cp"], errors="coerce").to_numpy(dtype=np.float64))
        if "result" in df.columns:
            have_result = True
            result_parts.append(_result_column_to_prob(df["result"]))
        if "wdl" in df.columns:
            have_wdl = True
            wdl_parts.append(pd.to_numeric(df["wdl"], errors="coerce").to_numpy(dtype=np.float64))

    n_total = len(fens)
    keep = [i for i, f in enumerate(fens) if _fen_is_trainable(f)]
    n_dropped = n_total - len(keep)

    def _sel(parts: list[np.ndarray], have: bool) -> np.ndarray | None:
        if not have:
            return None
        return np.concatenate(parts)[keep] if parts else np.zeros(0, dtype=np.float64)

    cp_arr = _sel(cp_parts, have_cp)
    result_arr = _sel(result_parts, have_result)
    wdl_arr = _sel(wdl_parts, have_wdl)
    fens_kept = [fens[i] for i in keep]
    is_black = np.array([f.split(" ")[1] == "b" for f in fens_kept], dtype=bool)

    if fens_kept:
        stm_idx, stm_off, opp_idx, opp_off = encode_fens_parallel(fens_kept, pool, n_workers)
    else:
        stm_idx = opp_idx = np.zeros(0, dtype=np.int32)
        stm_off = opp_off = np.zeros(0, dtype=np.int64)

    return {
        "stm_idx": stm_idx, "stm_off": stm_off, "opp_idx": opp_idx, "opp_off": opp_off,
        "is_black": is_black, "cp": cp_arr, "result_prob": result_arr, "wdl": wdl_arr,
        "n_rows": len(fens_kept), "n_dropped": n_dropped,
    }


def _load_or_build_parquet_file_cache(fpath: str, source: SourceSpec, chunk_rows: int,
                                       pool: "mp.pool.Pool", n_workers: int,
                                       use_cache: bool) -> dict:
    cache_path = _encoded_cache_path(fpath, source.suffix)
    if use_cache and os.path.exists(cache_path):
        try:
            data = torch.load(cache_path, map_location="cpu", weights_only=False)
            if not all(k in data for k in _ENCODED_CACHE_KEYS):
                raise ValueError("cache missing expected keys (stale schema)")
            stamp = data.get(_CACHE_STAMP_KEY)
            if stamp is None:
                raise ValueError("cache predates source-stamp validation")
            if tuple(stamp) != _source_stamp(fpath):
                raise ValueError(
                    f"source parquet changed since this cache was written "
                    f"(stamp {tuple(stamp)} != {_source_stamp(fpath)})")
            return data
        except Exception as e:
            print(f"  [encode_cache] WARNING: stale/incompatible cache at {cache_path} "
                  f"({e!r}). Delete stale '_encoded.pt' files if this repeats. Re-encoding.")

    data = _build_parquet_file_cache(fpath, source, chunk_rows, pool, n_workers)
    data[_CACHE_STAMP_KEY] = _source_stamp(fpath)
    if use_cache:
        try:
            torch.save(data, cache_path)
        except Exception as e:
            print(f"  [encode_cache] WARNING: could not write cache {cache_path}: {e!r}")
    return data


def _select_bag_rows(flat_idx: np.ndarray, offsets: np.ndarray, row_idx: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Select an arbitrary (not necessarily contiguous or sorted) subset
    of rows from an EmbeddingBag-style (flat_idx, offsets) pair, returning
    a new pair with offsets rebased to start at 0. Used to apply the
    train/val split and train_pct resample directly to a whole file's
    cached encoded tensors."""
    if len(row_idx) == 0:
        return np.zeros(0, dtype=flat_idx.dtype), np.zeros(0, dtype=offsets.dtype)
    n = len(offsets)
    ends = np.empty(n, dtype=np.int64)
    ends[:-1] = offsets[1:]
    ends[-1] = len(flat_idx)
    lengths = (ends - offsets)[row_idx]
    new_offsets = np.zeros(len(row_idx), dtype=offsets.dtype)
    np.cumsum(lengths[:-1], out=new_offsets[1:])
    parts = [flat_idx[offsets[i]:ends[i]] for i in row_idx]
    new_flat = np.concatenate(parts) if parts else np.zeros(0, dtype=flat_idx.dtype)
    return new_flat, new_offsets


def iter_parquet_rows(source: SourceSpec, chunk_rows: int, split: Literal["train", "val"],
                       split_seed: int, resample_seed: int,
                       pool: "mp.pool.Pool" = None, n_workers: int = 1,
                       use_cache: bool = ENCODE_CACHE):
    """Yield (stm_idx, stm_off, opp_idx, opp_off, y) ALREADY-ENCODED chunks
    from every *.parquet file matching source.path, deterministically
    split row-by-row into train/val via a seeded hash of the row's
    position within its file (so the split is stable across runs without
    needing to persist an index file).

    Each file's FENs are encoded exactly once (via
    _load_or_build_parquet_file_cache's sibling '_encoded.pt' cache) and
    every subsequent train/val split, epoch resample, and lam-target
    blend is computed on the CACHED arrays — see the cache docstring
    above for what is cached raw vs. recomputed on every load.

    Two independent RNGs are used: `split_seed` drives the train/val
    partition (kept fixed across epochs, so a row's train/val membership
    never changes mid-run), while `resample_seed` drives the train_pct
    subsampling of the *train* rows (varies per epoch when
    --resample-each-epoch is on, so a fresh random subset of the training
    rows is drawn each epoch instead of reusing the same one forever —
    see module docstring point 0 / v3.14's RESAMPLE_EACH_EPOCH).
    """
    files = sorted(glob.glob(os.path.join(source.path, "*" + source.suffix)))
    if not files:
        # Also accept `source.path` being a direct glob pattern.
        files = sorted(glob.glob(source.path))
    if not files:
        raise FileNotFoundError(
            f"No files matching '*{source.suffix}' for source {source.name!r} at {source.path!r}")

    # pct_mode='sample-files': drop whole FILES up front and
    # never open them, instead of reading every file and discarding rows.
    # On a 1460-file dataset at train_pct=0.02 this is the difference
    # between reading 18.5M rows and reading 370k. Only the train split is
    # subsampled — validation always sees its full share, otherwise the
    # val loss would be computed on a different amount of data each epoch
    # and stop being comparable across epochs.
    if source.pct_mode == "sample-files" and source.train_pct < 1.0 and split == "train":
        shard_rng = np.random.default_rng(
            np.random.SeedSequence([resample_seed, 0x5A4D5, zlib.crc32(source.name.encode())]))
        pick = shard_rng.random(len(files)) < source.train_pct
        if not pick.any():          # never starve a source to nothing
            pick[shard_rng.integers(len(files))] = True
        files = [f for f, k in zip(files, pick) if k]

    # CRITICAL: these two streams must be genuinely independent.
    #
    # Seeding them with two scalars (default_rng(split_seed) and
    # default_rng(resample_seed)) is NOT enough: whenever the two scalars
    # happen to be equal — which is exactly the case on epoch 0, and on
    # every epoch when --no-resample-each-epoch is used — both generators
    # emit the IDENTICAL uniform sequence. Then `keep = u < train_pct` and
    # `is_val = u < val_frac` are nested rather than independent events, so
    # `~is_val & keep` is EMPTY whenever train_pct <= val_frac: the training
    # split silently yields zero rows and the run reports "0 batches" while
    # validation looks perfectly healthy. (When train_pct > val_frac it
    # doesn't crash, it just quietly trains on train_pct - val_frac of the
    # data instead of train_pct.)
    #
    # Spawning from a SeedSequence with distinct spawn keys gives streams
    # that are independent by construction, whatever the scalars are.
    rng = np.random.default_rng(np.random.SeedSequence([split_seed, 0xA11CE]))
    resample_rng = np.random.default_rng(np.random.SeedSequence([resample_seed, 0xB0B]))

    for fpath in files:
        cache = _load_or_build_parquet_file_cache(
            fpath, source, chunk_rows, pool, n_workers, use_cache)
        n_rows = cache["n_rows"]
        if cache["n_dropped"]:
            _note_dropped(cache["n_dropped"], cache["n_dropped"] + n_rows)
        if n_rows == 0:
            continue

        # Deterministic per-row train/val split (stable across epochs).
        # Drawn once per whole file now that encoding is cached at file
        # granularity (was once per pyarrow read-batch before Fix 3's
        # caching; still adequate for an initial v400 trainer, not a hard
        # scientific reproducibility guarantee — see module note above).
        row_u = rng.random(n_rows)
        is_val = row_u < source.val_frac
        mask = is_val if split == "val" else ~is_val

        if source.pct_mode == "sample-rows" and source.train_pct < 1.0 and split == "train":
            # Independent RNG stream so this resamples per epoch without
            # perturbing the train/val split above.
            keep = resample_rng.random(n_rows) < source.train_pct
            mask = mask & keep

        if not mask.any():
            continue
        idx = np.nonzero(mask)[0]

        # Rebuild the lam-blended target from the CACHED raw columns (not
        # a frozen cached target — see cache docstring: lam is a per-run
        # knob and must not be baked in).
        cols = {}
        if cache["cp"] is not None:
            cols["cp"] = cache["cp"][idx]
        if cache["result_prob"] is not None:
            cols["result"] = cache["result_prob"][idx]
        if cache["wdl"] is not None:
            cols["wdl"] = cache["wdl"][idx]
        if cols:
            y = blend_target(pd.DataFrame(cols), source)
        else:
            # No label column at all survived caching for this file — fall
            # back to a neutral target rather than crashing; blend_target's
            # own "legacy_wdl" fallback path handles the far more common
            # case of a lone 'wdl' column, which IS cached above.
            y = np.full(len(idx), 0.5, dtype=np.float64)

        # White-relative -> STM-relative flip (APPENDIX D.1/D.2).
        is_black_sel = cache["is_black"][idx]
        y = y.copy()
        y[is_black_sel] = 1.0 - y[is_black_sel]

        stm_idx_sel, stm_off_sel = _select_bag_rows(cache["stm_idx"], cache["stm_off"], idx)
        opp_idx_sel, opp_off_sel = _select_bag_rows(cache["opp_idx"], cache["opp_off"], idx)
        yield stm_idx_sel, stm_off_sel, opp_idx_sel, opp_off_sel, y.astype(np.float32)


# ════════════════════════════════════════════════════════════════════════
#  Bin source: memmap read + train/val split
# ════════════════════════════════════════════════════════════════════════

def iter_bin_rows(source: SourceSpec, chunk_rows: int, split: Literal["train", "val"],
                   split_seed: int, resample_seed: int):
    """Yield (boards, stms, y) chunks from a MultiShardSelfplay dataset,
    applying wl_target() with this source's K and a deterministic
    train/val split over the GLOBAL row index (stable regardless of
    chunking).

    Yields the raw mailbox `board`/`stm` arrays straight from the `.bin`
    records — NOT fens — so the caller can feed them to
    encoding.encode_mailbox_batch() directly. Building a FEN string here
    only to parse it straight back with python-chess in the caller would
    be exactly the wasteful array -> string -> array round-trip this path
    exists to avoid (see encode_mailbox_batch's docstring); use
    dataset.records_to_fens_and_targets() only where a FEN is genuinely
    needed (parquet path, ad-hoc scripts, tests).

    As in iter_parquet_rows, `split_seed` fixes the train/val partition
    across epochs and `resample_seed` (varies per epoch when
    --resample-each-epoch is on) independently drives which train rows are
    kept under train_pct < 1.0, so the train subsample can be refreshed
    every epoch without disturbing the val set.
    """
    ds = dataset.MultiShardSelfplay.from_glob(source.path)
    n = len(ds)
    if n == 0:
        return

    # zlib.crc32, NOT hash(): Python's builtin hash() for str is salted
    # per-process (PYTHONHASHSEED), so using it here would give a DIFFERENT
    # train/val split on every run — silently destroying the reproducibility
    # this seed is supposed to provide, and leaking validation rows into
    # training across runs of the same experiment.
    name_salt = zlib.crc32(source.name.encode("utf-8")) & 0xFFFFFFFF
    split_rng = np.random.default_rng(split_seed ^ name_salt)
    perm = split_rng.permutation(n)
    n_val = max(1, int(n * source.val_frac)) if n > 1 else 0
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    if source.train_pct < 1.0:
        resample_rng = np.random.default_rng(resample_seed ^ name_salt)
        keep = resample_rng.random(len(train_idx)) < source.train_pct
        train_idx = train_idx[keep]

    idx = val_idx if split == "val" else train_idx
    idx = np.sort(idx)   # sorted access is friendlier to the memmap page cache

    for start in range(0, len(idx), chunk_rows):
        chunk_idx = idx[start:start + chunk_rows]
        records = ds.get_batch(chunk_idx)
        # wl_target() is the only part of records_to_fens_and_targets()
        # this path needs — it operates on eval_cp/game_result directly,
        # no FEN involved. The board/stm arrays are handed to the caller
        # as-is for encode_mailbox_batch().
        y = dataset.wl_target(records["eval_cp"], records["game_result"], k=source.k)
        yield records["board"], records["stm"], y


# ════════════════════════════════════════════════════════════════════════
#  Chunk streaming across all sources (train or val)
# ════════════════════════════════════════════════════════════════════════

# ── Position-validity filter ────────────────────────────────────────
#
# Why this exists: the parquet datasets under data/ mix real positions
# with synthetic stress boards that are NOT legal chess (observed: 47
# non-king pieces, sides with no king, 31-32 piece boards). v3.14's dense
# encoder set bits for them and trained on them without complaint. Those
# rows teach the net nothing and cost real training time, so they are
# dropped here — loudly, with a count, rather than silently.
#
# The check is deliberately cheap (string scan, no python-chess Board
# construction): it runs on every row of a multi-million-row stream.
_DROP_STATS = {"dropped": 0, "seen": 0, "warned": False}


def _fen_is_trainable(fen: str) -> bool:
    """True if `fen`'s piece placement is a plausible legal position:
    exactly one king per side and at most 30 non-king pieces."""
    board_part = fen.split(" ", 1)[0]
    n_nonking = 0
    wk = bk = 0
    for ch in board_part:
        if ch == "/" or ch.isdigit():
            continue
        if ch == "K":
            wk += 1
        elif ch == "k":
            bk += 1
        else:
            n_nonking += 1
            if n_nonking > LEGAL_MAX_ACTIVE_FEATURES:
                return False
    return wk == 1 and bk == 1


# Zchezz mailbox piece codes for the two kings (COL_W=8+type6=14, COL_B=16+type6=22).
_WK_CODE = 8 | 6
_BK_CODE = 16 | 6


def _mailbox_is_trainable(boards: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of `_fen_is_trainable`, operating directly on
    a (N, 64) mailbox board batch (no FEN string built for the check
    either — see encode_mailbox_batch's docstring for why the `.bin` path
    avoids FEN strings end to end). Returns an (N,) bool mask."""
    n_wk = (boards == _WK_CODE).sum(axis=1)
    n_bk = (boards == _BK_CODE).sum(axis=1)
    n_nonking = (boards != 0).sum(axis=1) - n_wk - n_bk
    return (n_wk == 1) & (n_bk == 1) & (n_nonking <= LEGAL_MAX_ACTIVE_FEATURES)


def _note_dropped(n: int, batch: int) -> None:
    _DROP_STATS["dropped"] += n
    _DROP_STATS["seen"] += batch
    if not _DROP_STATS["warned"]:
        _DROP_STATS["warned"] = True
        print(f"  [filter] dropping non-legal positions from the stream "
              f"(first batch: {n}/{batch}). Running total reported at end of epoch.")


def stream_split(sources: list[SourceSpec], split: Literal["train", "val"],
                  args: argparse.Namespace, pool: "mp.pool.Pool", epoch: int = 0):
    """Round-robins over every source's row-chunk generator, encodes each
    chunk's FENs in parallel, accumulates encoded chunks until
    `args.max_positions_chunk` positions are buffered, then yields one
    big encoded+labeled chunk as torch tensors ready for a DataLoader.

    This mirrors v3.14's two-level chunking (`yield_safe_chunks_mixed`):
    small per-source row chunks feed a larger position-count-bounded
    buffer that is what actually gets pushed to the training device.

    The train/val partition always uses the fixed `args.seed` (stable
    across epochs). When `split == "train"` and `args.resample_each_epoch`
    is set, the train_pct row subsample is redrawn every epoch using
    `args.seed + epoch` (restores v3.14's RESAMPLE_EACH_EPOCH=True
    default — see module docstring point 0); otherwise it reuses
    `args.seed` every time, pinning the same subsample forever.
    """
    resample_seed = args.seed
    if split == "train" and getattr(args, "resample_each_epoch", True):
        resample_seed = args.seed + epoch

    # Each generator is tagged with its source kind so the round-robin loop
    # below knows which accumulator to feed. Since Fix 3 (encoded-chunk
    # caching), parquet's iter_parquet_rows() already yields fully-encoded
    # (stm_idx, stm_off, opp_idx, opp_off, y) tuples straight from its
    # per-file cache — the legality filter and encode_fens_parallel() call
    # now happen once, at cache-build time, inside
    # _build_parquet_file_cache(), not here. bin still yields raw
    # (boards, stms, y) and is filtered/encoded here via
    # encode_mailbox_batch() (see that function's docstring for why the
    # `.bin` path stays FEN-free end to end). The two encoded results are
    # merged back into one batch-wide (stm_idx, stm_off, opp_idx, opp_off,
    # y) tuple at flush time.
    generators: list[tuple[Literal["parquet", "bin"], object]] = []
    for src in sources:
        if src.kind == "parquet":
            generators.append(("parquet", iter_parquet_rows(
                src, args.parquet_chunk_rows, split, args.seed, resample_seed,
                pool=pool, n_workers=args.workers,
                use_cache=getattr(args, "encode_cache", ENCODE_CACHE))))
        else:
            generators.append(("bin", iter_bin_rows(
                src, args.parquet_chunk_rows, split, args.seed, resample_seed)))

    acc_pq_stm_idx: list[np.ndarray] = []
    acc_pq_stm_off: list[np.ndarray] = []
    acc_pq_opp_idx: list[np.ndarray] = []
    acc_pq_opp_off: list[np.ndarray] = []
    acc_pq_y: list[np.ndarray] = []
    acc_boards: list[np.ndarray] = []
    acc_stms: list[np.ndarray] = []
    acc_bin_y: list[np.ndarray] = []
    acc_n = 0

    def _flush():
        nonlocal acc_pq_stm_idx, acc_pq_stm_off, acc_pq_opp_idx, acc_pq_opp_off, acc_pq_y
        nonlocal acc_boards, acc_stms, acc_bin_y, acc_n
        if acc_n == 0:
            return None

        stm_idx_parts: list[np.ndarray] = []
        stm_off_parts: list[np.ndarray] = []
        opp_idx_parts: list[np.ndarray] = []
        opp_off_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        stm_running = 0
        opp_running = 0

        # ── Parquet part: already encoded+filtered by iter_parquet_rows'
        #    per-file cache — just re-base offsets and concatenate. ──
        if acc_pq_y:
            for stm_idx, stm_off, opp_idx, opp_off, y_flat in zip(
                    acc_pq_stm_idx, acc_pq_stm_off, acc_pq_opp_idx, acc_pq_opp_off, acc_pq_y):
                stm_idx_parts.append(stm_idx)
                stm_off_parts.append(stm_off + stm_running)
                stm_running += len(stm_idx)
                opp_idx_parts.append(opp_idx)
                opp_off_parts.append(opp_off + opp_running)
                opp_running += len(opp_idx)
                y_parts.append(y_flat)

        # ── Bin/mailbox part: filter invalid, encode via encode_mailbox_batch ──
        if acc_boards:
            boards = np.concatenate(acc_boards, axis=0)
            stms = np.concatenate(acc_stms, axis=0)
            y_flat = np.concatenate(acc_bin_y).astype(np.float32)
            keep_mask = _mailbox_is_trainable(boards)
            n_dropped = int((~keep_mask).sum())
            if n_dropped:
                _note_dropped(n_dropped, len(boards))
            if keep_mask.any():
                boards = boards[keep_mask]
                stms = stms[keep_mask]
                y_flat = y_flat[keep_mask]
                stm_idx, stm_off, opp_idx, opp_off = encode_mailbox_batch(boards, stms)
                stm_idx_parts.append(stm_idx)
                stm_off_parts.append(stm_off + stm_running)
                stm_running += len(stm_idx)
                opp_idx_parts.append(opp_idx)
                opp_off_parts.append(opp_off + opp_running)
                opp_running += len(opp_idx)
                y_parts.append(y_flat)

        acc_pq_stm_idx, acc_pq_stm_off, acc_pq_opp_idx, acc_pq_opp_off, acc_pq_y = [], [], [], [], []
        acc_boards, acc_stms, acc_bin_y = [], [], []
        acc_n = 0

        if not y_parts:
            return None

        stm_idx = (np.concatenate(stm_idx_parts) if stm_idx_parts
                   else np.zeros(0, dtype=np.int32))
        stm_off = (np.concatenate(stm_off_parts) if stm_off_parts
                   else np.zeros(0, dtype=np.int64))
        opp_idx = (np.concatenate(opp_idx_parts) if opp_idx_parts
                   else np.zeros(0, dtype=np.int32))
        opp_off = (np.concatenate(opp_off_parts) if opp_off_parts
                   else np.zeros(0, dtype=np.int64))
        y = np.concatenate(y_parts)
        return (
            torch.from_numpy(stm_idx).long(), torch.from_numpy(stm_off),
            torch.from_numpy(opp_idx).long(), torch.from_numpy(opp_off),
            torch.from_numpy(y),
        )

    active = list(generators)
    while active:
        still_active = []
        for kind, gen in active:
            try:
                if kind == "parquet":
                    stm_idx, stm_off, opp_idx, opp_off, y = next(gen)
                    acc_pq_stm_idx.append(stm_idx)
                    acc_pq_stm_off.append(stm_off)
                    acc_pq_opp_idx.append(opp_idx)
                    acc_pq_opp_off.append(opp_off)
                    acc_pq_y.append(y)
                    acc_n += len(stm_off)
                else:
                    boards, stms, y = next(gen)
                    acc_boards.append(boards)
                    acc_stms.append(stms)
                    acc_bin_y.append(y)
                    acc_n += len(boards)
            except StopIteration:
                continue
            still_active.append((kind, gen))

            if acc_n >= args.max_positions_chunk:
                chunk = _flush()
                if chunk is not None:
                    yield chunk

        active = still_active

    chunk = _flush()
    if chunk is not None:
        yield chunk


# ════════════════════════════════════════════════════════════════════════
#  Checkpoint I/O
# ════════════════════════════════════════════════════════════════════════

ARCH_DICT = {
    "input": 2560,
    "h1": 512,
    "concat": 1024,
    "h2": 32,
    "encoding": "halfkp_4bucket",
}


def find_latest_checkpoint(ckpt_dir: str) -> str | None:
    if not os.path.isdir(ckpt_dir):
        return None
    candidates = [f for f in os.listdir(ckpt_dir) if f.startswith("nnue_v400_") and f.endswith(".pt")]
    if not candidates:
        return None
    candidates.sort(key=lambda f: os.path.getmtime(os.path.join(ckpt_dir, f)))
    return os.path.join(ckpt_dir, candidates[-1])


def save_checkpoint(ckpt_dir: str, dataset_name: str, epoch: int, model: NNUE,
                     optimizer: torch.optim.Optimizer, train_loss: float, val_loss: float | None,
                     train_mae: float | None = None, val_mae: float | None = None) -> str:
    os.makedirs(ckpt_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(ckpt_dir, f"nnue_v400_{dataset_name}_epoch{epoch:02d}_{timestamp}.pt")
    torch.save({
        "epoch": epoch,
        "dataset": dataset_name,
        "avg_loss": train_loss,
        "val_loss": val_loss,
        "train_mae": train_mae,   # mean |sigmoid(pred) - target| over the epoch (restores v3.14's epoch_mae)
        "val_mae": val_mae,       # same, on the held-out split — new vs. v3.14, which had no val pass at all
        "lr": optimizer.param_groups[0]["lr"],
        "timestamp": timestamp,
        "arch": ARCH_DICT,
        "qat": True,
        "qa": QA,
        "qb": QB,
        "weights": model.state_dict(),
    }, path)
    return path


# ════════════════════════════════════════════════════════════════════════
#  Training loop
# ════════════════════════════════════════════════════════════════════════

def train(args: argparse.Namespace) -> None:
    # No --source on the command line? Use the DATASETS block at the top of
    # this file. That block — not the CLI — is the normal way to configure a
    # run (CLAUDE.md rule 8); --source exists only to override it for
    # scripted sweeps.
    if not args.sources:
        args.sources = sources_from_config()
        if not args.sources:
            raise SystemExit(
                "train_nnue.py: no data sources. Either set pct > 0 for at least one\n"
                "entry in the DATASETS block at the top of this file, or pass --source.")
        print("Sources (from the DATASETS block at the top of this file):")
    else:
        print("Sources (from --source on the command line, overriding the DATASETS block):")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Resample each epoch: {args.resample_each_epoch} "
          f"(sources with train_pct<1.0 get a fresh row subsample per epoch when True)")
    for src in args.sources:
        print(f"  source[{src.kind}] {src.name}: path={src.path} "
              f"{'target_col=' + src.target_col if src.kind == 'parquet' else 'k=' + str(src.k)} "
              f"train_pct={src.train_pct} val_frac={src.val_frac}")

    model = NNUE().to(device)

    start_epoch = 0
    resume_lr = args.lr
    latest = find_latest_checkpoint(args.ckpt_dir)
    if latest:
        # Mirrors v3.14 mixtrain.py's try/except around checkpoint load: a
        # corrupt file, an incompatible/renamed state_dict key, or any other
        # load-time error must fall back to a fresh random-init run rather
        # than crashing the whole training script.
        try:
            ckpt = torch.load(latest, map_location=device)
            model.load_state_dict(ckpt["weights"])
            if ckpt.get("dataset") == args.dataset_name:
                start_epoch = ckpt["epoch"]
                resume_lr = ckpt.get("lr", args.lr)
                print(f"Resumed from {latest} (same dataset) at epoch {start_epoch}, lr={resume_lr:.2e}")
            else:
                start_epoch = 0
                resume_lr = args.transfer_lr
                print(f"Weight-transfer from {latest} (dataset {ckpt.get('dataset')!r} != "
                      f"{args.dataset_name!r}); starting epoch 0 at lr={resume_lr:.2e}")
        except Exception as e:
            start_epoch = 0
            resume_lr = args.lr
            print(f"WARNING: checkpoint at {latest} is incompatible or corrupt ({e!r}); "
                  f"falling back to fresh random-init training at epoch 0, lr={resume_lr:.2e}.")
    else:
        print("No checkpoint found — training from random init.")

    optimizer = torch.optim.Adam(model.parameters(), lr=resume_lr, weight_decay=args.weight_decay)
    if start_epoch > 0:
        # CosineAnnealingLR requires 'initial_lr' to already be present in
        # each param_group whenever it's constructed with last_epoch != -1
        # (it only sets that key itself on a fresh, last_epoch=-1 init) —
        # without this, resuming raises KeyError('initial_lr').
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", resume_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=1e-6,
        last_epoch=start_epoch - 1 if start_epoch > 0 else -1,
    )
    loss_fn = nn.BCELoss()

    # Previous epoch's batch count, used to give the mid-epoch heartbeat a
    # rough ETA (progress_frac = batches_so_far / prev_epoch_batches).
    # None on the very first epoch of the run, when there is no prior
    # epoch to compare against.
    prev_epoch_batches: int | None = None

    ctx = mp.get_context("spawn" if os.name == "nt" else "fork")
    with ctx.Pool(processes=args.workers) as pool:
        for epoch in range(start_epoch, args.epochs):
            print(f"\n{'=' * 80}\nEPOCH {epoch + 1}/{args.epochs}  lr={optimizer.param_groups[0]['lr']:.2e}\n{'=' * 80}")

            model.train()
            epoch_loss, epoch_mae, epoch_batches = 0.0, 0.0, 0
            t0 = time.time()

            for stm_idx, stm_off, opp_idx, opp_off, y in stream_split(args.sources, "train", args, pool, epoch=epoch):
                n = len(stm_off)

                # Batches are sliced directly from the already-built bag
                # tensors (no torch DataLoader/TensorDataset here): a
                # DataLoader shuffling OFFSET rows would still need the
                # per-sample index slices re-derived from the shuffled
                # offsets, which is exactly what this direct slicing does
                # up front, without the extra indirection. Within-chunk
                # position order is already effectively randomized by
                # stream_split's round-robin interleaving of sources, and
                # each source's own row order comes from a seeded
                # permutation (parquet: per-row RNG draw; bin: np.random
                # permutation of the source's global index range).
                for start in range(0, n, args.batch_size):
                    end = min(start + args.batch_size, n)
                    b_stm_idx, b_stm_off = _slice_bag(stm_idx, stm_off, start, end)
                    b_opp_idx, b_opp_off = _slice_bag(opp_idx, opp_off, start, end)
                    yb = y[start:end].to(device)

                    optimizer.zero_grad(set_to_none=True)
                    preds = model(
                        b_stm_idx.to(device), b_stm_off.to(device),
                        b_opp_idx.to(device), b_opp_off.to(device),
                    )
                    loss = loss_fn(preds, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    clamp_weights_(model)

                    with torch.no_grad():
                        # Mean absolute error between the model's sigmoid
                        # output and the same training target the BCE loss
                        # used — restores v3.14 mixtrain.py's epoch_mae.
                        epoch_mae += torch.abs(preds - yb).mean().item()
                    epoch_loss += loss.item()
                    epoch_batches += 1

                    # Mid-epoch progress heartbeat (restores v3.14's
                    # PRINT_EVERY_N_CHUNKS status line, ~every N batches
                    # instead of every N chunks — see HEARTBEAT_EVERY_BATCHES).
                    if epoch_batches % args.heartbeat_every_batches == 0:
                        elapsed_h = time.time() - t0
                        eta_str = "n/a"
                        if prev_epoch_batches:
                            frac = min(1.0, epoch_batches / prev_epoch_batches)
                            if frac > 0:
                                eta_s = max(0.0, elapsed_h / frac - elapsed_h)
                                eta_str = f"{eta_s / 60:.1f}m"
                        print(f"    [heartbeat] batch {epoch_batches} | "
                              f"loss {epoch_loss / epoch_batches:.5f} | "
                              f"mae {epoch_mae / epoch_batches:.5f} | "
                              f"elapsed {elapsed_h:.0f}s | ETA {eta_str}")

            scheduler.step()
            avg_train_loss = epoch_loss / max(1, epoch_batches)
            avg_train_mae = epoch_mae / max(1, epoch_batches)
            elapsed = time.time() - t0
            print(f"  train loss: {avg_train_loss:.5f}  mae: {avg_train_mae:.5f}  "
                  f"({epoch_batches} batches, {elapsed:.1f}s)")
            prev_epoch_batches = epoch_batches

            val_loss = val_mae = None
            if (epoch + 1) % args.val_every == 0:
                val_loss, val_mae = evaluate(model, args, pool, device, loss_fn)
                print(f"  val   loss: {val_loss:.5f}  mae: {val_mae:.5f}")

            ckpt_path = save_checkpoint(args.ckpt_dir, args.dataset_name, epoch + 1,
                                         model, optimizer, avg_train_loss, val_loss,
                                         train_mae=avg_train_mae, val_mae=val_mae)
            print(f"  checkpoint: {ckpt_path}")
            if device.type == "cuda":
                print(f"  Peak GPU VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    print("\nTraining complete.")


def _slice_bag(flat_idx: torch.Tensor, offsets: torch.Tensor, start: int, end: int
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice a [flat_idx, offsets] EmbeddingBag pair to the sample range
    [start, end) and return a new (flat_idx, offsets) pair with offsets
    rebased to 0, ready to feed to F.embedding_bag for just that batch."""
    n = offsets.shape[0]
    lo = int(offsets[start].item())
    hi = int(offsets[end].item()) if end < n else flat_idx.shape[0]
    batch_idx = flat_idx[lo:hi]
    batch_off = offsets[start:end] - offsets[start]
    return batch_idx, batch_off


@torch.no_grad()
def evaluate(model: NNUE, args: argparse.Namespace, pool: "mp.pool.Pool",
             device: torch.device, loss_fn: nn.Module) -> tuple[float, float]:
    """Returns (avg_loss, avg_mae) over the held-out validation split.
    val MAE is a genuine v400 improvement over v3.14, which only ever
    computed MAE on training minibatches (see module docstring point 3)."""
    model.eval()
    total_loss, total_mae, total_batches = 0.0, 0.0, 0
    for stm_idx, stm_off, opp_idx, opp_off, y in stream_split(args.sources, "val", args, pool):
        n = len(stm_off)
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            b_stm_idx, b_stm_off = _slice_bag(stm_idx, stm_off, start, end)
            b_opp_idx, b_opp_off = _slice_bag(opp_idx, opp_off, start, end)
            yb = y[start:end].to(device)
            preds = model(
                b_stm_idx.to(device), b_stm_off.to(device),
                b_opp_idx.to(device), b_opp_off.to(device),
            )
            loss = loss_fn(preds, yb)
            total_loss += loss.item()
            total_mae += torch.abs(preds - yb).mean().item()
            total_batches += 1
    model.train()
    return total_loss / max(1, total_batches), total_mae / max(1, total_batches)


if __name__ == "__main__":
    parser = build_arg_parser()
    parsed_args = parser.parse_args()
    if parsed_args.show_config:
        print("Resolved configuration:")
        width = max(len(k) for k in vars(parsed_args))
        for key, val in sorted(vars(parsed_args).items()):
            if key not in ("show_config", "sources"):
                print(f"  {key.ljust(width)} = {val!r}")
        for d in DATASETS:
            print(f"  dataset: {d['name']}  pct={d['pct']}  mode={d['mode']}  lam={d['lam']}")
        raise SystemExit(0)
    train(parsed_args)
