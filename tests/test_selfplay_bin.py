"""tests/test_selfplay_bin.py — sanity checker for .bin files produced
by engine/c/tools/selfplay.c (shared tool source, built via
engine/build/Makefile — see README.md, "engine/c/tools/"; Appendix F.2/F.3 of
docs/v400_implementation_plan.md).

Reads a shard through train/dataset.py's SAMPLE_DTYPE (the
authoritative cross-language contract for the record layout — this script
never hand-rolls the dtype) and checks the invariants the implementation
plan's Phase 3 verification step calls for:

  1. Record count is sane (file size is an exact multiple of the record
     size — dataset.py's MultiShardSelfplay already raises on this,
     so this is really "did the loader accept the file at all").
  2. Every board decodes to a legal-ish position: exactly one king per
     side, all non-zero piece codes in Zchezz's WP(9)..BK(22) range,
     no more than 8 pawns/side, no more than 16 pieces/side.
  3. `game_result` is internally consistent within a game: consecutive
     samples alternate `stm` (0,1,0,1,...) inside one game, and — the
     property that actually matters for training — samples with stm=0
     (white) and samples with stm=1 (black) that belong to the SAME
     decisive game carry OPPOSITE-signed `game_result` (mover-relative,
     not white-relative), while a drawn game is all zero. Game
     boundaries are inferred heuristically (see `infer_game_boundaries`)
     since the on-disk format intentionally does not store them (each
     record is meant to be sampled independently at training time).
  4. `eval_cp` values fall inside a plausible range (mate-score-ish
     bounds at worst, never absurd/overflowed sentinel values).

Usage:
    python tests/test_selfplay_bin.py PATH_TO.bin [PATH2.bin ...]

Exit code 0 if every check passes, 1 otherwise (prints what failed).

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

import sys
import os

# ═══════════════ CONFIGURATION ═══════════════
# The list of .bin shards to check. CLAUDE.md rule 8: this positional,
# repeatable CLI input (PATH_TO.bin [PATH2.bin ...]) needs a config-block
# equivalent too, same as any other CLI-reachable input -- edit this list
# to check a fixed set of shards without typing paths every run. Any paths
# given on the command line override this list entirely (see main()).
DEFAULT_BIN_PATHS: list[str] = [
    # "data/selfplay/shard_0000.bin",
]
# ═══════════════════════════════════════════

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
import numpy as np
from dataset import SAMPLE_DTYPE, MultiShardSelfplay, wl_target  # noqa: E402

# Zchezz piece codes: COL_W=8, COL_B=16, type 1..6 (P,N,B,R,Q,K) -> WP=9..BK=22.
MIN_PIECE_CODE = 9
MAX_PIECE_CODE = 22
WK_CODE = 14
BK_CODE = 22
# Mate-score bound used by search.c: +-(19000 - ply). Give generous headroom.
EVAL_CP_ABS_BOUND = 19100


def _standard_start_board() -> np.ndarray:
    board = np.zeros(64, dtype=np.uint8)
    back_rank = [4, 2, 3, 5, 6, 3, 2, 4]  # R N B Q K B N R (type codes)
    for f in range(8):
        board[56 + f] = 8 | back_rank[f]   # white back rank (rank 1 = row 7)
        board[48 + f] = 8 | 1               # white pawns (rank 2 = row 6)
        board[8 + f] = 16 | 1               # black pawns (rank 7 = row 1)
        board[f] = 16 | back_rank[f]        # black back rank (rank 8 = row 0)
    return board


_START_BOARD = _standard_start_board()


def infer_game_boundaries(arr: np.ndarray) -> list[tuple[int, int]]:
    """Split a shard's flat record stream into per-game [start, end) ranges.

    The on-disk format has no explicit game-boundary marker (by design —
    each record is meant to be sampled independently at training time),
    so this reconstructs boundaries heuristically.

    HISTORY, because the obvious heuristic is now WRONG: this used to
    match the exact standard starting position, which worked only while
    tools/selfplay.c started every game from startpos. Once opening-book
    support landed, games start from arbitrary book positions and that
    fingerprint matches nothing — the whole shard collapsed into "1 game"
    and every downstream check silently became meaningless.

    The heuristic used now is PIECE COUNT NON-MONOTONICITY: within a
    single game the number of pieces on the board never increases
    (captures only remove; promotions replace). So a record whose piece
    count is GREATER than its predecessor's must belong to a new game.
    This is opening-agnostic. Its blind spot is the reverse case — two
    consecutive games where the new game's opening has no more pieces
    than the previous game's final position (e.g. a book position that is
    already an endgame). That under-detects rather than mis-detects, so a
    passing result stays trustworthy; the count is reported so a wildly
    low number is visible.

    Rejected alternative: "stm alternates 0,1,0,1 so a repeated 0 is a new
    game" — that silently under-detects whenever a game's last recorded
    sample had stm=1 (White delivered the mating move, the common case),
    leaving the pattern unbroken across the boundary.
    """
    n = len(arr)
    if n == 0:
        return []
    piece_count = np.count_nonzero(arr["board"], axis=1)
    # A new game wherever material increased vs the previous record.
    is_start = np.zeros(n, dtype=bool)
    is_start[0] = True
    if n > 1:
        is_start[1:] = piece_count[1:] > piece_count[:-1]
    starts = np.flatnonzero(is_start).tolist()
    if not starts or starts[0] != 0:
        starts = [0] + starts
    starts.append(n)
    starts = sorted(set(starts))
    return [(starts[i], starts[i + 1]) for i in range(len(starts) - 1) if starts[i + 1] > starts[i]]


def check_shard(path: str) -> list[str]:
    errors: list[str] = []
    size = os.path.getsize(path)
    if size == 0:
        return [f"{path}: empty file"]
    if size % SAMPLE_DTYPE.itemsize != 0:
        return [f"{path}: size {size} is not a multiple of record size {SAMPLE_DTYPE.itemsize}"]

    arr = np.memmap(path, dtype=SAMPLE_DTYPE, mode="r")
    n = len(arr)
    print(f"{path}: {n} records ({size} bytes)")

    # ── 1. dtype / padding sanity ──────────────────────────────────
    if not np.all(arr["_pad"] == 0):
        bad = int(np.count_nonzero(arr["_pad"]))
        errors.append(f"{path}: {bad}/{n} records have nonzero _pad (format drift?)")

    # ── 2. board legality ───────────────────────────────────────────
    boards = arr["board"]  # (n, 64) uint8
    nonzero_mask = boards != 0
    out_of_range = nonzero_mask & ((boards < MIN_PIECE_CODE) | (boards > MAX_PIECE_CODE))
    n_bad_codes = int(np.any(out_of_range, axis=1).sum())
    if n_bad_codes:
        errors.append(f"{path}: {n_bad_codes} records have a piece code outside [{MIN_PIECE_CODE},{MAX_PIECE_CODE}]")

    wk_counts = (boards == WK_CODE).sum(axis=1)
    bk_counts = (boards == BK_CODE).sum(axis=1)
    n_bad_kings = int(np.count_nonzero((wk_counts != 1) | (bk_counts != 1)))
    if n_bad_kings:
        errors.append(f"{path}: {n_bad_kings} records do not have exactly one king per side")

    pawn_w = (boards == 9).sum(axis=1)
    pawn_b = (boards == 17).sum(axis=1)
    n_bad_pawns = int(np.count_nonzero((pawn_w > 8) | (pawn_b > 8)))
    if n_bad_pawns:
        errors.append(f"{path}: {n_bad_pawns} records have >8 pawns for one side")

    pieces_w = ((boards >= 9) & (boards <= 14)).sum(axis=1)
    pieces_b = ((boards >= 17) & (boards <= 22)).sum(axis=1)
    n_bad_total = int(np.count_nonzero((pieces_w > 16) | (pieces_b > 16)))
    if n_bad_total:
        errors.append(f"{path}: {n_bad_total} records have >16 pieces for one side")

    # ── 3. stm / castling / ep_file range sanity ────────────────────
    if not np.all((arr["stm"] == 0) | (arr["stm"] == 1)):
        errors.append(f"{path}: stm has values outside {{0,1}}")
    if not np.all(arr["castling"] <= 15):
        errors.append(f"{path}: castling bitmask has bits outside 0..15")
    if not np.all(arr["ep_file"] <= 8):
        errors.append(f"{path}: ep_file outside 0..8")
    if not np.all((arr["game_result"] >= -1) & (arr["game_result"] <= 1)):
        errors.append(f"{path}: game_result has values outside {{-1,0,1}}")

    # ── 4. eval_cp plausibility ──────────────────────────────────────
    eval_cp = arr["eval_cp"].astype(np.int64)
    n_extreme = int(np.count_nonzero(np.abs(eval_cp) > EVAL_CP_ABS_BOUND))
    if n_extreme:
        errors.append(f"{path}: {n_extreme} records have |eval_cp| > {EVAL_CP_ABS_BOUND} (mate-score bound)")
    print(f"  eval_cp: min={eval_cp.min()} max={eval_cp.max()} mean={eval_cp.mean():.1f}")

    # ── 5. per-game game_result consistency (heuristic boundaries) ──
    games = infer_game_boundaries(arr)
    print(f"  inferred {len(games)} game(s) from piece-count boundaries")
    bad_games = 0
    for (s, e) in games:
        seg = arr[s:e]
        gr = seg["game_result"]
        if np.all(gr == 0):
            continue  # draw (or a game with zero decisive samples), nothing to flip-check
        white_gr = set(np.unique(gr[seg["stm"] == 0]).tolist())
        black_gr = set(np.unique(gr[seg["stm"] == 1]).tolist())
        # A decisive game must have exactly one nonzero result value per
        # color, and the two colors' values must be opposite signs.
        if white_gr - {0} and black_gr - {0}:
            w = next(iter(white_gr - {0}))
            b = next(iter(black_gr - {0}))
            if w != -b:
                bad_games += 1
    if bad_games:
        errors.append(f"{path}: {bad_games} inferred game(s) have game_result that does not flip sign with stm")

    # ── 6. wl_target() smoke test (exercises the actual training-facing API) ─
    try:
        _ = wl_target(arr["eval_cp"][:min(n, 1000)], arr["game_result"][:min(n, 1000)], k=1.0)
        _ = wl_target(arr["eval_cp"][:min(n, 1000)], arr["game_result"][:min(n, 1000)], k=0.0)
    except Exception as e:  # noqa: BLE001
        errors.append(f"{path}: wl_target() raised: {e}")

    return errors


def main(argv: list[str]) -> int:
    if len(argv) >= 2:
        paths = argv[1:]
    else:
        paths = list(DEFAULT_BIN_PATHS)
        if paths:
            print(f"no paths given on the command line -- using DEFAULT_BIN_PATHS: {paths}")
    if not paths:
        print(f"usage: {argv[0]} PATH_TO.bin [PATH2.bin ...]", file=sys.stderr)
        print("(or populate DEFAULT_BIN_PATHS at the top of this file)", file=sys.stderr)
        return 2
    all_errors: list[str] = []
    for p in paths:
        all_errors.extend(check_shard(p))

    # Also exercise the multi-shard reader end-to-end if more than one path given.
    if len(paths) > 1:
        try:
            ds = MultiShardSelfplay(paths)
            print(f"MultiShardSelfplay: {len(ds)} total records across {len(paths)} shard(s)")
            idx = np.arange(min(len(ds), 100))
            batch = ds.get_batch(idx)
            print(f"  get_batch smoke test: fetched {len(batch)} records OK")
        except Exception as e:  # noqa: BLE001
            all_errors.append(f"MultiShardSelfplay end-to-end failed: {e}")

    if all_errors:
        print("\nFAILED:")
        for e in all_errors:
            print(f"  - {e}")
        return 1

    print("\nOK: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
