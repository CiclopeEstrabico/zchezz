#!/usr/bin/env python3
"""tests/test_mailbox_encoder.py — parity check for encoding.encode_mailbox_batch()

train/encoding.py's `encode_mailbox_batch()` encodes HalfKP-4Bucket features
directly from a `.bin` record's packed mailbox `board[]`/`stm` array, with
numpy, instead of the old FEN-string round-trip (`dataset.py`'s
`records_to_fens_and_targets()` -> `chess.Board(fen)` -> `piece_map()`,
walked one square at a time by python-chess).

A silent divergence between the two encoders would poison every future net
without ever raising an error (both paths produce syntactically valid
feature-index arrays; only their VALUES could differ). This script is the
guard against that: for every record in a real self-play `.bin` file, it
computes the active feature-index SET for both perspectives via both paths
and asserts they are identical.

Sets, not arrays, are compared: EmbeddingBag(mode='sum') / index_add are
order-invariant reductions (see encoding.py's module docstring), so index
ORDER within a position is not part of the contract, only the SET of
active indices is.

Usage:
  python tests/test_mailbox_encoder.py [path/to/file.bin]

  With no argument, generates a fresh fixture via engine/build/selfplay.exe
  (8 games, 2 threads, movetime=40ms, tests/random_nnu4.bin weights) into
  the scratch dir below, exactly as this test is meant to be exercised.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np
import chess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
import encoding as enc  # noqa: E402
import dataset  # noqa: E402

# ═══════════════════════════════ CONFIGURATION ═══════════════════════════════
BIN_PATH = None                 # path to an existing .bin fixture; None = generate one
SELFPLAY_GAMES = 8              # games to generate when BIN_PATH is None
SELFPLAY_THREADS = 2            # worker threads for the generator run
SELFPLAY_MOVETIME_MS = 40       # per-move budget, milliseconds
SELFPLAY_NNUE = os.path.join(os.path.dirname(__file__), "random_nnu4.bin")
SELFPLAY_EXE = os.path.join(os.path.dirname(__file__), "..", "engine", "build", "selfplay.exe")
BATCH_SIZE = 50_000              # rows encoded per encode_mailbox_batch() call
BIN_PATH_DEFAULT = ""            # "" keeps BIN_PATH = None (generate a fixture);
                                 # a path here is the same as passing --bin
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════ COMMAND LINE ═══════════════════════════════
# `python tests/test_mailbox_encoder.py` runs exactly what the block above
# says. The legacy positional form (`... path\to\fixture.bin`) still works and
# means --bin. `--show-config` prints the resolved settings and exits.
# ════════════════════════════════════════════════════════════════════════════
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
from cliconf import override_from_cli  # noqa: E402

CLI = [
    ("BIN_PATH_DEFAULT",     "--bin",       str, "existing .bin fixture; empty = generate one"),
    ("SELFPLAY_GAMES",       "--games",     int, "games generated when no fixture is given"),
    ("SELFPLAY_THREADS",     "--threads",   int, "worker threads for the generator run"),
    ("SELFPLAY_MOVETIME_MS", "--movetime",  int, "per-move budget for the generator, ms"),
    ("SELFPLAY_NNUE",        "--nnue",      str, "NNUE weights used by the generator"),
    ("SELFPLAY_EXE",         "--selfplay-exe", str, "selfplay.exe used to generate the fixture"),
    ("BATCH_SIZE",           "--batch-size", int, "rows per encode_mailbox_batch() call"),
]


def _generate_fixture() -> str:
    """Runs engine/build/selfplay.exe to produce a small real .bin fixture,
    exercising real search output (not synthetic boards) — the same
    command documented in the task: 8 games, 2 threads, movetime=40ms."""
    if not os.path.isfile(SELFPLAY_EXE):
        raise FileNotFoundError(
            f"selfplay.exe not found at {SELFPLAY_EXE!r} — build it first: "
            f"cd engine/build && mingw32-make ENGINE=v400 selfplay")
    tmp_dir = tempfile.mkdtemp(prefix="zchezz_mailbox_parity_")
    out_path = os.path.join(tmp_dir, "p.bin")
    cmd = [
        SELFPLAY_EXE,
        "--games", str(SELFPLAY_GAMES),
        "--threads", str(SELFPLAY_THREADS),
        "--movetime", str(SELFPLAY_MOVETIME_MS),
        "--nnue", SELFPLAY_NNUE,
        "--out", out_path,
    ]
    print(f"[fixture] generating: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=os.path.dirname(SELFPLAY_EXE))
    return out_path


def _fen_reference_indices(fen: str) -> tuple[set[int], set[int]]:
    """The OLD path: FEN string -> chess.Board -> halfkp_active_indices()."""
    board = chess.Board(fen)
    stm_idx, opp_idx = enc.encode_position_indices(board)
    return set(stm_idx.tolist()), set(opp_idx.tolist())


def main() -> int:
    bin_path = BIN_PATH_DEFAULT or BIN_PATH or _generate_fixture()
    size = os.path.getsize(bin_path)
    assert size % dataset.SAMPLE_DTYPE.itemsize == 0, (
        f"{bin_path}: size {size} is not a multiple of record size "
        f"{dataset.SAMPLE_DTYPE.itemsize}")
    n_rows = size // dataset.SAMPLE_DTYPE.itemsize
    records = np.fromfile(bin_path, dtype=dataset.SAMPLE_DTYPE, count=n_rows)
    print(f"[parity] {bin_path}: {n_rows} records")
    assert n_rows > 0, "fixture produced zero records — nothing to test"

    # ── Coverage bookkeeping — confirm the awkward cases actually occur ──
    coverage = {
        "black_to_move": 0,
        "castling_present": 0,
        "castling_absent": 0,
        "ep_available": 0,
        "endgame_le8_pieces": 0,
    }

    mismatches = 0
    first_mismatch = None
    n_checked = 0

    for start in range(0, n_rows, BATCH_SIZE):
        chunk = records[start:start + BATCH_SIZE]
        boards = chunk["board"]
        stms = chunk["stm"]

        # New path: straight from the mailbox array, batched.
        stm_idx, stm_off, opp_idx, opp_off = enc.encode_mailbox_batch(boards, stms)

        # Old path: FEN string per record, one at a time (this is exactly
        # the round-trip being removed from the real training loop; kept
        # here ONLY as the reference oracle).
        fens, _y = dataset.records_to_fens_and_targets(chunk, k=1.0)

        m = len(chunk)
        for i in range(m):
            rec = chunk[i]
            fen = fens[i]

            if int(rec["stm"]) == 1:
                coverage["black_to_move"] += 1
            if int(rec["castling"]) != 0:
                coverage["castling_present"] += 1
            else:
                coverage["castling_absent"] += 1
            if int(rec["ep_file"]) != 8:
                coverage["ep_available"] += 1
            n_pieces = int((rec["board"] != 0).sum())
            if n_pieces <= 8:
                coverage["endgame_le8_pieces"] += 1

            s0, e0 = int(stm_off[i]), int(stm_off[i + 1]) if i + 1 < m else len(stm_idx)
            o0, oe = int(opp_off[i]), int(opp_off[i + 1]) if i + 1 < m else len(opp_idx)
            got_stm = set(stm_idx[s0:e0].tolist())
            got_opp = set(opp_idx[o0:oe].tolist())

            ref_stm, ref_opp = _fen_reference_indices(fen)

            n_checked += 1
            if got_stm != ref_stm or got_opp != ref_opp:
                mismatches += 1
                if first_mismatch is None:
                    first_mismatch = {
                        "global_row": start + i,
                        "fen": fen,
                        "stm_field": int(rec["stm"]),
                        "board": rec["board"].tolist(),
                        "castling": int(rec["castling"]),
                        "ep_file": int(rec["ep_file"]),
                        "got_stm": sorted(got_stm),
                        "ref_stm": sorted(ref_stm),
                        "got_opp": sorted(got_opp),
                        "ref_opp": sorted(ref_opp),
                        "stm_diff": sorted(got_stm ^ ref_stm),
                        "opp_diff": sorted(got_opp ^ ref_opp),
                    }

    print(f"[parity] checked {n_checked} records")
    print(f"[coverage] black to move           : {coverage['black_to_move']}")
    print(f"[coverage] castling rights present  : {coverage['castling_present']}")
    print(f"[coverage] castling rights absent   : {coverage['castling_absent']}")
    print(f"[coverage] en-passant available     : {coverage['ep_available']}")
    print(f"[coverage] endgame (<=8 pieces)      : {coverage['endgame_le8_pieces']}")

    if mismatches:
        print(f"\n[FAIL] {mismatches}/{n_checked} records mismatch. First mismatch:")
        for k, v in first_mismatch.items():
            print(f"  {k}: {v}")
        return 1

    print(f"\n[PASS] 0 mismatches over {n_checked} records — "
          f"encode_mailbox_batch matches the FEN+python-chess path exactly.")
    return 0


if __name__ == "__main__":
    # Legacy positional form: `python tests/test_mailbox_encoder.py <fixture.bin>`.
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        sys.argv[1:2] = ["--bin", sys.argv[1]]
    override_from_cli(globals(), CLI, description=__doc__,
                      prog="test_mailbox_encoder.py")
    raise SystemExit(main())
