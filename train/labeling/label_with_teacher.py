#!/usr/bin/env python3
"""Label Zchezz native self-play positions with a stronger UCI teacher.

The v5.00 training loop deliberately separates POSITION DISTRIBUTION from
TARGET QUALITY:

* positions come from zchezz_v500 self-play, so the net trains where the
  engine actually searches and makes mistakes;
* cp comes from a stronger teacher (v3.14 by default), not from the same
  student that generated the positions;
* real game result is preserved separately.  The trainer decides the blend
  later; this script never bakes lambda into the dataset.

Input is the packed .bin format from engine/c/tools/selfplay.c.  Output is
Parquet with canonical WHITE-relative columns:

    fen, cp, result, student_cp, teacher_delta_cp, generation

`cp` is the teacher score. `student_cp` is the score stored by the v5.00
self-play generator. Both are WHITE-relative in the Parquet output.

Example:
    python train/labeling/label_with_teacher.py \
        --input "data/v500_selfplay/gen00/*.bin" \
        --output data/v500_teacher_v314/gen00 \
        --teacher engine/c/zchezz_v314/zchezz.exe \
        --nodes 10000 --max-rows 100000 --workers 8 --generation 0
"""

from __future__ import annotations

import argparse
import atexit
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
TRAIN_DIR = ROOT / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

import dataset  # noqa: E402


# Keep mate labels finite. sigmoid(3000/320) is already effectively 1.0.
CP_CLIP = 3000
DEFAULT_NODES = 10_000
DEFAULT_MAX_ROWS = 100_000
DEFAULT_WORKERS = max(1, min(8, (os.cpu_count() or 4) // 2))
DEFAULT_SHARD_ROWS = 10_000
DEFAULT_SEED = 500_314

_PIECE_TO_FEN = {
    9: "P", 10: "N", 11: "B", 12: "R", 13: "Q", 14: "K",
    17: "p", 18: "n", 19: "b", 20: "r", 21: "q", 22: "k",
}

_ENGINE = None
_LIMIT = None


def sample_to_fen(rec: np.void) -> str:
    """Convert one packed SelfplaySample to a legal FEN.

    Zchezz mailbox square 0 is a8, so the stored 64-byte board is already in
    FEN rank order. ep_file stores only the file; the target rank follows
    from side to move (white to move -> black just moved -> rank 6).
    """
    board = rec["board"]
    ranks: list[str] = []
    for r in range(8):
        empty = 0
        out: list[str] = []
        for f in range(8):
            p = int(board[r * 8 + f])
            if p == 0:
                empty += 1
                continue
            if empty:
                out.append(str(empty))
                empty = 0
            try:
                out.append(_PIECE_TO_FEN[p])
            except KeyError as exc:
                raise ValueError(f"unknown Zchezz piece code {p}") from exc
        if empty:
            out.append(str(empty))
        ranks.append("".join(out))

    stm = int(rec["stm"])
    side = "w" if stm == 0 else "b"
    ca = int(rec["castling"])
    castling = ""
    if ca & 1: castling += "K"
    if ca & 2: castling += "Q"
    if ca & 4: castling += "k"
    if ca & 8: castling += "q"
    if not castling:
        castling = "-"

    ep_file = int(rec["ep_file"])
    if 0 <= ep_file <= 7:
        ep = chr(ord("a") + ep_file) + ("6" if stm == 0 else "3")
    else:
        ep = "-"

    return f"{'/'.join(ranks)} {side} {castling} {ep} {int(rec['rule50'])} 1"


def _worker_init(engine_path: str, nodes: int) -> None:
    global _ENGINE, _LIMIT
    import chess.engine

    _ENGINE = chess.engine.SimpleEngine.popen_uci(engine_path)
    _LIMIT = chess.engine.Limit(nodes=nodes)

    def _close() -> None:
        global _ENGINE
        if _ENGINE is not None:
            try:
                _ENGINE.quit()
            except Exception:
                pass
            _ENGINE = None

    atexit.register(_close)


def _teacher_label(item: tuple[str, float, int]) -> tuple[str, float, int, int, int]:
    """Return fen/result/stm/student_cp_white/teacher_cp_white."""
    fen, result_white, student_cp_white = item
    import chess

    board = chess.Board(fen)
    info = _ENGINE.analyse(board, _LIMIT)
    score = info["score"].pov(chess.WHITE).score(mate_score=CP_CLIP)
    if score is None:
        score = 0
    score = max(-CP_CLIP, min(CP_CLIP, int(score)))
    return fen, result_white, student_cp_white, score, score - student_cp_white


def _payload_from_records(records: np.ndarray) -> list[tuple[str, float, int]]:
    payload: list[tuple[str, float, int]] = []
    for rec in records:
        stm = int(rec["stm"])
        g = float(rec["game_result"])
        result_stm = (g + 1.0) / 2.0
        result_white = result_stm if stm == 0 else 1.0 - result_stm
        student_stm = int(rec["eval_cp"])
        student_white = student_stm if stm == 0 else -student_stm
        payload.append((sample_to_fen(rec), result_white, student_white))
    return payload


def label(args: argparse.Namespace) -> int:
    import pandas as pd

    teacher = Path(args.teacher).resolve()
    if not teacher.is_file():
        raise SystemExit(f"teacher engine not found: {teacher}")

    ds = dataset.MultiShardSelfplay.from_glob(args.input)
    total = len(ds)
    if total <= 0:
        raise SystemExit("input contains zero self-play records")

    take = total if args.max_rows <= 0 else min(total, args.max_rows)
    rng = np.random.default_rng(args.seed)
    if take == total:
        indices = np.arange(total, dtype=np.int64)
    else:
        indices = np.sort(rng.choice(total, size=take, replace=False).astype(np.int64))

    print(f"[teacher] input_rows={total:,} sampled={take:,} nodes={args.nodes:,} workers={args.workers}")
    records = ds.get_batch(indices)
    payload = _payload_from_records(records)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    shard = 0
    done = 0

    def flush() -> None:
        nonlocal rows, shard
        if not rows:
            return
        dst = out_dir / f"teacher_{shard:05d}.parquet"
        pd.DataFrame.from_records(rows).to_parquet(dst, index=False)
        print(f"[teacher] wrote {dst} rows={len(rows):,}")
        rows = []
        shard += 1

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init,
        initargs=(str(teacher), args.nodes),
    ) as ex:
        for fen, result_white, student_cp, teacher_cp, delta in ex.map(
            _teacher_label, payload, chunksize=max(1, args.chunksize)
        ):
            rows.append({
                "fen": fen,
                "cp": int(teacher_cp),
                "result": float(result_white),
                "student_cp": int(student_cp),
                "teacher_delta_cp": int(delta),
                "generation": int(args.generation),
            })
            done += 1
            if len(rows) >= args.shard_rows:
                flush()
            if done % 1000 == 0:
                print(f"[teacher] {done:,}/{take:,}")

    flush()
    print(f"[teacher] complete rows={done:,} shards={shard}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Label v5.00 self-play with a stronger UCI teacher")
    p.add_argument("--input", required=True, help="glob for packed .bin self-play shards")
    p.add_argument("--output", required=True, help="directory for teacher_*.parquet shards")
    p.add_argument("--teacher", default=str(ROOT / "engine/c/zchezz_v314/zchezz.exe"))
    p.add_argument("--nodes", type=int, default=DEFAULT_NODES)
    p.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS,
                   help="0 = label every input row")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--chunksize", type=int, default=4)
    p.add_argument("--shard-rows", type=int, default=DEFAULT_SHARD_ROWS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--generation", type=int, default=0)
    return p


if __name__ == "__main__":
    raise SystemExit(label(build_parser().parse_args()))
