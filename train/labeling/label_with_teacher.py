#!/usr/bin/env python3
"""Label Zchezz native self-play positions with a stronger UCI teacher.

The v5.00 training loop deliberately separates POSITION DISTRIBUTION from
TARGET QUALITY:

* positions come from zchezz_v500 self-play, so the net trains where the
  engine actually searches and makes mistakes;
* cp comes from a stronger teacher (v3.14 by default), not from the same
  student that generated the positions;
* real game result is preserved separately. The trainer decides the blend
  later; this script never bakes lambda into the dataset.

Input is the packed .bin format from engine/c/tools/selfplay.c. Output is
Parquet with canonical WHITE-relative columns:

    fen, cp, result, student_cp, teacher_delta_cp, generation

`cp` is the teacher score. `student_cp` is the score stored by the v5.00
self-play generator. Both are WHITE-relative in the Parquet output.

The teacher protocol is intentionally bounded: each UCI `go nodes` command
has a wall-clock timeout. A timed-out/crashed teacher is killed, restarted,
and the position is retried once. This follows the robust subprocess pattern
already used by train/labeling/process_positions.py and prevents one bad UCI
search from hanging a multi-hour generation.

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
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
TRAIN_DIR = ROOT / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

import dataset  # noqa: E402


CP_CLIP = 3000
DEFAULT_NODES = 10_000
DEFAULT_MAX_ROWS = 100_000
DEFAULT_WORKERS = max(1, min(8, (os.cpu_count() or 4) // 2))
DEFAULT_SHARD_ROWS = 10_000
DEFAULT_SEED = 500_314
DEFAULT_TIMEOUT = 30.0
DEFAULT_HASH_MB = 16

_PIECE_TO_FEN = {
    9: "P", 10: "N", 11: "B", 12: "R", 13: "Q", 14: "K",
    17: "p", 18: "n", 19: "b", 20: "r", 21: "q", 22: "k",
}

_ENGINE = None


def sample_to_fen(rec: np.void) -> str:
    """Convert one packed SelfplaySample to a legal FEN.

    Zchezz mailbox square 0 is a8 (the same order board_to_fen() uses), so
    the stored 64-byte board is already in FEN rank order. ep_file stores
    only the file; the target rank follows from side to move.
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


class UciTeacher:
    """Persistent single-thread UCI teacher with a hard per-position timeout."""

    def __init__(self, path: str, nodes: int, timeout: float, hash_mb: int) -> None:
        self.path = path
        self.nodes = nodes
        self.timeout = timeout
        self.hash_mb = hash_mb
        self.proc: subprocess.Popen | None = None
        self.re_cp = re.compile(r"score cp (-?\d+)")
        self.re_mate = re.compile(r"score mate (-?\d+)")
        self.start()

    def write(self, command: str) -> None:
        if not self.proc or self.proc.poll() is not None or not self.proc.stdin:
            raise RuntimeError("teacher process is not alive")
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def readline(self, timeout: float) -> str:
        if not self.proc or not self.proc.stdout:
            return ""
        box = [""]
        event = threading.Event()

        def reader() -> None:
            try:
                box[0] = self.proc.stdout.readline() if self.proc and self.proc.stdout else ""
            except Exception:
                box[0] = ""
            event.set()

        threading.Thread(target=reader, daemon=True).start()
        return box[0] if event.wait(timeout) else ""

    def read_until(self, marker: str, timeout: float) -> list[str]:
        lines: list[str] = []
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"teacher did not emit {marker!r} in {timeout:.1f}s")
            line = self.readline(remaining)
            if not line:
                raise TimeoutError(f"teacher stdout closed/stalled waiting for {marker!r}")
            lines.append(line)
            if marker in line:
                return lines

    def start(self) -> None:
        self.close()
        self.proc = subprocess.Popen(
            [self.path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self.write("uci")
        self.read_until("uciok", min(10.0, self.timeout))
        self.write(f"setoption name Hash value {self.hash_mb}")
        self.write("setoption name Threads value 1")
        self.write("isready")
        self.read_until("readyok", min(10.0, self.timeout))

    def close(self) -> None:
        p = self.proc
        self.proc = None
        if not p:
            return
        try:
            if p.poll() is None and p.stdin:
                p.stdin.write("quit\n")
                p.stdin.flush()
                p.wait(timeout=1.0)
        except Exception:
            pass
        if p.poll() is None:
            try:
                p.kill()
            except Exception:
                pass
        try:
            p.wait(timeout=1.0)
        except Exception:
            pass

    def _evaluate_once(self, fen: str) -> int:
        # Zchezz UCI scores are already WHITE-relative. search_best()
        # converts its root negamax score to wb_score before main.c emits
        # `info score cp`. A second Black-to-move flip here corrupts half
        # of the teacher corpus.
        self.write(f"position fen {fen}")
        self.write(f"go nodes {self.nodes}")
        last_cp = 0
        have_score = False
        for line in self.read_until("bestmove", self.timeout):
            m = self.re_mate.search(line)
            if m:
                last_cp = CP_CLIP if int(m.group(1)) > 0 else -CP_CLIP
                have_score = True
                continue
            m = self.re_cp.search(line)
            if m:
                last_cp = int(m.group(1))
                have_score = True
        if not have_score:
            last_cp = 0
        return max(-CP_CLIP, min(CP_CLIP, last_cp))

    def evaluate(self, fen: str) -> int:
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                return self._evaluate_once(fen)
            except Exception as exc:
                last_error = exc
                self.start()
        raise RuntimeError(f"teacher failed twice for FEN {fen!r}: {last_error}")


def _worker_init(engine_path: str, nodes: int, timeout: float, hash_mb: int) -> None:
    global _ENGINE
    _ENGINE = UciTeacher(engine_path, nodes, timeout, hash_mb)

    def _close() -> None:
        global _ENGINE
        if _ENGINE is not None:
            try:
                _ENGINE.close()
            except Exception:
                pass
            _ENGINE = None

    atexit.register(_close)


def _teacher_label(item: tuple[str, float, int]) -> tuple[str, float, int, int, int]:
    fen, result_white, student_cp_white = item
    teacher_cp = int(_ENGINE.evaluate(fen))
    return fen, result_white, student_cp_white, teacher_cp, teacher_cp - student_cp_white


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

    print(
        f"[teacher] input_rows={total:,} sampled={take:,} nodes={args.nodes:,} "
        f"workers={args.workers} timeout={args.timeout:.1f}s"
    )
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
        initargs=(str(teacher), args.nodes, args.timeout, args.hash_mb),
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
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                   help="hard wall-clock timeout for one teacher position; timed-out engines are restarted")
    p.add_argument("--hash-mb", type=int, default=DEFAULT_HASH_MB)
    return p


if __name__ == "__main__":
    raise SystemExit(label(build_parser().parse_args()))
