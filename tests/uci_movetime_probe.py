#!/usr/bin/env python3
"""Probe short UCI searches on identical legal positions.

This is a diagnostic, not an Elo test.  It measures what a requested
``go movetime`` actually buys each engine: completed depth, reported nodes,
wall time, and invalid/null bestmove responses while legal moves still exist.
Engines are queried one at a time and the query order rotates by position to
reduce thermal/order bias.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import chess


DEPTH_RE = re.compile(r"\bdepth\s+(\d+)")
NODES_RE = re.compile(r"\bnodes\s+(\d+)")
NPS_RE = re.compile(r"\bnps\s+(\d+)")


@dataclass
class Engine:
    name: str
    path: str
    proc: subprocess.Popen[str]


def read_until(engine: Engine, prefix: str, timeout_s: float) -> list[str]:
    deadline = time.monotonic() + timeout_s
    lines: list[str] = []
    while time.monotonic() < deadline:
        line = engine.proc.stdout.readline()
        if line == "":
            raise RuntimeError(f"{engine.name}: engine pipe closed")
        line = line.strip()
        lines.append(line)
        if line.startswith(prefix):
            return lines
    raise TimeoutError(f"{engine.name}: timeout waiting for {prefix!r}")


def send(engine: Engine, command: str) -> None:
    engine.proc.stdin.write(command + "\n")
    engine.proc.stdin.flush()


def start_engine(name: str, path: str, hash_mb: int) -> Engine:
    proc = subprocess.Popen(
        [path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    e = Engine(name=name, path=path, proc=proc)
    send(e, "uci")
    read_until(e, "uciok", 10.0)
    send(e, f"setoption name Hash value {hash_mb}")
    send(e, "setoption name Threads value 1")
    send(e, "setoption name OwnBook value false")
    send(e, "isready")
    read_until(e, "readyok", 10.0)
    return e


def stop_engine(engine: Engine) -> None:
    try:
        send(engine, "quit")
        engine.proc.wait(timeout=2.0)
    except Exception:
        engine.proc.kill()


def make_positions(seed: int, count: int) -> list[str]:
    rng = random.Random(seed)
    out: list[str] = []
    seen: set[str] = set()
    while len(out) < count:
        board = chess.Board()
        target = rng.randint(8, 54)
        for _ in range(target):
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(rng.choice(legal))
            if board.is_game_over(claim_draw=True):
                break
        if board.is_game_over(claim_draw=True) or board.legal_moves.count() < 2:
            continue
        fen = board.fen()
        if fen in seen:
            continue
        seen.add(fen)
        out.append(fen)
    return out


def parse_last_info(lines: list[str]) -> tuple[int | None, int | None, int | None]:
    depth = nodes = nps = None
    for line in lines:
        if not line.startswith("info"):
            continue
        m = DEPTH_RE.search(line)
        if m:
            depth = int(m.group(1))
        m = NODES_RE.search(line)
        if m:
            nodes = int(m.group(1))
        m = NPS_RE.search(line)
        if m:
            nps = int(m.group(1))
    return depth, nodes, nps


def probe(engine: Engine, fen: str, movetime_ms: int) -> dict:
    board = chess.Board(fen)
    send(engine, "ucinewgame")
    send(engine, "isready")
    read_until(engine, "readyok", 10.0)
    send(engine, f"position fen {fen}")

    t0 = time.perf_counter_ns()
    send(engine, f"go movetime {movetime_ms}")
    lines = read_until(engine, "bestmove", max(5.0, movetime_ms / 1000.0 + 3.0))
    wall_ms = (time.perf_counter_ns() - t0) / 1e6

    best_line = next(line for line in reversed(lines) if line.startswith("bestmove"))
    parts = best_line.split()
    best = parts[1] if len(parts) > 1 else ""
    depth, nodes, nps = parse_last_info(lines)

    legal_count = board.legal_moves.count()
    null_with_legal = best in {"", "0000", "(none)"} and legal_count > 0
    invalid = False
    if best not in {"", "0000", "(none)"}:
        try:
            invalid = chess.Move.from_uci(best) not in board.legal_moves
        except ValueError:
            invalid = True

    return {
        "bestmove": best,
        "depth": depth,
        "nodes": nodes,
        "reported_nps": nps,
        "wall_ms": wall_ms,
        "legal_count": legal_count,
        "null_with_legal": null_with_legal,
        "invalid_bestmove": invalid,
    }


def median(values):
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def mean(values):
    vals = [v for v in values if v is not None]
    return statistics.mean(vals) if vals else None


def summarize(rows: list[dict]) -> dict:
    nodes = [r["nodes"] for r in rows]
    depth = [r["depth"] for r in rows]
    wall = [r["wall_ms"] for r in rows]
    return {
        "count": len(rows),
        "median_nodes": median(nodes),
        "mean_nodes": mean(nodes),
        "median_depth": median(depth),
        "mean_depth": mean(depth),
        "median_wall_ms": median(wall),
        "mean_wall_ms": mean(wall),
        "null_with_legal": sum(bool(r["null_with_legal"]) for r in rows),
        "invalid_bestmove": sum(bool(r["invalid_bestmove"]) for r in rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", action="append", required=True, help="NAME=PATH")
    ap.add_argument("--positions", type=int, default=128)
    ap.add_argument("--movetime", type=int, default=200)
    ap.add_argument("--seed", type=int, default=806200)
    ap.add_argument("--hash-mb", type=int, default=16)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    specs: list[tuple[str, str]] = []
    for spec in args.engine:
        name, path = spec.split("=", 1)
        specs.append((name, path))

    positions = make_positions(args.seed, args.positions)
    engines = [start_engine(name, path, args.hash_mb) for name, path in specs]
    rows: dict[str, list[dict]] = {e.name: [] for e in engines}

    try:
        for i, fen in enumerate(positions):
            order = engines[i % len(engines):] + engines[: i % len(engines)]
            for e in order:
                r = probe(e, fen, args.movetime)
                r["position_index"] = i
                rows[e.name].append(r)
            if (i + 1) % 16 == 0:
                print(f"positions {i + 1}/{len(positions)}", flush=True)
    finally:
        for e in engines:
            stop_engine(e)

    summary = {name: summarize(rs) for name, rs in rows.items()}
    base = specs[0][0]
    ratios = {}
    base_nodes = summary[base]["median_nodes"]
    for name, _ in specs[1:]:
        n = summary[name]["median_nodes"]
        if base_nodes and n:
            ratios[f"{base}_over_{name}_median_nodes"] = base_nodes / n

    out = {
        "movetime_ms": args.movetime,
        "seed": args.seed,
        "positions": positions,
        "summary": summary,
        "ratios": ratios,
        "rows": rows,
    }
    Path(args.json).write_text(json.dumps(out, indent=2))

    print(json.dumps({"movetime_ms": args.movetime, "summary": summary, "ratios": ratios}, indent=2))


if __name__ == "__main__":
    main()
