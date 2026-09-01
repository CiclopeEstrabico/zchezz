#!/usr/bin/env python3
"""Reference perft suite for Zchezz move generation and make/unmake."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "utils"))
from repo_paths import engine_executable, latest_version  # noqa: E402

# ═══════════════ CONFIGURATION ═══════════════
VERSION = ""
ENGINE_EXE = ""
TIMEOUT_S = 120.0
MAX_DEPTH = 0
ONLY: list[str] = []
STOP_ON_FAIL = False
# ═════════════════════════════════════════════

PERFT_SUITE = [
    (
        "Startpos",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        [(1, 20), (2, 400), (3, 8902), (4, 197281), (5, 4865609)],
    ),
    (
        "Kiwipete (EP, castling, promotions, pins)",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -",
        [(1, 48), (2, 2039), (3, 97862), (4, 4085603)],
    ),
    (
        "Position 3 (discovered check, rook pins)",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -",
        [(1, 14), (2, 191), (3, 2812), (4, 43238), (5, 674624)],
    ),
    (
        "Position 4 (under-promotion, castling rights)",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        [(1, 6), (2, 264), (3, 9467), (4, 422333)],
    ),
    (
        "Position 5 (promotion captures)",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        [(1, 44), (2, 1486), (3, 62379), (4, 2103487)],
    ),
    (
        "Position 6 (complex middlegame, pins)",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        [(1, 46), (2, 2079), (3, 89890), (4, 3894594)],
    ),
    (
        "EP special (EP exposes check = illegal)",
        "3k4/3p4/8/K1P4r/8/8/8/8 b - - 0 1",
        [(1, 18), (2, 92), (3, 1670), (4, 10138), (5, 185429), (6, 1134888)],
    ),
    (
        "Castling rights after rook capture",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        [(1, 26), (2, 568), (3, 13744), (4, 314346), (5, 7594526)],
    ),
]

def parse_bool(text: str) -> bool:
    value = text.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {text!r}")

def resolve_engine(version: str, explicit: str) -> Path:
    return Path(explicit) if explicit else engine_executable(version or latest_version())

def run_perft(exe: Path, fen: str, depth: int, timeout: float) -> int | None:
    text = f"position fen {fen}\nperft {depth}\nquit\n"
    try:
        proc = subprocess.run(
            [str(exe)],
            cwd=exe.parent,
            input=text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None
    output = proc.stdout + "\n" + proc.stderr
    for line in output.splitlines():
        if "Nodes searched:" in line:
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy_version", nargs="?", default="")
    parser.add_argument("--version", default=VERSION)
    parser.add_argument("--exe", default=ENGINE_EXE)
    parser.add_argument("--timeout", type=float, default=TIMEOUT_S)
    parser.add_argument("--max-depth", type=int, default=MAX_DEPTH)
    parser.add_argument("--only", action="append", default=list(ONLY))
    parser.add_argument("--stop-on-fail", type=parse_bool, default=STOP_ON_FAIL)
    args = parser.parse_args()

    version = args.version or args.legacy_version or VERSION
    exe = resolve_engine(version, args.exe)
    if not exe.is_file():
        print(f"ERROR: engine not found: {exe}")
        return 1

    suite = PERFT_SUITE
    if args.only:
        wanted = [x.lower() for x in args.only]
        suite = [item for item in suite if any(key in item[0].lower() for key in wanted)]
        if not suite:
            print(f"ERROR: --only matched no position: {args.only}")
            return 2

    total = passed = failed = 0
    start = time.monotonic()
    print(f"Engine: {exe}")
    print(f"Positions: {len(suite)}; max-depth={args.max_depth or 'all'}")

    for name, fen, depths in suite:
        print(f"\n=== {name} ===")
        for depth, expected in depths:
            if args.max_depth and depth > args.max_depth:
                continue
            total += 1
            actual = run_perft(exe, fen, depth, args.timeout)
            if actual == expected:
                passed += 1
                print(f"PASS perft({depth}) = {actual:,}")
            else:
                failed += 1
                shown = "timeout/no result" if actual is None else f"{actual:,}"
                print(f"FAIL perft({depth}) = {shown}; expected {expected:,}")
                if args.stop_on_fail:
                    print(f"Results: {passed}/{total} passed, {failed} failed")
                    return 1

    elapsed = time.monotonic() - start
    print(f"\nResults: {passed}/{total} passed, {failed} failed ({elapsed:.1f}s)")
    return 1 if failed else 0

if __name__ == "__main__":
    raise SystemExit(main())
