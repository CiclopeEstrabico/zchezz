#!/usr/bin/env python3
"""Comprehensive perft test for Zchezz — validates make/unmake correctness.

Tests ALL move generation edge cases:
  - Normal moves, double pawn push
  - En passant (including EP that exposes check)
  - Castling (all 4 types, rights revocation)
  - Promotions (queen, rook, bishop, knight)
  - Promotion captures
  - Discovered check, double check
  - Pins (bishop, rook)
  - Complex middlegame positions

Every position must match reference node counts EXACTLY.
Any mismatch = make/unmake or move generation bug.
"""
import subprocess, sys, os, time

# Force UTF-8 output on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

ENGINE_DIR = os.path.join(os.path.dirname(__file__), "..", "engine", "c")

# ═══════════════ CONFIGURATION ═══════════════
# DEFAULT_VERSION: engine folder suffix used when no version is given on the
# command line (`python tests/test_perft.py vXXX`). Keep it pointing at the
# engine version currently under development.
DEFAULT_VERSION = "v400"
# ═══════════════════════════════════════════

# Reference positions with known perft values
# Format: (name, fen, [(depth, expected_nodes), ...])
PERFT_SUITE = [
    ("Startpos",
     "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
     [(1, 20), (2, 400), (3, 8902), (4, 197281), (5, 4865609)]),

    ("Kiwipete (EP, castling, promotions, pins)",
     "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -",
     [(1, 48), (2, 2039), (3, 97862), (4, 4085603)]),

    ("Position 3 (discovered check, rook pins)",
     "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -",
     [(1, 14), (2, 191), (3, 2812), (4, 43238), (5, 674624)]),

    ("Position 4 (under-promotion, castling rights)",
     "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
     [(1, 6), (2, 264), (3, 9467), (4, 422333)]),

    ("Position 5 (promotion captures)",
     "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
     [(1, 44), (2, 1486), (3, 62379), (4, 2103487)]),

    ("Position 6 (complex middlegame, pins)",
     "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
     [(1, 46), (2, 2079), (3, 89890), (4, 3894594)]),

    ("EP special (EP exposes check = illegal)",
     "3k4/3p4/8/K1P4r/8/8/8/8 b - - 0 1",
     [(1, 18), (2, 92), (3, 1670), (4, 10138), (5, 185429), (6, 1134888)]),

    ("Castling rights after rook capture",
     "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
     [(1, 26), (2, 568), (3, 13744), (4, 314346), (5, 7594526)]),
]


def run_perft(exe, fen, depth):
    """Run perft and return node count."""
    cmds = f"position fen {fen}\nperft {depth}\nquit\n"
    try:
        result = subprocess.run(
            [exe], input=cmds, capture_output=True, text=True, timeout=120
        )
        output = result.stdout + result.stderr
        for line in output.split('\n'):
            if 'Nodes searched:' in line:
                return int(line.split(':')[1].strip())
    except subprocess.TimeoutExpired:
        return -1
    return -1


def main():
    version = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VERSION
    exe = os.path.join(ENGINE_DIR, f"zchezz_{version}", "zchezz.exe")
    if not os.path.exists(exe):
        print(f"ERROR: {exe} not found")
        sys.exit(1)

    print(f"Engine: {exe}")
    print(f"Perft Suite: {len(PERFT_SUITE)} positions\n")

    total = 0
    passed = 0
    failed = 0
    t0 = time.time()

    for name, fen, depths in PERFT_SUITE:
        print(f"=== {name} ===")
        print(f"  FEN: {fen}")
        for depth, expected in depths:
            total += 1
            nodes = run_perft(exe, fen, depth)
            ok = nodes == expected
            if ok:
                passed += 1
                print(f"  OK perft({depth}) = {nodes:>12,}")
            else:
                failed += 1
                print(f"  FAIL perft({depth}) = {nodes:>12,}  (expected {expected:>12,})")
        print()

    elapsed = time.time() - t0
    print("=" * 50)
    print(f"Results: {passed}/{total} passed, {failed} failed  ({elapsed:.1f}s)")
    print("=" * 50)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
