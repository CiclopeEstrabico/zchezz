#!/usr/bin/env python3
"""run_verify.py — drive zchezz_verify.exe (compiled with -DNNUE_VERIFY_ACC)
over UCI for a battery of positions chosen to exercise the HalfKP-4Bucket
accumulator: bucket-crossing king walks, castling (rook-is-a-feature,
king-is-not), en-passant, king captures, and ordinary middlegame play.
Also runs one case under Threads=4 to check per-thread accumulators
under Lazy SMP.

A verify-instrumented build calls abort() the instant a live accumulator
disagrees with a from-scratch rebuild, so "process exited 0 for every
case" is direct evidence the incremental update matched the ground truth
on all positions probed.
"""
import subprocess, sys, os, time

# ═══════════════ CONFIGURATION ═══════════════
# Edit these and run with no arguments; every one is also a CLI flag that
# overrides it (CLAUDE.md rule 8, see the COMMAND LINE block below).
EXE       = "zchezz_verify.exe"  # verify-instrumented engine binary (-DNNUE_VERIFY_ACC)
TIMEOUT_S = 60                   # per-case wall-clock limit
ONLY      = []                   # [] = every case; else keep cases whose name
                                 # contains one of these substrings
RUN_THREADS_CASE = True          # also run the Threads=4 case (Lazy SMP check)
# ═══════════════════════════════════════════

# ═══════════════ COMMAND LINE ═══════════════
# `python tests/test_nnue_accumulator.py` runs exactly what the block above
# says. The legacy positional form (`... path\to\zchezz_verify.exe`) still
# works and means --exe. See utils/cliconf.py.
# ════════════════════════════════════════════
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
from cliconf import override_from_cli  # noqa: E402

CLI = [
    ("EXE",              "--exe",     str,  "verify-instrumented engine binary"),
    ("TIMEOUT_S",        "--timeout", float,"per-case wall-clock limit, seconds"),
    ("ONLY",             "--only",    list, "run only cases whose name contains this (repeatable)"),
    ("RUN_THREADS_CASE", "--threads-case", bool, "also run the Threads=4 case"),
]

CASES = [
    ("startpos", "position startpos\ngo depth 8\n"),
    ("castle both sides",
     "position fen r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1\ngo depth 8\n"),
    ("KP endgame, kings cross bucket borders",
     "position fen 8/8/8/4k3/8/4K3/4P3/8 w - - 0 1\ngo depth 10\n"),
    ("en passant available",
     "position fen rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3\ngo depth 8\n"),
    ("king captures a pawn",
     "position fen 8/8/8/3k4/3p4/3K4/8/8 w - - 0 1\ngo depth 8\n"),
    ("middlegame: Italian-ish",
     "position fen r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4\ngo depth 8\n"),
    ("middlegame: open center",
     "position fen r2qkbnr/ppp2ppp/2np4/1B2p3/4P1b1/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 5\ngo depth 8\n"),
    ("king endgame, few pawns",
     "position fen 8/5pk1/6p1/8/7P/6P1/5PK1/8 w - - 0 1\ngo depth 10\n"),
]

THREADS_CASE = ("startpos, Threads=4",
    "setoption name Threads value 4\nposition startpos\ngo depth 8\n")


def run(name, cmds):
    # IMPORTANT: do NOT append "quit\n" here. cmd_go() starts the search on
    # a background thread and returns immediately; the "quit" handler sets
    # g_stop_flag and joins right away, which can abort the search before
    # it reaches the requested depth (best move can even come back 0000).
    # Closing stdin (EOF) instead hits the *other* code path in main.c's
    # loop ("EOF reached - wait for any running search to complete before
    # exit"), which pthread_join()s the search thread to completion first.
    # subprocess.run(input=...) closes stdin after writing, which is
    # exactly the EOF we want here.
    script = cmds
    t0 = time.time()
    try:
        r = subprocess.run([EXE], input=script, capture_output=True, text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired as e:
        print(f"[TIMEOUT] {name}")
        print(e.stdout or "")
        print(e.stderr or "")
        return False, float(TIMEOUT_S)
    dt = time.time() - t0
    ok = (r.returncode == 0) and ("MISMATCH" not in r.stderr) and ("MISMATCH" not in r.stdout)
    status = "OK" if ok else "FAIL"
    print(f"[{status}] {name}  (rc={r.returncode}, {dt:.2f}s)")
    if not ok:
        print("  --- stdout tail ---")
        print("\n".join(r.stdout.splitlines()[-20:]))
        print("  --- stderr tail ---")
        print("\n".join(r.stderr.splitlines()[-20:]))
    return ok, dt


def main():
    print(f"Engine: {EXE}\n")
    cases = CASES
    if ONLY:
        cases = [c for c in CASES if any(o.lower() in c[0].lower() for o in ONLY)]
        if not cases:
            print(f"ERROR: --only {ONLY} matched no case")
            sys.exit(1)
    all_ok = True
    total_time = 0.0
    for name, cmds in cases:
        ok, dt = run(name, cmds)
        all_ok &= ok
        total_time += dt
    if RUN_THREADS_CASE and not ONLY:
        ok, dt = run(*THREADS_CASE)
        all_ok &= ok
        total_time += dt

    print()
    print("=" * 60)
    print(f"ALL PASSED" if all_ok else "SOME FAILED")
    print(f"Total wall time: {total_time:.2f}s")
    print("=" * 60)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    # Legacy positional form: `python tests/test_nnue_accumulator.py <exe>`.
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        sys.argv[1:2] = ["--exe", sys.argv[1]]
    override_from_cli(globals(), CLI, description=__doc__,
                      prog="test_nnue_accumulator.py")
    main()
