#!/usr/bin/env python3
"""
run_selfplay_native.py — driver for engine/c/tools/selfplay.c
===============================================================

selfplay.exe already knows how to play self-play games entirely
in-process (N worker THREADS, no subprocess/UCI text, no PGN
re-parse, no separate Stockfish labelling pass — see
engine/c/tools/selfplay.c's own header comment for the full design,
implementation plan Appendix F.2/F.3). It parses its own CLI, builds
its own worker pool, and writes the packed .bin sample format
straight to disk (optionally alongside a standard PGN — rule 9: the
two are additive, not either/or).

What selfplay.exe does NOT know how to do is decide whether its own
binary is stale relative to the sources it was built from, or build
itself. That is this script's job, mirrored closely on
tests/run_arena.py's HEAD-build path (see that file's "BUILD
RESOLUTION" section for the identical mtime-check idea applied to
zchezz.exe instead of selfplay.exe): before every run, this script
mtime-checks selfplay.exe against every source file the Makefile's
`selfplay` target actually depends on (SELFPLAY_SRCS — see
BUILD_DEPS below, kept in sync BY HAND with engine/build/Makefile)
and rebuilds via `mingw32-make`/`make ENGINE=v402 selfplay` whenever
any of them is newer. This is the "auto-compile every time we start
a selfplay run" behavior requested for this tool. Everything else —
game playing, temperature move selection, TT policy, opening
resolution, .bin/.pgn output — is selfplay.exe's job; this script
does not reimplement any of it, it only shells out and streams
selfplay.exe's own stdout/stderr through (same spirit as
run_arena.py streaming arena.exe's output).

── RELATIONSHIP TO tests/run_selfplay.py (the OLDER driver) ─────────

  tests/run_selfplay.py is the pre-existing Python/UCI-subprocess
  self-play generator: it spawns N zchezz.exe PROCESSES, drives each
  one over the UCI text protocol, writes PGN, re-parses that PGN, and
  hands every position to a separate Stockfish process for labelling
  before serializing to Parquet. Per CLAUDE.md rule 9 ("native and
  non-native paths are a BLEND, not a replacement"), this script is
  an ADDITION alongside it, not a replacement — tests/run_selfplay.py
  is untouched and stays in the suite. Use the native path
  (selfplay.exe via this script) for throughput (one process, N
  threads, direct search_best() calls, no UCI/PGN/Stockfish
  round-trips); use tests/run_selfplay.py where its externally-judged
  Stockfish labels or its Parquet output are specifically wanted.

── OUTPUT FORMAT ─────────────────────────────────────────────────────

  selfplay.exe writes train/dataset.py's SAMPLE_DTYPE-compatible
  packed .bin records (see engine/c/tools/sample.h) to --out
  (append mode) and, optionally, standard PGN to --pgn and/or EPD
  (FEN + c0/c1/c2/c3 eval opcodes, same convention as
  tests/run_selfplay.py's EPD export) to --epd (rule 9: not
  either/or — .bin/.pgn/.epd can all be requested in the same run).

── USAGE ──────────────────────────────────────────────────────────────

  python tests/run_selfplay_native.py --out data/gen7.bin \\
      --games 20000 --threads 16 --movetime 100 \\
      --openings openings/ --opening-mode all

  python tests/run_selfplay_native.py --out data/gen7.bin --pgn data/gen7.pgn \\
      --games 500 --movetime 30 --multipv 4 --temperature 1.0 --temp-plies 24

Every flag selfplay.exe accepts (--games, --threads, --movetime,
--nodes, --depth, --multipv, --temperature, --temp-scale,
--temp-plies, --temp-final, --max-plies, --seed, --separate-tt,
--tt-mb, --nnue, --out, --pgn, --epd, --save-opening-in-epd,
--no-save-opening-in-epd, --openings, --opening-mode,
--random-plies, --book-portion, --same-opening-twice,
--no-same-opening-twice, --save-opening-samples) is forwarded — this
script does not re-parse or duplicate their meaning, see
FORWARD_VALUE_FLAGS/FORWARD_BOOL_FLAGS and the DEFAULT CONFIGURATION
block below (project convention, CLAUDE.md rule 8: every CLI-settable
option needs a documented default here too, and the CLI default must
literally BE that constant, never a second copy of the literal —
mirrors selfplay.c's own SP_DEFAULT_* block and run_arena.py's
DEFAULT_* block).

── CONFIGURATION NAMES ─────────────────────────────────────────────────────

Every tool uses the same constant name for the same concept, so a value
means the same thing whichever tool you are reading. The full list lives in
ONE place: utils/cliconf.py, section "SHARED CONFIGURATION VOCABULARY".
A `DEFAULT_` prefix marks a knob specific to this tool alone.

SAVE_PGN / SAVE_EPD / SAVE_BIN are independent switches, not a choice of
one: any combination can be on in the same run.
"""

# Windows consoles default to cp1252, which cannot encode the em-dashes and box
# characters used in this file's docstring and reports — printing then dies with
# UnicodeEncodeError before any output appears. Force UTF-8 (same as
# tests/bench_nps.py).
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass  # already-redirected or non-reconfigurable stream

import os
import sys
import time
import subprocess
import glob
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.path.join(REPO_ROOT, "engine", "build")
TOOLS_DIR = os.path.join(REPO_ROOT, "engine", "c", "tools")
ENGINE_DIR = os.path.join(REPO_ROOT, "engine", "c", "zchezz_v402")
# selfplay.exe is a SHARED tool binary, built in engine/build/ (NOT inside
# any engine/c/zchezz_vXXX/ folder) — see header comment above and
# README.md's engine/c/tools/ file inventory.
SELFPLAY_EXE = os.path.join(BUILD_DIR, "selfplay.exe")

# engine/build/Makefile's SELFPLAY_SRCS, mirrored here BY HAND so the
# staleness check knows exactly which files invalidate the cached exe.
# Keep in sync with that Makefile target if it ever changes.
BUILD_DEPS = [
    os.path.join(TOOLS_DIR, "selfplay.c"),
    os.path.join(TOOLS_DIR, "opening_pool.c"),
    os.path.join(TOOLS_DIR, "opening_pool.h"),
    os.path.join(TOOLS_DIR, "sample.h"),
    os.path.join(ENGINE_DIR, "board.c"),
    os.path.join(ENGINE_DIR, "search.c"),
    os.path.join(ENGINE_DIR, "nnue.c"),
]
# Headers the above .c files pull in from ENGINE_DIR — not in the Makefile's
# own prerequisite list (make itself doesn't track them either), but a
# changed header with an unchanged .c mtime is exactly the kind of stale
# build this check exists to catch, so they're included defensively.
BUILD_DEPS += glob.glob(os.path.join(ENGINE_DIR, "*.h"))

# Direct-gcc fallback command, used only if neither `make` nor
# `mingw32-make` is on PATH — mirrors engine/build/Makefile's `selfplay`
# target verbatim (see that Makefile and engine/c/tools/selfplay.c's own
# header "BUILD" section for the exact same command spelled out).
DIRECT_GCC_BUILD_CMD = [
    "gcc", "-O3", "-ffast-math", "-D_GNU_SOURCE", "-std=c11", "-mavxvnni", "-mavx2",
    "-I" + os.path.join("..", "c", "zchezz_v402"), "-DNO_TABLEBASES", "-DNO_BOOK",
    "-Wno-unused-variable", "-Wno-unused-but-set-variable",
    "-Wno-maybe-uninitialized", "-Wno-misleading-indentation",
    "-Wno-sign-compare", "-Wno-unused-function", "-Wno-parentheses",
    "-o", "selfplay.exe",
    os.path.join("..", "c", "tools", "selfplay.c"),
    os.path.join("..", "c", "tools", "opening_pool.c"),
    os.path.join("..", "c", "zchezz_v402", "board.c"),
    os.path.join("..", "c", "zchezz_v402", "search.c"),
    os.path.join("..", "c", "zchezz_v402", "nnue.c"),
    "-static", "-lm", "-pthread",
]

# ═══════════════════════════════════════════════════════════════════
# ===== DEFAULT CONFIGURATION =====
#
# Project-wide convention (same as tests/run_arena.py, tests/
# run_tournament.py): every option this script accepts on the command
# line has a documented default HERE, and argparse's own `default=`
# below reads FROM these constants — never a second copy of the same
# literal. These mirror selfplay.c's own SP_DEFAULT_* block BY HAND
# (selfplay.exe would apply those same defaults anyway if a flag were
# omitted, but per the project's config-block convention this script
# forwards an explicit value for every numeric/boolean flag rather
# than relying on the child process's own defaults silently). Path-
# like flags default to "" (not forwarded — see PATH_LIKE_FLAGS)
# since "no file" is selfplay.exe's own meaningful default for those.
# ═══════════════════════════════════════════════════════════════════
SAVE_PGN                 = False    # save games to standard PGN file in RESULTS_DIR
SAVE_EPD                 = False    # save positions + eval to EPD file in RESULTS_DIR
SAVE_BIN                 = True     # save packed training samples to BIN file in RESULTS_DIR
RESULTS_DIR              = r"tests\selfplay_results"

DEFAULT_OUT_PATH             = ""       # explicit BIN override path (if empty and SAVE_BIN=True, uses RESULTS_DIR)
DEFAULT_PGN_PATH             = ""       # explicit PGN override path (if empty and SAVE_PGN=True, uses RESULTS_DIR)
DEFAULT_EPD_PATH             = ""       # explicit EPD override path (if empty and SAVE_EPD=True, uses RESULTS_DIR)
GAMES                        = 100      # number of self-play games
CONCURRENCY                  = 0        # worker threads (concurrent games, not threads per game); 0 = selfplay.exe autodetects logical cores
MOVETIME_MS                  = 100      # per-move time budget, ms; 0 = use --nodes/--depth instead
NODES                        = 0        # per-move node budget; used only when movetime==0 and depth==0
DEFAULT_DEPTH                = 0        # per-move depth budget, plies; 0 = use movetime/nodes instead
DEFAULT_MULTIPV               = 4        # root candidates sampled for temperature move choice
DEFAULT_TEMPERATURE          = 1.0      # softmax T0 for the first --temp-plies SEARCH plies
DEFAULT_TEMP_SCALE           = 100.0    # centipawns per softmax unit before applying T
DEFAULT_TEMP_PLIES           = 24       # search plies using T0 before switching to --temp-final
DEFAULT_TEMP_FINAL           = 0.05     # softmax T1 after --temp-plies (near-argmax, not exactly)
DEFAULT_TEMP_ARGMAX_EPS      = 0.0      # T <= this is EXACT argmax: that ply searches multipv=1 and skips the
                                        # softmax. 0.0 = only a literal T of 0 takes the fast path. Set
                                        # TEMPERATURE=0 and TEMP_FINAL=0 for the deterministic generator at
                                        # deterministic cost; raise just above TEMP_FINAL (e.g. 0.06) to get a
                                        # diverse opening and a fast argmax remainder.
MAX_PLIES                    = 400      # game-length safety cap -> counted as a draw
SEED                         = 1        # RNG base seed, per-game deterministic
DEFAULT_SEPARATE_TT          = False    # False = shared TT between colors within a game (default)
TT_MB                        = 8.0      # per-TTable memory budget, MB
DEFAULT_NNUE_PATH            = os.path.join(ENGINE_DIR, "nnue_weights.bin")  # v4.02: absolute path into the engine folder (selfplay runs with the caller's cwd)
DEFAULT_SAVE_OPENING_IN_EPD  = True     # True = include forced opening plies in the EPD output (default ON,
                                         # matches tests/run_selfplay.py's SAVE_OPENING_IN_EPD=True; deliberately
                                         # the OPPOSITE default of DEFAULT_SAVE_OPENING_SAMPLES below)
# Relative paths resolve against the caller's cwd (normally the repo root).
# Leave this non-empty: an empty value makes OPENING_MODE="book" fall back to
# random openings without saying so.
OPENING_FOLDER                = r"openings\lines"  # walked recursively for .pgn/.epd files
DEFAULT_OPENING_MODE         = "book"   # how each game's starting position is chosen:
                                        #   "book"        from an opening in OPENING_FOLDER
                                        #   "random"      after RANDOM_PLIES random legal plies
                                        #   "book+random" BOOK_PORTION from the book, rest random
                                        # ("all" accepted as the old name for "book+random")
DEFAULT_RANDOM_PLIES         = 6        # random legal opening plies (random/all modes)
DEFAULT_BOOK_PORTION         = 0.97     # "book+random" only: fraction of (paired) games using a book opening
DEFAULT_SAME_OPENING_TWICE   = True     # True = pair games (2k,2k+1) on the same opening (variance reduction)
DEFAULT_SAVE_OPENING_SAMPLES = False    # False = exclude forced opening-phase plies from the .bin
# ═══════════════════════════════════════════════════════════════════

# Flags accepted by selfplay.c's parse_args() that this script forwards
# verbatim (value-taking flags -> "--flag VALUE"). --out is handled
# separately below (it's required, not defaulted). Anything the CLI
# passes through argparse ends up here; nothing is re-parsed.
FORWARD_VALUE_FLAGS = {
    "--games", "--threads", "--movetime", "--nodes", "--depth", "--multipv",
    "--temperature", "--temp-scale", "--temp-plies", "--temp-final",
    "--temp-argmax-eps",
    "--max-plies", "--seed", "--tt-mb", "--nnue", "--pgn", "--epd", "--openings",
    "--opening-mode", "--random-plies", "--book-portion",
}
# Path-like flags: an empty-string default means "don't forward this flag
# at all" (selfplay.exe's own meaningful default), never "forward an
# empty path". NOTE --openings is normally non-empty (see
# OPENING_FOLDER above) and IS forwarded by default.
PATH_LIKE_FLAGS = {"--pgn", "--epd", "--openings"}


def log(msg):
    print(f"[run_selfplay_native] {msg}", file=sys.stderr)


def run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def ensure_selfplay_built():
    """mtime-check selfplay.exe against BUILD_DEPS (mirrors run_arena.py's
    resolve_head() mtime check, applied to selfplay.exe instead of
    zchezz.exe) and rebuild via the Makefile's `selfplay` target if stale
    or missing. This is the 'auto-compile every time we start a selfplay
    run' behavior."""
    needs_build = True
    if os.path.isfile(SELFPLAY_EXE):
        exe_mtime = os.path.getmtime(SELFPLAY_EXE)
        stale = [d for d in BUILD_DEPS if os.path.isfile(d) and os.path.getmtime(d) > exe_mtime]
        if stale:
            log(f"selfplay.exe is older than {len(stale)} source(s) (e.g. {stale[0]}) — rebuilding")
        needs_build = bool(stale)
    else:
        log("selfplay.exe not found — building")

    if not needs_build:
        log(f"selfplay.exe is up to date, reusing {SELFPLAY_EXE}")
        return True

    for make_cmd in (["make", "ENGINE=v402", "selfplay"], ["mingw32-make", "ENGINE=v402", "selfplay"]):
        try:
            r = run(make_cmd, cwd=BUILD_DIR)
        except FileNotFoundError:
            continue
        if r.returncode == 0:
            log(f"built selfplay.exe via `{' '.join(make_cmd)}`")
            return True
        log(f"`{' '.join(make_cmd)}` failed:\n{r.stdout}\n{r.stderr}")

    log("no working `make`/`mingw32-make` found — building selfplay.exe directly with gcc")
    r = run(DIRECT_GCC_BUILD_CMD, cwd=BUILD_DIR)
    if r.returncode != 0:
        log(f"direct gcc build of selfplay.exe also failed:\n{r.stdout}\n{r.stderr}")
        return False
    log("built selfplay.exe via direct gcc fallback")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT_PATH,
                     help="output .bin file (required, append mode — forwarded to selfplay.exe's --out)")
    ap.add_argument("--games", type=int, default=GAMES)
    ap.add_argument("--threads", type=int, default=CONCURRENCY)
    ap.add_argument("--movetime", type=int, default=MOVETIME_MS)
    ap.add_argument("--nodes", type=int, default=NODES)
    ap.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    ap.add_argument("--multipv", type=int, default=DEFAULT_MULTIPV)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--temp-scale", type=float, default=DEFAULT_TEMP_SCALE)
    ap.add_argument("--temp-plies", type=int, default=DEFAULT_TEMP_PLIES)
    ap.add_argument("--temp-final", type=float, default=DEFAULT_TEMP_FINAL)
    ap.add_argument("--temp-argmax-eps", type=float, default=DEFAULT_TEMP_ARGMAX_EPS)
    ap.add_argument("--max-plies", type=int, default=MAX_PLIES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--separate-tt", action="store_true", default=DEFAULT_SEPARATE_TT,
                     help="give each color its own TTable (default: shared within a game)")
    ap.add_argument("--tt-mb", type=float, default=TT_MB)
    ap.add_argument("--nnue", default=DEFAULT_NNUE_PATH)
    ap.add_argument("--pgn", default=DEFAULT_PGN_PATH,
                     help="write all games as standard PGN to PATH (rule 9: additive, not either/or with --out)")
    ap.add_argument("--epd", default=DEFAULT_EPD_PATH,
                     help="write every position as EPD (FEN + c0/c1/c2/c3 eval opcodes) to PATH "
                          "(rule 9: additive, not either/or with --out/--pgn; same convention as "
                          "tests/run_selfplay.py's EPD export)")
    epd_open_group = ap.add_mutually_exclusive_group()
    epd_open_group.add_argument("--save-opening-in-epd", dest="save_opening_in_epd",
                                 action="store_true", default=DEFAULT_SAVE_OPENING_IN_EPD)
    epd_open_group.add_argument("--no-save-opening-in-epd", dest="save_opening_in_epd",
                                 action="store_false")
    ap.add_argument("--openings", default=OPENING_FOLDER,
                     help="opening corpus: a .pgn/.epd file OR a directory walked recursively")
    ap.add_argument("--opening-mode", choices=("book", "random", "book+random", "all"),
                    default=DEFAULT_OPENING_MODE,
                    help="book | random | book+random ('all' = old name for book+random)")
    ap.add_argument("--random-plies", type=int, default=DEFAULT_RANDOM_PLIES)
    ap.add_argument("--book-portion", type=float, default=DEFAULT_BOOK_PORTION)
    same_group = ap.add_mutually_exclusive_group()
    same_group.add_argument("--same-opening-twice", dest="same_opening_twice",
                             action="store_true", default=DEFAULT_SAME_OPENING_TWICE)
    same_group.add_argument("--no-same-opening-twice", dest="same_opening_twice",
                             action="store_false")
    ap.add_argument("--save-opening-samples", action="store_true", default=DEFAULT_SAVE_OPENING_SAMPLES)
    ap.add_argument("--show-config", action="store_true",
                    help="print the resolved configuration and exit "
                         "(runs nothing, writes nothing)")
    args = ap.parse_args()
    if getattr(args, "show_config", False):
        print("Resolved configuration:")
        _w = max(len(k) for k in vars(args))
        for _k, _v in sorted(vars(args).items()):
            if _k != "show_config":
                print(f"  {_k.ljust(_w)} = {_v!r}")
        sys.exit(0)


    out_path = args.out
    pgn_path = args.pgn
    epd_path = args.epd

    if SAVE_PGN or SAVE_EPD or SAVE_BIN or out_path or pgn_path or epd_path:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        if SAVE_BIN and not out_path:
            out_path = os.path.join(RESULTS_DIR, f"selfplay_{ts}.bin")
        if SAVE_PGN and not pgn_path:
            pgn_path = os.path.join(RESULTS_DIR, f"selfplay_{ts}.pgn")
        if SAVE_EPD and not epd_path:
            epd_path = os.path.join(RESULTS_DIR, f"selfplay_{ts}.epd")

    if not out_path and not pgn_path and not epd_path:
        log("error: at least one output (SAVE_BIN, SAVE_PGN, or SAVE_EPD) must be enabled")
        sys.exit(1)

    if not ensure_selfplay_built():
        sys.exit(1)

    cmd = [SELFPLAY_EXE]
    if out_path:
        cmd += ["--out", out_path]

    arg_map = {
        "--games": args.games, "--threads": args.threads, "--movetime": args.movetime,
        "--nodes": args.nodes, "--depth": args.depth, "--multipv": args.multipv,
        "--temperature": args.temperature, "--temp-scale": args.temp_scale,
        "--temp-plies": args.temp_plies, "--temp-final": args.temp_final,
        "--temp-argmax-eps": args.temp_argmax_eps,
        "--max-plies": args.max_plies, "--seed": args.seed, "--tt-mb": args.tt_mb,
        "--nnue": args.nnue, "--pgn": pgn_path, "--epd": epd_path, "--openings": args.openings,
        # selfplay.exe still spells this mode "all"; translate at the boundary
        # so the Python side can use the self-describing name.
        "--opening-mode": ("all" if args.opening_mode == "book+random"
                           else args.opening_mode),
        "--random-plies": args.random_plies,
        "--book-portion": args.book_portion,
    }
    for flag, val in arg_map.items():
        if flag in PATH_LIKE_FLAGS and not val:
            continue   # "" means "selfplay.exe's own default applies" — don't forward an empty path
        cmd += [flag, str(val)]
    if args.separate_tt:
        cmd.append("--separate-tt")
    if args.same_opening_twice:
        cmd.append("--same-opening-twice")
    else:
        cmd.append("--no-same-opening-twice")
    if args.save_opening_samples:
        cmd.append("--save-opening-samples")
    if args.save_opening_in_epd:
        cmd.append("--save-opening-in-epd")
    else:
        cmd.append("--no-save-opening-in-epd")

    log(f"running: {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
