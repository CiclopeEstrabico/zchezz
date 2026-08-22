#!/usr/bin/env python3
"""
run_arena.py — driver for engine/c/tools/arena.c
=================================================

arena.exe already knows how to play two IN-PROCESS kinds of player
("net:<weights.nnu4>" and "uci:<already-built .exe>") against each
other or in a round-robin/gauntlet — see engine/c/tools/arena.c's
header comment for the full design (Appendix F.4 of
docs/v400_implementation_plan.md).

NOTE ON THE BUILD LAYOUT: engine/c/tools/arena.c is SHARED across
engine versions (it tracks the current engine API only, see
README.md's engine/c/tools/ file inventory) and is built via engine/build/Makefile,
which places arena.exe in engine/build/ — NOT inside any
engine/c/zchezz_vXXX/ folder. zchezz.exe, by contrast, is still built
per-version inside its own engine/c/zchezz_vXXX/ folder (see
ENGINE_DIR_FOR_HEAD below) since it IS the per-version release
artifact.

What arena.exe does NOT know how to do is turn "compare against
v3.14" or "compare against a git tag" into an actual .exe. That is
this script's ONLY job: resolve engine references from git into built
zchezz.exe binaries, register them as uci: players, and hand the
whole player list to arena.exe. Everything else (game playing, Elo,
SPRT, cross-table, anchors) is arena.exe's job — this script does not
reimplement any of that, it only shells out and streams arena.exe's
own stdout/stderr through.

NOTE ON ABSOLUTE/ANCHOR ELO: arena.c deliberately does NOT have a
Stockfish-anchor / known-strength mode (see that file's header comment
"NO STOCKFISH-ANCHORED ABSOLUTE ELO HERE") — this native path exists
to compare two engine VERSIONS quickly (Appendix F.4/F.5's SPRT gate),
not to benchmark against an external reference at a calibrated
strength. For absolute Elo vs Stockfish, use
tests/run_tournament.py's ANCHORS mechanism directly; this script has
no anchor syntax to forward.

── PLAYER REFERENCE SYNTAX ─────────────────────────────────────────

  net:<path.nnu4>       passed straight through to arena.exe unchanged.
  uci:<path>            passed straight through to arena.exe unchanged
                         (already-built .exe — including Stockfish).
  ref:<gitref>          a git ref (branch, tag, or commit sha) OR the
                         literal word HEAD. This script builds it (see
                         "BUILD RESOLUTION" below) and rewrites it to
                         a uci:<built exe path> before invoking
                         arena.exe. This is the ONLY spec kind this
                         script actually does work for.

── BUILD RESOLUTION ────────────────────────────────────────────────

  ref == "HEAD" (case-insensitive): built IN PLACE, in this checkout's
  own engine/c/zchezz_v400/ (the current v4.00 dev folder — see
  ENGINE_DIR_FOR_HEAD below), no git worktree involved. This is the
  common "test my current uncommitted work" case — a worktree can only
  check out a COMMITTED ref, so a dirty working tree has to be built
  where it lives. Rebuilt only if any of its own sources is newer than
  the cached zchezz.exe (mtime check), so repeated runs during
  iterative development don't pay a rebuild every time.

  Any other ref: resolved to a commit sha via `git rev-parse <ref>`
  (this is the CACHE KEY — re-running with the same ref that already
  resolves to a cached sha never rebuilds, even if the ref's branch
  tip has since moved — pin an explicit sha if you want that). A
  temporary `git worktree add --detach` checks out that commit, this
  script finds the ENGINE FOLDER inside it (the project's convention
  is one folder per version — engine/c/zchezz_vXXX/ — and which one
  existed at an arbitrary historical commit is not knowable in
  advance, so this script picks the highest-numbered
  engine/c/zchezz_v* folder present in that worktree), builds it with
  the exact native-build command from CLAUDE.md's "Build Instructions"
  section (mirrored here, see BUILD_CMD_TEMPLATE), copies the
  resulting zchezz.exe (+ nnue_weights.bin, if the folder has one — a
  build without its own weights would silently fall back to whatever
  nnue_weights.bin happens to be in the CURRENT working directory when
  arena.exe spawns it, which is almost never what you want when
  comparing two different versions) into a persistent cache directory
  keyed by the commit sha, then removes the temporary worktree.

  A ref that fails to resolve/build is reported to stderr and DROPPED
  from the player list rather than aborting the whole run — matching
  the task's explicit requirement that one bad ref must not sink an
  otherwise-runnable match. If fewer than 2 players survive, this
  script exits with an error (arena.exe itself would also refuse).

── CACHE LAYOUT ─────────────────────────────────────────────────────

  <repo_root>/.arena_build_cache/<sha>/zchezz.exe (+ nnue_weights.bin,
  tablebases/ symlink if present). Safe to delete entirely at any
  time — everything in it is reproducible from git.

── USAGE ────────────────────────────────────────────────────────────

  python tests/run_arena.py --player ref:HEAD --player ref:v3.14 \\
      --games 100 --movetime 100 --threads 8 \\
      --openings openings/lines/8moves_v3.pgn --sprt --elo0 0 --elo1 5

  python tests/run_arena.py --player net:gen6.nnu4 --player net:gen5.nnu4 \\
      --games 400 --movetime 30 --threads 16 --sprt --elo0 0 --elo1 5 \\
      --json gate_result.json          # Appendix F.5 promote/discard gate

  python tests/run_arena.py --player ref:HEAD --player ref:v3.14 --player ref:main \\
      --games 50 --movetime 100 --threads 8 --pgn cross_version.pgn   # mini-tournament

Every arena.exe flag not related to player resolution (--games,
--threads, --movetime, --nodes, --max-plies, --openings,
--opening-plies, --seed, --tt-mb, --gauntlet, --sprt, --elo0, --elo1,
--alpha, --beta, --uci-timeout, --json, --pgn, --bin, --epd) is
forwarded — this script does not re-parse or duplicate their meaning,
see FORWARD_VALUE_FLAGS and the DEFAULT CONFIGURATION block below
(project convention: every CLI-settable option needs a documented
top-of-file default here too, same as the v3.14 Python test harnesses
already do).

── OUTPUT FORMATS: --pgn / --bin / --epd (additive, not either/or) ───

  --pgn FILE   standard PGN, GUI-loadable.
  --bin FILE   packed training samples (sample.h/dataset.py format).
               arena.exe HARD-REFUSES this unless every resolved
               player is net: and shares the SAME weight file — see
               arena.c's --bin gate. Since this script's ref:/uci:
               players always resolve to uci: (a built .exe, not an
               in-process net), --bin only works with --player
               net:<path> specs, and only when they're the same file.
               This is deliberate, not a limitation to work around:
               sample.h's eval_cp has no per-row evaluator-identity
               field, so a mixed-engine --bin run would silently blend
               two different engines' opinions into one training
               column (CLAUDE.md rule 10).
  --epd FILE   FEN + c0/c1/c2/c3 opcode rows. NO such restriction —
               c2 records the actual mover's player name per row, so
               a ref:HEAD-vs-ref:v3.14 or net:-vs-Stockfish run is
               fine here; a downstream consumer can filter/group on
               c2 if it only wants one evaluator's rows.

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
import subprocess
import shutil
import glob
import time
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(REPO_ROOT, ".arena_build_cache")
ENGINE_DIR_FOR_HEAD = os.path.join(REPO_ROOT, "engine", "c", "zchezz_v402")
BUILD_DIR = os.path.join(REPO_ROOT, "engine", "build")
TOOLS_DIR = os.path.join(REPO_ROOT, "engine", "c", "tools")
# arena.exe is a SHARED tool binary, built in engine/build/ (NOT inside
# ENGINE_DIR_FOR_HEAD) — see the header comment above and
# README.md's engine/c/tools/ file inventory.
ARENA_EXE = os.path.join(BUILD_DIR, "arena.exe")

# ═══════════════════════════════════════════════════════════════════
# ===== DEFAULT CONFIGURATION =====
#
# Project-wide convention (same as tests/run_tournament.py and
# tests/run_tournament_quick.py): every option this script accepts on
# the command line has a documented default HERE, and argparse's own
# `default=` below reads FROM these constants — never a second copy
# of the same literal. Edit these to change this script's defaults
# without touching argparse; edit argparse only to change flag names.
#
# These are kept in sync BY HAND with arena.c's own ARENA_DEFAULT_*
# block (engine/c/tools/arena.c) — arena.exe would apply
# those same defaults anyway if a flag were omitted, but per the
# project's config-block convention this script forwards an explicit
# value for every numeric/boolean flag rather than relying on the
# child process's own defaults silently. Path-like flags (openings/
# json/pgn/gauntlet) default to "" (not forwarded at all — see
# arg_map below) since "no file" is arena.exe's own meaningful default
# for those, not a value to forward.
# ═══════════════════════════════════════════════════════════════════
GAMES                    = 2000      # games per pairing (>2 players) or total (2 players)
CONCURRENCY              = 0        # concurrent GAMES (not threads per game); 0 = arena.exe autodetects logical cores
MOVETIME_MS              = 100      # per-move time budget, ms; 0 = use NODES instead
NODES                    = 0        # per-move node budget, used only when movetime==0
MAX_PLIES                = 400      # game-length safety cap -> counted as a draw (400 is the project-wide value, see CLAUDE.md)
SEED                     = 1        # opening-cycling RNG seed
TT_MB                    = 8.0      # per-instance TTable memory budget, MB
DEFAULT_OPENING_PLIES    = 2        # PGN opening file: plies replayed per game before recording its FEN
DEFAULT_SPRT_ENABLED     = False    # run as a sequential SPRT gate instead of a fixed-length
                                    # match. Requires exactly 2 players (arena.exe enforces this).
DEFAULT_ELO0             = 0.0      # SPRT null hypothesis Elo
DEFAULT_ELO1             = 5.0      # SPRT alt hypothesis Elo
DEFAULT_ALPHA            = 0.05     # SPRT false-accept-H1 rate
DEFAULT_BETA             = 0.05     # SPRT false-accept-H0 rate
DEFAULT_UCI_TIMEOUT_MS   = 15000    # max wait for a uci: player's "bestmove" line
SAVE_PGN                 = True    # save games to standard PGN file in RESULTS_DIR
SAVE_EPD                 = False   # EPD requires eval_cp from net: players; off for uci: runs
SAVE_BIN                 = False   # BIN requires eval_cp from in-process net: players; off for uci: runs
RESULTS_DIR              = r"tests\arena_results"

OPENING_FOLDER           = r"openings\lines"  # walked recursively for opening files
DEFAULT_JSON_PATH        = ""       # "" = no --json output
DEFAULT_PGN_PATH         = ""       # explicit PGN override path (if empty and SAVE_PGN=True, uses RESULTS_DIR)
DEFAULT_BIN_PATH         = ""       # explicit BIN override path (if empty and SAVE_BIN=True, uses RESULTS_DIR)
DEFAULT_EPD_PATH         = ""       # explicit EPD override path (if empty and SAVE_EPD=True, uses RESULTS_DIR)
DEFAULT_GAUNTLET_SPEC    = ""       # "" = round-robin (no gauntlet candidate)

# --player is repeatable (net:/uci:/ref:), so it can't sit next to a scalar
# DEFAULT_* constant the way every flag above does -- this list IS its
# config-block equivalent (CLAUDE.md rule 8: every CLI-reachable input,
# including repeated/list-valued ones, must be settable by editing this
# file). Used only when --player is not given at all on the command line;
# any --player on the CLI overrides this list in its entirety (see main()).
# Same net:/uci:/ref: syntax as the CLI flag.
DEFAULT_PLAYERS: list[str] = [
    r"uci:engine\c\zchezz_v402\zchezz.exe",
    r"uci:engine\c\zchezz_v402\zchezz.exe",
]

# Mirrors CLAUDE.md's "Build Instructions -> Native (Windows/Linux)"
# block verbatim (minus the trailing backslash-continuations) so a
# built-from-git-worktree engine gets EXACTLY the flags the project's
# own docs say a release build uses — no drift between "how a human
# builds v3.14" and "how this script builds v3.14".
BUILD_CMD_TEMPLATE = [
    "gcc", "-O3", "-ffast-math", "-D_GNU_SOURCE", "-std=c11", "-mavxvnni", "-mavx2",
    "-I.", "-Wno-unused-variable", "-Wno-unused-but-set-variable",
    "-Wno-maybe-uninitialized", "-Wno-misleading-indentation",
    "-Wno-sign-compare", "-Wno-unused-function", "-Wno-parentheses",
    "-o", "zchezz.exe",
    "main.c", "board.c", "search.c", "nnue.c", "syzygy.c", "tbprobe.c", "book.c",
    "-static", "-lm", "-pthread",
]

# Flags accepted by arena.c's parse_args() that this script forwards
# verbatim (name -> "value" or "flag"). Anything not in this table
# that arrives as a --player is intercepted for ref: resolution;
# anything else unknown is passed through as-is (forward-compatible
# with new arena.exe flags this script doesn't know about yet).
FORWARD_VALUE_FLAGS = {
    "--games", "--threads", "--movetime", "--nodes", "--max-plies",
    "--openings", "--opening-plies", "--seed", "--tt-mb", "--gauntlet",
    "--elo0", "--elo1", "--alpha", "--beta", "--uci-timeout", "--json", "--pgn",
    "--bin", "--epd",
}
FORWARD_BOOL_FLAGS = {"--sprt"}
# Path-like flags: an empty-string default means "don't forward this
# flag at all" (arena.exe's own meaningful default), never "forward an
# empty path". Numeric/boolean flags are always forwarded explicitly.
PATH_LIKE_FLAGS = {"--openings", "--json", "--pgn", "--gauntlet", "--bin", "--epd"}


def log(msg):
    print(f"[run_arena] {msg}", file=sys.stderr)


def run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def git_rev_parse(ref, cwd=REPO_ROOT):
    r = run(["git", "rev-parse", ref], cwd=cwd)
    if r.returncode != 0:
        return None
    return r.stdout.strip()


def find_engine_dir(root):
    """Highest-numbered engine/c/zchezz_v* folder under `root` — see
    header comment 'BUILD RESOLUTION' for why this has to be a
    heuristic (the project's one-folder-per-version convention means
    the folder name at an arbitrary historical commit isn't knowable
    in advance)."""
    candidates = glob.glob(os.path.join(root, "engine", "c", "zchezz_v*"))
    candidates = [c for c in candidates if os.path.isdir(c) and os.path.isfile(os.path.join(c, "board.c"))]
    if not candidates:
        return None

    def version_key(path):
        name = os.path.basename(path)
        digits = "".join(ch for ch in name if ch.isdigit())
        try:
            return int(digits) if digits else -1
        except ValueError:
            return -1

    candidates.sort(key=version_key)
    return candidates[-1]


def build_in_dir(engine_dir, label):
    """Runs BUILD_CMD_TEMPLATE inside `engine_dir`. Returns the path to
    the resulting zchezz.exe, or None on failure (logged, not raised —
    caller decides whether that's fatal for this one player)."""
    log(f"building {label} in {engine_dir} ...")
    t0 = time.time()
    r = run(BUILD_CMD_TEMPLATE, cwd=engine_dir)
    if r.returncode != 0:
        log(f"BUILD FAILED for {label}:")
        log(r.stdout[-2000:])
        log(r.stderr[-2000:])
        return None
    exe = os.path.join(engine_dir, "zchezz.exe")
    if not os.path.isfile(exe):
        log(f"BUILD FAILED for {label}: gcc reported success but zchezz.exe is missing")
        return None
    log(f"built {label} in {time.time() - t0:.1f}s -> {exe}")
    return exe


def resolve_head():
    """ref:HEAD — build in place, no worktree. Reuses the cached exe
    if every source file in the folder is older than it."""
    exe = os.path.join(ENGINE_DIR_FOR_HEAD, "zchezz.exe")
    needs_build = True
    if os.path.isfile(exe):
        exe_mtime = os.path.getmtime(exe)
        srcs = glob.glob(os.path.join(ENGINE_DIR_FOR_HEAD, "*.c")) + \
               glob.glob(os.path.join(ENGINE_DIR_FOR_HEAD, "*.h"))
        needs_build = any(os.path.getmtime(s) > exe_mtime for s in srcs)
    if needs_build:
        exe = build_in_dir(ENGINE_DIR_FOR_HEAD, "HEAD")
        if not exe:
            return None
    else:
        log(f"HEAD: reusing up-to-date {exe}")
    return f"uci:{exe}"


def resolve_git_ref(ref):
    sha = git_rev_parse(ref)
    if not sha:
        log(f"ref '{ref}' does not resolve (git rev-parse failed) — SKIPPING this player")
        return None

    cache_dir = os.path.join(CACHE_DIR, sha)
    cached_exe = os.path.join(cache_dir, "zchezz.exe")
    if os.path.isfile(cached_exe):
        log(f"ref '{ref}' ({sha[:10]}) -> cached build, reusing {cached_exe}")
        return f"uci:{cached_exe}"

    log(f"ref '{ref}' -> {sha[:10]}, not cached — creating worktree")
    worktree_dir = os.path.join(CACHE_DIR, f"_wt_{sha[:12]}")
    if os.path.exists(worktree_dir):
        shutil.rmtree(worktree_dir, ignore_errors=True)
    r = run(["git", "worktree", "add", "--detach", worktree_dir, sha])
    if r.returncode != 0:
        log(f"ref '{ref}': git worktree add failed — SKIPPING this player\n{r.stderr}")
        return None

    try:
        engine_dir = find_engine_dir(worktree_dir)
        if not engine_dir:
            log(f"ref '{ref}' ({sha[:10]}): no engine/c/zchezz_v* folder found in worktree — SKIPPING")
            return None
        exe = build_in_dir(engine_dir, f"{ref} ({sha[:10]})")
        if not exe:
            return None   # build_in_dir already logged the failure — SKIP, don't abort the run
        os.makedirs(cache_dir, exist_ok=True)
        shutil.copy2(exe, cached_exe)
        weights = os.path.join(engine_dir, "nnue_weights.bin")
        if os.path.isfile(weights):
            shutil.copy2(weights, os.path.join(cache_dir, "nnue_weights.bin"))
        else:
            log(f"NOTE: {ref} ({sha[:10]}) has no nnue_weights.bin in its engine folder — "
                f"if it needs one at runtime, arena.exe spawns it with cwd={cache_dir}, "
                f"so it will fail to find weights unless you place one there manually.")
        book = os.path.join(engine_dir, "book.bin")
        if os.path.isfile(book):
            shutil.copy2(book, os.path.join(cache_dir, "book.bin"))
    finally:
        run(["git", "worktree", "remove", "--force", worktree_dir])

    return f"uci:{cached_exe}"


def resolve_player_spec(spec):
    """Returns a (possibly rewritten) spec string ready for arena.exe,
    or None if this spec should be dropped (failed ref resolution)."""
    if spec.startswith("net:") or spec.startswith("uci:"):
        return spec
    if not spec.startswith("ref:"):
        log(f"unrecognized --player spec (expected net:/uci:/ref:): {spec} — SKIPPING")
        return None

    ref = spec[4:]
    if ref.strip().upper() == "HEAD":
        return resolve_head()
    return resolve_git_ref(ref)


# arena.exe's own build command (engine/build/Makefile's `arena` target,
# mirrored here) — used as a fallback when neither `make` nor
# `mingw32-make` is on PATH (observed on at least one dev machine in this
# project: MSYS2/MinGW ships `mingw32-make`, not a `make` alias, and plain
# `gcc`-only setups may have neither). Paths are relative to BUILD_DIR,
# matching how the Makefile itself invokes gcc (ENGINE=v400 baked in here
# since this fallback only serves resolve_head()'s HEAD-in-place case).
ARENA_BUILD_CMD_TEMPLATE = [
    "gcc", "-O3", "-ffast-math", "-D_GNU_SOURCE", "-std=c11", "-mavxvnni", "-mavx2",
    "-I" + os.path.join("..", "c", "zchezz_v402"), "-DNO_TABLEBASES", "-DNO_BOOK",
    "-Wno-unused-variable", "-Wno-unused-but-set-variable",
    "-Wno-maybe-uninitialized", "-Wno-misleading-indentation",
    "-Wno-sign-compare", "-Wno-unused-function", "-Wno-parentheses",
    "-o", "arena.exe",
    os.path.join("..", "c", "tools", "arena.c"),
    os.path.join("..", "c", "tools", "opening_pool.c"),
    os.path.join("..", "c", "zchezz_v402", "board.c"),
    os.path.join("..", "c", "zchezz_v402", "search.c"),
    os.path.join("..", "c", "zchezz_v402", "nnue.c"),
    "-static", "-lm", "-pthread",
]


def ensure_arena_built():
    if os.path.isfile(ARENA_EXE):
        src = os.path.join(TOOLS_DIR, "arena.c")
        if os.path.getmtime(ARENA_EXE) >= os.path.getmtime(src):
            return True
        log("arena.exe is older than engine/c/tools/arena.c — rebuilding")

    for make_cmd in (["make", "ENGINE=v402", "arena"], ["mingw32-make", "ENGINE=v402", "arena"]):
        try:
            r = run(make_cmd, cwd=BUILD_DIR)
        except FileNotFoundError:
            continue
        if r.returncode == 0:
            return True
        log(f"`{' '.join(make_cmd)}` failed:\n{r.stdout}\n{r.stderr}")

    log("no working `make`/`mingw32-make` found — building arena.exe directly with gcc")
    r = run(ARENA_BUILD_CMD_TEMPLATE, cwd=BUILD_DIR)
    if r.returncode != 0:
        log(f"direct gcc build of arena.exe also failed:\n{r.stdout}\n{r.stderr}")
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--player", action="append", default=None,
                     help="net:<path> | uci:<path> | ref:<gitref-or-HEAD> (repeat, 2+ required). "
                          "Omit entirely to use the DEFAULT_PLAYERS config block above "
                          "(edit-the-file workflow); any --player on the CLI overrides it.")
    # Forwarded flags — same names/semantics as arena.exe. Defaults come
    # from the DEFAULT CONFIGURATION block above, not a second literal.
    ap.add_argument("--games", type=int, default=GAMES)
    ap.add_argument("--threads", type=int, default=CONCURRENCY)
    ap.add_argument("--movetime", type=int, default=MOVETIME_MS)
    ap.add_argument("--nodes", type=int, default=NODES)
    ap.add_argument("--max-plies", type=int, default=MAX_PLIES)
    ap.add_argument("--openings", default=OPENING_FOLDER)
    ap.add_argument("--opening-plies", type=int, default=DEFAULT_OPENING_PLIES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tt-mb", type=float, default=TT_MB)
    ap.add_argument("--gauntlet", default=DEFAULT_GAUNTLET_SPEC,
                     help="0-based index or name/path substring, forwarded as-is")
    ap.add_argument("--sprt", action="store_true", default=DEFAULT_SPRT_ENABLED)
    ap.add_argument("--elo0", type=float, default=DEFAULT_ELO0)
    ap.add_argument("--elo1", type=float, default=DEFAULT_ELO1)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--beta", type=float, default=DEFAULT_BETA)
    ap.add_argument("--uci-timeout", type=int, default=DEFAULT_UCI_TIMEOUT_MS)
    ap.add_argument("--json", default=DEFAULT_JSON_PATH)
    ap.add_argument("--pgn", default=DEFAULT_PGN_PATH,
                     help="write all games as standard PGN to FILE (GUI-loadable)")
    ap.add_argument("--bin", default=DEFAULT_BIN_PATH,
                     help="append packed training samples to FILE (sample.h/dataset.py format). "
                          "arena.exe requires every resolved player to be net: AND share the same "
                          "weight file -- a ref:/uci: player (or two different net: weight files) "
                          "will make arena.exe refuse to start; use --epd instead for mixed engines.")
    ap.add_argument("--epd", default=DEFAULT_EPD_PATH,
                     help="append FEN + c0/c1/c2/c3 opcode rows to FILE. Safe for any player mix "
                          "(ref:/uci:/net:, even different engines) -- c2 records the actual mover's "
                          "player name per row, so evaluator identity is never blended across engines.")
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


    players = args.player if args.player is not None else list(DEFAULT_PLAYERS)
    if args.player is None:
        log(f"no --player given -- using DEFAULT_PLAYERS config block: {players}")
    if len(players) < 2:
        log(f"error: need at least 2 --player entries (got {len(players)}, "
            f"from {'CLI' if args.player is not None else 'DEFAULT_PLAYERS'}). Aborting.")
        sys.exit(1)

    if not ensure_arena_built():
        sys.exit(1)

    resolved = []
    for spec in players:
        r = resolve_player_spec(spec)
        if r is not None:
            resolved.append(r)
        else:
            log(f"'{spec}' was dropped — continuing with the remaining players")

    if len(resolved) < 2:
        log(f"error: only {len(resolved)} player(s) resolved successfully, need >= 2. Aborting.")
        sys.exit(1)

    cmd = [ARENA_EXE]
    for spec in resolved:
        cmd += ["--player", spec]

    # Auto-resolve output paths from RESULTS_DIR if SAVE_* boolean flags are set
    pgn_path = args.pgn
    epd_path = args.epd
    bin_path = args.bin

    if SAVE_PGN or SAVE_EPD or SAVE_BIN or pgn_path or epd_path or bin_path:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        if SAVE_PGN and not pgn_path:
            pgn_path = os.path.join(RESULTS_DIR, f"arena_{ts}.pgn")
        if SAVE_EPD and not epd_path:
            epd_path = os.path.join(RESULTS_DIR, f"arena_{ts}.epd")
        if SAVE_BIN and not bin_path:
            bin_path = os.path.join(RESULTS_DIR, f"arena_{ts}.bin")

    arg_map = {
        "--games": args.games, "--threads": args.threads, "--movetime": args.movetime,
        "--nodes": args.nodes, "--max-plies": args.max_plies, "--openings": args.openings,
        "--opening-plies": args.opening_plies, "--seed": args.seed, "--tt-mb": args.tt_mb,
        "--gauntlet": args.gauntlet, "--elo0": args.elo0, "--elo1": args.elo1,
        "--alpha": args.alpha, "--beta": args.beta, "--uci-timeout": args.uci_timeout,
        "--json": args.json, "--pgn": pgn_path, "--bin": bin_path, "--epd": epd_path,
    }
    for flag, val in arg_map.items():
        if flag in PATH_LIKE_FLAGS and not val:
            continue   # "" means "arena.exe's own default applies" — don't forward an empty path
        cmd += [flag, str(val)]
    if args.sprt:
        cmd.append("--sprt")

    log(f"players resolved: {resolved}")
    log(f"running: {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
