#!/usr/bin/env python3
"""
run_tournament_quick.py — Quick 200-game H2H regression test
=========================================================

Same engine as run_tournament.py — this file is a clean copy with only the
config block edited (ANCHORS=[], SELF_PLAY=True, MOVETIME_MS lowered to 100,
GAMES_SELF_PLAY set to 300). Everything below "LOGGING" is intentionally
byte-for-byte identical to run_tournament.py; if you fix a bug in the
shared machinery (OpeningIndex, EngineInstance, play_game, the reporting
functions) here, mirror the fix there too — there is no shared module
between them, on purpose, so each file stays a single edit-and-run script
per CLAUDE.md rule 8's config-block convention. Like the other two, the
config block is the primary interface and every constant is ALSO a CLI flag
that overrides it (see the COMMAND LINE block after the config;
`--show-config` prints the resolved settings and runs nothing).

  python tests/run_tournament_quick.py

Default: v4.00 (engine under test) vs v3.14 (previous stable baseline),
         200 games (100 openings x 2 colors), 14 workers, 100ms/move,
         PGN + log saved, no EPD, no .bin.

OUTPUT FORMATS: SAVE_PGN / SAVE_EPD / SAVE_BIN are independent and additive
(CLAUDE.md rule 9) — see run_tournament.py's SAVE_BIN comment (mirrored
here) for why .bin, unlike EPD, is restricted to one engine's own moves
per run (SAVE_BIN_LABEL) rather than recording every mover's eval.

The two engine folders under engine/c/ are v4.00 and v3.14, which is the
pair this preset compares.

── RELATION TO OTHER TOOLS — WHEN TO USE WHICH ────────────────────────────

  This is the "fast sanity check" preset of run_tournament.py's H2H mode:
  200 games at 100ms/move finishes in minutes and is the thing to run
  right after a rebuild, before committing to a longer CLAUDE.md-rule-1
  regression run (500 games @ 100ms + 200 @ 200ms, 14 concurrency). It is
  NOT that regression run itself — it's a smaller, faster proxy for "did I
  obviously break something."

  For the ACTUAL promotion-gate decision (a version regressed or not, with
  a real sequential-test stopping rule), use run_arena.py — the native
  SPRT-driven A/B path (engine/c/tools/arena.c) — which is both faster
  per-game (no per-game Python process spawn/UCI-handshake overhead, see
  run_tournament.py's header docstring) and statistically the right tool
  (SPRT, not a fixed-N ELO-difference estimate). This script exists
  because sometimes you want a quick, no-git-worktree-building, "just point
  it at two already-built .exe paths" check before reaching for that.

  run_selfplay.py is unrelated to this file's purpose (training-data
  generation via persistent engine processes, not H2H regression testing).

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


import os, json, subprocess, time, math, random, threading, datetime, tempfile, signal, sys, io, struct, glob
from queue import Queue, Empty
import chess
import chess.pgn

# Import shared ELO calculator (trinomial model with proper 95% CI)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elo_calc import elo_difference as _elo_calc, estimated_elo as _estimated_elo

# train/ is imported ONLY for SAMPLE_DTYPE (the packed .bin record layout) —
# see run_tournament.py's identical comment (this file mirrors that shared
# machinery byte-for-byte, see header docstring). Cheap import even with
# SAVE_BIN off (dataset.py only imports numpy/os/glob at module scope).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))
import numpy as np
from dataset import SAMPLE_DTYPE

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — v4.01 vs v4.00 sanity check (1T, 100 games)
# ═══════════════════════════════════════════════════════════════════════════════

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Time Control ──────────────────────────────────────────────────────────────
MOVETIME_MS             = 100    # ms per move

# ── v4.01 (trained) vs v4.00 (placeholder weights) sanity check ───────────────
MY_ENGINES = [
    {
        "path":     r"engine\c\zchezz_v401\zchezz.exe",
        "label":    "v401-1T",
        "tc_mode":  "movetime",
        "tc_value": MOVETIME_MS,
        "tc_inc":   0,
        "options":  {"Threads": "1"},
    },
    {
        "path":     r"engine\c\zchezz_v400\zchezz.exe",
        "label":    "v400-1T",
        "tc_mode":  "movetime",
        "tc_value": MOVETIME_MS,
        "tc_inc":   0,
        "options":  {"Threads": "1"},
    },
]

# ── No anchors ───────────────────────────────────────────────────────────────
ANCHORS = []

# ── Tournament Parameters ─────────────────────────────────────────────────────
GAMES_VS_EACH_ANCHOR = 0
GAMES_SELF_PLAY      = 50     # 50 openings x 2 colors = 100 games (sanity check)
SELF_PLAY            = True
COLOR_SWAP           = True

CONCURRENCY          = 14     # 14 concurrent (1T engines)
MAX_PLIES            = 400
MOVE_TIMEOUT_S         = 38.0
REPORT_PERFORMANCE_METRICS = True

# ── Opening Book ──────────────────────────────────────────────────────────────
OPENING_FOLDER       = r"openings\lines"
OPENING_FILES        = ["8moves_v3.pgn", "Blitz_Testing_4moves.pgn"]

# ── Output ────────────────────────────────────────────────────────────────────
# SAVE_PGN / SAVE_EPD / SAVE_BIN are independent and additive (CLAUDE.md rule
# 9) — see run_tournament.py's SAVE_BIN comment for the full design rationale
# behind SAVE_BIN's single-evaluator restriction (mirrored here verbatim).
SAVE_PGN             = True   # save games for review
SAVE_EPD             = False  # no training data for quick tests
SAVE_BIN             = False  # no training data for quick tests (same
                               # single-evaluator restriction as
                               # run_tournament.py's SAVE_BIN; set
                               # SAVE_BIN_LABEL below if you turn this on)
SAVE_BIN_LABEL       = ""     # required (non-empty) whenever SAVE_BIN=True —
                               # exact match against an engine "label" above
RESULTS_DIR          = r"tests\quick_results"
COUNTER_EVERY        = 10
REPORT_LOOPS         = 200    # report at end (200 openings)

MATE_SCORE           = 999999  # sentinel value for mate scores

# ── Head-to-head overrides (change the pair without editing MY_ENGINES) ──────
# MY_ENGINES above stays the authoritative description of the two players
# (it carries per-engine UCI options a flag cannot express); these let a
# scripted A/B point the same setup at a different pair of builds.
ENGINE_A_PATH  = ""    # "" = keep MY_ENGINES[0]["path"]
ENGINE_A_LABEL = ""    # "" = keep MY_ENGINES[0]["label"]
ENGINE_B_PATH  = ""    # "" = keep MY_ENGINES[1]["path"]
ENGINE_B_LABEL = ""    # "" = keep MY_ENGINES[1]["label"]
THREADS        = 0     # 0 = keep each engine's own Threads option; >0 = set both

# ═══════════════════════════════════════════════════════════════════════════════
#  COMMAND LINE — every constant above is also a flag (CLAUDE.md rule 8)
#
#  The config block IS the interface: a bare `python tests/run_tournament_quick.py`
#  runs exactly what the constants say. The flags only OVERRIDE them;
#  `--show-config` prints the resolved values and exits. See utils/cliconf.py.
# ═══════════════════════════════════════════════════════════════════════════════
sys.path.insert(0, os.path.join(_BASE_DIR, "utils"))
from cliconf import override_from_cli  # noqa: E402

CLI = [
    ("MOVETIME_MS",          "--movetime",        int,   "ms per move"),
    ("GAMES_SELF_PLAY",      "--games",           int,   "openings per engine pair (x2 with color swap)"),
    ("GAMES_VS_EACH_ANCHOR", "--games-vs-anchor", int,   "openings against each anchor (ANCHORS is empty here)"),
    ("SELF_PLAY",            "--self-play",       bool,  "engines in MY_ENGINES play each other"),
    ("COLOR_SWAP",           "--color-swap",      bool,  "repeat each opening with colors reversed"),
    ("CONCURRENCY",          "--concurrency",     int,   "concurrent games (worker threads)"),
    ("MAX_PLIES",            "--max-plies",       int,   "half-move cap before a forced draw"),
    ("MOVE_TIMEOUT_S",         "--move-timeout",    float, "seconds to wait for a move before a timeout loss"),
    ("OPENING_FOLDER",       "--openings",        str,   "folder with .epd/.pgn opening files"),
    ("OPENING_FILES",        "--opening-file",    list,  "load ONLY these files from --openings"),
    ("SAVE_PGN",             "--pgn",             bool,  "save all games as PGN"),
    ("SAVE_EPD",             "--epd",             bool,  "save positions + eval as EPD"),
    ("SAVE_BIN",             "--bin",             bool,  "save packed .bin training samples"),
    ("SAVE_BIN_LABEL",       "--bin-label",       str,   "engine label whose moves go into the .bin"),
    ("RESULTS_DIR",          "--results-dir",     str,   "where the .log/.pgn/.epd/.bin land"),
    ("COUNTER_EVERY",        "--counter-every",   int,   "progress line every N games"),
    ("REPORT_LOOPS",         "--report-loops",    int,   "full cross-table every N mini-cycles"),
    ("REPORT_PERFORMANCE_METRICS", "--perf",      bool,  "show NPS / time-per-move in the reports"),
    ("ENGINE_A_PATH",        "--engine-a",        str,   "override MY_ENGINES[0] path"),
    ("ENGINE_A_LABEL",       "--label-a",         str,   "override MY_ENGINES[0] label"),
    ("ENGINE_B_PATH",        "--engine-b",        str,   "override MY_ENGINES[1] path"),
    ("ENGINE_B_LABEL",       "--label-b",         str,   "override MY_ENGINES[1] label"),
    ("THREADS",              "--threads",         int,   "set the Threads UCI option on BOTH engines (0 = leave as configured)"),
]

if __name__ == "__main__":
    override_from_cli(globals(), CLI, description=__doc__,
                      prog="run_tournament_quick.py")
    _overrides = ((ENGINE_A_PATH, ENGINE_A_LABEL), (ENGINE_B_PATH, ENGINE_B_LABEL))
    for _eng, (_p, _l) in zip(MY_ENGINES, _overrides):
        if _p: _eng["path"]  = _p
        if _l: _eng["label"] = _l
        if THREADS:
            _eng.setdefault("options", {})["Threads"] = str(THREADS)
        # tc_value was copied from MOVETIME_MS when the list was defined, so
        # it has to follow a --movetime override.
        if _eng.get("tc_mode") == "movetime":
            _eng["tc_value"] = MOVETIME_MS

# ═══════════════════════════════════════════════════════════════════════════════
#  LOGGING — All output goes to both console and log file
# ═══════════════════════════════════════════════════════════════════════════════

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_FILE  = os.path.join(RESULTS_DIR, f"torneio_{TIMESTAMP}.log")
ALL_PGNS  = os.path.join(RESULTS_DIR, f"torneio_{TIMESTAMP}.pgn")
ALL_EPDS  = os.path.join(RESULTS_DIR, f"torneio_{TIMESTAMP}.epd")
ALL_BINS  = os.path.join(RESULTS_DIR, f"torneio_{TIMESTAMP}.bin")

if SAVE_BIN and not SAVE_BIN_LABEL:
    print("[run_tournament_quick] FATAL: SAVE_BIN=True requires a non-empty SAVE_BIN_LABEL "
          "(the exact 'label' of the one engine whose moves/evals get recorded to "
          ".bin — see SAVE_BIN's config comment). Aborting before any games are played.")
    sys.exit(1)

_log_lock = threading.Lock()

def log(*args):
    msg = " ".join(map(str, args))
    with _log_lock:
        try: print(msg, flush=True)
        except UnicodeEncodeError: print(msg.encode("ascii","replace").decode(), flush=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f: f.write(msg + "\n")

# ═══════════════════════════════════════════════════════════════════════════════
#  OPENING INDEX — Memory-efficient: stores only byte offsets, not positions
#  Supports .epd and .pgn files. Caches EPD offsets as .idx binary files.
# ═══════════════════════════════════════════════════════════════════════════════

class OpeningIndex:
    def __init__(self):
        self._index = []
        self._handles = {}
        self._lock = threading.Lock()

    def __len__(self):  return len(self._index)
    def __bool__(self): return bool(self._index)

    def _open(self, fpath):
        with self._lock:
            if fpath not in self._handles:
                self._handles[fpath] = open(fpath, "rb")
            return self._handles[fpath]

    def _build_epd(self, fpath):
        idx_path = fpath + ".idx"
        if os.path.exists(idx_path) and os.path.getmtime(idx_path) >= os.path.getmtime(fpath):
            try:
                data = open(idx_path, "rb").read()
                offsets = list(struct.unpack_from(f"<{len(data)//8}Q", data))
                for off in offsets: self._index.append((fpath, off, "epd"))
                log(f"  [EPD] {os.path.basename(fpath)}: {len(offsets):,} pos (cached)")
                return
            except Exception: pass
        offsets = []
        with open(fpath, "rb") as f:
            while True:
                off = f.tell(); raw = f.readline()
                if not raw: break
                s = raw.strip()
                if s and not s.startswith(b"#") and len(s.split()) >= 4:
                    offsets.append(off)
        try:
            with open(idx_path, "wb") as f:
                f.write(struct.pack(f"<{len(offsets)}Q", *offsets))
        except Exception: pass
        for off in offsets: self._index.append((fpath, off, "epd"))
        log(f"  [EPD] {os.path.basename(fpath)}: {len(offsets):,} pos indexed")

    def _build_pgn(self, fpath):
        count = 0
        with open(fpath, "rb") as f:
            game_start = None
            while True:
                off = f.tell(); raw = f.readline()
                if not raw:
                    if game_start is not None: self._index.append((fpath, game_start, "pgn")); count += 1
                    break
                line = raw.decode("utf-8", errors="ignore")
                if line.startswith("[Event "):
                    game_start = off
                elif line.strip() == "" and game_start is not None:
                    self._index.append((fpath, game_start, "pgn")); count += 1; game_start = None
        log(f"  [PGN] {os.path.basename(fpath)}: {count:,} games")

    def shuffle(self): random.shuffle(self._index)

    def fetch(self, idx) -> dict:
        fpath, offset, kind = self._index[idx]
        if kind == "epd":
            fh = self._open(fpath)
            with self._lock:
                fh.seek(offset)
                raw = fh.readline().decode("utf-8", errors="ignore").strip()
            parts = raw.split()
            fen = " ".join(parts[:4]) + " 0 1"
            try: chess.Board(fen); return {"moves": [], "fen": fen}
            except Exception: return {"moves": [], "fen": None}
        else:
            with open(fpath, "rb") as f:
                f.seek(offset)
                block, blanks = [], 0
                while True:
                    raw = f.readline()
                    if not raw: break
                    line = raw.decode("utf-8", errors="ignore")
                    block.append(line)
                    if line.strip() == "":
                        blanks += 1
                        if blanks >= 2: break
                    else: blanks = 0
            game = chess.pgn.read_game(io.StringIO("".join(block)))
            if not game: return {"moves": [], "fen": None}
            uci_seq, board = [], game.board()
            for move in game.mainline_moves():
                if move in board.legal_moves: uci_seq.append(move.uci()); board.push(move)
                else: break
            return {"moves": uci_seq, "fen": None}

    def random_pick(self) -> dict:
        return self.fetch(random.randrange(len(self._index)))

def load_all_openings(folder: str, filter_files: list = None) -> OpeningIndex:
    """Load openings from folder. If filter_files is non-empty, only load those files."""
    idx = OpeningIndex()
    if not os.path.exists(folder): return idx
    # Walk recursively: openings/ is split into lines/ (.pgn move sequences
    # played from the start position) and positions/ (.epd start positions).
    # A flat listdir() would silently find nothing and report 0 openings.
    all_files = sorted(
        os.path.relpath(os.path.join(root, f), folder)
        for root, _dirs, files in os.walk(folder) for f in files
    )
    if filter_files:
        # Match on either the bare filename or the lines/foo.pgn relative path,
        # so OPENING_FILES entries stay valid after the subfolder split.
        all_files = [f for f in all_files
                     if f in filter_files or os.path.basename(f) in filter_files]
        if not all_files:
            log(f"  [WARNING] No matching files in {folder} for filter: {filter_files}")
    for f in all_files:
        fp = os.path.join(folder, f)
        if f.endswith(".epd"): idx._build_epd(fp)
        elif f.endswith(".pgn"): idx._build_pgn(fp)
    log(f"  [Openings] Total: {len(idx):,} positions from {len(all_files)} file(s)")
    return idx

# ═══════════════════════════════════════════════════════════════════════════════
#  ENGINE INSTANCE — Persistent UCI engine process
#  Each worker thread creates its own EngineInstance pair per game.
#  Supports movetime, depth, nodes, and fixedtime time controls.
#  Auto-detects NNUE weights (nnue_weights.bin) in the engine directory.
# ═══════════════════════════════════════════════════════════════════════════════

class EngineInstance:
    def __init__(self, path, label, tc_mode="depth", tc_value=8, tc_inc=0,
                 options=None, go_cmd_override=None, elo=None, **_):
        self.path     = os.path.abspath(path)
        self.label    = label
        self.tc_mode  = tc_mode
        self.tc_value = tc_value
        self.tc_inc   = tc_inc
        self.options  = options or {}
        self.go_cmd_override = go_cmd_override
        self.elo      = elo   # known elo (anchors only)
        self.process  = None
        self.queue    = Queue()
        self.start_fen = None

    def _reader(self):
        try:
            for line in iter(self.process.stdout.readline, ""):
                line = line.strip()
                if line: self.queue.put(line)
        except Exception: pass

    def start(self):
        engine_dir = os.path.dirname(self.path)
        try:
            args = [self.path]
            if os.path.exists(os.path.join(engine_dir, "nnue_weights.bin")):
                args += ["--nnue", "nnue_weights.bin"]
            self.process = subprocess.Popen(
                args, cwd=engine_dir,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            )
            threading.Thread(target=self._reader, daemon=True).start()
            threading.Thread(target=lambda: self.process.stderr.read(), daemon=True).start()

            self._send("uci")
            t0 = time.time(); ok = False
            while time.time() - t0 < 15:
                try:
                    if "uciok" in self.queue.get(timeout=0.1).lower(): ok = True; break
                except Empty:
                    if self.process.poll() is not None: break
            if not ok: return False

            for name, val in self.options.items():
                self._send(f"setoption name {name} value {val}")

            self._send("isready")
            t0 = time.time()
            while time.time() - t0 < 15:
                try:
                    if "readyok" in self.queue.get(timeout=0.1).lower(): break
                except Empty: pass

            self._send("ucinewgame")
            self._send("isready")
            t0 = time.time()
            while time.time() - t0 < 10:
                try:
                    if "readyok" in self.queue.get(timeout=0.1).lower(): break
                except Empty: pass
            return True
        except Exception as e:
            log(f"  [ERRO] {self.label}: {e}")
            return False

    def _send(self, cmd):
        if self.process and self.process.stdin:
            try:
                self.process.stdin.write(cmd + "\n")
                self.process.stdin.flush()
                return True
            except Exception: pass
        return False

    def _build_go(self, wtime, btime):
        if self.go_cmd_override:
            return f"go {self.go_cmd_override}"
        if self.tc_mode == "depth":
            return f"go depth {self.tc_value}"
        if self.tc_mode == "movetime":
            return f"go movetime {self.tc_value}"
        if self.tc_mode == "nodes":
            return f"go nodes {self.tc_value}"
        # fixedtime
        return f"go wtime {wtime} btime {btime} winc {self.tc_inc} binc {self.tc_inc}"

    def get_move(self, board, wtime=0, btime=0):
        if not self.process or self.process.poll() is not None:
            if not self.start(): return None, None

        while not self.queue.empty():
            try: self.queue.get_nowait()
            except Empty: break

        moves = " ".join(m.uci() for m in board.move_stack)
        if self.start_fen:
            pos = f"position fen {self.start_fen}" + (f" moves {moves}" if moves else "")
        else:
            pos = "position startpos" + (f" moves {moves}" if moves else "")
        self._send(pos)
        self._send(self._build_go(wtime, btime))

        move = None; score = None; nodes = 0; depth = 0
        t0 = time.time()
        while time.time() - t0 < MOVE_TIMEOUT_S:
            try:
                line = self.queue.get(timeout=0.2)
                if line.startswith("info "):
                    parts = line.split()
                    if "nodes" in parts:
                        try: nodes = int(parts[parts.index("nodes")+1])
                        except Exception: pass
                    if "depth" in parts:
                        try: depth = int(parts[parts.index("depth")+1])
                        except Exception: pass
                if "score cp" in line:
                    parts = line.split()
                    try:
                        i = parts.index("cp"); score = int(parts[i+1])
                        # Convert to white's perspective: if it's black's turn, negate
                        if board.turn == chess.BLACK: score = -score
                    except Exception: pass
                elif "score mate" in line:
                    parts = line.split()
                    try:
                        i = parts.index("mate"); m_val = int(parts[i+1])
                        # positive mate = current side mates; from white's side:
                        if board.turn == chess.WHITE:
                            score = MATE_SCORE if m_val > 0 else -MATE_SCORE
                        else:
                            score = MATE_SCORE if m_val < 0 else -MATE_SCORE
                    except Exception: pass
                if line.startswith("bestmove"):
                    parts = line.split()
                    if len(parts) > 1 and parts[1] not in ("(none)", "0000"):
                        move = parts[1]
                    break
            except Empty:
                if self.process.poll() is not None: break
        return move, score, nodes, depth

    def stop(self):
        if self.process:
            try: self._send("quit")
            except Exception: pass
            try: self.process.terminate(); self.process.wait(timeout=1.0)
            except Exception: pass
            self.process = None

# ═══════════════════════════════════════════════════════════════════════════════
#  ELO ESTIMATOR — Maximum Likelihood Estimation across anchors
#  Groups results by anchor ELO, computes per-anchor estimate via elo_calc,
#  then combines them using inverse-variance weighting for a final estimate.
# ═══════════════════════════════════════════════════════════════════════════════

class EloEstimator:
    def __init__(self):
        self.results = []  # list of (opponent_elo_float, score_float)

    def add(self, opp_elo, score):
        self.results.append((opp_elo, score))

    def estimate_mle(self):
        if not self.results: return None, None
        # Group by anchor ELO
        per_anchor = {}
        for anchor_elo, score in self.results:
            if anchor_elo not in per_anchor:
                per_anchor[anchor_elo] = {"w": 0, "d": 0, "l": 0}
            if score == 1.0: per_anchor[anchor_elo]["w"] += 1
            elif score == 0.0: per_anchor[anchor_elo]["l"] += 1
            else: per_anchor[anchor_elo]["d"] += 1
        
        # Weighted average across anchors
        weighted_elo = 0.0
        combined_var = 0.0
        for a_elo, s in per_anchor.items():
            est, ci = _estimated_elo(s["w"], s["d"], s["l"], a_elo)
            se = ci / 1.96 if ci < float('inf') and ci > 0 else float('inf')
            if se == 0 or se == float('inf'): continue
            weight = 1.0 / se ** 2
            weighted_elo += est * weight
            combined_var += weight
        
        if combined_var > 0:
            final_elo = weighted_elo / combined_var
            final_se = 1.0 / math.sqrt(combined_var)
            return final_elo, final_se * 1.96
        return None, None


# ═══════════════════════════════════════════════════════════════════════════════
#  STATS & ELO HELPERS — Global state for tracking results
#  Thread-safe via _stats_lock. Updated after each game completes.
# ═══════════════════════════════════════════════════════════════════════════════

_stats_lock = threading.Lock()
_all_labels    = []   # filled in main()
_my_labels     = []   # MY_ENGINES labels
stats          = {}   # label -> {w,l,d,pts,games,total_ms,total_moves,to,err}
h2h            = {}   # (l1,l2) -> {w,l,d}
elo_estimators = {}   # my_label -> EloEstimator
completed_games = 0
epd_positions   = 0
bin_samples     = 0
start_time      = time.time()


def _new_stats():
    return {"w": 0, "l": 0, "d": 0, "pts": 0, "games": 0, "to": 0, "err": 0, 
            "total_ms": 0, "total_moves": 0, "total_nodes": 0, "total_depth": 0}


def elo_diff(wins, losses, draws):
    N = wins + losses + draws
    if N == 0: return "N/A"
    elo, ci, _ = _elo_calc(wins, draws, losses)
    sign = "+" if elo >= 0 else ""
    return f"{sign}{elo:.1f} ±{ci:.1f}"


def format_eval(score, elapsed_ms):
    t = f" | {elapsed_ms}ms" if elapsed_ms is not None else ""
    if score is None:
        return ("{ " + str(elapsed_ms) + "ms } ") if elapsed_ms else ""
    if abs(score) >= MATE_SCORE // 2:
        side = "+" if score > 0 else "-"
        return "{ #" + side + "M" + t + " } "
    cp = score / 100.0
    sign = "+" if cp >= 0 else ""
    return "{ " + sign + f"{cp:.2f}" + t + " } "

# ═══════════════════════════════════════════════════════════════════════════════
#  PLAY GAME — One complete game between two engine instances
#  Creates engine processes, applies opening, plays until game-over or MAX_PLIES.
#  Returns result via results_queue (thread-safe).
# ═══════════════════════════════════════════════════════════════════════════════

def play_game(game_id, white_cfg, black_cfg, results_queue, opening=None):
    w_eng = EngineInstance(**white_cfg)
    b_eng = EngineInstance(**black_cfg)

    opening_fen = opening.get("fen") if opening else None
    forced_moves = opening.get("moves") if opening else []

    w_eng.start_fen = opening_fen
    b_eng.start_fen = opening_fen

    if not w_eng.start() or not b_eng.start():
        results_queue.put({"id": game_id, "error": True,
                           "white": w_eng.label, "black": b_eng.label})
        w_eng.stop(); b_eng.stop()
        return

    board    = chess.Board(opening_fen) if opening_fen else chess.Board()
    pgn_game = chess.pgn.Game()
    pgn_game.headers["Event"]   = "Tournament Complete"
    pgn_game.headers["Round"]   = str(game_id)
    pgn_game.headers["White"]   = w_eng.label
    pgn_game.headers["Black"]   = b_eng.label
    pgn_game.headers["WhiteTC"] = f"{w_eng.tc_mode} {w_eng.tc_value}"
    pgn_game.headers["BlackTC"] = f"{b_eng.tc_mode} {b_eng.tc_value}"
    if opening_fen:
        pgn_game.headers["FEN"]   = opening_fen
        pgn_game.headers["SetUp"] = "1"

    # Apply forced opening moves
    node = pgn_game
    for uci in (forced_moves or []):
        try:
            mv = chess.Move.from_uci(uci)
            if mv in board.legal_moves: board.push(mv); node = node.add_main_variation(mv)
            else: break
        except Exception: break

    # positions list: each entry = {"fen": str, "score_white": int|None}
    # We collect (fen_before_move, score_from_white) for every engine move
    positions = []
    # Record opening positions (no score)
    tmp_board = chess.Board(opening_fen) if opening_fen else chess.Board()
    positions.append({"fen": tmp_board.fen(), "score_white": None, "evaluator": "book"})
    for uci in (forced_moves or []):
        try:
            mv = chess.Move.from_uci(uci)
            if mv in tmp_board.legal_moves:
                tmp_board.push(mv)
                positions.append({"fen": tmp_board.fen(), "score_white": None, "evaluator": "book"})
            else: break
        except Exception: break

    wtime = w_eng.tc_value if w_eng.tc_mode == "fixedtime" else 0
    btime = b_eng.tc_value if b_eng.tc_mode == "fixedtime" else 0

    ply = 0
    winner = None
    result_reason = "Normal"
    error_occurred = False
    timeout_occurred = False
    timing = {w_eng.label: {"ms": 0, "moves": 0, "nodes": 0, "depth": 0}, 
              b_eng.label: {"ms": 0, "moves": 0, "nodes": 0, "depth": 0}}

    try:
        while not board.is_game_over() and ply < MAX_PLIES:
            is_white = board.turn == chess.WHITE
            active   = w_eng if is_white else b_eng

            t0 = time.time()
            move_str, score, move_nodes, move_depth = active.get_move(board, wtime, btime)
            elapsed_ms = int((time.time() - t0) * 1000)

            timing[active.label]["ms"]    += elapsed_ms
            timing[active.label]["moves"] += 1
            timing[active.label]["nodes"] += move_nodes
            timing[active.label]["depth"] += move_depth

            if active.tc_mode == "fixedtime":
                if is_white: wtime = max(0, wtime - elapsed_ms + w_eng.tc_inc)
                else:        btime = max(0, btime - elapsed_ms + b_eng.tc_inc)

            if not move_str:
                timeout_occurred = True
                result_reason = f"Timeout ({active.label})"
                winner = "black" if is_white else "white"
                break

            try:
                move = board.parse_uci(move_str)
                if move not in board.legal_moves or move == chess.Move.null():
                    raise ValueError(f"Illegal: {move_str}")
                # Record position before push
                positions.append({"fen": board.fen(), "score_white": score, "evaluator": active.label, "move": move})
                board.push(move)
                comment = format_eval(score, elapsed_ms)
                node = node.add_main_variation(move, comment=comment)
                ply += 1
            except Exception as e:
                error_occurred = True
                result_reason = f"Lance Inválido ({move_str}) [{active.label}]"
                winner = "black" if is_white else "white"
                break

        # Determine result
        if not timeout_occurred and not error_occurred:
            if ply >= MAX_PLIES:
                result_reason = "Empate (Limite de lances)"; winner = None
            elif board.is_checkmate():
                result_reason = "Checkmate"
                winner = "white" if board.turn == chess.BLACK else "black"
            elif board.is_stalemate():
                result_reason = "Afogamento"; winner = None
            elif board.is_insufficient_material():
                result_reason = "Material Insuficiente"; winner = None
            elif board.is_fifty_moves():
                result_reason = "Regra dos 50 Lances"; winner = None
            elif board.is_repetition(3):
                result_reason = "Triplice Repetição"; winner = None
            else:
                result_reason = "Empate"; winner = None

        # Add final position
        positions.append({"fen": board.fen(), "score_white": None})

        pgn_result = "1-0" if winner == "white" else ("0-1" if winner == "black" else "1/2-1/2")
        pgn_game.headers["Result"]      = pgn_result
        pgn_game.headers["Termination"] = result_reason

        exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=True)
        pgn_str  = pgn_game.accept(exporter)

        results_queue.put({
            "id":        game_id,
            "white":     w_eng.label,
            "black":     b_eng.label,
            "winner":    winner,
            "reason":    result_reason,
            "result":    pgn_result,
            "pgn":       pgn_str,
            "positions": positions,
            "timing":    timing,
            "timeout":   timeout_occurred,
            "error":     error_occurred,
        })
    finally:
        w_eng.stop()
        b_eng.stop()

# ═══════════════════════════════════════════════════════════════════════════════
#  REPORTS — Cross-table, ELO estimates, H2H, and performance metrics
# ═══════════════════════════════════════════════════════════════════════════════

def print_counter():
    elapsed = time.time() - start_time
    parts = []
    for lb in _my_labels:
        s = stats.get(lb, _new_stats())
        g = s["games"]
        parts.append(f"{lb}: {s['pts']:.1f}pts/{g}g")
    bin_part = f" | BIN: {bin_samples} samples" if SAVE_BIN else ""
    log(f"  [{completed_games}] {' | '.join(parts)} | EPD: {epd_positions} pos{bin_part} | {elapsed:.0f}s")


def print_cross_table(is_final=False):
    elapsed = time.time() - start_time
    gps = completed_games / elapsed if elapsed > 0 else 0
    title = "RELATÓRIO FINAL" if is_final else f"RELATÓRIO (após {completed_games} jogos)"
    log(f"\n{'='*64}")
    log(f" {title}")
    bin_part = f" | BIN: {bin_samples} samples" if SAVE_BIN else ""
    log(f" Jogos: {completed_games} | {elapsed:.0f}s | {gps:.2f} j/s | EPD: {epd_positions} pos{bin_part}")
    log(f"{'='*64}")

    all_p = _all_labels
    col_w = max(14, max(len(lb) for lb in all_p) + 2)
    row_w = col_w

    # header
    hdr = f"{'':>{row_w}}"
    for lb in all_p: hdr += f" {lb:^{col_w}}"
    log(hdr)
    log("-" * len(hdr))

    # rows
    for lb in all_p:
        row = f"{lb:>{row_w}}"
        for lb2 in all_p:
            if lb == lb2:
                row += f" {'---':^{col_w}}"
            else:
                s = h2h.get((lb, lb2), {"w":0,"l":0,"d":0})
                g = s["w"] + s["l"] + s["d"]
                pts = s["w"] + s["d"] * 0.5
                cell = f"{pts:.1f}/{g}" if g > 0 else "-"
                row += f" {cell:^{col_w}}"
        # overall
        st = stats.get(lb, _new_stats())
        row += f"  {st['pts']:.1f}/{st['games']}"
        log(row)

    log("")
    # ELO MLE for my engines
    log(" ELO Estimado (MLE):")
    for lb in _my_labels:
        est = elo_estimators.get(lb)
        if est:
            e, m = est.estimate_mle()
            if e is not None:
                log(f"   {lb}: {e:.1f} +/-{m:.1f} Elo")
            else:
                log(f"   {lb}: (sem dados suficientes)")

    # H2H between my engines
    if len(_my_labels) > 1:
        log("")
        log(" H2H Direto (Meus Engines):")
        for i in range(len(_my_labels)):
            for j in range(i+1, len(_my_labels)):
                n1, n2 = _my_labels[i], _my_labels[j]
                s = h2h.get((n1, n2), {"w":0,"l":0,"d":0})
                if s["w"]+s["l"]+s["d"] == 0: continue
                ed = elo_diff(s["w"], s["l"], s["d"])
                log(f"   {n1} vs {n2}: {ed} Elo  (V:{s['w']} E:{s['d']} D:{s['l']})")

    log(f"{'='*64}\n")

    if REPORT_PERFORMANCE_METRICS:
        log(" Performance (Médias por Engine):")
        log(f" {'Engine':<16} | {'n/s':>10} | {'time/move':>12} | {'nodes/move':>12} | {'depth/move':>10}")
        log("-" * 64)
        for lb in _all_labels:
            st = stats.get(lb, _new_stats())
            tmv = st["total_moves"]
            if tmv > 0:
                t_sec = st["total_ms"] / 1000.0
                nps = st["total_nodes"] / t_sec if t_sec > 0 else 0
                t_pm = st["total_ms"] / tmv
                n_pm = st["total_nodes"] / tmv
                d_pm = st["total_depth"] / tmv
                log(f" {lb:<16} | {nps:>10.0f} | {t_pm:>10.0f}ms | {n_pm:>12.0f} | {d_pm:>10.1f}")
        log(f"{'='*64}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  SCHEDULE BUILDER — Creates the full game schedule
#  Interleaves anchor matches and self-play, applies COLOR_SWAP,
#  then shuffles for fair concurrency distribution.
# ═══════════════════════════════════════════════════════════════════════════════

def build_schedule(op_idx):
    my_cfgs     = MY_ENGINES
    anchor_cfgs = ANCHORS
    schedule    = []
    gid         = 1

    pairs = []
    # MY_ENGINES vs ANCHORS
    for my in my_cfgs:
        for anc in anchor_cfgs:
            pairs.append((my, anc, GAMES_VS_EACH_ANCHOR))

    # MY_ENGINES self-play
    if SELF_PLAY:
        for i in range(len(my_cfgs)):
            for j in range(i+1, len(my_cfgs)):
                pairs.append((my_cfgs[i], my_cfgs[j], GAMES_SELF_PLAY))

    max_games = max((p[2] for p in pairs), default=0)
    for round_num in range(max_games):
        round_matches = []
        for w_cfg, b_cfg, total_games in pairs:
            if round_num < total_games:
                idx = random.randrange(len(op_idx)) if op_idx and len(op_idx) > 0 else None
                round_matches.append((w_cfg, b_cfg, idx))
                if COLOR_SWAP:
                    round_matches.append((b_cfg, w_cfg, idx))
        
        random.shuffle(round_matches)
        for w_cfg, b_cfg, idx in round_matches:
            schedule.append((gid, w_cfg, b_cfg, idx))
            gid += 1

    return schedule

# ═══════════════════════════════════════════════════════════════════════════════
#  EPD WRITER — Saves positions + evaluations for NNUE training data
#  Format: FEN c0 "result"; c1 "score_cp"; c2 "evaluator"; c3 "ply_index";
# ═══════════════════════════════════════════════════════════════════════════════

_epd_lock = threading.Lock()

def write_epd(f_epd, positions, winner):
    """
    positions: list of {"fen": str, "score_white": int|None, "evaluator": str}
    winner: "white" | "black" | None
    """
    if winner == "white": pgn_res = "1-0"
    elif winner == "black": pgn_res = "0-1"
    else: pgn_res = "1/2-1/2"
    
    lines = []
    for idx, pos in enumerate(positions):
        fen = pos["fen"]
        sc  = pos["score_white"]
        
        norm_score = 0
        if sc is not None:
            if abs(sc) >= MATE_SCORE // 2:
                norm_score = MATE_SCORE if sc > 0 else -MATE_SCORE
            else:
                norm_score = sc
                
        lines.append(f"{fen} c0 \"{pgn_res}\"; c1 \"{norm_score}\"; c2 \"{pos.get('evaluator', '')}\"; c3 \"{idx}\";\n")
        
    with _epd_lock:
        f_epd.writelines(lines)
        f_epd.flush()
    return len(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  .bin WRITER — packed training samples, SAME schema as tools/selfplay.c /
#  tools/arena.c (train/dataset.py's SAMPLE_DTYPE). Mirrors
#  run_tournament.py's identical section byte-for-byte (see header
#  docstring) — see SAVE_BIN's config-block comment for the design
#  rationale: only SAVE_BIN_LABEL's own moves are encoded, so one .bin file
#  never blends two evaluators' opinions.
# ═══════════════════════════════════════════════════════════════════════════════

_bin_lock = threading.Lock()

def _board_to_zchezz_mailbox(board: "chess.Board") -> bytes:
    """python-chess Board -> Zchezz mailbox bytes[64] (0=empty, WP=9..BK=22,
    zsq 0=a8..63=h1). zsq = python-chess-square ^ 56 (CLAUDE.md coordinate
    invariant: sq_w = zsq ^ 56, self-inverse)."""
    arr = bytearray(64)
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece is None:
            continue
        zsq = sq ^ 56
        color_bit = 8 if piece.color == chess.WHITE else 16  # COL_W=8, COL_B=16
        arr[zsq] = color_bit + piece.piece_type
    return bytes(arr)

def _castling_byte(board: "chess.Board") -> int:
    """CA_WK=1, CA_WQ=2, CA_BK=4, CA_BQ=8 (board.h)."""
    ca = 0
    if board.has_kingside_castling_rights(chess.WHITE):  ca |= 1
    if board.has_queenside_castling_rights(chess.WHITE): ca |= 2
    if board.has_kingside_castling_rights(chess.BLACK):  ca |= 4
    if board.has_queenside_castling_rights(chess.BLACK): ca |= 8
    return ca

def _ep_file_byte(board: "chess.Board") -> int:
    """0..7 file index, 8 = no en-passant target (sample.h)."""
    if board.ep_square is None: return 8
    return chess.square_file(board.ep_square)

def _pack_move16(move: "chess.Move", board: "chess.Board") -> int:
    """Low-16-bits move pack matching sample.h's sample_pack_move(): from[0:5]
    to[6:11] prom[12:14] epc[15]. `board` must still be PRE-move."""
    from_zsq = move.from_square ^ 56
    to_zsq   = move.to_square ^ 56
    prom     = move.promotion or 0
    epc      = 1 if board.is_en_passant(move) else 0
    return (from_zsq & 63) | ((to_zsq & 63) << 6) | ((prom & 7) << 12) | (epc << 15)

def build_bin_records(positions, winner, label):
    """See run_tournament.py's identical function for the full contract."""
    rows = [p for p in positions if p.get("evaluator") == label and p.get("move") is not None]
    if not rows:
        return b""
    rec = np.zeros(len(rows), dtype=SAMPLE_DTYPE)
    for i, pos in enumerate(rows):
        b = chess.Board(pos["fen"])
        is_white_stm = (b.turn == chess.WHITE)
        raw_score = pos["score_white"]
        if raw_score is None:
            eval_cp = 0
        else:
            stm_score = raw_score if is_white_stm else -raw_score
            if abs(stm_score) >= MATE_SCORE // 2:
                stm_score = MATE_SCORE if stm_score > 0 else -MATE_SCORE
            eval_cp = max(-32000, min(32000, stm_score))
        if winner is None:
            game_result = 0
        else:
            white_won = (winner == "white")
            game_result = 1 if (is_white_stm == white_won) else -1

        rec[i]["board"]       = np.frombuffer(_board_to_zchezz_mailbox(b), dtype=np.uint8)
        rec[i]["stm"]         = 0 if is_white_stm else 1
        rec[i]["rule50"]      = b.halfmove_clock
        rec[i]["castling"]    = _castling_byte(b)
        rec[i]["ep_file"]     = _ep_file_byte(b)
        rec[i]["eval_cp"]     = eval_cp
        rec[i]["game_result"] = game_result
        rec[i]["move_played"] = _pack_move16(pos["move"], b)
        rec[i]["_pad"]        = 0
    return rec.tobytes()

def write_bin(f_bin, positions, winner, label):
    data = build_bin_records(positions, winner, label)
    if not data:
        return 0
    n = len(data) // SAMPLE_DTYPE.itemsize
    with _bin_lock:
        f_bin.write(data)
        f_bin.flush()
    return n


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN — Tournament orchestration
#  Loads openings, builds schedule, spawns worker threads, collects results,
#  writes PGN/EPD, prints periodic reports, and handles graceful shutdown.
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    global completed_games, epd_positions, bin_samples, start_time
    global _all_labels, _my_labels, stats, h2h, elo_estimators

    start_time = time.time()

    # Build label lists
    _my_labels  = [e["label"] for e in MY_ENGINES]
    _anc_labels = [a["label"] for a in ANCHORS]
    _all_labels = _my_labels + _anc_labels

    # Init stats & h2h
    for lb in _all_labels:
        stats[lb] = _new_stats()
    for l1 in _all_labels:
        for l2 in _all_labels:
            if l1 != l2: h2h[(l1, l2)] = {"w":0,"l":0,"d":0}

    # Anchor ELO map
    anchor_elo_map = {a["label"]: a["elo"] for a in ANCHORS}

    # Init ELO estimators for my engines
    for lb in _my_labels:
        elo_estimators[lb] = EloEstimator()

    # Load openings
    log("\n" + "="*60)
    log("  TOURNAMENT COMPLETE — Iniciando")
    log(f"  Meus Engines  : {', '.join(_my_labels)}")
    log(f"  Anchors       : {', '.join(_anc_labels)}")
    log(f"  Self-play     : {SELF_PLAY}")
    log(f"  Color swap    : {COLOR_SWAP}")
    log(f"  Concorrência  : {CONCURRENCY}")
    log("="*60)

    op_idx = load_all_openings(OPENING_FOLDER, OPENING_FILES or None)
    op_idx.shuffle()

    schedule = build_schedule(op_idx)
    total_games = len(schedule)

    # mini-ciclo: todos os pares jogam 1 abertura (x2 se COLOR_SWAP)
    # Independente de GAMES_VS_EACH_ANCHOR — representa 1 passagem por todos os pares
    swap = 2 if COLOR_SWAP else 1
    n_my  = len(MY_ENGINES)
    n_anc = len(ANCHORS)
    pairs_anc  = n_my * n_anc
    pairs_self = (n_my * (n_my - 1) // 2) if (SELF_PLAY and n_my > 1) else 0
    loop_size  = max(1, (pairs_anc + pairs_self) * swap)
    report_every = loop_size * max(1, REPORT_LOOPS)

    log(f"  Total de jogos : {total_games}")
    log(f"  Mini-ciclo     : {loop_size} jogos ({pairs_anc} pares-anchor + {pairs_self} pares-self x{swap} cores)")
    log(f"  Report a cada  : {report_every} jogos (REPORT_LOOPS={REPORT_LOOPS})")
    log(f"  CPUs em uso    : {CONCURRENCY}")
    log(f"  Ordem dos jogos: aleatória (schedule embaralhado)")
    log("")

    task_queue    = Queue()
    results_queue = Queue()
    for item in schedule:
        task_queue.put(item)

    def worker_loop():
        while True:
            try:
                item = task_queue.get_nowait()
            except Empty:
                break
            gid, w_cfg, b_cfg, op_idx_val = item
            op = op_idx.fetch(op_idx_val) if op_idx_val is not None else {"moves": [], "fen": None}
            play_game(gid, w_cfg, b_cfg, results_queue, op)
            task_queue.task_done()

    threads = [threading.Thread(target=worker_loop, daemon=True) for _ in range(CONCURRENCY)]
    for t in threads: t.start()

    f_pgn = open(ALL_PGNS, "w", encoding="utf-8") if SAVE_PGN else None
    f_epd = open(ALL_EPDS, "w", encoding="utf-8") if SAVE_EPD else None
    f_bin = open(ALL_BINS, "ab") if SAVE_BIN else None

    def _cleanup():
        if f_pgn: f_pgn.close()
        if f_epd: f_epd.close()
        if f_bin: f_bin.close()

    try:
        while completed_games < total_games:
            try:
                res = results_queue.get(timeout=1.0)
            except Empty:
                if all(not t.is_alive() for t in threads): break
                continue

            completed_games += 1
            w, b = res["white"], res["black"]

            with _stats_lock:
                for lb in (w, b):
                    if lb not in stats: stats[lb] = _new_stats()
                stats[w]["games"] += 1
                stats[b]["games"] += 1
                t_w = res["timing"].get(w, {"ms":0,"moves":0,"nodes":0,"depth":0})
                t_b = res["timing"].get(b, {"ms":0,"moves":0,"nodes":0,"depth":0})
                stats[w]["total_ms"]    += t_w["ms"]
                stats[w]["total_moves"] += t_w["moves"]
                stats[w]["total_nodes"] += t_w.get("nodes", 0)
                stats[w]["total_depth"] += t_w.get("depth", 0)
                stats[b]["total_ms"]    += t_b["ms"]
                stats[b]["total_moves"] += t_b["moves"]
                stats[b]["total_nodes"] += t_b.get("nodes", 0)
                stats[b]["total_depth"] += t_b.get("depth", 0)
                if res.get("timeout"): stats[w]["to"]+=1; stats[b]["to"]+=1
                if res.get("error"):   stats[w]["err"]+=1; stats[b]["err"]+=1

                winner = res.get("winner")
                if winner == "white":
                    stats[w]["w"]+=1; stats[w]["pts"]+=1; stats[b]["l"]+=1
                    if (w,b) in h2h: h2h[(w,b)]["w"]+=1
                    if (b,w) in h2h: h2h[(b,w)]["l"]+=1
                elif winner == "black":
                    stats[b]["w"]+=1; stats[b]["pts"]+=1; stats[w]["l"]+=1
                    if (w,b) in h2h: h2h[(w,b)]["l"]+=1
                    if (b,w) in h2h: h2h[(b,w)]["w"]+=1
                else:
                    stats[w]["d"]+=1; stats[w]["pts"]+=0.5
                    stats[b]["d"]+=1; stats[b]["pts"]+=0.5
                    if (w,b) in h2h: h2h[(w,b)]["d"]+=1
                    if (b,w) in h2h: h2h[(b,w)]["d"]+=1

                # Update ELO estimators (apenas contra anchors fixos)
                for my_lb in _my_labels:
                    if my_lb not in (w, b): continue
                    opp_lb = b if w == my_lb else w
                    if opp_lb not in anchor_elo_map: continue
                    
                    score = (1.0 if winner == ("white" if w==my_lb else "black")
                             else 0.0 if winner == ("black" if w==my_lb else "white")
                             else 0.5)
                    opp_elo = float(anchor_elo_map[opp_lb])
                    elo_estimators[my_lb].add(opp_elo, score)

            # Write PGN
            if f_pgn and res.get("pgn") and not res.get("error") and not res.get("timeout"):
                f_pgn.write(res["pgn"] + "\n\n")
                f_pgn.flush()

            # Write EPD
            if f_epd and res.get("positions"):
                n = write_epd(f_epd, res["positions"], winner)
                with _stats_lock:
                    epd_positions += n

            # Write .bin (only SAVE_BIN_LABEL's own moves — see SAVE_BIN's
            # config-block comment and build_bin_records() above).
            if f_bin and res.get("positions"):
                n = write_bin(f_bin, res["positions"], winner, SAVE_BIN_LABEL)
                with _stats_lock:
                    bin_samples += n

            # Per-game log
            # Log por jogo apenas no counter_every
            if completed_games % COUNTER_EVERY == 0:
                print_counter()

            # Full cross-table (aligned to complete loops)
            if completed_games % report_every == 0:
                print_cross_table(is_final=False)

    except KeyboardInterrupt:
        log("\n[!] Torneio interrompido pelo utilizador")
    finally:
        _cleanup()

    print_cross_table(is_final=True)


def _sigint(sig, frame):
    log("\n[!] SIGINT recebido")
    print_cross_table(is_final=True)
    sys.exit(0)

signal.signal(signal.SIGINT, _sigint)

if __name__ == "__main__":
    main()
