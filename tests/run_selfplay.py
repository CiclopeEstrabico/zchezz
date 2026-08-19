"""
run_selfplay.py — High-volume self-play generator (PERSISTENT engine processes)
================================================================================

Plays a large batch of games between the engine configs in ENGINES_CFG
(usually the SAME engine playing itself, hence "self-play") and writes every
position reached, tagged with the game's final result and the mover's own
eval, to a single EPD file — this is a TRAINING DATA GENERATOR first, a
tournament runner second. If you want a human-readable cross-table / ELO
comparison between two DIFFERENT engine versions, use run_tournament.py or
run_tournament_quick.py instead; those are built around that use case
(ELO reporting, anchors, MultiPV-style output) and only optionally emit EPD.
This script inverts that: EPD is unconditional (the `fe` file handle below is
always opened), PGN is the opt-in extra (SAVE_PGN).

CONFIG BLOCK FIRST, CLI SECOND — this is an edit-the-config-block-and-run
script: every knob has a documented default in the CONFIGURATION block below
and a bare `python tests/run_selfplay.py` runs exactly what those constants
say. Each one is ALSO a command-line flag that overrides it for scripted or
one-off runs (see the COMMAND LINE block right after the config, CLAUDE.md
rule 8, and utils/cliconf.py). `--show-config` prints the resolved settings
and exits without playing a game; `--help` lists every flag with the value
the config block gives it.

── RELATION TO OTHER TOOLS ─────────────────────────────────────────────────

  engine/c/tools/selfplay.c (native) — the FASTER path for the same job
  (per CLAUDE.md rule 9, "native and non-native paths are a blend, not a
  replacement"). Prefer the native tool for raw generation throughput once
  it covers your scenario; this Python harness stays because it is the only
  one of the two with a UCI-subprocess-based engine driver, which matters
  when you want to selfplay engines that AREN'T zchezz.exe builds (anything
  speaking UCI works here — the native tool only knows zchezz's in-process
  net:/uci: player kinds via arena.c's machinery).

  run_tournament.py / run_tournament_quick.py — H2H and anchor-ELO
  reporting between distinct engine builds; EPD export there is optional
  and secondary. This script's EPD export is the point.

  run_arena.py (native A/B gate) — SPRT-driven promote/discard between two
  VERSIONS. Not what this script is for at all; this script doesn't compute
  ELO or SPRT, it only tallies W/L/D and writes training positions.

── PERSISTENT ENGINE PROCESSES (the key design difference from
   run_tournament.py) ────────────────────────────────────────────────────

  run_tournament.py's play_game() spawns a FRESH EngineInstance (new OS
  process, full UCI handshake) for every single game. At self-play's game
  volumes (thousands to tens of thousands of games) that process-spawn
  overhead dominates wall-clock time. Here, each worker thread starts ONE
  EngineInstance per ENGINES_CFG entry when the worker thread starts, and
  reuses those same processes for every game the worker plays — only
  `ucinewgame`/`isready` is sent between games (EngineInstance.reset_state),
  never a process restart. This is the whole reason self-play can sustain
  CONCURRENCY=16 persistent engine pairs for 20000 games in one run.

── OPENING BOOK — GAMES IS A SLOT BUDGET, NOT A GAME COUNT ──────────

  GAMES is divided across engine pairs into per-pair "opening
  iterations" (raw_iterations = GAMES // num_pairs), and EACH
  iteration expands into 2 actual games because COLOR_SWAP is always on
  (every opening is played from both sides) — so the real game count
  (_real_total_games) is usually double what GAMES might suggest,
  before SAME_OPENING_TWICE is even considered. If book positions run out
  before raw_n_book iterations are satisfied, book iterations are silently
  capped to what OPENING_FOLDER actually has and the shortfall is made up
  with random-opening games (see the "AVISO" log line) — a run never fails
  just because a book is smaller than requested, it degrades toward
  OPENING_MODE="random" for the overflow instead.

  Same OpeningIndex design as run_tournament.py: only byte offsets are
  scanned/stored per file (not full positions), EPD offsets are cached to a
  sibling `.idx` file keyed by mtime, and openings/ is walked recursively
  (lines/*.pgn move sequences, positions/*.epd start positions) — a flat
  listdir() would silently see 0 openings after the folder was split into
  subfolders.

── OUTPUT ──────────────────────────────────────────────────────────────────

  EPD  (always): `tests/selfplay_results/selfplay_<timestamp>.epd` — one
  line per position reached (opening positions included when
  SAVE_OPENING_IN_EPD), tagged `c0 "<pgn result>"; c1 "<white-relative cp,
  clamped to ±MATE_SCORE>"; c2 "<mover label>"; c3 "<ply index>";` — the
  same convention run_tournament.py's write_epd() uses, so the two tools'
  EPD output is interchangeable as training data.
  PGN  (opt-in, SAVE_PGN): same directory, `.pgn` extension.
  LOG  (always): full console transcript, same directory, `.log` extension.

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

import os, sys, json, subprocess, time, math, random, threading, datetime, tempfile, signal, io
import collections
from queue import Queue, Empty
import chess
import chess.pgn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elo_calc import elo_difference as _elo_difference

# train/ is imported ONLY for SAMPLE_DTYPE (the packed .bin record layout) so
# this script's .bin output is byte-compatible with engine/c/tools/selfplay.c's
# — see CLAUDE.md's "TRAINING-DATA NAMING CONVENTION" and sample.h's own
# comment on why two independently-written record layouts must never exist.
# dataset.py only imports numpy/os/glob at module scope (no torch), so this
# stays a cheap import even when SAVE_BIN is off.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))
import numpy as np
from dataset import SAMPLE_DTYPE

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit this block and run; every constant is also a CLI flag
#  (see the COMMAND LINE block after the config, and CLAUDE.md rule 8)
# ═══════════════════════════════════════════════════════════════════════════════

def kill_leftover_engines():
    """Kill any leftover zchezz*.exe processes system-wide before starting.

    A previous run that was Ctrl-C'd or crashed can leave orphaned engine
    processes behind (still holding stdin/stdout pipes); starting the next
    run without this cleanup risks a stale process answering for a worker
    slot this run thinks it owns.

    Called from main(), NOT at import time: `--help` and `--show-config`
    must not kill a colleague's running engines just to print text.
    """
    cmd = (
        'Get-CimInstance Win32_Process -Filter "Name = \'zchezz*\'" | '
        'Invoke-CimMethod -MethodName Terminate'
    )
    subprocess.run(["powershell", "-Command", cmd], capture_output=True)
    print("[OK] Processos Anteriores encerrados.")

# ── Engines ("players") — one persistent process per entry, per worker thread ──
# Two identical entries (as below) = true self-play (one engine playing
# itself). Different paths/labels = an N-way round robin between distinct
# UCI engines, still with EPD export — this is NOT limited to 2 engines.
ENGINES_CFG = [
    {
        "path":     r"engine\c\zchezz_v401\zchezz.exe",
        "label":    "Zchezz-v401",
        "tc_mode":  "movetime",   # "movetime" | "depth" | "nodes" | "fixedtime"
        "tc_value": 200,          # units depend on tc_mode (ms for movetime)
        "tc_inc":   0             # increment in ms, "fixedtime" mode only
    },
    {
        "path":     r"engine\c\zchezz_v401\zchezz.exe",
        "label":    "Zchezz-v401",
        "tc_mode":  "movetime",
        "tc_value": 200,
        "tc_inc":   0
    },
]

# ── Match Volume & Safety ────────────────────────────────────────────────────
GAMES  = 20000  # opening-slot budget, NOT the final game count — see
                       # "OPENING BOOK" note in the header docstring above
CONCURRENCY  = 14      # parallel worker threads (= persistent engine-pair count)
MAX_PLIES    = 400     # half-moves before a game is forced to a draw
MOVE_TIMEOUT_S = 35.0    # seconds to wait for one "bestmove" before declaring TIMEOUT
REPORT_EVERY = 10      # print a status block every N completed games

# ── Output ────────────────────────────────────────────────────────────────────
# EPD is always saved (see header docstring — this script IS the EPD export).
# .bin and .pgn are opt-in extras, same status: additive, not either/or
# (CLAUDE.md rule 9) — any combination of SAVE_BIN/SAVE_PGN can be on at once.
SAVE_PGN            = True  # save every game as standard PGN alongside the EPD
SAVE_EPD            = True   # save positions + eval as EPD
SAVE_BIN            = True  # save every game as a packed .bin (train/dataset.py's
                              # SAMPLE_DTYPE — same schema engine/c/tools/selfplay.c
                              # writes, so .bin files from both tools are interchangeable)
SAVE_OPENING_IN_EPD = True   # include the forced book/random opening plies in the EPD output
SAVE_OPENING_IN_BIN = True  # include forced opening plies in the .bin output (default OFF,
                              # matches selfplay.c's SP_DEFAULT_SAVE_OPENING_SAMPLES=0: those
                              # plies were not chosen by search, so eval_cp is meaningless for them)
RESULTS_DIR         = r"tests\selfplay_results"  # output directory for .log/.pgn/.epd/.bin — all
                                                   # four share this dir and only differ by extension

# ── Opening Book ──────────────────────────────────────────────────────────────
OPENING_MODE        = "book"  # how each game's starting position is chosen:
                              #   "book"        every game starts from an opening in OPENING_FOLDER
                              #   "random"      every game starts after RANDOM_PLIES random legal plies
                              #   "book+random" BOOK_PORTION of games from the book, the rest random
                              # ("all" is accepted as the old name for "book+random")
BOOK_PORTION        = 0.97    # "book+random" only: fraction of iterations drawn from the book
OPENING_FOLDER      = r"openings\lines"  # walked recursively for .pgn/.epd files, see header docstring
RANDOM_PLIES        = 6       # plies of random legal moves, for "random" and the random half of "book+random"
SAME_OPENING_TWICE  = True    # True: color-swap pair reuses the SAME opening; False: advances to the next one
COLOR_SWAP          = True    # every opening is always played from both sides (not actually optional — see main())

# ── Engine overrides (change the players without editing ENGINES_CFG) ────────
# ENGINES_CFG above stays the authoritative description of the field; these
# let a scripted run point the same setup at another build / time control.
ENGINE_PATH   = ""    # "" = keep each ENGINES_CFG entry's own path; else set ALL of them
ENGINE_LABEL  = ""    # "" = keep each entry's own label
MOVETIME_MS   = 0     # 0 = keep each entry's tc_mode/tc_value; >0 = force movetime on all

# ═══════════════════════════════════════════════════════════════════════════════
#  COMMAND LINE — every constant above is also a flag (CLAUDE.md rule 8)
#
#  The config block IS the interface: a bare `python tests/run_selfplay.py`
#  runs exactly what the constants say. The flags only OVERRIDE them, and
#  `--show-config` prints the resolved values and exits without playing a
#  single game. See utils/cliconf.py.
#
#  This must run BEFORE the timestamped output paths below are built, so a
#  --results-dir override actually reaches them.
# ═══════════════════════════════════════════════════════════════════════════════
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils"))
from cliconf import override_from_cli  # noqa: E402

CLI = [
    ("GAMES",         "--games",           int,   "opening-slot budget (see header: not the final game count)"),
    ("CONCURRENCY",         "--concurrency",     int,   "parallel worker threads (persistent engine pairs)"),
    ("MAX_PLIES",           "--max-plies",       int,   "half-move cap before a forced draw"),
    ("MOVE_TIMEOUT_S",        "--move-timeout",    float, "seconds to wait for one bestmove"),
    ("REPORT_EVERY",        "--report-every",    int,   "status block every N completed games"),
    ("SAVE_PGN",            "--pgn",             bool,  "save every game as standard PGN"),
    ("SAVE_EPD",            "--epd",             bool,  "save positions + eval as EPD"),
    ("SAVE_BIN",            "--bin",             bool,  "save every game as packed .bin samples"),
    ("SAVE_OPENING_IN_EPD", "--opening-in-epd",  bool,  "include forced opening plies in the EPD"),
    ("SAVE_OPENING_IN_BIN", "--opening-in-bin",  bool,  "include forced opening plies in the .bin"),
    ("RESULTS_DIR",         "--results-dir",     str,   "output directory for .log/.pgn/.epd/.bin"),
    ("OPENING_MODE",        "--opening-mode",    str,   "book | random | book+random",
     ("book", "random", "book+random", "all")),
    ("BOOK_PORTION",        "--book-portion",    float, "'book+random' mode: fraction drawn from the book"),
    ("OPENING_FOLDER",      "--openings",        str,   "folder walked recursively for .pgn/.epd openings"),
    ("RANDOM_PLIES",        "--random-plies",    int,   "ply count for the random opening mode"),
    ("SAME_OPENING_TWICE",  "--same-opening-twice", bool, "color-swap pair reuses the same opening"),
    ("ENGINE_PATH",         "--engine",          str,   "override the path of EVERY ENGINES_CFG entry"),
    ("ENGINE_LABEL",        "--engine-label",    str,   "override the label of EVERY ENGINES_CFG entry"),
    ("MOVETIME_MS",         "--movetime",        int,   "force movetime (ms) on every engine; 0 = keep configured TC"),
]

if __name__ == "__main__":
    override_from_cli(globals(), CLI, description=__doc__, prog="run_selfplay.py")
    for _eng in ENGINES_CFG:
        if ENGINE_PATH:  _eng["path"]  = ENGINE_PATH
        if ENGINE_LABEL: _eng["label"] = ENGINE_LABEL
        if MOVETIME_MS:
            _eng["tc_mode"]  = "movetime"
            _eng["tc_value"] = MOVETIME_MS

# ═══════════════════════════════════════════════════════════════════════════════
#  SAÍDA E LOGS
# ═══════════════════════════════════════════════════════════════════════════════
os.makedirs(RESULTS_DIR, exist_ok=True)
TIMESTAMP  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE   = os.path.join(RESULTS_DIR, f"selfplay_{TIMESTAMP}.log")
PGN_FILE   = os.path.join(RESULTS_DIR, f"selfplay_{TIMESTAMP}.pgn")
EPD_FILE   = os.path.join(RESULTS_DIR, f"selfplay_{TIMESTAMP}.epd")
BIN_FILE   = os.path.join(RESULTS_DIR, f"selfplay_{TIMESTAMP}.bin")

MATE_SCORE = 99999999   # clamp value for mate scores in EPD/.bin output (both formats clamp to this)

_log_lock = threading.Lock()
def log(*args):
    msg = " ".join(map(str, args))
    with _log_lock:
        print(msg, flush=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE INSTANCE (PERSISTENTE)
# ═══════════════════════════════════════════════════════════════════════════════

class EngineInstance:
    def __init__(self, path: str, label: str, tc_mode="depth", tc_value=7, tc_inc=0):
        self.path      = os.path.abspath(path)
        self.label     = label
        self.tc_mode   = tc_mode
        self.tc_value  = tc_value
        self.tc_inc    = tc_inc
        self.process   = None
        self.queue     = Queue()
        self.suppress_nnue = False
        self.start_fen = None  # Set per-game; None means standard startpos

    def _reader(self):
        try:
            for line in iter(self.process.stdout.readline, ''):
                line = line.strip()
                if line: self.queue.put(line)
        except Exception: pass

    def _stderr_reader(self):
        try:
            for line in iter(self.process.stderr.readline, ''):
                line = line.strip()
                if not line: continue
                if self.suppress_nnue and any(x in line.lower() for x in ["nnue", "loading"]):
                    continue
                if any(x in line.lower() for x in ["error", "exception", "nnue", "loading"]):
                    log(f"  [Native/{self.label}] {line}")
        except Exception: pass

    def start(self):
        if self.process and self.process.poll() is None:
            return True # Já rodando

        engine_dir = os.path.dirname(self.path)
        try:
            args = [self.path]
            if os.path.exists(os.path.join(engine_dir, "nnue_weights.bin")):
                args += ["--nnue", "nnue_weights.bin"]
            
            # Global NNUE suppression logic
            global _nnue_logged
            with _nnue_logged_lock:
                if self.path in _nnue_logged: self.suppress_nnue = True
                else: _nnue_logged.add(self.path); self.suppress_nnue = False

            self.process = subprocess.Popen(
                args, cwd=engine_dir,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            threading.Thread(target=self._reader,        daemon=True).start()
            threading.Thread(target=self._stderr_reader, daemon=True).start()

            self._send("uci")
            t0 = time.time()
            while time.time() - t0 < 5:
                try:
                    if "uciok" in self.queue.get(timeout=0.1): break
                except Empty:
                    if self.process.poll() is not None: return False
            
            self._send("isready")
            t0 = time.time()
            while time.time() - t0 < 5:
                try:
                    if "readyok" in self.queue.get(timeout=0.1): break
                except Empty: pass
            
            self._send("ucinewgame")
            self._send("isready")
            return self.process.poll() is None
        except Exception as e:
            log(f"  [ERRO] Falha ao iniciar {self.label}: {e}")
            return False

    def reset_state(self):
        """Prepara o motor para uma nova partida sem reiniciar processo."""
        self._send("ucinewgame")
        self._send("isready")
        t0 = time.time()
        while time.time() - t0 < 2:
            try:
                if "readyok" in self.queue.get(timeout=0.1): break
            except Empty: pass

    def _send(self, cmd):
        if self.process and self.process.stdin:
            try:
                self.process.stdin.write(cmd + "\n")
                self.process.stdin.flush()
                return True
            except: pass
        return False

    def get_move(self, board, wtime=0, btime=0):
        if not self.process or self.process.poll() is not None:
            if not self.start(): return None, 0

        # Limpa queue
        while not self.queue.empty():
            try: self.queue.get_nowait()
            except Empty: break

        m_list = " ".join([m.uci() for m in board.move_stack])
        if self.start_fen:
            pos_cmd = f"position fen {self.start_fen}" + (f" moves {m_list}" if m_list else "")
        else:
            pos_cmd = f"position startpos" + (f" moves {m_list}" if m_list else "")
        self._send(pos_cmd)

        if self.tc_mode == "depth":
            tc_cmd = f"go depth {self.tc_value}"
        elif self.tc_mode == "movetime":
            tc_cmd = f"go movetime {self.tc_value}"
        elif self.tc_mode == "nodes":
            tc_cmd = f"go nodes {self.tc_value}"
        else:
            tc_cmd = f"go wtime {wtime} btime {btime} winc {self.tc_inc} binc {self.tc_inc}"
        self._send(tc_cmd)
        
        move, score, nodes, depth = None, None, 0, 0
        t_start = time.time()
        while time.time() - t_start < MOVE_TIMEOUT_S:
            try:
                line = self.queue.get(timeout=0.1)
                if line.startswith("info "):
                    parts = line.split()
                    if "nodes" in parts:
                        try: nodes = int(parts[parts.index("nodes")+1])
                        except: pass
                    if "depth" in parts:
                        try: depth = int(parts[parts.index("depth")+1])
                        except: pass
                if "score cp" in line:
                    parts = line.split()
                    try: idx = parts.index("cp"); score = int(parts[idx+1])
                    except: pass
                elif "score mate" in line:
                    parts = line.split()
                    try:
                        m_val = int(parts[parts.index("mate")+1])
                        if board.turn == chess.WHITE:
                            score = MATE_SCORE if m_val > 0 else -MATE_SCORE
                        else:
                            score = MATE_SCORE if m_val < 0 else -MATE_SCORE
                    except: pass
                if line.startswith("bestmove"):
                    parts = line.split()
                    if len(parts) > 1: move = parts[1]
                    break
            except Empty:
                if self.process.poll() is not None: break
                continue
        
        if not move: return "TIMEOUT" if self.process.poll() is None else None, 0, 0, 0
        
        # Convert eval score to white's perspective
        if board.turn == chess.BLACK and score is not None:
            score = -score
            
        return move, score, nodes, depth

    def stop(self):
        if self.process:
            try:
                self._send("quit")
                self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=1.0)
            except: pass
            self.process = None

# Estatísticas globais
completed_games   = 0
_real_total_games = 0
start_time        = 0.0   # set in main() after opening indexing
_stats_lock       = threading.Lock()
_file_lock        = threading.Lock()
_nnue_logged      = set()
_nnue_logged_lock = threading.Lock()
_cfg_labels = [c["label"] for c in ENGINES_CFG]
stats = {n: {"w":0,"l":0,"d":0,"pts":0.0,"to":0,"err":0,"games":0,"total_ms":0,"total_moves":0,"total_nodes":0,"total_depth":0} for n in _cfg_labels}
# defaultdict, not a comprehension over (n1,n2) pairs with n1!=n2: the
# default config plays one label against itself (true self-play), and an
# n1==n2 key must not KeyError the very first completed game.
h2h = collections.defaultdict(lambda: {"w":0,"l":0,"d":0})

# ═══════════════════════════════════════════════════════════════════════════════
# AUXILIARES (UI e Elo)
# ═══════════════════════════════════════════════════════════════════════════════

class OpeningIndex:
    """
    Memory-efficient opening index.
    Scans files once at startup recording only byte offsets — no positions
    are kept in memory. A single seek() + readline() fetches any entry
    on demand, so 100 MB+ EPD files are handled without RAM issues.
    """
    def __init__(self):
        # Each entry: (filepath, byte_offset, kind)  kind = 'epd' | 'pgn'
        self._index = []

    def __len__(self):
        return len(self._index)

    def __bool__(self):
        return bool(self._index)

    def _build_epd(self, fpath):
        idx_path = fpath + ".idx"
        import struct
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
                offset = f.tell()
                raw = f.readline()
                if not raw: break
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line or line.startswith("#"): continue
                if len(line.split()) >= 4:
                    offsets.append(offset)
        try:
            import struct
            with open(idx_path, "wb") as f:
                f.write(struct.pack(f"<{len(offsets)}Q", *offsets))
        except Exception: pass
        for off in offsets: self._index.append((fpath, off, "epd"))
        log(f"  [EPD] {os.path.basename(fpath)}: {len(offsets):,} pos indexed")

    def _build_pgn(self, fpath):
        with open(fpath, "rb") as f:
            game_start = None
            while True:
                offset = f.tell()
                raw = f.readline()
                if not raw:
                    if game_start is not None:
                        self._index.append((fpath, game_start, "pgn"))
                    break
                line = raw.decode("utf-8", errors="ignore")
                if line.startswith("[Event "):
                    game_start = offset
                elif line.strip() == "" and game_start is not None:
                    self._index.append((fpath, game_start, "pgn"))
                    game_start = None

    def shuffle(self):
        random.shuffle(self._index)

    def fetch(self, idx):
        """Fetch one opening by index. Returns {"moves": [...], "fen": None|str}."""
        fpath, offset, kind = self._index[idx]
        if kind == "epd":
            with open(fpath, "rb") as f:
                f.seek(offset)
                raw = f.readline().decode("utf-8", errors="ignore").strip()
            parts = raw.split()
            fen = " ".join(parts[:4]) + " 0 1"
            try:
                board = chess.Board(fen)
                return {"moves": [], "fen": board.fen()}
            except Exception:
                return {"moves": [], "fen": None}
        else:  # pgn
            with open(fpath, "rb") as f:
                f.seek(offset)
                block = []
                blank_count = 0
                while True:
                    raw = f.readline()
                    if not raw: break
                    line = raw.decode("utf-8", errors="ignore")
                    block.append(line)
                    if line.strip() == "":
                        blank_count += 1
                        if blank_count >= 2: break
                    else:
                        blank_count = 0
            buf = io.StringIO("".join(block))
            game = chess.pgn.read_game(buf)
            if game is None: return {"moves": [], "fen": None}
            uci_seq, board = [], game.board()
            for move in game.mainline_moves():
                if move in board.legal_moves: uci_seq.append(move.uci()); board.push(move)
                else: break
            return {"moves": uci_seq, "fen": None}

    def random_pick(self):
        return self.fetch(random.randrange(len(self._index)))


def load_all_openings(folder_path):
    """
    Build an OpeningIndex from all .pgn and .epd files in folder_path.
    Only byte offsets are stored — no positions loaded into RAM.
    """
    idx = OpeningIndex()
    if not os.path.exists(folder_path): return idx
    # Walk recursively: openings/ is split into lines/ (.pgn move sequences
    # played from the start position) and positions/ (.epd start positions).
    # A flat listdir() would silently find nothing and report 0 openings.
    for root, _dirs, files in os.walk(folder_path):
        for f_name in sorted(files):
            fpath = os.path.join(root, f_name)
            if f_name.endswith(".epd"):   idx._build_epd(fpath)
            elif f_name.endswith(".pgn"): idx._build_pgn(fpath)
    log(f"[Openings] Indexed {len(idx)} positions from '{folder_path}' (offsets only, no RAM load)")
    return idx

def random_opening(n_plies: int, start_board=None) -> dict:
    board = start_board.copy() if start_board else chess.Board()
    moves = []
    for _ in range(n_plies):
        legal = list(board.legal_moves)
        if not legal or board.is_game_over(): break
        move = random.choice(legal)
        moves.append(move.uci()); board.push(move)
    return {"moves": moves, "fen": None}

def elo_diff(wins, losses, draws):
    N = wins + losses + draws
    if N == 0: return "N/A"
    elo, ci, _ = _elo_difference(wins, draws, losses)
    return f"{'+' if elo>=0 else ''}{elo:.1f} ±{ci:.0f}"

def print_status(is_final: bool = False):
    global completed_games, _real_total_games, start_time, stats, h2h
    elapsed = time.time() - start_time
    gps_val = completed_games / elapsed if elapsed > 1 else 0
    gps     = f"{gps_val:.2f}" if elapsed > 1 else "—"

    if not is_final and gps_val > 0 and completed_games < _real_total_games:
        remaining = (_real_total_games - completed_games) / gps_val
        h, m, s   = int(remaining // 3600), int((remaining % 3600) // 60), int(remaining % 60)
        eta_str   = f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"
        eta_line  = f"ETA    : ~{eta_str} (a {datetime.datetime.now() + datetime.timedelta(seconds=remaining):%H:%M:%S})\n"
    else:
        eta_line  = ""

    title = "RESULTADO FINAL" if is_final else "STATUS"
    log(f"\n{'='*52}\n=== {title}\nJogos  : {completed_games}/{_real_total_games} | Vel: {gps} j/s\nTempo  : {elapsed:.1f}s\n{eta_line}{'-'*52}")
    for r, (name, s) in enumerate(sorted(stats.items(), key=lambda x: x[1]["pts"], reverse=True), 1):
        g = s["games"]
        mean_ms = int(s["total_ms"]/s["total_moves"]) if s["total_moves"] else 0
        pct = lambda n: f"{n/g*100:.1f}%" if g else "0.0%"
        alerts = f"  [!] T:{s['to']} Err:{s['err']}" if s["to"] or s["err"] else ""
        log(f"{r}. {name} | Pts:{s['pts']}/{g} ({pct(s['pts'])})\n   V:{s['w']}({pct(s['w'])}) E:{s['d']}({pct(s['d'])}) D:{s['l']}({pct(s['l'])}) T.med:{mean_ms}ms{alerts}")
    log(f"{'='*52}")

def format_eval_comment(score, ms):
    if score is None: return f"{{ {ms}ms }}"
    if abs(score) > 89000:
        mate = math.ceil((900000 - abs(score)) / 2) + 1
        return f"{{ #{'' if score>=0 else '-'}{mate} | {ms}ms }}"
    return f"{{ {'+' if score>=0 else ''}{score/100:.2f} | {ms}ms }}"

# ═══════════════════════════════════════════════════════════════════════════════
# .bin ENCODING — python-chess Board -> train/dataset.py's SAMPLE_DTYPE record
# ═══════════════════════════════════════════════════════════════════════════════
# Mirrors Zchezz's own mailbox/castling/move-pack conventions exactly (see
# CLAUDE.md "Piece encoding" and engine/c/tools/sample.h's field comments) so
# the records written here are byte-identical in meaning to selfplay.exe's.

def board_to_zchezz_mailbox(board: "chess.Board") -> bytes:
    """python-chess Board -> Zchezz mailbox bytes[64] (0=empty, WP=9..BK=22,
    zsq 0=a8..63=h1). zsq = python-chess-square ^ 56 (CLAUDE.md coordinate
    invariant: sq_w = zsq ^ 56, which is self-inverse)."""
    arr = bytearray(64)
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece is None:
            continue
        zsq = sq ^ 56
        color_bit = 8 if piece.color == chess.WHITE else 16  # COL_W=8, COL_B=16
        arr[zsq] = color_bit + piece.piece_type  # piece_type 1..6 == Zchezz's PC_TYPE (P..K)
    return bytes(arr)

def castling_byte(board: "chess.Board") -> int:
    """Bitmask matching board.h: CA_WK=1, CA_WQ=2, CA_BK=4, CA_BQ=8."""
    ca = 0
    if board.has_kingside_castling_rights(chess.WHITE):  ca |= 1
    if board.has_queenside_castling_rights(chess.WHITE): ca |= 2
    if board.has_kingside_castling_rights(chess.BLACK):  ca |= 4
    if board.has_queenside_castling_rights(chess.BLACK): ca |= 8
    return ca

def ep_file_byte(board: "chess.Board") -> int:
    """0..7 file index, 8 = no en-passant target (matches sample.h)."""
    if board.ep_square is None: return 8
    return chess.square_file(board.ep_square)

def pack_move16(move: "chess.Move", board: "chess.Board") -> int:
    """Low-16-bits move pack, matching sample.h's sample_pack_move() /
    selfplay.c's pack_move16(): from[0:5] to[6:11] prom[12:14] epc[15].
    `board` must still be in the PRE-move state (needed for is_en_passant)."""
    from_zsq = move.from_square ^ 56
    to_zsq   = move.to_square ^ 56
    prom     = move.promotion or 0   # python-chess promotion 2..5 == Zchezz PC_TYPE N..Q
    epc      = 1 if board.is_en_passant(move) else 0
    return (from_zsq & 63) | ((to_zsq & 63) << 6) | ((prom & 7) << 12) | (epc << 15)

def build_bin_records(states, winner) -> bytes:
    """states: list of {"fen": <pre-move FEN>, "score": <white-relative cp|None>,
    "move": <chess.Move>} in play order. winner: 'white'|'black'|None (draw).
    Returns packed bytes in SAMPLE_DTYPE layout — same schema selfplay.exe
    writes (see train/dataset.py's SAMPLE_DTYPE / engine/c/tools/sample.h)."""
    if not states: return b""
    rec = np.zeros(len(states), dtype=SAMPLE_DTYPE)
    for i, st in enumerate(states):
        b = chess.Board(st["fen"])
        is_white_stm = (b.turn == chess.WHITE)
        raw_score = st["score"]
        # SAMPLE_DTYPE's eval_cp is STM-relative; this script stores
        # White-relative scores (see get_move()'s own conversion) — flip back.
        if raw_score is None:
            eval_cp = 0   # forced opening plies (not chosen by search) — matches selfplay.c default
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

        rec[i]["board"]       = np.frombuffer(board_to_zchezz_mailbox(b), dtype=np.uint8)
        rec[i]["stm"]         = 0 if is_white_stm else 1
        rec[i]["rule50"]      = b.halfmove_clock
        rec[i]["castling"]    = castling_byte(b)
        rec[i]["ep_file"]     = ep_file_byte(b)
        rec[i]["eval_cp"]     = eval_cp
        rec[i]["game_result"] = game_result
        rec[i]["move_played"] = pack_move16(st["move"], b) if st.get("move") is not None else 0
        rec[i]["_pad"]        = 0
    return rec.tobytes()

# ═══════════════════════════════════════════════════════════════════════════════
# GAME LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def play_game(game_id, w_engine, b_engine, opening, results_queue):
    start_t = time.time()
    # Reset state instead of starting process
    w_engine.reset_state(); b_engine.reset_state()

    # Opening is a dict: {"moves": [...], "fen": None|"<fen>"}
    opening_fen_base = opening.get("fen")  # None for PGN openings
    forced_moves     = opening.get("moves", [])

    # Set start_fen on engines so get_move sends the right position command
    w_engine.start_fen = opening_fen_base
    b_engine.start_fen = opening_fen_base

    board = chess.Board(opening_fen_base) if opening_fen_base else chess.Board()
    pgn_game = chess.pgn.Game()
    if opening_fen_base:
        pgn_game.headers["FEN"]      = opening_fen_base
        pgn_game.headers["SetUp"]    = "1"
    pgn_game.headers.update({"Event": "Selfplay EXE", "Round": str(game_id), "White": w_engine.label, "Black": b_engine.label})

    node, ply, applied = pgn_game, 0, []
    # Capture opening positions unconditionally (cheap) — EPD/.bin each decide
    # independently at write time whether to include them (SAVE_OPENING_IN_EPD
    # vs SAVE_OPENING_IN_BIN; the two formats have different defaults, see
    # CONFIGURATION block).
    opening_history_states = []
    for uci in forced_moves:
        try:
            move = chess.Move.from_uci(uci)
            if move in board.legal_moves:
                pre_fen = board.fen()
                board.push(move); node = node.add_main_variation(move); applied.append(uci); ply += 1
                opening_history_states.append({"fen": pre_fen, "score": None, "evaluator": "book", "move": move})
            else: break
        except: break
    
    opening_fen = board.fen()
    winner, reason, to, err = None, "Normal", False, False
    timing = {w_engine.label: {"ms":0,"moves":0,"nodes":0,"depth":0}, b_engine.label: {"ms":0,"moves":0,"nodes":0,"depth":0}}
    
    wtime = w_engine.tc_value if w_engine.tc_mode == "fixedtime" else 0
    btime = b_engine.tc_value if b_engine.tc_mode == "fixedtime" else 0

    # Store each state to save as EPD later
    history_states = []

    try:
        while not board.is_game_over() and ply < MAX_PLIES:
            is_w = board.turn == chess.WHITE
            active = w_engine if is_w else b_engine
            t0 = time.time()
            
            # Captura o fen antes do motor realizar a jogada
            current_fen = board.fen()
            
            move_str, score, move_nodes, move_depth = active.get_move(board, wtime, btime)
            ms = int((time.time() - t0) * 1000)
            timing[active.label]["ms"]    += ms
            timing[active.label]["moves"] += 1
            timing[active.label]["nodes"] += move_nodes
            timing[active.label]["depth"] += move_depth

            if active.tc_mode == "fixedtime":
                if is_w: wtime = max(0, wtime - ms + w_engine.tc_inc)
                else:    btime = max(0, btime - ms + b_engine.tc_inc)

            if move_str == "TIMEOUT": to = True; reason = f"Timeout ({active.label})"; winner = "black" if is_w else "white"; break
            if not move_str: err = True; reason = f"Erro ({active.label})"; winner = "black" if is_w else "white"; break

            try:
                move = board.parse_uci(move_str)
                # Reject null moves (0000 / (none)) and moves outside the legal
                # set.  Pushing a null move corrupts the repetition counter and
                # can trigger a false draw before real 3-fold has occurred.
                if move == chess.Move.null() or move not in board.legal_moves:
                    raise ValueError(f"Null/illegal: {move_str}")
                board.push(move)
                node = node.add_main_variation(move, comment=format_eval_comment(score, ms))
                ply += 1
                history_states.append({"fen": current_fen, "score": score, "evaluator": active.label, "move": move})
            except:
                err = True
                _is_null = move_str in ("0000", "(none)")
                reason = (
                    f"Vitoria por Lance Nulo ({active.label} jogou {move_str})"
                    if _is_null else
                    f"Lance Invalido ({move_str}) [{active.label}]"
                )
                winner = "black" if is_w else "white"
                # Annotate the last PGN node so the infraction is visible on replay
                node.comment = (
                    f"[RESULTADO: {active.label} jogou lance nulo/ilegal '{move_str}' — "
                    f"vitoria para {'Pretas' if is_w else 'Brancas'}]"
                )
                break

        if not to and not err:
            res = board.result()
            if res == "1-0": winner = "white"; reason = "Checkmate" if board.is_checkmate() else "Vitoria"
            elif res == "0-1": winner = "black"; reason = "Checkmate" if board.is_checkmate() else "Vitoria"
            else: winner = None; reason = "Empate"

        # Set PGN Result and Termination headers
        if   winner == "white": pgn_result = "1-0"
        elif winner == "black": pgn_result = "0-1"
        else:                   pgn_result = "1/2-1/2"
        pgn_game.headers["Result"]      = pgn_result
        pgn_game.headers["Termination"] = reason

        # Normalise scores: None→0, mate→±MATE_SCORE (matches run_tournament.py)
        def normalise_score(sc):
            if sc is None: return 0
            if abs(sc) >= MATE_SCORE // 2:
                return MATE_SCORE if sc > 0 else -MATE_SCORE
            return sc

        # Combine opening positions (if the format's own opening flag says so) + game positions
        epd_states = (opening_history_states if SAVE_OPENING_IN_EPD else []) + history_states
        bin_states = (opening_history_states if SAVE_OPENING_IN_BIN else []) + history_states

        epd_lines = []
        final_res = pgn_result
        for idx, st in enumerate(epd_states):
            norm_score = normalise_score(st['score'])
            epd_lines.append(f"{st['fen']} c0 \"{final_res}\"; c1 \"{norm_score}\"; c2 \"{st['evaluator']}\"; c3 \"{idx}\";")

        # Empty string when no positions — main loop guards against writing blank lines
        final_epd_str = "\n".join(epd_lines) if SAVE_EPD else None

        bin_bytes = build_bin_records(bin_states, winner) if SAVE_BIN else b""

        results_queue.put({
            "id": game_id, "white": w_engine.label, "black": b_engine.label, "winner": winner, "reason": reason,
            "pgn": str(pgn_game) if SAVE_PGN else None,
            "epd": final_epd_str,
            "bin": bin_bytes,
            "timeout": to, "error": err, "timing": timing, "duration": time.time() - start_t
        })
    except Exception as e:
        log(f"  [ERRO JOGO {game_id}] Exception: {e}")

def main():
    global completed_games, _real_total_games, start_time, _stats_lock, _file_lock, stats, h2h
    kill_leftover_engines()
    all_openings = load_all_openings(OPENING_FOLDER) if OPENING_MODE != "random" else OpeningIndex()

    # Shuffle the index once so book picks are non-repeating across the run
    all_openings.shuffle()

    # ── Opening count summary ──────────────────────────────────────────────
    n_book_positions = len(all_openings)
    log(f"[Openings] {n_book_positions:,} linhas de abertura indexadas de '{OPENING_FOLDER}'")

    N = len(ENGINES_CFG)
    pairs_idx      = [(i,j) for i in range(N) for j in range(i+1, N)]
    pair_multiplier = 2  # COLOR_SWAP é sempre ativo: cada abertura gera 2 jogos (um por cor)

    # ── Cap book games to available openings ──────────────────────────────
    # GAMES representa o total de posições de abertura alocadas no torneio.
    # Se SAME_OPENING_TWICE for True, o número real de jogos dobra.
    raw_iterations = max(1, GAMES // max(1, len(pairs_idx)))

    if OPENING_MODE == "book":
        raw_n_book, raw_n_rand = raw_iterations, 0
    elif OPENING_MODE == "random":
        raw_n_book, raw_n_rand = 0, raw_iterations
    else:  # "book+random" (or its old name, "all")
        raw_n_book = int(raw_iterations * BOOK_PORTION)
        raw_n_rand = raw_iterations - raw_n_book

    # Cap: each pair can consume at most n_book_positions openings
    # (they cycle via modulo, so the real cap is the full index).
    # For a fair round-robin each pair should get the same number, so cap
    # globally: max book iterations per pair = n_book_positions // len(pairs)
    # (or len(pairs)==0 guard).
    if n_book_positions > 0 and raw_n_book > 0:
        max_book_iters = max(1, n_book_positions // max(1, len(pairs_idx)))
        if raw_n_book > max_book_iters:
            capped_n_book = max_book_iters
            log(f"[Openings] AVISO: {raw_n_book} iterações de livro por par solicitadas, "
                f"mas apenas {n_book_positions} posições disponíveis para {len(pairs_idx)} par(es). "
                f"Reduzindo para {capped_n_book} iterações de livro por par.")
            raw_n_book = capped_n_book
            raw_n_rand += (raw_iterations - raw_n_book - raw_n_rand)  # make up diff with random

    iterations     = raw_n_book + raw_n_rand
    n_book_f       = raw_n_book
    n_rand_f       = raw_n_rand
    games_per_pair = iterations * pair_multiplier
    _real_total_games = games_per_pair * len(pairs_idx)

    book_games_total = n_book_f * 2 * len(pairs_idx)
    rand_games_total = n_rand_f * 2 * len(pairs_idx)
    log(f"[Openings] Jogos com livro : {book_games_total:,}  |  Jogos aleatórios: {rand_games_total:,}  |  Total: {_real_total_games:,}  (color-swap sempre ativo)")

    schedule = []
    gid = 1

    # Global counter so book picks cycle through the shuffled index without repeating
    book_cursor = 0

    for (i, j) in pairs_idx:
        n_b = min(n_book_f, len(all_openings))
        n_r = n_rand_f + (n_book_f - n_b)

        sequence = []
        for k in range(n_b):
            sequence.append(("book", (book_cursor + k) % max(1, len(all_openings))))
        book_cursor += n_b
        for _ in range(n_r):
            sequence.append(("random", None))

        for s_idx, (kind, oidx) in enumerate(sequence):
            # COLOR_SWAP sempre ativo: joga a abertura nos dois sentidos
            schedule.append((gid, i, j, kind, oidx)); gid += 1
            if SAME_OPENING_TWICE:
                # Repete a MESMA abertura com as cores invertidas
                schedule.append((gid, j, i, kind, oidx)); gid += 1
            else:
                # Usa a próxima abertura da sequência para o jogo de cor invertida
                next_oidx = (oidx + 1) % max(1, len(all_openings)) if kind == "book" else None
                schedule.append((gid, j, i, kind, next_oidx)); gid += 1

    random.shuffle(schedule)

    # start_time set here, AFTER indexing, so ETA reflects actual game time
    start_time = time.time()
    log(f"\n=== SELFPLAY EXE (Motores Persistentes) ===\nJogos: {_real_total_games} | Paralelismo: {CONCURRENCY}\n")

    task_q, res_q = Queue(), Queue()
    for item in schedule: task_q.put(item)

    def worker_loop():
        # Initialize engine processes
        instances = []
        for cfg in ENGINES_CFG:
            instances.append(EngineInstance(**cfg))
        for e in instances: e.start()

        while True:
            try:
                gid, w_idx, b_idx, kind, oidx = task_q.get_nowait()
                # Resolve opening lazily — only one seek+readline per game
                if kind == "book":
                    opening = all_openings.fetch(oidx)
                else:
                    opening = random_opening(RANDOM_PLIES)
                play_game(gid, instances[w_idx], instances[b_idx], opening, res_q)
                task_q.task_done()
            except Empty: break

        for e in instances: e.stop()

    threads = [threading.Thread(target=worker_loop, daemon=True) for _ in range(CONCURRENCY)]
    for t in threads: t.start()

    fe = open(EPD_FILE, "w", encoding="utf-8") if SAVE_EPD else None
    fp = open(PGN_FILE, "w", encoding="utf-8") if SAVE_PGN else None
    fb = open(BIN_FILE, "ab") if SAVE_BIN else None  # append mode, matches selfplay.exe's --out convention

    try:
        while completed_games < _real_total_games:
            try: res = res_q.get(timeout=1.0)
            except Empty:
                if all(not t.is_alive() for t in threads): break
                continue
            
            completed_games += 1
            w, b = res["white"], res["black"]
            with _stats_lock:
                stats[w]["games"] += 1; stats[b]["games"] += 1
                stats[w]["total_ms"]    += res["timing"][w]["ms"];    stats[w]["total_moves"] += res["timing"][w]["moves"]
                stats[w]["total_nodes"] += res["timing"][w]["nodes"]; stats[w]["total_depth"] += res["timing"][w]["depth"]
                stats[b]["total_ms"]    += res["timing"][b]["ms"];    stats[b]["total_moves"] += res["timing"][b]["moves"]
                stats[b]["total_nodes"] += res["timing"][b]["nodes"]; stats[b]["total_depth"] += res["timing"][b]["depth"]
                if res["timeout"]: stats[w]["to"] += 1; stats[b]["to"] += 1
                if res["error"]:   stats[w]["err"] += 1; stats[b]["err"] += 1
                if res["winner"] == "white": stats[w]["w"]+=1; stats[w]["pts"]+=1; stats[b]["l"]+=1; h2h[(w,b)]["w"]+=1; h2h[(b,w)]["l"]+=1
                elif res["winner"] == "black": stats[b]["w"]+=1; stats[b]["pts"]+=1; stats[w]["l"]+=1; h2h[(w,b)]["l"]+=1; h2h[(b,w)]["w"]+=1
                else: stats[w]["d"]+=1; stats[w]["pts"]+=0.5; stats[b]["d"]+=1; stats[b]["pts"]+=0.5; h2h[(w,b)]["d"]+=1; h2h[(b,w)]["d"]+=1
            
            with _file_lock:
                # Fix: only write EPD if there are positions to write (avoids blank lines)
                if fe and res["epd"]:
                    fe.write(res["epd"] + "\n"); fe.flush()
                if fp and res["pgn"]: fp.write(res["pgn"] + "\n\n"); fp.flush()
                if fb and res["bin"]: fb.write(res["bin"]); fb.flush()

            if completed_games % REPORT_EVERY == 0: print_status()

    finally:
        if fe: fe.close()
        if fp: fp.close()
        if fb: fb.close()

    log("\nTORNEIO FINALIZADO!")
    print_status(True)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda s,f: sys.exit(0))
    main()
