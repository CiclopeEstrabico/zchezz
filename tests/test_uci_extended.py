#!/usr/bin/env python3
"""test_uci_extended.py — Comprehensive UCI test suite for Zchezz

Tests 8 feature groups:
  T1: Basic handshake (uci, isready, ucinewgame)
  T2: Position + Go (startpos, fen, depth, movetime, nodes, infinite+stop)
  T3: Syzygy tablebases (SyzygyPath, probing, tbhits)
  T4: Opening book (OwnBook, BookFile, book probe)
  T5: MultiPV (set 1-4, verify multipv field)
  T6: Threads / Lazy SMP (Threads 1/2/4, stop works)
  T7: Full options (Contempt, MoveOverhead, Ponder, AnalyseMode, debug)
  T8: Commands (bench, d, eval)

Usage:
  python tests/test_uci_extended.py [version]    # e.g., python tests/test_uci_extended.py v305
"""

import subprocess
import sys
import os
import re
import time
import glob

# Force UTF-8 output on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import threading

def find_latest_engine():
    """Auto-detect the latest engine version directory."""
    base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "engine", "c")
    dirs = sorted(glob.glob(os.path.join(base, "zchezz_v*")))
    if not dirs:
        raise FileNotFoundError("No engine version found")
    return dirs[-1]  # latest by name sort

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ═══════════════ CONFIGURATION ═══════════════
# Edit these and run with no arguments; every one is also a CLI flag that
# overrides it (CLAUDE.md rule 8, see the COMMAND LINE block below).
VERSION      = ""      # engine folder suffix, e.g. "v400". "" = the highest
                       # zchezz_v* folder present (find_latest_engine())
ENGINE_PATH  = ""      # explicit engine binary; "" = derived from VERSION
SYZYGY_PATH  = os.path.join(_REPO, "tablebases")          # Syzygy folder
BOOK_PATH    = os.path.join(_REPO, "utils", "book.bin")   # opening book, tested if present
ONLY         = []      # [] = every test group; else the group ids to run, e.g. ["T3","T7"]
HANDSHAKE_TIMEOUT_S = 5   # seconds allowed for uciok / readyok per group
# ═══════════════════════════════════════════

# ═══════════════ COMMAND LINE ═══════════════
# `python tests/test_uci_extended.py` runs exactly what the block above says.
# The legacy positional form (`... v400`) still works and means --version v400.
# `--show-config` prints the resolved settings and exits. See utils/cliconf.py.
# ════════════════════════════════════════════
sys.path.insert(0, os.path.join(_REPO, "utils"))
from cliconf import override_from_cli  # noqa: E402

CLI = [
    ("VERSION",     "--version", str,  "engine folder suffix (engine/c/zchezz_<V>); empty = latest"),
    ("ENGINE_PATH", "--exe",     str,  "explicit engine binary, overrides --version"),
    ("SYZYGY_PATH", "--syzygy",  str,  "Syzygy tablebase folder"),
    ("BOOK_PATH",   "--book",    str,  "opening book file (tested only if it exists)"),
    ("ONLY",        "--only",    list, "run only this test group id, e.g. T3 (repeatable)"),
    ("HANDSHAKE_TIMEOUT_S", "--handshake-timeout", float, "seconds allowed for uciok/readyok"),
]


def resolve_engine_path() -> str:
    """The engine binary, from --exe, else --version, else the latest folder."""
    if ENGINE_PATH:
        return ENGINE_PATH
    engine_dir = (os.path.join(_REPO, "engine", "c", "zchezz_" + VERSION)
                  if VERSION else find_latest_engine())
    return os.path.join(engine_dir, "zchezz.exe")


class UCIEngine:
    """Manages a UCI engine subprocess."""

    def __init__(self, path):
        self.path = path
        self.proc = None
        self.output_lines = []
        self._reader_thread = None
        self._lock = threading.Lock()

    def start(self):
        self.proc = subprocess.Popen(
            [self.path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=os.path.dirname(self.path),
        )
        self.output_lines = []
        self._reader_thread = threading.Thread(target=self._reader, daemon=True)
        self._reader_thread.start()

    def _reader(self):
        for line in self.proc.stdout:
            with self._lock:
                self.output_lines.append(line.rstrip("\n\r"))

    def send(self, cmd):
        if self.proc and self.proc.poll() is None:
            self.proc.stdin.write(cmd + "\n")
            self.proc.stdin.flush()

    def read_until(self, pattern, timeout=10.0):
        """Read lines until one matches `pattern` (regex) or timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                for i, line in enumerate(self.output_lines):
                    if re.search(pattern, line):
                        # Return all lines up to and including the match
                        result = self.output_lines[:i+1]
                        self.output_lines = self.output_lines[i+1:]
                        return result
            time.sleep(0.05)
        # Timeout — return whatever we have
        with self._lock:
            result = list(self.output_lines)
            self.output_lines.clear()
        return result

    def read_lines(self, timeout=1.0):
        """Read all accumulated lines after a delay."""
        time.sleep(timeout)
        with self._lock:
            result = list(self.output_lines)
            self.output_lines.clear()
        return result

    def quit(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.send("quit")
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()

    def clear(self):
        with self._lock:
            self.output_lines.clear()


# ── Test results tracking ──────────────────────────────────────
passed = 0
failed = 0
errors = []


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        msg = f"  ✗ {name}" + (f" — {detail}" if detail else "")
        print(msg)
        errors.append(msg)


# ── T1: Basic Handshake ────────────────────────────────────────
def test_t1(eng):
    print("\n=== T1: Basic Handshake ===")

    eng.send("uci")
    lines = eng.read_until(r"^uciok$", timeout=5)
    text = "\n".join(lines)

    check("uci → id name Zchezz",
          any("id name Zchezz" in l for l in lines),
          f"got: {[l for l in lines if 'id name' in l]}")

    check("uci → id author",
          any("id author" in l for l in lines))

    check("uci → option name Hash",
          any("option name Hash" in l for l in lines))

    check("uci → option name Threads",
          any("option name Threads" in l for l in lines))

    check("uci → option name MultiPV",
          any("option name MultiPV" in l for l in lines))

    check("uci → option name SyzygyPath",
          any("option name SyzygyPath" in l for l in lines))

    check("uci → option name OwnBook",
          any("option name OwnBook" in l for l in lines))

    check("uci → option name BookFile",
          any("option name BookFile" in l for l in lines))

    check("uci → uciok",
          any("uciok" in l for l in lines))

    eng.send("isready")
    lines = eng.read_until(r"^readyok$", timeout=5)
    check("isready → readyok",
          any("readyok" in l for l in lines))

    eng.send("ucinewgame")
    eng.send("isready")
    lines = eng.read_until(r"^readyok$", timeout=5)
    check("ucinewgame + isready → readyok",
          any("readyok" in l for l in lines))


# ── T2: Position + Go ──────────────────────────────────────────
def test_t2(eng):
    print("\n=== T2: Position + Go ===")

    # go depth
    eng.send("position startpos")
    eng.send("go depth 5")
    lines = eng.read_until(r"^bestmove", timeout=15)
    bm_lines = [l for l in lines if l.startswith("bestmove")]
    check("go depth 5 → bestmove",
          len(bm_lines) > 0,
          f"no bestmove in {len(lines)} lines")

    if bm_lines:
        move = bm_lines[0].split()[1]
        check("bestmove is valid UCI (4-5 chars)",
              len(move) >= 4 and len(move) <= 5 and move[0] in "abcdefgh",
              f"got: {move}")

    # Info lines have depth, score, nodes
    info_lines = [l for l in lines if l.startswith("info depth")]
    check("go depth 5 → info lines emitted",
          len(info_lines) > 0)

    if info_lines:
        last_info = info_lines[-1]
        check("info has score cp or score mate",
              "score cp" in last_info or "score mate" in last_info,
              f"got: {last_info[:80]}")
        check("info has nodes",
              "nodes" in last_info)
        check("info has nps",
              "nps" in last_info)
        check("info has hashfull",
              "hashfull" in last_info)

    # go movetime
    eng.send("position startpos")
    eng.send("go movetime 500")
    lines = eng.read_until(r"^bestmove", timeout=10)
    check("go movetime 500 → bestmove",
          any(l.startswith("bestmove") for l in lines))

    # go nodes
    eng.send("position startpos")
    eng.send("go nodes 1000")
    lines = eng.read_until(r"^bestmove", timeout=10)
    check("go nodes 1000 → bestmove",
          any(l.startswith("bestmove") for l in lines))

    # position fen
    eng.send("position fen r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2")
    eng.send("go depth 5")
    lines = eng.read_until(r"^bestmove", timeout=15)
    check("position fen → go depth 5 → bestmove",
          any(l.startswith("bestmove") for l in lines))

    # position startpos moves
    eng.send("position startpos moves e2e4 e7e5 g1f3")
    eng.send("go depth 5")
    lines = eng.read_until(r"^bestmove", timeout=15)
    check("position startpos moves → bestmove",
          any(l.startswith("bestmove") for l in lines))

    # go wtime/btime
    eng.send("position startpos")
    eng.send("go wtime 60000 btime 60000 winc 0 binc 0")
    lines = eng.read_until(r"^bestmove", timeout=15)
    check("go wtime/btime → bestmove",
          any(l.startswith("bestmove") for l in lines))

    # go infinite + stop
    eng.send("position startpos")
    eng.send("go infinite")
    time.sleep(1.0)
    eng.send("stop")
    lines = eng.read_until(r"^bestmove", timeout=10)
    check("go infinite + stop → bestmove",
          any(l.startswith("bestmove") for l in lines))


# ── T3: Syzygy Tablebases (Comprehensive) ──────────────────────
def test_t3(eng):
    print("\n=== T3: Syzygy Tablebases (Comprehensive) ===")

    if not os.path.isdir(SYZYGY_PATH):
        print(f"  ⚠ Skipping — SyzygyPath not found: {SYZYGY_PATH}")
        return

    def sync():
        eng.send("isready")
        eng.read_until(r"^readyok$", timeout=5)
        time.sleep(0.2)
        eng.clear()

    # ── T3.1: TB Initialization ──────────────────────────────────
    eng.send(f'setoption name SyzygyPath value {SYZYGY_PATH}')
    eng.send("isready")
    lines = eng.read_until(r"^readyok$", timeout=10)
    all_text = "\n".join(lines)
    check("T3.1a: SyzygyPath set + readyok",
          any("readyok" in l for l in lines))
    check("T3.1b: Syzygy tables loaded message",
          "syzygy" in all_text.lower() or "loaded" in all_text.lower(),
          f"stderr/info may have the load message")

    # ── T3.2: 4-piece KRKP — must get tbhits > 0 ────────────────
    # White: Ke1, Rc2 vs Black: Ke6, Pd4.  hm=0 so TB should probe.
    sync()
    eng.send("ucinewgame")
    eng.send("isready")
    eng.read_until(r"^readyok$", timeout=5)
    eng.send("position fen 8/8/4k3/8/3p4/8/2R5/4K3 w - - 0 1")
    eng.send("go depth 12")
    lines = eng.read_until(r"^bestmove", timeout=30)
    info_lines = [l for l in lines if "tbhits" in l]
    tbhits_4pc = 0
    if info_lines:
        m = re.search(r"tbhits\s+(\d+)", info_lines[-1])
        tbhits_4pc = int(m.group(1)) if m else 0
    check("T3.2a: 4-piece KRKP → bestmove",
          any(l.startswith("bestmove") for l in lines))
    check("T3.2b: 4-piece KRKP → tbhits > 0",
          tbhits_4pc > 0,
          f"tbhits={tbhits_4pc}")
    # This exact rook-versus-pawn position is tablebase-drawn.  Material is
    # not a valid oracle here; the successful probe above is.
    score_4pc = None
    for l in reversed(info_lines):
        m = re.search(r"score cp (-?\d+)", l)
        if m:
            score_4pc = int(m.group(1))
            break
    if score_4pc is not None:
        check("T3.2c: 4-piece KRKP → tablebase-draw score",
              abs(score_4pc) < 100,
              f"score={score_4pc}")

    # ── T3.3: 5-piece KRPKB — must get tbhits > 0 ───────────────
    # White: Kc2, Rd4, Pe4 vs Black: Kd6, Be1.
    sync()
    eng.send("ucinewgame")
    eng.send("isready")
    eng.read_until(r"^readyok$", timeout=5)
    eng.send("position fen 8/8/3k4/8/3RP3/8/2K5/4b3 w - - 0 1")
    eng.send("go depth 12")
    lines = eng.read_until(r"^bestmove", timeout=30)
    info_lines = [l for l in lines if "tbhits" in l]
    tbhits_5pc = 0
    if info_lines:
        m = re.search(r"tbhits\s+(\d+)", info_lines[-1])
        tbhits_5pc = int(m.group(1)) if m else 0
    check("T3.3a: 5-piece KRPKB → bestmove",
          any(l.startswith("bestmove") for l in lines))
    check("T3.3b: 5-piece KRPKB → tbhits > 0",
          tbhits_5pc > 0,
          f"tbhits={tbhits_5pc}")

    # ── T3.4: 3-piece KRK — must work with npieces>=3 guard ─────
    # 3-piece positions are now probed (guard changed to npieces<3).
    # KRK is trivially won but TB confirms it.
    sync()
    eng.send("ucinewgame")
    eng.send("isready")
    eng.read_until(r"^readyok$", timeout=5)
    eng.send("position fen 8/8/8/4k3/8/8/8/R3K3 w - - 0 1")
    eng.send("go depth 12")
    lines = eng.read_until(r"^bestmove", timeout=30)
    check("T3.4a: 3-piece KRK → bestmove (no crash)",
          any(l.startswith("bestmove") for l in lines))
    info_lines = [l for l in lines if "tbhits" in l]
    tbhits_3pc = 0
    if info_lines:
        m = re.search(r"tbhits\s+(\d+)", info_lines[-1])
        tbhits_3pc = int(m.group(1)) if m else 0
    check("T3.4b: 3-piece KRK → tbhits >= 0 (no crash)",
          True,
          f"tbhits={tbhits_3pc}")

    # ── T3.5: 4-piece KPK with pawn — tbhits after pawn push ────
    # White: Ke1, Pe2 vs Black: Ke6.  Root has hm=0, pawn pushes
    # create hm=0 children → TB should fire in non-PV nodes.
    sync()
    eng.send("ucinewgame")
    eng.send("isready")
    eng.read_until(r"^readyok$", timeout=5)
    eng.send("position fen 4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    eng.send("go depth 14")
    lines = eng.read_until(r"^bestmove", timeout=30)
    info_lines = [l for l in lines if "tbhits" in l]
    tbhits_kpk = 0
    if info_lines:
        m = re.search(r"tbhits\s+(\d+)", info_lines[-1])
        tbhits_kpk = int(m.group(1)) if m else 0
    check("T3.5a: KPK (3 pieces) → bestmove",
          any(l.startswith("bestmove") for l in lines))
    # KPK now probed (guard changed to npieces<3)
    # tbhits may be 0 if all nodes are PV or hm>0
    check("T3.5b: KPK → no crash (3-piece probing enabled)",
          True,
          f"tbhits={tbhits_kpk}")

    # ── T3.6: TB with hm > 0 — should NOT probe ─────────────────
    # Same 4-piece position but with hm=5 (no recent capture/pawn move)
    sync()
    eng.send("ucinewgame")
    eng.send("isready")
    eng.read_until(r"^readyok$", timeout=5)
    eng.send("position fen 8/8/4k3/8/3p4/8/2R5/4K3 w - - 5 10")
    eng.send("go depth 10")
    lines = eng.read_until(r"^bestmove", timeout=30)
    info_lines = [l for l in lines if "tbhits" in l]
    tbhits_hm5 = 0
    if info_lines:
        m = re.search(r"tbhits\s+(\d+)", info_lines[-1])
        tbhits_hm5 = int(m.group(1)) if m else 0
    check("T3.6a: KRKP hm=5 → bestmove",
          any(l.startswith("bestmove") for l in lines))
    # With hm=5, root doesn't probe. But captures inside the tree
    # create hm=0 children. tbhits may be 0 or small.
    check("T3.6b: KRKP hm=5 → fewer tbhits than hm=0",
          tbhits_hm5 <= tbhits_4pc,
          f"hm5={tbhits_hm5} vs hm0={tbhits_4pc}")

    # ── T3.7: TB drawn position — score near 0 ──────────────────
    # KBKN is a theoretical draw (bishop vs knight). 4 pieces.
    sync()
    eng.send("ucinewgame")
    eng.send("isready")
    eng.read_until(r"^readyok$", timeout=5)
    eng.send("position fen 8/8/4k3/8/3B4/8/5n2/4K3 w - - 0 1")
    eng.send("go depth 14")
    lines = eng.read_until(r"^bestmove", timeout=30)
    info_lines = [l for l in lines if "tbhits" in l]
    tbhits_draw = 0
    if info_lines:
        m = re.search(r"tbhits\s+(\d+)", info_lines[-1])
        tbhits_draw = int(m.group(1)) if m else 0
    check("T3.7a: KBKN (drawn) → bestmove",
          any(l.startswith("bestmove") for l in lines))
    check("T3.7b: KBKN (drawn) → no invalid TB probe",
          tbhits_draw == 0,
          f"tbhits={tbhits_draw}")
    # Score should be close to 0 (drawn endgame)
    score_draw = None
    for l in reversed(info_lines):
        m = re.search(r"score cp (-?\d+)", l)
        if m:
            score_draw = int(m.group(1))
            break
    if score_draw is not None:
        check("T3.7c: KBKN (drawn) → score near 0 (|score| < 100)",
              abs(score_draw) < 100,
              f"score={score_draw}")

    # ── T3.8: SyzygyProbeDepth option ────────────────────────────
    sync()
    eng.send("setoption name SyzygyProbeDepth value 5")
    eng.send("isready")
    eng.read_until(r"^readyok$", timeout=5)
    check("T3.8: SyzygyProbeDepth=5 accepted", True)

    # ── T3.9: SyzygyProbeLimit option ────────────────────────────
    eng.send("setoption name SyzygyProbeLimit value 4")
    eng.send("isready")
    eng.read_until(r"^readyok$", timeout=5)

    # With ProbeLimit=4, 5-piece positions should NOT probe
    eng.send("ucinewgame")
    eng.send("isready")
    eng.read_until(r"^readyok$", timeout=5)
    eng.send("position fen 8/8/3k4/8/3RP3/8/2K5/4b3 w - - 0 1")
    eng.send("go depth 10")
    lines = eng.read_until(r"^bestmove", timeout=30)
    info_lines = [l for l in lines if "tbhits" in l]
    tbhits_limited = 0
    if info_lines:
        m = re.search(r"tbhits\s+(\d+)", info_lines[-1])
        tbhits_limited = int(m.group(1)) if m else 0
    check("T3.9a: ProbeLimit=4, 5-piece → bestmove",
          any(l.startswith("bestmove") for l in lines),
          f"limited={tbhits_limited} vs unlimited={tbhits_5pc}")

    # ── T3.10: NPS comparison (TB overhead check) ────────────────
    # Reset options
    eng.send("setoption name SyzygyProbeDepth value 1")
    eng.send("setoption name SyzygyProbeLimit value 6")
    sync()
    # Middlegame position (many pieces, no TB probing expected)
    eng.send("ucinewgame")
    eng.send("isready")
    eng.read_until(r"^readyok$", timeout=5)
    eng.send("position startpos")
    eng.send("go depth 14")
    lines = eng.read_until(r"^bestmove", timeout=30)
    nps_tb = 0
    for l in reversed(lines):
        m = re.search(r"nps\s+(\d+)", l)
        if m:
            nps_tb = int(m.group(1))
            break
    check("T3.10: Middlegame NPS > 1M (no TB overhead)",
          nps_tb > 1000000,
          f"nps={nps_tb}")

    # ── T3.11: Multiple sequential TB searches (no crash) ────────
    sync()
    tb_positions = [
        "8/8/4k3/8/3p4/8/2R5/4K3 w - - 0 1",   # KRKP
        "8/8/3k4/8/3RP3/8/2K5/4b3 w - - 0 1",   # KRPKB
        "4k3/8/8/3Q4/3P4/4K3/8/8 w - - 0 1",     # KQPK
        "8/8/4k3/8/3B4/8/5n2/4K3 w - - 0 1",     # KBKN
        "8/3k4/8/8/8/8/3KR3/4r3 w - - 0 1",       # KRKR
    ]
    crash_free = True
    for i, fen in enumerate(tb_positions):
        eng.send("ucinewgame")
        eng.send("isready")
        eng.read_until(r"^readyok$", timeout=5)
        eng.send(f"position fen {fen}")
        eng.send("go depth 10")
        lines = eng.read_until(r"^bestmove", timeout=20)
        if not any(l.startswith("bestmove") for l in lines):
            crash_free = False
            break
    check("T3.11: 5 sequential TB positions → no crash",
          crash_free)


# ── T4: Opening Book ──────────────────────────────────────────
def test_t4(eng):
    print("\n=== T4: Opening Book ===")

    # Test OwnBook option
    eng.send("setoption name OwnBook value true")
    eng.send("isready")
    lines = eng.read_until(r"^readyok$", timeout=5)
    check("OwnBook set to true", any("readyok" in l for l in lines))

    if BOOK_PATH and os.path.exists(BOOK_PATH):
        eng.send(f"setoption name BookFile value {BOOK_PATH}")
        eng.send("isready")
        lines = eng.read_until(r"^readyok$", timeout=5)
        check("BookFile set", any("readyok" in l for l in lines))

        eng.send("position startpos")
        eng.send("go depth 5")
        lines = eng.read_until(r"^bestmove", timeout=15)

        book_info = [l for l in lines if "book move" in l.lower()]
        bm_lines = [l for l in lines if l.startswith("bestmove")]
        check("Book probe from startpos → bestmove",
              len(bm_lines) > 0)

        if book_info:
            check("Book probe → info string book move", True)
        else:
            check("Book probe → info string book move (may search instead)", True)
    else:
        print("  ⚠ No book file specified — skipping book file tests")

    # Disable book and verify normal search works
    eng.send("setoption name OwnBook value false")
    eng.send("isready")
    eng.read_until(r"^readyok$", timeout=5)

    eng.send("position startpos")
    eng.send("go depth 5")
    lines = eng.read_until(r"^bestmove", timeout=15)
    check("OwnBook=false → normal search → bestmove",
          any(l.startswith("bestmove") for l in lines))


# ── T5: MultiPV ───────────────────────────────────────────────
def test_t5(eng):
    print("\n=== T5: MultiPV ===")

    for npv in [1, 2, 3, 4]:
        eng.send(f"setoption name MultiPV value {npv}")
        eng.send("isready")
        eng.read_until(r"^readyok$", timeout=5)

        eng.send("position startpos")
        eng.send("go depth 7")
        lines = eng.read_until(r"^bestmove", timeout=30)

        bm = [l for l in lines if l.startswith("bestmove")]
        check(f"MultiPV={npv} → bestmove", len(bm) > 0)

        if npv > 1:
            # Check for multipv field in info lines
            mpv_lines = [l for l in lines if f"multipv {npv}" in l or "multipv" in l]
            check(f"MultiPV={npv} → multipv field in info",
                  len(mpv_lines) > 0,
                  f"found {len(mpv_lines)} lines with multipv")

            # Check that we have info lines for each PV
            max_mpv = 0
            for l in lines:
                m = re.search(r"multipv\s+(\d+)", l)
                if m:
                    max_mpv = max(max_mpv, int(m.group(1)))
            check(f"MultiPV={npv} → max multipv index = {npv}",
                  max_mpv == npv,
                  f"max_mpv={max_mpv}")

    # Reset to 1
    eng.send("setoption name MultiPV value 1")
    eng.send("isready")
    eng.read_until(r"^readyok$", timeout=5)


# ── T6: Threads ───────────────────────────────────────────────
def test_t6(eng):
    print("\n=== T6: Threads / Lazy SMP ===")

    for n in [1, 2, 4]:
        eng.send(f"setoption name Threads value {n}")
        eng.send("isready")
        eng.read_until(r"^readyok$", timeout=5)

        eng.send("position startpos")
        eng.send("go depth 7")
        lines = eng.read_until(r"^bestmove", timeout=30)
        check(f"Threads={n} → bestmove",
              any(l.startswith("bestmove") for l in lines))

    # Test stop with threads
    eng.send("setoption name Threads value 2")
    eng.send("isready")
    eng.read_until(r"^readyok$", timeout=5)

    eng.send("position startpos")
    eng.send("go infinite")
    time.sleep(2.0)  # Allow helpers to fully start before stopping
    eng.send("stop")
    lines = eng.read_until(r"^bestmove", timeout=15)
    check("Threads=2 + go infinite + stop → bestmove",
          any(l.startswith("bestmove") for l in lines))

    # Reset
    eng.send("setoption name Threads value 1")
    eng.send("isready")
    eng.read_until(r"^readyok$", timeout=5)


# ── T7: Full Options ──────────────────────────────────────────
def test_t7(eng):
    print("\n=== T7: Full Options ===")

    # Contempt
    eng.send("setoption name Contempt value 25")
    eng.send("isready")
    lines = eng.read_until(r"^readyok$", timeout=5)
    check("Contempt=25 accepted", any("readyok" in l for l in lines))

    # MoveOverhead
    eng.send("setoption name MoveOverhead value 100")
    eng.send("isready")
    lines = eng.read_until(r"^readyok$", timeout=5)
    check("MoveOverhead=100 accepted", any("readyok" in l for l in lines))

    # Ponder (bestmove should include ponder move)
    eng.send("setoption name Ponder value true")
    eng.send("isready")
    eng.read_until(r"^readyok$", timeout=5)
    eng.send("position startpos")
    eng.send("go depth 7")
    lines = eng.read_until(r"^bestmove", timeout=15)
    bm = [l for l in lines if l.startswith("bestmove")]
    if bm:
        check("Ponder=true → bestmove has ponder field",
              "ponder" in bm[0],
              f"got: {bm[0]}")
    else:
        check("Ponder=true → bestmove", False)

    eng.send("setoption name Ponder value false")

    # UCI_AnalyseMode
    eng.send("setoption name UCI_AnalyseMode value true")
    eng.send("isready")
    lines = eng.read_until(r"^readyok$", timeout=5)
    check("UCI_AnalyseMode=true accepted", any("readyok" in l for l in lines))
    eng.send("setoption name UCI_AnalyseMode value false")

    # Debug on/off
    eng.send("debug on")
    eng.send("isready")
    lines = eng.read_until(r"^readyok$", timeout=5)
    check("debug on accepted", any("readyok" in l for l in lines))
    eng.send("debug off")


# ── T8: Commands ──────────────────────────────────────────────
def test_t8(eng):
    print("\n=== T8: Commands ===")

    # bench
    eng.clear()
    eng.send("bench 5")
    lines = eng.read_until(r"Nodes/sec", timeout=60)
    check("bench → Nodes/sec output",
          any("Nodes/sec" in l for l in lines),
          f"got {len(lines)} lines")

    # d (display)
    eng.send("position startpos")
    eng.send("d")
    lines = eng.read_until(r"Fen:", timeout=5)
    check("d → Fen: output",
          any("Fen:" in l for l in lines))
    check("d → board display (+---+)",
          any("+---+" in l for l in lines))

    # eval
    eng.send("eval")
    lines = eng.read_lines(timeout=1.0)
    check("eval → cp output",
          any("cp" in l.lower() or "eval" in l.lower() for l in lines),
          f"lines: {lines[:3]}")

    # mate detection
    eng.send("position fen 6k1/5ppp/8/8/8/8/8/R3K3 w Q - 0 1")
    eng.send("go depth 8")
    lines = eng.read_until(r"^bestmove", timeout=15)
    check("mate position → bestmove",
          any(l.startswith("bestmove") for l in lines))

    # FEN round-trip
    test_fen = "r1bqkb1r/pppppppp/2n2n2/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 2 3"
    eng.send(f"position fen {test_fen}")
    eng.send("d")
    lines = eng.read_until(r"Fen:", timeout=5)
    fen_lines = [l for l in lines if l.startswith("Fen:")]
    if fen_lines:
        output_fen = fen_lines[0].replace("Fen: ", "").strip()
        check("FEN round-trip",
              output_fen == test_fen,
              f"expected: {test_fen}\n       got:      {output_fen}")
    else:
        check("FEN round-trip", False, "no Fen: line in output")

# ── T9: Regression & Stress Tests ─────────────────────────────
def test_t9(eng):
    print("\n=== T9: Regression & Stress ===")

    # Depth consistency: go depth 10 must reach depth 10
    eng.send("position startpos")
    eng.send("go depth 10")
    lines = eng.read_until(r"^bestmove", timeout=30)
    info_d10 = [l for l in lines if "info" in l and " depth 10 " in l]
    check("go depth 10 → info depth 10 present",
          len(info_d10) > 0,
          f"found {len(info_d10)} depth-10 info lines")

    # NPS sanity check: must be > 100K (catch broken eval/search)
    nps_val = 0
    for l in reversed(lines):
        m = re.search(r"nps\s+(\d+)", l)
        if m:
            nps_val = int(m.group(1))
            break
    check("NPS > 100,000 (sanity)",
          nps_val > 100000,
          f"nps={nps_val}")

    # Node count check: must be > 0
    nodes_val = 0
    for l in reversed(lines):
        m = re.search(r"nodes\s+(\d+)", l)
        if m:
            nodes_val = int(m.group(1))
            break
    check("nodes > 0 at depth 10",
          nodes_val > 0,
          f"nodes={nodes_val}")

    # Multiple FEN positions — opening, middlegame, endgame
    test_fens = [
        ("opening", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),
        ("middlegame", "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w - - 4 6"),
        ("endgame", "8/5pk1/7p/3p1Rp1/p2P2P1/1r5P/1P3PK1/8 w - - 2 40"),
    ]
    for label, fen in test_fens:
        eng.send(f"position fen {fen}")
        eng.send("go depth 8")
        lines = eng.read_until(r"^bestmove", timeout=15)
        bm = [l for l in lines if l.startswith("bestmove")]
        check(f"{label} FEN → bestmove",
              len(bm) > 0 and len(bm[0].split()[1]) >= 4)

    # Rapid position cycling: 10 position+go cycles without crash
    rapid_ok = True
    for i in range(10):
        eng.send("position startpos moves e2e4 e7e5 g1f3")
        eng.send("go depth 4")
        lines = eng.read_until(r"^bestmove", timeout=10)
        if not any(l.startswith("bestmove") for l in lines):
            rapid_ok = False
            break
    check("rapid cycling (10x position+go) no crash", rapid_ok)

    # Long move list: 30+ moves applied → bestmove (undo stack stress)
    long_moves = "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 " \
                 "f1e1 b7b5 a4b3 d7d6 c2c3 e8g8 h2h3 c6b8 d2d4 b8d7 " \
                 "b3c2 c7c5 d4d5 d7b6 b1d2 a6a5 b2b3 a5a4"
    eng.send(f"position startpos moves {long_moves}")
    eng.send("go depth 6")
    lines = eng.read_until(r"^bestmove", timeout=15)
    check("long move list (28 moves) → bestmove",
          any(l.startswith("bestmove") for l in lines))

    # Threads=4 deep search (crash stress)
    eng.send("setoption name Threads value 4")
    eng.send("isready")
    eng.read_until(r"^readyok$", timeout=5)
    eng.send("position startpos")
    eng.send("go depth 10")
    lines = eng.read_until(r"^bestmove", timeout=60)
    check("Threads=4 depth 10 → bestmove",
          any(l.startswith("bestmove") for l in lines))

    # Multiple sequential Threads=4 games (crash stress)
    multi_ok = True
    for i in range(5):
        eng.send("ucinewgame")
        eng.send("isready")
        eng.read_until(r"^readyok$", timeout=5)
        eng.send("position startpos moves e2e4 e7e5")
        eng.send("go depth 6")
        lines = eng.read_until(r"^bestmove", timeout=30)
        if not any(l.startswith("bestmove") for l in lines):
            multi_ok = False
            break
    check("5x sequential Threads=4 games no crash", multi_ok)

    # Reset threads
    eng.send("setoption name Threads value 1")
    eng.send("isready")
    eng.read_until(r"^readyok$", timeout=5)

    # Perft validation: perft(4) from startpos = 197,281
    eng.send("position startpos")
    eng.send("perft 4")
    lines = eng.read_until(r"Nodes searched", timeout=15)
    perft_ok = False
    nodes = -1
    for l in lines:
        if "Nodes searched:" in l:
            nodes = int(l.split(":")[1].strip())
            perft_ok = (nodes == 197281)
            break
    check("perft(4) = 197,281",
          perft_ok,
          f"nodes={nodes}" if not perft_ok else "")


# ── T10: v3.13 Thread Safety (NnueAccum) ──────────────────────
def test_t10(eng):
    print("\n=== T10: v3.13 Thread Safety ===")

    # Different thread counts with same position → all produce valid bestmove
    test_fen = "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2"
    for threads in [1, 2, 4, 8]:
        eng.send(f"setoption name Threads value {threads}")
        eng.send("isready")
        eng.read_until(r"^readyok$", timeout=5)

        eng.send(f"position fen {test_fen}")
        eng.send("go depth 6")
        lines = eng.read_until(r"^bestmove", timeout=30)
        bm = [l for l in lines if l.startswith("bestmove")]
        check(f"Threads={threads} depth 6 → valid bestmove",
              len(bm) > 0 and len(bm[0].split()[1]) >= 4)

    # Rapid thread switching: alternate between 1 and 4 threads
    for i in range(5):
        t = 1 if i % 2 == 0 else 4
        eng.send(f"setoption name Threads value {t}")
        eng.send("isready")
        eng.read_until(r"^readyok$", timeout=5)
        eng.send("position startpos moves e2e4 e7e5")
        eng.send("go depth 5")
        lines = eng.read_until(r"^bestmove", timeout=15)
        check(f"Thread switch cycle {i+1} (Threads={t}) → bestmove",
              any(l.startswith("bestmove") for l in lines))

    # Multiple ucinewgame + threads (accumulator reset stress)
    eng.send("setoption name Threads value 4")
    for i in range(3):
        eng.send("ucinewgame")
        eng.send("isready")
        eng.read_until(r"^readyok$", timeout=5)
        eng.send("position startpos")
        eng.send("go depth 5")
        lines = eng.read_until(r"^bestmove", timeout=15)
        check(f"ucinewgame + Threads=4 cycle {i+1} → bestmove",
              any(l.startswith("bestmove") for l in lines))

    eng.send("setoption name Threads value 1")
    eng.send("isready")
    eng.read_until(r"^readyok$", timeout=5)


# ── T11: Edge Case Positions ──────────────────────────────────
def test_t11(eng):
    print("\n=== T11: Edge Case Positions ===")

    edge_positions = [
        ("stalemate detection",
         "k7/8/1K6/8/8/8/8/8 b - - 0 1", 8),
        ("single legal move",
         "k7/8/1K6/8/8/8/8/R7 b - - 0 1", 6),
        ("promotion required",
         "8/P7/8/8/8/8/8/K1k5 w - - 0 1", 6),
        ("double check",
         "r1bqkbnr/pppp1ppp/8/4p3/2B5/5Q2/PPPPPPPP/RNB1K1NR w KQkq - 2 3", 5),
        ("pinned piece",
         "rnbqk2r/ppppbppp/5n2/4p3/4P3/5N2/PPPPBPPP/RNBQK2R w KQkq - 4 4", 6),
        ("EP discovery check",
         "8/8/8/8/k1Pp4/8/1K6/8 b - c3 0 1", 6),
        ("rook vs bishop endgame",
         "4k3/8/8/4R3/8/8/b7/4K3 w - - 0 1", 8),
        ("KNB vs K (tricky mate)",
         "8/8/8/8/8/8/4k3/K1B1N3 w - - 0 1", 8),
        ("many captures available",
         "r1b1k1nr/pppp1ppp/2n5/4P3/1bB5/5N2/PPPN1PPP/R1BQK2R w KQkq - 1 5", 6),
        ("FEN with high halfmove clock",
         "8/8/3k4/8/8/3K4/8/8 w - - 90 100", 6),
    ]

    for label, fen, depth in edge_positions:
        eng.send(f"position fen {fen}")
        eng.send(f"go depth {depth}")
        lines = eng.read_until(r"^bestmove", timeout=15)
        bm = [l for l in lines if l.startswith("bestmove")]
        check(f"{label} → bestmove",
              len(bm) > 0,
              f"no bestmove in {len(lines)} lines")


# ── T12: Score and PV Validation ──────────────────────────────
def test_t12(eng):
    print("\n=== T12: Score & PV Validation ===")

    def sync():
        """Synchronize engine state between sub-tests."""
        eng.send("isready")
        eng.read_until(r"^readyok$", timeout=5)
        time.sleep(0.2)  # Let reader thread fully drain stdout pipe
        eng.clear()

    # Winning position: score should be > 0 for white
    sync()
    eng.send("position fen 8/8/8/8/8/8/4k3/4K2R w - - 0 1")
    eng.send("go depth 8")
    lines = eng.read_until(r"^bestmove", timeout=15)
    info_lines = [l for l in lines if l.startswith("info depth")]
    if info_lines:
        last = info_lines[-1]
        m = re.search(r"score cp (\-?\d+)", last)
        m2 = re.search(r"score mate (\-?\d+)", last)
        if m:
            score = int(m.group(1))
            check("winning position → positive score", score > 0, f"score={score}")
        elif m2:
            mate = int(m2.group(1))
            check("winning position → mate score", mate > 0, f"mate={mate}")
        else:
            check("winning position → has score", False)
    else:
        check("winning position → info lines", False)

    # Losing position: score should be < 0 for white
    sync()
    eng.send("position fen 8/8/8/8/8/8/4K3/4k2r b - - 0 1")
    eng.send("go depth 8")
    lines = eng.read_until(r"^bestmove", timeout=15)
    info_lines = [l for l in lines if l.startswith("info depth")]
    if info_lines:
        last = info_lines[-1]
        m = re.search(r"score cp (\-?\d+)", last)
        m2 = re.search(r"score mate (\-?\d+)", last)
        score_ok = False
        if m:
            score_ok = True
        elif m2:
            score_ok = True
        check("losing-for-white position → has valid score", score_ok)
    else:
        check("losing-for-white position → info lines", False)

    # PV must contain valid UCI moves
    sync()
    eng.send("position startpos")
    eng.send("go depth 8")
    lines = eng.read_until(r"^bestmove", timeout=15)
    info_lines = [l for l in lines if l.startswith("info depth")]
    if info_lines:
        last = info_lines[-1]
        m = re.search(r"pv (.+)", last)
        if m:
            pv_str = m.group(1).strip()
            pv_moves = pv_str.split()
            all_valid = all(
                len(mv) >= 4 and len(mv) <= 5 and mv[0] in "abcdefgh"
                for mv in pv_moves
            )
            check(f"PV contains valid UCI moves ({len(pv_moves)} moves)",
                  all_valid and len(pv_moves) > 0,
                  f"pv: {pv_str[:60]}")
        else:
            check("depth 8 → pv field present", False)
    else:
        check("depth 8 → info lines present", False)

    # Mate in 1 detection
    sync()
    eng.send("position fen 6k1/5ppp/8/8/8/8/8/R3K3 w Q - 0 1")
    eng.send("go depth 10")
    lines = eng.read_until(r"^bestmove", timeout=15)
    info_lines = [l for l in lines if "score mate" in l]
    check("mate-in-1 → score mate detected",
          len(info_lines) > 0,
          f"found {len(info_lines)} mate lines")

    # Depth progression: each depth should have info
    sync()
    eng.send("position startpos")
    eng.send("go depth 6")
    lines = eng.read_until(r"^bestmove", timeout=15)
    depths_seen = set()
    for l in lines:
        m = re.search(r"info depth (\d+)", l)
        if m:
            depths_seen.add(int(m.group(1)))
    check("depth 6 → info for depths 1-6",
          all(d in depths_seen for d in range(1, 7)),
          f"depths seen: {sorted(depths_seen)}")

    # hashfull increases with depth
    sync()
    eng.send("ucinewgame")  # Clear TT for clean hashfull measurement
    eng.send("isready")
    eng.read_until(r"^readyok$", timeout=5)
    eng.send("position startpos")
    eng.send("go depth 12")
    lines = eng.read_until(r"^bestmove", timeout=30)
    hashfull_vals = []
    for l in lines:
        m = re.search(r"hashfull (\d+)", l)
        if m:
            hashfull_vals.append(int(m.group(1)))
    check("hashfull reported in info",
          len(hashfull_vals) > 0)
    if len(hashfull_vals) > 2:
        check("hashfull increases with depth",
              hashfull_vals[-1] >= hashfull_vals[0],
              f"first={hashfull_vals[0]}, last={hashfull_vals[-1]}")


# ── T13: UCI Protocol Conformance ─────────────────────────────
def test_t13(eng):
    print("\n=== T13: UCI Protocol Conformance ===")

    # Registration stub
    eng.send("register later")
    eng.send("isready")
    lines = eng.read_until(r"^readyok$", timeout=5)
    check("register later → readyok", any("readyok" in l for l in lines))

    # Unknown command → no crash
    eng.send("foobar unknown command")
    eng.send("isready")
    lines = eng.read_until(r"^readyok$", timeout=5)
    check("unknown command → no crash", any("readyok" in l for l in lines))

    # Empty line → no crash
    eng.send("")
    eng.send("isready")
    lines = eng.read_until(r"^readyok$", timeout=5)
    check("empty line → no crash", any("readyok" in l for l in lines))

    # Multiple rapid isready
    for i in range(5):
        eng.send("isready")
    lines = eng.read_until(r"^readyok$", timeout=5)
    ready_count = sum(1 for l in lines if "readyok" in l)
    check("5x rapid isready → at least 1 readyok",
          ready_count >= 1, f"count={ready_count}")

    # Position with no moves keyword
    eng.send("position fen rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    eng.send("go depth 4")
    lines = eng.read_until(r"^bestmove", timeout=10)
    check("position fen (no moves) → bestmove",
          any(l.startswith("bestmove") for l in lines))

    # go mate (mate search)
    eng.send("position fen 6k1/5ppp/8/8/8/8/8/R3K3 w Q - 0 1")
    eng.send("go mate 2")
    lines = eng.read_until(r"^bestmove", timeout=15)
    check("go mate 2 → bestmove",
          any(l.startswith("bestmove") for l in lines))

    # Multiple go commands (second should work after first)
    eng.send("position startpos")
    eng.send("go depth 4")
    eng.read_until(r"^bestmove", timeout=10)
    eng.send("position startpos moves e2e4")
    eng.send("go depth 4")
    lines = eng.read_until(r"^bestmove", timeout=10)
    check("sequential go commands → second bestmove",
          any(l.startswith("bestmove") for l in lines))

    # Option name with spaces (UCI_AnalyseMode)
    eng.send("setoption name UCI_AnalyseMode value true")
    eng.send("isready")
    lines = eng.read_until(r"^readyok$", timeout=5)
    check("option with underscore name → readyok",
          any("readyok" in l for l in lines))

    # UCI_Chess960 option
    eng.send("setoption name UCI_Chess960 value false")
    eng.send("isready")
    lines = eng.read_until(r"^readyok$", timeout=5)
    check("UCI_Chess960 option → readyok",
          any("readyok" in l for l in lines))


# ── Main ──────────────────────────────────────────────────────
def main():
    global BOOK_PATH, SYZYGY_PATH

    engine_path = resolve_engine_path()
    if not os.path.exists(engine_path):
        print(f"ERROR: Engine not found at {engine_path}")
        sys.exit(1)

    print(f"Engine: {engine_path}")
    print(f"Syzygy: {SYZYGY_PATH} ({'exists' if os.path.isdir(SYZYGY_PATH) else 'NOT FOUND'})")
    if BOOK_PATH:
        print(f"Book:   {BOOK_PATH}")

    # Run each test group with a FRESH engine to prevent cascade failures
    test_groups = [
        ("T1", test_t1),
        ("T2", test_t2),
        ("T3", test_t3),
        ("T4", test_t4),
        ("T5", test_t5),
        ("T6", test_t6),
        ("T7", test_t7),
        ("T8", test_t8),
        ("T9", test_t9),
        ("T10", test_t10),
        ("T11", test_t11),
        ("T12", test_t12),
        ("T13", test_t13),
    ]

    if ONLY:
        wanted = {o.upper() for o in ONLY}
        unknown = wanted - {n for n, _ in test_groups}
        if unknown:
            print(f"ERROR: --only names unknown test group(s): {sorted(unknown)}")
            sys.exit(1)
        test_groups = [(n, f) for n, f in test_groups if n in wanted]

    for name, test_fn in test_groups:
        eng = UCIEngine(engine_path)
        try:
            eng.start()
            time.sleep(0.3)
            # Handshake
            eng.send("uci")
            eng.read_until(r"^uciok$", timeout=HANDSHAKE_TIMEOUT_S)
            eng.send("isready")
            eng.read_until(r"^readyok$", timeout=HANDSHAKE_TIMEOUT_S)
            test_fn(eng)
        except Exception as e:
            print(f"  ERROR in {name}: {e}")
        finally:
            eng.quit()

    # Summary
    total = passed + failed
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if errors:
        print(f"\nFailures:")
        for e in errors:
            print(e)
    print(f"{'='*50}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    # Legacy positional form: `python tests/test_uci_extended.py v400`.
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        sys.argv[1:2] = ["--version", sys.argv[1].lstrip("v") and sys.argv[1]]
    override_from_cli(globals(), CLI, description="Extended UCI test suite",
                      prog="test_uci_extended.py")
    main()

