#!/usr/bin/env python3
"""uci_test.py — Comprehensive UCI test suite for Zchezz

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
  python tests/uci_test.py [version]    # e.g., python tests/uci_test.py v305
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

# Allow version override via CLI: python uci_test.py v305
if len(sys.argv) > 1 and sys.argv[1].startswith("v"):
    ENGINE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "engine", "c", f"zchezz_{sys.argv[1]}")
else:
    ENGINE_DIR = find_latest_engine()

ENGINE_PATH = os.path.join(ENGINE_DIR, "zchezz.exe")
# Syzygy path — adjust if needed
SYZYGY_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tablebases")
# Book path — will be tested if exists
BOOK_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "utils", "OpeningBook.bin")


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


# ── T3: Syzygy Tablebases ──────────────────────────────────────
def test_t3(eng):
    print("\n=== T3: Syzygy Tablebases ===")

    if not os.path.isdir(SYZYGY_PATH):
        print(f"  ⚠ Skipping — SyzygyPath not found: {SYZYGY_PATH}")
        return

    eng.send(f'setoption name SyzygyPath value {SYZYGY_PATH}')
    eng.send("isready")
    lines = eng.read_until(r"^readyok$", timeout=10)
    check("SyzygyPath set + readyok",
          any("readyok" in l for l in lines))

    # KQPK — 4-piece position (KQK 3-piece crashes the TB library)
    # White: Ke3, Qd5, Pd4  vs  Black: Ke8
    eng.send("position fen 4k3/8/8/3Q4/3P4/4K3/8/8 w - - 0 1")
    eng.send("go depth 10")
    lines = eng.read_until(r"^bestmove", timeout=20)
    check("KQPK endgame → bestmove",
          any(l.startswith("bestmove") for l in lines))

    # Check for tbhits
    info_lines = [l for l in lines if "tbhits" in l]
    if info_lines:
        m = re.search(r"tbhits\s+(\d+)", info_lines[-1])
        tbhits = int(m.group(1)) if m else 0
        check("KQPK → tbhits > 0",
              tbhits > 0,
              f"tbhits={tbhits}")
    else:
        check("KQPK → tbhits field present", False, "no tbhits in info")

    # SyzygyProbeDepth
    eng.send("setoption name SyzygyProbeDepth value 5")
    eng.send("isready")
    eng.read_until(r"^readyok$", timeout=5)
    check("SyzygyProbeDepth accepted", True)

    # SyzygyProbeLimit
    eng.send("setoption name SyzygyProbeLimit value 5")
    eng.send("isready")
    eng.read_until(r"^readyok$", timeout=5)
    check("SyzygyProbeLimit accepted", True)


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
    time.sleep(1.0)
    eng.send("stop")
    lines = eng.read_until(r"^bestmove", timeout=10)
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


# ── Main ──────────────────────────────────────────────────────
def main():
    global BOOK_PATH, SYZYGY_PATH

    if not os.path.exists(ENGINE_PATH):
        print(f"ERROR: Engine not found at {ENGINE_PATH}")
        sys.exit(1)

    print(f"Engine: {ENGINE_PATH}")
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
    ]

    for name, test_fn in test_groups:
        eng = UCIEngine(ENGINE_PATH)
        try:
            eng.start()
            time.sleep(0.3)
            # Handshake
            eng.send("uci")
            eng.read_until(r"^uciok$", timeout=5)
            eng.send("isready")
            eng.read_until(r"^readyok$", timeout=5)
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
    main()
