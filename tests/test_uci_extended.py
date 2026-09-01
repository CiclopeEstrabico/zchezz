#!/usr/bin/env python3
"""Deterministic UCI functional suite for Zchezz.

Groups:
  T1  handshake and required option inventory
  T2  position/go command behavior
  T3  Syzygy probing when runtime tablebases are available
  T4  opening-book option behavior
  T5  MultiPV
  T6  Threads / stop behavior
  T7  supported option setting
  T8  engine extension commands
  T9  crash/stress sequences

This suite tests functional contracts. It deliberately avoids fixed NPS
thresholds, exact diagnostic wording, and other machine-dependent assertions.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "utils"))

from repo_paths import (  # noqa: E402
    default_opening_book,
    engine_executable,
    latest_version,
    tablebase_root,
)

# ═══════════════ CONFIGURATION ═══════════════
VERSION = ""
ENGINE_PATH = ""
SYZYGY_PATH = str(tablebase_root())
BOOK_PATH = str(default_opening_book() or "")
ONLY: list[str] = []
HANDSHAKE_TIMEOUT_S = 5.0
SEARCH_TIMEOUT_S = 20.0
# ═════════════════════════════════════════════

REQUIRED_OPTIONS = {
    "Hash", "Threads", "MultiPV", "SyzygyPath", "OwnBook", "BookFile",
}

PASSED = 0
FAILED = 0
SKIPPED = 0
FAILURES: list[str] = []

def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS  {name}")
    else:
        FAILED += 1
        msg = f"  FAIL  {name}" + (f" — {detail}" if detail else "")
        print(msg)
        FAILURES.append(msg)

def skip(name: str, reason: str) -> None:
    global SKIPPED
    SKIPPED += 1
    print(f"  SKIP  {name} — {reason}")

class UCIEngine:
    def __init__(self, path: Path):
        self.path = path
        self.proc: subprocess.Popen[str] | None = None
        self.lines: list[str] = []
        self.lock = threading.Lock()
        self.reader: threading.Thread | None = None

    def start(self) -> None:
        self.proc = subprocess.Popen(
            [str(self.path)],
            cwd=self.path.parent,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            with self.lock:
                self.lines.append(line.rstrip("\r\n"))

    def send(self, command: str) -> None:
        assert self.proc and self.proc.stdin and self.proc.poll() is None
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def read_until(self, pattern: str, timeout: float) -> list[str]:
        rx = re.compile(pattern)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                for i, line in enumerate(self.lines):
                    if rx.search(line):
                        result = self.lines[: i + 1]
                        del self.lines[: i + 1]
                        return result
            time.sleep(0.02)
        with self.lock:
            result = list(self.lines)
            self.lines.clear()
        return result

    def read_after(self, delay: float = 0.25) -> list[str]:
        time.sleep(delay)
        with self.lock:
            result = list(self.lines)
            self.lines.clear()
        return result

    def sync(self) -> bool:
        self.send("isready")
        return any(line == "readyok" for line in self.read_until(r"^readyok$", HANDSHAKE_TIMEOUT_S))

    def clear(self) -> None:
        with self.lock:
            self.lines.clear()

    def quit(self) -> None:
        if not self.proc:
            return
        if self.proc.poll() is None:
            try:
                self.send("quit")
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()
                self.proc.wait(timeout=3)

def resolve_engine() -> Path:
    if ENGINE_PATH:
        path = Path(ENGINE_PATH)
        if not path.is_absolute():
            path = ROOT / path
        return path.resolve()
    return engine_executable(VERSION or latest_version()).resolve()

def handshake(eng: UCIEngine) -> tuple[list[str], set[str]]:
    eng.send("uci")
    lines = eng.read_until(r"^uciok$", HANDSHAKE_TIMEOUT_S)
    options = set()
    for line in lines:
        m = re.match(r"option name (.+?)(?: type |$)", line)
        if m:
            options.add(m.group(1).strip())
    return lines, options

def bestmove(eng: UCIEngine, go: str, *, timeout: float = SEARCH_TIMEOUT_S) -> list[str]:
    eng.send(go)
    return eng.read_until(r"^bestmove\b", timeout)

def t1(eng: UCIEngine, options: set[str]) -> None:
    print("\n=== T1: handshake ===")
    eng.send("uci")
    lines = eng.read_until(r"^uciok$", HANDSHAKE_TIMEOUT_S)
    check("id name Zchezz", any("id name Zchezz" in line for line in lines))
    check("id author", any(line.startswith("id author") for line in lines))
    check("uciok", any(line == "uciok" for line in lines))
    found = set()
    for line in lines:
        m = re.match(r"option name (.+?)(?: type |$)", line)
        if m:
            found.add(m.group(1).strip())
    missing = sorted(REQUIRED_OPTIONS - found)
    check("required UCI options", not missing, f"missing={missing}")
    check("isready -> readyok", eng.sync())
    eng.send("ucinewgame")
    check("ucinewgame preserves readiness", eng.sync())

def t2(eng: UCIEngine, options: set[str]) -> None:
    print("\n=== T2: position and go ===")
    scenarios = [
        ("startpos depth", "position startpos", "go depth 5"),
        ("startpos movetime", "position startpos", "go movetime 200"),
        ("startpos nodes", "position startpos", "go nodes 1000"),
        (
            "FEN depth",
            "position fen r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
            "go depth 5",
        ),
        ("move list", "position startpos moves e2e4 e7e5 g1f3", "go depth 5"),
    ]
    for label, position, go in scenarios:
        eng.send(position)
        lines = bestmove(eng, go)
        bm = [line for line in lines if line.startswith("bestmove ")]
        check(f"{label} -> bestmove", bool(bm))
        if bm:
            move = bm[-1].split()[1]
            check(f"{label} bestmove syntax", bool(re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", move)), move)

    eng.send("position startpos")
    eng.send("go infinite")
    time.sleep(0.5)
    eng.send("stop")
    lines = eng.read_until(r"^bestmove\b", SEARCH_TIMEOUT_S)
    check("go infinite + stop -> bestmove", any(line.startswith("bestmove ") for line in lines))

def _last_tbhits(lines: list[str]) -> int:
    for line in reversed(lines):
        m = re.search(r"\btbhits\s+(\d+)", line)
        if m:
            return int(m.group(1))
    return 0

def t3(eng: UCIEngine, options: set[str]) -> None:
    print("\n=== T3: Syzygy ===")
    path = Path(SYZYGY_PATH)
    if "SyzygyPath" not in options:
        skip("Syzygy group", "engine does not advertise SyzygyPath")
        return
    if not path.is_dir():
        skip("Syzygy group", f"tablebase directory not found: {path}")
        return

    eng.send(f"setoption name SyzygyPath value {path}")
    check("SyzygyPath accepted", eng.sync())

    eng.send("ucinewgame")
    eng.sync()
    eng.send("position fen 8/8/4k3/8/3p4/8/2R5/4K3 w - - 0 1")
    lines = bestmove(eng, "go depth 12", timeout=30)
    check("KRKP -> bestmove", any(line.startswith("bestmove ") for line in lines))
    hits = _last_tbhits(lines)
    check("KRKP functional probe", hits > 0, f"tbhits={hits}")

    eng.send("ucinewgame")
    eng.sync()
    eng.send("position fen 8/8/8/4k3/8/8/8/R3K3 w - - 0 1")
    lines = bestmove(eng, "go depth 10", timeout=30)
    check("KRK -> bestmove", any(line.startswith("bestmove ") for line in lines))

def t4(eng: UCIEngine, options: set[str]) -> None:
    print("\n=== T4: opening book ===")
    if "OwnBook" not in options:
        skip("opening-book group", "engine does not advertise OwnBook")
        return

    eng.send("setoption name OwnBook value false")
    check("OwnBook=false accepted", eng.sync())
    eng.send("position startpos")
    check("book disabled normal search", any(line.startswith("bestmove ") for line in bestmove(eng, "go depth 5")))

    book = Path(BOOK_PATH) if BOOK_PATH else None
    if not book or not book.is_file():
        skip("book-file probe", "no opening book file available")
        return
    if "BookFile" not in options:
        skip("book-file probe", "engine does not advertise BookFile")
        return

    eng.send(f"setoption name BookFile value {book}")
    eng.send("setoption name OwnBook value true")
    check("BookFile/OwnBook accepted", eng.sync())
    eng.send("position startpos")
    lines = bestmove(eng, "go depth 5")
    check("book-enabled startpos -> bestmove", any(line.startswith("bestmove ") for line in lines))
    eng.send("setoption name OwnBook value false")
    eng.sync()

def t5(eng: UCIEngine, options: set[str]) -> None:
    print("\n=== T5: MultiPV ===")
    if "MultiPV" not in options:
        check("MultiPV option advertised", False)
        return
    for count in (1, 2, 4):
        eng.send(f"setoption name MultiPV value {count}")
        check(f"MultiPV={count} accepted", eng.sync())
        eng.send("position startpos")
        lines = bestmove(eng, "go depth 7", timeout=30)
        check(f"MultiPV={count} -> bestmove", any(line.startswith("bestmove ") for line in lines))
        if count > 1:
            indices = {
                int(m.group(1))
                for line in lines
                if (m := re.search(r"\bmultipv\s+(\d+)", line))
            }
            check(f"MultiPV={count} emits indices 1..{count}", set(range(1, count + 1)) <= indices, str(sorted(indices)))
    eng.send("setoption name MultiPV value 1")
    eng.sync()

def t6(eng: UCIEngine, options: set[str]) -> None:
    print("\n=== T6: Threads ===")
    if "Threads" not in options:
        check("Threads option advertised", False)
        return
    for count in (1, 2, 4):
        eng.send(f"setoption name Threads value {count}")
        check(f"Threads={count} accepted", eng.sync())
        eng.send("position startpos")
        lines = bestmove(eng, "go depth 6", timeout=30)
        check(f"Threads={count} -> bestmove", any(line.startswith("bestmove ") for line in lines))
    eng.send("setoption name Threads value 2")
    eng.sync()
    eng.send("position startpos")
    eng.send("go infinite")
    time.sleep(0.75)
    eng.send("stop")
    lines = eng.read_until(r"^bestmove\b", 20)
    check("Threads=2 stop -> bestmove", any(line.startswith("bestmove ") for line in lines))
    eng.send("setoption name Threads value 1")
    eng.sync()

def t7(eng: UCIEngine, options: set[str]) -> None:
    print("\n=== T7: supported options ===")
    cases = [
        ("Contempt", "25"),
        ("MoveOverhead", "100"),
        ("Ponder", "true"),
        ("UCI_AnalyseMode", "true"),
    ]
    exercised = 0
    for name, value in cases:
        if name not in options:
            skip(name, "option not advertised by this engine")
            continue
        exercised += 1
        eng.send(f"setoption name {name} value {value}")
        check(f"{name} accepted", eng.sync())
    if exercised == 0:
        skip("supported-options group", "none of the optional cases are advertised")

def t8(eng: UCIEngine, options: set[str]) -> None:
    print("\n=== T8: extension commands ===")
    eng.clear()
    eng.send("bench 5")
    lines = eng.read_until(r"(?i)nodes/sec", 60)
    check("bench reports Nodes/sec", any(re.search(r"(?i)nodes/sec", line) for line in lines))

    eng.send("position startpos")
    eng.send("d")
    lines = eng.read_until(r"^Fen:", 5)
    check("d reports FEN", any(line.startswith("Fen:") for line in lines))

    eng.send("eval")
    lines = eng.read_after(0.5)
    check("eval produces output", bool(lines), f"lines={lines[:3]}")

def t9(eng: UCIEngine, options: set[str]) -> None:
    print("\n=== T9: stress ===")
    ok = True
    for _ in range(10):
        eng.send("position startpos moves e2e4 e7e5 g1f3")
        if not any(line.startswith("bestmove ") for line in bestmove(eng, "go depth 4", timeout=10)):
            ok = False
            break
    check("10 rapid position/search cycles", ok)

    long_moves = (
        "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 "
        "f1e1 b7b5 a4b3 d7d6 c2c3 e8g8 h2h3 c6b8 d2d4 b8d7"
    )
    eng.send(f"position startpos moves {long_moves}")
    check("long move list", any(line.startswith("bestmove ") for line in bestmove(eng, "go depth 5")))

    if "Threads" in options:
        for count in (4, 1, 4, 1):
            eng.send(f"setoption name Threads value {count}")
            if not eng.sync():
                check(f"thread switch to {count}", False)
                return
            eng.send("position startpos")
            lines = bestmove(eng, "go depth 5", timeout=20)
            check(f"thread switch {count} -> bestmove", any(line.startswith("bestmove ") for line in lines))

GROUPS = {
    "T1": t1, "T2": t2, "T3": t3, "T4": t4, "T5": t5,
    "T6": t6, "T7": t7, "T8": t8, "T9": t9,
}

def run_group(name: str, fn, engine_path: Path) -> None:
    global FAILED
    eng = UCIEngine(engine_path)
    try:
        eng.start()
        _, options = handshake(eng)
        if not eng.sync():
            check(f"{name} pre-test readiness", False)
            return
        fn(eng, options)
    except Exception as exc:
        FAILED += 1
        msg = f"  FAIL  {name} raised {type(exc).__name__}: {exc}"
        FAILURES.append(msg)
        print(msg)
    finally:
        eng.quit()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy_version", nargs="?", default="")
    parser.add_argument("--version", default=VERSION)
    parser.add_argument("--exe", default=ENGINE_PATH)
    parser.add_argument("--syzygy", default=SYZYGY_PATH)
    parser.add_argument("--book", default=BOOK_PATH)
    parser.add_argument("--only", action="append", default=list(ONLY))
    parser.add_argument("--handshake-timeout", type=float, default=HANDSHAKE_TIMEOUT_S)
    parser.add_argument("--search-timeout", type=float, default=SEARCH_TIMEOUT_S)
    return parser.parse_args()

def main() -> int:
    global VERSION, ENGINE_PATH, SYZYGY_PATH, BOOK_PATH
    global HANDSHAKE_TIMEOUT_S, SEARCH_TIMEOUT_S

    args = parse_args()
    VERSION = args.version or args.legacy_version or VERSION
    ENGINE_PATH = args.exe
    SYZYGY_PATH = args.syzygy
    BOOK_PATH = args.book
    HANDSHAKE_TIMEOUT_S = args.handshake_timeout
    SEARCH_TIMEOUT_S = args.search_timeout

    engine_path = resolve_engine()
    if not engine_path.is_file():
        print(f"ERROR: engine not found: {engine_path}")
        return 1

    wanted = [item.upper() for item in args.only] if args.only else list(GROUPS)
    unknown = sorted(set(wanted) - set(GROUPS))
    if unknown:
        print(f"ERROR: unknown test group(s): {unknown}")
        return 2

    print(f"Engine: {engine_path}")
    print(f"Syzygy: {SYZYGY_PATH}")
    print(f"Book: {BOOK_PATH or '(none)'}")

    for name in wanted:
        run_group(name, GROUPS[name], engine_path)

    total = PASSED + FAILED
    print("\n" + "=" * 60)
    print(f"Results: {PASSED}/{total} assertions passed, {FAILED} failed, {SKIPPED} skipped")
    if FAILURES:
        print("Failures:")
        for item in FAILURES:
            print(item)
    print("=" * 60)
    return 1 if FAILED else 0

if __name__ == "__main__":
    raise SystemExit(main())
