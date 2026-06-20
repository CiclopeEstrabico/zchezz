#!/usr/bin/env python3
"""match.py — Unified engine-vs-engine match runner.

Run 2+ engines head-to-head with proper paired openings, concurrent games,
live ELO output, and score adjudication.

Usage:
  # Auto-detect latest vs previous version (200 games)
  python tests/match.py

  # Two engines, 600 games, 15 workers
  python tests/match.py engine/c/zchezz_v309/zchezz.exe engine/c/zchezz_v306/zchezz.exe \\
      --names v309 v306 --games 600 --workers 15

  # With tablebases for engine A
  python tests/match.py ENGINE_A ENGINE_B --tb-a "C:\\tablebases\\3-4-5"

  # 3-engine round-robin
  python tests/match.py ENGINE_A ENGINE_B ENGINE_C --games 200

  # Quick smoke test
  python tests/match.py --games 20 --movetime 50
"""

import argparse
import glob
import math
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty

# Shared ELO calculator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elo_calc import elo_difference

try:
    import chess
    import chess.pgn
    HAS_CHESS = True
except ImportError:
    HAS_CHESS = False

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Adjudication defaults
ADJ_WIN_CP    = 1000   # centipawns to consider a win
ADJ_WIN_MOVES = 4      # consecutive moves above threshold
ADJ_DRAW_CP   = 5      # centipawns to consider a draw
ADJ_DRAW_MOVES = 8     # consecutive moves below threshold
ADJ_DRAW_MIN_PLY = 80  # minimum ply before draw adjudication


# ── Auto-detect engines ───────────────────────────────────────────────────────
def find_engine_versions():
    """Find all zchezz engine versions, sorted by version number."""
    pattern = os.path.join(BASE_DIR, "engine", "c", "zchezz_v*")
    dirs = sorted(glob.glob(pattern))
    results = []
    for d in dirs:
        exe = os.path.join(d, "zchezz.exe")
        if os.path.exists(exe):
            name = os.path.basename(d).replace("zchezz_", "")
            results.append({"path": exe, "name": name, "dir": d})
    return results


def find_nnue(engine_path):
    """Find NNUE weights in the same directory as the engine."""
    d = os.path.dirname(os.path.abspath(engine_path))
    nnue = os.path.join(d, "nnue_weights.bin")
    return nnue if os.path.exists(nnue) else None


# ── Opening loader ────────────────────────────────────────────────────────────
def load_openings_pgn(pgn_path, max_n=5000):
    """Load openings from PGN file. Returns list of FEN strings."""
    if not HAS_CHESS or not os.path.exists(pgn_path):
        return []
    openings = []
    with open(pgn_path, "r", errors="ignore") as f:
        while len(openings) < max_n:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            board = g.board()
            for m in g.mainline_moves():
                board.push(m)
            openings.append(board.fen())
    return openings


def find_openings():
    """Find the best opening book available."""
    candidates = [
        os.path.join(BASE_DIR, "openings", "8moves_v3.pgn"),
        os.path.join(BASE_DIR, "openings", "Blitz_Testing_4moves.pgn"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return load_openings_pgn(c)
    # Search for any PGN in openings/
    pgns = glob.glob(os.path.join(BASE_DIR, "openings", "*.pgn"))
    if pgns:
        return load_openings_pgn(pgns[0])
    return []


# ── UCI Engine ─────────────────────────────────────────────────────────────────
class UCIEngine:
    """Persistent UCI engine process."""

    def __init__(self, path, options=None, name=""):
        self.name = name or os.path.basename(path)
        self.path = os.path.abspath(path)
        self.options = options or {}
        self._queue = Queue()
        self._alive = True
        self._start_fen = None

        cwd = os.path.dirname(self.path)
        try:
            self._proc = subprocess.Popen(
                [self.path],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1, cwd=cwd,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)
            )
        except Exception as e:
            raise RuntimeError(f"Failed to start {self.name}: {e}")

        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=lambda: self._proc.stderr.read(), daemon=True).start()

        self._send("uci")
        self._wait("uciok", 10)
        for k, v in self.options.items():
            self._send(f"setoption name {k} value {v}")
        self._send("isready")
        self._wait("readyok", 30)

    def _reader(self):
        try:
            for line in self._proc.stdout:
                self._queue.put(line.strip())
        except Exception:
            self._alive = False

    def _send(self, cmd):
        try:
            self._proc.stdin.write(cmd + "\n")
            self._proc.stdin.flush()
        except Exception:
            self._alive = False

    def _wait(self, keyword, timeout=10):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                line = self._queue.get(timeout=0.05)
                if keyword in line:
                    return line
            except Empty:
                pass
        return None

    def _drain(self):
        """Drain pending output."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Empty:
                break

    def new_game(self):
        self._send("ucinewgame")
        self._send("isready")
        self._wait("readyok", 10)

    def go(self, fen, moves_uci, movetime_ms):
        """Send position + go movetime. Returns (bestmove, score_cp)."""
        self._drain()

        if moves_uci:
            self._send(f"position fen {fen} moves {moves_uci}")
        else:
            self._send(f"position fen {fen}")
        self._send(f"go movetime {movetime_ms}")

        bestmove = None
        score = 0
        is_mate = False
        t0 = time.time()
        timeout = max(30, movetime_ms / 1000 + 10)

        while time.time() - t0 < timeout:
            try:
                line = self._queue.get(timeout=0.1)
                if "score cp " in line:
                    try:
                        idx = line.index("score cp ") + 9
                        score = int(line[idx:].split()[0])
                        is_mate = False
                    except (ValueError, IndexError):
                        pass
                elif "score mate " in line:
                    try:
                        idx = line.index("score mate ") + 11
                        mate_in = int(line[idx:].split()[0])
                        score = 30000 if mate_in > 0 else -30000
                        is_mate = True
                    except (ValueError, IndexError):
                        pass
                if line.startswith("bestmove"):
                    parts = line.split()
                    bestmove = parts[1] if len(parts) > 1 else None
                    break
            except Empty:
                if not self.is_alive():
                    break

        return bestmove, score

    def is_alive(self):
        return self._alive and self._proc.poll() is None

    def quit(self):
        try:
            self._send("quit")
            self._proc.wait(timeout=2)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass


# ── Game logic ─────────────────────────────────────────────────────────────────
def play_game(eng_w, eng_b, opening_fen, movetime_ms, max_plies=400,
              adjudicate=True):
    """Play one game. Returns (result, ply_count).
    result: 1.0 = white wins, 0.0 = black wins, 0.5 = draw.
    """
    if not HAS_CHESS:
        raise ImportError("python-chess required")

    board = chess.Board(opening_fen) if opening_fen else chess.Board()
    eng_w._start_fen = board.fen()
    eng_b._start_fen = board.fen()
    eng_w.new_game()
    eng_b.new_game()

    start_fen = board.fen()
    moves_played = []

    # Adjudication counters
    win_w_count = 0
    win_b_count = 0
    draw_count = 0

    for ply in range(max_plies):
        is_white = board.turn == chess.WHITE
        eng = eng_w if is_white else eng_b

        moves_str = " ".join(moves_played) if moves_played else ""
        bestmove, score_cp = eng.go(start_fen, moves_str, movetime_ms)

        if not bestmove or bestmove in ("(none)", "0000", "none", ""):
            return (0.0 if is_white else 1.0), ply

        try:
            move = chess.Move.from_uci(bestmove)
            if move not in board.legal_moves:
                # Try partial match
                for lm in board.legal_moves:
                    if lm.uci()[:4] == bestmove[:4]:
                        move = lm
                        break
                else:
                    return (0.0 if is_white else 1.0), ply
            board.push(move)
            moves_played.append(move.uci())
        except Exception:
            return (0.0 if is_white else 1.0), ply

        # Check game over
        if board.is_game_over(claim_draw=True):
            r = board.result(claim_draw=True)
            if r == "1-0":
                return 1.0, ply + 1
            elif r == "0-1":
                return 0.0, ply + 1
            else:
                return 0.5, ply + 1

        # Adjudication
        if adjudicate:
            # Convert to white's perspective
            w_score = score_cp if is_white else -score_cp

            if w_score >= ADJ_WIN_CP:
                win_w_count += 1
                win_b_count = 0
                draw_count = 0
            elif w_score <= -ADJ_WIN_CP:
                win_b_count += 1
                win_w_count = 0
                draw_count = 0
            elif abs(w_score) <= ADJ_DRAW_CP:
                draw_count += 1
                win_w_count = 0
                win_b_count = 0
            else:
                win_w_count = 0
                win_b_count = 0
                draw_count = 0

            if win_w_count >= ADJ_WIN_MOVES:
                return 1.0, ply + 1
            if win_b_count >= ADJ_WIN_MOVES:
                return 0.0, ply + 1
            if draw_count >= ADJ_DRAW_MOVES * 2 and ply >= ADJ_DRAW_MIN_PLY:
                return 0.5, ply + 1

    return 0.5, max_plies


# ── Match runner ───────────────────────────────────────────────────────────────
def run_match(eng_a_cfg, eng_b_cfg, openings, n_games, workers, movetime_ms,
              max_plies=400, adjudicate=True):
    """Run a head-to-head match with paired openings and concurrent games."""
    a_name = eng_a_cfg["name"]
    b_name = eng_b_cfg["name"]
    n_pairs = n_games // 2

    print(f"\n{'=' * 70}")
    print(f"  {a_name} vs {b_name}")
    print(f"  {n_games} games ({n_pairs} paired openings) | "
          f"{movetime_ms}ms/move | {workers} workers")
    print(f"{'=' * 70}")

    # Build game schedule: paired openings, swapped colors
    games = []
    for i in range(n_pairs):
        fen = openings[i % len(openings)] if openings else STARTPOS
        games.append({"id": i * 2, "fen": fen, "a_white": True})
        games.append({"id": i * 2 + 1, "fen": fen, "a_white": False})

    random.shuffle(games)

    w, d, l = 0, 0, 0
    done = 0
    lock = threading.Lock()
    t0 = time.time()

    def play_one(game_info):
        """Worker: create engines, play game, return result."""
        try:
            eng_w_cfg = eng_a_cfg if game_info["a_white"] else eng_b_cfg
            eng_b_cfg2 = eng_b_cfg if game_info["a_white"] else eng_a_cfg

            eng_w = UCIEngine(eng_w_cfg["path"], eng_w_cfg["options"],
                              eng_w_cfg["name"])
            eng_b = UCIEngine(eng_b_cfg2["path"], eng_b_cfg2["options"],
                              eng_b_cfg2["name"])

            result, plies = play_game(eng_w, eng_b, game_info["fen"],
                                      movetime_ms, max_plies, adjudicate)

            eng_w.quit()
            eng_b.quit()

            # Convert to A's perspective
            if game_info["a_white"]:
                a_score = result
            else:
                a_score = 1.0 - result

            return a_score, plies
        except Exception as e:
            return 0.5, 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(play_one, g): g for g in games}

        for f in as_completed(futures):
            a_score, plies = f.result()
            with lock:
                if a_score == 1.0:
                    w += 1
                elif a_score == 0.0:
                    l += 1
                else:
                    d += 1
                done += 1

                if done % 10 == 0 or done == n_games:
                    elo, ci, _ = elo_difference(w, d, l)
                    elapsed = time.time() - t0
                    gps = done / elapsed if elapsed > 0 else 0
                    pct = (w + d * 0.5) / done * 100
                    print(f"  [{done:3d}/{n_games}] +{w} ={d} -{l} "
                          f"({pct:.1f}%) | ELO: {elo:+.0f} ±{ci:.0f} "
                          f"| {gps:.1f} g/s")

    elapsed = time.time() - t0
    elo, ci, _ = elo_difference(w, d, l)
    pct = (w + d * 0.5) / (w + d + l) * 100 if (w + d + l) > 0 else 50

    print(f"\n{'=' * 70}")
    print(f"  FINAL: {a_name} vs {b_name}")
    print(f"  Score: +{w} ={d} -{l} ({pct:.1f}%)")
    print(f"  ELO: {a_name} is {elo:+.0f} ±{ci:.0f} vs {b_name}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'=' * 70}")

    return {"a": a_name, "b": b_name, "w": w, "d": d, "l": l,
            "elo": elo, "ci": ci}


def run_round_robin(engine_cfgs, openings, games_per_pair, workers,
                    movetime_ms, max_plies=400):
    """Run round-robin between 3+ engines."""
    results = {}
    pairs = [(i, j) for i in range(len(engine_cfgs))
             for j in range(i + 1, len(engine_cfgs))]

    for idx, (i, j) in enumerate(pairs):
        print(f"\n  Match {idx + 1}/{len(pairs)}")
        r = run_match(engine_cfgs[i], engine_cfgs[j], openings,
                      games_per_pair, workers, movetime_ms, max_plies)
        results[(i, j)] = r

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  ROUND-ROBIN SUMMARY")
    print(f"{'=' * 70}")
    for (i, j), r in results.items():
        print(f"  {r['a']:12s} vs {r['b']:12s}: "
              f"+{r['w']} ={r['d']} -{r['l']} | "
              f"ELO: {r['elo']:+.0f} ±{r['ci']:.0f}")
    print(f"{'=' * 70}")

    return results


# ── CLI ────────────────────────────────────────────────────────────────────────
def build_engine_cfg(path, name=None, tb_path=None, threads=1, extra_opts=None):
    """Build engine config dict."""
    abs_path = os.path.abspath(path)
    nnue = find_nnue(abs_path)
    opts = {}
    if nnue:
        opts["NNUE"] = nnue
    if threads > 1:
        opts["Threads"] = str(threads)
    if tb_path:
        opts["SyzygyPath"] = tb_path
    if extra_opts:
        for kv in extra_opts:
            k, v = kv.split("=", 1)
            opts[k] = v

    return {
        "path": abs_path,
        "name": name or os.path.basename(os.path.dirname(abs_path)),
        "options": opts,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Unified engine-vs-engine match runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("engines", nargs="*",
                        help="Paths to engine executables (2+ for round-robin)")
    parser.add_argument("--names", nargs="*",
                        help="Names for each engine (same order as paths)")
    parser.add_argument("--games", type=int, default=200,
                        help="Total games (default: 200)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent games (default: 8)")
    parser.add_argument("--movetime", type=int, default=200,
                        help="Milliseconds per move (default: 200)")
    parser.add_argument("--max-plies", type=int, default=400,
                        help="Max plies per game (default: 400)")
    parser.add_argument("--openings",
                        help="Path to openings PGN file")
    parser.add_argument("--tb-a",
                        help="SyzygyPath for engine A")
    parser.add_argument("--tb-b",
                        help="SyzygyPath for engine B")
    parser.add_argument("--threads", type=int, default=1,
                        help="Threads per engine (default: 1)")
    parser.add_argument("--options-a", nargs="*",
                        help="Extra UCI options for A (key=value)")
    parser.add_argument("--options-b", nargs="*",
                        help="Extra UCI options for B (key=value)")
    parser.add_argument("--no-adjudicate", action="store_true",
                        help="Disable score adjudication")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()
    random.seed(args.seed)

    # Auto-detect engines if none specified
    if not args.engines:
        versions = find_engine_versions()
        if len(versions) < 2:
            print("ERROR: Need at least 2 engines. Specify paths or have "
                  "2+ versions in engine/c/")
            sys.exit(1)
        # Latest vs previous
        latest = versions[-1]
        prev = versions[-2]
        args.engines = [latest["path"], prev["path"]]
        if not args.names:
            args.names = [latest["name"], prev["name"]]
        print(f"  Auto-detected: {latest['name']} vs {prev['name']}")

    # Build engine configs
    names = args.names or [None] * len(args.engines)
    if len(names) < len(args.engines):
        names.extend([None] * (len(args.engines) - len(names)))

    engine_cfgs = []
    for i, (path, name) in enumerate(zip(args.engines, names)):
        tb = args.tb_a if i == 0 else (args.tb_b if i == 1 else None)
        extra = args.options_a if i == 0 else (args.options_b if i == 1 else None)
        cfg = build_engine_cfg(path, name, tb, args.threads, extra)
        engine_cfgs.append(cfg)

    # Load openings
    if args.openings:
        openings = load_openings_pgn(args.openings)
        print(f"  Loaded {len(openings)} openings from {os.path.basename(args.openings)}")
    else:
        openings = find_openings()
        if openings:
            print(f"  Auto-loaded {len(openings)} openings")
        else:
            print("  WARNING: No openings found, using startpos")
            openings = [STARTPOS]

    # Make games even
    if args.games % 2 != 0:
        args.games -= 1

    # Run
    if len(engine_cfgs) == 2:
        run_match(engine_cfgs[0], engine_cfgs[1], openings, args.games,
                  args.workers, args.movetime, args.max_plies,
                  not args.no_adjudicate)
    else:
        run_round_robin(engine_cfgs, openings, args.games, args.workers,
                        args.movetime, args.max_plies)


if __name__ == "__main__":
    main()
