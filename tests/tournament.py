#!/usr/bin/env python3
"""tournament.py — Estimate absolute ELO vs Stockfish anchors.

Play the engine against Stockfish at known strength levels to estimate
its absolute ELO rating with proper confidence intervals.

Usage:
  # Latest engine vs SF-2800 (600 games, 15 workers)
  python tests/tournament.py --anchors 2800 --games 600 --workers 15

  # Multiple anchors
  python tests/tournament.py --anchors 2800 2900 --games 300

  # With tablebases
  python tests/tournament.py --anchors 2800 --games 600 --tb "C:\\tablebases\\3-4-5"

  # Custom engine and Stockfish path
  python tests/tournament.py engine/c/zchezz_v309/zchezz.exe \\
      --anchors 2800 --sf engine/stockfish/stockfish.exe
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

# Shared modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elo_calc import elo_difference, estimated_elo

try:
    import chess
    import chess.pgn
    HAS_CHESS = True
except ImportError:
    HAS_CHESS = False

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


# ── Auto-detect ───────────────────────────────────────────────────────────────
def find_latest_engine():
    """Find the latest zchezz engine version."""
    pattern = os.path.join(BASE_DIR, "engine", "c", "zchezz_v*")
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        return None
    latest = dirs[-1]
    exe = os.path.join(latest, "zchezz.exe")
    name = os.path.basename(latest).replace("zchezz_", "")
    nnue = os.path.join(latest, "nnue_weights.bin")
    return {"path": exe, "name": name, "nnue": nnue}


def find_stockfish():
    """Find Stockfish executable."""
    candidates = [
        os.path.join(BASE_DIR, "engine", "stockfish", "stockfish.exe"),
        os.path.join(BASE_DIR, "engine", "stockfish_fast", "stockfish.exe"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def find_openings():
    """Find the best opening book available."""
    candidates = [
        os.path.join(BASE_DIR, "openings", "8moves_v3.pgn"),
        os.path.join(BASE_DIR, "openings", "Blitz_Testing_4moves.pgn"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    pgns = glob.glob(os.path.join(BASE_DIR, "openings", "*.pgn"))
    return pgns[0] if pgns else None


def load_openings_pgn(pgn_path, max_n=5000):
    """Load openings from PGN as FEN list."""
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


# ── UCI Engine ─────────────────────────────────────────────────────────────────
class UCIEngine:
    """Persistent UCI engine process."""

    def __init__(self, path, options=None, name=""):
        self.name = name or os.path.basename(path)
        self.path = os.path.abspath(path)
        self.options = options or {}
        self._queue = Queue()
        self._alive = True

        cwd = os.path.dirname(self.path)
        self._proc = subprocess.Popen(
            [self.path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, cwd=cwd,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        )
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
        self._drain()
        if moves_uci:
            self._send(f"position fen {fen} moves {moves_uci}")
        else:
            self._send(f"position fen {fen}")
        self._send(f"go movetime {movetime_ms}")

        bestmove = None
        score = 0
        t0 = time.time()
        timeout = max(30, movetime_ms / 1000 + 10)

        while time.time() - t0 < timeout:
            try:
                line = self._queue.get(timeout=0.1)
                if "score cp " in line:
                    try:
                        idx = line.index("score cp ") + 9
                        score = int(line[idx:].split()[0])
                    except (ValueError, IndexError):
                        pass
                elif "score mate " in line:
                    try:
                        idx = line.index("score mate ") + 11
                        mate_in = int(line[idx:].split()[0])
                        score = 30000 if mate_in > 0 else -30000
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
ADJ_WIN_CP = 1000
ADJ_WIN_MOVES = 4
ADJ_DRAW_CP = 5
ADJ_DRAW_MOVES = 8
ADJ_DRAW_MIN_PLY = 80


def play_game(eng_w, eng_b, opening_fen, movetime_ms, max_plies=400):
    """Play one game. Returns 1.0/0.5/0.0 from white's perspective."""
    if not HAS_CHESS:
        raise ImportError("python-chess required")

    board = chess.Board(opening_fen) if opening_fen else chess.Board()
    start_fen = board.fen()
    eng_w.new_game()
    eng_b.new_game()
    moves_played = []
    win_w, win_b, draw_streak = 0, 0, 0

    for ply in range(max_plies):
        is_white = board.turn == chess.WHITE
        eng = eng_w if is_white else eng_b
        moves_str = " ".join(moves_played) if moves_played else ""
        bestmove, score_cp = eng.go(start_fen, moves_str, movetime_ms)

        if not bestmove or bestmove in ("(none)", "0000", ""):
            return 0.0 if is_white else 1.0

        try:
            move = chess.Move.from_uci(bestmove)
            if move not in board.legal_moves:
                for lm in board.legal_moves:
                    if lm.uci()[:4] == bestmove[:4]:
                        move = lm
                        break
                else:
                    return 0.0 if is_white else 1.0
            board.push(move)
            moves_played.append(move.uci())
        except Exception:
            return 0.0 if is_white else 1.0

        if board.is_game_over(claim_draw=True):
            r = board.result(claim_draw=True)
            if r == "1-0":
                return 1.0
            elif r == "0-1":
                return 0.0
            return 0.5

        # Adjudication
        w_score = score_cp if is_white else -score_cp
        if w_score >= ADJ_WIN_CP:
            win_w += 1; win_b = 0; draw_streak = 0
        elif w_score <= -ADJ_WIN_CP:
            win_b += 1; win_w = 0; draw_streak = 0
        elif abs(w_score) <= ADJ_DRAW_CP:
            draw_streak += 1; win_w = 0; win_b = 0
        else:
            win_w = win_b = draw_streak = 0

        if win_w >= ADJ_WIN_MOVES:
            return 1.0
        if win_b >= ADJ_WIN_MOVES:
            return 0.0
        if draw_streak >= ADJ_DRAW_MOVES * 2 and ply >= ADJ_DRAW_MIN_PLY:
            return 0.5

    return 0.5


# ── Tournament runner ──────────────────────────────────────────────────────────
def run_anchor_match(engine_cfg, sf_path, anchor_elo, openings,
                     n_games, workers, movetime_ms, max_plies):
    """Run one anchor match. Returns W/D/L dict."""
    engine_name = engine_cfg["name"]
    sf_label = f"SF-{anchor_elo}"
    sf_opts = {"UCI_LimitStrength": "true", "UCI_Elo": str(anchor_elo),
               "Threads": "1"}
    n_pairs = n_games // 2

    print(f"\n  === {engine_name} vs {sf_label} ({n_games} games, "
          f"{workers} workers) ===")

    w, d, l = 0, 0, 0
    done = 0
    lock = threading.Lock()
    t0 = time.time()

    games = []
    for i in range(n_pairs):
        fen = openings[i % len(openings)] if openings else STARTPOS
        games.append({"fen": fen, "zchezz_white": True})
        games.append({"fen": fen, "zchezz_white": False})
    random.shuffle(games)

    def play_one(game_info):
        try:
            eng_z = UCIEngine(engine_cfg["path"], engine_cfg["options"],
                              engine_name)
            eng_s = UCIEngine(sf_path, sf_opts, sf_label)

            if game_info["zchezz_white"]:
                result = play_game(eng_z, eng_s, game_info["fen"],
                                   movetime_ms, max_plies)
                z_score = result
            else:
                result = play_game(eng_s, eng_z, game_info["fen"],
                                   movetime_ms, max_plies)
                z_score = 1.0 - result

            eng_z.quit()
            eng_s.quit()
            return z_score
        except Exception:
            return 0.5

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(play_one, g): g for g in games}

        for f in as_completed(futures):
            z_score = f.result()
            with lock:
                if z_score == 1.0:
                    w += 1
                elif z_score == 0.0:
                    l += 1
                else:
                    d += 1
                done += 1

                if done % 10 == 0 or done == n_games:
                    elo_diff, ci, _ = elo_difference(w, d, l)
                    est = anchor_elo + elo_diff
                    elapsed = time.time() - t0
                    gps = done / elapsed if elapsed > 0 else 0
                    w_label = engine_name if game_info.get("zchezz_white") else sf_label
                    b_label = sf_label if game_info.get("zchezz_white") else engine_name
                    print(f"  [{done:3d}/{n_games}] {w_label:15s} vs {b_label:15s} "
                          f"| +{w} ={d} -{l} | ELO: {est:.0f} ±{ci:.0f} "
                          f"| {gps:.1f} g/s")

    elo_diff, ci, _ = elo_difference(w, d, l)
    est = anchor_elo + elo_diff
    pct = (w + d * 0.5) / (w + d + l) * 100 if (w + d + l) > 0 else 50

    print(f"\n  {engine_name} vs {sf_label} FINAL: +{w} ={d} -{l} ({pct:.1f}%)")
    print(f"  ESTIMATED ELO: {est:.0f} ±{ci:.0f}")

    return {"w": w, "d": d, "l": l, "anchor_elo": anchor_elo,
            "elo_diff": elo_diff, "ci": ci, "estimated": est}


def main():
    parser = argparse.ArgumentParser(
        description="Estimate ELO vs Stockfish anchors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("engine", nargs="?",
                        help="Path to engine exe (auto-detects latest if omitted)")
    parser.add_argument("--name",
                        help="Engine name")
    parser.add_argument("--anchors", nargs="+", type=int, default=[2800],
                        help="Stockfish ELO anchors (default: 2800)")
    parser.add_argument("--games", type=int, default=600,
                        help="Games per anchor (default: 600)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent games (default: 8)")
    parser.add_argument("--movetime", type=int, default=200,
                        help="ms per move (default: 200)")
    parser.add_argument("--max-plies", type=int, default=400,
                        help="Max plies per game (default: 400)")
    parser.add_argument("--sf",
                        help="Path to Stockfish exe")
    parser.add_argument("--tb",
                        help="SyzygyPath for the engine")
    parser.add_argument("--openings",
                        help="Path to openings PGN")
    parser.add_argument("--threads", type=int, default=1,
                        help="Engine threads (default: 1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()
    random.seed(args.seed)

    # Auto-detect engine
    if args.engine:
        engine_path = os.path.abspath(args.engine)
        engine_name = args.name or os.path.basename(os.path.dirname(engine_path))
        nnue_path = os.path.join(os.path.dirname(engine_path), "nnue_weights.bin")
    else:
        info = find_latest_engine()
        if not info:
            print("ERROR: No engine found. Specify path.")
            sys.exit(1)
        engine_path = info["path"]
        engine_name = args.name or info["name"]
        nnue_path = info["nnue"]

    # Build engine options
    engine_opts = {}
    if os.path.exists(nnue_path):
        engine_opts["NNUE"] = os.path.abspath(nnue_path)
    if args.threads > 1:
        engine_opts["Threads"] = str(args.threads)
    if args.tb:
        engine_opts["SyzygyPath"] = args.tb

    engine_cfg = {"path": engine_path, "name": engine_name,
                  "options": engine_opts}

    # Find Stockfish
    sf_path = args.sf or find_stockfish()
    if not sf_path or not os.path.exists(sf_path):
        print("ERROR: Stockfish not found. Use --sf path/to/stockfish.exe")
        sys.exit(1)

    # Load openings
    openings_pgn = args.openings or find_openings()
    openings = load_openings_pgn(openings_pgn) if openings_pgn else []
    if openings:
        print(f"  Loaded {len(openings)} openings")
    else:
        openings = [STARTPOS]
        print("  WARNING: No openings found, using startpos")

    # Make games even
    if args.games % 2 != 0:
        args.games -= 1

    # Header
    tb_str = f" +TB({args.tb})" if args.tb else ""
    print(f"\n{'=' * 70}")
    print(f"  ELO Estimation: {engine_name}{tb_str}")
    print(f"  Anchors: {', '.join(f'SF-{a}' for a in args.anchors)}")
    print(f"  {args.games} games/anchor | {args.movetime}ms/move | "
          f"{args.workers} workers")
    print(f"{'=' * 70}")

    # Run each anchor
    all_results = {}
    for anchor in args.anchors:
        r = run_anchor_match(engine_cfg, sf_path, anchor, openings,
                             args.games, args.workers, args.movetime,
                             args.max_plies)
        all_results[anchor] = r

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"  FINAL ELO ESTIMATION: {engine_name}{tb_str}")
    print(f"{'=' * 70}")

    weighted_elo = 0.0
    combined_var = 0.0
    for anchor, r in all_results.items():
        est, ci = estimated_elo(r["w"], r["d"], r["l"], anchor)
        se = ci / 1.96 if ci < float('inf') and ci > 0 else float('inf')
        total = r["w"] + r["d"] + r["l"]
        pct = (r["w"] + r["d"] * 0.5) / total * 100 if total > 0 else 50
        print(f"  vs SF-{anchor}: +{r['w']} ={r['d']} -{r['l']} ({pct:.1f}%) "
              f"| Est: {est:.0f} ±{ci:.0f}")

        if se > 0 and se < float('inf'):
            weight = 1.0 / se ** 2
            weighted_elo += est * weight
            combined_var += weight

    if combined_var > 0:
        final_elo = weighted_elo / combined_var
        final_ci = 1.96 / math.sqrt(combined_var)
        print(f"\n  >>> ESTIMATED ELO: {final_elo:.0f} ±{final_ci:.0f} <<<")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
