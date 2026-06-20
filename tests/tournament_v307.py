#!/usr/bin/env python3
"""Test: v307+TB (WDL probe depth=1) vs v307 (no TB) — only 3 concurrent games."""

import subprocess, threading, time, os, sys, math, random
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, as_completed
import chess, chess.pgn

MOVETIME = 100
MAX_PLIES = 400
GAMES = 200
WORKERS = 3   # Low concurrency to minimize disk I/O contention

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OPENINGS = os.path.join(BASE, "openings", "8moves_v3.pgn")


def load_openings(max_n=100):
    openings = []
    if not os.path.exists(OPENINGS):
        return openings
    with open(OPENINGS, "r") as f:
        while len(openings) < max_n:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            b = g.board()
            moves = list(g.mainline_moves())
            for m in moves:
                b.push(m)
            openings.append({"fen": b.fen(), "moves": " ".join(m.uci() for m in moves)})
    random.seed(42)
    random.shuffle(openings)
    return openings

OPENING_LIST = load_openings()
print(f"  Loaded {len(OPENING_LIST)} openings")


class UCIEngine:
    def __init__(self, path, options=None, name=""):
        self.name = name
        self.queue = Queue()
        self.start_fen = None
        cwd = os.path.dirname(os.path.abspath(path))
        self.p = subprocess.Popen(
            [os.path.abspath(path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, cwd=cwd,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        threading.Thread(target=lambda: self.p.stderr.read(), daemon=True).start()
        threading.Thread(target=self._reader, daemon=True).start()
        self._send("uci")
        self._wait("uciok", 10)
        for k, v in (options or {}).items():
            self._send(f"setoption name {k} value {v}")
        self._send("isready")
        self._wait("readyok", 10)

    def _reader(self):
        try:
            for line in self.p.stdout:
                self.queue.put(line.strip())
        except:
            pass

    def _send(self, cmd):
        try:
            self.p.stdin.write(cmd + "\n")
            self.p.stdin.flush()
        except:
            pass

    def _wait(self, keyword, timeout):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                line = self.queue.get(timeout=0.1)
                if keyword in line:
                    return line
            except Empty:
                pass
        return None

    def get_move(self, board):
        self._send("stop")
        t0 = time.time()
        while time.time() - t0 < 0.5:
            try:
                line = self.queue.get(timeout=0.05)
                if line.startswith("bestmove"):
                    break
            except Empty:
                break
        self._send("isready")
        t0 = time.time()
        while time.time() - t0 < 10:
            try:
                line = self.queue.get(timeout=0.1)
                if "readyok" in line:
                    break
            except Empty:
                pass
        while not self.queue.empty():
            try: self.queue.get_nowait()
            except Empty: break

        m_list = " ".join([m.uci() for m in board.move_stack])
        if self.start_fen:
            pos_cmd = f"position fen {self.start_fen}" + (f" moves {m_list}" if m_list else "")
        else:
            pos_cmd = "position startpos" + (f" moves {m_list}" if m_list else "")
        self._send(pos_cmd)
        self._send(f"go movetime {MOVETIME}")

        t0 = time.time()
        while time.time() - t0 < 30:
            try:
                line = self.queue.get(timeout=0.1)
                if line.startswith("bestmove"):
                    parts = line.split()
                    return parts[1] if len(parts) > 1 else None
            except Empty:
                pass
        return None

    def quit(self):
        try:
            self._send("quit")
            self.p.wait(timeout=2)
        except:
            try: self.p.kill()
            except: pass


def play_game(white_cfg, black_cfg, opening):
    try:
        w_eng = UCIEngine(white_cfg["path"], white_cfg["options"], white_cfg.get("name", "W"))
        b_eng = UCIEngine(black_cfg["path"], black_cfg["options"], black_cfg.get("name", "B"))
    except Exception:
        return 0.5, 0

    board = chess.Board()
    if opening:
        board = chess.Board(opening["fen"])
        w_eng.start_fen = opening["fen"]
        b_eng.start_fen = opening["fen"]

    ply = 0
    result = None
    while not board.is_game_over(claim_draw=True) and ply < MAX_PLIES:
        eng = w_eng if board.turn == chess.WHITE else b_eng
        mv_str = eng.get_move(board)
        if not mv_str:
            result = 0.0 if board.turn == chess.WHITE else 1.0
            break
        try:
            move = board.parse_uci(mv_str)
            if move not in board.legal_moves:
                result = 0.0 if board.turn == chess.WHITE else 1.0
                break
            board.push(move)
            ply += 1
        except:
            result = 0.0 if board.turn == chess.WHITE else 1.0
            break

    if result is None:
        if board.is_game_over(claim_draw=True):
            r = board.result(claim_draw=True)
            if r == "1-0": result = 1.0
            elif r == "0-1": result = 0.0
            else: result = 0.5
        else:
            result = 0.5

    w_eng.quit()
    b_eng.quit()
    return result, ply


def main():
    random.seed(42)

    NNUE = os.path.join(BASE, "engine", "c", "zchezz_v307", "nnue_weights.bin")
    TB   = os.path.join(BASE, "tablebases")

    # v307+TB with WDL probing enabled via UCI (SyzygyProbeDepth=1)
    a_cfg = {
        "name": "v307+TB",
        "path": os.path.join(BASE, "engine", "c", "zchezz_v307", "zchezz.exe"),
        "options": {"NNUE": NNUE, "SyzygyPath": TB, "SyzygyProbeDepth": "1"},
    }
    b_cfg = {
        "name": "v307",
        "path": os.path.join(BASE, "engine", "c", "zchezz_v307", "zchezz.exe"),
        "options": {"NNUE": NNUE},
    }

    a_name, b_name = a_cfg["name"], b_cfg["name"]
    print(f"\n{'=' * 60}")
    print(f"  {a_name} vs {b_name} ({GAMES} games)")
    print(f"  Movetime: {MOVETIME}ms | Workers: {WORKERS} (low I/O contention)")
    print(f"  TB in-tree WDL probe enabled via UCI (SyzygyProbeDepth=1)")
    print(f"  Root DTZ probe DISABLED in code")
    print(f"{'=' * 60}")

    n_openings = min(GAMES // 2, len(OPENING_LIST))
    games = []
    for i in range(n_openings):
        opening = OPENING_LIST[i]
        games.append((a_cfg, b_cfg, opening, a_name, b_name))
        games.append((b_cfg, a_cfg, opening, b_name, a_name))

    w, d, l = 0, 0, 0
    game_num = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {}
        for g_info in games:
            f = pool.submit(play_game, g_info[0], g_info[1], g_info[2])
            futures[f] = g_info

        for f in as_completed(futures):
            g_info = futures[f]
            w_name, b_name_g = g_info[3], g_info[4]
            try:
                score_w, plies = f.result()
            except:
                score_w, plies = 0.5, 0

            game_num += 1
            if w_name == a_name:
                a_score = score_w
            else:
                a_score = 1.0 - score_w

            if a_score == 1.0:
                w += 1; res_str = f"{a_name} WIN"
            elif a_score == 0.0:
                l += 1; res_str = f"{b_name} WIN"
            else:
                d += 1; res_str = "DRAW"

            total_a = w + d * 0.5
            pct = total_a / game_num * 100
            print(f"  [{game_num:3d}/{len(games)}] {w_name:10s}(W) vs {b_name_g:10s}(B) | "
                  f"{res_str:14s} ({plies:3d}p) | "
                  f"{a_name}: +{w} ={d} -{l} ({pct:.1f}%)")

            # Early stop if clearly losing after 100+ games
            if game_num >= 100:
                n = w + d + l
                s = w + d * 0.5
                p_val = s / n
                if p_val < 0.38 or p_val > 0.62:
                    elo = -400 * math.log10(1.0 / p_val - 1) if 0 < p_val < 1 else 0
                    print(f"\n  *** EARLY STOP at {game_num} games: {pct:.1f}% ({elo:+.0f} ELO) — clear result ***")
                    break

    total_a = w + d * 0.5
    n = w + d + l
    pct = total_a / n if n > 0 else 0.5
    if 0 < pct < 1:
        elo_diff = -400 * math.log10(1.0 / pct - 1)
    elif pct >= 1:
        elo_diff = 400
    else:
        elo_diff = -400

    print(f"\n  {'=' * 56}")
    print(f"  {a_name} vs {b_name} FINAL: +{w} ={d} -{l}")
    print(f"  Score: {total_a:.1f}/{n} ({pct*100:.1f}%)")
    print(f"  ELO diff: {a_name} is {elo_diff:+.0f} vs {b_name}")
    print(f"  {'=' * 56}")


if __name__ == "__main__":
    main()
