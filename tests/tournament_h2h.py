#!/usr/bin/env python3
"""Head-to-head round-robin: v215 vs v305+TB vs v305 (no TB)
No Stockfish — direct comparison between the three engines."""

import subprocess, threading, time, os, sys, math, random
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, as_completed
import chess, chess.pgn

# ─── Config ──────────────────────────────────────────────────────────
MOVETIME = 100        # ms per move
MAX_PLIES = 400       # max plies per game
GAMES_PER_PAIR = 100  # games per pair (50 as white, 50 as black)
WORKERS = 14          # concurrent games

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NNUE_305 = os.path.join(BASE, "engine", "c", "zchezz_v305", "nnue_weights.bin")
NNUE_215 = os.path.join(BASE, "engine", "old", "c", "zchezz_v215", "nnue_weights.bin")
TB_PATH  = os.path.join(BASE, "tablebases")
OPENINGS = os.path.join(BASE, "tests", "openings.pgn")

ENGINES = {
    "v215": {
        "path": os.path.join(BASE, "engine", "old", "c", "zchezz_v215", "zchezz.exe"),
        "options": {"NNUE": NNUE_215},
    },
    "v305+TB": {
        "path": os.path.join(BASE, "engine", "c", "zchezz_v305", "zchezz.exe"),
        "options": {"NNUE": NNUE_305, "SyzygyPath": TB_PATH},
    },
    "v305": {
        "path": os.path.join(BASE, "engine", "c", "zchezz_v305", "zchezz.exe"),
        "options": {"NNUE": NNUE_305},
    },
}


# ─── Load openings ──────────────────────────────────────────────────
def load_openings():
    openings = []
    if not os.path.exists(OPENINGS):
        return openings
    with open(OPENINGS, "r") as f:
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            b = g.board()
            moves = list(g.mainline_moves())
            for m in moves:
                b.push(m)
            openings.append({"fen": b.fen(), "moves": " ".join(m.uci() for m in moves)})
    return openings

OPENING_LIST = load_openings()
print(f"  Loaded {len(OPENING_LIST)} openings")


# ─── UCI Engine wrapper ─────────────────────────────────────────────
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
            try:
                self.queue.get_nowait()
            except Empty:
                break

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
            try:
                self.p.kill()
            except:
                pass


# ─── Play one game ──────────────────────────────────────────────────
def play_game(white_cfg, black_cfg, opening):
    """Returns: 1.0=white wins, 0.0=black wins, 0.5=draw"""
    try:
        w_eng = UCIEngine(white_cfg["path"], white_cfg["options"], white_cfg.get("name", "W"))
        b_eng = UCIEngine(black_cfg["path"], black_cfg["options"], black_cfg.get("name", "B"))
    except Exception as e:
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
            if r == "1-0":
                result = 1.0
            elif r == "0-1":
                result = 0.0
            else:
                result = 0.5
        else:
            result = 0.5

    w_eng.quit()
    b_eng.quit()
    return result, ply


# ─── Main ───────────────────────────────────────────────────────────
def main():
    random.seed(42)

    eng_names = list(ENGINES.keys())
    pairs = []
    for i in range(len(eng_names)):
        for j in range(i + 1, len(eng_names)):
            pairs.append((eng_names[i], eng_names[j]))

    print("=" * 70)
    print("  Head-to-Head Round Robin (no Stockfish)")
    print(f"  Engines: {', '.join(eng_names)}")
    print(f"  Movetime: {MOVETIME}ms | Workers: {WORKERS}")
    print(f"  Games per pair: {GAMES_PER_PAIR} ({GAMES_PER_PAIR}//2 each color)")
    print("=" * 70)

    # Track results per pair
    pair_results = {}
    for a, b in pairs:
        pair_results[(a, b)] = {"w": 0, "d": 0, "l": 0}  # from a's perspective

    # Track overall score per engine
    engine_score = {n: 0.0 for n in eng_names}
    engine_games = {n: 0 for n in eng_names}

    for a_name, b_name in pairs:
        a_cfg = ENGINES[a_name]
        b_cfg = ENGINES[b_name]
        a_cfg["name"] = a_name
        b_cfg["name"] = b_name

        print(f"\n  {'=' * 60}")
        print(f"  {a_name} vs {b_name} ({GAMES_PER_PAIR} games)")
        print(f"  {'=' * 60}")

        # Build game list: alternate colors
        games = []
        for i in range(GAMES_PER_PAIR):
            opening = OPENING_LIST[i % len(OPENING_LIST)] if OPENING_LIST else None
            if i % 2 == 0:
                games.append((a_cfg, b_cfg, opening, a_name, b_name))
            else:
                games.append((b_cfg, a_cfg, opening, b_name, a_name))

        game_num = 0
        w, d, l = 0, 0, 0

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

                # Update from a_name's perspective
                if w_name == a_name:
                    a_score = score_w
                else:
                    a_score = 1.0 - score_w

                if a_score == 1.0:
                    w += 1
                    res_str = f"{a_name} WIN"
                elif a_score == 0.0:
                    l += 1
                    res_str = f"{b_name} WIN"
                else:
                    d += 1
                    res_str = "DRAW"

                engine_score[a_name] += a_score
                engine_score[b_name] += (1.0 - a_score)
                engine_games[a_name] += 1
                engine_games[b_name] += 1

                total_a = w + d * 0.5
                pct = total_a / game_num * 100

                print(f"  [{game_num:3d}/{GAMES_PER_PAIR}] {w_name:8s}(W) vs {b_name_g:8s}(B) | "
                      f"{res_str:12s} ({plies:3d}p) | "
                      f"{a_name}: +{w} ={d} -{l} ({pct:.1f}%)")

        pair_results[(a_name, b_name)] = {"w": w, "d": d, "l": l}
        total_a = w + d * 0.5
        print(f"\n  Result: {a_name} vs {b_name}: +{w} ={d} -{l} "
              f"({total_a:.1f}/{game_num} = {total_a/game_num*100:.1f}%)")

    # ─── Final Standings ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL STANDINGS")
    print("=" * 70)

    # Crosstable
    print(f"\n  {'':12s}", end="")
    for n in eng_names:
        print(f"{n:>12s}", end="")
    print(f"{'Score':>10s}{'Games':>8s}{'%':>8s}")

    for a in eng_names:
        print(f"  {a:12s}", end="")
        for b in eng_names:
            if a == b:
                print(f"{'---':>12s}", end="")
            else:
                key = (a, b) if (a, b) in pair_results else (b, a)
                pr = pair_results[key]
                if key[0] == a:
                    w_val, d_val, l_val = pr["w"], pr["d"], pr["l"]
                    cell = f"+{w_val}={d_val}-{l_val}"
                    print(f"{cell:>12s}", end="")
                else:
                    w_val, d_val, l_val = pr["l"], pr["d"], pr["w"]
                    cell = f"+{w_val}={d_val}-{l_val}"
                    print(f"{cell:>12s}", end="")
            pass
        s = engine_score[a]
        g = engine_games[a]
        pct = s / g * 100 if g > 0 else 0
        print(f"{s:>10.1f}{g:>8d}{pct:>7.1f}%")

    # ELO differences (from pairwise scores)
    print(f"\n  Pairwise ELO differences:")
    for a_name, b_name in pairs:
        pr = pair_results[(a_name, b_name)]
        n = pr["w"] + pr["d"] + pr["l"]
        s = pr["w"] + pr["d"] * 0.5
        pct = s / n if n > 0 else 0.5
        if 0 < pct < 1:
            elo_diff = -400 * math.log10(1.0 / pct - 1)
        elif pct >= 1:
            elo_diff = 400
        else:
            elo_diff = -400
        print(f"  {a_name} vs {b_name}: {s:.1f}/{n} ({pct*100:.1f}%) -> "
              f"{a_name} is {elo_diff:+.0f} ELO vs {b_name}")

    print("\nDone!")


if __name__ == "__main__":
    main()
