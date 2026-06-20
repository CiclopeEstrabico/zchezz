#!/usr/bin/env python3
"""3-way comparison: v215 vs v305+TB vs v305 (no TB)
Uses the same tournament infrastructure but runs each engine separately."""

import subprocess, threading, time, os, sys, math, re
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, as_completed
import chess, chess.pgn
from io import StringIO

# ─── Config ──────────────────────────────────────────────────────────
MOVETIME = 100        # ms per move
MAX_PLIES = 400       # max plies per game
GAMES_PER_ANCHOR = 100  # games per anchor per engine (200 total per engine)
WORKERS = 14          # concurrent games (leave 2 CPUs for OS)

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NNUE_305 = os.path.join(BASE, "engine", "c", "zchezz_v305", "nnue_weights.bin")
NNUE_215 = os.path.join(BASE, "engine", "old", "c", "zchezz_v215", "nnue_weights.bin")
TB_PATH  = os.path.join(BASE, "tablebases")
OPENINGS = os.path.join(BASE, "tests", "openings.pgn")

SF_PATH  = os.path.join(BASE, "engine", "stockfish", "stockfish.exe")

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

ANCHORS = [
    {"name": "SF-2800", "elo": 2800, "options": {"UCI_LimitStrength": "true", "UCI_Elo": "2800"}},
    {"name": "SF-2900", "elo": 2900, "options": {"UCI_LimitStrength": "true", "UCI_Elo": "2900"}},
]


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
            fen = b.fen()
            uci_moves = " ".join(m.uci() for m in moves)
            openings.append({"fen": fen, "moves": uci_moves, "name": g.headers.get("Event", "?")})
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
        # Stop + drain
        self._send("stop")
        t0 = time.time()
        while time.time() - t0 < 0.5:
            try:
                line = self.queue.get(timeout=0.05)
                if line.startswith("bestmove"):
                    break
            except Empty:
                break

        # Sync
        self._send("isready")
        t0 = time.time()
        while time.time() - t0 < 10:
            try:
                line = self.queue.get(timeout=0.1)
                if "readyok" in line:
                    break
            except Empty:
                pass

        # Flush
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Empty:
                break

        # Position
        m_list = " ".join([m.uci() for m in board.move_stack])
        if self.start_fen:
            pos_cmd = f"position fen {self.start_fen}" + (f" moves {m_list}" if m_list else "")
        else:
            pos_cmd = "position startpos" + (f" moves {m_list}" if m_list else "")
        self._send(pos_cmd)

        # Go
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
def play_game(test_cfg, anchor_cfg, opening, test_white):
    """Returns (result_for_test, plies)"""
    try:
        test_eng = UCIEngine(test_cfg["path"], test_cfg["options"], test_cfg.get("name", "test"))
        sf_eng = UCIEngine(SF_PATH, anchor_cfg["options"], anchor_cfg["name"])
    except Exception as e:
        return None, 0

    board = chess.Board()
    if opening:
        board = chess.Board(opening["fen"])
        test_eng.start_fen = opening["fen"]
        sf_eng.start_fen = opening["fen"]

    ply = 0
    result = None
    while not board.is_game_over(claim_draw=True) and ply < MAX_PLIES:
        is_test_turn = (board.turn == chess.WHITE) == test_white
        eng = test_eng if is_test_turn else sf_eng
        mv_str = eng.get_move(board)
        if not mv_str:
            # Engine failed — other side wins
            result = 0.0 if is_test_turn else 1.0
            break
        try:
            move = board.parse_uci(mv_str)
            if move not in board.legal_moves:
                result = 0.0 if is_test_turn else 1.0
                break
            board.push(move)
            ply += 1
        except:
            result = 0.0 if is_test_turn else 1.0
            break

    if result is None:
        if board.is_game_over(claim_draw=True):
            r = board.result(claim_draw=True)
            if r == "1-0":
                result = 1.0 if test_white else 0.0
            elif r == "0-1":
                result = 0.0 if test_white else 1.0
            else:
                result = 0.5
        else:
            result = 0.5  # max plies

    test_eng.quit()
    sf_eng.quit()
    return result, ply


# ─── ELO calculation ────────────────────────────────────────────────
def compute_elo(results, anchors_used):
    """Given list of (score, anchor_elo), compute engine ELO via MLE."""
    if not results:
        return 0, 0
    total_score = sum(r[0] for r in results)
    n = len(results)
    avg_score = total_score / n
    if avg_score <= 0.001:
        return 800, 999
    if avg_score >= 0.999:
        return 3500, 999

    # MLE: find ELO that maximizes likelihood
    best_elo, best_ll = 2000, -1e18
    for elo_try in range(1500, 3200):
        ll = 0
        for score, anch_elo in results:
            expected = 1.0 / (1.0 + 10 ** ((anch_elo - elo_try) / 400.0))
            if score == 1.0:
                ll += math.log(max(expected, 1e-10))
            elif score == 0.0:
                ll += math.log(max(1 - expected, 1e-10))
            else:
                ll += math.log(max(expected * (1 - expected), 1e-15)) * 0.5 + math.log(2) * 0.5
        if ll > best_ll:
            best_ll = ll
            best_elo = elo_try

    # Error estimate
    n = len(results)
    se = math.sqrt(avg_score * (1 - avg_score) / n) if n > 1 else 0.5
    elo_err = 400 * se / max(avg_score * (1 - avg_score), 0.01)
    return best_elo, round(elo_err, 1)


# ─── Main ───────────────────────────────────────────────────────────
def main():
    import random
    random.seed(42)

    print("=" * 70)
    print("  3-Way Comparison: v215 vs v305+TB vs v305 (no TB)")
    print(f"  Movetime: {MOVETIME}ms | Workers: {WORKERS} | Games/anchor: {GAMES_PER_ANCHOR}")
    print("=" * 70)

    for eng_name, eng_cfg in ENGINES.items():
        eng_cfg["name"] = eng_name
        results = []  # [(score, anchor_elo), ...]
        wins, draws, losses = 0, 0, 0
        game_num = 0
        total_games = GAMES_PER_ANCHOR * len(ANCHORS)

        print(f"\n  Testing: {eng_name}")
        print(f"  {'-' * 60}")

        # Build game list
        games = []
        for anchor in ANCHORS:
            for i in range(GAMES_PER_ANCHOR):
                opening = OPENING_LIST[i % len(OPENING_LIST)] if OPENING_LIST else None
                test_white = (i % 2 == 0)
                games.append((eng_cfg, anchor, opening, test_white))

        random.shuffle(games)

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {}
            for g_info in games:
                f = pool.submit(play_game, g_info[0], g_info[1], g_info[2], g_info[3])
                futures[f] = g_info

            for f in as_completed(futures):
                g_info = futures[f]
                anchor = g_info[1]
                try:
                    score, plies = f.result()
                except Exception as e:
                    score, plies = 0.5, 0

                if score is None:
                    score = 0.5

                game_num += 1
                results.append((score, anchor["elo"]))

                if score == 1.0:
                    wins += 1
                    res_str = "WIN"
                elif score == 0.0:
                    losses += 1
                    res_str = "LOSS"
                else:
                    draws += 1
                    res_str = "DRAW"

                elo, err = compute_elo(results, None)
                print(f"  [{game_num:3d}/{total_games}] vs {anchor['name']:8s} | "
                      f"{res_str:4s} ({plies:3d}p) | "
                      f"W:{wins} D:{draws} L:{losses} | "
                      f"ELO: {elo:6.1f} +/- {err:5.1f}")

        # Final stats
        elo, err = compute_elo(results, None)
        total = wins + draws + losses
        print(f"\n  {'=' * 60}")
        print(f"  {eng_name} FINAL: ELO {elo:.0f} +/- {err:.1f}")
        print(f"  W/D/L: {wins}/{draws}/{losses} ({total} games)")
        print(f"  Score: {sum(r[0] for r in results):.1f}/{total}")
        print(f"  {'=' * 60}")

    print("\nAll done!")


if __name__ == "__main__":
    main()
