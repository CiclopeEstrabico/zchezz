#!/usr/bin/env python3
"""Tournament: Zchezz v305+TB vs Stockfish anchors (2800, 2900)
600+ games, 8 concurrent games, updates every 50 games per anchor.
"""
import subprocess, os, sys, time, math, random, threading, re
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elo_calc import elo_difference, estimated_elo

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tournament_live.log")

def log(msg):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# ── Paths ──────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZCHEZZ = os.path.join(BASE, "engine", "c", "zchezz_v305", "zchezz.exe")
NNUE = os.path.join(BASE, "engine", "c", "zchezz_v305", "nnue_weights.bin")
SF = os.path.join(BASE, "engine", "stockfish", "stockfish.exe")
TB_PATH = os.path.join(BASE, "tablebases")
OPENINGS = os.path.join(BASE, "openings", "Blitz_Testing_4moves.pgn")

MOVETIME = 200       # ms per move
MAX_MOVES = 200
WORKERS = 8          # 8 concurrent games = 16 processes = 16 cores

# ── Anchors ────────────────────────────────────────────
ANCHORS = [
    {"elo": 2800, "games": 300},
    {"elo": 2900, "games": 300},
]

# ── Load openings ─────────────────────────────────────
def load_openings(pgn_path):
    try:
        import chess, chess.pgn
        openings = []
        with open(pgn_path, "r", errors="ignore") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                board = game.board()
                uci_moves = []
                for move in game.mainline_moves():
                    uci_moves.append(move.uci())
                    board.push(move)
                if uci_moves:
                    openings.append(" ".join(uci_moves))
        return openings
    except ImportError:
        return None

FALLBACK_OPENINGS = [
    "e2e4 e7e5 g1f3 b8c6 f1c4", "e2e4 c7c5 g1f3 d7d6 d2d4",
    "e2e4 e7e6 d2d4 d7d5 b1c3", "d2d4 d7d5 c2c4 e7e6 b1c3",
    "d2d4 g8f6 c2c4 g7g6 b1c3", "e2e4 e7e5 g1f3 b8c6 f1b5",
    "d2d4 d7d5 c2c4 c7c6 g1f3", "e2e4 c7c6 d2d4 d7d5 b1c3",
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4", "c2c4 e7e5 b1c3 g8f6 g1f3",
    "e2e4 e7e5 g1f3 b8c6 d2d4", "e2e4 d7d6 d2d4 g8f6 b1c3",
    "d2d4 d7d5 c2c4 d5c4 e2e3", "d2d4 f7f5 c2c4 g8f6 g1f3",
    "d2d4 d7d5 c1f4 g8f6 e2e3", "d2d4 g8f6 c2c4 c7c5 d4d5",
]

# ── UCI Engine ─────────────────────────────────────────
class UCIEngine:
    def __init__(self, path, options=None):
        self.proc = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
            cwd=os.path.dirname(path)
        )
        self._read_until("uciok", send="uci")
        if options:
            for k, v in options.items():
                self._send(f"setoption name {k} value {v}")
        self._send("isready")
        self._read_until("readyok")

    def _send(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _read_until(self, token, send=None, timeout=15):
        if send:
            self._send(send)
        start = time.time()
        while True:
            line = self.proc.stdout.readline().strip()
            if token in line:
                return line
            if time.time() - start > timeout:
                return ""

    def new_game(self):
        self._send("ucinewgame")
        self._send("isready")
        self._read_until("readyok")

    def go(self, fen, moves_str):
        if moves_str:
            self._send(f"position fen {fen} moves {moves_str}")
        else:
            self._send(f"position fen {fen}")
        self._send(f"go movetime {MOVETIME}")
        start = time.time()
        last_score = 0
        is_mate = False
        while True:
            line = self.proc.stdout.readline().strip()
            if "score cp " in line:
                try:
                    idx = line.index("score cp ") + 9
                    rest = line[idx:].split()[0]
                    last_score = int(rest)
                    is_mate = False
                except:
                    pass
            elif "score mate " in line:
                try:
                    idx = line.index("score mate ") + 11
                    rest = line[idx:].split()[0]
                    mate_in = int(rest)
                    last_score = 30000 if mate_in > 0 else -30000
                    is_mate = True
                except:
                    pass
            if line.startswith("bestmove"):
                parts = line.split()
                move = parts[1] if len(parts) > 1 else "(none)"
                return move, last_score
            if time.time() - start > 60:
                return "(none)", 0

    def quit(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=3)
        except:
            self.proc.kill()

# ── Adjudication thresholds ────────────────────────────
WIN_CP = 1000       # 10 pawns = clearly won
WIN_COUNT = 4        # must persist for 4 consecutive moves
DRAW_CP = 5          # both sides < 5cp
DRAW_COUNT = 8       # for 8 consecutive moves

# ── Play one game ──────────────────────────────────────
def play_game(anchor_elo, opening, zchezz_white):
    zchezz_opts = {"NNUE": NNUE, "Threads": "1", "SyzygyPath": TB_PATH}
    sf_opts = {"UCI_LimitStrength": "true", "UCI_Elo": str(anchor_elo), "Threads": "1"}

    try:
        eng_z = UCIEngine(ZCHEZZ, zchezz_opts)
        eng_s = UCIEngine(SF, sf_opts)
    except Exception:
        return 0.5

    eng_w = eng_z if zchezz_white else eng_s
    eng_b = eng_s if zchezz_white else eng_z
    eng_w.new_game()
    eng_b.new_game()

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    moves = opening.split() if opening else []

    # Adjudication counters (from white's perspective)
    white_winning_count = 0  # consecutive plies where white is winning
    black_winning_count = 0  # consecutive plies where black is winning
    draw_count = 0           # consecutive plies where both agree ~0

    for ply in range(len(moves), MAX_MOVES * 2):
        eng = eng_w if ply % 2 == 0 else eng_b
        moves_str = " ".join(moves)
        move, score_cp = eng.go(fen, moves_str)

        if move in ("(none)", "0000", "none", ""):
            eng_w.quit(); eng_b.quit()
            if ply % 2 == 0:
                return 0.0 if zchezz_white else 1.0
            else:
                return 1.0 if zchezz_white else 0.0

        # Convert score to white's perspective
        white_score = score_cp if ply % 2 == 0 else -score_cp

        # Track winning streaks
        if white_score >= WIN_CP:
            white_winning_count += 1
            black_winning_count = 0
            draw_count = 0
        elif white_score <= -WIN_CP:
            black_winning_count += 1
            white_winning_count = 0
            draw_count = 0
        elif abs(white_score) <= DRAW_CP:
            draw_count += 1
            white_winning_count = 0
            black_winning_count = 0
        else:
            white_winning_count = 0
            black_winning_count = 0
            draw_count = 0

        # Adjudicate
        if white_winning_count >= WIN_COUNT:
            eng_w.quit(); eng_b.quit()
            return 1.0 if zchezz_white else 0.0  # white wins
        if black_winning_count >= WIN_COUNT:
            eng_w.quit(); eng_b.quit()
            return 0.0 if zchezz_white else 1.0  # black wins
        if draw_count >= DRAW_COUNT and ply >= 80:
            eng_w.quit(); eng_b.quit()
            return 0.5

        moves.append(move)

    eng_w.quit(); eng_b.quit()
    return 0.5

# ── Run one anchor match ──────────────────────────────
def run_anchor(anchor_elo, total_games, openings):
    n_pairs = total_games // 2
    w = d = l = 0
    completed = 0
    next_report = 10
    lock = threading.Lock()
    start_time = time.time()

    shuffled = openings.copy()
    random.shuffle(shuffled)

    def play_pair(pair_idx):
        opening = shuffled[pair_idx % len(shuffled)]
        r1 = play_game(anchor_elo, opening, True)
        r2 = play_game(anchor_elo, opening, False)
        return r1, r2

    workers_per_anchor = WORKERS // len(ANCHORS)

    try:
        with ThreadPoolExecutor(max_workers=workers_per_anchor) as pool:
            futures = {pool.submit(play_pair, i): i for i in range(n_pairs)}

            for future in as_completed(futures):
                try:
                    r1, r2 = future.result()
                except Exception as e:
                    log(f"  [vs SF {anchor_elo}] ERROR in pair: {e}")
                    continue
                with lock:
                    for r in (r1, r2):
                        if r == 1.0: w += 1
                        elif r == 0.0: l += 1
                        else: d += 1
                    completed = w + d + l

                    if completed >= next_report or completed == total_games:
                        pts = w + d * 0.5
                        pct = pts / completed * 100
                        diff, err, _ = elo_difference(w, d, l)
                        est = anchor_elo + diff
                        elapsed = time.time() - start_time
                        gps = completed / elapsed if elapsed > 0 else 0
                        log(f"  [vs SF {anchor_elo}] {completed:3d}/{total_games} | W={w} D={d} L={l} | {pct:.1f}% | diff={diff:+.1f} ±{err:.1f} | Est: {est:.0f} | {gps:.1f} g/s")
                        next_report = ((completed // 50) + 1) * 50
    except Exception as e:
        log(f"  [vs SF {anchor_elo}] FATAL ERROR: {e}")

    return {"w": w, "d": d, "l": l}

# ── Main ───────────────────────────────────────────────
def main():
    log("=" * 78)
    log("  Zchezz v305 + Tablebases  vs  Stockfish Anchors")
    log(f"  Movetime: {MOVETIME}ms | Workers: {WORKERS} concurrent games")
    log("=" * 78)

    openings = load_openings(OPENINGS) if os.path.exists(OPENINGS) else None
    if openings:
        log(f"  Loaded {len(openings)} openings from PGN")
    else:
        openings = FALLBACK_OPENINGS
        log(f"  Using {len(openings)} built-in openings")
    log('')

    results = {}
    threads = []

    def run_and_store(anchor):
        r = run_anchor(anchor["elo"], anchor["games"], openings)
        results[anchor["elo"]] = r

    for anchor in ANCHORS:
        log(f"  Starting match vs SF {anchor['elo']} ({anchor['games']} games)...")
        t = threading.Thread(target=run_and_store, args=(anchor,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # ── Final summary ──
    log("\n" + "=" * 78)
    log("  FINAL RESULTS")
    log("=" * 78)

    elos = []
    for anchor in ANCHORS:
        elo = anchor["elo"]
        r = results.get(elo, {"w":0,"d":0,"l":0})
        total = r["w"] + r["d"] + r["l"]
        if total == 0: continue
        pts = r["w"] + r["d"] * 0.5
        pct = pts / total * 100
        diff, err, _ = elo_difference(r["w"], r["d"], r["l"])
        est = elo + diff
        elos.append(est)
        log(f"  vs SF {elo}: {pts:.1f}/{total} ({pct:.1f}%) | "
              f"W={r['w']} D={r['d']} L={r['l']} | "
              f"Δ={diff:+.1f} ±{err:.1f} | Estimated: {est:.0f} ±{err:.0f}")

    if elos:
        avg = sum(elos) / len(elos)
        log(f"\n  >>> Average estimated Elo: {avg:.0f} <<<")
    log("=" * 78)

if __name__ == "__main__":
    main()
