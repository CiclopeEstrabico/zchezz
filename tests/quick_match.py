#!/usr/bin/env python3
"""Quick match with paired openings from PGN file: each opening played twice (swap colors).

Usage:
    python quick_match.py ENGINE_A ENGINE_B NAME_A NAME_B [GAMES] [OPENINGS_PGN]
"""
import subprocess, random, os, sys, time, math, re, glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elo_calc import elo_difference

# Auto-detect the latest engine version
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE_C_DIR = os.path.join(BASE_DIR, "engine", "c")
_engine_dirs = sorted(glob.glob(os.path.join(ENGINE_C_DIR, "zchezz_v*")))
_latest = _engine_dirs[-1] if _engine_dirs else os.path.join(ENGINE_C_DIR, "zchezz_v305")
_default_exe = os.path.join(_latest, "zchezz.exe")

# ── Configure via CLI args ─────────────────────────────
ENGINE_A = sys.argv[1] if len(sys.argv) > 1 else _default_exe
ENGINE_B = sys.argv[2] if len(sys.argv) > 2 else _default_exe
NAME_A   = sys.argv[3] if len(sys.argv) > 3 else "A"
NAME_B   = sys.argv[4] if len(sys.argv) > 4 else "B"
GAMES    = int(sys.argv[5]) if len(sys.argv) > 5 else 40
OPENINGS_PGN = sys.argv[6] if len(sys.argv) > 6 else r"c:\Zchezz\openings\Blitz_Testing_4moves.pgn"
NNUE_A   = os.path.join(os.path.dirname(ENGINE_A), "nnue_weights.bin")
NNUE_B   = os.path.join(os.path.dirname(ENGINE_B), "nnue_weights.bin")

MOVETIME = 50
MAX_MOVES = 200

# ── PGN Opening Loader ─────────────────────────────────
SAN_PIECE = {"K": 6, "Q": 5, "R": 4, "B": 3, "N": 2}

def san_to_uci_simple(san, board_fen):
    """Minimal SAN->UCI via python subprocess calling the engine itself? No.
    We'll use a minimal approach: parse PGN moves via regex and convert."""
    # For robustness, we use python-chess if available, else fallback to regex
    try:
        import chess
        board = chess.Board(board_fen)
        move = board.parse_san(san)
        return move.uci(), board.fen()
    except ImportError:
        return None, None

def load_openings_pgn(pgn_path, max_openings=5000):
    """Load openings from PGN file. Returns list of UCI move strings."""
    try:
        import chess
        import chess.pgn
    except ImportError:
        print("WARNING: python-chess not installed. Using fallback openings.", flush=True)
        return None

    openings = []
    with open(pgn_path, "r", errors="ignore") as f:
        while len(openings) < max_openings:
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

# Fallback built-in openings (UCI format)
FALLBACK_OPENINGS = [
    "e2e4 e7e5 g1f3 b8c6 f1c4",         # Italian
    "e2e4 c7c5 g1f3 d7d6 d2d4",          # Sicilian
    "e2e4 e7e6 d2d4 d7d5 b1c3",          # French
    "e2e4 c7c6 d2d4 d7d5 b1c3",          # Caro-Kann
    "d2d4 d7d5 c2c4 e7e6 b1c3",          # QGD
    "d2d4 d7d5 c2c4 d5c4 e2e3",          # QGA
    "d2d4 d7d5 c2c4 c7c6 g1f3",          # Slav
    "d2d4 g8f6 c2c4 g7g6 b1c3",          # King's Indian
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4",     # Nimzo-Indian
    "c2c4 e7e5 b1c3 g8f6 g1f3",          # English
    "e2e4 e7e5 g1f3 b8c6 f1b5",          # Ruy Lopez
    "e2e4 e7e5 g1f3 b8c6 d2d4",          # Scotch
    "e2e4 d7d6 d2d4 g8f6 b1c3",          # Pirc
    "e2e4 d7d5 e4d5 d8d5 b1c3",          # Scandinavian
    "d2d4 f7f5 c2c4 g8f6 g1f3",          # Dutch
    "d2d4 d7d5 c2c4 e7e6 g2g3",          # Catalan
    "d2d4 d7d5 c1f4 g8f6 e2e3",          # London
    "d2d4 g8f6 g1f3 e7e6 c1g5",          # Torre
    "d2d4 g8f6 c2c4 c7c5 d4d5",          # Benoni
    "d2d4 g8f6 c2c4 g7g6 b1c3 d7d5",     # Grünfeld
    "e2e4 e7e5 g1f3 g8f6 f3e5",          # Petrov
    "e2e4 e7e5 b1c3 g8f6 f1c4",          # Vienna
    "e2e4 e7e5 g1f3 b8c6 b1c3 g8f6",     # Four Knights
    "e2e4 e7e5 g1f3 d7d6 d2d4",          # Philidor
    "e2e4 g8f6 e4e5 f6d5 d2d4",          # Alekhine
]

class UCIEngine:
    def __init__(self, path, nnue):
        self.proc = subprocess.Popen(
            [path, "--nnue", nnue],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1
        )
        self._read_until("uciok", send="uci")
        self._send("isready")
        self._read_until("readyok")

    def _send(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _read_until(self, token, send=None, timeout=10):
        if send: self._send(send)
        start = time.time()
        while True:
            line = self.proc.stdout.readline().strip()
            if token in line: return line
            if time.time() - start > timeout: return ""

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
        while True:
            line = self.proc.stdout.readline().strip()
            if line.startswith("bestmove"):
                parts = line.split()
                return parts[1] if len(parts) > 1 else "(none)"
            if time.time() - start > 30:  # 30s hard timeout per move
                return "(none)"

    def quit(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=2)
        except:
            self.proc.kill()

def play_game(eng_w, eng_b, fen, opening_moves):
    eng_w.new_game()
    eng_b.new_game()
    moves = opening_moves.split() if opening_moves else []
    for ply in range(len(moves), MAX_MOVES * 2):
        eng = eng_w if ply % 2 == 0 else eng_b
        moves_str = " ".join(moves)
        move = eng.go(fen, moves_str)
        if move in ("(none)", "0000", "none", ""):
            return 0.0 if ply % 2 == 0 else 1.0
        moves.append(move)
    return 0.5



def main():
    # Load openings from PGN
    openings = None
    if os.path.exists(OPENINGS_PGN):
        openings = load_openings_pgn(OPENINGS_PGN)
        if openings:
            print(f"Loaded {len(openings)} openings from {os.path.basename(OPENINGS_PGN)}", flush=True)

    if not openings:
        openings = FALLBACK_OPENINGS
        print(f"Using {len(openings)} built-in openings", flush=True)

    random.seed(42)
    random.shuffle(openings)
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    wins_a = draws = wins_b = 0
    total = 0
    n_pairs = GAMES // 2

    # Ensure we have enough openings for all pairs (no reuse ideally)
    if len(openings) < n_pairs:
        print(f"WARNING: Only {len(openings)} openings for {n_pairs} pairs. Some will repeat.", flush=True)

    print(f"Match: {NAME_A} vs {NAME_B}", flush=True)
    print(f"Games: {GAMES} (={n_pairs} paired openings) | Movetime: {MOVETIME}ms | Unique openings: {min(len(openings), n_pairs)}", flush=True)
    print("=" * 70, flush=True)

    t_start = time.time()

    for pair in range(n_pairs):
        opening = openings[pair % len(openings)]

        # Game 1: Engine A as White
        eng_w = UCIEngine(ENGINE_A, NNUE_A)
        eng_b = UCIEngine(ENGINE_B, NNUE_B)
        result = play_game(eng_w, eng_b, fen, opening)
        if result == 1.0: wins_a += 1
        elif result == 0.0: wins_b += 1
        else: draws += 1
        eng_w.quit(); eng_b.quit()
        total += 1

        # Game 2: Engine B as White (same opening, swapped colors)
        eng_w = UCIEngine(ENGINE_B, NNUE_B)
        eng_b = UCIEngine(ENGINE_A, NNUE_A)
        result = play_game(eng_w, eng_b, fen, opening)
        if result == 0.0: wins_a += 1
        elif result == 1.0: wins_b += 1
        else: draws += 1
        eng_w.quit(); eng_b.quit()
        total += 1

        if total % 10 == 0:
            pts_a = wins_a + draws * 0.5
            pct = pts_a / total * 100 if total > 0 else 50
            elo, err, _ = elo_difference(wins_a, draws, wins_b)
            elapsed = time.time() - t_start
            print(f"  [{total:3d}] {NAME_A}: {pts_a:.1f}/{total} ({pct:.1f}%) "
                  f"| W={wins_a} D={draws} L={wins_b} | ELO: {elo:+.1f} ±{err:.1f} [{elapsed:.0f}s]", flush=True)

    pts_a = wins_a + draws * 0.5
    pct = pts_a / total * 100 if total > 0 else 50
    elo, err, _ = elo_difference(wins_a, draws, wins_b)
    elapsed = time.time() - t_start

    print("\n" + "=" * 70, flush=True)
    print(f"FINAL: {NAME_A} vs {NAME_B}", flush=True)
    print(f"  {NAME_A}: {pts_a:.1f}/{total} ({pct:.1f}%)", flush=True)
    print(f"  W={wins_a} D={draws} L={wins_b}", flush=True)
    print(f"  ELO difference: {elo:+.1f} ±{err:.1f}", flush=True)
    print(f"  Elapsed: {elapsed:.0f}s", flush=True)
    print("=" * 70, flush=True)

    return 0 if elo > 0 else 1

if __name__ == "__main__":
    sys.exit(main())
