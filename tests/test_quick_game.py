"""Quick 2-game test to verify tournament_elo produces decisive results with NNUE"""
import subprocess, time, re, os, sys
import chess
import chess.pgn

os.chdir(r"c:\Zchezz")

ZCHEZZ = r"engine\c\zchezz_v305\zchezz.exe"
STOCKFISH = r"engine\stockfish\stockfish.exe"

class Engine:
    def __init__(self, path, options=None, name=""):
        self.name = name
        self.path = path
        self.p = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, text=True, bufsize=1,
            cwd=os.path.dirname(os.path.abspath(path)),
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        import threading
        threading.Thread(target=lambda: self.p.stderr.read(), daemon=True).start()
        self._send("uci")
        self._wait_for("uciok", 10)
        if options:
            for k, v in options.items():
                self._send(f"setoption name {k} value {v}")
        self._send("isready")
        self._wait_for("readyok", 10)
    
    def _send(self, cmd):
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()
    
    def _wait_for(self, keyword, timeout):
        t0 = time.time()
        while time.time() - t0 < timeout:
            line = self.p.stdout.readline().strip()
            if keyword in line:
                return line
        return None
    
    def get_move(self, board):
        moves_str = " ".join([m.uci() for m in board.move_stack])
        pos = "position startpos" + (f" moves {moves_str}" if moves_str else "")
        self._send(pos)
        self._send("go movetime 200")
        t0 = time.time()
        while time.time() - t0 < 10:
            line = self.p.stdout.readline().strip()
            if line.startswith("bestmove"):
                parts = line.split()
                return parts[1] if len(parts) > 1 else None
        return None
    
    def quit(self):
        try:
            self._send("quit")
            self.p.wait(timeout=2)
        except:
            self.p.kill()

def play_game(white_eng, black_eng, max_plies=400):
    board = chess.Board()
    ply = 0
    while not board.is_game_over(claim_draw=True) and ply < max_plies:
        active = white_eng if board.turn == chess.WHITE else black_eng
        move_str = active.get_move(board)
        if not move_str:
            # Engine failed = loss
            return "0-1" if board.turn == chess.WHITE else "1-0", ply, "engine_fail"
        try:
            move = board.parse_uci(move_str)
            if move in board.legal_moves:
                board.push(move)
                ply += 1
            else:
                return "0-1" if board.turn == chess.WHITE else "1-0", ply, "illegal_move"
        except:
            return "0-1" if board.turn == chess.WHITE else "1-0", ply, "parse_error"
    
    if board.is_game_over(claim_draw=True):
        return board.result(claim_draw=True), ply, "natural"
    return "1/2-1/2", ply, "max_plies"

print("=== QUICK GAME TEST ===")
print()

# Game 1: Zchezz (White) vs SF-2800
print("--- Game 1: Zchezz vs SF-2800 ---")
z1 = Engine(ZCHEZZ, {"NNUE": os.path.abspath(r"engine\c\zchezz_v305\nnue_weights.bin"), 
                       "SyzygyPath": os.path.abspath("tablebases")}, "Zchezz")
sf1 = Engine(STOCKFISH, {"UCI_LimitStrength": "true", "UCI_Elo": "2800"}, "SF-2800")
result1, plies1, reason1 = play_game(z1, sf1, max_plies=200)
print(f"  Result: {result1} after {plies1} plies ({reason1})")
z1.quit(); sf1.quit()

# Game 2: SF-2800 (White) vs Zchezz
print("--- Game 2: SF-2800 vs Zchezz ---")
sf2 = Engine(STOCKFISH, {"UCI_LimitStrength": "true", "UCI_Elo": "2800"}, "SF-2800")
z2 = Engine(ZCHEZZ, {"NNUE": os.path.abspath(r"engine\c\zchezz_v305\nnue_weights.bin"),
                       "SyzygyPath": os.path.abspath("tablebases")}, "Zchezz")
result2, plies2, reason2 = play_game(sf2, z2, max_plies=200)
print(f"  Result: {result2} after {plies2} plies ({reason2})")
sf2.quit(); z2.quit()

print()
print(f"Game 1: {result1} ({reason1}, {plies1} plies)")
print(f"Game 2: {result2} ({reason2}, {plies2} plies)")

# Sanity check
has_decisive = result1 != "1/2-1/2" or result2 != "1/2-1/2"
both_long = plies1 > 30 and plies2 > 30
print()
if both_long:
    print("[PASS] Both games played long enough (>30 plies)")
else:
    print(f"[WARN] Short game detected: {plies1}, {plies2} plies")
    
print(f"Results: {result1}, {result2}")
print("=== TEST COMPLETE ===")
