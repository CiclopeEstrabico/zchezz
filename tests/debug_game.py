"""Debug: find which move causes the engine to desync"""
import subprocess, time, os, threading
import chess

os.chdir(r"c:\Zchezz")

exe = r"engine\c\zchezz_v305\zchezz.exe"

def test_engine():
    p = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE, text=True, bufsize=1,
                         cwd=os.path.dirname(os.path.abspath(exe)))
    
    stderr_lines = []
    def read_stderr():
        for line in p.stderr:
            stderr_lines.append(line.strip())
    threading.Thread(target=read_stderr, daemon=True).start()
    
    def send(cmd):
        p.stdin.write(cmd + "\n")
        p.stdin.flush()
    
    def read_bestmove(timeout=10):
        t0 = time.time()
        infos = []
        while time.time() - t0 < timeout:
            line = p.stdout.readline().strip()
            if "info depth" in line:
                infos.append(line)
            if line.startswith("bestmove"):
                return line.split()[1] if len(line.split()) > 1 else None, infos
        return None, infos
    
    def read_until(kw, timeout=5):
        t0 = time.time()
        while time.time() - t0 < timeout:
            line = p.stdout.readline().strip()
            if kw in line: return True
        return False
    
    send("uci"); read_until("uciok")
    send("setoption name NNUE value " + os.path.abspath(r"engine\c\zchezz_v305\nnue_weights.bin"))
    send("isready"); read_until("readyok", 10)
    
    board = chess.Board()
    sf = subprocess.Popen([r"engine\stockfish\stockfish.exe"], 
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE, text=True, bufsize=1,
                          creationflags=subprocess.CREATE_NO_WINDOW)
    threading.Thread(target=lambda: sf.stderr.read(), daemon=True).start()
    sf.stdin.write("uci\n"); sf.stdin.flush()
    time.sleep(0.5)
    while True:
        l = sf.stdout.readline().strip()
        if "uciok" in l: break
    sf.stdin.write("setoption name UCI_LimitStrength value true\n"); sf.stdin.flush()
    sf.stdin.write("setoption name UCI_Elo value 2800\n"); sf.stdin.flush()
    sf.stdin.write("isready\n"); sf.stdin.flush()
    while True:
        l = sf.stdout.readline().strip()
        if "readyok" in l: break
    
    moves = []
    for ply in range(100):
        is_zchezz_turn = (board.turn == chess.WHITE)
        moves_str = " ".join(moves)
        pos_cmd = "position startpos" + (f" moves {moves_str}" if moves_str else "")
        
        if is_zchezz_turn:
            # Zchezz's turn - send isready sync first
            send("isready")
            read_until("readyok", 5)
            send(pos_cmd)
            send("go movetime 100")
            mv, infos = read_bestmove(10)
            engine_name = "Zchezz"
        else:
            # Stockfish's turn
            sf.stdin.write("isready\n"); sf.stdin.flush()
            while True:
                l = sf.stdout.readline().strip()
                if "readyok" in l: break
            sf.stdin.write(pos_cmd + "\n"); sf.stdin.flush()
            sf.stdin.write("go movetime 100\n"); sf.stdin.flush()
            t0 = time.time()
            infos = []
            mv = None
            while time.time() - t0 < 10:
                line = sf.stdout.readline().strip()
                if "info depth" in line: infos.append(line)
                if line.startswith("bestmove"):
                    mv = line.split()[1] if len(line.split()) > 1 else None
                    break
            engine_name = "SF-2800"
        
        if not mv:
            print(f"Ply {ply}: {engine_name} returned None!")
            break
        
        try:
            move = board.parse_uci(mv)
            if move not in board.legal_moves:
                print(f"Ply {ply}: {engine_name} ILLEGAL: '{mv}'")
                print(f"  FEN: {board.fen()}")
                print(f"  Legal: {[m.uci() for m in board.legal_moves]}")
                if infos:
                    print(f"  Last info: {infos[-1][:150]}")
                print(f"  Pos cmd: {pos_cmd[:200]}")
                # Check stderr
                if stderr_lines:
                    print(f"  Stderr: {stderr_lines[-5:]}")
                break
            board.push(move)
            moves.append(mv)
            print(f"Ply {ply}: {engine_name} {mv} OK  (moves_len={len(pos_cmd)})")
        except Exception as e:
            print(f"Ply {ply}: {engine_name} PARSE ERROR: '{mv}' - {e}")
            print(f"  FEN: {board.fen()}")
            if infos:
                print(f"  Last info: {infos[-1][:150]}")
            print(f"  Pos cmd len: {len(pos_cmd)}")
            if stderr_lines:
                print(f"  Stderr: {stderr_lines[-5:]}")
            break
        
        if board.is_game_over(claim_draw=True):
            print(f"Game over: {board.result(claim_draw=True)} after {ply+1} plies")
            break
    
    send("quit"); p.wait(timeout=3)
    sf.stdin.write("quit\n"); sf.stdin.flush(); sf.wait(timeout=3)

test_engine()
