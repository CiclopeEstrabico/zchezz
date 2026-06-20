"""Test: verify Zchezz correctly applies long move sequences"""
import subprocess, time, os, threading

os.chdir(r"c:\Zchezz")

# The exact moves from the failed game
moves = "e2e4 c7c5 d2d4 c5d4 g1f3 g7g6 f3d4 g8f6 b1c3 b8c6 d4c6 b7c6 e4e5 f6h5 f1e2 h5g7 e1g1 g7e6 f2f4 f8g7 g2g4 g6g5 f4f5 e6f4 c3e4 d8b6 g1h1 g7e5 e4g5 f4e2 d1e2 f7f6 c2c4 c6c5 g5f3 c8b7 h1g1 e5c7 g4g5 b6c6 c1d2 e8c8 e2e7 h8f8 e7e2 f6g5 f3d4 c5d4 e2e4 c6e4"

exe = r"engine\c\zchezz_v305\zchezz.exe"
p = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
                     cwd=os.path.dirname(os.path.abspath(exe)))
threading.Thread(target=lambda: [print("STDERR:", l.strip()) for l in p.stderr], daemon=True).start()

def send(cmd):
    print(f">>> {cmd[:200]}{'...' if len(cmd) > 200 else ''}")
    p.stdin.write(cmd + "\n"); p.stdin.flush()

def read_until(keyword, timeout=5):
    t0 = time.time()
    while time.time() - t0 < timeout:
        line = p.stdout.readline().strip()
        if line: print(f"  < {line[:200]}")
        if keyword in line: return line
    return None

send("uci"); read_until("uciok")
send(r"setoption name NNUE value C:\Zchezz\engine\c\zchezz_v305\nnue_weights.bin")
send("isready"); read_until("readyok", 10)

# Test incremental: apply moves one at a time and check eval makes sense
move_list = moves.split()
for i in range(len(move_list)):
    partial = " ".join(move_list[:i+1])
    cmd = f"position startpos moves {partial}"
    send(cmd)
    send("d")  # display board
    time.sleep(0.3)
    # Read display output
    lines = []
    t0 = time.time()
    while time.time() - t0 < 2:
        line = p.stdout.readline().strip()
        if line: 
            lines.append(line)
            print(f"  < {line}")
        if "Fen:" in line:
            # Read a few more lines then stop
            for _ in range(5):
                line = p.stdout.readline().strip()
                if line:
                    lines.append(line)
                    print(f"  < {line}")
            break
    print(f"  -- After move {i+1}: {move_list[i]} --")
    print()

# Now do a search from the full position
send(f"position startpos moves {moves}")
send("go movetime 200")
read_until("bestmove", 10)

send("quit")
p.wait(timeout=3)
