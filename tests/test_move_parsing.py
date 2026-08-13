"""Test: verify Zchezz correctly applies long move sequences"""
import subprocess, time, os, threading

# ═══════════════ CONFIGURATION ═══════════════
# Edit these and run with no arguments; every one is also a CLI flag that
# overrides it (CLAUDE.md rule 8, see the COMMAND LINE block below).
REPO_DIR = r"c:\Zchezz"                          # working directory the engine is launched from
ENGINE_EXE = r"engine\c\zchezz_v400\zchezz.exe"  # engine binary under test
NNUE_WEIGHTS = r"C:\Zchezz\engine\c\zchezz_v400\nnue_weights.bin"  # NNUE weights loaded via setoption
DISPLAY_READ_TIMEOUT_S = 2   # seconds to read the "d" board-display output
SEARCH_TIMEOUT_S = 10        # seconds to wait for "bestmove"
# The exact moves from the failed game this test was written for. Replace it
# (or pass --moves) to replay any other suspect game through the same check.
MOVES = "e2e4 c7c5 d2d4 c5d4 g1f3 g7g6 f3d4 g8f6 b1c3 b8c6 d4c6 b7c6 e4e5 f6h5 f1e2 h5g7 e1g1 g7e6 f2f4 f8g7 g2g4 g6g5 f4f5 e6f4 c3e4 d8b6 g1h1 g7e5 e4g5 f4e2 d1e2 f7f6 c2c4 c6c5 g5f3 c8b7 h1g1 e5c7 g4g5 b6c6 c1d2 e8c8 e2e7 h8f8 e7e2 f6g5 f3d4 c5d4 e2e4 c6e4"
# ═══════════════════════════════════════════

# ═══════════════ COMMAND LINE ═══════════════
# `python tests/test_move_parsing.py` runs exactly what the block above says;
# the flags only override it. See utils/cliconf.py.
# ════════════════════════════════════════════
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
from cliconf import override_from_cli  # noqa: E402

CLI = [
    ("REPO_DIR",     "--repo-dir", str,  "working directory the engine is launched from"),
    ("ENGINE_EXE",   "--exe",      str,  "engine binary under test"),
    ("NNUE_WEIGHTS", "--nnue",     str,  "NNUE weights file loaded via setoption"),
    ("MOVES",        "--moves",    str,  "space-separated UCI move list to replay"),
    ("DISPLAY_READ_TIMEOUT_S", "--display-timeout", float, "seconds to read the 'd' output"),
    ("SEARCH_TIMEOUT_S",       "--search-timeout",  float, "seconds to wait for bestmove"),
]

if __name__ == "__main__":
    override_from_cli(globals(), CLI, description=__doc__, prog="test_move_parsing.py")

os.chdir(REPO_DIR)

moves = MOVES

exe = ENGINE_EXE
p = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
                     cwd=os.path.dirname(os.path.abspath(exe)))
threading.Thread(target=lambda: [print("STDERR:", l.strip()) for l in p.stderr], daemon=True).start()

def send(cmd):
    print(f">>> {cmd[:200]}{'...' if len(cmd) > 200 else ''}")
    p.stdin.write(cmd + "\n"); p.stdin.flush()

def read_until(keyword, timeout=DISPLAY_READ_TIMEOUT_S):
    t0 = time.time()
    while time.time() - t0 < timeout:
        line = p.stdout.readline().strip()
        if line: print(f"  < {line[:200]}")
        if keyword in line: return line
    return None

send("uci"); read_until("uciok")
send(f"setoption name NNUE value {NNUE_WEIGHTS}")
send("isready"); read_until("readyok", SEARCH_TIMEOUT_S)

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
    while time.time() - t0 < DISPLAY_READ_TIMEOUT_S:
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
read_until("bestmove", SEARCH_TIMEOUT_S)

send("quit")
p.wait(timeout=3)
