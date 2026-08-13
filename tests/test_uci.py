import subprocess, time, re, sys, os

# ═══════════════ CONFIGURATION ═══════════════
# No CLI in this script — edit these constants directly.
REPO_DIR = r"c:\Zchezz"                          # working directory the engine is launched from
ENGINE_EXE = r"engine\c\zchezz_v400\zchezz.exe"  # engine binary under test (relative to REPO_DIR)
NNUE_WEIGHTS = r"C:\Zchezz\engine\c\zchezz_v400\nnue_weights.bin"  # NNUE weights loaded via setoption
SYZYGY_PATH = r"C:\Zchezz\tablebases"   # Syzygy tablebase directory loaded via setoption
READYOK_TIMEOUT_S = 5   # seconds to wait for "readyok"/"uciok" before giving up
# ═══════════════════════════════════════════

os.chdir(REPO_DIR)

print("=== UCI ENGINE TEST ===\n")

exe = ENGINE_EXE
p = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

def send(cmd):
    print(f">>> {cmd}")
    p.stdin.write(cmd + "\n")
    p.stdin.flush()

def read_until(keyword, timeout=READYOK_TIMEOUT_S):
    lines = []
    t0 = time.time()
    while time.time() - t0 < timeout:
        line = p.stdout.readline().strip()
        if line:
            print(f"  < {line}")
            lines.append(line)
        if keyword in line:
            break
    return lines

PASS = 0
FAIL = 0

def check(name, condition):
    global PASS, FAIL
    if condition:
        print(f"  [PASS] {name}")
        PASS += 1
    else:
        print(f"  [FAIL] {name}")
        FAIL += 1

# Test 1: UCI handshake
print("--- TEST 1: UCI handshake ---")
send("uci")
lines = read_until("uciok")
check("uciok received", any("uciok" in l for l in lines))
check("NNUE option present", any("option name NNUE" in l for l in lines))
check("SyzygyPath option present", any("option name SyzygyPath" in l for l in lines))
print()

# Test 2: Set options + isready
print("--- TEST 2: Set NNUE + Syzygy + isready ---")
send(f"setoption name NNUE value {NNUE_WEIGHTS}")
send(f"setoption name SyzygyPath value {SYZYGY_PATH}")
send("isready")
lines = read_until("readyok", timeout=10)
check("readyok received", any("readyok" in l for l in lines))
print()

# Test 3: Startpos search - verify NNUE eval
print("--- TEST 3: Search startpos movetime 500 ---")
send("position startpos")
send("go movetime 500")
lines = read_until("bestmove", timeout=10)
scores = [l for l in lines if "score cp" in l]
bestmove_lines = [l for l in lines if "bestmove" in l]

nnue_working = False
bm = None
if scores:
    m = re.search(r"score cp (-?\d+)", scores[-1])
    cp = int(m.group(1)) if m else None
    print(f"  Last eval: {cp} cp")
    nnue_working = (cp is not None and cp != 0)
if bestmove_lines:
    bm = bestmove_lines[0].split()[1]
    print(f"  Best move: {bm}")

check("NNUE eval is non-zero", nnue_working)
reasonable = ["e2e4","d2d4","c2c4","g1f3","b1c3","e2e3","d2d3"]
check("Reasonable opening move", bm in reasonable)
print()

# Test 4: Play 20 half-moves from startpos (both sides)
print("--- TEST 4: Play 20 half-moves ---")
moves = []
all_ok = True
non_zero_evals = 0
for i in range(20):
    pos = "position startpos"
    if moves:
        pos += " moves " + " ".join(moves)
    send(pos)
    send("go movetime 100")
    lines = read_until("bestmove", timeout=10)
    bm_line = [l for l in lines if "bestmove" in l]
    if bm_line:
        mv = bm_line[0].split()[1]
        moves.append(mv)
        sc_lines = [l for l in lines if "score cp" in l]
        if sc_lines:
            m = re.search(r"score cp (-?\d+)", sc_lines[-1])
            if m and int(m.group(1)) != 0:
                non_zero_evals += 1
        depth_lines = [l for l in lines if "info depth" in l]
        last_depth = 0
        if depth_lines:
            dm = re.search(r"depth (\d+)", depth_lines[-1])
            last_depth = int(dm.group(1)) if dm else 0
        print(f"  Move {i+1}: {mv} (depth {last_depth})")
    else:
        print(f"  Move {i+1}: FAILED!")
        all_ok = False
        break

check("All 20 moves played", len(moves) == 20)
check("NNUE evals non-zero (>10 of 20)", non_zero_evals > 10)
print(f"  Moves: {' '.join(moves)}")
print()

# Test 5: TB endgame (KRK)
print("--- TEST 5: Tablebase probe (KRK) ---")
send("position fen 8/8/8/4k3/8/8/8/R3K3 w - - 0 1")
send("go movetime 500")
lines = read_until("bestmove", timeout=10)
tb_lines = [l for l in lines if "tbhits" in l]
tbhits = 0
if tb_lines:
    m = re.search(r"tbhits (\d+)", tb_lines[-1])
    tbhits = int(m.group(1)) if m else 0
    print(f"  TB hits: {tbhits}")
bm_line = [l for l in lines if "bestmove" in l]
if bm_line:
    print(f"  Best move: {bm_line[0].split()[1]}")
check("TB probed (tbhits > 0)", tbhits > 0)
print()

# Test 6: Auto-load NNUE (no setoption, just isready)
print("--- TEST 6: NNUE auto-load (new engine, no setoption) ---")
send("quit")
p.wait(timeout=3)

# Start a fresh engine WITHOUT sending setoption name NNUE
p2 = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
def send2(cmd):
    print(f">>> {cmd}")
    p2.stdin.write(cmd + "\n")
    p2.stdin.flush()

def read2(keyword, timeout=READYOK_TIMEOUT_S):
    lines = []
    t0 = time.time()
    while time.time() - t0 < timeout:
        line = p2.stdout.readline().strip()
        if line:
            print(f"  < {line}")
            lines.append(line)
        if keyword in line:
            break
    return lines

send2("uci")
read2("uciok")
send2("isready")  # This should trigger auto_load_nnue
read2("readyok", timeout=10)
send2("position startpos")
send2("go movetime 200")
lines = read2("bestmove", timeout=10)
scores = [l for l in lines if "score cp" in l]
auto_nnue_ok = False
if scores:
    m = re.search(r"score cp (-?\d+)", scores[-1])
    cp = int(m.group(1)) if m else 0
    print(f"  Eval without explicit setoption: {cp} cp")
    auto_nnue_ok = (cp != 0)
bm_line = [l for l in lines if "bestmove" in l]
if bm_line:
    auto_move = bm_line[0].split()[1]
    print(f"  Move: {auto_move}")
    # Without NNUE, it plays a2a3 garbage. With NNUE, reasonable move
    auto_nnue_ok = auto_nnue_ok and auto_move in reasonable
check("NNUE auto-loaded (non-zero eval + good move)", auto_nnue_ok)
send2("quit")
p2.wait(timeout=3)
print()

# Summary
print("=" * 50)
print(f"RESULTS: {PASS} passed, {FAIL} failed")
if FAIL > 0:
    print("*** SOME TESTS FAILED ***")
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
    sys.exit(0)
