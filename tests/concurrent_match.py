#!/usr/bin/env python3
"""Concurrent match runner: runs N independent quick_match instances in parallel.

Usage:
    python concurrent_match.py ENGINE_A ENGINE_B NAME_A NAME_B TOTAL_GAMES WORKERS [OPENINGS_PGN]

Example (400 games, 4 workers = 4x100 games in parallel):
    python concurrent_match.py eng_a.exe eng_b.exe v216K v215 400 4 openings.pgn
"""
import subprocess, sys, os, time, math, re, tempfile, threading, glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elo_calc import elo_difference

# Auto-detect the latest engine version
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE_C_DIR = os.path.join(BASE_DIR, "engine", "c")
_engine_dirs = sorted(glob.glob(os.path.join(ENGINE_C_DIR, "zchezz_v*")))
_latest = _engine_dirs[-1] if _engine_dirs else os.path.join(ENGINE_C_DIR, "zchezz_v305")
_default_exe = os.path.join(_latest, "zchezz.exe")

ENGINE_A = sys.argv[1] if len(sys.argv) > 1 else _default_exe
ENGINE_B = sys.argv[2] if len(sys.argv) > 2 else _default_exe
NAME_A   = sys.argv[3] if len(sys.argv) > 3 else "A"
NAME_B   = sys.argv[4] if len(sys.argv) > 4 else "B"
TOTAL_GAMES = int(sys.argv[5]) if len(sys.argv) > 5 else 400
WORKERS  = int(sys.argv[6]) if len(sys.argv) > 6 else 4
OPENINGS_PGN = sys.argv[7] if len(sys.argv) > 7 else r"c:\Zchezz\openings\Blitz_Testing_4moves.pgn"

QUICK_MATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quick_match.py")



# Results aggregator
lock = threading.Lock()
results = {"wins_a": 0, "draws": 0, "wins_b": 0, "done": 0}

def run_worker(worker_id, games_per_worker):
    """Run a quick_match subprocess and parse results."""
    cmd = [
        sys.executable, "-u", QUICK_MATCH,
        ENGINE_A, ENGINE_B, NAME_A, NAME_B,
        str(games_per_worker), OPENINGS_PGN
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, bufsize=1)
    
    local_w = local_d = local_l = 0
    for line in proc.stdout:
        line = line.strip()
        # Parse final result line: "  W=23 D=11 L=16"
        m = re.search(r'W=(\d+)\s+D=(\d+)\s+L=(\d+)', line)
        if m:
            local_w = int(m.group(1))
            local_d = int(m.group(2))
            local_l = int(m.group(3))
        # Print progress from workers with prefix
        if line.startswith('[') or 'FINAL' in line or 'ELO' in line:
            with lock:
                print(f"  [W{worker_id}] {line}", flush=True)
    
    proc.wait()
    
    with lock:
        results["wins_a"] += local_w
        results["draws"] += local_d
        results["wins_b"] += local_l
        results["done"] += 1
        
        total = results["wins_a"] + results["draws"] + results["wins_b"]
        elo, err, _ = elo_difference(results["wins_a"], results["draws"], results["wins_b"])
        print(f"  [AGG {results['done']}/{WORKERS}] {total} games: "
              f"W={results['wins_a']} D={results['draws']} L={results['wins_b']} "
              f"| ELO: {elo:+.1f} ±{err:.1f}", flush=True)

def main():
    games_per_worker = TOTAL_GAMES // WORKERS
    # Make even for paired openings
    if games_per_worker % 2 != 0:
        games_per_worker -= 1
    
    print(f"Concurrent Match: {NAME_A} vs {NAME_B}", flush=True)
    print(f"Total: {games_per_worker * WORKERS} games | Workers: {WORKERS} | "
          f"Games/worker: {games_per_worker}", flush=True)
    print(f"Openings: {os.path.basename(OPENINGS_PGN)}", flush=True)
    print("=" * 70, flush=True)
    
    t0 = time.time()
    
    threads = []
    for i in range(WORKERS):
        t = threading.Thread(target=run_worker, args=(i+1, games_per_worker))
        t.start()
        threads.append(t)
        time.sleep(0.5)  # Stagger starts slightly
    
    for t in threads:
        t.join()
    
    elapsed = time.time() - t0
    total = results["wins_a"] + results["draws"] + results["wins_b"]
    pts_a = results["wins_a"] + results["draws"] * 0.5
    pct = pts_a / total * 100 if total > 0 else 50
    elo, err, _ = elo_difference(results["wins_a"], results["draws"], results["wins_b"])
    
    print("\n" + "=" * 70, flush=True)
    print(f"FINAL: {NAME_A} vs {NAME_B} ({total} games)", flush=True)
    print(f"  {NAME_A}: {pts_a:.1f}/{total} ({pct:.1f}%)", flush=True)
    print(f"  W={results['wins_a']} D={results['draws']} L={results['wins_b']}", flush=True)
    print(f"  ELO difference: {elo:+.1f} ±{err:.1f}", flush=True)
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    print("=" * 70, flush=True)
    
    return 0 if elo > 0 else 1

if __name__ == "__main__":
    sys.exit(main())
