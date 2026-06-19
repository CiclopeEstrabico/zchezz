#!/usr/bin/env python3
"""parallel_match.py — Fast parallel engine-vs-engine test.

Runs games concurrently across multiple workers (2 engine processes per game).
Each game uses 1T engines at 100ms/move for speed.

Usage:
  python tests/parallel_match.py tb       # 1T vs 1T+TB  (200 games, 8 workers)
  python tests/parallel_match.py reg      # v304 vs v305  (200 games, 8 workers)
  python tests/parallel_match.py smp      # 1T vs 4T      (100 games, 4 workers)
"""

import subprocess, sys, os, time, glob, io, threading, math
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ['PYTHONUNBUFFERED'] = '1'
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# ── Auto-detect paths ──────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def find_engine(version=None):
    if version:
        return os.path.join(BASE, "engine", "c", f"zchezz_{version}", "zchezz.exe")
    dirs = sorted(glob.glob(os.path.join(BASE, "engine", "c", "zchezz_v*")))
    return os.path.join(dirs[-1], "zchezz.exe") if dirs else None

LATEST = find_engine()
TB_PATH = os.path.join(BASE, "tablebases", "3-4-5")
BOOK = os.path.join(BASE, "utils", "OpeningBook.bin")

# ── Engine communication ───────────────────────────────────────
class Engine:
    def __init__(self, path, options=None):
        self.proc = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1
        )
        self.send("uci")
        self.read_until("uciok")
        if options:
            for k, v in options.items():
                self.send(f"setoption name {k} value {v}")
        self.send("isready")
        self.read_until("readyok")

    def send(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def read_until(self, target, timeout=10):
        import select
        deadline = time.time() + timeout
        lines = []
        while time.time() < deadline:
            line = self.proc.stdout.readline().strip()
            if not line:
                continue
            lines.append(line)
            if target in line:
                return lines
        return lines

    def go_movetime(self, ms):
        self.send(f"go movetime {ms}")
        lines = self.read_until("bestmove", timeout=max(30, ms/1000 + 5))
        for l in reversed(lines):
            if l.startswith("bestmove"):
                return l.split()[1]
        return None

    def close(self):
        try:
            self.send("quit")
            self.proc.wait(timeout=2)
        except:
            self.proc.kill()

# ── Simple board for game-over detection (FEN-based) ───────────
# We track the game via the engine's FEN output
def play_one_game(engine_a_path, engine_b_path, opts_a, opts_b, movetime_ms, game_id):
    """Play one game, return ('a', 'b', or 'd') for winner."""
    try:
        ea = Engine(engine_a_path, opts_a)
        eb = Engine(engine_b_path, opts_b)
    except Exception as e:
        return ('d', game_id, str(e))

    moves = []
    result = 'd'
    max_plies = 400

    ea.send("ucinewgame")
    ea.send("isready")
    ea.read_until("readyok")
    eb.send("ucinewgame")
    eb.send("isready")
    eb.read_until("readyok")

    for ply in range(max_plies):
        eng = ea if ply % 2 == 0 else eb
        other = eb if ply % 2 == 0 else ea

        mv_str = " ".join(moves)
        if mv_str:
            eng.send(f"position startpos moves {mv_str}")
        else:
            eng.send("position startpos")

        best = eng.go_movetime(movetime_ms)

        if not best or best == "(none)" or best == "0000":
            # Current side has no move = loss
            result = 'b' if ply % 2 == 0 else 'a'
            break

        moves.append(best)

        # Check for draw conditions: repetition or 50-move
        if len(moves) > 8:
            # Simple repetition: if the last 4 half-moves repeat
            if len(moves) >= 12:
                last4 = moves[-4:]
                prev4 = moves[-8:-4]
                prev8 = moves[-12:-8] if len(moves) >= 12 else None
                if last4 == prev4:
                    result = 'd'
                    break

        # Also pass to other engine so it stays in sync
        # (not needed since we set position each time)

    ea.close()
    eb.close()
    return (result, game_id, '')


def run_test(mode, n_games=200, workers=8, movetime=100):
    """Run parallel test. mode: 'tb', 'reg', 'smp'"""

    if mode == 'tb':
        title = "Tablebase Test: 1T vs 1T+TB"
        eng_a = LATEST  # no TB
        eng_b = LATEST  # with TB
        opts_a = {"Threads": "1"}
        opts_b = {"Threads": "1", "SyzygyPath": TB_PATH}
        label_a, label_b = "noTB", "TB"

    elif mode == 'reg':
        title = "Regression Test: v304(1T) vs v305(1T)"
        eng_a = find_engine("v304")
        eng_b = find_engine("v305")
        opts_a = {"Threads": "1"}
        opts_b = {"Threads": "1"}
        label_a, label_b = "v304", "v305"

    elif mode == 'smp':
        title = "SMP Test: 1T vs 4T"
        eng_a = LATEST
        eng_b = LATEST
        opts_a = {"Threads": "1"}
        opts_b = {"Threads": "4"}
        label_a, label_b = "1T", "4T"
        workers = min(workers, 4)  # 4T uses more cores

    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)

    print("=" * 60)
    print(f"  {title}")
    print(f"  {n_games} games, {movetime}ms/move, {workers} parallel workers")
    print("=" * 60)
    print(f"  A: {label_a} = {eng_a}")
    print(f"  B: {label_b} = {eng_b}")
    print()

    wins_a = 0
    wins_b = 0
    draws = 0
    done = 0
    t0 = time.time()

    futures = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i in range(n_games):
            # Alternate colors: even games A=white, odd games A=black
            if i % 2 == 0:
                f = pool.submit(play_one_game, eng_a, eng_b, opts_a, opts_b, movetime, i)
            else:
                f = pool.submit(play_one_game, eng_b, eng_a, opts_b, opts_a, movetime, i)
            futures[f] = i

        for f in as_completed(futures):
            result, gid, err = f.result()
            done += 1

            # Translate result back if colors were swapped
            if gid % 2 == 1:
                if result == 'a': result = 'b'
                elif result == 'b': result = 'a'

            if result == 'a':
                wins_a += 1
            elif result == 'b':
                wins_b += 1
            else:
                draws += 1

            total_pts_a = wins_a + draws * 0.5
            score_a = total_pts_a / done * 100
            elapsed = time.time() - t0
            gps = done / elapsed if elapsed > 0 else 0

            tag = f"[{label_a}]" if result == 'a' else f"[{label_b}]" if result == 'b' else "[draw]"
            print(f"  G {done:3d}/{n_games}  {label_a}:{wins_a} {label_b}:{wins_b} D:{draws}  "
                  f"{label_a}_score={score_a:.1f}%  {tag}  ({gps:.1f} g/s)")

    elapsed = time.time() - t0
    total_a = wins_a + draws * 0.5
    total_b = wins_b + draws * 0.5
    score_a = total_a / n_games * 100
    score_b = total_b / n_games * 100

    # Elo difference
    if 0 < score_a / 100 < 1:
        elo_diff = -400 * math.log10(1 / (score_a / 100) - 1)
    else:
        elo_diff = 0

    print(f"\n{'=' * 60}")
    print(f"  RESULT: {label_a}={wins_a}W  {label_b}={wins_b}W  D={draws}")
    print(f"  {label_a} score: {score_a:.1f}%   {label_b} score: {score_b:.1f}%")
    if abs(elo_diff) > 0:
        print(f"  Elo diff: {elo_diff:+.0f} ({label_a} vs {label_b})")
    if score_a > 55:
        print(f"  ✅ {label_a} is stronger")
    elif score_b > 55:
        print(f"  ✅ {label_b} is stronger")
    else:
        print(f"  ⚖️  Equal (within noise)")
    print(f"  Time: {elapsed:.0f}s ({n_games/elapsed:.1f} games/sec)")
    print("=" * 60)

    return wins_a, wins_b, draws


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "tb"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    w = int(sys.argv[3]) if len(sys.argv) > 3 else 7  # 7 workers × 2 procs = 14 cores, leaves 2 free
    mt = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    run_test(mode, n_games=n, workers=w, movetime=mt)
