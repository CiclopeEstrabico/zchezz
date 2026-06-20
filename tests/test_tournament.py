#!/usr/bin/env python3
"""
3-way round-robin tournament v2 (FEN-based):
  A = Zchezz 1 Thread (no TB)
  B = Zchezz 4 Threads (no TB)
  C = Zchezz 4 Threads + TB

200 games per pairing × 3 pairings = 600 games total.
FIX: Send position fen <currentFEN> instead of startpos+moves
     (engine ucinewgame doesn't fully reset hash state)
"""
import subprocess, threading, time, sys, io, math, random, traceback, os, glob
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elo_calc import elo_difference as _elo_diff_fn

def find_latest_engine():
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engine", "c")
    dirs = sorted(glob.glob(os.path.join(base, "zchezz_v*")))
    return dirs[-1] if dirs else None

ENGINE_DIR = find_latest_engine()
ENGINE   = os.path.join(ENGINE_DIR, 'zchezz.exe')
NNUE     = os.path.join(ENGINE_DIR, 'nnue_weights.bin')
TB_PATH  = r'c:\Zchezz\tablebases'
BOOK     = r'c:\Zchezz\utils\OpeningBook.bin'

GAMES_PER_PAIR = 200
MOVETIME_MS    = 300
MAX_PLIES      = 400
ADJ_WIN_CP     = 500
ADJ_WIN_N      = 5
ADJ_DRAW_CP    = 15
ADJ_DRAW_N     = 8

import chess
import chess.polyglot

CONFIGS = {
    'A_1T': {'NNUE': NNUE, 'Threads': '1'},
    'B_4T': {'NNUE': NNUE, 'Threads': '4'},
    'C_4T_TB': {
        'NNUE': NNUE, 'Threads': '4',
        'SyzygyPath': TB_PATH, 'SyzygyProbeDepth': '1', 'SyzygyProbeLimit': '6',
    },
}
PAIRINGS = [('A_1T', 'B_4T'), ('A_1T', 'C_4T_TB'), ('B_4T', 'C_4T_TB')]

class UCIEngine:
    def __init__(self, name, options):
        self.name = name
        self.options = options
        self.crashes = 0
        self.timeouts = 0
        self.illegal = 0
        self._start()

    def _start(self):
        import os
        eng_dir = os.path.dirname(ENGINE)
        self.proc = subprocess.Popen(
            [ENGINE], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1, cwd=eng_dir)
        self.out = []
        self.lock = threading.Lock()
        self.alive = True
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=lambda:[None for _ in self.proc.stderr], daemon=True).start()
        time.sleep(0.1)  # let reader thread start
        self._send('uci')
        r = self._wait('uciok', 60)
        if not r or not any('uciok' in l for l in r):
            print(f"    [WARN] {self.name}: uciok timeout! alive={self.is_alive()}", flush=True)
            return
        for k, v in self.options.items():
            self._send(f'setoption name {k} value {v}')
        self._send('isready')
        r = self._wait('readyok', 120)
        if not r or not any('readyok' in l for l in r):
            print(f"    [WARN] {self.name}: readyok timeout! alive={self.is_alive()}", flush=True)

    def _read(self):
        try:
            for line in self.proc.stdout:
                with self.lock: self.out.append(line.strip())
        except: self.alive = False

    def _send(self, cmd):
        try: self.proc.stdin.write(cmd + '\n'); self.proc.stdin.flush()
        except: self.alive = False

    def _wait(self, token, timeout=30):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                for i, l in enumerate(self.out):
                    if token in l:
                        result = self.out[:i+1]; self.out = self.out[i+1:]; return result
            time.sleep(0.005)
        self.timeouts += 1
        with self.lock: r = list(self.out); self.out.clear()
        return r

    def is_alive(self): return self.alive and self.proc.poll() is None

    def go_fen(self, fen, mt):
        """Send just the FEN (no move list) to avoid stale hash issues."""
        if not self.is_alive():
            self.crashes += 1; self.restart(); return None, 0
        self._send(f'position fen {fen}')
        self._send(f'go movetime {mt}')
        lines = self._wait('bestmove', 30)
        bm, sc = None, 0
        for l in lines:
            if l.startswith('bestmove'):
                parts = l.split(); bm = parts[1] if len(parts) > 1 else None
            if 'score cp' in l:
                parts = l.split(); idx = parts.index('cp'); sc = int(parts[idx+1])
            elif 'score mate' in l:
                parts = l.split(); idx = parts.index('mate')
                sc = 30000 if int(parts[idx+1]) > 0 else -30000
        return bm, sc

    def newgame(self):
        """Send ucinewgame + isready (FEN-mode avoids stale hash issue)."""
        if not self.is_alive():
            self.crashes += 1; self.restart(); return
        self._send('ucinewgame')
        self._send('isready')
        r = self._wait('readyok', 10)
        if not r or not any('readyok' in l for l in r):
            # Engine didn't respond — restart it
            self.crashes += 1; self.restart()

    def restart(self):
        try: self.proc.kill()
        except: pass
        try: self._start()
        except: self.alive = False

    def quit(self):
        try: self._send('quit'); self.proc.wait(timeout=3)
        except:
            try: self.proc.kill()
            except: pass

def book_move(board):
    try:
        with chess.polyglot.open_reader(BOOK) as reader:
            entries = list(reader.find_all(board))
            if entries:
                total = sum(e.weight for e in entries)
                r = random.random() * total
                c = 0
                for e in entries:
                    c += e.weight
                    if r <= c: return e.move
    except: pass
    return None

def play_game(eng_w, eng_b):
    board = chess.Board()
    bk = random.randint(4, 10)
    for _ in range(bk):
        if board.is_game_over(): break
        mv = book_move(board)
        if mv is None or mv not in board.legal_moves: break
        board.push(mv)

    win_w, win_b, draw_streak = 0, 0, 0
    ply = len(board.move_stack)

    while not board.is_game_over() and ply < MAX_PLIES:
        is_wt = board.turn == chess.WHITE
        eng = eng_w if is_wt else eng_b
        fen = board.fen()

        if not eng.is_alive():
            eng.crashes += 1; eng.restart()
            return ('b' if is_wt else 'w'), 'crash', ply

        bm, sc = eng.go_fen(fen, MOVETIME_MS)

        if not bm or bm in ('(none)', '0000'):
            return ('b' if is_wt else 'w'), 'no_bestmove', ply

        try:
            mv = chess.Move.from_uci(bm)
            if mv not in board.legal_moves:
                for lm in board.legal_moves:
                    if lm.uci()[:4] == bm[:4]: mv = lm; break
                else:
                    eng.illegal += 1
                    return ('b' if is_wt else 'w'), 'illegal_move', ply
        except:
            eng.illegal += 1
            return ('b' if is_wt else 'w'), 'bad_uci', ply

        board.push(mv); ply += 1

        if sc < -ADJ_WIN_CP:
            if is_wt: win_b += 1; win_w = 0
            else: win_w += 1; win_b = 0
        elif sc > ADJ_WIN_CP:
            if is_wt: win_w += 1; win_b = 0
            else: win_b += 1; win_w = 0
        else: win_w = 0; win_b = 0

        if abs(sc) < ADJ_DRAW_CP: draw_streak += 1
        else: draw_streak = 0

        if win_w >= ADJ_WIN_N: return 'w', 'resign', ply
        if win_b >= ADJ_WIN_N: return 'b', 'resign', ply
        if draw_streak >= ADJ_DRAW_N * 2 and ply > 60: return 'd', 'adjudication', ply

    r = board.result()
    if r == '1-0': return 'w', 'checkmate', ply
    elif r == '0-1': return 'b', 'checkmate', ply
    return 'd', 'max_plies' if ply >= MAX_PLIES else 'stalemate', ply

def elo_diff(w, l, d):
    elo, ci, _ = _elo_diff_fn(w, d, l)
    return elo, ci

def main():
    random.seed(42)
    print("=" * 65)
    print("  ZCHEZZ v3.04 — 3-Way Round Robin Tournament v2")
    print(f"  A = 1 Thread    B = 4 Threads    C = 4 Threads + TB")
    print(f"  {GAMES_PER_PAIR} games/pair x 3 pairs = {GAMES_PER_PAIR*3} games")
    print(f"  Movetime: {MOVETIME_MS}ms  Book: 4-10 plies  FEN-mode")
    print("=" * 65)

    engines = {}
    for name, opts in CONFIGS.items():
        print(f"  Starting {name}...", end=' ', flush=True)
        engines[name] = UCIEngine(name, opts)
        print("OK", flush=True)
    print(flush=True)

    results = {}
    for a, b in PAIRINGS:
        results[(a, b)] = {'w': 0, 'l': 0, 'd': 0}

    term_reasons = {}
    total_plies = 0
    total_games = 0

    for pair_idx, (name_a, name_b) in enumerate(PAIRINGS):
        eng_a, eng_b = engines[name_a], engines[name_b]
        print(f"\n{'—'*65}")
        print(f"  Match {pair_idx+1}/3: {name_a} vs {name_b}  ({GAMES_PER_PAIR} games)")
        print(f"{'—'*65}")

        for g in range(GAMES_PER_PAIR):
            a_white = (g % 2 == 0)
            w_eng = eng_a if a_white else eng_b
            b_eng = eng_b if a_white else eng_a
            w_eng.newgame(); b_eng.newgame()

            try:
                result, reason, ply = play_game(w_eng, b_eng)
            except Exception as ex:
                result, reason, ply = 'd', f'exception', 0
                traceback.print_exc()

            total_plies += ply; total_games += 1
            term_reasons[reason] = term_reasons.get(reason, 0) + 1

            r = results[(name_a, name_b)]
            if result == 'w':
                if a_white: r['w'] += 1
                else: r['l'] += 1
            elif result == 'b':
                if not a_white: r['w'] += 1
                else: r['l'] += 1
            else: r['d'] += 1

            if (g+1) % 20 == 0 or result != 'd' or (g+1) == GAMES_PER_PAIR:
                t = r['w'] + r['l'] + r['d']
                elo, ese = elo_diff(r['w'], r['l'], r['d'])
                pct = (r['w'] + r['d']*0.5) / t * 100 if t > 0 else 50
                tag = ""
                if result != 'd':
                    winner = name_a if (result == 'w' and a_white) or (result == 'b' and not a_white) else name_b
                    tag = f" [{winner} {reason}]"
                print(f"  G{g+1:3d}/{GAMES_PER_PAIR}  +{r['w']} ={r['d']} -{r['l']}  "
                      f"{pct:.0f}%  E={elo:+.0f}+/-{ese:.0f}{tag}")

    print("\n" + "=" * 65)
    print("  FINAL RESULTS")
    print("=" * 65)

    names = list(CONFIGS.keys())
    print(f"\n  {'':12s} | {'A_1T':>10s} | {'B_4T':>10s} | {'C_4T_TB':>10s} | {'Score':>7s}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*7}")
    for n in names:
        row = f"  {n:12s} |"
        total_pts, total_g = 0.0, 0
        for m in names:
            if n == m: row += f" {'---':>10s} |"; continue
            if (n, m) in results:
                r = results[(n, m)]; w, l, d = r['w'], r['l'], r['d']
            else:
                r = results[(m, n)]; w, l, d = r['l'], r['w'], r['d']
            pts = w + d * 0.5; g = w + l + d
            total_pts += pts; total_g += g
            row += f" {w:>2d}W{d:>3d}D{l:>2d}L |"
        pct = total_pts / total_g * 100 if total_g > 0 else 50
        row += f" {pct:5.1f}%"
        print(row)

    print(f"\n  ELO Differences:")
    for name_a, name_b in PAIRINGS:
        r = results[(name_a, name_b)]
        elo, ese = elo_diff(r['w'], r['l'], r['d'])
        t = r['w'] + r['l'] + r['d']
        pct = (r['w'] + r['d']*0.5) / t * 100 if t > 0 else 50
        print(f"    {name_a} vs {name_b}: +{r['w']} ={r['d']} -{r['l']}  "
              f"({pct:.1f}%)  ELO: {elo:+.0f} +/-{ese:.0f}")

    r_mt = results[('A_1T', 'B_4T')]
    elo_mt, ese_mt = elo_diff(r_mt['l'], r_mt['w'], r_mt['d'])
    print(f"\n  * Multithread boost (1T->4T): {elo_mt:+.0f} +/-{ese_mt:.0f} ELO")
    r_tb = results[('B_4T', 'C_4T_TB')]
    elo_tb, ese_tb = elo_diff(r_tb['l'], r_tb['w'], r_tb['d'])
    print(f"  * Tablebase boost (4T->4T+TB): {elo_tb:+.0f} +/-{ese_tb:.0f} ELO")
    r_both = results[('A_1T', 'C_4T_TB')]
    elo_both, ese_both = elo_diff(r_both['l'], r_both['w'], r_both['d'])
    print(f"  * Combined boost (1T->4T+TB): {elo_both:+.0f} +/-{ese_both:.0f} ELO")

    print(f"\n  Health Report:")
    for n in names:
        e = engines[n]
        print(f"    {n}: crashes={e.crashes} timeouts={e.timeouts} illegal={e.illegal}")
    print(f"\n  Termination reasons:")
    for k, v in sorted(term_reasons.items(), key=lambda x: -x[1]):
        print(f"    {k}: {v}")
    avg_ply = total_plies / total_games if total_games > 0 else 0
    print(f"\n  Total games: {total_games}  Avg plies: {avg_ply:.0f}")
    print("=" * 65)
    for e in engines.values(): e.quit()

if __name__ == '__main__':
    main()
