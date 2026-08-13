"""
run_suite.py — Zchezz UCI Engine Test Suite Runner
======================================================

Runs one or two UCI engines against EPD tactical/positional test suites
(WAC, STS, ERET, etc., anything dropped into SUITES_DIR) and scores each
position pass/fail (or partial credit for STS-style `c0` scored-move
lines) — this measures TACTICAL SOLVING STRENGTH at a fixed depth/movetime,
not playing strength. It does not play games, has no opponent, and
produces no ELO — for that, see run_tournament.py / run_tournament_quick.py
(H2H and anchor ELO) or run_arena.py (native SPRT gate between versions).
Those tools tell you whether a change made the engine stronger in real
games; this one tells you whether it still finds the textbook tactic in
suite X at depth/movetime Y — useful as a fast, single-position-level
regression check between full game-based runs, and the natural tool to
reach for after a search-heuristic change (pruning, LMR, extensions)
where you want to see WHICH positions started failing, not just an
aggregate ELO delta.

CONFIG-BLOCK-AND-CLI, not CLI-only: configure everything in the CONFIG
block below and just run:

    python run_suite.py

Every CLI flag below OVERRIDES its CONFIG counterpart when passed — CONFIG
is not bypassed, it is the default the CLI patches (CLAUDE.md rule 8: the
CLI's own `default=` is `None` for every flag and args are resolved as
`args.x if args.x is not None else CONFIG_X` in main(), never a second
copy of the literal). Examples:

    python run_suite.py                              # usa tudo do CONFIG
    python run_suite.py --e1 nova.exe --depth 12    # override parcial
    python run_suite.py --n 30 --exclude SSM OTS    # override parcial
    python run_suite.py --movetime 500 --pct 0.50
    python run_suite.py --all --include STS WAC
    python run_suite.py --help

── NON-OBVIOUS BEHAVIOR ────────────────────────────────────────────────────

  SUITES_DIR default is "tests/suites" — moved there (previously a
  top-level path) as part of the repo restructure; if suites go missing,
  check that path first.

  Sampling (SAMPLE_N / SAMPLE_PCT / SAMPLE_ALL) is mutually exclusive —
  set exactly one, leave the others at None/False; SAMPLE_SEED is fixed so
  the SAME sampled positions are used for engine 1 and engine 2 in a
  comparison run, which is what makes the per-suite delta meaningful
  (otherwise a score change could just be sampling noise).

  Each suite runs with a small pool of reusable UCIEngine processes
  (ThreadPoolExecutor + a get_eng()/put_eng() pool, see run_suite()) rather
  than one process per position or one process for the whole suite — this
  is the same "avoid per-unit process spawn cost" tradeoff run_selfplay.py
  makes for games, applied here to individual EPD positions, sized by
  WORKERS (defaults to logical cores - 1).

  Scoring: if an EPD line has a `c0` scored-move list (STS-style, e.g.
  `c0 "Nf3=10, Ng1=6"`), partial credit is awarded per the matched move's
  point value; otherwise it's straight pass/fail against `bm`
  (best-move-must-match) or `am` (avoid-this-move) — see score_position().

OUTPUT: a JSON file (RESULTS_DIR, or --output) with full per-position
results for both engines plus the printed comparison table — no PGN/EPD
export, this tool doesn't produce training data or game records.

── SHARED CONFIGURATION VOCABULARY ─────────────────────────────────────────

Every tests/run_*.py harness uses the SAME name for the same concept, so a
value means the same thing whichever harness you are reading (CLAUDE.md
rule 8: the constant lives in the CONFIGURATION block once, and the CLI's
`default=` IS that constant, never a second copy of the literal).

  GAMES           number of games to play
  CONCURRENCY     parallel GAMES, not threads per game. 0 = autodetect cores
                  in the harnesses that wrap a C exe (arena, selfplay_native)
  MOVETIME_MS     per-move budget in milliseconds
  NODES           per-move node budget; used only when MOVETIME_MS == 0
  MAX_PLIES       half-move cap before a game is scored a draw (400 project-wide)
  SEED            RNG seed for opening cycling / game order
  TT_MB           per-TTable memory budget in MB
  OPENING_FOLDER  openings/lines, walked recursively for .pgn/.epd files
  RESULTS_DIR     tests/<tool>_results — every harness writes under tests/
  SAVE_PGN        write standard PGN game records
  SAVE_EPD        write positions + eval as EPD
  SAVE_BIN        write packed training samples (train/dataset.py SAMPLE_DTYPE)

SAVE_PGN / SAVE_EPD / SAVE_BIN are INDEPENDENT flags, not a choice of one
(CLAUDE.md rule 9: the native path is a faster path for the same job, never a
reduced one — losing PGN because a tool prefers .bin is a regression).

A `DEFAULT_` prefix marks a knob specific to ONE tool (DEFAULT_TEMPERATURE,
DEFAULT_ALPHA, ...) and is deliberately not shared.
"""

# Windows consoles default to cp1252, which cannot encode the em-dashes and box
# characters used in this file's docstring and reports — printing then dies with
# UnicodeEncodeError before any output appears. Force UTF-8 (same as
# tests/bench_nps.py).
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass  # already-redirected or non-reconfigurable stream


import argparse, os, re, subprocess, threading, time, random, json, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  —  edite aqui, sem precisar de linha de comando
# ═══════════════════════════════════════════════════════════════════════════════

# Repo-root-anchored paths, so this script behaves identically no matter which
# directory it is launched from — the same convention run_arena.py and
# run_selfplay_native.py use (they call this REPO_ROOT too).
REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE_DIR = os.path.join(REPO_ROOT, "engine", "c", "zchezz_v400")

# ── Engines ───────────────────────────────────────────────────────────────────
# ENGINE_2 = None  →  modo single-engine (sem tabela comparativa).
# NNUE_*   = None  →  auto-descobre nnue_weights.bin ao lado do binary.
#                      Se não encontrar, roda com eval clássico.

ENGINE_1  = os.path.join(ENGINE_DIR, "zchezz.exe")   # current engine version
ENGINE_2  = None                    # ex: .../zchezz_v314/zchezz.exe para comparação

NNUE_1    = None                    # r"C:\engines\nnue_weights.bin"  ou None
NNUE_2    = None

# ── Controle de busca ─────────────────────────────────────────────────────────
# Se MOVETIME_MS for definido (não None), DEPTH é ignorado.

DEPTH       = 8                     # profundidade de busca
MOVETIME_MS = None                  # ex: 500  (ms por posição); None = usa DEPTH

# ── Diretórios ────────────────────────────────────────────────────────────────
SUITES_DIR  = r"tests\suites"       # pasta com os arquivos .epd
RESULTS_DIR = r"tests\suite_results"  # onde salvar os JSONs de resultado.
                                    # Matches the project-wide layout: every
                                    # harness writes under tests\<tool>_results
                                    # (arena_results, selfplay_results,
                                    # complete_results, quick_results).

# ── Amostragem ────────────────────────────────────────────────────────────────
# Escolha UMA das três. As demais devem ficar como None / False.
#
#   SAMPLE_N   = 50    →  exatamente 50 posições aleatórias por suite
#                          (se N > tamanho da suite, roda a suite inteira)
#   SAMPLE_PCT = 0.20  →  20% de cada suite (mínimo 1 posição)
#   SAMPLE_ALL = True  →  todas as posições, na ordem original do EPD
#
# Se todas ficarem None/False → padrão 10% de cada suite.

SAMPLE_N   = None                   # ex: 50
SAMPLE_PCT = 0.10                   # ex: 0.20  (20%)
SAMPLE_ALL = False                  # True  →  ignora SAMPLE_N e SAMPLE_PCT

SAMPLE_SEED = 42                    # semente fixa → mesmas posições nos dois engines

# ── Seleção de suites ─────────────────────────────────────────────────────────
# []  →  usa todas as suites encontradas em SUITES_DIR
#
# Exemplos:
#   INCLUDE_SUITES = ["STS", "WAC", "ERET"]
#   EXCLUDE_SUITES = ["SSM", "OTS", "MATE5"]

INCLUDE_SUITES = []                 # []  →  todas
EXCLUDE_SUITES = []                 # []  →  nenhuma excluída

# ── Performance ───────────────────────────────────────────────────────────────
WORKERS = max(1, (os.cpu_count() or 4) - 1)   # recomendado: núcleos físicos - 1

# ── Timeout por posição ───────────────────────────────────────────────────────
ENGINE_TIMEOUT_S = 60               # segundos antes de abortar a busca

# ═══════════════════════════════════════════════════════════════════════════════
#  FIM DO CONFIG
# ═══════════════════════════════════════════════════════════════════════════════


# ── EPD PARSER ─────────────────────────────────────────────────────────────────

def _parse_epd_line(line: str):
    line = line.strip()
    if not line or line.startswith('#'):
        return None

    parts = line.split(None, 6)
    if len(parts) < 5:
        return None

    opcode_start = 4
    fen_parts    = parts[:4]
    if len(parts) > 5:
        try:
            int(parts[4]); int(parts[5])
            fen_parts    = parts[:6]
            opcode_start = 6
        except (ValueError, IndexError):
            pass

    fen  = ' '.join(fen_parts)
    if len(fen.split()) == 4:
        fen += ' 0 1'

    rest = ' '.join(parts[opcode_start:]) if len(parts) > opcode_start else ''
    pos  = {'fen': fen, 'bm': [], 'am': [], 'type': 'bm', 'id': '', 'scored_moves': {}}

    for op in re.split(r';', rest):
        op = op.strip()
        if not op:
            continue
        m = re.match(r'^(\w+)\s+(.*)', op, re.DOTALL)
        if not m:
            continue
        key, val = m.group(1).lower(), m.group(2).strip().strip('"')

        if key == 'bm':
            pos['bm'] = val.split(); pos['type'] = 'bm'
        elif key == 'am':
            pos['am'] = val.split(); pos['type'] = 'am'
        elif key == 'id':
            pos['id'] = val
        elif key == 'c0':
            scored = {}
            for tok in re.split(r'[,;]', val):
                tok = tok.strip()
                eq = re.match(r'^(\S+)=(\d+)$', tok)
                if eq:
                    scored[eq.group(1)] = int(eq.group(2)); continue
                sp = re.match(r'^(\S+)\s+(\d+)$', tok)
                if sp:
                    scored[sp.group(1)] = int(sp.group(2))
            pos['scored_moves'] = scored

    if not pos['bm'] and not pos['am']:
        return None
    return pos


def load_epd(path: str):
    positions = []
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            p = _parse_epd_line(line)
            if p:
                if not p['id']:
                    p['id'] = f"{Path(path).stem}.{len(positions)+1:04d}"
                positions.append(p)
    return positions


def discover_suites(suites_dir, include, exclude):
    d = Path(suites_dir)
    if not d.exists():
        print(f"[ERRO] Pasta não encontrada: {d.resolve()}")
        sys.exit(1)

    inc_up = [x.upper() for x in include] if include else []
    exc_up = [x.upper() for x in exclude]
    suites = {}

    for epd_file in sorted(d.glob("*.epd")):
        name = epd_file.stem.upper()
        if inc_up and name not in inc_up:
            continue
        if name in exc_up:
            print(f"  [skip] {name}")
            continue
        positions = load_epd(str(epd_file))
        if positions:
            suites[name] = positions
            print(f"  [load] {name:<12}  {len(positions):>5} posições  ({epd_file.name})")
        else:
            print(f"  [warn] {name} — 0 posições, pulando")
    return suites


# ── AMOSTRAGEM ─────────────────────────────────────────────────────────────────

def sample_positions(positions, n, pct, seed=42):
    """
    n   → exatamente N posições; se N >= total, retorna tudo embaralhado
    pct → round(total * pct) posições, mínimo 1
    ambos None → retorna na ordem original (modo SAMPLE_ALL)
    """
    total = len(positions)
    rng   = random.Random(seed)

    if n is not None:
        if n >= total:
            out = list(positions); rng.shuffle(out); return out
        return [positions[i] for i in sorted(rng.sample(range(total), n))]

    if pct is not None:
        k = max(1, round(total * pct))
        if k >= total:
            out = list(positions); rng.shuffle(out); return out
        return [positions[i] for i in sorted(rng.sample(range(total), k))]

    return list(positions)


# ── UCI ENGINE WRAPPER ─────────────────────────────────────────────────────────

class UCIEngine:
    def __init__(self, binary, nnue_path=None):
        args = [binary]
        if nnue_path and os.path.exists(nnue_path):
            args += ['--nnue', nnue_path]

        kw = {}
        if sys.platform == 'win32':
            kw['creationflags'] = subprocess.CREATE_NO_WINDOW

        self._proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, **kw,
        )
        self._lock         = threading.Lock()
        self._stderr_lines = []

        def _drain():
            for line in iter(self._proc.stderr.readline, ''):
                self._stderr_lines.append(line.rstrip())
        threading.Thread(target=_drain, daemon=True).start()

        self._send('uci');     self._wait_for('uciok',   10)
        self._send('isready'); self._wait_for('readyok', 10)
        self.nnue_loaded = any('[NNUE] Loaded' in l for l in self._stderr_lines)

    def _send(self, cmd):
        self._proc.stdin.write(cmd + '\n')
        self._proc.stdin.flush()

    def _readline_timeout(self, timeout):
        if sys.platform != 'win32':
            import select
            if select.select([self._proc.stdout], [], [], timeout)[0]:
                return self._proc.stdout.readline().rstrip()
            return ''
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            line = self._proc.stdout.readline()
            if line: return line.rstrip()
            time.sleep(0.002)
        return ''

    def _wait_for(self, marker, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self._readline_timeout(min(0.1, deadline - time.monotonic()))
            if line and marker in line:
                return
        raise TimeoutError(f"UCI: '{marker}' não recebido em {timeout}s")

    def search(self, fen, depth, movetime_ms):
        with self._lock:
            self._send(f'position fen {fen}')
            if movetime_ms:
                self._send(f'go movetime {movetime_ms}')
            else:
                self._send(f'go depth {depth or 8}')

            best_uci = None; score = None; nodes = 0
            t0       = time.monotonic()
            deadline = t0 + ENGINE_TIMEOUT_S

            while time.monotonic() < deadline:
                line = self._readline_timeout(min(0.1, deadline - time.monotonic()))
                if not line: continue
                if line.startswith('info'):
                    m = re.search(r'score cp (-?\d+)', line)
                    if m: score = int(m.group(1))
                    m = re.search(r'score mate (-?\d+)', line)
                    if m: score = 100_000 if int(m.group(1)) > 0 else -100_000
                    m = re.search(r'nodes (\d+)', line)
                    if m: nodes = int(m.group(1))
                elif line.startswith('bestmove'):
                    parts = line.split()
                    best_uci = parts[1] if len(parts) > 1 else None
                    break

            return {'uci': best_uci, 'score': score, 'nodes': nodes,
                    'elapsed_ms': int((time.monotonic() - t0) * 1000)}

    def close(self):
        try: self._send('quit'); self._proc.wait(timeout=3)
        except Exception: pass
        try: self._proc.terminate()
        except Exception: pass


# ── SAN → UCI ──────────────────────────────────────────────────────────────────

def san_to_uci(fen, san):
    try:
        import chess
        board = chess.Board(fen)
        move  = board.parse_san(san.rstrip('+#!?'))
        return move.uci()
    except Exception:
        return None


# ── SCORING ────────────────────────────────────────────────────────────────────

def score_position(pos, engine_uci):
    if engine_uci is None:
        return 0, 10

    if pos['scored_moves']:
        for san, pts in pos['scored_moves'].items():
            if san_to_uci(pos['fen'], san) == engine_uci:
                return pts, 10
        return 0, 10

    if pos['type'] == 'am':
        am_ucis = [u for u in (san_to_uci(pos['fen'], s) for s in pos['am']) if u]
        return (0 if engine_uci in am_ucis else 10), 10

    bm_ucis = [u for u in (san_to_uci(pos['fen'], s) for s in pos['bm']) if u]
    return (10 if engine_uci in bm_ucis else 0), 10


# ── SUITE RUNNER ───────────────────────────────────────────────────────────────

def run_suite(suite_name, positions, binary, depth, movetime_ms, workers):
    label      = os.path.basename(binary)
    search_tag = f"movetime={movetime_ms}ms" if movetime_ms else f"depth={depth}"
    print(f"\n  [{suite_name}] {label}  —  {len(positions)} posições  ({search_tag})")

    results   = [None] * len(positions)
    tot_score = tot_max = correct = 0
    t_start   = time.monotonic()

    pool = []; pool_lock = threading.Lock()

    def get_eng():
        with pool_lock:
            if pool: return pool.pop()
        return UCIEngine(binary)

    def put_eng(e):
        with pool_lock: pool.append(e)

    def run_one(idx_pos):
        idx, pos = idx_pos
        eng = get_eng()
        try:
            raw = eng.search(pos['fen'], depth, movetime_ms)
        except Exception:
            raw = {'uci': None, 'score': None, 'nodes': 0, 'elapsed_ms': 0}
        finally:
            put_eng(eng)

        pts, mx = score_position(pos, raw.get('uci'))

        if pos['scored_moves']:
            best_san = max(pos['scored_moves'], key=pos['scored_moves'].get)
            exp = f"{best_san}({san_to_uci(pos['fen'], best_san) or '?'})"
        elif pos['type'] == 'am':
            exp = f"avoid:{'/'.join(pos['am'][:2])}"
        else:
            s = pos['bm'][0] if pos['bm'] else '?'
            exp = f"{s}({san_to_uci(pos['fen'], s) or '?'})"

        icon = 'OK' if pts > 0 else '--'
        print(f"    [{icon}] {pos['id'][:38]:<38}  exp:{exp:<22}"
              f"  got:{(raw.get('uci') or 'none'):<8}"
              f"  {raw.get('elapsed_ms', 0):>5}ms"
              f"  sc:{raw.get('score', '?')}")

        return idx, {
            'id': pos['id'], 'fen': pos['fen'], 'type': pos['type'],
            'expected':   '/'.join(pos['bm'] or pos['am']),
            'engine_uci': raw.get('uci'),
            'score':      raw.get('score'),
            'nodes':      raw.get('nodes', 0),
            'points':     pts, 'max_points': mx,
            'pass':       pts > 0,
            'elapsed_ms': raw.get('elapsed_ms', 0),
        }

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_one, ip) for ip in enumerate(positions)]
        for fut in as_completed(futs):
            idx, r = fut.result()
            results[idx]  = r
            correct      += int(r['pass'])
            tot_score    += r['points']
            tot_max      += r['max_points']

    for e in pool: e.close()

    elapsed = time.monotonic() - t_start
    pct_val = tot_score / tot_max * 100 if tot_max else 0
    avg_ms  = elapsed / max(len(positions), 1) * 1000
    print(f"\n  [{suite_name}] Score: {tot_score}/{tot_max} ({pct_val:.1f}%)  "
          f"Pass: {correct}/{len(positions)}  "
          f"Tempo: {elapsed:.1f}s  Avg: {avg_ms:.0f}ms/pos")

    return {
        'suite': suite_name, 'engine': label,
        'depth': depth, 'movetime_ms': movetime_ms,
        'total': len(positions), 'correct': correct,
        'score': tot_score, 'max_score': tot_max,
        'pct': round(pct_val, 2), 'elapsed_ms': int(elapsed * 1000),
        'results': results,
    }


# ── TABELA COMPARATIVA ─────────────────────────────────────────────────────────

def print_table(e1_res, e2_res, e1_label, e2_label):
    W = 110
    print("\n" + "=" * W)
    if e2_res:
        print(f"  COMPARACAO: {e1_label}  x  {e2_label}")
    else:
        print(f"  RESULTADOS: {e1_label}")
    print("=" * W)

    if e2_res:
        print(f"  {'Suite':<12}  {'Pos':>5}  "
              f"{'E1 Score':>10}  {'E1%':>6}  "
              f"{'E2 Score':>10}  {'E2%':>6}  "
              f"{'Delta':>7}  {'Melhora':>8}  {'Piora':>6}")
    else:
        print(f"  {'Suite':<12}  {'Pos':>5}  "
              f"{'Score':>10}  {'%':>6}  {'Pass':>6}  {'Avg ms':>8}")
    print("  " + "-" * (W - 2))

    e1_ts = e1_tm = e1_tp = 0
    e2_ts = e2_tm = e2_tp = 0

    for name, r1 in e1_res.items():
        e1_ts += r1['score']; e1_tm += r1['max_score']; e1_tp += r1['correct']
        r2 = e2_res.get(name) if e2_res else None

        if r2:
            e2_ts += r2['score']; e2_tm += r2['max_score']; e2_tp += r2['correct']
            imp = reg = 0
            for a, b in zip(r1['results'], r2['results']):
                if a and b:
                    imp += int(not a['pass'] and b['pass'])
                    reg += int(    a['pass'] and not b['pass'])
            delta = r2['score'] - r1['score']
            print(f"  {name:<12}  {r1['total']:>5}  "
                  f"{r1['score']:>5}/{r1['max_score']:<4}  {r1['pct']:>5.1f}%  "
                  f"{r2['score']:>5}/{r2['max_score']:<4}  {r2['pct']:>5.1f}%  "
                  f"{delta:>+7}  {imp:>8}  {reg:>6}")
        else:
            avg = r1['elapsed_ms'] / max(r1['total'], 1)
            print(f"  {name:<12}  {r1['total']:>5}  "
                  f"{r1['score']:>5}/{r1['max_score']:<4}  {r1['pct']:>5.1f}%  "
                  f"{r1['correct']:>6}  {avg:>7.0f}ms")

    print("  " + "-" * (W - 2))
    e1p = e1_ts / e1_tm * 100 if e1_tm else 0
    if e2_res and e2_tm:
        e2p   = e2_ts / e2_tm * 100
        delta = e2_ts - e1_ts
        print(f"  {'TOTAL':<12}  {'':>5}  "
              f"{e1_ts:>5}/{e1_tm:<4}  {e1p:>5.1f}%  "
              f"{e2_ts:>5}/{e2_tm:<4}  {e2p:>5.1f}%  "
              f"{delta:>+7}")
    else:
        print(f"  {'TOTAL':<12}  {'':>5}  "
              f"{e1_ts:>5}/{e1_tm:<4}  {e1p:>5.1f}%  {e1_tp:>6}")

    print("=" * W)
    print("  STS: pontuacao parcial (1-10/pos). Demais suites: acerto/erro (10/0).")
    if e2_res:
        print("  Melhora = posicoes em que E2 acertou e E1 errou")
        print("  Piora   = posicoes em que E1 acertou e E2 errou")


# ── HELPERS ────────────────────────────────────────────────────────────────────

def find_nnue(binary_path):
    p = Path(binary_path).parent / 'nnue_weights.bin'
    return str(p) if p.exists() else None


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN — resolve CONFIG vs CLI e executa
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Zchezz UCI Engine Test Suite Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Todos os parametros tem default no bloco CONFIG no topo do arquivo.\n"
            "Os argumentos abaixo sobrescrevem o CONFIG quando fornecidos.\n\n"
            "Exemplos:\n"
            "  python run_suite.py                              # usa CONFIG\n"
            "  python run_suite.py --e1 nova.exe --depth 12\n"
            "  python run_suite.py --n 30 --exclude SSM OTS\n"
            "  python run_suite.py --movetime 500 --pct 0.50\n"
            "  python run_suite.py --all --include STS WAC\n"
        )
    )

    parser.add_argument('--e1',    default=None, metavar='PATH', help='Engine 1 (override ENGINE_1)')
    parser.add_argument('--e2',    default=None, metavar='PATH', help='Engine 2 (override ENGINE_2)')
    parser.add_argument('--nnue1', default=None, metavar='PATH', help='NNUE weights engine 1')
    parser.add_argument('--nnue2', default=None, metavar='PATH', help='NNUE weights engine 2')

    sg = parser.add_mutually_exclusive_group()
    sg.add_argument('--depth',    type=int,   default=None, metavar='N',  help='Profundidade (override DEPTH)')
    sg.add_argument('--movetime', type=int,   default=None, metavar='MS', help='ms por posicao (override MOVETIME_MS)')

    parser.add_argument('--suites-dir', default=None, metavar='DIR',  help='Pasta .epd (override SUITES_DIR)')
    parser.add_argument('--include', nargs='+', default=None, metavar='SUITE', help='Rodar somente estas suites')
    parser.add_argument('--exclude', nargs='+', default=None, metavar='SUITE', help='Pular estas suites')

    am = parser.add_mutually_exclusive_group()
    am.add_argument('--n',   type=int,   default=None, metavar='N', help='N posicoes por suite (override SAMPLE_N)')
    am.add_argument('--pct', type=float, default=None, metavar='F', help='Fracao 0-1 por suite (override SAMPLE_PCT)')
    am.add_argument('--all', action='store_true',                   help='Todas as posicoes (override SAMPLE_ALL)')

    parser.add_argument('--seed',    type=int, default=None, help='Semente aleatoria (override SAMPLE_SEED)')
    parser.add_argument('--workers', type=int, default=None, help='Workers paralelos (override WORKERS)')
    parser.add_argument('--output',  default=None, metavar='FILE', help='Arquivo JSON de saida')

    args = parser.parse_args()

    # ── Resolve CONFIG vs CLI ─────────────────────────────────────────────────
    e1_path    = args.e1    or ENGINE_1
    e2_path    = args.e2    or ENGINE_2
    nnue1_path = args.nnue1 or NNUE_1
    nnue2_path = args.nnue2 or NNUE_2
    suites_dir = args.suites_dir or SUITES_DIR
    workers    = args.workers if args.workers is not None else WORKERS
    seed       = args.seed    if args.seed    is not None else SAMPLE_SEED

    # search mode: CLI > CONFIG; MOVETIME_MS > DEPTH
    if   args.movetime is not None: depth = None;         movetime = args.movetime
    elif args.depth    is not None: depth = args.depth;   movetime = None
    elif MOVETIME_MS   is not None: depth = None;         movetime = MOVETIME_MS
    else:                           depth = DEPTH;        movetime = None

    # suite lists
    include = args.include if args.include is not None else INCLUDE_SUITES
    exclude = args.exclude if args.exclude is not None else EXCLUDE_SUITES

    # sampling: CLI > CONFIG; priority all > n > pct
    if   args.all:              sample_n = None;     sample_pct = None
    elif args.n   is not None:  sample_n = args.n;   sample_pct = None
    elif args.pct is not None:  sample_n = None;     sample_pct = args.pct
    elif SAMPLE_ALL:            sample_n = None;     sample_pct = None
    elif SAMPLE_N is not None:  sample_n = SAMPLE_N; sample_pct = None
    else:                       sample_n = None;     sample_pct = SAMPLE_PCT

    # ── Validacoes ───────────────────────────────────────────────────────────
    if not e1_path:
        print("[ERRO] ENGINE_1 nao configurado. Edite o CONFIG ou use --e1.")
        sys.exit(1)
    for lbl, p in [("Engine 1", e1_path), ("Engine 2", e2_path)]:
        if p and not os.path.exists(p):
            print(f"[ERRO] {lbl} nao encontrado: {p}")
            sys.exit(1)

    nnue1_path = nnue1_path or find_nnue(e1_path)
    nnue2_path = nnue2_path or (find_nnue(e2_path) if e2_path else None)

    # ── Header ───────────────────────────────────────────────────────────────
    W = 72
    print("=" * W)
    print("  ZCHEZZ UCI ENGINE TEST SUITE RUNNER")
    print("=" * W)
    print(f"  Engine 1 : {e1_path}")
    print(f"  NNUE 1   : {nnue1_path or '(nao encontrado — eval classico)'}")
    if e2_path:
        print(f"  Engine 2 : {e2_path}")
        print(f"  NNUE 2   : {nnue2_path or '(nao encontrado — eval classico)'}")
    search_str = f"movetime={movetime}ms" if movetime else f"depth={depth}"
    print(f"  Busca    : {search_str}   Workers: {workers}   Seed: {seed}")
    if include: print(f"  Include  : {', '.join(include)}")
    if exclude: print(f"  Exclude  : {', '.join(exclude)}")
    print()

    # ── Carrega suites ───────────────────────────────────────────────────────
    print(f"Carregando suites de: {suites_dir}")
    suites = discover_suites(suites_dir, include, exclude)
    if not suites:
        print("[ERRO] Nenhuma suite carregada.")
        sys.exit(1)

    # ── Amostragem ───────────────────────────────────────────────────────────
    if sample_n is None and sample_pct is None:
        mode_str = "TODAS (ordem original)"
    elif sample_n is not None:
        mode_str = f"n={sample_n} por suite"
    else:
        mode_str = f"pct={sample_pct:.0%} por suite"

    print(f"\nAmostragem — {mode_str} (seed={seed}):")
    sampled = {}
    for name, positions in suites.items():
        s = sample_positions(positions, sample_n, sample_pct, seed=seed)
        sampled[name] = s
        print(f"  {name:<12}  {len(s):>5} / {len(positions)}")

    # ── Engine 1 ─────────────────────────────────────────────────────────────
    print(f"\n{'-'*W}\n  Engine 1: {os.path.basename(e1_path)}\n{'-'*W}")
    e1_results = {}
    for name, positions in sampled.items():
        e1_results[name] = run_suite(name, positions, e1_path, depth, movetime, workers)

    # ── Engine 2 (opcional) ──────────────────────────────────────────────────
    e2_results = None
    if e2_path:
        print(f"\n{'-'*W}\n  Engine 2: {os.path.basename(e2_path)}\n{'-'*W}")
        e2_results = {}
        for name, positions in sampled.items():
            e2_results[name] = run_suite(name, positions, e2_path, depth, movetime, workers)

    # ── Tabela ───────────────────────────────────────────────────────────────
    print_table(e1_results, e2_results,
                os.path.basename(e1_path),
                os.path.basename(e2_path) if e2_path else None)

    # ── Salva JSON ───────────────────────────────────────────────────────────
    out_path = args.output
    if not out_path:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        ts       = time.strftime('%Y%m%d_%H%M%S')
        out_path = os.path.join(RESULTS_DIR, f"suite_results_{ts}.json")

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'meta': {
                'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'engine1': e1_path, 'engine2': e2_path,
                'depth': depth, 'movetime_ms': movetime,
                'seed': seed, 'sample_n': sample_n, 'sample_pct': sample_pct,
            },
            'engine1': e1_results,
            'engine2': e2_results,
        }, f, indent=2)
    print(f"\n  Resultados salvos em: {out_path}")


if __name__ == '__main__':
    main()
