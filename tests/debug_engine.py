"""
debug_engine.py — Diagnóstico completo do engine UCI nativo (zchezz.exe / zchezz).

Rode ANTES de qualquer torneio para verificar que tudo está funcionando:
    python debug_engine.py

Testa cada camada separadamente e mostra stdout + stderr raw de cada etapa.
"""

import subprocess, time, os, sys, threading, re

# ── CONFIGURE AQUI ────────────────────────────────────────────
ENGINE_PATH = r"zchezz.exe"          # ou caminho completo: r"C:\engines\zchezz.exe"
NNUE_PATH   = None                   # None = auto-descobre ao lado do binary
DEPTH       = 6
TEST_FEN    = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKB1R b KQkq e3 0 1"
# ──────────────────────────────────────────────────────────────

SEP = "─" * 62


def section(title: str):
    print(f"\n{SEP}\n  {title}\n{SEP}")


# ══════════════════════════════════════════════════════════════
# 1. Binary existe?
# ══════════════════════════════════════════════════════════════
section("1. Binary do engine existe?")
engine_abs = os.path.abspath(ENGINE_PATH)
engine_dir = os.path.dirname(engine_abs)
print(f"  Caminho absoluto : {engine_abs}")
print(f"  Existe           : {os.path.exists(engine_abs)}")
if not os.path.exists(engine_abs):
    print("  ❌ Binary não encontrado. Ajuste ENGINE_PATH.")
    sys.exit(1)
print(f"  Tamanho          : {os.path.getsize(engine_abs):,} bytes")

# ══════════════════════════════════════════════════════════════
# 2. nnue_weights.bin
# ══════════════════════════════════════════════════════════════
section("2. NNUE weights (.bin)")
if NNUE_PATH is None:
    NNUE_PATH = os.path.join(engine_dir, "nnue_weights.bin")
print(f"  Procurando em    : {NNUE_PATH}")
nnue_exists = os.path.exists(NNUE_PATH)
print(f"  Existe           : {nnue_exists}")
if nnue_exists:
    print(f"  Tamanho          : {os.path.getsize(NNUE_PATH):,} bytes")
    nnue_arg = ['--nnue', NNUE_PATH]
else:
    print("  ⚠  .bin não encontrado — engine usará eval clássico (mais fraco)")
    nnue_arg = []
    print(f"\n  Conteúdo de {engine_dir}:")
    try:
        for fname in sorted(os.listdir(engine_dir)):
            fp = os.path.join(engine_dir, fname)
            print(f"    {fname}  ({os.path.getsize(fp):,} bytes)")
    except Exception as e:
        print(f"    (erro ao listar: {e})")

# ══════════════════════════════════════════════════════════════
# 3. Handshake UCI
# ══════════════════════════════════════════════════════════════
section("3. Handshake UCI (uci / isready)")

args = [engine_abs] + nnue_arg
print(f"  Comando: {' '.join(args)}")

try:
    proc = subprocess.Popen(
        args,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0,
    )
except Exception as e:
    print(f"  ❌ Falha ao iniciar o processo: {e}")
    sys.exit(1)

stderr_lines: list[str] = []

def _drain_stderr():
    for line in iter(proc.stderr.readline, ''):
        line = line.rstrip()
        if line:
            stderr_lines.append(line)
            print(f"  [stderr] {line}")

threading.Thread(target=_drain_stderr, daemon=True).start()

def send(cmd: str):
    proc.stdin.write(cmd + '\n')
    proc.stdin.flush()

def read_until(marker: str, timeout: float = 10.0) -> list[str]:
    lines = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if sys.platform != 'win32':
            import select
            ready = select.select([proc.stdout], [], [], 0.1)
            if not ready[0]:
                continue
            line = proc.stdout.readline().rstrip()
        else:
            line = ''
            t0 = time.monotonic()
            while time.monotonic() - t0 < 0.1:
                line = proc.stdout.readline()
                if line:
                    line = line.rstrip()
                    break
                time.sleep(0.005)
        if line:
            print(f"  [stdout] {line}")
            lines.append(line)
            if marker in line:
                return lines
    print(f"  ⚠  Timeout aguardando '{marker}'")
    return lines

send('uci')
uci_lines = read_until('uciok', timeout=8)
uci_ok    = any('uciok' in l for l in uci_lines)
print(f"\n  uciok recebido   : {'✓' if uci_ok else '✗'}")

send('isready')
ready_lines = read_until('readyok', timeout=10)
ready_ok    = any('readyok' in l for l in ready_lines)
print(f"  readyok recebido : {'✓' if ready_ok else '✗'}")

if not ready_ok:
    print("\n  ❌ Engine não respondeu ao isready. Abortando.")
    proc.terminate()
    sys.exit(1)

# ══════════════════════════════════════════════════════════════
# 4. Busca real
# ══════════════════════════════════════════════════════════════
section(f"4. Busca real — depth {DEPTH}")
print(f"  FEN: {TEST_FEN}")

send(f'position fen {TEST_FEN}')
send(f'go depth {DEPTH}')

best_uci  = None
score_cp  = None
nodes     = 0
t0        = time.monotonic()
deadline  = t0 + 60.0

while time.monotonic() < deadline:
    if sys.platform != 'win32':
        import select
        ready = select.select([proc.stdout], [], [], 0.1)
        if not ready[0]:
            continue
        line = proc.stdout.readline().rstrip()
    else:
        line = ''
        t_inner = time.monotonic()
        while time.monotonic() - t_inner < 0.1:
            line = proc.stdout.readline()
            if line:
                line = line.rstrip()
                break
            time.sleep(0.005)

    if not line:
        continue

    print(f"  [stdout] {line}")

    if line.startswith('info'):
        m = re.search(r'score cp (-?\d+)', line)
        if m:
            score_cp = int(m.group(1))
        m = re.search(r'score mate (-?\d+)', line)
        if m:
            score_cp = 100000 if int(m.group(1)) > 0 else -100000
        m = re.search(r'nodes (\d+)', line)
        if m:
            nodes = int(m.group(1))

    elif line.startswith('bestmove'):
        parts    = line.split()
        best_uci = parts[1] if len(parts) > 1 else None
        break

elapsed = time.monotonic() - t0

send('quit')
try:
    proc.wait(timeout=3)
except subprocess.TimeoutExpired:
    proc.terminate()

# ══════════════════════════════════════════════════════════════
# RESUMO
# ══════════════════════════════════════════════════════════════
section("RESUMO")

nnue_loaded  = any('[NNUE] Loaded' in l for l in stderr_lines)
nnue_missing = any('not found' in l.lower() for l in stderr_lines)

print(f"  Binary       : {'✓' if os.path.exists(engine_abs) else '✗'}  {engine_abs}")
print(f"  UCI handshake: {'✓' if uci_ok and ready_ok else '✗'}")

if nnue_loaded:
    print(f"  NNUE         : ✓ carregado  ({NNUE_PATH})")
elif nnue_missing:
    print(f"  NNUE         : ✗ .bin não encontrado — eval clássico em uso")
else:
    print(f"  NNUE         : ?  (sem mensagem de NNUE no stderr)")

if best_uci:
    print(f"  Lance        : ✓  {best_uci}  score={score_cp} cp  nodes={nodes:,}  {elapsed*1000:.0f}ms")
else:
    print(f"  Lance        : ✗  nenhum bestmove recebido (timeout {elapsed:.1f}s)")
    print("\n  Possíveis causas:")
    print("    • Engine crashou (veja [stderr] acima)")
    print("    • Depth muito alto — tente reduzir DEPTH para 4")
    print("    • Posição inválida em TEST_FEN")
