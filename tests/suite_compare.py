"""
suite_compare.py — Orquestrador de testes para Engines Zchezz (arquivos JS).
Executa as 6 suites de teste individuais, amostrando conforme as porcentagens
e consolidando os resultados em uma tabela comparativa (até 4 engines).
"""

import os, json, subprocess, re, time, random, shutil, threading
from queue import Queue

# ── CONFIGURAÇÕES DOS ENGINES ─────────────────────────────────
# Adicione o caminho e a profundidade: (caminho_js, depth)
ENGINE_V1 = (r"engine\worker\zchezz_worker_v112.js", 8)
ENGINE_V2 = (r"engine\worker_html_bin\146_v2\zchezz_worker_v146.js", 8)
ENGINE_V3 = None
ENGINE_V4 = None

# QUANTIDADE DE NÚCLEOS / TESTES EM PARALELO
CONCURRENCY = 12

# ── AMOSTRAGEM POR SUITE (0.0 a 1.0) ─────────────────────────
PCT_STS   = 0.05   # Strategic Test Suite (450 pos)
PCT_SSM   = 0.05   # Endgame Tablebase (300 pos)
PCT_WAC   = 0.05   # Win at Chess (300 pos)
PCT_ERET  = 0.05   # Engine Test Suite (300 pos)
PCT_MATE5 = 0.05   # Mates em 5 (100 pos)
PCT_OTS   = 0.05   # One-Two-Three (150 pos)

# Nome do comando Node.js (tente "node" ou o caminho completo se necessário)
# Nome do comando Node.js (caminho completo para garantir no Windows)
NODE_CMD = r"C:\Program Files\nodejs\node.exe"

# ── SUITES DEFINIDAS ──────────────────────────────────────────
SUITES = {
    'STS':   {'file': r'tests\sts_test_suite.js',   'pct': PCT_STS},
    'SSM':   {'file': r'tests\ssm_test_suite.js',   'pct': PCT_SSM},
    'WAC':   {'file': r'tests\wac_test_suite.js',   'pct': PCT_WAC},
    'ERET':  {'file': r'tests\eret_test_suite.js',  'pct': PCT_ERET},
    'MATE5': {'file': r'tests\mate5_test_suite.js', 'pct': PCT_MATE5},
    'OTS':   {'file': r'tests\ots_test_suite.js',   'pct': PCT_OTS},
}

# ── MOTOR DE EXECUÇÃO EM JS ───────────────────────────────────
# O script Python vai gerar um mini-harness temporário em JS para rodar o engine 
# individualmente e retornar apenas o JSON com os resultados de performance.
# ── PASSO 1: descobre tamanho de cada suite via Node (uma vez) ──────────────
# Depois amostra índices em Python com semente fixa → todos os engines
# recebem exactamente as mesmas posições, em exactamente a mesma quantidade.

JS_GET_LENGTH_TMPL = """
const suite = require('./{SUITE_FILE}');
const VAR_MAP = {
  STS:'STS_POSITIONS', SSM:'SSM_POSITIONS', WAC:'WAC_POSITIONS',
  ERET:'ERET_POSITIONS', MATE5:'M5_POSITIONS', OTS:'OTS_POSITIONS'
};
const arr = suite[VAR_MAP['{SUITE_KEY}']];
console.log(arr ? arr.length : 0);
"""

JS_SINGLE_RUNNER_TMPL = """
const fs   = require('fs');
const path = require('path');
const suite = require('./{SUITE_FILE}');

// ── NNUE patches ──────────────────────────────────────────────────────────────
let engineCode = fs.readFileSync('{ENGINE_FILE}', 'utf8');

// Patch 1: declare _nnueMode — strict-mode new Function() forbids implicit globals
engineCode = engineCode.replace(
    /^(\\s*['"]use strict['"]\\s*;)/m,
    '$1\\nvar _nnueMode = false;'
);
// Patch 2: replace _nnueBootstrap to load .bin from the engine's own directory
const engineDir = path.dirname(path.resolve('{ENGINE_FILE}'));
const binPath   = path.join(engineDir, 'nnue_weights.bin');
if (fs.existsSync(binPath)) {
    const nnueBinB64 = fs.readFileSync(binPath).toString('base64');
    engineCode = engineCode.replace(
        /\\(function _nnueBootstrap\\(\\)\\s*\\{[\\s\\S]*?\\}\\)\\(\\);/,
        `(function _nnueBootstrap() {
            try {
                const buf = Buffer.from('` + nnueBinB64 + `', 'base64');
                const ab  = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
                _nnueParseBinary(ab);
                _nnueMode = true;
                process.stderr.write('[NNUE] Loaded (' + ab.byteLength + ' bytes)\\\\n');
            } catch(e) {
                process.stderr.write('[NNUE] Error: ' + e.message + '\\\\n');
            }
        })();`
    );
} else {
    process.stderr.write('[suite] nnue_weights.bin not found at: ' + binPath + '\\n');
}

// ── SAN→UCI helper injected into the engine closure ───────────────────────────
function buildEngine(code) {
  const self = { postMessage: null, onmessage: null, _sanToUci: null };
  const helper = `
    const _FILES = 'abcdefgh'; const _RANKS = '87654321';
    function _parseSanPromotion(san){ const m=san.match(/=?([QRBN])$/); return m?m[1].toLowerCase():null; }
    function _pieceTypeFromChar(ch){ return {'P':1,'N':2,'B':3,'R':4,'Q':5,'K':6}[ch]||0; }
    self._sanToUci = function(fen, san) {
      try {
        const st = loadFen(fen); const moves = st.legalMoves();
        const s = san.replace(/[+#!?]/g,'').trim();
        if(s==='O-O'  ||s==='0-0')  { for(const m of moves){ const u=moveToUci(m); if(u==='e1g1'||u==='e8g8') return u; } return null; }
        if(s==='O-O-O'||s==='0-0-0'){ for(const m of moves){ const u=moveToUci(m); if(u==='e1c1'||u==='e8c8') return u; } return null; }
        const promChar  = _parseSanPromotion(s);
        const promPiece = promChar ? {q:5,r:4,b:3,n:2}[promChar] : null;
        const sBase     = s.replace(/[=QRBN]$/,'');
        let pieceType=1, rest=sBase;
        if(sBase.length>0 && 'NBRQK'.includes(sBase[0])){ pieceType=_pieceTypeFromChar(sBase[0]); rest=sBase.slice(1); }
        rest = rest.replace('x','');
        const dest=rest.slice(-2);
        const destFile=_FILES.indexOf(dest[0]); const destRank=_RANKS.indexOf(dest[1]);
        if(destFile<0||destRank<0) return null;
        const destSq=destRank*8+destFile;
        const disambig=rest.slice(0,rest.length-2);
        for(const m of moves){
          if(promPiece&&m.prom!==promPiece) continue;
          if(!promPiece&&m.prom) continue;
          if(m.to!==destSq) continue;
          if((st.b[m.from]&7)!==pieceType) continue;
          if(disambig.length>0){
            const fF=_FILES[m.from&7]; const fR=_RANKS[m.from>>3];
            if(disambig.length===1){ if(disambig!==fF&&disambig!==fR) continue; }
            else if(disambig.length===2){ if(disambig!==(fF+fR)) continue; }
          }
          return moveToUci(m);
        }
        return null;
      } catch(e){ return null; }
    };
  `;
  return (new Function('self', code + '\\n' + helper + '\\nreturn self;'))(self);
}

// ── Suite-aware scoring ───────────────────────────────────────────────────────
//
//  Suite  | bm field type  | type field      | Scoring
//  -------|----------------|-----------------|----------------------------------
//  STS    | string (1 SAN) | absent          | scored_moves[engineSAN] → 0-10
//  SSM    | string (1 SAN) | absent          | pass/fail: 10 or 0
//  WAC    | array of SAN   | always "bm"     | pass/fail: 10 or 0
//  ERET   | array of SAN   | "bm" or "am"    | bm=pass/fail; am=10 if NOT played
//  MATE5  | array of SAN   | always "bm"     | pass/fail: 10 or 0
//  OTS    | array of SAN   | absent          | pass/fail: 10 or 0

const SUITE_KEY  = '{SUITE_KEY}';
const VAR_MAP    = { STS:'STS_POSITIONS', SSM:'SSM_POSITIONS', WAC:'WAC_POSITIONS',
                     ERET:'ERET_POSITIONS', MATE5:'M5_POSITIONS', OTS:'OTS_POSITIONS' };
const allPos     = suite[VAR_MAP[SUITE_KEY]];
const positions  = {SAMPLED_INDICES}.map(i => allPos[i]);

const eng        = buildEngine(engineCode);
let totalScore=0, correct=0;
const start = Date.now();

for (const p of positions) {
  let result = null;
  eng.postMessage = (d) => { result = d; };
  eng.onmessage({ data: { fen: p.fen, depth: {DEPTH}, id: 1 } });
  const engineUci = result && result.uci ? result.uci : null;
  let posScore = 0;

  if (SUITE_KEY === 'STS') {
    // bm is a single SAN string; scored_moves = {SAN: points, ...}
    // Convert each scored_moves key to UCI and compare.
    if (engineUci && p.scored_moves) {
      for (const [san, pts] of Object.entries(p.scored_moves)) {
        if (eng._sanToUci(p.fen, san) === engineUci) { posScore = pts; break; }
      }
    }

  } else if (SUITE_KEY === 'SSM') {
    // bm is a single SAN string
    if (engineUci && eng._sanToUci(p.fen, p.bm) === engineUci) posScore = 10;

  } else if (SUITE_KEY === 'ERET') {
    // bm is an array; type can be "bm" or "am"
    const bmUcis = (p.bm || []).map(s => eng._sanToUci(p.fen, s)).filter(Boolean);
    if (p.type === 'am') {
      // Avoid move: score if engine does NOT play the listed move(s)
      if (engineUci && bmUcis.length > 0 && !bmUcis.includes(engineUci)) posScore = 10;
    } else {
      if (engineUci && bmUcis.includes(engineUci)) posScore = 10;
    }

  } else {
    // WAC, MATE5, OTS — bm is an array of SAN strings
    const bmUcis = (p.bm || []).map(s => eng._sanToUci(p.fen, s)).filter(Boolean);
    if (engineUci && bmUcis.includes(engineUci)) posScore = 10;
  }

  totalScore += posScore;
  if (posScore > 0) correct++;
}

console.log(JSON.stringify({
  correct:  correct,
  total:    positions.length,
  score:    totalScore,
  maxScore: positions.length * 10,
  timeMs:   Date.now() - start
}));
"""

# Cache: suite_key → lista de índices amostrados (computada uma única vez)
_sampled_indices_cache: dict = {}

def get_sampled_indices(suite_key: str) -> list:
    """
    Descobre o total de posições da suite rodando um Node mínimo,
    depois sorteia índices com semente fixa (42).
    Resultado é cacheado: todos os engines recebem os MESMOS índices.
    """
    if suite_key in _sampled_indices_cache:
        return _sampled_indices_cache[suite_key]

    s = SUITES[suite_key]
    if not os.path.exists(s['file']):
        _sampled_indices_cache[suite_key] = []
        return []

    js = JS_GET_LENGTH_TMPL \
        .replace('{SUITE_FILE}', s['file'].replace('\\', '/')) \
        .replace('{SUITE_KEY}',  suite_key)
    tmp = f"tmp_len_{suite_key}.js"
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(js)
    try:
        res = subprocess.run([NODE_CMD, tmp], capture_output=True, text=True, timeout=15)
        total = int(res.stdout.strip()) if res.returncode == 0 and res.stdout.strip().isdigit() else 0
    except Exception:
        total = 0
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    if total == 0:
        print(f"  [Suite {suite_key}] AVISO: não foi possível determinar o total de posições.")
        _sampled_indices_cache[suite_key] = []
        return []

    rng = random.Random(42)  # semente fixa → mesmos índices em toda execução
    n_sample = max(1, round(total * s['pct']))
    indices = sorted(rng.sample(range(total), min(n_sample, total)))
    print(f"  [Suite {suite_key}] {total} posições → {len(indices)} amostradas (seed=42, pct={s['pct']:.0%})")
    _sampled_indices_cache[suite_key] = indices
    return indices


def run_single_test(engine_file, suite_key, depth):
    s = SUITES[suite_key]
    if not os.path.exists(s['file']) or not os.path.exists(engine_file):
        return None

    indices = get_sampled_indices(suite_key)
    if not indices:
        return None

    js_content = JS_SINGLE_RUNNER_TMPL \
        .replace('{SUITE_FILE}',      s['file'].replace('\\', '/')) \
        .replace('{ENGINE_FILE}',     engine_file.replace('\\', '/')) \
        .replace('{SUITE_KEY}',       suite_key) \
        .replace('{SAMPLED_INDICES}', json.dumps(indices)) \
        .replace('{DEPTH}',           str(depth))

    # pid no nome evita colisão quando múltiplas threads rodam a mesma suite
    tmp_file = f"tmp_run_{suite_key}_{os.getpid()}_{threading.get_ident()}.js"
    with open(tmp_file, 'w', encoding='utf-8') as f:
        f.write(js_content)

    try:
        res = subprocess.run([NODE_CMD, tmp_file], capture_output=True, text=True, timeout=120)
        # Show NNUE/error messages from stderr (only first run per engine to avoid spam)
        if res.stderr.strip():
            for line in res.stderr.strip().splitlines():
                print(f"    [Node] {line}")
        if res.returncode == 0:
            return json.loads(res.stdout.strip())
        else:
            print(f"  [Erro Node na suite {suite_key}]: {res.stderr[:300]}")
            return None
    except Exception as e:
        print(f"  [Exceção na suite {suite_key}]: {e}")
        return None
    finally:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

# ── MAIN ──────────────────────────────────────────────────────
def main():
    print("="*80)
    print("      ZCHEZZ ENGINE SUITE COMPARATOR (JS RUNNER)      ")
    print("="*80)
    
    # 1. Checar Node (shutil.which não funciona para caminhos completos no Windows;
    #    verificamos diretamente com subprocess)
    try:
        r = subprocess.run([NODE_CMD, "--version"], capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            raise RuntimeError(r.stderr)
        print(f"Node.js: {r.stdout.strip()}")
    except Exception as e:
        print(f"❌ Erro: Node não encontrado em '{NODE_CMD}'. Ajuste NODE_CMD.\n  {e}")
        return

    # 2. Configurar engines
    engines_paths = [ENGINE_V1, ENGINE_V2, ENGINE_V3, ENGINE_V4]
    targets = []
    for config in engines_paths:
        if config and isinstance(config, (list, tuple)):
            path, depth = config
            if os.path.exists(path):
                targets.append((path, depth))
            else:
                print(f"  AVISO: engine não encontrado: {path}")
        elif config and isinstance(config, str) and os.path.exists(config):
            targets.append((config, 3))

    if not targets:
        print("Nenhum engine configurado para teste (ENGINE_V1/2/3/4).")
        return

    # 3. PRÉ-AMOSTRAGEM — roda UMA vez por suite, antes de qualquer thread
    #    Garante que todos os engines recebam exatamente os mesmos índices.
    print(f"\nPré-amostrado das suites (seed=42)...")
    for suite_key in SUITES:
        get_sampled_indices(suite_key)  # preenche o cache

    table_data = []
    results_queue = Queue()

    def test_worker(engine_info):
        engine_path, engine_depth = engine_info
        label = os.path.basename(engine_path)
        engine_res = {'label': label, 'depth': engine_depth, 'suites': {},
                      'total_correct': 0, 'total_pos': 0, 'total_time': 0}
        for suite_key in SUITES.keys():
            res = run_single_test(engine_path, suite_key, engine_depth)
            if res:
                engine_res['suites'][suite_key] = res
                engine_res['total_correct'] += res['correct']
                engine_res['total_pos']     += res['total']
                engine_res['total_time']    += res['timeMs']
        results_queue.put(engine_res)

    # 4. Executar testes em paralelo (um thread por engine)
    print(f"\nIniciando testes ({len(targets)} engines, {CONCURRENCY} núcleos)...")
    threads = []
    for info in targets:
        t = threading.Thread(target=test_worker, args=(info,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    while not results_queue.empty():
        table_data.append(results_queue.get())

    # 5. Renderizar tabela
    print("\n" + "="*105)
    header = f"{'Engine':<28} | {'Score%':<7} | {'Score':<12} | {'Avg ms':<8} | Stats por Suite (score/max)"
    print(header)
    print("-" * 105)
    for r in table_data:
        total_score   = sum(v['score']    for v in r['suites'].values())
        total_max     = sum(v['maxScore'] for v in r['suites'].values())
        total_time    = sum(v['timeMs']   for v in r['suites'].values())
        total_pos     = sum(v['total']    for v in r['suites'].values())
        pct     = (total_score / total_max * 100) if total_max > 0 else 0
        avg_t   = (total_time  / total_pos)       if total_pos > 0 else 0
        # Per-suite: show score/maxScore so STS partial points are visible
        stats_str = " ".join([f"{k}:{v['score']}/{v['maxScore']}" for k, v in r['suites'].items()])
        label = f"{r['label']} (D{r['depth']})"
        line  = f"{label:<28} | {pct:>5.1f}%  | {total_score:>5}/{total_max:<5} | {avg_t:>6.1f}ms | {stats_str}"
        print(line)
    print("="*105)
    print("Nota: STS usa pontuação parcial (1-10 por posição). Demais suites: 10=acerto, 0=erro.")
    print("      Mesmas posições para todos os engines (amostragem determinística, seed=42).")

if __name__ == "__main__":
    main()