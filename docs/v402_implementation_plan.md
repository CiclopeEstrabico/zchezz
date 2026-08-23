# Zchezz v4.00/v4.01 — Implementation Plan (Status & Roadmap)

_Last updated: 2026-08-23_

> **Status da branch `v403-lc0-training` (2026-08-23):** 10 receitas de treino
> com dados LCZ test91 (fresh e warm-start da v402) nao superaram a v402 —
> teto = paridade (-3 +/-29, soup de dois warm-starts). Diagnostico: calibracao
> dos rotulos LCZ vs margins desta busca; paridade Python<->C confirmada sem
> mismatches. Proximo passo em andamento: loop de self-play proprio
> (`tools/selfplay.c`, ~6-7 jogos/s a 20k nodes/lance); iteracao 1 pausada com
> ~1.64M posicoes. Bug avistado uma vez no log: `[BUG] undo overflow`.

---

## OBJETIVO

Elevar o Zchezz de ~2900 ELO (v3.14) para >3000 ELO via:

- **HalfKP-4Bucket**: dependencia de rei nas features -> maior expressividade
- **L1 = 512** (era 256), **SCReLU** (era ClippedReLU), **concat [stm|opp]**
- **Eliminacao dos 31 extras** -> 100% incremental, sem merge no hot path
- **Bootstrap loop** (selfplay -> treino -> SPRT -> promove) para treinar a rede iterativamente

---

## ARQUITETURA NOVA (HalfKP-4Bucket)

```
Feature input: 2560 (4 buckets x 640 features/perspectiva, sem rei)
L1: 2560 -> 512  int16, acumulador incremental, ativacao SCReLU
Concat: [stm 512 | opp 512] -> 1024 uint8
L2: 1024 -> 32  int8, maddubs AVX2/WASM
L3: 32 -> 1   int8 -> float32
Pesos: NNU4 (~2.6 MB), quantizacao QA=255 QB=64
```

**Invariantes criticos:**

- Concat sempre `[stm, opp]` — trocar invalida toda a rede
- Mudanca de bucket do rei invalida a perspectiva inteira (flag dirty por frame do stack)
- TT probe ANTES de TB probe
- Selfplay compartilha TT (tt_clear() por jogo); Arena isola TT por jogador
- Somente a thread principal incrementa TT_GEN

---

## O QUE FOI FEITO (HISTORICO RESUMIDO)

### Infraestrutura (CONCLUIDO)

- **TTable/NnueNet por instancia** (`create`/`destroy`), sem arrays globais
- **`tools/arena.c`** + `run_arena.py`: dois tipos de player (`net:`/`uci:`), TT isolada por lado, SPRT real
- **`tools/selfplay.c`** + `run_selfplay_native.py`: selfplay in-process, `.bin` direto, TT compartilhada com `tt_clear()`, abertura por book, PGN opcional
- **`tools/opening_pool.c`**: pool de aberturas com lock, EPD e PGN suportados
- **Makefile unificado** (`engine/build/`) com `ENGINE=vXXX`, alvos `native`/`arena`/`selfplay`/`wasm`
- **`utils/cliconf.py`**: bloco de configuracao no topo de toda ferramenta, `--show-config` universal

### Arquitetura NNUE (CONCLUIDO)

- `nnue.h`/`nnue.c` implementam os 2560 features, L1/L2/L3, SCReLU, lazy bucket refresh
- Acumulador incremental com stack de 128 frames, flags de dirty por perspectiva
- Loader NNU4 com validacao de dims e magic
- Paridade C/Python verificada via `train/check_parity.py`

### Pipeline de treino (CONCLUIDO)

- `train/train_nnue.py`: dois estagios (warmup + finetune), CosineAnnealingLR, BCE com alvo continuo (piso ~0.62, nao 0.693)
- `train/export_nnu4.py`: checkpoint `.pt` -> `nnue_weights.bin` (NNU4)
- `train/dataset.py`: blend `target = lambda*result + (1-lambda)*wdl`, lambda por dataset
- `train/labeling/process_positions.py`: conversor universal EPD/PGN/parquet/bin
- Datasets em `data/` com convencao de nomes codificando fonte/tipo/data

### v4.01 — Primeira rede treinada (CONCLUIDO 2026-08-16)

- Treino completo: warmup 12 epocas + finetune 200 epocas
- Checkpoint final: `checkpoints/v400/nnue_v400_halfkp4b_v400_ft_epoch200_2026-08-16_21-45-51.pt`
  - `avg_loss = 0.6094`, `val_loss = 0.6255`, `lr = 1e-6`
- Exportado para `engine/c/zchezz_v401/nnue_weights.bin` (2,656,464 bytes, OK)
- Compilado: `engine/c/zchezz_v401/zchezz.exe` + `arena.exe` + `selfplay.exe` com `ENGINE=v401`

### Phase 1 checks v4.01 (CONCLUIDO 2026-08-16)

- Perft: 37/37
- UCI extended: 119/120 (falha T3.2c KRKP draw score posicao limitrofe de TB — nao bloqueia)
- bench_nps: v401-noTB = 1,440,320 NPS avg (+21% vs v314), eval sanity 8/8, TB sanity 2/2
- NPS opening/middle: v401 = 1.70M vs v314 = 2.17M (-21%) — rede maior e mais cara, compensado pela eval mais precisa

---

## RESULTADO DE TORNEIO — BASELINE GEN 1 (2026-08-16)

```
================================================================
                  v401-1T        v314-1T
--------------------------------------------
       v401-1T      ---         117.5/600     117.5/600
       v314-1T   482.5/600         ---        482.5/600
```

**v4.01 Gen 1 perde ~-480 ELO vs v3.14.** Esperado e normal — e a primeira rede treinada para a nova arquitetura. v3.14 tem rede madura com muitas geracoes de bootstrap. O gap fecha iteracao a iteracao.

---

## RESULTADO DA CAMPANHA DE BUSCA — V4.02 (CONCLUIDO 2026-08-22)

Branch `v402-search-strength` (mesclado em `v400-data-and-harness`).
Sem tocar na rede: apenas politica de busca, memoria e constantes.
Cada mudanca passou por arena cega de 800 jogos a 100 ms/lance antes
de ser mantida; tudo que regrediu foi revertido (lista no README,
secao Search).

| confronto                    | Elo                |
|------------------------------|--------------------|
| v402 vs v401                 | **+157 +/- 20**    |
| v402 vs v314                 | **-146 +/- 22**    |
| v401 vs v314 (linha de base) | -268 +/- 30        |

Ganhos principais: geracao da TT estavel dentro da partida + corte de
raiz proibido + nada gravado apos aborto (+109 so isso), ajuste fino
das 13 constantes por algoritmo genetico (+23), poda SEE de silenciosos
(-24% nos nos), entradas de TT compactas (AoS 24 B), nucleo AVX-VNNI na
camada L2 (+7% NPS), futilidade antes do make, stores da qsearch.
Rejeitados com SPRT negativo: historico de capturas, 3o slot de CMH,
remocao do envelhecimento de historico, stores de stand-pat,
persistencia de counter-moves, xeque direto exato por peca.

Ferramentas: pipeline de treino (selfplay/arena/suite/torneio)
reapontado para a v402; `ga_tune.exe`, `arena.exe`, `selfplay.exe`
recompilados contra ela. Smoke test de auto-jogo OK (4 jogos, 339
posicoes).

Proximo passo do roadmap (abaixo) continua valido, agora sobre a base
de busca v402: selfplay Gen-1 -> treino Gen-2 -> gate SPRT. O folder
nao-rastreado `engine/c/zchezz_v403/` e o branch `v403-lc0-training`
abrigam o esforco de dados LC0 para esse ciclo.

---

## ROADMAP — A PARTIR DE AGORA

### O loop de bootstrap

```
1. SELFPLAY   -> gerar dados com a rede atual (Gen N)
2. TREINO     -> treinar Gen N+1 nos dados novos
3. SPRT GATE  -> arena Gen N+1 vs Gen N
   promovido  -> Gen N+1 vira current, volta para 1
   descartado -> ajustar LR/lambda/dados, volta para 2
```

---

### Passo 1 — Selfplay Gen 1 (PROXIMO PASSO IMEDIATO)

Gerar dados de selfplay com a rede v4.01 Gen 1 para alimentar o treino da Gen 2.

```powershell
# Selfplay nativo (rapido, in-process)
python tests/run_selfplay_native.py --games 50000 --movetime 30 --threads 14
```

**Configuracoes importantes:**

- `MOVETIME_MS = 30` ms (volume, nao profundidade)
- `GAMES = 50000-200000` por iteracao
- Saida em `.bin` (treino) + `.pgn` (inspecao)
- Seeds diferentes por iteracao para diversidade de aberturas

**Target:** ~50K-200K jogos por geracao, guardados em `data/selfplay_gen1/`

---

### Passo 2 — Treino Gen 2

```python
# CONFIGURATION BLOCK (train_nnue.py)
STAGE    = "finetune"   # partir dos pesos de v401 Gen 1
LR       = 1e-4         # um pouco maior que o finetune final
EPOCHS   = 50

DATASETS = [
    # Dados de selfplay Gen 1 (lambda alto: confiar no resultado do jogo)
    {"path": "data/selfplay_gen1", "type": "bin", "pct": 1.0, "lam": 0.75},
    # Mistura com dados historicos para nao esquecer taticas
    {"path": "data/sf50k_cp_wdl_zchezz_2026", "type": "parquet", "pct": 0.3, "lam": 0.2},
]
```

**Checkpoint de partida:** `--ckpt checkpoints/v400/nnue_v400_halfkp4b_v400_ft_epoch200_...pt`

---

### Passo 3 — SPRT Gate (arena Gen 2 vs Gen 1)

```powershell
# Exportar Gen 2
python train/export_nnu4.py --ckpt checkpoints/v400/<gen2_epoch_best>.pt `
    --dst checkpoints/v400/gen2.nnu4

# SPRT gate
python tests/run_arena.py `
    --player net:checkpoints/v400/gen2.nnu4 `
    --player net:engine/c/zchezz_v402/nnue_weights.bin `
    --games 400 --movetime 30 --threads 14 `
    --sprt --elo0 0 --elo1 5 `
    --json checkpoints/v400/gen2_gate_result.json
```

**Se promovido:**

- Copiar `gen2.nnu4` -> `engine/c/zchezz_v402/nnue_weights.bin`
- Recompilar: `mingw32-make ENGINE=v401 native`
- Phase 1 checks: `test_perft.py v401`, `bench_nps.py`
- Repetir o ciclo com mais dados

---

### Evolucao esperada por geracao

| Geracao       | ELO gap vs v314 |
| ------------- | --------------- |
| Gen 1 (atual) | -480 ELO        |
| Gen 2-3       | -300 a -400 ELO |
| Gen 4-6       | -100 a -200 ELO |
| Gen 7-10      | 0 a +100 ELO    |
| Gen 15+       | +150 a +250 ELO |

**Alavancas de melhora:**

- Aumentar `MOVETIME_MS` do selfplay (30->50->100ms) conforme a rede melhora
- Aumentar volume de dados por geracao
- Reduzir LR a cada ciclo: 1e-4 -> 5e-5 -> 2e-5 -> 1e-5
- Aumentar lambda (confiar mais no resultado real conforme a rede amadurece)

---

### Checks obrigatórios a cada geracao promovida (AGENTS.md Phase 1)

```powershell
python tests/test_perft.py v401
python tests/test_uci_extended.py v401
python tests/bench_nps.py
```

Fase 5 completa (regressao 700 jogos vs v314) so quando a rede estiver perto de empatar.

---

## ARQUITETURA NNUE — REFERENCIA RAPIDA

```
NNU4 layout (nnue.c _load_weights / export_nnu4.py):
  "NNU4" magic + epoch + dims[5] + scales[4]
  L1W [2560][512] int16  (2.5 MB, feature-major)
  L1B [512]       int32
  L2W [32][1024]  int8   (output-major, NO transpose)
  L2B [32]        int32
  L3W [32]        int8
  L3B              float32

Quantizacao: QA=255 (L1), QB=64 (L2/L3), SHIFT=8, OUT_SCALE=320/4096=0.078125
CP_TO_WDL_T = 320 — mesmo em nnue.c, export_nnu4.py e train_nnue.py
```

**NNUE concat order:** SEMPRE `[stm, opp]`. Trocar = engine avalia posicoes ganhas como perdidas.

**Lazy bucket refresh:** mudanca de bucket invalida a perspectiva inteira. Flag dirty no frame do stack do acumulador (nao num campo flat), para sobreviver pushes/pops corretos.

---
