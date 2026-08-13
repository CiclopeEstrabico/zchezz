# Zchezz v400 — Implementation Plan
## Agente: Claude Opus | Objetivo: Arquitetura HalfKP-4Bucket com acumulador total

---

## OBJETIVO

Elevar o Zchezz de ~2700 ELO para >3000 ELO eliminando o principal limitador arquitetural:
**a ausência de dependência de rei nas features**. Simultaneamente, remover todos os
features não-acumuláveis (os 31 extras), atingindo **100% de incrementalidade** no
acumulador — zero recomputação por posição fora do rebuild de rei.

Ganhos esperados:
- **+150 a +250 ELO** via HalfKP-4bucket (dependência de rei + L1 maior + SCReLU)
- **+10–20% nodes/s** via eliminação do merge `ext_buf` no hot path
- **Maior depth efetivo** via eval mais precisa → melhores cortes alpha-beta

---

## VISÃO GERAL EM FASES

O plano inteiro (arquitetura nova + infra de treino/tuning nova, Apêndice
F) está organizado em 6 fases sequenciais. Cada `PARTE`/`APÊNDICE` do
corpo do documento carrega a tag `(FASE N)` no próprio título, então esta
seção é só o mapa — os detalhes de cada item continuam onde já estavam.

| Fase | Nome | Objetivo | Depende de | Onde no documento |
|---|---|---|---|---|
| **0** | TT por instância | Tirar a TT de arrays globais do processo, empacotar num `TTable` alocado — sem isso nada de F.2/F.3/F.4 é possível (não dá pra ter 2 buscas independentes no mesmo processo) | nada (pode começar em paralelo com a Fase 1, é código de `search.c` isolado) | Apêndice F.1, base: Apêndice B |
| **1** | Arquitetura NNUE v400 | HalfKP-4Bucket, só cabeça de valor — o "v400" original deste documento | nada | Partes 1-8, Apêndices A, B, C, D, E |
| **2** | Arena nativa + SPSA | `tools/arena_native.c` (A/B em processo) + `tools/tune_spsa.c` (afinar constantes de busca automaticamente) | Fase 0 | Apêndice F.4 |
| **3** | Selfplay nativo + bootstrap Estágio 1 | `tools/selfplay.c`: gera `.bin` direto (sem PGN/parquet), rótulo `wl_target` com blend `K` (F.3.0), amostra de lance por temperatura sobre scores de Multi-PV | Fases 0, 1 (usa a NNUE v400 pra avaliar, mas também funciona com v3.14 pra validar o loop mais cedo) | Apêndice F.2, F.3 (intro + F.3.0 + Estágio 1) |
| **4** | Cabeça de política + MCTS (bootstrap Estágio 2) | Segunda cabeça na NNUE, MCTS PUCT como gerador de dados, temperatura sobre visitas da raiz (`π(a) ∝ N(a)^(1/T)`) | Fase 3 (loop de bootstrap já validado) | Apêndice F.3 Estágio 2 |
| **5** | Ciclo de bootstrap contínuo | Loop `gerar (Fase 3/4) → treinar → Arena (Fase 2) julga se a geração nova é mais forte via SPRT → promove ou descarta → repete` | Fases 2, 3 (e opcionalmente 4) | Ver F.5 (fim do Apêndice F) |

**Ordem prática recomendada**: Fase 0 e Fase 1 podem correr em paralelo
(são código desacoplado — TT é `search.c`/`search.h`, NNUE é
`nnue.c`/`nnue.h`/`train/`). Fase 2 destrava o primeiro uso útil da Fase 0
sozinha (dá pra rodar SPSA na v3.14 antes mesmo da v400 estar pronta, se
quiser afinar constantes de busca que não mudam com a arquitetura de
rede). Fases 3→4→5 são estritamente sequenciais.

---

## STATUS ATUAL (atualizado 2026-08-12)

Evidência: **[M]** medido/testado nesta sessão · **[O]** relatado pelo dono
do projeto · **[?]** não verificado — não assumir pronto nem faltando.

| Fase | Item | Status | Evidência |
|---|---|---|---|
| **0** | TT por instância (`TTable`/`NnueNet`) | **PRONTO** | `search.h`/`nnue.h` já expõem `create`/`destroy`; `g_tt`/`g_nnue_net` são a instância default do processo. Documentado em CLAUDE.md ("Per-instance objects"). Neutralidade provada por paridade exata de nodes vs binário baseline pré-refactor com eval=0 (busca determinística): startpos `go depth 10` → 3520 nós, `a2a3` antes e depois; meio-jogo `go depth 10` → 5339 nós, `c3c4` antes e depois **[M]** |
| **1** | Arquitetura NNUE v400 (HalfKP-4Bucket) | **PRONTO estruturalmente, rede NÃO treinada** | `nnue.h`/`nnue.c` implementam os 2560 features, L1/L2/L3, SCReLU — Partes 1-8 deste documento. `nnue_weights.bin` existe (~2.6 MB) mas com pesos aleatórios/placeholder — Fases 2-9 de teste de força (CLAUDE.md) não são válidas até haver um treino real |
| **2** | Arena nativa (`tools/arena.c`) | **PRONTO** (nomeado `arena.c`, não `arena_native.c` como o plano original previa — mudança de nome, não de escopo) | `run_arena.py` + `arena.c` confirmados nesta sessão: 2 tipos de player (`net:`/`uci:`), TT isolada por lado, SPRT real (`sprt_llr`, `--elo0`/`--elo1`) **[M]** |
| **2** | SPSA tuner (`tools/tune_spsa.c`) | **NÃO EXISTE** | Nenhum arquivo `*spsa*`/`*tune*` em `engine/c/tools/` **[M]** |
| **3** | Selfplay nativo (`tools/selfplay.c`) | **PRONTO, com driver Python novo** | `engine/c/tools/selfplay.c` (1330 linhas) implementa todos os requisitos do Apêndice F.2: threads in-process, `.bin` (`SelfplaySample`), TT compartilhada com `tt_clear()` físico por jogo, amostra por temperatura sobre MultiPV. Nesta sessão: build limpo confirmado, 2 jogos reais rodados via o novo `tests/run_selfplay_native.py` (driver com auto-rebuild por mtime, modelado em `run_arena.py`), saída `.bin` validada byte-a-byte (76 posições × 75 bytes) **[M]**. PGN, opening books (`openings/lines`, `openings/positions`) e random plies confirmados presentes — não é `.bin`-only (regra 9 do CLAUDE.md) **[M]**. Novo: `engine/build/build_selfplay.bat` para build manual |
| **3** | K-blend (`wl_target`, F.3.0) | **PRONTO** | `train/dataset.py:154` implementa `wl_target = k*result_prob + (1-k)*ev_prob` exatamente como especificado **[M]** |
| **3** | Bootstrap Estágio 1 (alpha-beta existente, sem política/MCTS) | **PRONTO** (é o que `selfplay.c` já faz — chama `search_best()` direto) | mesma evidência do selfplay nativo acima |
| **4** | Cabeça de política + MCTS (bootstrap Estágio 2) | **NÃO EXISTE** | Nenhum arquivo/símbolo relacionado a MCTS ou segunda cabeça de política em `train/` ou `engine/c/`; arquitetura atual da NNUE (CLAUDE.md) é só cabeça de valor **[M]** |
| **5** | Ciclo de bootstrap contínuo (gerar→treinar→arena→promove) | **NÃO EXISTE** | Nenhum script de orquestração encontrado; as peças (`selfplay.exe`, `train_nnue.py`, `arena.exe`) existem soltas mas não há um loop automático que as encadeia **[M]** |
| **1** | Primeiro treino real da rede v400 | **EM ANDAMENTO, primeira leva -200 ELO vs v3.14** | Feito por **outra sessão**, fora deste chat, usando o `CKPT_DIR` externo antigo (`C:/nnue_checkpoints/checkpoints_v400/`, o path que este documento já registra como corrigido para `checkpoints/v400/` — a outra sessão rodou antes dessa correção ou passou `--ckpt-dir` manualmente). Dois runs medidos em disco: `halfkp4b_v400` (6 épocas, terminou 16:11) e `strong_sf_stage2` (8 épocas, terminou 17:22, `--dataset-name` diferente — provavelmente uma segunda tentativa com outro dataset/config depois da primeira leva não ir bem) **[M]**. `engine/c/zchezz_v400/nnue_weights.bin` tem mtime 16:12 — bate com o fim do primeiro run, ou seja **o `.bin` hoje no repo provavelmente já é a rede exportada, não mais placeholder aleatório** **[M, inferido pelo timestamp — não confirmado lendo o binário]**. O resultado "-200 ELO vs v3.14" foi relatado pelo dono do projeto **[O]**, não há arquivo de resultado de torneio no repo para conferir o número exato ou a config do teste (quantos jogos, movetime, qual checkpoint exportado) — próximo passo é achar/pedir esse log antes de reagir ao número. |

**Outras correções feitas nesta sessão, fora do escopo original deste
documento mas relevantes para o pipeline de treino (v400 Parte 5 /
`train/train_nnue.py`):**
- Checkpoints migrados de `torch.save`(`.pt`) — não mais o JSON antigo
  do `mixtrain.py` — para `checkpoints/v400/` (antes apontava para fora
  do repo, `C:/nnue_checkpoints/...`).
- Restaurado: métrica MAE (treino + validação), cache de FEN codificado
  por arquivo parquet (`*_encoded.pt`), relatório de pico de VRAM,
  heartbeat de progresso intra-época, fallback gracioso ao carregar um
  checkpoint corrompido/incompatível.
- Bug real encontrado e corrigido nesta sessão: resumir treino
  (`--dataset-name` igual ao checkpoint) quebrava com
  `KeyError: 'initial_lr'` no `CosineAnnealingLR` — faltava popular
  `initial_lr` nos `param_groups` do otimizador antes de construir o
  scheduler com `last_epoch != -1`. Testado (2 épocas, resume real) após
  o fix **[M]**.
- Datasets em `data/` renomeados para convenção que codifica
  `cp`/`wdl`/`res`/`filter`/fonte(`sf`/`zchezz`/`virichess`)/data direto
  no nome da pasta (ver `data/Data.md`, que também foi reescrito como
  tabela única). `extra-quiet-n5k_sf` + `extra_quiet_raw_wdl` (mesmas
  posições, extraídas duas vezes) foram unidas por `fen` num único
  parquet, dobrando o corpus lambda-ativo.

**Documentação (fora do escopo original, feito nesta sessão):**
- Novo driver `tests/run_selfplay_native.py` (modelado em `run_arena.py`:
  auto-rebuild por mtime, todos os flags do `selfplay.exe` também como
  constantes de configuração no topo do arquivo — regra 8 do CLAUDE.md)
  + `engine/build/build_selfplay.bat` para build manual. Testado: build
  limpo + 2 jogos reais rodados via o driver **[M]**.
- `tests/run_selfplay.py`, `run_tournament.py`, `run_tournament_quick.py`,
  `run_suite.py` — documentação de cabeçalho expandida ao nível de
  `run_arena.py` (o que cada um faz, relação com os outros, decisões de
  design não-óbvias). `run_selfplay.py` também ganhou o bloco
  `CONFIGURATION` no formato da regra 8 (antes eram globais soltas).
  Nenhuma mudança de comportamento; todos os 4 arquivos re-verificados
  (`ast.parse` + `--help`) depois da edição **[M]**.
- `CLAUDE.md` reduzido de regras+documentação misturadas para só regras
  (pedido explícito do dono do projeto). O conteúdo descritivo que só
  existia lá (Architecture Notes do v4.00 — que era a ÚNICA descrição
  correta da arquitetura v4.00 no repo, já que a seção NNUE do
  `README.md` ainda descrevia a v3.14 como se fosse atual) foi migrado
  para `README.md`, que agora tem `### v4.00 (current)` e
  `### v3.14 (legacy, frozen)` claramente separados. `CLAUDE.md` ficou
  com: regras críticas 1-10, uma lista curta de "CRITICAL INVARIANTS"
  (as armadilhas tipo "trocar a ordem do concat quebra a avaliação"),
  convenção de nomes, e o essencial de `TTable`/`NnueNet` — o resto
  aponta para `README.md`. Também preenchidas lacunas que uma auditoria
  encontrou: `opening_pool.c`/`.h` e `sample.h` não apareciam em nenhum
  inventário de arquivos; agora aparecem no `README.md`.

## CONTEXTO: O QUE EXISTE HOJE (v3.14)

### Arquitetura atual
```
Input: 768 HM (Half-Mirror, sem dependência de rei) + 31 extras manuais = 799
L1:    799 → 256  (int16, acumulado incrementalmente para as 768 HM)
L2:    256 → 64   (int8, maddubs AVX2/WASM)
L3:    64  → 1    (int8 → float32)
Ativação: ClippedReLU [0, 255]
Arquivo de pesos: nnue_weights.bin, ~427 KB, formato NNU3
```

### O problema dos 31 extras
Os 31 features extras (contagens de peças, peões passados por coluna, distância Chebyshev
entre reis) **não podem ser acumulados** porque dependem de múltiplas peças
simultaneamente. Em `nnue_eval` e `nnue_eval_bb`, a cada nó do search:

```
_compute_extra_feat(...)     ← recomputa 31 features do estado do tabuleiro
_project_feat_full(...)      ← multiplica 31 features × 256 neurônios (SIMD)
acc_HM + ext_buf + bias      ← merge antes do ReLU
```

Isso é **O(31 × 256) SIMD por nó**, não eliminável com cache porque muda a cada lance.
Com HalfKP, esses recursos emergem organicamente das features locais — sem custo extra.

### O problema do HM (sem rei)
`feature = (cor × 6 + tipo_peça) × 64 + casa` → 768 features.
A rede não distingue "Torre em e4 com rei em g1" de "Torre em e4 com rei em a8".
Isso é o principal teto de força.

---

## ARQUITETURA NOVA: HalfKP-4Bucket (FASE 1)

### Definição das features

```
Perspectiva Branca:
  king_bucket(b->wk) = (b->wk % 8 >= 4 ? 1 : 0) | (b->wk / 8 >= 4 ? 2 : 0)
  → 4 buckets: 0=queenside-baixo, 1=kingside-baixo, 2=queenside-alto, 3=kingside-alto

  Para cada peça não-rei (p, sq) no tabuleiro:
    feature_W = bucket × 640 + piece_color_type_idx(p) × 64 + sq_from_white_pov(sq)
    → 0 .. 2559

Perspectiva Preta:
  king_bucket(b->bk) calculado sobre bk espelhado verticalmente:
    bk_mirrored = b->bk ^ 56
    bucket_B = (bk_mirrored % 8 >= 4 ? 1 : 0) | (bk_mirrored / 8 >= 4 ? 2 : 0)

  Para cada peça não-rei (p, sq):
    sq_from_black_pov = sq ^ 56   (espelha verticalmente)
    cor relativa ao preto: se preta → offset 0, se branca → offset 6
    feature_B = bucket_B × 640 + piece_color_type_idx_relative_to_black(p) × 64 + sq_from_black_pov
    → 0 .. 2559
```

`piece_color_type_idx`: P=0..5 (branco), P=6..11 (preto), sem rei (rei não entra nas features).
Total de features possíveis: `4 × 640 = 2560` por perspectiva.

**INVARIANTE CRÍTICO:** O rei não é incluído como feature de nenhuma perspectiva.
Apenas as outras 10 peças (P, N, B, R, Q × 2 cores).

### Camadas

```
Feature input:  2560 (HalfKP-4bucket, por perspectiva)
L1:             2560 → 512  (int16, acumulador incremental por perspectiva)
Ativação L1:    SCReLU: c = clamp(x / QA, 0, 1); out = c * c  (em float durante treino)
Concat:         [acc_W(512), acc_B(512)] → 1024 uint8 (após SCReLU + quantização)
L2:             1024 → 32   (int8, maddubs)
Ativação L2:    ClippedReLU [0, QB=64]
L3:             32  → 1     (int8 → float32)
```

### Arquivo de pesos (NNU4)

```
Magic:      "NNU4" (4 bytes)
Epoch:      uint32 (4 bytes)
Dims:       5 × uint32: [L1_IN=2560, L1_OUT=512, L2_IN=1024, L2_OUT=32, L3_IN=32]
Scales:     4 × float32: [QA=255, QB=64, SHIFT=8, OUT_SCALE]
L1W:        [2560][512] int16   (2.5 MB)
L1B:        [512]       int32
L2W:        [32][1024]  int8   (row-major por output, para maddubs)
L2B:        [32]        int32
L3W:        [32]        int8
L3B:        float32
```

Tamanho total estimado: **~2.6 MB** (viável para WASM).

### Memória por thread (NnueAccum)

```
acc_stack_w[128][512]  int16  → 128 KB
acc_stack_b[128][512]  int16  → 128 KB
acc_w[512], acc_b[512]        → 2 KB (scratch para rebuild)
acc_dirty, acc_ptr            → 8 B
bucket_w, bucket_b            → 2 bytes (bucket atual de cada rei)
needs_refresh_w, needs_refresh_b → 2 bytes (flag para refresh de bucket)
Total: ~258 KB por thread (vs ~68 KB atual)
```

---

## LISTA COMPLETA DE ARQUIVOS A MODIFICAR (FASE 1)

### Diretório base: `engine/c/zchezz_v314/`
Renomear a pasta para `zchezz_v400/` ao terminar.

```
1. nnue.h           ← constants, NnueAccum struct
2. nnue.c           ← feature encoding, forward pass, acumulador, loader
3. board.h          ← nenhuma mudança estrutural necessária
4. board.c          ← board_make: detectar mudança de bucket ao mover rei
5. main.c           ← atualizar version string, verificar heap alloc de NnueAccum
```

### Diretório: `train/`
```
6. train_nnue.py      ← nova encode_chunk, novo modelo NNUE, SCReLU, dims
7. export_nnu4.py  ← novo conversor NNU4 com novas dims
```

---

## PARTE 1 (FASE 1): `nnue.h` — Constantes e NnueAccum

### Mudanças nas constantes

```c
/* REMOVER: */
#define NN_L1_IN   799
#define NN_HM_IN   768
#define NN_EXTRA    31

/* ADICIONAR: */
#define NN_FEAT_IN      2560   /* HalfKP-4bucket: 4 buckets × 640 features por perspectiva */
#define NN_L1_IN        2560   /* alias de NN_FEAT_IN */
#define NN_KING_BUCKETS    4

/* MODIFICAR: */
#define NN_L1_OUT   512   /* era 256 */
#define NN_L2_IN   1024   /* era 256 — agora concat de 2 perspectivas */
#define NN_L2_OUT    32   /* era 64 */
#define NN_L3_IN     32   /* deve igualar NN_L2_OUT */

/* MANTIDOS: */
#define NN_QA       255
#define NN_QB        64
#define NN_SHIFT      8
#define NN_ACC_STACK  128
#define NN_ACC_DEPTH  512
```

### NnueAccum — nova struct

```c
typedef struct {
    /* Acumuladores por perspectiva — HalfKP-4bucket, 512 neurônios */
    int16_t  acc_stack_w[NN_ACC_STACK][NN_L1_OUT] __attribute__((aligned(32)));
    int16_t  acc_stack_b[NN_ACC_STACK][NN_L1_OUT] __attribute__((aligned(32)));
    int      acc_ptr;

    /* Scratch para rebuild */
    int16_t  acc_w[NN_L1_OUT] __attribute__((aligned(32)));
    int16_t  acc_b[NN_L1_OUT] __attribute__((aligned(32)));

    /* Estado do bucket do rei — determina quando fazer refresh */
    uint8_t  bucket_w;          /* bucket atual do rei branco */
    uint8_t  bucket_b;          /* bucket atual do rei preto */
    uint8_t  needs_refresh_w;   /* 1 → rebuild acc_w na próxima eval */
    uint8_t  needs_refresh_b;   /* 1 → rebuild acc_b na próxima eval */

    /* Flag de rebuild total (como antes) */
    int      acc_dirty;
} NnueAccum;
```

**REMOVER da struct:** `ext_buf`, `ext_feat`, `ext_dirty`, `cache_key`, `cache_buf`.
Esses campos não existem mais — não há features extras.

### API pública — manter iguais as assinaturas:
```c
void nnue_rebuild(NnueAccum *na, const uint8_t *board);
void nnue_push_na(NnueAccum *na, const uint8_t *board, const NNMove *m);
void nnue_pop_na(NnueAccum *na);
int  nnue_eval(NnueAccum *na, int stm, const uint8_t *board);
int  nnue_eval_bb(NnueAccum *na, int stm, const uint8_t *board,
                  const uint64_t bb[12], uint64_t board_hash);
int  nnue_load(const char *path);
int  nnue_load_from_mem(const uint8_t *data, size_t len);
```

**REMOVER da API pública:** `nnue_reset` (substituída por limpar acc_dirty + buckets).
Ou manter `nnue_reset` com nova implementação sem ext_cache.

### NNMove — sem mudança
```c
typedef struct {
    uint8_t from_sq;
    uint8_t to_sq;
    uint8_t prom;
    uint8_t is_epc;
    uint8_t castle;
} NNMove;
```

---

## PARTE 2 (FASE 1): `nnue.c` — Feature Encoding e Forward Pass

### 2.1 Função auxiliar: king_bucket

```c
/* Converte casa do rei (0..63) no layout do Zchezz (0=a8, 63=h1)
 * em bucket 0..3.
 *
 * Layout do Zchezz: sq=0 é a8, sq=63 é h1.
 *   file = sq % 8   (0=file a, 7=file h)
 *   rank = sq / 8   (0=rank 8, 7=rank 1)
 *
 * Bucket:
 *   bit 0: kingside (file >= 4)
 *   bit 1: "baixo" no tabuleiro de preto = rank 5-8 = rank_idx >= 4 no Zchezz
 *
 * Para perspectiva branca: usa sq direto.
 * Para perspectiva preta:  usa sq ^ 56 (espelho vertical) antes de calcular.
 */
static inline int king_bucket(int sq) {
    int file = sq % 8;
    int rank = sq / 8;
    return (file >= 4 ? 1 : 0) | (rank >= 4 ? 2 : 0);
}
```

### 2.2 Função auxiliar: halfkp_feature_index

```c
/* Retorna o índice HalfKP-4bucket para uma peça, dado o bucket do rei
 * da perspectiva que está sendo calculada.
 *
 * bucket:     0..3 (resultado de king_bucket do rei da perspectiva)
 * piece_color_rel: 0=aliado P, 1=aliado N, 2=aliado B, 3=aliado R, 4=aliado Q,
 *                  5=inimigo P, 6=inimigo N, 7=inimigo B, 8=inimigo R, 9=inimigo Q
 *                  (0..9, nunca inclui rei)
 * piece_sq:   0..63, já na perspectiva correta (espelhado se for perspectiva preta)
 *
 * Retorna: 0 .. 2559
 */
static inline int halfkp_feat(int bucket, int piece_color_rel, int piece_sq) {
    return bucket * 640 + piece_color_rel * 64 + piece_sq;
}

/* Calcula piece_color_rel a partir da peça e da perspectiva.
 *
 * p:        código de peça do Zchezz (WP=9..BK=22)
 * is_white_pov: 1 se calculando perspectiva branca, 0 se preta
 *
 * Tipos: P=0, N=1, B=2, R=3, Q=4 (rei = retorna -1, não incluir)
 * Aliado: mesmo lado da perspectiva → offset 0
 * Inimigo: lado oposto → offset 5
 */
static inline int halfkp_piece_rel(uint8_t p, int is_white_pov) {
    int t = PC_TYPE(p) - 1;   /* P=0..Q=4, K=5 */
    if (t < 0 || t > 5) return -1;
    if (t == 5) return -1;   /* rei não é feature */
    int is_white_piece = (PC_COLOR(p) == COL_W);
    int is_ally = (is_white_pov == is_white_piece);
    return (is_ally ? 0 : 5) + t;
}
```

### 2.3 _acc_add_piece e _acc_sub_piece — nova versão

Os helpers agora recebem o bucket do rei de cada perspectiva.

```c
/*
 * Atualiza acumuladores de AMBAS as perspectivas para uma peça em sq.
 * wk_sq: casa atual do rei branco (no layout Zchezz)
 * bk_sq: casa atual do rei preto  (no layout Zchezz)
 *
 * Calcula dois índices de feature (um por perspectiva) e adiciona
 * as duas colunas correspondentes de _nnL1WT ao acc.
 */
static inline void _acc_add_piece(int16_t *accW, int16_t *accB,
                                  uint8_t p, int sq,
                                  int wk_sq, int bk_sq) {
    /* Perspectiva branca */
    int rel_w = halfkp_piece_rel(p, 1);
    if (rel_w >= 0) {
        int bkt_w = king_bucket(wk_sq);
        int sq_w  = sq ^ 56;   /* Zchezz sq 0=a8; Python/branco-POV: sq 0=a1 → XOR56 */
        int fidx_w = halfkp_feat(bkt_w, rel_w, sq_w);
        const int16_t *row_w = _nnL1WT + fidx_w * NN_L1_OUT;
        /* AVX2 / WASM / scalar: idêntico ao atual mas com NN_L1_OUT=512 */
        for (int o = 0; o < NN_L1_OUT; o += 16) { /* AVX2: 16 int16 por reg */ ... }
    }

    /* Perspectiva preta */
    int rel_b = halfkp_piece_rel(p, 0);
    if (rel_b >= 0) {
        int bk_mirrored = bk_sq ^ 56;
        int bkt_b = king_bucket(bk_mirrored);
        int sq_b  = sq;   /* perspectiva preta: sq não espelha o sq da peça? */
        /* ATENÇÃO: o sq da peça na perspectiva preta É sq ^ 56 (espelhado).
         * Confirmar convenção com Python encode_chunk (ver Parte 6).
         * CONVENÇÃO ADOTADA: sq da peça também espelha para perspectiva preta. */
        sq_b = sq ^ 56;
        int fidx_b = halfkp_feat(bkt_b, rel_b, sq_b);
        const int16_t *row_b = _nnL1WT + fidx_b * NN_L1_OUT;
        for (int o = 0; o < NN_L1_OUT; o += 16) { ... }
    }
}
```

**NOTA CRÍTICA para o agente:** A convenção de `sq ^ 56` deve ser **idêntica** entre
`_acc_add_piece` no C e `encode_chunk` no Python. Valide com um teste de simetria:
para uma posição espelhada, o output da rede deve ser o mesmo. O Zchezz já usa `sq ^ 56`
para a perspectiva branca no `_acc_add_piece` atual — manter isso.

A assinatura de `_acc_sub_piece` é a mesma (subtrai ao invés de somar).

### 2.4 nnue_rebuild — nova implementação

```c
void nnue_rebuild(NnueAccum *na, const uint8_t *board) {
    /* Recalcular buckets dos reis */
    int wk_sq = -1, bk_sq = -1;
    for (int sq = 0; sq < 64; sq++) {
        if (board[sq] == WK) wk_sq = sq;
        if (board[sq] == BK) bk_sq = sq;
    }
    na->bucket_w = (uint8_t)king_bucket(wk_sq);
    na->bucket_b = (uint8_t)king_bucket(bk_sq ^ 56);
    na->needs_refresh_w = 0;
    na->needs_refresh_b = 0;

    /* Inicializar acumuladores com bias */
    int16_t *dW = na->acc_stack_w[0];
    int16_t *dB = na->acc_stack_b[0];
    /* Copiar bias L1 (int32 → int16) para seed */
    for (int o = 0; o < NN_L1_OUT; o++) {
        dW[o] = (int16_t)_nnL1B[o];
        dB[o] = (int16_t)_nnL1B[o];
    }

    /* Adicionar cada peça */
    for (int sq = 0; sq < 64; sq++) {
        uint8_t p = board[sq];
        if (!p || PC_TYPE(p) == 6) continue;   /* vazio ou rei: pular */
        _acc_add_piece(dW, dB, p, sq, wk_sq, bk_sq);
    }

    na->acc_dirty = 0;
    na->acc_ptr = 0;
    /* Copiar para scratchpads */
    memcpy(na->acc_w, dW, NN_L1_OUT * sizeof(int16_t));
    memcpy(na->acc_b, dB, NN_L1_OUT * sizeof(int16_t));
}
```

**DIFERENÇA DO ATUAL:** O loop de scan agora chama `_acc_add_piece` com `wk_sq` e `bk_sq`.
Também: seed com bias em int16 (se o bias couber; se não, fazer como antes com int32 no eval).

**NOTA:** No design atual, o bias é adicionado no forward pass (`_nnL1B` é int32 e somado
durante o ClippedReLU). Para manter compatibilidade com o SIMD existente, **não** colocar
o bias no acumulador — mantê-lo no forward pass como hoje. O `nnue_rebuild` então só
zera os acumuladores e chama `_acc_add_piece` para cada peça, **sem** seed de bias.

### 2.5 nnue_push_na — bucket detection

```c
void nnue_push_na(NnueAccum *na, const uint8_t *board, const NNMove *m) {
    int src = na->acc_ptr, dst = src + 1;
    if (dst >= NN_ACC_STACK) { na->acc_dirty = 1; return; }
    if (na->acc_dirty) { nnue_rebuild(na, board); src = 0; dst = 1; }

    /* Copiar acumuladores para o próximo slot */
    memcpy(na->acc_stack_w[dst], na->acc_stack_w[src], NN_L1_OUT * sizeof(int16_t));
    memcpy(na->acc_stack_b[dst], na->acc_stack_b[src], NN_L1_OUT * sizeof(int16_t));

    int16_t *cW = na->acc_stack_w[dst];
    int16_t *cB = na->acc_stack_b[dst];

    /* Detectar o rei branco e preto ANTES do lance (usando board[]) */
    int wk_sq = -1, bk_sq = -1;
    /* OPTIMIZAÇÃO: Board já tem b->wk e b->bk — passar via NNMove ou ler do board.
     * Por ora: ler do board (igual ao rebuild). Depois: passar wk/bk via parâmetro. */
    for (int s = 0; s < 64; s++) {
        if (board[s] == WK) wk_sq = s;
        if (board[s] == BK) bk_sq = s;
    }

    /* Aplicar o lance ao acumulador */
    if (m->castle) {
        /* Castling: mover rei e torre. Só remove/adiciona não-reis relevantes.
         * NOTA: o rei não entra nas features. A torre sim.
         * Detectar novo king_sq após castling para atualizar bucket. */
        const int *sq_table = _castle_sq[m->castle];
        /* sq_table: {kf, kt, rf, rt} */
        int kf = sq_table[0], kt = sq_table[1];
        int rf = sq_table[2], rt = sq_table[3];
        /* Torre: remover de rf, adicionar em rt */
        _acc_sub_piece(cW, cB, board[rf], rf, wk_sq, bk_sq);
        _acc_add_piece(cW, cB, board[rf], rt, kt,    bk_sq);  /* kt = novo wk após roque */
        /* Atualizar bucket se o rei branco ou preto mudou de bucket */
        int is_white_castle = (board[kf] == WK);
        if (is_white_castle) {
            int new_bucket = king_bucket(kt);
            if (new_bucket != na->bucket_w) {
                na->needs_refresh_w = 1;
                na->bucket_w = (uint8_t)new_bucket;
            }
        } else {
            int new_bucket = king_bucket(kt ^ 56);
            if (new_bucket != na->bucket_b) {
                na->needs_refresh_b = 1;
                na->bucket_b = (uint8_t)new_bucket;
            }
        }
    } else {
        int f = m->from_sq, to = m->to_sq;
        uint8_t p = board[f], cap = board[to];

        /* Detectar lance de rei */
        if (PC_TYPE(p) == 6) {
            /* Rei moveu — atualizar bucket */
            int is_white_king = (PC_COLOR(p) == COL_W);
            if (is_white_king) {
                wk_sq = to;   /* novo wk_sq para calcular features das peças abaixo */
                int new_bucket = king_bucket(to);
                if (new_bucket != na->bucket_w) {
                    na->needs_refresh_w = 1;
                    na->bucket_w = (uint8_t)new_bucket;
                }
            } else {
                bk_sq = to;
                int new_bucket = king_bucket(to ^ 56);
                if (new_bucket != na->bucket_b) {
                    na->needs_refresh_b = 1;
                    na->bucket_b = (uint8_t)new_bucket;
                }
            }
            /* Rei não é feature — não chama _acc_add/sub para o rei */
        } else {
            /* Peça não-rei: update incremental normal */
            _acc_sub_piece(cW, cB, p, f, wk_sq, bk_sq);
            if (cap && PC_TYPE(cap) != 6)
                _acc_sub_piece(cW, cB, cap, to, wk_sq, bk_sq);
            if (m->is_epc) {
                int epsq = (PC_COLOR(p) == COL_W) ? to + 8 : to - 8;
                if (board[epsq] && PC_TYPE(board[epsq]) != 6)
                    _acc_sub_piece(cW, cB, board[epsq], epsq, wk_sq, bk_sq);
            }
            uint8_t landing = m->prom ? (uint8_t)(PC_COLOR(p) | m->prom) : p;
            _acc_add_piece(cW, cB, landing, to, wk_sq, bk_sq);
        }
    }

    na->acc_ptr = dst;
    /* ext_dirty removido — não existe mais */
}
```

**LÓGICA DO NEEDS_REFRESH:**
Quando `needs_refresh_w = 1` ou `needs_refresh_b = 1`, o acumulador daquele slot está
**errado** (foi calculado com o bucket antigo). O `nnue_eval` deve detectar isso e fazer
um rebuild parcial (só da perspectiva afetada) antes de avaliar.

Implementação no `nnue_eval_bb`:
```c
if (na->needs_refresh_w) {
    /* Rebuild perspectiva branca a partir do board atual */
    _rebuild_perspective_w(na, board, wk_sq);
    na->needs_refresh_w = 0;
}
```

Essa abordagem é chamada **"lazy refresh"** e é o que Stockfish usa desde NNUE-5.

### 2.6 nnue_pop_na — sem mudança
```c
void nnue_pop_na(NnueAccum *na) {
    if (na->acc_ptr > 0) na->acc_ptr--;
    /* Após pop, restaurar bucket para o bucket do slot anterior.
     * PROBLEMA: o bucket não está no stack, está como campo plano.
     * SOLUÇÃO: incluir bucket_w e bucket_b no acc_stack como campos extras,
     * ou fazer rebuild se needs_refresh. */
}
```

**ATENÇÃO — problema de pop com bucket:**
Quando há pop após um lance de rei que mudou o bucket, o `na->bucket_w` ficou incorreto
(aponta para o bucket pós-lance, mas voltamos para pré-lance). **Solução:** incluir
`bucket_w` e `bucket_b` no `acc_stack` como dois uint8 extras em cada frame, ou usar
um stack separado de 2 bytes × 128.

**Implementação recomendada:**
```c
/* Em NnueAccum, adicionar: */
uint8_t bucket_stack_w[NN_ACC_STACK];
uint8_t bucket_stack_b[NN_ACC_STACK];
```

No `push_na`, antes do `memcpy`:
```c
na->bucket_stack_w[dst] = na->bucket_w;   /* salva bucket pré-lance */
na->bucket_stack_b[dst] = na->bucket_b;
```
No `pop_na`:
```c
na->acc_ptr--;
na->bucket_w = na->bucket_stack_w[na->acc_ptr];
na->bucket_b = na->bucket_stack_b[na->acc_ptr];
```

### 2.7 Forward pass — nnue_eval_bb

#### Step 1: SCReLU no lugar de ClippedReLU para L1

```c
/* Após soma acc + bias, aplicar SCReLU:
 *   c = clamp(x / QA, 0, 1)   → x clamped entre 0 e QA em int
 *   out = c * c / QA           → produto int, depois shift
 *
 * Implementação int: o acumulador está em escala int16 (não dividido por QA).
 * ClippedReLU atual: out = clamp(x + bias, 0, QA) → uint8
 * SCReLU: out = (clamp(x + bias, 0, QA))^2 / QA → uint8
 *
 * Para manter uint8 [0, 255]:
 *   c = clamp(x + bias, 0, 255)          (int32)
 *   out = (c * c) >> 8                    (= c²/256 ≈ c²/QA)
 *
 * AVX2: requer mulhi_epi16 ou conversão para int32.
 * IMPLEMENTAÇÃO SIMPLES (compatível com AVX2 e WASM):
 *   1. Clamp acc+bias para [0, 255] → uint8 como hoje
 *   2. Widen para int16
 *   3. Multiplicar int16 × int16 → int32 (via madd ou mullo)
 *   4. Shift >> 8 → uint8
 */
```

**Nota sobre L1 → L2 concat:**
Com NN_L1_OUT=512 e duas perspectivas, `relu1[1024] uint8`. O índice correto é:
- `relu1[0..511]` = perspectiva STM (quem move)
- `relu1[512..1023]` = perspectiva do oponente

O L2 weight matrix é `[32][1024]`, e o kernel AVX2 de `maddubs_epi16` funciona
identicamente com `NN_L2_IN=1024` — só muda o número de iterações (de 256/32=8 para
1024/32=32).

#### Steps 2-6: quase iguais ao atual

```
Step 2: SCReLU → relu1[1024] uint8
Step 3: L2 maddubs kernel (NN_L2_IN=1024, NN_L2_OUT=32) → acc2[32] int32
Step 4: shift + ClippedReLU → relu2[32] uint8
Step 5: L3 dot product int32
Step 6: scale + bias → centipawns
```

**O kernel maddubs não precisa mudar** — só processa mais iterações (1024 vs 256 inputs).
O unroll 4-wide continua funcionando: `for (o = 0; o < 32; o += 4)`.

### 2.8 Loader NNU4

```c
/* Magic: "NNU4" */
/* Verificar magic, ler dims, ler scales */
/* L1_SZ = NN_L1_IN * NN_L1_OUT = 2560 * 512 */
/* L2_SZ = NN_L2_OUT * NN_L2_IN = 32 * 1024 */
/* Sem transpose no L2W (igual ao NNU3 atual — já sem transpose desde v3.14) */
```

Atualizar `nnue_load` e `nnue_load_from_mem` para magic "NNU4" e novos offsets.

---

## PARTE 3 (FASE 1): `board.c` — Detecção de Mudança de Bucket

A única mudança em `board.c` é passar `wk_sq` e `bk_sq` para `nnue_push_na`.

**Opção A (mais limpa):** Adicionar `wk_sq` e `bk_sq` ao `NNMove`:
```c
typedef struct {
    uint8_t from_sq;
    uint8_t to_sq;
    uint8_t prom;
    uint8_t is_epc;
    uint8_t castle;
    uint8_t wk_sq;   /* NOVO: casa do rei branco ANTES do lance */
    uint8_t bk_sq;   /* NOVO: casa do rei preto  ANTES do lance */
} NNMove;
```

Em `board_make`, antes de chamar `nnue_push_na`:
```c
NNMove nm;
nm.from_sq = (uint8_t)f;
nm.to_sq   = (uint8_t)to;
nm.prom    = m->prom;
nm.is_epc  = m->epc;
nm.castle  = m->castle ? (col==COL_W ? (to>f ? 1 : 2) : (to>f ? 3 : 4)) : 0;
nm.wk_sq   = b->wk;   /* NOVO */
nm.bk_sq   = b->bk;   /* NOVO */
nnue_push_na(b->nnue, b->b, &nm);
```

Isso elimina o scan `for (s=0; s<64; s++)` dentro de `nnue_push_na`.

**Opção B (sem mudança em NNMove):** `nnue_push_na` faz o scan do board como hoje.
Mais lento mas mais simples. Aceitar para o primeiro draft e otimizar depois.

---

## PARTE 4 (FASE 1): `main.c`

1. Atualizar string de versão: `"Zchezz v400"`
2. `sizeof(NnueAccum)` agora é ~258 KB — verificar que `zmalloc32` no `helper_thread_fn`
   aloca corretamente (já usa `sizeof(NnueAccum)`, não hardcoded).
3. Atualizar comentário do header.

---

## PARTE 5 (FASE 1): `train/train_nnue.py` — Novo Modelo

### 5.1 Constantes novas

```python
INPUT_MAIN    = 2560   # HalfKP-4bucket (era 768)
INPUT_EXTRA   = 0      # REMOVIDO
INPUT_TOTAL   = 2560   # (era 799)
HIDDEN1       = 512    # L1 por perspectiva (era 256)
HIDDEN2       = 32     # L2 (era 64)
CONCAT_DIM    = 1024   # HIDDEN1 * 2 (novo)
```

### 5.2 SCReLU

```python
class SCReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.clamp(0.0, 1.0)
        return c * c
```

### 5.3 Modelo NNUE com perspectivas explícitas

```python
class NNUE(nn.Module):
    def __init__(self):
        super().__init__()
        # L1: mesmos pesos para ambas as perspectivas
        # Input: (batch, 2560) por perspectiva → dois tensores separados
        self.l1    = nn.Linear(INPUT_MAIN, HIDDEN1)   # shared weights
        self.act1  = SCReLU()
        self.l2    = nn.Linear(HIDDEN1 * 2, HIDDEN2)  # concat de perspectivas
        self.act2  = ClippedReLU(1.0)                 # ClippedReLU no L2
        self.l3    = nn.Linear(HIDDEN2, 1)

    def forward(self, x_stm: torch.Tensor, x_opp: torch.Tensor) -> torch.Tensor:
        """
        x_stm: (batch, 2560) features da perspectiva do jogador que move
        x_opp: (batch, 2560) features da perspectiva do oponente
        """
        # QAT fake-quant
        w1 = fake_quant_int16(self.l1.weight, QA)
        b1 = fake_quant_bias_int32(self.l1.bias, float(QA))

        h_stm = self.act1(F.linear(x_stm, w1, b1))
        h_opp = self.act1(F.linear(x_opp, w1, b1))

        # QAT: fake-quant da ativação
        h_stm_q = (h_stm * QA).round().clamp(0, QA) / QA
        h_stm = h_stm + (h_stm_q - h_stm).detach()
        h_opp_q = (h_opp * QA).round().clamp(0, QA) / QA
        h_opp = h_opp + (h_opp_q - h_opp).detach()

        # Concatenar perspectivas: STM sempre primeiro
        h = torch.cat([h_stm, h_opp], dim=1)   # (batch, 1024)

        w2 = fake_quant_int8(self.l2.weight, QB)
        b2 = fake_quant_bias_int32(self.l2.bias, float(QA * QB))
        h2 = self.act2(F.linear(h, w2, b2))

        h2_q = (h2 * QB).round().clamp(0, QB) / QB
        h2 = h2 + (h2_q - h2).detach()

        w3 = fake_quant_int8(self.l3.weight, QB)
        return torch.sigmoid(F.linear(h2, w3, self.l3.bias))
```

**NOTA:** `l1.weight` é `[HIDDEN1, INPUT_MAIN] = [512, 2560]`. **Os mesmos pesos são
usados para STM e OPP** — isso é a "perspectiva compartilhada" que garante
eficiência e simetria.

### 5.4 encode_chunk — nova versão

```python
PIECE_MAP = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN:  4
    # KING não mapeado — não é feature
}

def king_bucket(king_sq: int) -> int:
    """king_sq em coordenadas Python-chess (0=a1, 63=h8)"""
    file_ = king_sq % 8
    rank_ = king_sq // 8
    return (1 if file_ >= 4 else 0) | (2 if rank_ >= 4 else 0)

def halfkp_features(board: chess.Board, pov_is_white: bool) -> np.ndarray:
    """
    Retorna array (2560,) uint8 com as features HalfKP-4bucket
    da perspectiva pov_is_white.

    CONVENÇÃO DE COORDENADAS:
    Python-chess: sq 0=a1, 63=h8.
    Perspectiva branca: usa sq direto.
    Perspectiva preta:  espelha verticalmente = sq ^ 56 (file=mesmo, rank=invertido).
    Isso é consistente com board.mirror() que o encode atual já usa.
    """
    feats = np.zeros(2560, dtype=np.uint8)

    if pov_is_white:
        king_sq = board.king(chess.WHITE)
        if king_sq is None: return feats
        bucket = king_bucket(king_sq)
    else:
        king_sq = board.king(chess.BLACK)
        if king_sq is None: return feats
        bucket = king_bucket(king_sq ^ 56)   # espelhar antes do bucket

    for sq, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue   # rei não é feature

        piece_type_idx = PIECE_MAP.get(piece.piece_type, -1)
        if piece_type_idx < 0:
            continue

        # Cor relativa à perspectiva
        if pov_is_white:
            is_ally = (piece.color == chess.WHITE)
            sq_pov  = sq   # perspectiva branca: sq direto
        else:
            is_ally = (piece.color == chess.BLACK)
            sq_pov  = sq ^ 56   # espelhar para perspectiva preta

        color_offset = 0 if is_ally else 5
        feat_idx = bucket * 640 + (color_offset + piece_type_idx) * 64 + sq_pov
        feats[feat_idx] = 1

    return feats

def encode_chunk(fens):
    """
    Retorna (X_stm, X_opp) onde cada um é (N, 2560) uint8.
    STM = perspectiva do jogador que move.
    OPP = perspectiva do oponente.
    """
    import chess
    N = len(fens)
    X_stm = np.zeros((N, 2560), dtype=np.uint8)
    X_opp = np.zeros((N, 2560), dtype=np.uint8)

    for i, fen in enumerate(fens):
        board = chess.Board(fen)
        stm_is_white = (board.turn == chess.WHITE)

        X_stm[i] = halfkp_features(board, pov_is_white=stm_is_white)
        X_opp[i] = halfkp_features(board, pov_is_white=not stm_is_white)

    return X_stm, X_opp
```

**REMOVER** do `encode_chunk`: todo o código de peões passados, distâncias, contagens.

**Adaptar** `build_tensors_parallel` para retornar `(X_stm, X_opp, y)` e o training loop
para chamar `model(X_stm_batch, X_opp_batch)`.

**REMOVER** `out_extra`, `INPUT_EXTRA`, o dataloader de features extras, e a concatenação
`torch.cat([X_bits, X_extra], dim=1)`.

**REMOVER** espelhamento com `board.mirror()` no encode — agora o encode já lida com
perspectiva explicitamente via `pov_is_white`. **Manter** a lógica de inverter `y_base`
para pretas: `y_base[is_black] = 1.0 - y_base[is_black]` (label continua do ponto de
vista branco).

### 5.5 Atualizar arch dict no checkpoint

```python
'arch': {
    'input':  INPUT_MAIN,     # 2560
    'h1':     HIDDEN1,        # 512
    'concat': CONCAT_DIM,     # 1024
    'h2':     HIDDEN2,        # 32
    'encoding': 'halfkp_4bucket'
}
```

---

## PARTE 6 (FASE 1): `train/export_nnu4.py` — Conversor NNU4

```python
L1_IN  = 2560
L1_OUT = 512
L2_IN  = 1024
L2_OUT = 32
L3_IN  = 32

QA = 255.0
QB = 64.0
SHIFT = 8.0
OUT_SCALE = 320.0 / (QB * QB)   # = 320 / 4096 ≈ 0.0781

# Extrair tensores
L1W = np.array(w["l1.weight"], dtype=np.float32)   # [512, 2560]
L1B = np.array(w["l1.bias"],   dtype=np.float32)   # [512]
L2W = np.array(w["l2.weight"], dtype=np.float32)   # [32, 1024]
L2B = np.array(w["l2.bias"],   dtype=np.float32)   # [32]
L3W = np.array(w["l3.weight"], dtype=np.float32)   # [1, 32]
L3B = np.array(w["l3.bias"],   dtype=np.float32)   # [1]

# Quantizar
L1W_q = quant16(L1W, QA)         # [512, 2560] int16
L1B_q = quant_bias_int32(L1B, QA)
L2W_q = quant8(L2W, QB)          # [32, 1024] int8  (já row-major por output — SEM transpose)
L2B_q = quant_bias_int32(L2B, QA * QB)
L3W_q = quant8(L3W.flatten(), QB)
L3B_f = float(L3B[0])

# Transpor L1W para layout C: [L1_IN][L1_OUT] = [2560][512]
# O modelo PyTorch armazena [out_features][in_features] = [512][2560]
# O C acessa _nnL1WT[feat_idx * L1_OUT + o], ou seja, [feat_idx][o]
# → precisamos transpor: L1W_T = L1W_q.T  → [2560, 512]
L1W_T = np.ascontiguousarray(L1W_q.T)   # [2560, 512] int16

# Escrever NNU4
with open(DST, 'wb') as f:
    f.write(b'NNU4')
    f.write(struct.pack('<I', epoch))
    f.write(struct.pack('<5I', L1_IN, L1_OUT, L2_IN, L2_OUT, L3_IN))
    f.write(struct.pack('<4f', QA, QB, SHIFT, OUT_SCALE))
    f.write(L1W_T.tobytes())    # [2560][512] int16
    f.write(L1B_q.tobytes())    # [512] int32
    f.write(L2W_q.tobytes())    # [32][1024] int8  (row-major por output)
    f.write(L2B_q.tobytes())    # [32] int32
    f.write(L3W_q.tobytes())    # [32] int8
    f.write(struct.pack('<f', L3B_f))
```

**VERIFICAR:** O layout da L1W em memória é `[feat_idx * NN_L1_OUT + neuron_idx]`.
Isso significa que para adicionar feature `k` ao acumulador, somamos a linha inteira
`_nnL1WT[k * 512 .. k*512 + 511]`. O transpose de `[512, 2560]` para `[2560, 512]`
garante que as linhas sejam contíguas em memória — idêntico ao que já é feito hoje.

---

## PARTE 7 (FASE 1): WASM — Impacto e Compatibilidade

O WASM usa `nnue_load_from_mem` e o acumulador `g_nnue_accum` (estático global).

**Mudanças necessárias:**
1. `g_nnue_accum` agora é 258 KB (vs 68 KB). Verificar que o WASM stack/heap suporta.
   Se necessário, usar `malloc` no init do WASM em vez de variável global.
2. O kernel WASM SIMD128 de L1 precisa ser atualizado para processar
   `NN_L1_OUT=512` (era 256) — só muda o loop count, lógica idêntica.
3. O kernel WASM SIMD128 de L2 processa `NN_L2_IN=1024` inputs — `NN_L2_OUT=32`
   outputs, loop count aumenta de 16 para 64, lógica idêntica.
4. Magic "NNU4" no lugar de "NNU3".

---

## PARTE 8 (FASE 1): TESTES E VALIDAÇÃO

### 8.1 Verificação de simetria (crítica)
Após implementar o encode Python:
```python
# Espelhar o tabuleiro e verificar que a avaliação é negada
board_orig = chess.Board(fen)
board_mirror = board_orig.mirror()
# Avaliação de board_orig com brancas = -Avaliação de board_mirror com brancas
```

### 8.2 Verificação de paridade C/Python
Para uma posição fixa, calcular manualmente os feature indices em Python e em C
e verificar que são iguais.

### 8.3 Perft
Rodar `perft` em posições padrão após as mudanças em `board.c`/`nnue.c` para
confirmar que nenhuma mudança afetou a geração de lances.

### 8.4 Regression de eval
Comparar avaliações de v3.14 vs v400 em um conjunto de FENs — esperar divergência
(rede nova), mas verificar que v400 não retorna 0 nem constante.

### 8.5 Teste de acumulador
Verificar que `nnue_rebuild(na, board)` + N rounds de `nnue_push_na` + `nnue_eval`
produz o mesmo resultado que `nnue_rebuild` direto na posição pós-lances.

---

## ORDEM DE IMPLEMENTAÇÃO RECOMENDADA PARA O AGENTE

```
1. train/train_nnue.py
   a. Definir halfkp_features() e king_bucket() em Python
   b. Reescrever encode_chunk() → retorna (X_stm, X_opp)
   c. Atualizar constantes INPUT_MAIN, HIDDEN1, HIDDEN2, CONCAT_DIM
   d. Implementar SCReLU
   e. Reescrever classe NNUE com perspectivas explícitas
   f. Atualizar training loop e dataloader

2. train/export_nnu4.py
   a. Atualizar dimensões e magic "NNU4"
   b. Transpor L1W corretamente para [2560][512]
   c. L2W sem transpose (row-major por output)

3. engine/c/nnue.h
   a. Atualizar constantes
   b. Reescrever NnueAccum (remover ext_*, adicionar bucket_stack)

4. engine/c/nnue.c
   a. Implementar king_bucket(), halfkp_piece_rel(), halfkp_feat()
   b. Reescrever _acc_add_piece e _acc_sub_piece
   c. Reescrever nnue_rebuild
   d. Reescrever nnue_push_na com bucket detection
   e. Reescrever nnue_pop_na com bucket stack
   f. Implementar SCReLU no forward pass (L1 activation)
   g. Atualizar concat de perspectivas (stm + opp → 1024)
   h. Atualizar L2 kernel para NN_L2_IN=1024, NN_L2_OUT=32
   i. Atualizar loader para NNU4
   j. Remover toda lógica de ext_feat, ext_buf, ext_dirty, ext_cache

5. engine/c/board.c
   a. Adicionar wk_sq, bk_sq ao NNMove (opcional, ver Opção A/B)
   b. Preencher nm.wk_sq = b->wk, nm.bk_sq = b->bk antes de nnue_push_na

6. engine/c/main.c
   a. Atualizar version string
   b. Verificar sizeof(NnueAccum) no heap alloc de helpers

7. Testes (seção 8)
```

---

## SUMÁRIO DE CONSTANTES

| Constante       | v3.14     | v400      | Delta |
|-----------------|-----------|-----------|-------|
| NN_L1_IN        | 799       | 2560      | +3×   |
| NN_HM_IN        | 768       | —         | removido |
| NN_EXTRA        | 31        | —         | removido |
| NN_L1_OUT       | 256       | 512       | +2×   |
| NN_L2_IN        | 256       | 1024      | +4×   |
| NN_L2_OUT       | 64        | 32        | -2×   |
| NN_L3_IN        | 64        | 32        | -2×   |
| NN_KING_BUCKETS | —         | 4         | novo  |
| Ativação L1     | ClippedReLU | SCReLU  | mudou |
| Ativação L2     | ClippedReLU | ClippedReLU | igual |
| Pesos (KB)      | ~427      | ~2600     | +6×   |
| NnueAccum (KB)  | ~68       | ~258      | +4×   |
| Magic           | NNU3      | NNU4      | mudou |

---

## NOTAS FINAIS PARA O AGENTE

1. **Não reutilizar pesos existentes.** A mudança de feature encoding é incompatível com
   os pesos atuais. É necessário treinar do zero.

2. **Convenção de coordenadas é o risco principal.** O `sq ^ 56` deve ser idêntico
   entre Python e C. Implementar o teste de paridade (seção 8.2) antes de treinar.

3. **O lazy refresh de perspectiva** (needs_refresh_w/b) é necessário apenas para lances
   de rei que mudam o bucket. Na prática, roque + ~1% dos lances de rei.
   Implementar como rebuild parcial: só percorre as peças e reconstrói o acc_stack_w
   ou acc_stack_b do slot atual, sem tocar o outro.

4. **O L2 kernel AVX2/WASM não precisa reescrita lógica.** Só mudam os loop bounds:
   `NN_L2_IN` vai de 256 para 1024 (mais iterações internas), `NN_L2_OUT` vai de 64
   para 32 (menos iterações externas). O 4-wide unroll continua — só o outer loop tem
   8 iterações (32/4) ao invés de 16 (64/4).

5. **SCReLU tem impacto numérico na quantização.** O OUT_SCALE e os limites de clamp
   precisam ser recalibrados. A fórmula `OUT_SCALE = 320 / (QB * QB)` continua válida
   porque a saída de L3 está na mesma escala independente da ativação de L1.

6. **Manter legacy nnue_push/pop para WASM.** Atualizar com a mesma lógica de bucket
   que nnue_push_na.
```

---

## APÊNDICE A (FASE 1): MULTIPV — Cuidados com a Nova Arquitetura

### Como o Multi-PV funciona hoje

O Multi-PV roda N iterações completas de iterative deepening. Após cada PV encontrar o
melhor lance, esse lance é adicionado a `ss->excluded_root[]`. A próxima iteração do
`alpha_beta` checa `is_excluded_root()` em `ply==0` e pula os lances excluídos.

```
PV1: search all moves → best=e2e4 → add e2e4 to excluded_root[]
PV2: search minus {e2e4} → best=d2d4 → add d2d4 to excluded_root[]
PV3: search minus {e2e4, d2d4} → ...
```

### Impacto do v400 no Multi-PV

**Nenhuma mudança de lógica é necessária.** O Multi-PV opera no `search.c` e usa
`excluded_root[]` no nível de lance — completamente independente da arquitetura NNUE.

**O único ponto de atenção:** o TT cutoff em `ply==0` já está desabilitado quando
`excluded_root_n > 0`. Isso previne que o score da PV1 seja retornado como cutoff para
a PV2. Esse comportamento deve ser **mantido** — não alterar essa lógica.

### Tempo por PV com o novo modelo

O modelo v400 é mais lento por avaliação (L1 maior, L2 mais larga em inputs). Para
Multi-PV, o time budget é dividido entre as N linhas (já implementado: cada PV reseta
`ss->deadline_ms`). Com o engine mais forte, o Multi-PV vai ser usado tipicamente em
profundidades menores — o tradeoff é aceitável.

**Verificação obrigatória após implementação:** rodar `go movetime 1000 multipv 3` em
posição aberta e confirmar que as 3 linhas são distintas e a terceira tem `depth >= 6`.

---

## APÊNDICE B (FASE 1, base p/ FASE 0): MULTITHREAD / LAZY SMP — Cuidados Específicos

### O que muda com NnueAccum maior

A `NnueAccum` cresce de ~68 KB para ~258 KB. Em `helper_thread_fn`:

```c
/* ATUAL: */
NnueAccum *my_nnue = (NnueAccum *)zmalloc32(sizeof(NnueAccum));
memset(my_nnue, 0, sizeof(NnueAccum));
```

**Esse código não muda** — usa `sizeof(NnueAccum)` dinamicamente. Mas há dois cuidados:

**1. Stack size dos helpers:** O stack de 8 MB já alocado com `pthread_attr_setstacksize`
é suficiente para a NnueAccum no heap. A NnueAccum é alocada no heap via `zmalloc32`,
não na stack, então não há risco de stack overflow.

**2. Inicialização dos bucket_stack:** O `memset(my_nnue, 0, sizeof(NnueAccum))` zera
todos os campos incluindo `bucket_w`, `bucket_b`, e os `bucket_stack_w/b`. Isso é
correto porque `nnue_rebuild` vai recalcular os buckets quando chamado em `search_best`.
**Nenhuma inicialização especial necessária.**

**3. `zmalloc32` em main.c:** A função local `zmalloc32` em `main.c` (cópia da de
`nnue.c`) aloca com `posix_memalign(..., 32, bytes)`. Com `bytes = sizeof(NnueAccum) ≈
258 KB`, isso é seguro — não há limite prático para `posix_memalign` nesse tamanho.

### Thread safety do bucket state

Os campos `bucket_w`, `bucket_b`, `needs_refresh_w/b`, e `bucket_stack_w/b` são todos
**dentro de `NnueAccum`**, que é privado por thread. Não há compartilhamento nem race
condition. O mesmo isolamento que existe hoje para `acc_stack_w/b` se aplica.

### Acumulador correto em cada thread após `board_make` de helper

Cada helper thread recebe uma **cópia privada do Board** (`g_helper_args[i].board =
sta->board`). Depois, em `search_best`, chama `nnue_rebuild(b->nnue, b->b)` para
inicializar o acumulador da posição raiz, **incluindo os buckets**. O `nnue_rebuild`
deve:
1. Escanear o board para encontrar `wk_sq` e `bk_sq`
2. Calcular `na->bucket_w = king_bucket(wk_sq)` e `na->bucket_b = king_bucket(bk_sq^56)`
3. Inicializar `bucket_stack_w[0] = na->bucket_w`, `bucket_stack_b[0] = na->bucket_b`
4. Zerar `needs_refresh_w = needs_refresh_b = 0`

**Crítico:** se `nnue_rebuild` não inicializar os `bucket_stack[0]`, o primeiro `pop_na`
após o primeiro `push_na` vai restaurar `bucket_w = 0` incorretamente.

### O `nnue_reset` e o ciclo de vida por busca

O ciclo em `search_best`:
```c
if (nnue_ready()) nnue_rebuild(b->nnue, b->b);
```

Essa linha já substitui a função de `nnue_reset` (que apenas marcava `acc_dirty=1`). Em
v400, `nnue_rebuild` deve também resetar `needs_refresh_w/b = 0` e preencher o
`bucket_stack[0]`. **Manter esse padrão** — não chamar `nnue_reset` antes do rebuild.

### WASM: g_nnue_accum como variável global

No WASM, `g_nnue_accum` é uma variável estática global de ~258 KB. O módulo WASM tem
heap configurável — verificar que o `TOTAL_MEMORY` do Emscripten acomoda o aumento
(~190 KB a mais vs v3.14). Tipicamente o WASM já tem 64+ MB de heap configurado, então
isso não é problema prático.

O legado `nnue_push`/`nnue_pop` (para WASM) deve ser atualizado com a mesma lógica de
`bucket_stack` que `nnue_push_na`/`nnue_pop_na`.

---

## APÊNDICE C (FASE 1): TABLEBASES (SYZYGY) — Integração com v400

### O que o TB probe usa da NNUE

**Nada diretamente.** O TB probe (`syzygy_probe_wdl`) não chama `nnue_eval`. Os dois
sistemas são independentes.

**Onde há interação indireta:**

**1. `board_make`/`board_unmake` dentro do root TB filtering:**
O root TB filtering (atualmente desabilitado, mas documentado como TODO) faz
`board_make` + `syzygy_probe_wdl` + `board_unmake` para cada lance raiz. Isso dispara
`nnue_push_na`/`nnue_pop_na`. Com v400, `nnue_push_na` agora detecta mudança de bucket.
Esse comportamento é correto — a acumulação está certa para cada posição filho, e o pop
restaura o estado anterior. **Nenhuma mudança necessária.**

**2. In-tree WDL probe em `alpha_beta`:**
Após um TB cutoff (`return tb_score`), o `nnue_eval_bb` não é chamado para esse nó.
O acumulador de v400 está no estado correto (foi atualizado pelo `nnue_push_na` do
`board_make` pai). O `nnue_pop_na` no `board_unmake` do pai vai restaurar corretamente.
**Nenhuma mudança necessária.**

**3. `needs_refresh` e TB probe:**
Se um lance de rei muda o bucket E um TB cutoff ocorre antes do `nnue_eval_bb` ser
chamado, o `needs_refresh_w/b` fica `1` mas nunca é consumido. No `nnue_pop_na` seguinte,
o bucket é restaurado via `bucket_stack[src]` — o `needs_refresh` residual não causa
problema porque o eval desse nó foi substituído pelo TB score.

**Garantia formal:** `needs_refresh` só é lido em `nnue_eval_bb`. Se o eval não é
chamado (TB cutoff), o flag fica sujo mas é irrelevante. No pop, o bucket volta ao valor
pré-lance via `bucket_stack`. O próximo push vai recalcular `needs_refresh` do zero.

### TT poisoning com scores de NNUE vs TB

O mecanismo atual de TT caching de TB scores (depth=127 para wins/losses) é
**independente da qualidade da NNUE**. Com v400, os TB scores continuam sendo
armazenados com `TT_LOWER`/`TT_UPPER` e depth alto. Os scores NNUE melhorados vão
produzir melhores ordenações de lances e mais cortes alpha-beta **antes** do TB probe,
reduzindo o número de probes necessários. Efeito líquido: mais eficiência, não conflito.

### Root TB filtering (quando implementado)

Quando o root filtering via `tb_probe_root_dtz()` for implementado (TODO existente), a
interação com v400 é:
- O `excluded_root[]` é preenchido por WDL relativo ao root
- O `nnue_eval_bb` de v400 é chamado nos nós sobreviventes
- Os buckets dos reis na posição raiz estão corretos após `nnue_rebuild`

**Nenhuma mudança de interface necessária para implementar o root TB filtering.**

---

## APÊNDICE D: CONVENÇÕES FIXAS DE WDL E CP

Este apêndice deve ser lido ANTES de modificar qualquer código de treino ou inferência.
Violar qualquer dessas convenções produz um engine que joga ao contrário.

### D.1 Perspectiva do Label no Treino

```
LABEL é WHITE-RELATIVE:
  wdl = 1.0  → Brancas vencem
  wdl = 0.5  → Empate
  wdl = 0.0  → Pretas vencem

APÓS encode_chunk, o label é convertido para STM-RELATIVE:
  is_black = (fen.split(' ')[1] == 'b')
  y[is_black] = 1.0 - y[is_black]
  → para posições de preto: 1.0 = pretas vencem (STM vence)

O MODELO RECEBE e APRENDE:
  input:  features na perspectiva do STM (x_stm) e do oponente (x_opp)
  output: sigmoid(L3) ∈ [0, 1]  =  probabilidade de vitória do STM
  loss:   BCELoss(output, y_stm_relative)

NÃO usar white-relative label diretamente como target do modelo.
NÃO usar cp labels sem converter para WDL via to_wdl.
```

### D.2 Perspectiva do Output no Conversor

```python
# to_wdl converte cp (white-relative) para WDL (white-relative):
def to_wdl(values, col):
    if col == 'wdl': return values.astype(np.float16)
    return (1.0 / (1.0 + np.exp(-values / 320.0))).astype(np.float16)
    # cp > 0 (white ahead) → WDL > 0.5 ✓

# SEGUIDO DE:
y_base[is_black] = 1.0 - y_base[is_black]
# Inverte para STM-relative ✓
```

O `320` no `to_wdl` é a **temperatura de conversão** que alinha o espaço de probabilidade
com o espaço de centipawns. Um CP de +320 corresponde a ~75% de vitória para as brancas.
**Esse valor deve ser idêntico** ao usado no step 6 do forward pass em C:
`cp = l3_sum * OUT_SCALE + L3B * 320.0f`.

### D.3 Perspectiva do Output em C

```c
/* eval_stm() em search.c: */
int eval_stm(Board *b) {
    return nnue_eval_bb(b->nnue, b->turn == COL_W ? 0 : 1, ...);
    // stm=0 → perspectiva branca como primária
    // stm=1 → perspectiva preta como primária
}

/* Em nnue_eval_bb com HalfKP v400: */
// relu1[0..511]   = SCReLU(acc_stack_w[ptr] + bias)  ← perspectiva STM
// relu1[512..1023] = SCReLU(acc_stack_b[ptr] + bias)  ← perspectiva opp
// concat → L2 → L3
// resultado = cp POSITIVO → bom para STM (quem move)

/* search.c usa eval_stm() sempre em negamax:
   score = -alpha_beta(...)
   eval_stm() retorna cp do ponto de vista de quem move ✓
*/
```

**INVARIANTE:** `nnue_eval_bb` retorna positivo quando a posição é boa para o jogador
cujo `stm` foi passado. Se `stm=0` (brancas), positivo = brancas estão ganhando.
Se `stm=1` (pretas), positivo = pretas estão ganhando. O negamax cuida da negação.

### D.4 Perspectiva das Features por Perspectiva no Treino

```python
# encode_chunk retorna DOIS tensores:
x_stm = halfkp_features(board, pov_is_white=stm_is_white)
x_opp = halfkp_features(board, pov_is_white=not stm_is_white)

# halfkp_features(board, pov_is_white=True):
#   - bucket calculado sobre b->wk (rei branco)
#   - aliado = peça branca → color_offset=0
#   - inimigo = peça preta  → color_offset=5
#   - sq_pov = sq  (coordenadas python-chess, 0=a1)

# halfkp_features(board, pov_is_white=False):
#   - bucket calculado sobre bk ^ 56 (rei preto espelhado)
#   - aliado = peça preta  → color_offset=0
#   - inimigo = peça branca → color_offset=5
#   - sq_pov = sq ^ 56  (espelho vertical)

# MODELO:
# model.forward(x_stm, x_opp):
#   h_stm = SCReLU(L1(x_stm))   # perspectiva de quem move
#   h_opp = SCReLU(L1(x_opp))   # perspectiva do oponente
#   h = concat([h_stm, h_opp])  # STM SEMPRE PRIMEIRO
#   → L2 → L3 → sigmoid
```

**INVARIANTE CRÍTICA:** STM sempre ocupa `h[0..511]`, oponente ocupa `h[512..1023]`.
Em C, isso deve ser refletido na ordem do concat no forward pass:
```c
// Step 2 em nnue_eval_bb:
int stm_is_w = (stm == 0);
int16_t *acc_stm = stm_is_w ? na->acc_stack_w[ptr] : na->acc_stack_b[ptr];
int16_t *acc_opp = stm_is_w ? na->acc_stack_b[ptr] : na->acc_stack_w[ptr];
// relu1[0..511] = SCReLU(acc_stm + bias)
// relu1[512..1023] = SCReLU(acc_opp + bias)
```

**ERRO COMUM:** Passar `(acc_opp, acc_stm)` ao invés de `(acc_stm, acc_opp)` faz o
engine jogar com a perspectiva invertida — vai avaliar posições vencedoras como perdidas.

---

## APÊNDICE E (FASE 1): QUANTIZAÇÃO — Guia Completo v400

### E.1 Cadeia de Quantização Completa

```
TREINAMENTO (float32 com fake_quant):

Entrada:
  x_stm, x_opp ∈ {0, 1}^2560  (binário: feature ativa ou não)

L1 forward:
  acc = L1W_float @ x + L1B_float        # float32
  QAT: L1W fake_quant → int16 range [-32767/QA, +32767/QA]
       L1B fake_quant → int32 range [-2^31/QA, +2^31/QA]  (na prática: [-128, 128])

Ativação SCReLU:
  c = clamp(acc, 0, 1)                   # float ∈ [0, 1]
  h1 = c * c                             # float ∈ [0, 1]
  fake_quant: h1_q = round(h1 * QA_EFF).clamp(0, QA_EFF) / QA_EFF
              QA_EFF = 254  (= 255² >> 8)
  STE: h1 = h1 + (h1_q - h1).detach()

L2 forward:
  acc2 = L2W_float @ h1 + L2B_float      # float32
  QAT: L2W fake_quant → int8 range [-127/QB, +127/QB]
       L2B fake_quant → int32 range (scale = QA_EFF * QB = 254 * 64 = 16256)

Ativação ClippedReLU (L2):
  h2 = clamp(acc2, 0, 1)                # float ∈ [0, 1]
  fake_quant: h2_q = round(h2 * QB).clamp(0, QB) / QB
  STE: h2 = h2 + (h2_q - h2).detach()

L3 forward:
  logit = L3W_float @ h2 + L3B_float    # float32
  QAT: L3W fake_quant → int8 range [-127/QB, +127/QB]
       L3B: NÃO quantizado (float32 no arquivo)

Saída:
  out = sigmoid(logit) ∈ [0, 1]         # WDL STM-relative
  loss = BCELoss(out, y_stm)
```

### E.2 Conversão para Inteiros (conversor NNU4)

```
L1W:
  L1W_q = round(L1W_float * QA).clip(-32767, 32767).astype(int16)
  # shape: [L1_OUT=512, L1_IN=2560] → transpor para [L1_IN=2560, L1_OUT=512]
  L1W_T = L1W_q.T.ascontiguousarray()   # escrita no arquivo

L1B:
  L1B_q = round(L1B_float * QA).astype(int32)
  # Bias é multiplicado por QA porque acc = sum(x_binary * L1W_q) + L1B_q
  # x é binário {0,1}, L1W_q está em escala QA → acc em escala QA

L2W:
  L2W_q = round(L2W_float * QB).clip(-127, 127).astype(int8)
  # shape: [L2_OUT=32, L2_IN=1024], row-major por output (SEM transpose)
  # NÃO transpor: o kernel maddubs lê [out][in] em memória

L2B:
  L2B_q = round(L2B_float * QA_EFF * QB).astype(int32)
  # Scale: QA_EFF * QB = 254 * 64 = 16256
  # porque relu1 (uint8 após SCReLU) ∈ [0, QA_EFF=254]
  #          L2W_q ∈ [-QB, QB]
  #          produto ∈ [-QA_EFF*QB, +QA_EFF*QB]
  # ALTERNATIVA ACEITÁVEL: usar QA * QB = 255 * 64 = 16320 (erro 0.39%)

L3W:
  L3W_q = round(L3W_float * QB).clip(-127, 127).astype(int8)

L3B:
  L3B_f = float(L3B_float)   # NÃO quantizar — escrito como float32

OUT_SCALE:
  OUT_SCALE = 320.0 / (QB * QB)   # = 320 / 4096 = 0.078125
  # Fórmula em C: cp = l3_sum_int * OUT_SCALE + L3B_f * 320.0
  # l3_sum_int escala: relu2 ∈ [0,QB] × L3W_q ∈ [-QB,QB] → QB²=4096 por elemento
```

### E.3 Forward Pass Inteiro em C (v400)

```c
/* Step 1: Acumulador já pronto (incremental ou rebuild) */
/* acc_stm[512], acc_opp[512] em int16 */

/* Step 2: SCReLU → relu1[1024] uint8 */
/* Para cada o em [0, 511]: */
/*   sum_stm = (int32)acc_stm[o] + _nnL1B[o]         (int32) */
/*   c_stm   = clamp(sum_stm, 0, 255)                 (uint8) */
/*   relu1[o] = (c_stm * c_stm) >> 8                  (uint8, ∈ [0,254]) */
/* Para cada o em [0, 511]: */
/*   sum_opp = (int32)acc_opp[o] + _nnL1B[o]          (int32) */
/*   c_opp   = clamp(sum_opp, 0, 255)                  (uint8) */
/*   relu1[512+o] = (c_opp * c_opp) >> 8               (uint8, ∈ [0,254]) */

/* Step 3: L2 — maddubs kernel, input=1024, output=32 */
/* _nnL2W: [32][1024] int8, _nnL2B: [32] int32 */
/* acc2[o] = _nnL2B[o] + sum_i(relu1[i] * _nnL2W[o][i])  (int32) */
/* 4-wide unroll: outer loop o = 0,4,8,...,28 (8 iterações, vs 16 no v3.14) */
/* inner loop i = 0,32,64,...,1024-32 (32 iterações, vs 8 no v3.14) */

/* Step 4: shift + ClippedReLU → relu2[32] uint8 */
/* relu2[o] = clamp(acc2[o] >> NN_SHIFT, 0, NN_QB)     (uint8, ∈ [0,64]) */

/* Step 5: L3 dot product */
/* l3_sum = sum_i(_nnL3W[i] * relu2[i])                (int32) */

/* Step 6: escala → centipawns */
/* cp = (float)l3_sum * _nnOutScale + _nnL3B * 320.0f  */
/* cp = clamp(cp, -2000.0f, 2000.0f)                   */
/* return (int)cp                                       */
```

### E.4 SCReLU AVX2 — Implementação do Step 2

```c
/* SCReLU para uma perspectiva (512 valores int16 → 512 valores uint8) */
/* Escreve em relu1_ptr (ponteiro para parte stm ou opp do buffer relu1[1024]) */

static void _screlu_perspective_avx2(
    const int16_t *acc,    /* acc_stm ou acc_opp, 512 int16 */
    uint8_t *relu1_ptr,    /* destino, 512 uint8 */
    const int32_t *bias    /* _nnL1B, 512 int32 */
) {
#ifdef __AVX2__
    /* Processar 16 int16 por iteração → 16 uint8 de saída */
    /* = 512 / 16 = 32 iterações por perspectiva */
    for (int o = 0; o < 512; o += 16) {
        /* Carregar 16 int16 do acumulador */
        __m128i a_lo = _mm_load_si128((const __m128i*)(acc + o));
        __m128i a_hi = _mm_load_si128((const __m128i*)(acc + o + 8));

        /* Widen para int32 e adicionar bias */
        __m256i s_lo = _mm256_add_epi32(
            _mm256_cvtepi16_epi32(a_lo),
            _mm256_load_si256((const __m256i*)(bias + o)));
        __m256i s_hi = _mm256_add_epi32(
            _mm256_cvtepi16_epi32(a_hi),
            _mm256_load_si256((const __m256i*)(bias + o + 8)));

        /* Clamp para [0, 255] → c ∈ [0, 255] uint8 */
        __m256i v0   = _mm256_setzero_si256();
        __m256i v255 = _mm256_set1_epi32(255);
        s_lo = _mm256_min_epi32(_mm256_max_epi32(s_lo, v0), v255);
        s_hi = _mm256_min_epi32(_mm256_max_epi32(s_hi, v0), v255);

        /* Narrow int32 → uint8 via packus */
        __m256i p16  = _mm256_packus_epi32(s_lo, s_hi);
        p16 = _mm256_permute4x64_epi64(p16, _MM_SHUFFLE(3,1,2,0));
        __m128i c8   = _mm_packus_epi16(
            _mm256_castsi256_si128(p16),
            _mm256_extracti128_si256(p16, 1));

        /* SCReLU: c * c >> 8 */
        /* Widen c (uint8) para uint16 para não overflow na multiplicação */
        __m128i c8_lo_half = _mm_unpacklo_epi8(c8, _mm_setzero_si128()); /* 8 uint16 */
        __m128i c8_hi_half = _mm_unpackhi_epi8(c8, _mm_setzero_si128()); /* 8 uint16 */
        __m128i sq_lo = _mm_mullo_epi16(c8_lo_half, c8_lo_half);  /* c² ∈ [0, 65025] uint16 */
        __m128i sq_hi = _mm_mullo_epi16(c8_hi_half, c8_hi_half);
        /* >> 8: shift right 8 bits, então narrow para uint8 */
        sq_lo = _mm_srli_epi16(sq_lo, 8);  /* ∈ [0, 254] */
        sq_hi = _mm_srli_epi16(sq_hi, 8);
        __m128i result = _mm_packus_epi16(sq_lo, sq_hi);  /* 16 uint8 ∈ [0, 254] */
        _mm_store_si128((__m128i*)(relu1_ptr + o), result);
    }
#endif
}
```

### E.5 SCReLU WASM SIMD128 — Step 2

```c
#elif defined(__wasm_simd128__)
    /* WASM: 128-bit = 8 int16 por vez, processar 512/8=64 iterações */
    for (int o = 0; o < 512; o += 8) {
        v128_t a = wasm_v128_load(acc + o);
        /* Widen low e high 4 int16 para int32 */
        v128_t a_lo32 = wasm_i32x4_extend_low_i16x8(a);
        v128_t a_hi32 = wasm_i32x4_extend_high_i16x8(a);
        v128_t b_lo32 = wasm_v128_load(bias + o);
        v128_t b_hi32 = wasm_v128_load(bias + o + 4);
        v128_t s_lo = wasm_i32x4_add(a_lo32, b_lo32);
        v128_t s_hi = wasm_i32x4_add(a_hi32, b_hi32);
        /* Clamp [0, 255] */
        s_lo = wasm_i32x4_min(wasm_i32x4_max(s_lo, wasm_i32x4_splat(0)),
                               wasm_i32x4_splat(255));
        s_hi = wasm_i32x4_min(wasm_i32x4_max(s_hi, wasm_i32x4_splat(0)),
                               wasm_i32x4_splat(255));
        /* Narrow i32→i16, depois i16→u8 */
        v128_t packed16 = wasm_i16x8_narrow_i32x4(s_lo, s_hi);  /* 8 int16, valores [0,255] */
        /* packed16 tem os valores c ∈ [0,255] como int16 */
        /* SCReLU: c*c >> 8 */
        v128_t c_lo = wasm_u32x4_extend_low_u16x8(packed16);   /* 4 uint32 */
        v128_t c_hi = wasm_u32x4_extend_high_u16x8(packed16);  /* 4 uint32 */
        /* Na WASM não há mul_epi16 eficiente. Usar i16x8_mul: */
        v128_t sq = wasm_i16x8_mul(packed16, packed16);  /* c² mod 65536 ∈ [0,65025] */
        v128_t sq_shifted = wasm_u16x8_shr(sq, 8);       /* c² >> 8 ∈ [0, 254] */
        /* Narrow u16→u8: os valores já estão em [0,254] que cabe em u8 */
        v128_t result8 = wasm_u8x16_narrow_i16x8(sq_shifted, sq_shifted);
        wasm_v128_store64_lane(relu1_ptr + o, result8, 0);  /* escreve 8 bytes */
    }
```

### E.6 Weight Clamping no Treino

```python
def clamp_weights_(model: NNUE) -> None:
    with torch.no_grad():
        lim1 = 32767.0 / QA           # L1W: int16 range → ±128.1
        model.l1.weight.clamp_(-lim1, lim1)
        model.l1.bias.clamp_(-lim1, lim1)   # L1B: mesma escala que L1W

        lim2 = 127.0 / QB             # L2W, L3W: int8 range → ±1.984
        model.l2.weight.clamp_(-lim2, lim2)
        model.l3.weight.clamp_(-lim2, lim2)
        # L2B e L3B: NÃO clampar (int32 e float têm range suficiente)

# CHAMAR a cada N batches (manter o ciclo atual de clamp_weights_)
# Clamping previne overflow no fake_quant durante backprop
```

### E.7 Overflow Check no Conversor

```python
def _quant16(arr, scale, name):
    q = np.round(arr * scale)
    n_ov = int(np.sum(np.abs(q) > 32767))
    if n_ov > 0:
        print(f"WARNING: {n_ov} valores {name} overflow int16 — serão clampados")
        print(f"  max_abs={np.abs(arr).max():.4f}, limit={32767/scale:.4f}")
    return np.clip(q, -32767, 32767).astype(np.int16)

def _quant8(arr, scale, name):
    q = np.round(arr * scale)
    n_ov = int(np.sum(np.abs(q) > 127))
    if n_ov > 0:
        print(f"WARNING: {n_ov} valores {name} overflow int8 — serão clampados")
        print(f"  max_abs={np.abs(arr).max():.4f}, limit={127/scale:.4f}")
    return np.clip(q, -127, 127).astype(np.int8)
```

Se `n_ov > 0` no L1W após treinamento extenso, aumentar o `clamp_weights_` ou reduzir
a learning rate nas últimas épocas.

### E.8 Tabela de Resumo de Constantes de Quantização

| Tensor | Tipo    | Scale             | Range int  | Nota |
|--------|---------|-------------------|------------|------|
| L1W    | int16   | QA = 255          | ±32767     | Transposto [2560][512] no arquivo |
| L1B    | int32   | QA = 255          | ±2^31      | Bias do acumulador |
| relu1  | uint8   | QA_EFF = 254      | [0, 254]   | Após SCReLU: c²>>8 |
| L2W    | int8    | QB = 64           | ±127       | Row-major [32][1024], sem transpose |
| L2B    | int32   | QA_EFF × QB = 16256 | ±2^31    | Aceita QA×QB=16320 (erro 0.39%) |
| relu2  | uint8   | QB = 64           | [0, 64]    | Após ClippedReLU + >>SHIFT |
| L3W    | int8    | QB = 64           | ±127       | |
| L3B    | float32 | 1.0 (não quant.)  | float      | Multiplicado por 320.0 em C |
| OUT_SCALE | float32 | — | 0.078125   | = 320 / QB² = 320 / 4096 |


---

## APÊNDICE F (FASES 0, 2, 3, 4, 5): IDEIAS DO ZQUORIDOR PARA v400 — ENGINE, TREINO, SELFPLAY E ARENA

Levantamento cruzado com o motor zquoridor (C++, negamax+NNUE+CAT). A maior
parte do zquoridor não se aplica ao zchezz (regras de Quoridor, CAT de
corredor, endgame_race — tudo específico do jogo). O que segue é só o que
tem análogo direto e vale a pena adotar. Ordenado por onde entra no plano
existente: primeiro a mudança estrutural que destrava tudo o mais (F.1),
depois engine/treino/selfplay/arena.

### F.0 Diferença de arquitetura que importa aqui

zquoridor é uma biblioteca C++ (`Negamax` é uma classe, `tt` é
`std::vector<TTEntry>` **membro da instância**) usada tanto pelo motor UCI
quanto por `tools/selfplay` e `tools/arena`, todos no mesmo processo,
multi-thread com `std::thread`, cada thread dona de sua(s) própria(s)
instância(s) de `Negamax`.

zchezz é C puro, e a TT é hoje um conjunto de **arrays globais do processo**
(`TT_H[]`, `TT_S[]`, `TT_D[]`, `TT_G[]`, `TT_M[]`, `TT_E[]`, `TT_GEN` em
`search.c`). Isso é perfeito para o caso de uso atual — Lazy SMP dentro de
UM processo UCI, uma busca de cada vez, helpers como threads compartilhando
a mesma TT globalmente. Mas significa que **não é possível ter dois jogos
sendo jogados ao mesmo tempo no mesmo processo** (duas buscas independentes
colidiriam na mesma TT), o que é exatamente o modelo que zquoridor usa em
`tools/selfplay` (N threads, cada uma jogando partidas sequenciais, TT por
thread) e que motiva a proposta em F.2.

Hoje o zchezz contorna isso jogando `CONCURRENCY=16` **processos** UCI
separados via `subprocess` (`tests/run_selfplay.py`, `tests/run_tournament.py`) —
funciona, mas paga overhead de processo (16 processos zchezz.exe residentes)
e overhead de protocolo UCI (parsing de texto por lance, mesmo em
`movetime=100ms`). Isso é aceitável para os torneios de ELO (que já
funcionam bem — ver F.4) mas é caro demais para gerar milhões de posições de
treino.

### F.1 (FASE 0) Engine — TTable por instância (pré-requisito para tudo abaixo)

**Prioridade: fazer isto ANTES de F.2.** Sem isto, F.2 não é possível.

Empacotar as 6 arrays globais + `TT_GEN` num struct `TTable` alocado
dinamicamente:

```c
// search.h
typedef struct {
    uint64_t *H;
    int32_t  *S;
    int32_t  *D;
    uint16_t *G;
    int32_t  *M;
    int32_t  *E;
    uint16_t gen;
    size_t   size;     /* TT_SIZE atual, pode variar (nativo vs WASM) */
    size_t   mask;
} TTable;

TTable *tt_create(size_t n_entries);
void    tt_destroy(TTable *tt);
void    tt_clear(TTable *tt);          /* memset físico — usar entre partidas nativas */
void    tt_new_generation(TTable *tt); /* bump lógico — usar dentro da mesma partida (equivalente ao ucinewgame atual) */
```

`tt_probe`/`tt_store` (hoje `static` em `search.c`, leem os globais direto)
passam a receber `TTable *tt` como primeiro argumento. `SearchState` ganha
um campo `TTable *tt` (helpers de Lazy SMP continuam recebendo o MESMO
ponteiro — nada muda no comportamento de busca única). O modo UCI padrão
continua alocando **uma** `TTable` global no `main()` do jeito que é hoje
(zero mudança de comportamento/performance ali — é só indireção de
ponteiro, o compilador deve conseguir manter em registrador dentro do loop
de busca já que `tt` não muda durante uma `alpha_beta`).

Isso é mecânico mas tem bastante superfície (toda chamada de `tt_probe`/
`tt_store`/`tt_score_store`/`tt_score_read` dentro de `qsearch`/
`alpha_beta`/`search_best`), então entra como sua própria PR antes de
qualquer coisa de F.2/F.3 — com o mesmo teste de paridade que o resto do
plano já usa (bench de nodes/s antes/depois deve bater igual, já que TT_GEN
segue exatamente a mesma semântica, só trocando array global por ponteiro).

Ganho secundário, não o motivo principal: também abre caminho para rodar
`arena.c` (F.4) A/B testando duas versões do motor no mesmo processo, cada
uma com sua `TTable`, sem risco de uma contaminar a outra — o problema que
o comentário-cabeçalho de `tools/arena/arena.cpp` descreve ter acontecido
de fato no zquoridor (torneio "fantasma" rodando a mesma engine dos dois
lados por engano). No zchezz esse risco específico não existe hoje porque
`tests/run_tournament.py` já aponta para dois `.exe` de verdade — mas vale
como validação: F.4 precisa continuar garantindo isso.

### F.2 (FASE 3) Selfplay nativo (novo: `tools/selfplay.c`)

Ideia central pedida: um gerador de dados de treino **em processo**,
análogo a `tools/selfplay/selfplay.hpp` do zquoridor, para substituir (só
para geração de dataset — não para os torneios de ELO, ver nota no final)
o caminho atual `tests/run_selfplay.py` → PGN/EPD → `label_*.py` →
parquet.

**Atualização**: a v1 deste apêndice assumia que o selfplay nativo seguiria
alimentando o pipeline rotulado por Stockfish (`label_*`). Isso mudou —
ver F.3: o objetivo agora é treino por bootstrapping, sem oráculo externo.
`eval_cp`/`game_result` abaixo são, portanto, 100% auto-gerados (a própria
NNUE + resultado real da partida), não mais um complemento ao rótulo de
Stockfish. O que não muda é o formato: sai um `.bin` compacto em vez de
PGN/EPD, do mesmo jeito que zquoridor grava `TrainingSample`.

**Por que o pipeline atual é lento comparado ao do zquoridor** (motivação
direta deste item — não é só "usar .bin em vez de PGN", é remover cada
etapa intermediária que hoje existe só por causa do protocolo UCI):

| Etapa | Pipeline atual (`tests/run_selfplay.py` → `label_*` → `train_nnue.py`) | zquoridor (`tools/selfplay` → `train_nnue.py`) | Proposto aqui (F.2) |
|---|---|---|---|
| Concorrência | N **processos** UCI via `subprocess` (`CONCURRENCY=16`) | N **threads** num só processo | N threads num só processo |
| Por lance | round-trip de texto UCI (`position ... \n go movetime ...` → parse de `bestmove`) | chamada de função direta (`Negamax::search`) | chamada de função direta (`search_best`) |
| Formato de partida | PGN completo (`chess.pgn`), depois reparseado | nenhum — grava `TrainingSample` direto no fim da partida | nenhum — grava `SelfplaySample` direto |
| Rotulagem | processo **separado**, depois do selfplay: `label_*.py` chama Stockfish (`SF_NODES=1_000_000`) por posição, em lote | nenhuma — `eval_cp` já sai da própria NNUE durante a busca de selfplay | idem zquoridor — `eval_cp` sai da busca, sem etapa separada |
| Formato em disco | Parquet (`chunk_NNNN.parquet`, via pandas) | binário packed, `numpy.fromfile` direto | binário packed, `numpy.fromfile` direto |
| Leitura no treino | pandas → conversão para tensores | `numpy.fromfile(dtype=SAMPLE_DTYPE)` já estruturado | idem zquoridor |

As duas maiores fontes de lentidão do pipeline atual — processo por jogo
(em vez de thread) e a etapa de rotulagem Stockfish como passo **separado**
depois do jogo terminar — são exatamente as duas que desaparecem com F.2 +
F.3: F.2 tira o subprocess/UCI/PGN/parquet do meio, F.3 tira o Stockfish do
loop (o `eval_cp` sai de graça, já dentro da mesma chamada de busca que
escolhe o lance — não é um passo a mais, é reaproveitar um número que a
busca já calculou). O que sobra depois das duas mudanças é estruturalmente
o mesmo pipeline do zquoridor: thread → busca → grava struct → repete,
sem nenhum processo/parse/arquivo intermediário entre a partida terminar e
a amostra virar bytes em disco.

**Formato de amostra proposto** (`selfplay_sample.h`, packed, análogo ao
`TrainingSample` de `selfplay.hpp`):

```c
#pragma pack(push, 1)
typedef struct {
    uint64_t pieces[2];      /* bitboards brancas/pretas, ou 12 planos — decidir junto do formato de treino atual */
    uint8_t  meta;           /* castling rights + ep file + stm, empacotado */
    int16_t  eval_cp;        /* eval própria do zchezz na hora da jogada, STM-relative */
    int8_t   game_result;    /* +1/0/-1 do ponto de vista de quem jogou esta posição */
    uint16_t move_played;    /* lance jogado, empacotado (mesmo pack_move de search.c) */
} SelfplaySample;
#pragma pack(pop)
```

(layout exato de `pieces`/`meta` deve seguir o que `train_nnue.py` já espera
como entrada mais barata de gerar a partir do FEN — checar `encode_chunk`
antes de fechar o layout, para não inventar um terceiro formato).

**Threading e uso de TT (a pergunta específica sobre "uso compartilhado de
TT")**: depende de F.1. Cada worker thread (pool de N, `N = --threads`,
default = núcleos) roda um laço:

```c
for (;;) {
    int g = atomic_fetch_add(&next_game, 1);
    if (g >= total_games) break;

    tt_clear(my_tt);          /* física — não física+lógica: início de partida nova, não custa reaproveitar zeros */
    play_one_game(board, my_tt, ss, cfg, out_buf);
    fwrite_locked(out_buf, ...);
}
```

Igual ao zquoridor, o ponto relevante é: **dentro de uma partida de
self-play, as duas cores usam a MESMA `TTable`** (`shared_tt=true` por
default, flag `--separate-tt` para desligar) — branco e preto são o MESMO
motor com os MESMOS pesos, então não há problema de correção em
compartilhar (ao contrário de um torneio A vs B de verdade, onde cada lado
precisa da sua própria TT — isso é o que `tools/arena` do zquoridor faz
diferente de `tools/selfplay`). Isso economiza metade da memória de TT por
thread (relevante quando se quer rodar 16-32 threads em paralelo, cada uma
com uma TT de alguns milhões de entradas) e, de brinde, cada lado enxerga
as entradas que o lado oposto já deixou na TT na mesma partida — mais
transposições batem, busca marginalmente mais rápida por partida.

`tt_clear` (física, não `tt_new_generation`) entre partidas é obrigatório
pelo mesmo motivo que o comentário de `selfplay.hpp` explica: scores de
repetição são dependentes do histórico da partida (o hash de Zobrist não
carrega "quantas vezes essa posição já apareceu NESTA partida"), então uma
entrada de "empate por repetição" da partida G1 pode ser lida errado como
score real na partida G2 se cair na mesma posição sem repetição. **Nota
tranquilizadora**: o motor UCI atual já evita esse mesmo problema
corretamente — `tt_probe` (search.c:239) descarta o *score* de qualquer
entrada de geração diferente (`TT_G[idx] != TT_GEN`), só reaproveita o
lance para ordenação, e `tests/run_selfplay.py`/`tests/run_tournament.py` já mandam
`ucinewgame` antes de cada partida — mas isso é geração *lógica* (bump de
contador), enquanto o problema de repetição intra-processo do selfplay
nativo precisa de limpeza *física* porque as duas cores jogam a MESMA
partida simultaneamente na MESMA TT (não há "geração anterior", é a
posição atual mesmo). Usar `tt_new_generation` sozinho aqui não resolveria
nada.

Sem UCI/stdio no meio, sem processo por jogo, sem parsing de texto por
lance — o ganho esperado é principalmente em taxa de partidas/segundo em
`movetime` curto (onde o overhead de protocolo/processo é proporcionalmente
maior), não em ELO.

**O que NÃO muda**: `tests/run_tournament.py`/`run_tournament_quick.py` continuam
sendo o caminho para medir ELO entre versões (PGN, EPD, compatibilidade com
GUIs externas, os dois `.exe` reais de cada versão — ver F.0). O nativo é
só um gerador de dataset mais rápido, não substitui a validação de força.

### F.3 (FASES 3 e 4) Train — bootstrapping puro (sem Stockfish), self-play com temperatura estilo MCTS

**Mudança de direção em relação à primeira versão deste apêndice**: não é
para usar Stockfish como fonte adicional — é para tirar o oráculo do loop
inteiramente. O objetivo é um ciclo fechado estilo AlphaZero/LC0:
gerar → treinar → substituir pesos → gerar de novo, sem nenhum sinal
externo além do resultado real das partidas e da própria rede.

Isso muda o que F.2 precisa gerar e adiciona uma peça que o v400 (só valor)
ainda não tem: **cabeça de política**. Segue em dois estágios, porque MCTS
de verdade só funciona bem com política — sem ela, cada expansão de nó
seria uniforme sobre ~35 lances legais médios e o MCTS gastaria a maior
parte do orçamento de simulações em lances óbviamente ruins.

#### F.3.0 — O blend K (WDL × avaliação própria), do jeito que o zquoridor faz

Correção importante em relação ao que ficou implícito acima: o alvo de
treino não é nem "só resultado da partida" nem "só avaliação da rede" —
é uma combinação convexa das duas, com um peso `k` escolhível, exatamente
como `training/train_nnue.py` do zquoridor faz (`wl_target = k *
game_result_prob + (1 - k) * ev_prob`, `k` configurável **por fonte de
dados**, default `k=1.0` = ignora a avaliação e usa só o resultado real).

```
ev_prob          = sigmoid(eval_cp_da_amostra / OUT_SCALE_cp)   # probabilidade que a PRÓPRIA rede já dava pra si mesma naquele momento (Apêndice D.2 já define essa conversão cp→prob, reusar igual)
game_result_prob = 1.0 se o mover da amostra venceu a partida, 0.0 se perdeu, 0.5 se empate

wl_target = K * game_result_prob + (1 - K) * ev_prob
```

- **`K = 1.0`**: ignora a avaliação, treina só contra o resultado final
  real da partida (TD(1) puro — o que o Estágio 1 abaixo faz "por
  omissão" se `K` não for exposto).
- **`K = 0.0`**: ignora o resultado da partida, treina a rede a imitar a
  própria avaliação que ela já deu no momento do lance — sozinho isso não
  serve pra nada (é um alvo que já é a própria saída da rede, colapsa em
  identidade), mas em blend com `game_result_prob` suaviza o ruído de
  "resultado da partida inteira" nas posições de abertura/meio-jogo, onde
  o resultado final tem baixa correlação causal com aquele lance
  específico.
- **`K` intermediário** (zquoridor usa valores por fonte, não um único
  global): dá um alvo mais denso e menos ruidoso que resultado puro sem
  descartar o sinal de verdade-fundamental do jogo real — é o que dá
  convergência mais rápida que TD(1) puro sem precisar de MCTS+política
  ainda.

`eval_cp` (necessário pro `ev_prob`) já está no `SelfplaySample` de F.2 —
então mesmo o Estágio 1 deve gravá-lo (correção da versão anterior deste
apêndice, que dizia "só `game_result`, sem `eval_cp` nenhum" — isso estava
incompleto: `eval_cp` continua sendo gravado, só não vem mais de
Stockfish, vem da própria busca que escolheu o lance). `K` vira parâmetro
de CLI do treino (`train/train_nnue.py` ou o novo leitor de `.bin`, F.3
final), por dataset/geração — permite, por exemplo, usar `K` mais alto
(mais peso em resultado real) nas primeiras gerações de bootstrap, quando
a própria avaliação ainda é fraca e não é uma referência confiável, e
baixar `K` (mais peso na própria avaliação) conforme a rede amadurece e
sua própria avaliação vira um alvo mais estável que o ruído de "quem
ganhou a partida inteira".

#### Estágio 1 — bootstrap com o alpha-beta que já existe (sem política, sem MCTS de verdade)

Não bloqueia em nada de novo: reusa o `search_best` atual como está,
avaliando com a própria NNUE (v400 quando pronta; v3.14 serve para
começar a validar o loop antes disso). A única mudança é **como o lance é
escolhido a partir da busca**, para injetar a aleatoriedade de exploração
que hoje só existe via abertura aleatória (`RANDOM_PLIES` em
`tests/run_selfplay.py`):

Em vez de sempre jogar `best_move` (argmax), amostrar entre os N melhores
lances da raiz (Multi-PV já existe em `search_best` — reusar diretamente,
setando `n_pvs` = número de candidatos a amostrar) com uma distribuição
softmax sobre os scores, escalada por temperatura:

```
P(lance_i) = exp(score_i / T) / Σ_j exp(score_j / T)
```

- **T → 0**: colapsa em argmax (joga sempre o melhor lance — modo torneio/avaliação).
- **T = 1**: proporcional linear ao score em escala de `exp` (bastante exploração).
- **T alto** (>>1): quase uniforme entre os candidatos.

Schedule recomendado, igual AlphaZero: `T` alto nas primeiras
`temp_plies` (ex.: 20-30) da partida, depois `T → 0` (argmax) dali em
diante — abertura variada, meio-jogo/final determinístico e forte. Expor
como dois parâmetros de CLI no selfplay nativo (F.2):

```
--temperature <T0>       # temperatura nas primeiras --temp-plies jogadas (default 1.0)
--temp-plies <N>         # quantos lances usam T0 antes de cair para T→0 (default 24)
--temp-final <T1>        # temperatura após temp-plies (default 0.05, ~argmax com cauda mínima)
```

Rótulo de treino nesse estágio: `wl_target = K * game_result_prob + (1-K)
* ev_prob` (F.3.0 acima), com `eval_cp` vindo da própria busca (nunca de
Stockfish) e `K` alto (perto de 1.0) nas primeiras gerações — na prática
próximo do TD(1) puro contra o resultado final, que é o esquema de treino
por reforço mais simples que existe. `K` pode baixar gradualmente nas
gerações seguintes conforme a própria avaliação amadurece (F.3.0). Ainda
assim, convergência mais lenta que MCTS+política (Estágio 2) porque a
única fonte de sinal por posição continua sendo o resultado da partida
inteira mais a própria avaliação corrente, sem distribuição de política
para regularizar a busca.

#### Estágio 2 — MCTS real com cabeça de política (o "bootstrapping" completo)

Isso é o pedido de "MCTS de verdade" e precisa de duas peças novas (a
cabeça de valor continua treinando exatamente como em F.3.0/Estágio 1 —
`wl_target` com o mesmo blend `K`; o que muda neste estágio é só que agora
também existe uma cabeça de política, treinada em paralelo, com sua
própria loss):

1. **Cabeça de política na NNUE.** No `NN_L2_OUT`/`NN_L3` do v400
   (`nnue.h`/`nnue.c`/`train_nnue.py`/`export_nnu4.py`), adicionar uma
   segunda saída paralela à de valor — mesmo tronco compartilhado (L1
   acumulado, SCReLU), duas cabeças L2/L3 separadas. Espaço de saída:
   diferente de Quoridor (209 lances possíveis, cabe num vetor denso), em
   xadrez o espaço de lances precisa de uma codificação tipo
   AlphaZero/LC0 (73 planos × 64 casas = 4672, ou a codificação
   "from-square + direção/underpromoção" do LC0) — decidir isso é o item
   de maior superfície de F.3, não é um detalhe pequeno. Treinado com
   cross-entropy contra a distribuição de visitas do MCTS (π abaixo), não
   contra o lance jogado sozinho (isso é o que distingue política MCTS de
   simples imitação de lance, e é o que dá o sinal denso que faz o
   bootstrapping convergir em razoável número de partidas).

2. **MCTS PUCT sobre a NNUE, como motor de self-play separado do
   alpha-beta.** Não substitui `alpha_beta`/`search.c` para jogo
   competitivo (UCI) — alpha-beta com NNUE de valor continua sendo o
   motor "de verdade" para força/ELO, isso já é bem estabelecido em
   engines clássicas mesmo com política disponível (a política ajuda
   ordenação de lances no alpha-beta também, ver ideia descartada em F.1
   original). O MCTS é só o gerador de dados de treino:
   - Seleção: PUCT (`Q(s,a) + c_puct * P(s,a) * sqrt(ΣN) / (1+N(s,a))`),
     `P(s,a)` vindo da cabeça de política, `Q` da média dos valores das
     simulações (ou direto da cabeça de valor em nós folha, sem rollout —
     estilo AlphaZero, não Monte Carlo clássico com rollout até o fim).
   - Expansão: 1 chamada de NNUE por nó folha novo (valor + política),
     igual ao acumulador incremental que a v400 já constrói para
     alpha-beta (reusar `NnueAccum`/`nnue_push_na`/`nnue_pop_na` para
     empilhar/desempilhar ao longo da árvore de MCTS, exatamente como já
     é feito ao longo da pilha de alpha-beta — é a mesma primitiva).
   - Orçamento de simulações por lance: fixo (ex. 400-800 simulações,
     como AlphaZero) ou por tempo — decidir junto de F.2 (throughput de
     partidas/segundo é o gargalo real de bootstrapping, então o número
     de simulações por lance é o dial mais importante de custo).
   - **Temperatura**: aplicada sobre a distribuição de **visitas** da
     raiz, não sobre scores de Multi-PV como no Estágio 1 —
     `π(a) ∝ N(a)^(1/T)`. Mesmos três parâmetros de CLI do Estágio 1
     (`--temperature`/`--temp-plies`/`--temp-final`), só troca a fórmula
     por baixo. `π` inteiro (não só o lance amostrado) é o alvo de
     treino da cabeça de política — essa é a diferença central para o
     Estágio 1: lá só se grava o lance escolhido, aqui se grava a
     distribuição completa de visitas.

`SelfplaySample` (F.2) ganha um campo a mais neste estágio: o vetor
esparso `(índice_lance, N(a))` da raiz, ou já normalizado como `π`
quantizado (ex. top-8 lances + probabilidade em uint8) — decidir formato
junto da codificação de lances do item 1, para não gravar dois esquemas
de índice de lance diferentes (política e `policyTarget`/`move_played`
precisam usar a MESMA codificação).

**Ordem recomendada dentro de F.3**: Estágio 1 primeiro e sozinho —
valida o loop de bootstrap inteiro (self-play → grava `.bin` → treina
valor puro → reconverte pesos → repete) com esforço pequeno, usando
infraestrutura que já existe quase toda. Só depois de ver esse ciclo
convergir (rede melhora partida sobre partida sem qualquer Stockfish)
vale investir no Estágio 2, que é uma adição estrutural de verdade (nova
cabeça de rede, novo tipo de busca, novo formato de amostra) — no
espírito do resto deste plano v400, que já separa "arquitetura nova" de
"validação incremental" em todo apêndice.

`train/` ganha, para os dois estágios, um leitor de `.bin` packed
(`numpy.fromfile` com dtype estruturado, igual
`training/read_selfplay.py` do zquoridor) para consumir a saída de F.2
direto, sem passar por parquet — os datasets antigos rotulados por
Stockfish (`label_*` → parquet, usados em `TRAIN_PCT_SET1..15` hoje)
não precisam ser descartados, mas deixam de crescer: novos dados vêm só
do bootstrap a partir daqui.

### F.4 (FASE 2) Arena nativa — A/B em processo (novo: `tools/arena_native.c`)

Depende de F.1. Ideia: um binário que, dado dois conjuntos de pesos NNUE
e/ou duas constantes de busca diferentes (não duas *versões de código* —
isso o `tests/run_tournament.py` já faz bem via dois `.exe` reais, ver F.0),
joga N partidas rápidas em processo, cada lado com sua própria `TTable`
(aqui NÃO compartilhada — são adversários de verdade, compartilhar TT
entre eles seria um vazamento de informação de um lado pro outro).

Uso principal: **destrava um tuner SPSA**, que o zchezz hoje não tem
(zquoridor tem `tools/spsa/tune_spsa.cpp` + `run_spsa.py` +
`plot_spsa.py`). O zchezz tem vários números "no olho" em `search.c` que
são candidatos naturais — `CONTEMPT=15`, o divisor implícito da tabela
`lmr_tab`, o delta inicial de aspiration window (20, dobrando até 500), as
margens de futility/NMP citadas no cabeçalho do arquivo. Um SPSA nativo
roda milhares de partidas curtíssimas (`movetime` bem baixo, tipo 10-20ms)
perturbando um vetor de constantes e medindo Elo relativo — infactível via
`subprocess`+UCI (overhead de processo dominaria o tempo de jogo em si),
direto ao ponto com F.1+F.4 porque troca de "constante" vira só reapontar
para um struct de parâmetros por instância, sem recompilar.

Ordem de implementação sugerida dentro do Apêndice F, em termos de fase:
**Fase 0 (F.1) → Fase 2 (F.4, arena+SPSA) → Fase 3 (F.2+F.3.0+Estágio 1,
selfplay+bootstrap) → Fase 4 (F.3 Estágio 2, política+MCTS) → Fase 5
(F.5, ciclo contínuo)**. A arena nativa entra logo depois da Fase 0 porque
é o consumidor mais simples do `TTable` por instância (dois lados
adversários, sem a sutileza de TT compartilhada de F.2) e já entrega valor
imediato (SPSA) antes de comprometer esforço no formato de dataset novo.

### F.5 (FASE 5) Ciclo de bootstrap contínuo — gerar, treinar, promover

Fecha o loop AlphaZero-style de verdade: sem isso, Fases 3/4 produzem só
"um treino a mais", não um processo que se auto-melhora.

```
gen0 = pesos atuais (v3.14 convertida, ou HalfKP-4Bucket recém-inicializada)
repita:
    1. Selfplay (Fase 3/4) com gen_i: N partidas, temperatura por schedule
       (F.3.0/Estágio 1 ou 2), grava .bin
    2. Treino: novo checkpoint gen_{i+1} = train(dados acumulados de
       gen_i e, opcionalmente, janela das últimas M gerações — não só a
       mais recente, para não esquecer padrões que gen_i já sabia)
    3. Gate via Arena nativa (Fase 2, F.4): gen_{i+1} vs gen_i, N partidas
       curtas, SPRT (H0: Elo(gen_{i+1}) <= Elo(gen_i)+0 vs H1: Elo
       maior). Só PROMOVE gen_{i+1} a "pesos atuais" se o SPRT aceitar
       H1 com significância definida (mesmos limiares que
       run_tournament_quick.py já usa para regressão, ver elo_calc.py)
    4. Se não promoveu: descarta gen_{i+1}, gera mais dados com gen_i e
       tenta de novo (ou ajusta hiperparâmetros de treino — LR, K,
       quantidade de dados)
    i += 1
```

Ponto crítico que falta em qualquer um dos passos acima sem F.4: sem
arena nativa rápida, o passo 3 (gate) teria que rodar via
`tests/run_tournament_quick.py` (subprocess+UCI) — funciona, mas cada ciclo
gerar→treinar→validar fica ordens de magnitude mais lento que gerar. É
por isso que F.4 vem antes de F.3 na ordem de fases: o gate é tão
frequente quanto a geração (uma vez por geração), então precisa ser tão
rápido quanto ela.

`K` (F.3.0) pode variar por geração dentro deste loop: gerações iniciais
com `K` alto (perto de 1.0 — confiar no resultado real, a rede ainda não
avalia bem o suficiente pra valer a pena imitar a si mesma) e reduzir `K`
gradualmente conforme as promoções em Fase 5 se acumulam (a própria
avaliação vira uma referência cada vez mais confiável, adicionar seu sinal
denso acelera convergência sem introduzir viés — mesma lógica de
annealing de `K` que faz o AlphaZero original converger sem precisar de
um oráculo externo em nenhum momento do processo).

