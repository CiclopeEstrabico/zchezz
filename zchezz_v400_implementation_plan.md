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

## ARQUITETURA NOVA: HalfKP-4Bucket

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

## LISTA COMPLETA DE ARQUIVOS A MODIFICAR

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
6. mixtrain.py      ← nova encode_chunk, novo modelo NNUE, SCReLU, dims
7. convert_nnue.py  ← novo conversor NNU4 com novas dims
```

---

## PARTE 1: `nnue.h` — Constantes e NnueAccum

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

## PARTE 2: `nnue.c` — Feature Encoding e Forward Pass

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

## PARTE 3: `board.c` — Detecção de Mudança de Bucket

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

## PARTE 4: `main.c`

1. Atualizar string de versão: `"Zchezz v400"`
2. `sizeof(NnueAccum)` agora é ~258 KB — verificar que `zmalloc32` no `helper_thread_fn`
   aloca corretamente (já usa `sizeof(NnueAccum)`, não hardcoded).
3. Atualizar comentário do header.

---

## PARTE 5: `train/mixtrain.py` — Novo Modelo

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

## PARTE 6: `train/convert_nnue.py` — Conversor NNU4

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

## PARTE 7: WASM — Impacto e Compatibilidade

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

## PARTE 8: TESTES E VALIDAÇÃO

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
1. train/mixtrain.py
   a. Definir halfkp_features() e king_bucket() em Python
   b. Reescrever encode_chunk() → retorna (X_stm, X_opp)
   c. Atualizar constantes INPUT_MAIN, HIDDEN1, HIDDEN2, CONCAT_DIM
   d. Implementar SCReLU
   e. Reescrever classe NNUE com perspectivas explícitas
   f. Atualizar training loop e dataloader

2. train/convert_nnue.py
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

## APÊNDICE A: MULTIPV — Cuidados com a Nova Arquitetura

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

## APÊNDICE B: MULTITHREAD / LAZY SMP — Cuidados Específicos

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

## APÊNDICE C: TABLEBASES (SYZYGY) — Integração com v400

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

AP�S encode_chunk, o label é convertido para STM-RELATIVE:
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

## APÊNDICE E: QUANTIZAÇÃO — Guia Completo v400

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

