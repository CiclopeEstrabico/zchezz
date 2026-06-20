/* search.h — Zchezz search layer
 *
 * Provides:
 *   search_init()     — call once after board_init()
 *   search_reset()    — call at start of every new search
 *   search_best()     — iterative deepening, returns best move + score
 *
 * All search parameters are passed through SearchParams.
 */
#pragma once
#include "board.h"

/* Forward declare SearchState for per-thread search data */
typedef struct SearchState SearchState;

#define MAX_PLY     64
#ifdef __EMSCRIPTEN__
#define TT_SIZE     (1 << 19)   /* 512K entries (~13 MB) — fits in WASM memory */
#else
#define TT_SIZE     (1 << 22)   /* 4M entries in 2M buckets (~104 MB) — native */
#endif
#define TT_BUCKETS  2           /* entries per bucket */
#define TT_SLOTS    (TT_SIZE / TT_BUCKETS)   /* number of logical slots */
#define TT_MASK     (TT_SLOTS - 1)
#define TT_EVAL_NONE 32001

/* TT entry flags (match JS: 0=exact, 1=lower, 2=upper) */
#define TT_EXACT  0
#define TT_LOWER  1
#define TT_UPPER  2

/* ── Transposition table entry (SoA layout for cache efficiency) ── */
extern uint64_t TT_H[TT_SIZE];   /* hash (64-bit) */
extern int32_t  TT_S[TT_SIZE];   /* score  */
extern int32_t  TT_D[TT_SIZE];   /* (depth<<8)|flag */
extern uint16_t TT_G[TT_SIZE];   /* generation */
extern int32_t  TT_M[TT_SIZE];   /* packed move */
extern int32_t  TT_E[TT_SIZE];   /* cached static eval */
extern uint16_t TT_GEN;

#define MAX_MULTI_PV 6

typedef struct {
    Move    best;           /* best move found (PV 1) */
    int     score;          /* score in cp (PV 1), White-relative */
    int     depth;          /* actual depth searched */
    long    nodes;
    long    tb_hits;        /* tablebase probes that returned a result */
    char    pv[256];        /* PV string for line 1 (UCI notation) */
    /* Multi-PV extension */
    int     num_pvs;        /* actual PVs returned (1..MAX_MULTI_PV) */
    int     scores[MAX_MULTI_PV];   /* scores for each PV (White-relative) */
    char    pvs[MAX_MULTI_PV][256]; /* PV strings for each line */
    Move    bests[MAX_MULTI_PV];    /* best move per PV line */
} SearchResult;

/* ── Search parameters ────────────────────────────────────────── */
typedef struct {
    int  max_depth;         /* hard depth limit */
    int  start_depth;       /* starting depth for iterative deepening (0 or 1 = start from 1) */
    int  time_limit_ms;     /* 0 = no time limit */
    long node_limit;        /* 0 = unlimited */
    int  multi_pv;          /* number of PV lines (1..MAX_MULTI_PV) */
    int  threads;           /* number of search threads (1 = single, Lazy SMP) */
    volatile int *stop;     /* pointer to shared stop flag (NULL = no external stop) */
    SearchState *search_state;  /* per-thread state (NULL = use global default) */
    void (*info_cb)(int depth, int score, long nodes, const char *pv, int turn, int multipv);
} SearchParams;

/* Syzygy tablebase probing configuration (set from UCI options) */
extern int g_tb_probe_depth;   /* minimum depth for WDL probing */
extern int g_tb_probe_limit;   /* maximum pieces for probing */

/* One-time initialisation (LMR table, MVV-LVA, clear TT) */
void search_init(void);

/* Reset per-search state (killers, history, counters) */
void search_reset(void);

/* Zera tabelas de história completamente (chamar em ucinewgame) */
void search_history_clear(void);

/* Set the active SearchState for this thread.
 * NULL resets to the default global state. */
void search_set_state(SearchState *state);

/* Allocate a new SearchState on the heap (for helper threads) */
SearchState *search_state_new(void);

/* Free a heap-allocated SearchState */
void search_state_free(SearchState *state);

/* Find best move.  Board b is not modified (make/unmake balanced). */
SearchResult search_best(Board *b, const SearchParams *p);

/* Stand-alone eval for a position (calls NNUE or fast classical) */
int eval_stm(Board *b);

/* WASM sret wrapper — called from JS worker via ccall */
void search_best_sret(SearchResult *out, Board *b, const SearchParams *p);

/* WASM helper — applies a space-separated UCI move list to *b (which must
 * already be loaded with board_load_fen).  Seeds b->hist so that
 * search_best can detect threefold repetitions correctly. */
void board_apply_moves(Board *b, const char *moves_str);
