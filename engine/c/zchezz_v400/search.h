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
#include <stddef.h>
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

/* ── Transposition table (SoA layout for cache efficiency) ──────
 * Packaged into a per-instance struct (v4.00) so that independent
 * searches (native self-play workers, A/B arena) can each own an
 * isolated TTable instead of sharing process-global arrays.  The
 * single-search / Lazy-SMP-helpers-share-one-TT behaviour used by
 * the UCI engine is unchanged: main() allocates ONE TTable (g_tt)
 * and every SearchParams in that process points at it. */
typedef struct {
    uint64_t *H;   /* hash (64-bit) */
    int32_t  *S;   /* score  */
    int32_t  *D;   /* (depth<<8)|flag */
    uint16_t *G;   /* generation */
    int32_t  *M;   /* packed move */
    int32_t  *E;   /* cached static eval */
    uint16_t gen;  /* current generation; incremented once per ID iteration
                    * by the main thread only (helpers read but don't write) */
    size_t   size; /* number of logical entries (== TT_SIZE today) */
    size_t   mask; /* slot mask = (size/TT_BUCKETS)-1 */
} TTable;

/* Allocate a new TTable with n_entries logical entries (must be a
 * power of two multiple of TT_BUCKETS).  Returns NULL on OOM. */
TTable *tt_create(size_t n_entries);

/* Free a TTable and all of its backing arrays. Safe to call with NULL. */
void    tt_destroy(TTable *tt);

/* Physical memset of the hash array + reset of static-eval sentinels
 * and generation.  Use between independent games (self-play/arena)
 * to fully wipe stale entries. */
void    tt_clear(TTable *tt);

/* Logical generation bump (current ucinewgame semantics) — old
 * entries are ignored on next probe but not physically erased. */
void    tt_new_generation(TTable *tt);

/* Process-wide default TTable used by the UCI engine (main.c).
 * Allocated once by search_init(); every SearchParams constructed by
 * main.c points p->tt at this (Lazy SMP helpers inherit it via the
 * struct copy of SearchParams — same pointer, intentional sharing). */
extern TTable *g_tt;

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
    TTable *tt;              /* transposition table to search with (NULL = use g_tt) */
    /* v4.00: MultiPV time-budget sharing.
     *   0 (default) = current/interactive behavior: EACH PV line gets its
     *       own full time_limit_ms budget (deadline reset every PV loop
     *       iteration).  This is what UCI `go movetime X multipv N` wants —
     *       every candidate line should be searched as deeply as a single
     *       PV search would be.
     *   1 = the total time_limit_ms is divided evenly across the n_pvs
     *       lines (time_limit_ms / n_pvs each), so the whole MultiPV call
     *       costs about the same as ONE ordinary search instead of N.
     *       Intended for callers that only want per-move root scores
     *       (e.g. the self-play generator sampling by temperature), not
     *       full per-line analysis depth.  See search_best() for the
     *       depth>=2 guarantee this relies on. */
    int  mpv_share_budget;
} SearchParams;

/* Syzygy tablebase probing configuration (set from UCI options) */
extern int g_tb_probe_depth;   /* minimum depth for WDL probing */
extern int g_tb_probe_limit;   /* maximum pieces for probing */

/* One-time initialisation (LMR table, MVV-LVA, clear TT) */
void search_init(void);

/* Reset per-search state (killers, history, counters) */
void search_reset(SearchState *ss);

/* Zera tabelas de história completamente (chamar em ucinewgame) */
void search_history_clear(SearchState *s);

/* Default global SearchState (used by main thread / single-threaded mode) */
extern SearchState g_ss;

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
