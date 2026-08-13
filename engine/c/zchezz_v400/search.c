/* search.c — Zchezz search layer (v3.14: fix Lazy SMP global ss race)
 *
 * Core search functions:
 *   see()           → static exchange evaluation (iterative, bitboard)
 *   score_move()    → move ordering scores (TT hit, MVV-LVA, killer, history)
 *   eval_stm()      → NNUE evaluation (v3.13: per-thread NnueAccum)
 *   qsearch()       → quiescence search with delta pruning
 *   alpha_beta()    → main negamax with PVS, LMR, NMP, futility, SEE pruning
 *   search_best()   → iterative deepening + aspiration windows
 *
 * ── v3.13 changes ─────────────────────────────────────────────
 *
 *  1. NNUE THREAD SAFETY:
 *     eval_stm() now reads b->nnue (NnueAccum pointer) from Board
 *     and passes it to nnue_eval_bb().  Each SMP helper thread
 *     has its own NnueAccum, so concurrent evaluations are safe.
 *
 *  2. TT/TB PROBE ORDERING (critical performance fix):
 *     In alpha_beta(), the TT probe is now BEFORE the TB probe.
 *     TB results are stored in TT at depth=127, so on revisits
 *     the TT hit serves the cached TB score without any disk I/O.
 *     This eliminates the ~10ms Fathom mmap latency on repeat visits
 *     that caused -41 ELO in v3.11.
 *
 *  3. TB PV-NODE SKIP:
 *     TB probing is skipped at PV nodes (!is_pv_early) following
 *     Stockfish convention. PV nodes need full search for accurate
 *     PV lines, and the TB score is still available via TT cache.
 *
 *  4. SEARCH_BEST NNUE LIFECYCLE:
 *     search_best() calls nnue_reset(b->nnue) then nnue_rebuild()
 *     at the start.  This ensures the per-thread accumulator is
 *     fully initialized before the first eval_stm() call.
 */

#define _POSIX_C_SOURCE 200809L
#include "search.h"
#include "board.h"
#include "nnue.h"
#include "syzygy.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ── TT (per-instance SoA arrays, v4.00) ─────────────────────────
 * Transposition table with Structure-of-Arrays layout for cache locality.
 * Each logical entry contains: hash, score, depth+flags, generation,
 * packed move, and static eval.  Two entries per slot (2-bucket scheme).
 *
 * tt->H[i]  — 64-bit Zobrist hash (full, not truncated)
 * tt->S[i]  — stored score (mate scores adjusted for ply distance)
 * tt->D[i]  — packed: bits [15:8]=depth, bits [1:0]=flag (EXACT/LOWER/UPPER)
 * tt->G[i]  — generation counter (16-bit, wraps at 65536)
 * tt->M[i]  — packed move (from|to<<6|prom<<12|epc<<15|castle<<16)
 * tt->E[i]  — static eval at the time of storage (for pruning decisions)
 * tt->gen   — current generation; incremented once per ID iteration
 *             by the main thread only (helpers read but don't write).
 *
 * Total memory: n_entries * (8+4+4+2+4+4) = n_entries * 26 bytes
 *   Native (4M entries): ~104 MB
 *   WASM   (512K entries): ~13 MB
 *
 * This used to be a set of process-global arrays (TT_H/TT_S/.../TT_GEN),
 * which made it impossible to run two independent searches (e.g. a
 * native self-play worker pool or an A/B arena) in one process without
 * them contaminating each other's TT.  It is now a dynamically
 * allocated TTable; the UCI engine still allocates exactly ONE
 * instance (g_tt, created in search_init()) and every SearchParams it
 * builds points at that same instance — including Lazy SMP helper
 * threads, which intentionally continue to SHARE the TT (struct-copy
 * of SearchParams copies the pointer, not the table). Behaviour is
 * unchanged; this is pointer indirection only. */
TTable *g_tt = NULL;

TTable *tt_create(size_t n_entries) {
    TTable *tt = (TTable *)calloc(1, sizeof(TTable));
    if (!tt) return NULL;
    tt->H = (uint64_t *)calloc(n_entries, sizeof(uint64_t));
    tt->S = (int32_t  *)calloc(n_entries, sizeof(int32_t));
    tt->D = (int32_t  *)calloc(n_entries, sizeof(int32_t));
    tt->G = (uint16_t *)calloc(n_entries, sizeof(uint16_t));
    tt->M = (int32_t  *)calloc(n_entries, sizeof(int32_t));
    tt->E = (int32_t  *)calloc(n_entries, sizeof(int32_t));
    if (!tt->H || !tt->S || !tt->D || !tt->G || !tt->M || !tt->E) {
        tt_destroy(tt);
        return NULL;
    }
    tt->size = n_entries;
    tt->mask = (n_entries / TT_BUCKETS) - 1;
    tt->gen  = 0;
    for (size_t i = 0; i < n_entries; i++) tt->E[i] = TT_EVAL_NONE;
    return tt;
}

void tt_destroy(TTable *tt) {
    if (!tt) return;
    free(tt->H); free(tt->S); free(tt->D);
    free(tt->G); free(tt->M); free(tt->E);
    free(tt);
}

void tt_clear(TTable *tt) {
    if (!tt) return;
    memset(tt->H, 0, tt->size * sizeof(uint64_t));
    for (size_t i = 0; i < tt->size; i++) tt->E[i] = TT_EVAL_NONE;
    tt->gen = 0;
}

void tt_new_generation(TTable *tt) {
    if (!tt) return;
    tt->gen = (tt->gen + 1) & 0xFFFF;
}



/* Contempt value (centipawns, STM-relative).
 * Returned instead of 0 on the first repetition (draw==3) so the engine
 * avoids cycling in winning positions.  15cp is enough to prefer any
 * real continuation over repeating, while still accepting a draw when
 * actually behind.  Raise to increase aversion to draws; lower to 0
 * to disable (reverts to flat-draw behaviour). */
#define CONTEMPT 15

/* ── LMR table ───────────────────────────────────────────────── */
#define LMR_D 64
#define LMR_M 128
static uint8_t lmr_tab[LMR_D * LMR_M];   /* built once from the DEFAULT lmr_divisor at
                                           * search_init() time; kept for reference/UCI-only
                                           * builds — the hot path below now reads
                                           * g_tune.lmr_divisor live instead of this table,
                                           * so a per-thread tuner can change it without a
                                           * rebuild (see "Tunable search constants" in
                                           * search.h). Left populated so nothing else that
                                           * might reference it in the future silently reads
                                           * zeros. */

/* ── Tunable search constants — defaults ─────────────────────────
 * Same numbers that used to be bare literals in alpha_beta()/the ID
 * loop below.  Kept as named macros (not just numbers in the struct
 * initializer) so a diff against the pre-tuner history line-matches
 * the original literals. See search.h's SearchTunables comment for
 * the full contract. */
#define TUNE_RAZOR_MARGIN_DEFAULT             200
#define TUNE_RFP_MULT_DEFAULT                  90
#define TUNE_RFP_IMPROVING_BONUS_DEFAULT       50
#define TUNE_NMP_BASE_DEFAULT                   3
#define TUNE_NMP_DEPTH_DIV_DEFAULT              3
#define TUNE_NMP_MAX_R_DEFAULT                  6
#define TUNE_NMP_EVAL_BONUS_THRESHOLD_DEFAULT 200
#define TUNE_PROBCUT_MARGIN_DEFAULT           200
#define TUNE_LMR_DIVISOR_DEFAULT              1.5
#define TUNE_FUT_MULT_DEFAULT                 150
#define TUNE_FUT_IMPROVING_ADJ_DEFAULT          50
#define TUNE_ASP_DELTA_INIT_DEFAULT            20
#define TUNE_ASP_DELTA_MAX_DEFAULT            500

_Thread_local SearchTunables g_tune = {
    .razor_margin              = TUNE_RAZOR_MARGIN_DEFAULT,
    .rfp_mult                  = TUNE_RFP_MULT_DEFAULT,
    .rfp_improving_bonus       = TUNE_RFP_IMPROVING_BONUS_DEFAULT,
    .nmp_base                  = TUNE_NMP_BASE_DEFAULT,
    .nmp_depth_div              = TUNE_NMP_DEPTH_DIV_DEFAULT,
    .nmp_max_r                 = TUNE_NMP_MAX_R_DEFAULT,
    .nmp_eval_bonus_threshold  = TUNE_NMP_EVAL_BONUS_THRESHOLD_DEFAULT,
    .probcut_margin            = TUNE_PROBCUT_MARGIN_DEFAULT,
    .lmr_divisor               = TUNE_LMR_DIVISOR_DEFAULT,
    .fut_mult                  = TUNE_FUT_MULT_DEFAULT,
    .fut_improving_adj         = TUNE_FUT_IMPROVING_ADJ_DEFAULT,
    .asp_delta_init            = TUNE_ASP_DELTA_INIT_DEFAULT,
    .asp_delta_max             = TUNE_ASP_DELTA_MAX_DEFAULT,
};

void search_tunables_apply(const SearchTunables *t) { g_tune = *t; }

SearchTunables search_tunables_defaults(void) {
    SearchTunables d = {
        .razor_margin              = TUNE_RAZOR_MARGIN_DEFAULT,
        .rfp_mult                  = TUNE_RFP_MULT_DEFAULT,
        .rfp_improving_bonus       = TUNE_RFP_IMPROVING_BONUS_DEFAULT,
        .nmp_base                  = TUNE_NMP_BASE_DEFAULT,
        .nmp_depth_div              = TUNE_NMP_DEPTH_DIV_DEFAULT,
        .nmp_max_r                 = TUNE_NMP_MAX_R_DEFAULT,
        .nmp_eval_bonus_threshold  = TUNE_NMP_EVAL_BONUS_THRESHOLD_DEFAULT,
        .probcut_margin            = TUNE_PROBCUT_MARGIN_DEFAULT,
        .lmr_divisor               = TUNE_LMR_DIVISOR_DEFAULT,
        .fut_mult                  = TUNE_FUT_MULT_DEFAULT,
        .fut_improving_adj         = TUNE_FUT_IMPROVING_ADJ_DEFAULT,
        .asp_delta_init            = TUNE_ASP_DELTA_INIT_DEFAULT,
        .asp_delta_max             = TUNE_ASP_DELTA_MAX_DEFAULT,
    };
    return d;
}

/* ── MVV-LVA table ───────────────────────────────────────────── */
/* victim 1-5, attacker 1-6  → index = victim*7+attacker */
static int32_t MVV_LVA[7*7];

/* ── SearchState — per-thread search data ─────────────────────
 * All mutable search state that must be private to each thread
 * in Lazy SMP.  Single-thread mode uses a single static instance.
 * Multi-thread mode allocates one per helper on the heap.         */
struct SearchState {
    TTable *tt;    /* transposition table this search reads/writes (v4.00:
                    * per-instance TT — Lazy SMP helpers get the SAME pointer
                    * as the main thread, intentionally shared) */
    long   nodes;
    long   nodes_total;
    long   node_limit;
    long   deadline_ms;
    int    time_up;
    /* While set, time_up() reports "keep going" no matter what the stop flag
     * or the clock say.  search_best() raises it for the FIRST depth of the
     * FIRST PV line of a search that starts at depth 1, so that search always
     * has a real move to return.  See search_best() for the full rationale. */
    int    stop_guard;
    volatile int *stop_flag;
    long   tb_hits;
    Move killers[MAX_PLY][2];
    int32_t mv_history[64*64];
    int32_t counter_move[64*64];
    int16_t cont_hist[2][64][64*64];
    int prev_ft[MAX_PLY];
    int prev_to_sq[MAX_PLY];
    int8_t sing_from[MAX_PLY];
    int8_t sing_to[MAX_PLY];
    int prev_static_eval[MAX_PLY];
    Move excluded_root[MAX_MOVES];  /* TB-filtered + Multi-PV excluded moves */
    int  excluded_root_n;
};

SearchState g_ss;
long s_tb_hits = 0;  /* legacy accessor for main.c */

/* Syzygy probing configuration (set from UCI options) */
int g_tb_probe_depth = 99;  /* disabled by default; set via UCI for single-game play */
int g_tb_probe_limit = 6;   /* maximum pieces for probing */

/* ── Wall clock helper ───────────────────────────────────────── */
/* Returns current wall time in milliseconds (monotonic, not affected
 * by system clock changes). Used for time management deadlines. */
static long now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long)(ts.tv_sec * 1000LL + ts.tv_nsec / 1000000LL);
}

/* Check whether the search should stop due to:
 *   1. External stop flag (UCI "stop" command sets *stop_flag = 1)
 *   2. Time deadline exceeded (checked every 8192 nodes to amortise
 *      the syscall cost of clock_gettime — ~1μs on Linux/Windows)
 * Once time_up is set, it stays set for the remainder of the search. */
static int time_up(SearchState *ss) {
    /* Depth-1 guarantee: a "stop"/"quit" that arrives before the search
     * thread has run a single node must not make the search return an
     * empty result (which surfaces as "bestmove 0000").  While the guard
     * is up neither the stop flag nor the deadline can end the search;
     * search_best() lowers it the moment the first depth completes, so
     * the stop is honored immediately after. */
    if (ss->stop_guard) return 0;
    if (ss->stop_flag && *ss->stop_flag) { ss->time_up = 1; return 1; }
    if ((ss->nodes_total & 8191) == 0 && ss->deadline_ms > 0)
        ss->time_up = (now_ms() >= ss->deadline_ms);
    return ss->time_up;
}

/* ── TT score adjustment for mate distances ──────────────────────
 * Mate scores encode distance-to-mate as ±(19000 - ply).  When storing
 * in TT, we need to make the score ply-independent so it's valid when
 * retrieved at a different ply.  We do this by adding/subtracting the
 * current ply:
 *   Store: +mate → score+ply (removes ply bias), -mate → score-ply
 *   Read:  +mate → score-ply (restores ply bias), -mate → score+ply
 * Threshold 9000 safely separates mate scores from eval scores. */
static inline int32_t tt_score_store(int score, int ply) {
    if (score >  9000) return score + ply;
    if (score < -9000) return score - ply;
    return score;
}
static inline int32_t tt_score_read(int score, int ply) {
    if (score >  9000) return score - ply;
    if (score < -9000) return score + ply;
    return score;
}

/* Pack move into 32 bits for TT storage — mirrors JS _packMove exactly.
 * Bit layout: [0:5]=from, [6:11]=to, [12:14]=prom type, [15]=epc flag,
 *             [16:19]=castle flags.  Total: 20 bits used of 32. */
static inline int32_t pack_move(const Move *m) {
    if (!m) return 0;
    int ci = m->castle; /* 0-4 already */
    return (m->from | (m->to<<6) | ((m->prom&7)<<12) |
            (m->epc ? (1<<15) : 0) | (ci<<16));
}

/* Unpack a TT-stored move back into the Move struct.
 * Zero value → null move (all fields zeroed). */
static inline void unpack_move(int32_t v, Move *m) {
    if (!v) { m->from=0; m->to=0; m->prom=0; m->epc=0; m->castle=0; m->score=0; return; }
    m->from   = v & 63;
    m->to     = (v >> 6) & 63;
    m->prom   = (v >> 12) & 7;
    m->epc    = (v >> 15) & 1;
    m->castle = (v >> 16) & 15;
    m->score  = 0;
}

/* TT entry returned by tt_probe — unpacked for caller convenience. */
typedef struct { int score; int depth; int flag; Move move; int static_eval; } TTE;

/* Store an entry in the transposition table.
 *
 * 2-bucket replacement scheme:
 *   Bucket 0: depth-preferred — replaced only if the new entry has
 *     equal or greater depth, or the existing entry is from an older
 *     generation.  If displaced, the old entry cascades to bucket 1.
 *   Bucket 1: always-replace — acts as a catch-all for recent entries
 *     that lost the depth competition in bucket 0.
 *
 * This balances depth-quality (deep entries survive longer) with
 * recency (new shallow entries still get stored somewhere). */
static void tt_store(TTable *tt, uint64_t hash, int score,
                     int depth, int flag, const Move *move,
                     int ply, int static_eval) {
    int slot = (int)(hash & tt->mask);
    int base = slot * TT_BUCKETS;
    int stored_score = tt_score_store(score, ply);
    int32_t packed_df = ((depth & 0xFF) << 8) | (flag & 3);
    int32_t packed_mv = pack_move(move);
    int32_t se = (static_eval != TT_EVAL_NONE) ? static_eval : TT_EVAL_NONE;

    /* Bucket 0: depth-preferred (replace only if deeper or stale generation) */
    int exist_depth0 = (tt->D[base] >> 8) & 0xFF;
    if (!tt->H[base] || tt->G[base] != tt->gen || depth >= exist_depth0) {
        /* Cascade displaced entry to bucket 1 (preserve it for move ordering) */
        if (tt->H[base] && tt->G[base] == tt->gen && depth >= exist_depth0) {
            tt->H[base+1] = tt->H[base]; tt->S[base+1] = tt->S[base];
            tt->D[base+1] = tt->D[base]; tt->G[base+1] = tt->G[base];
            tt->M[base+1] = tt->M[base]; tt->E[base+1] = tt->E[base];
        }
        tt->H[base] = hash; tt->S[base] = stored_score;
        tt->D[base] = packed_df; tt->G[base] = tt->gen;
        tt->M[base] = packed_mv; tt->E[base] = se;
        return;
    }

    /* Bucket 1: always-replace (catches shallow/recent entries) */
    tt->H[base+1] = hash; tt->S[base+1] = stored_score;
    tt->D[base+1] = packed_df; tt->G[base+1] = tt->gen;
    tt->M[base+1] = packed_mv; tt->E[base+1] = se;
}

static int tt_probe(TTable *tt, uint64_t hash, int ply, TTE *out) {
    int slot = (int)(hash & tt->mask);
    int base = slot * TT_BUCKETS;

    /* Check both buckets */
    for (int b = 0; b < TT_BUCKETS; b++) {
        int idx = base + b;
        if (tt->H[idx] != hash) continue;
        if (tt->G[idx] != tt->gen) {
            /* Stale generation: reuse the stored move for ordering, but not the score */
            out->score  = TT_EVAL_NONE;
            out->depth  = 0;
            out->flag   = TT_UPPER;
            out->static_eval = TT_EVAL_NONE;
            unpack_move(tt->M[idx], &out->move);
            return 2;   /* 2 = stale hit (move only) */
        }
        int d = tt->D[idx];
        out->score  = tt_score_read(tt->S[idx], ply);
        out->depth  = (d >> 8) & 0xFF;
        out->flag   = d & 3;
        out->static_eval = tt->E[idx];
        unpack_move(tt->M[idx], &out->move);
        return 1;   /* 1 = full hit */
    }
    return 0;
}

/* ── Eval ────────────────────────────────────────────────────── */
int eval_stm(Board *b) {
    if (!nnue_ready()) {
        static int warned = 0;
        if (!warned) {
            warned = 1;
            fprintf(stderr, "[NNUE] eval called but weights not loaded — returning 0\n");
        }
        return 0;
    }
    /* Fast eval path: precomputed bb[12] + per-thread ext feature cache.
     * HM accumulator (768 features) is updated incrementally by
     * nnue_push/pop in board_make/unmake — reads from _acc_buf_w[_acc_ptr].
     * Ext features (31 endgame: piece counts, passed pawns, king distance)
     * are recomputed on cache miss via _compute_extra_feat_bb. */
    return nnue_eval_bb(b->nnue, b->turn == COL_W ? 0 : 1, b->b, b->bb, b->hash);
}

/* ── SEE (pure bitboard — Phase 3) ────────────────────────────
 * No mailbox copy, no see_occ_bb, no see_lva.
 * Uses Board's bb[] arrays + magic lookups directly.
 * X-ray attacks are uncovered when attackers are removed from occ. */

/* Find all attackers of `sq` given occupancy `occ` */
static inline uint64_t see_attackers(const Board *bd, int sq, uint64_t occ) {
    return (bpawn_attacks_bb((uint64_t)1 << sq) & bd->bb[0])   /* WP */
         | (wpawn_attacks_bb((uint64_t)1 << sq) & bd->bb[6])   /* BP */
         | (NATK[sq] & (bd->bb[1] | bd->bb[7]))                /* N */
         | (bish_attacks(sq, occ) & (bd->bb[2] | bd->bb[4] | bd->bb[8] | bd->bb[10])) /* B+Q */
         | (rook_attacks(sq, occ) & (bd->bb[3] | bd->bb[4] | bd->bb[9] | bd->bb[10])) /* R+Q */
         | (KATK[sq] & (bd->bb[5] | bd->bb[11]));              /* K */
}

static int see_board(const Board *bd, int from, int to, int is_epc) {
    if (!bd->b[to] && !is_epc) return 0;

    int gained[32];
    int ng = 0;

    /* Initial capture value */
    int attacker_type = PC_TYPE(bd->b[from]);
    if (is_epc)
        gained[ng++] = MV_TAB[1]; /* pawn captured via ep */
    else
        gained[ng++] = MV_TAB[PC_TYPE(bd->b[to])];

    int last_cap_val = MV_TAB[attacker_type];

    /* Build occupancy from board's cached value, modified for the initial move */
    uint64_t occ = bd->occ;
    occ &= ~((uint64_t)1 << from);  /* remove attacker from its origin */
    occ |=  ((uint64_t)1 << to);    /* attacker now on target */
    if (is_epc) {
        int cap_sq = PC_COLOR(bd->b[from]) == COL_W ? to + 8 : to - 8;
        occ &= ~((uint64_t)1 << cap_sq);
    }

    int col = PC_COLOR(bd->b[from]) ^ 24; /* opponent moves next */

    /* Piece bitboard indices by color: [COL_W] = 0..5, [COL_B] = 6..11
     * Piece type order for LVA: pawn(1)=bb[0/6], knight(2)=bb[1/7],
     * bishop(3)=bb[2/8], rook(4)=bb[3/9], queen(5)=bb[4/10], king(6)=bb[5/11] */
    static const int VAL_ORDER[6] = {0, 1, 2, 3, 4, 5}; /* index offsets for P,N,B,R,Q,K */

    while (ng < 32) {
        /* Refresh attackers with updated occupancy (reveals X-ray attacks) */
        uint64_t attackers = see_attackers(bd, to, occ);

        /* Filter to only the current side's pieces that are still on the board */
        int base = (col == COL_W) ? 0 : 6;
        uint64_t side_atk = attackers & ((col == COL_W) ? (bd->bb[0]|bd->bb[1]|bd->bb[2]|bd->bb[3]|bd->bb[4]|bd->bb[5])
                                                        : (bd->bb[6]|bd->bb[7]|bd->bb[8]|bd->bb[9]|bd->bb[10]|bd->bb[11]));
        side_atk &= occ; /* only pieces still on the board */

        if (!side_atk) break;

        /* Find least valuable attacker */
        int found_sq = -1;
        int found_val = 0;
        for (int t = 0; t < 6; t++) {
            uint64_t piece_atk = side_atk & bd->bb[base + VAL_ORDER[t]];
            if (piece_atk) {
                found_sq = __builtin_ctzll(piece_atk);
                found_val = MV_TAB[t + 1]; /* type 1..6 */
                break;
            }
        }
        if (found_sq < 0) break;

        gained[ng++] = last_cap_val;
        last_cap_val = found_val;

        /* Remove this attacker from occupancy */
        occ &= ~((uint64_t)1 << found_sq);

        col ^= 24;
    }

    /* Retrograde minimax */
    int sc = 0;
    for (int i = ng - 1; i >= 1; i--)
        sc = gained[i] - sc > 0 ? gained[i] - sc : 0;
    return gained[0] - sc;
}


/* ── Move scoring + sorting ──────────────────────────────────── */
/*
 * Move ordering priority (descending score):
 *   2 000 000   PV / TT move
 *   1 700 000+  Promotions
 *   1 600 000+  Winning/equal captures (SEE >= 0)
 *     900 000   Killer 0
 *     800 000   Killer 1
 *     780 000   Counter move
 *     750 000+  Rook-to-7th / king-zone attack + history
 *     600 000+  History-positive quiet moves
 *     300 000   Quiet moves with zero/negative history (floor)
 *     100 000+  Losing captures (SEE < 0, score 100000..200000)
 *
 * The 300 000 floor for quiets keeps them above losing captures
 * (which cap at 200 000) regardless of CMH sign.
 *
 * `cmh0` = ss->prev_to_sq[ply-1],  -1 if not available
 * `cmh1` = ss->prev_to_sq[ply-2],  -1 if not available
 */
static int score_move(SearchState *ss, const Move *m, const Board *bd, int ply,
                      const Move *pv_move, int ok_sq,
                      int prev_ft_val, int cmh0, int cmh1) {
    const uint8_t *b = bd->b;
    int mfr = m->from, mto = m->to;
    if (pv_move && mfr==pv_move->from && mto==pv_move->to) return 2000000;

    uint8_t cap = b[mto];
    if (cap || m->epc) {
        int victim   = cap ? PC_TYPE(cap) : 1;
        int attacker = PC_TYPE(b[mfr]);
        int ep_bonus = m->epc ? 60 : 0;
        if (MV_TAB[attacker] <= MV_TAB[victim])
            return 1600000 + 1000 + MVV_LVA[victim*7+attacker] + ep_bonus;
        int sv = see_board(bd, mfr, mto, m->epc);
        if (sv >= 0)
            return 1600000 + sv*10 + MVV_LVA[victim*7+attacker] + ep_bonus;
        else {
            /* Losing capture: 100000..200000 */
            int s = 200000 + sv*2;
            return s > 100000 ? s : 100000;
        }
    }
    if (m->prom)   return 1700000 + MV_TAB[m->prom];
    if (m->castle) return 1500000;

    /* ── Quiet move scoring ──────────────────────────────────────
     * Combined bonus = history + CMH-slot-0 + CMH-slot-1.
     * Each component is int16_t (-32000..32000); combined fits int32.
     * We add a base of 300 000 so that a worst-case combined score of
     * 300 000 + 3*(-32000) = 204 000 stays above losing captures.     */
    if (ss->killers[ply][0].from==mfr && ss->killers[ply][0].to==mto &&
        (ss->killers[ply][0].from||ss->killers[ply][0].to)) return 900000;
    if (ss->killers[ply][1].from==mfr && ss->killers[ply][1].to==mto &&
        (ss->killers[ply][1].from||ss->killers[ply][1].to)) return 800000;
    if (prev_ft_val >= 0 && ss->counter_move[prev_ft_val] == (mfr*64+mto)) return 780000;

    int ft = mfr*64 + mto;
    int h  = ss->mv_history[ft];
    int c0 = (cmh0 >= 0) ? ss->cont_hist[0][cmh0][ft] : 0;
    int c1 = (cmh1 >= 0) ? ss->cont_hist[1][cmh1][ft] : 0;
    int combined = h + c0 + c1;   /* range: -96000..96000 */

    /* Rook-to-7th / king-zone attack: bump into a named tier */
    uint8_t piece = b[mfr];
    if (piece && PC_TYPE(piece)==4) {
        int to_rank = mto>>3;
        if ((PC_COLOR(piece)==COL_W && to_rank==1) ||
            (PC_COLOR(piece)==COL_B && to_rank==6))
            return 700000 + combined;
    }
    if (piece && ok_sq >= 0) {
        int kr=ok_sq>>3,kc=ok_sq&7,tr=mto>>3,tc=mto&7;
        int dr=tr-kr; if(dr<0)dr=-dr;
        int dc=tc-kc; if(dc<0)dc=-dc;
        if ((dr>dc?dr:dc) <= 2)
            return 600000 + combined;
    }

    /* General quiet: base 200 500 + combined bonus.
     * Range with worst combined (-48000): 200500 - 48000 = 152500 — above
     * the losing-capture floor (100 000) but below the best losing captures
     * (~200 000).  Moves that have been penalised repeatedly will therefore
     * naturally sort below bad captures, restoring the pruning density of
     * the original code while still honouring the CMH signal.
     *
     * Clamped from below at 50 000 to avoid negative scores in the array
     * (score field is int32, so no overflow, but very negative scores could
     * confuse pick-best in qsearch).  */
    int s = 200500 + combined;
    return s > 50000 ? s : 50000;
}

/* Insertion sort — stable, good for small n (typically < 50 moves) */
static void sort_moves(SearchState *ss, Move *moves, int n, const Board *bd, int ply,
                       const Move *pv_move, int ok_sq, int prev_ft_val,
                       int cmh0, int cmh1) {
    for (int i = 0; i < n; i++)
        moves[i].score = score_move(ss, &moves[i], bd, ply, pv_move, ok_sq,
                                    prev_ft_val, cmh0, cmh1);
    for (int i = 1; i < n; i++) {
        Move m = moves[i]; int j = i-1;
        while (j >= 0 && moves[j].score < m.score) { moves[j+1]=moves[j]; j--; }
        moves[j+1] = m;
    }
}

/* ── Quiescence search (QS) ──────────────────────────────────────
 * Extends the search beyond the horizon depth to resolve tactical
 * sequences (captures, promotions, check evasions).  Without QS,
 * the engine would misevaluate positions where a piece is hanging
 * ("horizon effect").
 *
 * Structure:
 *   1. Stand-pat: evaluate the position statically.  If stand-pat
 *      score >= beta, cut off (side to move can "stand pat" = do nothing).
 *   2. Delta pruning: if even capturing the best piece can't raise
 *      the score above alpha, skip all captures.
 *   3. SEE pruning: skip captures with negative SEE (losing exchanges).
 *   4. TT probe: check if this position was already evaluated at QS depth.
 *   5. Check evasion: if in check, search ALL moves (not just captures).
 *
 * Does NOT store results in TT (depth-0 entries would pollute the TT
 * and displace more valuable deeper entries). */
static int qsearch(SearchState *ss, Board *b, int alpha, int beta, int ply) {
    if (ply >= MAX_PLY-1) return eval_stm(b);
    /* node_limit is a budget for the WHOLE search, so it must be checked
     * against nodes_total: ss->nodes is reset at every iterative-deepening
     * depth (see search_best()), and checking that instead grants a fresh
     * budget per depth -- i.e. no overall stop condition at all when no
     * time limit is set.  UCI "go nodes N" means N total nodes. */
    if (ss->nodes_total >= ss->node_limit || time_up(ss)) return eval_stm(b);
    ss->nodes++; ss->nodes_total++;
    TTable *tt = ss->tt;   /* per-search TT, cached locally (never changes mid-search) */

    /* ── TT probe in qsearch ─────────────────────────────────── */
    uint64_t qs_hash = b->hash;
    TTE qs_tte;
    int qs_tte_hit = tt_probe(tt, qs_hash, ply, &qs_tte);
    Move qs_tt_move = {0};
    if (qs_tte_hit) {
        qs_tt_move = qs_tte.move;
        if (qs_tte_hit == 1 && qs_tte.depth >= 0) {
            /* TT cutoff in qsearch (depth 0 = qsearch depth) */
            if (qs_tte.flag == TT_EXACT) return qs_tte.score;
            if (qs_tte.flag == TT_LOWER && qs_tte.score >= beta) return qs_tte.score;
            if (qs_tte.flag == TT_UPPER && qs_tte.score <= alpha) return qs_tte.score;
        }
    }

    int in_check = board_in_check(b);
    if (in_check) {
        /* In check: generate all moves */
        Move moves[MAX_MOVES];
        int n = board_gen_moves(b, moves);
        int ok_sq = b->turn==COL_W ? b->bk : b->wk;
        sort_moves(ss, moves, n, b, ply, NULL, ok_sq, -1, -1, -1);
        int best = -99999, legal = 0;
        Move best_move_qs = {0};
        for (int i = 0; i < n; i++) {
            board_make(b, &moves[i]);
            int prev_turn = b->turn ^ 24;
            if (board_is_attacked(b, prev_turn==COL_W ? b->wk : b->bk, b->turn)) {
                board_unmake(b); continue;
            }
            legal++;
            int sc = -qsearch(ss, b, -beta, -alpha, ply+1);
            board_unmake(b);
            if (sc > best) { best = sc; best_move_qs = moves[i]; }
            if (sc > alpha) alpha = sc;
            if (alpha >= beta) {
                return beta;
            }
        }
        if (!legal) return -19000 + ply;   /* checkmate */
        /* Store result for check evasion — only cutoffs worth storing */
        return best;
    }

    /* Use TT-stored eval as stand-pat if available */
    int stand = (qs_tte_hit && qs_tte.static_eval != TT_EVAL_NONE)
                 ? qs_tte.static_eval : eval_stm(b);
    int qs_best = stand;
    int qs_orig_alpha = alpha;

    if (stand >= beta) return beta;
    /* Delta pruning */
    {
        int has_passer = (b->turn==COL_W) ? !!(b->bb[0] & 0x000000000000FF00ULL)
                                           : !!(b->bb[6] & 0x00FF000000000000ULL);
        int delta_margin = has_passer ? MV_TAB[5]*2+50 : MV_TAB[5]+50;
        if (stand + delta_margin < alpha) return alpha;
    }
    if (stand > alpha) alpha = stand;

    /* Captures + promotions — score all first, then pick-best */
    Move moves[MAX_MOVES];
    int n = board_gen_captures(b, moves);
    int ok_sq = b->turn==COL_W ? b->bk : b->wk;
    Move best_move_qs = {0};

    /* Use TT move for move ordering boost */
    const Move *qs_pv = (qs_tt_move.from || qs_tt_move.to) ? &qs_tt_move : NULL;

    for (int i = 0; i < n; i++) {
        /* SEE pruning: mark bad captures */
        if (!moves[i].prom && !moves[i].epc) {
            int sv = see_board(b, moves[i].from, moves[i].to, 0);
            if (sv < 0) { moves[i].score = -99999; continue; }
        }
        /* Delta pruning per move */
        uint8_t cap = b->b[moves[i].to];
        int gain = moves[i].epc ? MV_TAB[1] : cap ? MV_TAB[PC_TYPE(cap)] : 0;
        if (stand + gain + 50 < alpha && !moves[i].prom) { moves[i].score = -99999; continue; }
        moves[i].score = score_move(ss, &moves[i], b, ply, qs_pv, ok_sq, -1, -1, -1);
    }

    for (int i = 0; i < n; i++) {
        /* Pick-best */
        int best_idx = i;
        for (int j = i+1; j < n; j++)
            if (moves[j].score > moves[best_idx].score) best_idx = j;
        if (moves[best_idx].score <= -99999) break;
        if (best_idx != i) { Move tmp = moves[i]; moves[i] = moves[best_idx]; moves[best_idx] = tmp; }

        board_make(b, &moves[i]);
        int mover_col = b->turn ^ 24;
        int king_sq   = mover_col == COL_W ? b->wk : b->bk;
        if (board_is_attacked(b, king_sq, b->turn)) { board_unmake(b); continue; }
        int sc = -qsearch(ss, b, -beta, -alpha, ply+1);
        board_unmake(b);

        if (sc > qs_best) { qs_best = sc; best_move_qs = moves[i]; }
        if (sc >= beta) {
            return beta;
        }
        if (sc > alpha) alpha = sc;
    }

    /* Don't store non-cutoff qsearch results — they pollute the TT
     * with depth-0 entries that displace more valuable deeper entries */

    return alpha;
}

/* ── Alpha-Beta (negamax with PVS + all pruning) ─────────────────
 *
 * Main search function.  Returns the minimax score for the position
 * from the perspective of the side to move.
 *
 * Parameters:
 *   ss             — per-thread search state (killers, history, etc.)
 *   b              — board position (modified by make/unmake)
 *   depth          — remaining search depth (0 → drop to qsearch)
 *   alpha, beta    — search window (PV: wide; non-PV: zero-width)
 *   pv, pv_len     — output: best line found (PV array)
 *   ply            — distance from root (0 = root)
 *   in_check_hint  — 1 if known in check, 0 if known not, -1 if unknown
 *
 * Pruning techniques (in order of application):
 *   1. Draw detection (3-fold, 50-move, 1st repetition with contempt)
 *   2. TT probe (v3.13: BEFORE TB probe)
 *   3. Syzygy TB probe (non-root, non-PV, with TT caching)
 *   4. Razoring (depth 1, static eval + 200 < alpha)
 *   5. Reverse Futility Pruning (depth 2-9, eval - margin >= beta)
 *   6. Null Move Pruning (depth >= 3, non-endgame, eval >= beta)
 *   7. ProbCut (depth >= 5, shallow search with beta+200)
 *   8. IIR (Internal Iterative Reduction, no TT move at depth >= 4)
 *   9. Singular Extension (depth >= 7, TT move much better than alternatives)
 *  10. LMP (Late Move Pruning, quiet moves past count limit)
 *  11. History Pruning (quiet moves with very negative history)
 *  12. Futility Pruning (eval + margin <= alpha, quiet non-checking moves)
 *  13. SEE Pruning (losing captures below threshold)
 *  14. LMR (Late Move Reduction, quiet/losing-capture moves searched shallower)
 *
 * Move ordering (staged generation):
 *   Stage 1: TT/PV move (no generation needed)
 *   Stage 2: Captures + promotions (good captures first, losing deferred)
 *   Stage 3: Quiet moves (killers, counter, history-ordered)
 *   Stage 4: Losing captures (deferred from stage 2)
 */
/* Multi-PV root exclusion — forward declarations (defined after search_reset) */
static int  is_excluded_root(SearchState *ss, const Move *m);

/* Forward declaration */
static int alpha_beta(SearchState *ss, Board *b, int depth, int alpha, int beta,
                      Move *pv, int *pv_len, int ply, int in_check_hint);

static int alpha_beta(SearchState *ss, Board *b, int depth, int alpha, int beta,
                      Move *pv, int *pv_len, int ply, int in_check_hint) {
    if (ply >= MAX_PLY-1) { *pv_len=0; return eval_stm(b); }
    /* Whole-search budget -- see qsearch()'s note on why this is
     * nodes_total and not the per-depth ss->nodes. */
    if (ss->nodes_total >= ss->node_limit || time_up(ss)) {
        *pv_len=0;
        return depth<=0 ? eval_stm(b) : qsearch(ss,b,alpha,beta,ply);
    }
    ss->nodes++; ss->nodes_total++;
    *pv_len = 0;
    TTable *tt = ss->tt;   /* per-search TT, cached locally (never changes mid-search) */

    int mate_val = 19000 - ply;
    if (alpha >  mate_val) return  mate_val;
    if (beta  < -mate_val) return -mate_val;

    /* Draw check — only inside the tree (ply > 0).
     * At the root (ply == 0) we never return early: the arbiter (chess.js /
     * tournament.js) is responsible for claiming draws; the engine must always
     * return a legal move so the arbiter can decide.  Returning 0 at the root
     * left best_move as a null struct → "0000" sent to the UI → false draw.
     *
     * Sign: alpha_beta is negamax — the return value is negated by the parent.
     * So returning -CONTEMPT here means the PARENT (the side that chose to
     * play into this repeated position) sees +CONTEMPT — i.e. it looks good
     * to repeat, which is wrong.  We must return +CONTEMPT so the parent sees
     * -CONTEMPT (slightly bad for the side that moves into the repetition).
     *
     * draw==2 (true 3-fold) and draw==1 (50-move): return 0 exactly — these
     * are legally drawn positions; the arbiter will claim it anyway.
     * draw==3 (1st repetition): return +CONTEMPT so the mover into this
     * position is penalised, discouraging repetition when ahead. */
    if (ply > 0) {
        int draw = board_is_draw(b);
        if (draw == 2 || draw == 1) return 0;       /* true 3-fold / 50-move */
        if (draw == 3) return CONTEMPT;              /* penalise the mover into repetition */
    }

    /* ══════════════════════════════════════════════════════════════
     * TT PROBE  (v3.13: moved BEFORE TB probe — Stockfish ordering)
     * ══════════════════════════════════════════════════════════════
     *
     * WHY TT-BEFORE-TB MATTERS:
     * TB results are stored in TT at depth+6 (see below).  By probing
     * TT first, subsequent visits to the SAME position get the TB score
     * from fast RAM instead of slow disk/mmap I/O (~10ms per Fathom call).
     * In v3.11, TB was probed BEFORE TT, causing:
     *   - Redundant disk reads on revisits (depth+6 entry never used)
     *   - ~75 ELO penalty vs no-TB baseline
     *
     * Probe order (matches Stockfish):
     *   1. Draw detection  (above)
     *   2. TT probe        ← catches depth=127 cached TB entries
     *   3. TB probe        ← only fires on TT miss / shallow entry
     *
     * MULTI-PV GUARD (ply == 0 && excluded_root_n > 0):
     * At the root, when Multi-PV has excluded previous best moves,
     * we must NOT cut off from TT — we need to search past the excluded
     * moves to find the next-best PV line.  Without this guard, Multi-PV
     * lines 2+ would always return the same (cached) score as line 1. */
    TTE tte; int tte_hit = 0;
    Move pv_move = {0};
    tte_hit = tt_probe(tt, b->hash, ply, &tte);
    if (tte_hit) {
        pv_move = tte.move;
        if (tte_hit == 1 && tte.depth >= depth && !(ply == 0 && ss->excluded_root_n > 0)) {
            if (tte.flag == TT_EXACT) return tte.score;
            if (tte.flag == TT_LOWER && tte.score >= beta) return tte.score;
            if (tte.flag == TT_UPPER && tte.score <= alpha) return tte.score;
        }
    }

    /* ══════════════════════════════════════════════════════════════
     * SYZYGY TABLEBASE WDL PROBE  (in-tree, non-root, non-PV)
     * ══════════════════════════════════════════════════════════════
     *
     * CONDITIONS FOR PROBING (all must be true):
     *   - ply > 0        : never at root (root uses move-filtering, see search_best)
     *   - depth >= g_tb_probe_depth : skip shallow nodes (default: 6)
     *   - !is_pv_early   : skip PV nodes (beta - alpha > 1 = PV window)
     *     PV nodes need full search for accurate PV lines.
     *     The TB score is still available via TT cache from sibling nodes.
     *   - npieces <= g_tb_probe_limit : within loaded table range
     *
     * WDL VALUES (from Fathom, STM-relative):
     *   4 = win, 3 = cursed win, 2 = draw, 1 = blessed loss, 0 = loss
     *
     * TT CACHING STRATEGY:
     * After a successful probe, we store the result in TT.
     * - Wins/losses: depth=127 (absolute truth, never overridden by search).
     *   Win → TT_LOWER, Loss → TT_UPPER.
     * - Draws/cursed/blessed: depth=current_search_depth (so deeper
     *   NNUE searches can override). This prevents "TT poisoning" where
     *   a TB draw score of 0 replaces a NNUE eval of +3.0.
     *   Blessed loss → TT_LOWER (at least +1), Cursed win → TT_UPPER (at most -1).
     *
     * CUTOFF LOGIC:
     * Only definitive wins/losses cause immediate cutoffs.
     * Draws/cursed/blessed are stored in TT but the search continues
     * so NNUE can find practical advantages in TB-drawn positions.
     *
     * RULE50 GUARD (Stockfish: pos.rule50_count() == 0):
     * Syzygy WDL tables are only defined for positions with rule50==0
     * (just after a capture or pawn move).  Fathom returns FAILED for
     * rule50 > 0, but we add this guard to avoid even the function call
     * overhead on non-eligible positions.  Without this guard, the engine
     * was making ~3x more Fathom calls than Stockfish, losing 150 ELO. */
    int is_pv_early = (beta - alpha > 1);
    if (ply > 0 && !is_pv_early && b->hm == 0) {
        int npieces = __builtin_popcountll(b->occ);
        /* Stockfish cardinality filter:
         * pieces < limit → probe at ALL depths (cheap, always available)
         * pieces == limit → probe only at deep nodes (depth >= probe_depth)
         * This avoids probing overhead at shallow nodes where the TB
         * knowledge doesn't change the search result anyway. */
        if (npieces < g_tb_probe_limit
            || (npieces <= g_tb_probe_limit && depth >= g_tb_probe_depth)) {
            int wdl;
            if (syzygy_probe_wdl(b, &wdl)) {
                ss->tb_hits++;
                int tb_score, tb_flag;
                switch (wdl) {
                    case 4: tb_score = 18000 - ply; tb_flag = TT_LOWER; break;
                    case 3: tb_score = 1;           tb_flag = TT_LOWER; break;  /* blessed: at least +1 */
                    case 2: tb_score = 0;           tb_flag = TT_EXACT; break;
                    case 1: tb_score = -1;          tb_flag = TT_UPPER; break;  /* cursed: at most -1 */
                    case 0: default:
                        tb_score = -18000 + ply;    tb_flag = TT_UPPER; break;
                }
                /* Store TB result in TT.
                 * Definitive results (win/loss/draw) use depth+6 (Stockfish
                 * convention) for strong TT persistence.
                 * Blessed/cursed draws use current depth so NNUE searches
                 * can find practical advantages in 50-move-rule draws. */
                int tb_tt_depth = (wdl == 3 || wdl == 1) ? depth
                                : (depth + 6 < 127 ? depth + 6 : 127);
                tt_store(tt, b->hash, tb_score, tb_tt_depth, tb_flag, NULL, ply, TT_EVAL_NONE);
                /* Cut off for definitive results (Stockfish-style):
                 *   EXACT (draw WDL=2):  always return 0.
                 *   LOWER (win WDL=4):   return if tb_score >= beta.
                 *   UPPER (loss WDL=0):  return if tb_score <= alpha.
                 * Blessed/cursed (WDL 3/1): tighten bounds, continue search. */
                if (tb_flag == TT_EXACT
                    || (tb_flag == TT_LOWER && tb_score >= beta)
                    || (tb_flag == TT_UPPER && tb_score <= alpha)) {
                    return tb_score;
                }
                /* For wins that didn't cut off: tighten alpha */
                if (tb_flag == TT_LOWER && tb_score > alpha) alpha = tb_score;
                /* For losses that didn't cut off: tighten beta */
                if (tb_flag == TT_UPPER && tb_score < beta) beta = tb_score;
            }
        }
    }

    if (depth <= 0) return qsearch(ss, b, alpha, beta, ply);

    int in_check = in_check_hint >= 0 ? in_check_hint : board_in_check(b);
    if (in_check) depth++;

    int is_pv = (beta - alpha > 1);
    int static_eval = TT_EVAL_NONE;
    int raw_eval = TT_EVAL_NONE;  /* uncorrected eval for improving flag */
    if (!in_check) {
        raw_eval = (tte_hit && tte.static_eval != TT_EVAL_NONE)
                    ? tte.static_eval : eval_stm(b);
        static_eval = raw_eval;
        /* TT-score correction: use TT score as better eval estimate for pruning.
         * This is the "can_use_tt_value" technique from Stockfish. */
        if (tte_hit == 1 && tte.score != TT_EVAL_NONE) {
            if (tte.flag == TT_EXACT ||
                (tte.flag == TT_LOWER && tte.score > static_eval) ||
                (tte.flag == TT_UPPER && tte.score < static_eval)) {
                static_eval = tte.score;
            }
        }
    }

    /* ── Improving flag ────────────────────────────────────────────
     * Tracks whether the static eval is improving compared to 2 plies
     * ago (same side to move).  Used to tune pruning aggressiveness:
     *   - improving=true  → less pruning (position is getting better)
     *   - improving=false → more pruning (position is stagnant/worsening)
     * Uses raw_eval (not TT-corrected) for consistency across plies. */
    int improving = 0;
    if (!in_check) {
        ss->prev_static_eval[ply] = raw_eval;
        if (ply >= 2 && ss->prev_static_eval[ply-2] != TT_EVAL_NONE)
            improving = raw_eval > ss->prev_static_eval[ply-2];
    } else {
        ss->prev_static_eval[ply] = TT_EVAL_NONE;
    }

    /* ── Razoring ─────────────────────────────────────────────────
     * At depth 1, if eval is far below alpha (by 200cp), verify with
     * a qsearch.  If qsearch confirms, this node is hopeless — prune.
     * Only at depth 1 because deeper nodes have more tactical potential. */
    if (!in_check && !is_pv && depth==1 && static_eval+g_tune.razor_margin < alpha) {
        int qs = qsearch(ss, b, alpha-1, alpha, ply);
        if (qs < alpha) return qs;
    }

    /* ── Reverse Futility Pruning (RFP) ──────────────────────────
     * If eval is so far above beta that no reasonable score drop at
     * depth d could bring it below beta, return early.  The margin
     * is depth*90cp, reduced by 50cp if the position is improving.
     * Extended to depth 9 (Stockfish uses up to depth 9 too).
     * Guard: skip near mate scores to avoid pruning forced mates. */
    if (!in_check && !is_pv && depth>=2 && depth<=9 && beta<18000 && static_eval<18000) {
        int rfp_margin = depth*g_tune.rfp_mult - (improving ? g_tune.rfp_improving_bonus : 0);
        if (static_eval - rfp_margin >= beta) return static_eval;
    }

    /* Null Move Pruning */
    int not_endgame = 0;
    { uint64_t *bb = b->bb;
      /* New layout: bb[0]=WP bb[1]=WN bb[2]=WB bb[3]=WR bb[4]=WQ bb[5]=WK
       *             bb[6]=BP bb[7]=BN bb[8]=BB bb[9]=BR bb[10]=BQ bb[11]=BK */
      if (b->turn==COL_W) {
          not_endgame = bb[4]||bb[3]||((__builtin_popcountll(bb[1])+__builtin_popcountll(bb[2]))>=2);
      } else {
          not_endgame = bb[10]||bb[9]||((__builtin_popcountll(bb[7])+__builtin_popcountll(bb[8]))>=2);
      }
    }
    if (!in_check && !is_pv && depth>=3 && ply>0 && not_endgame && static_eval>=beta) {
        /* NMP reduction: base 3 + depth/3, capped at 6.
         * Add +1 if static_eval is much above beta (eval margin bonus). */
        int R = g_tune.nmp_base + depth / g_tune.nmp_depth_div;
        if (R > g_tune.nmp_max_r) R = g_tune.nmp_max_r;
        if (static_eval - beta > g_tune.nmp_eval_bonus_threshold) R += 1;
        /* Make null move */
        uint64_t save_hash = b->hash;
        int8_t   save_ep   = b->ep;
        int      save_turn = b->turn;
        int      save_hm   = b->hm;
        int      save_hist_len = b->hist_len;
        if (b->ep >= 0) { b->hash^=ZR_ep[b->ep&7]; b->ep=-1; }
        b->hash ^= ZR_side; b->turn ^= 24;
        b->hm++;
        /* Push null-move hash into b->hist so board_is_draw detects repetitions */
        if (b->hist_len < HIST_SIZE) {
            b->hist[b->hist_len] = b->hash;
        }
        b->hist_len++;
        Move dummy_pv[MAX_PLY]; int dummy_len=0;
        int null_score = -alpha_beta(ss, b, depth-1-R, -beta, -beta+1, dummy_pv, &dummy_len, ply+1, -1);
        b->hist_len = save_hist_len;
        b->turn = save_turn; b->ep = save_ep; b->hash = save_hash; b->hm = (uint8_t)save_hm;
        if (null_score >= beta) return beta;
    }

    /* ProbCut ────────────────────────────────────────────────────
     * At high depths, do a shallow search with a tighter beta.
     * If it fails high there it almost certainly fails high at
     * full depth — prune the node entirely.
     *
     * Conditions mirror Stockfish-lite practice:
     *   • depth >= 5  (not worth it below)
     *   • not in check (static_eval must be meaningful)
     *   • not PV (we don't prune PV nodes)
     *   • not near mate
     *   • enabled in endgames too (null move is disabled there,
     *     so ProbCut is the only forward-pruning in endgames) */
    if (!in_check && !is_pv && depth >= 5 && beta < 18000 && ply > 0) {
        int pc_beta  = beta + g_tune.probcut_margin;
        int pc_depth = depth - 4;   /* shallow probe: depth-4, min 1 */
        if (pc_depth < 1) pc_depth = 1;

        /* Generate captures + promotions only */
        Move pc_moves[MAX_MOVES];
        int  pc_n = board_gen_captures(b, pc_moves);
        /* Score and sort (reuse existing move scorer) */
        int pc_ok = b->turn == COL_W ? b->bk : b->wk;
        for (int pi = 0; pi < pc_n; pi++)
            pc_moves[pi].score = score_move(ss, &pc_moves[pi], b, ply,
                                            NULL, pc_ok, -1, -1, -1);
        for (int pi = 1; pi < pc_n; pi++) {
            Move tmp_m = pc_moves[pi]; int pj = pi-1;
            while (pj >= 0 && pc_moves[pj].score < tmp_m.score)
                { pc_moves[pj+1] = pc_moves[pj]; pj--; }
            pc_moves[pj+1] = tmp_m;
        }

        for (int pi = 0; pi < pc_n; pi++) {
            Move *pm = &pc_moves[pi];
            /* Skip bad captures (SEE < 0 relative to pc_beta margin) */
            if (!pm->prom && !pm->epc) {
                int sv = see_board(b, pm->from, pm->to, 0);
                if (sv < pc_beta - static_eval) continue;
            }

            board_make(b, pm);
            /* Legality */
            int mover_col_pc = b->turn ^ 24;
            int king_sq_pc   = mover_col_pc == COL_W ? b->wk : b->bk;
            if (board_is_attacked(b, king_sq_pc, b->turn)) { board_unmake(b); continue; }

            /* Quick qsearch verification, then shallow alpha_beta */
            int pc_sc = -qsearch(ss, b, -pc_beta, -pc_beta+1, ply+1);
            if (pc_sc >= pc_beta) {
                Move pc_pv[MAX_PLY]; int pc_len = 0;
                pc_sc = -alpha_beta(ss, b, pc_depth, -pc_beta, -pc_beta+1,
                                    pc_pv, &pc_len, ply+1, -1);
            }
            board_unmake(b);

            if (pc_sc >= pc_beta) return pc_sc;
        }
    }

    /* ── IIR (Internal Iterative Reduction) ───────────────────────
     * If we have no TT move (pv_move is null) at a deep node, the move
     * ordering will be poor.  Rather than wasting time on a full-depth
     * search with bad ordering, reduce depth by 1.  The shallow search
     * will populate the TT with a move for next time.  Conditions:
     *   depth >= 4, not in check, reduced depth still >= 2 */
    if (!pv_move.from && !pv_move.to && depth>=4 && !in_check && depth-1>=2) depth--;

    /* ── Singular Extension ──────────────────────────────────────
     * At deep nodes (depth >= 7) where the TT move is clearly best,
     * extend the search by 1 ply to resolve the position more accurately.
     *
     * Test: re-search the position at half depth, excluding the TT move,
     * with a window centred on tte.score - depth*2.  If ALL other moves
     * score below s_beta, the TT move is "singular" → extend it.
     *
     * If the re-search shows other moves are also good (s_beta >= beta),
     * this position is likely a "multi-cut" → prune with s_beta.
     *
     * sing_from/sing_to arrays track which move to exclude (prevents
     * infinite recursion in the singular search). */
    int sing_ext = 0;
    if (!in_check && depth>=7 && tte_hit && tte.depth>=depth-4 &&
        ss->sing_from[ply]<0 && ply>0 && (tte.flag==TT_EXACT||tte.flag==TT_LOWER)) {
        int s_beta  = tte.score - depth*2; if (s_beta < -18000) s_beta = -18000;
        int s_depth = (depth>>1)-1; if (s_depth < 1) s_depth = 1;
        ss->sing_from[ply] = (int8_t)pv_move.from;
        ss->sing_to  [ply] = (int8_t)pv_move.to;
        Move sp_pv[MAX_PLY]; int sp_len=0;
        int se_score = alpha_beta(ss, b, s_depth, s_beta-1, s_beta, sp_pv, &sp_len, ply, -1);
        ss->sing_from[ply] = -1; ss->sing_to[ply] = -1;
        if (se_score < s_beta) sing_ext = 1;    /* TT move is singular → extend */
        else if (s_beta >= beta) return s_beta;  /* multi-cut → prune */
    }

    /* ── Staged move generation (v310) ──────────────────────────────
     * Instead of generating all moves upfront, generate in stages:
     *   1. TT/PV move (no generation needed)
     *   2. Captures + promotions (board_gen_captures)
     *   3. Killers + counter move
     *   4. Quiet moves (board_gen_quiets)
     *   5. Losing captures (deferred from stage 2)
     * Score moves using the same score_move function as v309. */
    int ok_sq = b->turn==COL_W ? b->bk : b->wk;
    int cur_prev_ft = ply > 0 ? ss->prev_ft[ply-1] : -1;
    int cmh0 = ply >= 1 ? ss->prev_to_sq[ply-1] : -1;
    int cmh1 = ply >= 2 ? ss->prev_to_sq[ply-2] : -1;
    const Move *pv_ptr = (pv_move.from||pv_move.to) ? &pv_move : NULL;

    /* LMP limits — not tunable (see ga_tune.c's header comment "PARAMETERS
     * NOT TUNED, AND WHY": an 8-entry hand-shaped table doesn't reduce to
     * one or two scalars the way the futility margin does). */
    static const int lmp_limit[8] = {0,10,18,26,36,48,62,78};
    int fut_adj = improving ? 0 : g_tune.fut_improving_adj;
    /* fut_base(d) = g_tune.fut_mult * d — the original hardcoded table
     * {0,150,300,450,600,750,900,1050,1200} IS exactly 150*d for d=0..8,
     * so this is not an approximation, just the same table parameterized
     * by its one degree of freedom. */
    int fut_base[9];
    for (int fbi = 0; fbi < 9; fbi++) fut_base[fbi] = g_tune.fut_mult * fbi;

    int best = -99999, flag = TT_UPPER;
    Move best_move = {0};
    int legal_count = 0, quiet_count = 0;
    Move child_pv[MAX_PLY]; int child_len = 0;

    /* Track quiet moves searched for history penalty */
    Move searched_quiets[64];
    int n_searched_quiets = 0;

    /* ── STAGE 1: TT/PV Move ────────────────────────────────────── */
    int tt_tried = 0;
    if (pv_ptr) {
        Move *m = (Move *)pv_ptr;  /* safe: we only read */
        int mfr = m->from, mto = m->to;
        /* SMP guard: lockless TT can produce corrupt moves.
         * Validate that from-square has a piece of the side-to-move
         * to prevent P2BI[0]=-1 → b->bb[-1] crash in board_make. */
        uint8_t pp = b->b[mfr];
        if (pp && PC_COLOR(pp) == b->turn && m->castle <= 4) {
        int is_capture = !!(b->b[mto] || m->epc);
        int is_promo   = !!m->prom;
        int is_castle  = !!m->castle;
        int is_quiet   = !is_capture && !is_promo && !is_castle;

        /* Skip singular move */
        if (!(ss->sing_from[ply]>=0 && mfr==ss->sing_from[ply] && mto==ss->sing_to[ply])) {
            board_make(b, m);
            __builtin_prefetch(&tt->H[((b->hash ^ ZR_side) & tt->mask) * TT_BUCKETS], 0, 1);
            ss->prev_ft[ply]    = mfr*64 + mto;
            ss->prev_to_sq[ply] = mto;

            int mover_col = b->turn ^ 24;
            int king_sq   = mover_col == COL_W ? b->wk : b->bk;
            if (!board_is_attacked(b, king_sq, b->turn)) {
                /* Multi-PV: skip excluded root moves */
                if (ply == 0 && ss->excluded_root_n > 0 && is_excluded_root(ss, m)) {
                    board_unmake(b);
                } else {
                    tt_tried = 1;
                    legal_count++;

                    int gives_check = 0;
                    { uint8_t gpt=b->b[m->to]&7,gksq=b->turn==COL_W?b->wk:b->bk;
                      int gdr=((m->to>>3)-(gksq>>3)); if(gdr<0)gdr=-gdr;
                      int gdc=((m->to&7)-(gksq&7));   if(gdc<0)gdc=-gdc;
                      if (gpt>=3||gpt==2||m->prom||(gdr>gdc?gdr:gdc)<=2)
                          gives_check = board_in_check(b);
                    }

                    int check_ext = 0;
                    if (in_check && depth==1) check_ext=1;
                    if (sing_ext) check_ext = check_ext>1?check_ext:1;
                    if (!check_ext && !in_check) {
                        uint8_t pc = b->b[m->to];
                        if (pc && PC_TYPE(pc)==1) {
                            int to_rank = m->to >> 3;
                            if ((PC_COLOR(pc)==COL_W && to_rank==1) ||
                                (PC_COLOR(pc)==COL_B && to_rank==6))
                                check_ext = 1;
                        }
                    }

                    int sc;
                    child_len = 0;
                    sc = -alpha_beta(ss, b, depth-1+check_ext, -beta, -alpha, child_pv, &child_len, ply+1, gives_check);
                    board_unmake(b);

                    if (is_quiet && n_searched_quiets < 64 && sc <= alpha)
                        searched_quiets[n_searched_quiets++] = *m;

                    if (sc > best) {
                        best = sc; best_move = *m;
                        if (sc > alpha) {
                            alpha = sc; flag = TT_EXACT;
                            pv[0] = *m;
                            memcpy(pv+1, child_pv, child_len * sizeof(Move));
                            *pv_len = child_len + 1;
                        }
                    }
                    if (alpha >= beta) goto cutoff;
                }
            } else {
                board_unmake(b);
            }
        }
        }  /* SMP guard: pp && PC_COLOR(pp) == b->turn */
    }

    /* ── STAGE 2: Generate captures + promotions, score, pick-best ── */
    {
        Move caps[MAX_MOVES];
        int n_caps = board_gen_captures(b, caps);
        /* Score captures */
        for (int i = 0; i < n_caps; i++)
            caps[i].score = score_move(ss, &caps[i], b, ply, pv_ptr, ok_sq,
                                       cur_prev_ft, cmh0, cmh1);

        /* Winning/equal captures */
        Move bad_caps[MAX_MOVES];
        int n_bad = 0;

        for (int i = 0; i < n_caps; i++) {
            /* Pick-best */
            int best_idx = i;
            for (int j = i+1; j < n_caps; j++)
                if (caps[j].score > caps[best_idx].score) best_idx = j;
            if (best_idx != i) {
                Move tmp = caps[i]; caps[i] = caps[best_idx]; caps[best_idx] = tmp;
            }

            Move *m = &caps[i];
            int mfr = m->from, mto = m->to;
            /* Skip if this is the TT move (already tried) */
            if (tt_tried && pv_ptr && mfr==pv_ptr->from && mto==pv_ptr->to && m->prom==pv_ptr->prom)
                continue;

            int is_capture = !!(b->b[mto] || m->epc);
            int is_promo   = !!m->prom;

            /* Defer losing captures */
            if (is_capture && !is_promo && m->score >= 100000 && m->score <= 200000) {
                bad_caps[n_bad++] = *m;
                continue;
            }

            /* SEE pruning on losing captures */
            if (!in_check && is_capture && !is_promo && legal_count>0 && depth>2 && !is_pv) {
                if (m->score < 1600000) {
                    int sv = (m->score - 200000) / 2;
                    int see_thresh = depth<=4 ? -80 : depth<=6 ? -120 : -160;
                    if (sv < see_thresh) {
                        int tr=mto>>3,tc=mto&7,kr=ok_sq>>3,kc=ok_sq&7;
                        int dr=tr-kr;if(dr<0)dr=-dr;
                        int dc=tc-kc;if(dc<0)dc=-dc;
                        if ((dr>dc?dr:dc) > 1) continue;
                    }
                }
            }

            /* Skip singular move */
            if (ss->sing_from[ply]>=0 && mfr==ss->sing_from[ply] && mto==ss->sing_to[ply]) continue;

            board_make(b, m);
            __builtin_prefetch(&tt->H[((b->hash ^ ZR_side) & tt->mask) * TT_BUCKETS], 0, 1);
            ss->prev_ft[ply]    = mfr*64 + mto;
            ss->prev_to_sq[ply] = mto;

            int mover_col = b->turn ^ 24;
            int king_sq   = mover_col == COL_W ? b->wk : b->bk;
            if (board_is_attacked(b, king_sq, b->turn)) { board_unmake(b); continue; }

            if (ply == 0 && ss->excluded_root_n > 0 && is_excluded_root(ss, m)) {
                board_unmake(b); continue;
            }

            legal_count++;

            int gives_check = 0;
            { uint8_t gpt=b->b[m->to]&7,gksq=b->turn==COL_W?b->wk:b->bk;
              int gdr=((m->to>>3)-(gksq>>3)); if(gdr<0)gdr=-gdr;
              int gdc=((m->to&7)-(gksq&7));   if(gdc<0)gdc=-gdc;
              if (gpt>=3||gpt==2||m->prom||(gdr>gdc?gdr:gdc)<=2)
                  gives_check = board_in_check(b);
            }

            int check_ext = 0;
            if (in_check && depth==1) check_ext=1;
            if (sing_ext && pv_move.from==m->from && pv_move.to==m->to)
                check_ext = check_ext>1?check_ext:1;
            if (!check_ext && !in_check) {
                uint8_t pc = b->b[m->to];
                if (pc && PC_TYPE(pc)==1) {
                    int to_rank = m->to >> 3;
                    if ((PC_COLOR(pc)==COL_W && to_rank==1) ||
                        (PC_COLOR(pc)==COL_B && to_rank==6))
                        check_ext = 1;
                }
            }

            int sc;
            child_len = 0;
            if (legal_count == 1) {
                sc = -alpha_beta(ss, b, depth-1+check_ext, -beta, -alpha, child_pv, &child_len, ply+1, gives_check);
            } else {
                int reduce = 0;
                /* LMR for losing captures */
                if (is_capture && !is_promo && depth>=3 && legal_count>=4 && !in_check) {
                    if (m->score >= 100000 && m->score <= 200000) reduce = 1;
                }
                Move null_pv[MAX_PLY]; int null_len=0;
                sc = -alpha_beta(ss, b, depth-1-reduce+check_ext, -alpha-1, -alpha, null_pv, &null_len, ply+1, gives_check);
                if (sc > alpha && reduce > 0) {
                    sc = -alpha_beta(ss, b, depth-1+check_ext, -alpha-1, -alpha, null_pv, &null_len, ply+1, gives_check);
                }
                if (sc > alpha && sc < beta) {
                    sc = -alpha_beta(ss, b, depth-1+check_ext, -beta, -alpha, child_pv, &child_len, ply+1, gives_check);
                }
            }
            board_unmake(b);

            if (sc > best) {
                best = sc; best_move = *m;
                if (sc > alpha) {
                    alpha = sc; flag = TT_EXACT;
                    pv[0] = *m;
                    memcpy(pv+1, child_pv, child_len * sizeof(Move));
                    *pv_len = child_len + 1;
                }
            }
            if (alpha >= beta) goto cutoff;
        }

        /* ── STAGE 3: Quiet moves ────────────────────────────────── */
        {
            Move quiets[MAX_MOVES];
            int n_quiets = board_gen_quiets(b, quiets);
            /* Score quiets using the same scorer */
            for (int i = 0; i < n_quiets; i++)
                quiets[i].score = score_move(ss, &quiets[i], b, ply, pv_ptr, ok_sq,
                                             cur_prev_ft, cmh0, cmh1);

            for (int i = 0; i < n_quiets; i++) {
                /* Pick-best */
                int best_idx = i;
                for (int j = i+1; j < n_quiets; j++)
                    if (quiets[j].score > quiets[best_idx].score) best_idx = j;
                if (best_idx != i) {
                    Move tmp = quiets[i]; quiets[i] = quiets[best_idx]; quiets[best_idx] = tmp;
                }

                Move *m = &quiets[i];
                int mfr = m->from, mto = m->to;
                /* Skip TT move (already tried) */
                if (tt_tried && pv_ptr && mfr==pv_ptr->from && mto==pv_ptr->to && m->prom==pv_ptr->prom)
                    continue;

                int is_castle  = !!m->castle;
                int is_quiet   = !is_castle;  /* quiets are never captures */

                /* LMP */
                int is_killer = is_quiet && (
                    (ss->killers[ply][0].from==mfr&&ss->killers[ply][0].to==mto&&(ss->killers[ply][0].from||ss->killers[ply][0].to)) ||
                    (ss->killers[ply][1].from==mfr&&ss->killers[ply][1].to==mto&&(ss->killers[ply][1].from||ss->killers[ply][1].to)));
                if (!in_check && is_quiet && depth<=7 && legal_count>0 && !is_killer) {
                    quiet_count++;
                    int lmp_lim = lmp_limit[depth<8?depth:7];
                    if (!improving) lmp_lim = (lmp_lim + 1) / 2;
                    if (quiet_count > lmp_lim) continue;

                    /* History pruning */
                    if (depth <= 4) {
                        int ft_hp = mfr * 64 + mto;
                        int ch_hp = ss->mv_history[ft_hp];
                        if (cmh0 >= 0) ch_hp += ss->cont_hist[0][cmh0][ft_hp];
                        if (cmh1 >= 0) ch_hp += ss->cont_hist[1][cmh1][ft_hp];
                        int hp_thresh = -4000 * depth;
                        if (ch_hp < hp_thresh) continue;
                    }
                }

                /* Skip singular move */
                if (ss->sing_from[ply]>=0 && mfr==ss->sing_from[ply] && mto==ss->sing_to[ply]) continue;

                board_make(b, m);
                __builtin_prefetch(&tt->H[((b->hash ^ ZR_side) & tt->mask) * TT_BUCKETS], 0, 1);
                ss->prev_ft[ply]    = mfr*64 + mto;
                ss->prev_to_sq[ply] = mto;

                int mover_col = b->turn ^ 24;
                int king_sq   = mover_col == COL_W ? b->wk : b->bk;
                if (board_is_attacked(b, king_sq, b->turn)) { board_unmake(b); continue; }

                if (ply == 0 && ss->excluded_root_n > 0 && is_excluded_root(ss, m)) {
                    board_unmake(b); continue;
                }

                legal_count++;

                int gives_check = 0;
                { uint8_t gpt=b->b[m->to]&7,gksq=b->turn==COL_W?b->wk:b->bk;
                  int gdr=((m->to>>3)-(gksq>>3)); if(gdr<0)gdr=-gdr;
                  int gdc=((m->to&7)-(gksq&7));   if(gdc<0)gdc=-gdc;
                  if (gpt>=3||gpt==2||m->prom||(gdr>gdc?gdr:gdc)<=2)
                      gives_check = board_in_check(b);
                }

                /* Futility pruning */
                if (!in_check && is_quiet && !gives_check && depth>=1 && depth<=8 && legal_count>1) {
                    int fd = depth < 9 ? depth : 8;
                    if (static_eval + fut_base[fd] + fut_adj <= alpha) { board_unmake(b); continue; }
                }

                int check_ext = 0;
                if (in_check && depth==1) check_ext=1;
                if (sing_ext && pv_move.from==m->from && pv_move.to==m->to)
                    check_ext = check_ext>1?check_ext:1;
                if (!check_ext && !in_check) {
                    uint8_t pc = b->b[m->to];
                    if (pc && PC_TYPE(pc)==1) {
                        int to_rank = m->to >> 3;
                        if ((PC_COLOR(pc)==COL_W && to_rank==1) ||
                            (PC_COLOR(pc)==COL_B && to_rank==6))
                            check_ext = 1;
                    }
                }

                int sc;
                child_len = 0;
                if (legal_count == 1) {
                    sc = -alpha_beta(ss, b, depth-1+check_ext, -beta, -alpha, child_pv, &child_len, ply+1, gives_check);
                } else {
                    int reduce = 0;
                    if (legal_count>=3 && depth>=3 && !in_check && !gives_check && is_quiet && !is_killer) {
                        int ld = depth<LMR_D?depth:LMR_D-1;
                        int lm = legal_count<LMR_M?legal_count:LMR_M-1;
                        /* Computed live from g_tune.lmr_divisor rather than a
                         * precomputed lmr_tab[] lookup — see search.h's
                         * SearchTunables comment: lmr_tab is built once by
                         * search_init() on the main thread only, so a
                         * per-thread tuner could never see its own divisor
                         * reflected in it. Same formula/clamps that used to
                         * build the table (search_init() below), just
                         * evaluated per-node so it tracks THIS thread's
                         * g_tune. */
                        {
                            double lv = log((double)(ld>0?ld:1)) * log((double)(lm>0?lm:1)) / g_tune.lmr_divisor;
                            int lr = (int)lv;
                            if (lr < 1) lr = 1;
                            if (lr > ld-1) lr = ld-1;
                            reduce = (uint8_t)lr;
                        }
                        if (is_pv) reduce = reduce>0?reduce-1:0;
                        if (!improving) reduce += 1;
                        {
                            int ft_idx = mfr * 64 + mto;
                            int ch = ss->mv_history[ft_idx];
                            if (cmh0 >= 0) ch += ss->cont_hist[0][cmh0][ft_idx];
                            if (cmh1 >= 0) ch += ss->cont_hist[1][cmh1][ft_idx];
                            if (ch < -4000) reduce += 1;
                            if (ch < -8000) reduce += 1;
                            if (ch > 4000 && reduce > 0) reduce -= 1;
                        }
                        if (reduce >= depth - 1) reduce = depth - 2;
                        if (reduce < 0) reduce = 0;
                    }
                    Move null_pv[MAX_PLY]; int null_len=0;
                    sc = -alpha_beta(ss, b, depth-1-reduce+check_ext, -alpha-1, -alpha, null_pv, &null_len, ply+1, gives_check);
                    if (sc > alpha && reduce > 0) {
                        sc = -alpha_beta(ss, b, depth-1+check_ext, -alpha-1, -alpha, null_pv, &null_len, ply+1, gives_check);
                    }
                    if (sc > alpha && sc < beta) {
                        sc = -alpha_beta(ss, b, depth-1+check_ext, -beta, -alpha, child_pv, &child_len, ply+1, gives_check);
                    }
                }
                board_unmake(b);

                if (is_quiet && n_searched_quiets < 64 && sc <= alpha)
                    searched_quiets[n_searched_quiets++] = *m;

                if (sc > best) {
                    best = sc; best_move = *m;
                    if (sc > alpha) {
                        alpha = sc; flag = TT_EXACT;
                        pv[0] = *m;
                        memcpy(pv+1, child_pv, child_len * sizeof(Move));
                        *pv_len = child_len + 1;
                    }
                }
                if (alpha >= beta) goto cutoff;
            }
        }

        /* ── STAGE 4: Losing captures ────────────────────────────── */
        for (int i = 0; i < n_bad; i++) {
            Move *m = &bad_caps[i];
            int mfr = m->from, mto = m->to;
            /* Skip TT move */
            if (tt_tried && pv_ptr && mfr==pv_ptr->from && mto==pv_ptr->to && m->prom==pv_ptr->prom)
                continue;

            int is_promo = !!m->prom;

            /* SEE pruning */
            if (!in_check && !is_promo && legal_count>0 && depth>2 && !is_pv) {
                int sv = (m->score - 200000) / 2;
                int see_thresh = depth<=4 ? -80 : depth<=6 ? -120 : -160;
                if (sv < see_thresh) {
                    int tr=mto>>3,tc=mto&7,kr=ok_sq>>3,kc=ok_sq&7;
                    int dr=tr-kr;if(dr<0)dr=-dr;
                    int dc=tc-kc;if(dc<0)dc=-dc;
                    if ((dr>dc?dr:dc) > 1) continue;
                }
            }

            /* Skip singular move */
            if (ss->sing_from[ply]>=0 && mfr==ss->sing_from[ply] && mto==ss->sing_to[ply]) continue;

            board_make(b, m);
            __builtin_prefetch(&tt->H[((b->hash ^ ZR_side) & tt->mask) * TT_BUCKETS], 0, 1);
            ss->prev_ft[ply]    = mfr*64 + mto;
            ss->prev_to_sq[ply] = mto;

            int mover_col = b->turn ^ 24;
            int king_sq   = mover_col == COL_W ? b->wk : b->bk;
            if (board_is_attacked(b, king_sq, b->turn)) { board_unmake(b); continue; }

            if (ply == 0 && ss->excluded_root_n > 0 && is_excluded_root(ss, m)) {
                board_unmake(b); continue;
            }

            legal_count++;

            int gives_check = 0;
            { uint8_t gpt=b->b[m->to]&7,gksq=b->turn==COL_W?b->wk:b->bk;
              int gdr=((m->to>>3)-(gksq>>3)); if(gdr<0)gdr=-gdr;
              int gdc=((m->to&7)-(gksq&7));   if(gdc<0)gdc=-gdc;
              if (gpt>=3||gpt==2||m->prom||(gdr>gdc?gdr:gdc)<=2)
                  gives_check = board_in_check(b);
            }

            int check_ext = 0;
            if (in_check && depth==1) check_ext=1;
            if (sing_ext && pv_move.from==m->from && pv_move.to==m->to)
                check_ext = check_ext>1?check_ext:1;

            int sc;
            child_len = 0;
            if (legal_count == 1) {
                sc = -alpha_beta(ss, b, depth-1+check_ext, -beta, -alpha, child_pv, &child_len, ply+1, gives_check);
            } else {
                int reduce = 0;
                if (!is_promo && depth>=3 && legal_count>=4 && !in_check) {
                    if (m->score >= 100000 && m->score <= 200000) reduce = 1;
                }
                Move null_pv[MAX_PLY]; int null_len=0;
                sc = -alpha_beta(ss, b, depth-1-reduce+check_ext, -alpha-1, -alpha, null_pv, &null_len, ply+1, gives_check);
                if (sc > alpha && reduce > 0)
                    sc = -alpha_beta(ss, b, depth-1+check_ext, -alpha-1, -alpha, null_pv, &null_len, ply+1, gives_check);
                if (sc > alpha && sc < beta)
                    sc = -alpha_beta(ss, b, depth-1+check_ext, -beta, -alpha, child_pv, &child_len, ply+1, gives_check);
            }
            board_unmake(b);

            if (sc > best) {
                best = sc; best_move = *m;
                if (sc > alpha) {
                    alpha = sc; flag = TT_EXACT;
                    pv[0] = *m;
                    memcpy(pv+1, child_pv, child_len * sizeof(Move));
                    *pv_len = child_len + 1;
                }
            }
            if (alpha >= beta) goto cutoff;
        }
    }

cutoff:
    /* ── History / Killer / Counter-move updates on beta cutoff ────
     *
     * When a quiet move causes a beta cutoff, we:
     *   1. Store it as a killer (2 slots per ply, FIFO replacement)
     *   2. Store it as a counter move (indexed by previous from-to)
     *   3. Increase its history score (butterfly + CMH) by depth²
     *   4. Decrease history for all other quiets tried before it
     *
     * This "bonus/malus" scheme teaches the engine which quiet moves
     * are good in which contexts (history = butterfly table, CMH =
     * continuation history indexed by the move 1 and 2 plies prior).
     *
     * Bonus/malus are clamped to ±16384 to prevent saturation. */
    if (alpha >= beta && (best_move.from || best_move.to)) {
        int mfr = best_move.from, mto = best_move.to;
        int is_capture = !!(b->b[mto] || best_move.epc);
        int is_promo   = !!best_move.prom;
        int is_castle  = !!best_move.castle;
        int is_quiet   = !is_capture && !is_promo && !is_castle;

        if (is_quiet) {
            /* Killer moves: 2 slots per ply.  Slot 0 is most recent;
             * displaced killer moves to slot 1 (FIFO). */
            int km0_set = ss->killers[ply][0].from||ss->killers[ply][0].to;
            if (!km0_set || ss->killers[ply][0].from!=mfr || ss->killers[ply][0].to!=mto) {
                ss->killers[ply][1] = ss->killers[ply][0];
                ss->killers[ply][0] = best_move;
            }
            /* Counter move: "after opponent played X, this move cuts" */
            if (cur_prev_ft >= 0) ss->counter_move[cur_prev_ft] = mfr*64+mto;

            int bonus = depth*depth;  /* quadratic bonus: deeper cutoffs get more reward */
            if (bonus > 16384) bonus = 16384;

            /* Reward the cutoff move in all history tables */
            int ft = mfr*64+mto;
            {   int h = ss->mv_history[ft] + bonus;
                ss->mv_history[ft] = h < 16384 ? h : 16384; }
            if (cmh0 >= 0) {
                int c = ss->cont_hist[0][cmh0][ft] + bonus;
                ss->cont_hist[0][cmh0][ft] = (int16_t)(c < 16384 ? c : 16384);
            }
            if (cmh1 >= 0) {
                int c = ss->cont_hist[1][cmh1][ft] + bonus;
                ss->cont_hist[1][cmh1][ft] = (int16_t)(c < 16384 ? c : 16384);
            }

            /* Penalise all quiet moves that were searched before the cutoff
             * move but failed to produce a cutoff themselves.  This teaches
             * the engine that these moves are relatively weaker in this context. */
            for (int j = 0; j < n_searched_quiets; j++) {
                Move *pm = &searched_quiets[j];
                if (pm->from == mfr && pm->to == mto) continue;
                int pft = pm->from*64 + pm->to;
                {   int h = ss->mv_history[pft] - bonus;
                    ss->mv_history[pft] = h > -16384 ? h : -16384; }
                if (cmh0 >= 0) {
                    int c = ss->cont_hist[0][cmh0][pft] - bonus;
                    ss->cont_hist[0][cmh0][pft] = (int16_t)(c > -16384 ? c : -16384);
                }
                if (cmh1 >= 0) {
                    int c = ss->cont_hist[1][cmh1][pft] - bonus;
                    ss->cont_hist[1][cmh1][pft] = (int16_t)(c > -16384 ? c : -16384);
                }
            }
        }
        flag = TT_LOWER;  /* beta cutoff → score is a lower bound */
    }

    /* No legal moves: checkmate (in check) or stalemate (not in check) */
    if (!legal_count) return in_check ? (-19000+ply) : 0;
    /* Store best result in TT for future visits */
    if (best_move.from||best_move.to)
        tt_store(tt, b->hash, best, depth, flag, &best_move, ply, raw_eval);
    return best;
}

/* ── Iterative deepening + aspiration ────────────────────────── */

/* One-time search initialization: build lookup tables.
 * Must be called once at startup before any search. */
void search_init(void) {
    /* LMR (Late Move Reduction) table: lmr_tab[depth][move_count].
     * Formula: R = ln(d) * ln(m) / 1.5  (Stockfish-style logarithmic).
     * Divisor 1.5 gives moderate reductions; lower = more aggressive.
     * Clamped: min 1 (always reduce at least 1), max depth-1 (never
     * reduce to depth 0 — that would skip to qsearch prematurely). */
    for (int d = 1; d < LMR_D; d++)
        for (int m = 1; m < LMR_M; m++) {
            double v = log((double)d) * log((double)m) / 1.5;
            int r = (int)v;
            if (r < 1) r = 1;
            if (r > d-1) r = d-1;
            lmr_tab[d*LMR_M+m] = (uint8_t)r;
        }
    /* MVV-LVA (Most Valuable Victim – Least Valuable Attacker) table.
     * Score = (6-attacker_type) + victim_type*10.  Higher score for
     * captures of valuable pieces by cheap pieces (PxQ > QxQ). */
    for (int v = 1; v <= 5; v++)
        for (int a = 1; a <= 6; a++)
            MVV_LVA[v*7+a] = (6-a) + v*10;
    /* Allocate the process-wide default TT (v4.00: per-instance TTable).
     * tt_create() zero-initialises H/S/D/G/M (calloc) and sets every E[i]
     * to TT_EVAL_NONE — identical to the old memset(TT_H,...) + E-loop. */
    if (!g_tt) {
        g_tt = tt_create(TT_SIZE);
        if (!g_tt) {
            /* OOM at startup: fail loudly instead of leaving g_tt NULL,
             * which would crash inside tt_probe/tt_store on first search
             * with a confusing null-deref instead of a clear message. */
            fprintf(stderr, "[TT] fatal: tt_create(%d entries) failed (out of memory)\n", TT_SIZE);
            exit(1);
        }
    }
}

void search_history_clear(SearchState *s) {
    memset(s->mv_history,   0, sizeof(s->mv_history));
    memset(s->counter_move, 0, sizeof(s->counter_move));
    memset(s->cont_hist,    0, sizeof(s->cont_hist));
}

/* See search.h for why this exists alongside search_reset(). */
void search_clear_ordering(SearchState *ss) {
    memset(ss->killers,      0, sizeof(ss->killers));
    memset(ss->mv_history,   0, sizeof(ss->mv_history));
    memset(ss->counter_move, 0, sizeof(ss->counter_move));
    memset(ss->cont_hist,    0, sizeof(ss->cont_hist));
}

void search_reset(SearchState *ss) {
    memset(ss->killers,      0, sizeof(ss->killers));
    memset(ss->sing_from,   -1, sizeof(ss->sing_from));
    memset(ss->sing_to,     -1, sizeof(ss->sing_to));
    for (int i = 0; i < MAX_PLY; i++) {
        ss->prev_ft[i] = -1;
        ss->prev_to_sq[i] = -1;
        ss->prev_static_eval[i] = TT_EVAL_NONE;
    }
    /* Age history tables: divide by 4 to preserve relative ordering
     * while preventing saturation across games/iterations.
     * Loop is ~4K iterations for mv_history and ~512K for ss->cont_hist[2];
     * compiler will auto-vectorise with -O3 -march=native.            */
    for (int i = 0; i < 64*64; i++) ss->mv_history[i] >>= 2;
    memset(ss->counter_move, 0, sizeof(ss->counter_move));
    /* Age both CMH slots (int16_t arithmetic right-shift → signed divide by 4) */
    for (int s = 0; s < 2; s++)
        for (int i = 0; i < 64; i++)
            for (int j = 0; j < 64*64; j++)
                ss->cont_hist[s][i][j] >>= 2;
}

/* ── Per-thread SearchState management ───────────────────────── */

SearchState *search_state_new(void) {
    SearchState *s = (SearchState *)calloc(1, sizeof(SearchState));
    if (s) {
        memset(s->sing_from, -1, sizeof(s->sing_from));
        memset(s->sing_to,   -1, sizeof(s->sing_to));
        for (int i = 0; i < MAX_PLY; i++) {
            s->prev_ft[i] = -1;
            s->prev_to_sq[i] = -1;
            s->prev_static_eval[i] = TT_EVAL_NONE;
        }
    }
    return s;
}

void search_state_free(SearchState *state) {
    free(state);
}

/* is_excluded_root — definition (forward-declared before alpha_beta) */
static int is_excluded_root(SearchState *ss, const Move *m) {
    for (int i = 0; i < ss->excluded_root_n; i++)
        if (ss->excluded_root[i].from == m->from && ss->excluded_root[i].to == m->to &&
            ss->excluded_root[i].prom == m->prom)
            return 1;
    return 0;
}

SearchResult search_best(Board *b, const SearchParams *p) {
    /* Each thread needs its own SearchState. Helpers pass theirs via
     * p->search_state; main thread uses the default global g_ss. */
    SearchState *ss = p->search_state ? p->search_state : &g_ss;
    SearchResult res = {0};
    /* Per-instance TT (v4.00): use the caller-supplied table, or fall back
     * to the process-wide default.  Lazy SMP helpers receive the SAME
     * pointer as the main thread here (SearchParams is struct-copied in
     * main.c's search_thread_fn, so p->tt is identical for every helper
     * and the main thread) — sharing the TT across helpers is unchanged. */
    ss->tt = p->tt ? p->tt : g_tt;
    /* Only main thread increments TT generation — helpers share it */
    if (p->start_depth <= 1) tt_new_generation(ss->tt);
    nnue_reset(b->nnue);  /* v3.13: per-thread accumulator reset */

    int md = p->max_depth;
    int n_pvs = p->multi_pv;
    if (n_pvs < 1) n_pvs = 1;
    if (n_pvs > MAX_MULTI_PV) n_pvs = MAX_MULTI_PV;

    /* Node limit */
    if (p->time_limit_ms > 0) {
        ss->node_limit = p->node_limit > 0 ? p->node_limit : (long)2e9;
    } else if (p->node_limit > 0) {
        ss->node_limit = p->node_limit;
    } else {
        ss->node_limit = (long)2e9;
    }

    ss->deadline_ms = p->time_limit_ms > 0 ? now_ms() + p->time_limit_ms : 0;
    ss->time_up     = 0;
    ss->stop_guard  = 0;   /* per-thread state is reused across searches */
    ss->stop_flag   = p->stop;  /* may be NULL */
    ss->nodes       = 0;
    ss->nodes_total = 0;
    ss->tb_hits     = 0;

    search_reset(ss);
    *b->undo_top = 0;  /* reset undo stack for this search */

    /* Rebuild NNUE accumulator */
    if (nnue_ready()) nnue_rebuild(b->nnue, b->b);  /* v3.13: per-thread */

    /* ══════════════════════════════════════════════════════════════
     * ROOT SYZYGY TB MOVE FILTERING  (Stockfish-style)
     * ══════════════════════════════════════════════════════════════
     *
     * OVERVIEW:
     * At the root, instead of returning a TB score immediately (which
     * loses ~80 ELO because the engine can't distinguish between
     * equally-winning moves), we FILTER root moves by WDL class:
     *   1. Probe root position → get baseline WDL
     *   2. For each legal root move: make move, probe child WDL
     *   3. Keep only moves that achieve the best possible WDL
     *   4. Exclude TB-losing moves by adding them to excluded_root[]
     *   5. Run normal iterative deepening on the surviving moves
     *
     * This lets the search use NNUE evaluation to pick the BEST move
     * among the TB-equal moves (e.g., which winning move has shortest DTZ).
     *
     * GUARDS:
     * - start_depth <= 1: only main thread does root filtering (helpers
     *   start at depth 2+ and should NOT re-filter)
     * - g_tb_probe_depth < 99: TB is enabled (99 = disabled sentinel)
     * - npieces >= 4 && <= g_tb_probe_limit: within loaded TB range
     *
     * WDL NEGATION:
     * Fathom returns WDL from STM's perspective.  After board_make(),
     * the child's WDL is from the OPPONENT's perspective.  We negate
     * via (4 - child_wdl) to get our perspective:
     *   child=0 (opp wins) → our_wdl=4 (we win)
     *   child=4 (opp loses) → our_wdl=0 (we lose)
     *
     * INTERACTION WITH MULTI-PV:
     * TB-excluded moves are added to excluded_root[] BEFORE the Multi-PV
     * loop.  Multi-PV then ALSO adds its own exclusions (previous best
     * moves).  The tb_excluded_base counter tracks where TB exclusions
     * end, so Multi-PV resets only its own entries between PV iterations.
     *
     * INTERACTION WITH NNUE:
     * board_make/unmake here call nnue_push/pop, so the accumulator
     * state is correctly saved/restored for each probed child. */
    int tb_root_filtered = 0;
    /* ROOT TB MOVE FILTERING DISABLED (v3.13.1)
     *
     * Root filtering was causing -150 ELO regression because:
     * 1. It probes ALL legal root moves via board_make + syzygy_probe_wdl,
     *    adding ~20-50ms overhead at root on every search iteration
     * 2. After the rule50 fix, child probes after quiet moves fail,
     *    and fallback assigns root_wdl to all quiet moves — making the
     *    filtering meaningless for most moves
     * 3. The correct Stockfish approach uses tb_probe_root_dtz() for root
     *    filtering, which is a dedicated Fathom API that returns per-move
     *    DTZ scores in a single call (not N separate probes)
     *
     * In-tree WDL probing (alpha_beta, ply > 0) with TT caching at
     * depth=127 is sufficient for TB knowledge to propagate through
     * the search tree.  Root move filtering is unnecessary when in-tree
     * probing works correctly.
     *
     * TODO: Implement proper root filtering using tb_probe_root_dtz()
     * if we want Stockfish-level root TB accuracy. */

    /* ══════════════════════════════════════════════════════════════
     * MULTI-PV OUTER LOOP
     * ══════════════════════════════════════════════════════════════
     *
     * HOW MULTI-PV WORKS:
     * For multi_pv=N, we run N complete iterative-deepening searches.
     * After each PV line finds its best move, that move is added to
     * excluded_root[].  The NEXT PV iteration's alpha_beta() checks
     * is_excluded_root() at ply==0 and skips excluded moves, forcing
     * the search to find the second-best, third-best, etc.
     *
     * DATA FLOW:
     *   PV 1: search all moves → best = e2e4   → exclude e2e4
     *   PV 2: search minus e2e4 → best = d2d4  → exclude d2d4
     *   PV 3: search minus e2e4,d2d4 → best = c2c4 → ...
     *
     * TB INTERACTION:
     * TB root filtering has already populated excluded_root[0..tb_excluded_base-1].
     * Multi-PV appends after that.  Between PV iterations, we reset
     * excluded_root_n back to tb_excluded_base (not 0!) so TB exclusions
     * persist across all PV lines.
     *
     * TT INTERACTION:
     * The TT cutoff at ply==0 is disabled when excluded_root_n > 0
     * (see TT probe above).  This prevents TT from returning the
     * cached score of PV line 1 when searching for PV line 2.
     *
     * LAZY SMP INTERACTION:
     * Helper threads always run multi_pv=1 (silent, single PV).
     * Only the main thread runs multi_pv > 1 with info_cb output.
     * Helpers contribute to the shared TT, improving move ordering
     * for the main thread's subsequent PV searches. */
    int tb_excluded_base = ss->excluded_root_n;
    /* Starting depth for iterative deepening (Lazy SMP helpers start higher) */
    int sd = p->start_depth > 1 ? p->start_depth : 1;

    /* v4.00 shared MultiPV budget (mpv_share_budget==1):
     * Divide the caller's total time_limit_ms evenly across the n_pvs
     * lines instead of giving each line its own full budget.  This is
     * design choice (a) — a per-line divided budget — rather than (b) a
     * single wall-clock deadline for the whole loop.  (b) was rejected:
     * with a single shared deadline, PV line 1 (searched first, deepest
     * move ordering) can burn the ENTIRE budget before line 2 even starts,
     * leaving later lines at time_up=1 with zero depth searched — exactly
     * the "MultiPV freeze" bug the existing per-PV reset above was written
     * to avoid, just reintroduced across lines instead of within one line.
     * That would hand the temperature sampler an unfinished, effectively
     * garbage score for every PV after the first.  Dividing the budget (a)
     * guarantees every line gets its own nonzero, bounded slice.
     *
     * Depth>=2 guarantee: ss->time_up is only ever set from inside
     * alpha_beta's node-count check (every 8192 nodes), never checked
     * before a depth begins, and depths 1-2 below always run as a full
     * (non-aspiration) search regardless of prev_score_valid.  A depth-1
     * or depth-2 search from the root visits far fewer than 8192 nodes in
     * practice, so as long as the per-line slice is > 0ms every PV line
     * completes at least depth 2 before time_up can possibly fire — even
     * under selfplay's 50ms/4-line = ~12ms slices.  This is not a special
     * case in the code below; it falls out of the existing per-depth
     * structure.  (If a slice were ever exhausted mid-way through a rare
     * huge depth-1/2 search, the loop still returns whatever the LAST
     * fully-completed depth produced — never a partial/garbage score, see
     * the `update` gate a few lines down.) */
    long pv_budget_ms = p->time_limit_ms;
    if (p->mpv_share_budget && p->time_limit_ms > 0 && n_pvs > 1) {
        pv_budget_ms = p->time_limit_ms / n_pvs;
        if (pv_budget_ms < 1) pv_budget_ms = 1;
    }

    for (int mpv = 0; mpv < n_pvs; mpv++) {
        /* Reset time budget for each PV line.
         * Without this, PV line 1 at deep depths consumes the entire
         * time budget, leaving PV lines 2-N with time_up=1 and they
         * never get searched (the MultiPV "freeze" bug).
         * Each PV line gets its own full time allocation (or its share of
         * the total, if mpv_share_budget is set — see pv_budget_ms above,
         * which equals p->time_limit_ms and is therefore a no-op when
         * mpv_share_budget==0). */
        if (p->time_limit_ms > 0) {
            ss->deadline_ms = now_ms() + pv_budget_ms;
        }
        ss->time_up = 0;
        Move best_move = {0}; int best_score = 0;
        Move pv[MAX_PLY]; int pv_len = 0;
        int prev_score_valid = 0;
        int prev_score = 0;

        for (int depth = sd; depth <= md && !ss->time_up; depth++) {
            ss->nodes = 0;
            Move iter_pv[MAX_PLY]; int iter_len = 0;
            int score;

            /* Raise the depth-1 guarantee for the first depth of the first
             * PV line (see SearchState.stop_guard).  Conditions:
             *   sd == 1   — only the search that starts from scratch owes a
             *               move.  Lazy SMP helpers are staggered to start at
             *               depth 2/3/5/7 and their results are discarded, so
             *               guarding them would just make "quit" wait on a
             *               deep search for nothing.
             *   mpv == 0  — later MultiPV lines are extra, not the bestmove.
             * A depth-1 search is a handful of nodes, so the cost of ignoring
             * a stop for its duration is negligible. */
            ss->stop_guard = (sd == 1 && depth == sd && mpv == 0);

            if (depth <= 2 || !prev_score_valid) {
                score = alpha_beta(ss, b, depth, -99999, 99999, iter_pv, &iter_len, 0, -1);
            } else {
                /* Aspiration windows */
                int delta = g_tune.asp_delta_init, alpha2 = prev_score-delta, beta2 = prev_score+delta;
                int tries = 0; int exact = 0;
                while (tries < 6) {
                    tries++;
                    iter_len = 0;
                    score = alpha_beta(ss, b, depth, alpha2, beta2, iter_pv, &iter_len, 0, -1);
                    if (ss->time_up) break;
                    if      (score <= alpha2) { alpha2 -= delta; if(alpha2<-18000)alpha2=-18000; delta*=2; if(delta>g_tune.asp_delta_max)delta=g_tune.asp_delta_max; exact=0; }
                    else if (score >= beta2)  { beta2  += delta; if(beta2>18000)beta2=18000;   delta*=2; if(delta>g_tune.asp_delta_max)delta=g_tune.asp_delta_max; exact=0; }
                    else { exact=1; break; }
                    if (alpha2<=-18000 && beta2>=18000) break;
                }
                if (!exact && iter_len > 0) {}   /* use whatever we have */
            }

            /* First depth is in the bag — lower the guard so a pending stop
             * takes effect from the next depth on (the check just below,
             * and time_up() inside the next depth's alpha_beta). */
            ss->stop_guard = 0;

            if (ss->time_up && !iter_len) break;
            if (iter_len > 0) {
                /* Check if the best move from this iteration is excluded */
                if (is_excluded_root(ss, &iter_pv[0])) {
                    /* The root alpha-beta already skips excluded moves,
                     * but aspiration re-searches might not.  Accept it anyway. */
                }
                int update = !ss->time_up || (best_move.from == 0 && best_move.to == 0);
                if (update) {
                    best_move = iter_pv[0];
                    best_score = score;
                    memcpy(pv, iter_pv, iter_len * sizeof(Move));
                    pv_len = iter_len;
                    prev_score = score; prev_score_valid = 1;
                    if (mpv == 0) res.depth = depth;

                    /* Emit info with multipv tag */
                    /* Sync tb_hits so uci_info_cb reads the correct value */
                    s_tb_hits = ss->tb_hits;

                    if (p->info_cb) {
                        char pv_str[256]; int ppos = 0;
                        for (int pi = 0; pi < pv_len && pi < 12; pi++) {
                            if (pi) pv_str[ppos++] = ' ';
                            char uci[6]; move_to_uci(&pv[pi], uci);
                            int ul = (int)strlen(uci);
                            memcpy(pv_str+ppos, uci, ul); ppos += ul;
                        }
                        pv_str[ppos] = 0;
                        int wb_score = b->turn == COL_W ? score : -score;
                        p->info_cb(depth, wb_score, ss->nodes_total, pv_str, b->turn, mpv + 1);
                    }
                }
            }
            if (ss->time_up) break;
        }

        /* Store this PV line's results */
        int wb_score = b->turn == COL_W ? best_score : -best_score;
        res.scores[mpv] = wb_score;
        res.bests[mpv] = best_move;
        /* Build PV string for this line */
        char *out = res.pvs[mpv]; int pos = 0;
        for (int i = 0; i < pv_len && i < 12; i++) {
            if (i) { out[pos++]=' '; }
            char uci[6]; move_to_uci(&pv[i], uci);
            int len = (int)strlen(uci);
            memcpy(out+pos, uci, len); pos+=len;
        }
        out[pos] = 0;

        /* First PV also fills legacy fields */
        if (mpv == 0) {
            res.best  = best_move;
            res.score = wb_score;
            res.nodes = ss->nodes_total;
            res.tb_hits = ss->tb_hits;
            memcpy(res.pv, res.pvs[0], 256);
        }

        /* Exclude this best move from subsequent PV searches */
        if (best_move.from || best_move.to) {
            ss->excluded_root[ss->excluded_root_n++] = best_move;
        } else {
            break;  /* no legal move found, stop Multi-PV */
        }
    }

    res.num_pvs = ss->excluded_root_n > 0 ? ss->excluded_root_n : 1;
    ss->excluded_root_n = 0;  /* clean up for next search */
    return res;
}

/* ── WASM sret wrapper ─────────────────────────────────────────────────────
 * Emscripten cannot call C functions that return structs by value from JS.
 * This wrapper takes an explicit result pointer as the first argument,
 * matching the sret calling convention Emscripten uses internally.
 * Exported as _search_best_sret and called from the JS worker.              */
void search_best_sret(SearchResult *out, Board *b, const SearchParams *p) {
    /* JS fills SearchParams as raw bytes and cannot set function pointers
     * or volatile int* correctly.  Copy the struct locally and sanitize
     * the fields that would cause a WASM trap if non-NULL garbage. */
    SearchParams safe = *p;
    safe.info_cb      = NULL;   /* no callback in WASM mode */
    safe.stop         = NULL;   /* no external stop flag */
    safe.search_state = NULL;   /* use default global state */
    safe.tt           = NULL;   /* v4.00: JS never sets this — force the
                                  * g_tt fallback inside search_best() instead
                                  * of trusting whatever garbage/OOB byte was
                                  * read from the 32-byte SearchParams buffer
                                  * the JS worker allocates (see zchezz_wasm.html). */
    safe.mpv_share_budget = 0;  /* v4.00: JS never sets this either — force the
                                  * interactive per-PV-full-budget behavior so
                                  * the browser's MultiPV analysis panel always
                                  * gets full-depth lines, regardless of
                                  * whatever byte the JS buffer happened to have
                                  * at this offset. */
    SearchResult tmp = search_best(b, &safe);
    memcpy(out, &tmp, sizeof(SearchResult));
}

/* ── Move-application helper for WASM ─────────────────────────────────────
 * Applies a space-separated list of UCI moves to *b (which must already
 * have been loaded with board_load_fen).  This lets the JS worker seed the
 * board's internal history arrays so that repetition detection works
 * correctly when search_best reads b->hist / b->hist_len.
 *
 * Exported as _board_apply_moves.
 * Signature (JS side):  board_apply_moves(boardPtr, movesStrPtr)           */
void board_apply_moves(Board *b, const char *moves_str) {
    if (!moves_str || !*moves_str) return;
    const char *p = moves_str;
    char tok[8];
    while (1) {
        /* skip whitespace */
        while (*p == ' ' || *p == '\t') p++;
        if (!*p) break;
        /* read token */
        int i = 0;
        while (*p && *p != ' ' && *p != '\t' && i < 7)
            tok[i++] = *p++;
        tok[i] = '\0';
        if (!i) break;
        Move m;
        if (move_from_uci(b, tok, &m))
            board_make(b, &m);
    }
}
