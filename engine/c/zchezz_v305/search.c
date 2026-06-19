/* search.c — Zchezz search layer
 *
 * Direct port of zchezz_worker_v146.js:
 */
#define _POSIX_C_SOURCE 200809L
/*
 *   see()           → static exchange evaluation
 *   score_move()    → move ordering scores
 *   eval_stm()      → NNUE or fast classical (fastEval)
 *   qsearch()       → quiescence search
 *   alpha_beta()    → main negamax + all pruning
 *   search_best()   → iterative deepening + aspiration
 */

#include "search.h"
#include "board.h"
#include "nnue.h"
#include "syzygy.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ── TT (global SoA arrays) ──────────────────────────────────── */
uint64_t TT_H[TT_SIZE];
int32_t  TT_S[TT_SIZE];
int32_t  TT_D[TT_SIZE];
uint16_t TT_G[TT_SIZE];
int32_t  TT_M[TT_SIZE];
int32_t  TT_E[TT_SIZE];
uint16_t TT_GEN = 0;



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
static uint8_t lmr_tab[LMR_D * LMR_M];

/* ── MVV-LVA table ───────────────────────────────────────────── */
/* victim 1-5, attacker 1-6  → index = victim*7+attacker */
static int32_t MVV_LVA[7*7];

/* ── Per-search state (thread-local for Lazy SMP) ──────────── */
static _Thread_local long   s_nodes;
static _Thread_local long   s_nodes_total;   /* acumulado em todas as iterações ID */
static _Thread_local long   s_node_limit;
static _Thread_local long   s_deadline_ms;   /* wall-clock ms deadline, 0=no limit */
static _Thread_local int    s_time_up;
static _Thread_local volatile int *s_stop_flag;  /* external stop signal (from UCI thread) */

/* Killer moves: killers[ply][0..1] */
static _Thread_local Move killers[MAX_PLY][2];

/* History heuristic: history[from*64+to] */
static _Thread_local int32_t mv_history[64*64];

/* Counter move: counter[from*64+to] = packed (from*64+to) of refutation */
static _Thread_local int32_t counter_move[64*64];

/* Continuation history: cont_hist[slot][prev_to][from*64+to]
 *
 * Two-ply context (standard in modern engines):
 *   slot 0 — conditioned on where the move at ply-1 landed  (1-ply CMH)
 *   slot 1 — conditioned on where the move at ply-2 landed  (2-ply CMH)
 *
 * Dimensions per slot: 64 × 4096 × int16 = 512 KB.
 * Total: 2 × 512 KB = 1 MB.  Well within L2 on any modern CPU.
 *
 * Both slots use the same bonus/penalty formula (depth*depth) so the
 * combined signal is symmetric and won't bias the history table. */
static _Thread_local int16_t cont_hist[2][64][64*64];

/* Previous from*64+to per ply */
static _Thread_local int prev_ft[MAX_PLY];

/* Previous landing square per ply (for both CMH lookups) */
static _Thread_local int prev_to_sq[MAX_PLY];

/* Singular extension state */
static _Thread_local int8_t sing_from[MAX_PLY];
static _Thread_local int8_t sing_to  [MAX_PLY];

/* Previous static eval per ply */
static _Thread_local int prev_static_eval[MAX_PLY];

/* Syzygy tablebase hit counter (non-static: read by main.c for info output) */
_Thread_local long s_tb_hits = 0;

/* Syzygy probing configuration (set from UCI options) */
int g_tb_probe_depth = 6;   /* minimum depth for in-tree WDL probing (high to reduce I/O) */
int g_tb_probe_limit = 6;   /* maximum pieces for probing */

/* ── Wall clock helper ───────────────────────────────────────── */
static long now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long)(ts.tv_sec * 1000LL + ts.tv_nsec / 1000000LL);
}

static int time_up(void) {
    if (s_stop_flag && *s_stop_flag) { s_time_up = 1; return 1; }
    if ((s_nodes_total & 8191) == 0 && s_deadline_ms > 0)
        s_time_up = (now_ms() >= s_deadline_ms);
    return s_time_up;
}

/* ── TT helpers ──────────────────────────────────────────────── */
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

/* Pack move into 32 bits — mirrors JS _packMove exactly */
static inline int32_t pack_move(const Move *m) {
    if (!m) return 0;
    int ci = m->castle; /* 0-4 already */
    return (m->from | (m->to<<6) | ((m->prom&7)<<12) |
            (m->epc ? (1<<15) : 0) | (ci<<16));
}

static inline void unpack_move(int32_t v, Move *m) {
    if (!v) { m->from=0; m->to=0; m->prom=0; m->epc=0; m->castle=0; m->score=0; return; }
    m->from   = v & 63;
    m->to     = (v >> 6) & 63;
    m->prom   = (v >> 12) & 7;
    m->epc    = (v >> 15) & 1;
    m->castle = (v >> 16) & 15;
    m->score  = 0;
}

typedef struct { int score; int depth; int flag; Move move; int static_eval; } TTE;

static void tt_store(uint64_t hash, int score,
                     int depth, int flag, const Move *move,
                     int ply, int static_eval) {
    int slot = (int)(hash & TT_MASK);
    int base = slot * TT_BUCKETS;
    int stored_score = tt_score_store(score, ply);
    int32_t packed_df = ((depth & 0xFF) << 8) | (flag & 3);
    int32_t packed_mv = pack_move(move);
    int32_t se = (static_eval != TT_EVAL_NONE) ? static_eval : TT_EVAL_NONE;

    /* Bucket 0: depth-preferred (replace only if deeper or same generation) */
    int exist_depth0 = (TT_D[base] >> 8) & 0xFF;
    if (!TT_H[base] || TT_G[base] != TT_GEN || depth >= exist_depth0) {
        /* If bucket 0 had a valid entry being displaced, move it to bucket 1 */
        if (TT_H[base] && TT_G[base] == TT_GEN && depth >= exist_depth0) {
            TT_H[base+1] = TT_H[base]; TT_S[base+1] = TT_S[base];
            TT_D[base+1] = TT_D[base]; TT_G[base+1] = TT_G[base];
            TT_M[base+1] = TT_M[base]; TT_E[base+1] = TT_E[base];
        }
        TT_H[base] = hash; TT_S[base] = stored_score;
        TT_D[base] = packed_df; TT_G[base] = TT_GEN;
        TT_M[base] = packed_mv; TT_E[base] = se;
        return;
    }

    /* Bucket 1: always-replace */
    TT_H[base+1] = hash; TT_S[base+1] = stored_score;
    TT_D[base+1] = packed_df; TT_G[base+1] = TT_GEN;
    TT_M[base+1] = packed_mv; TT_E[base+1] = se;
}

static int tt_probe(uint64_t hash, int ply, TTE *out) {
    int slot = (int)(hash & TT_MASK);
    int base = slot * TT_BUCKETS;

    /* Check both buckets */
    for (int b = 0; b < TT_BUCKETS; b++) {
        int idx = base + b;
        if (TT_H[idx] != hash) continue;
        if (TT_G[idx] != TT_GEN) {
            /* Stale generation: reuse the stored move for ordering, but not the score */
            out->score  = TT_EVAL_NONE;
            out->depth  = 0;
            out->flag   = TT_UPPER;
            out->static_eval = TT_EVAL_NONE;
            unpack_move(TT_M[idx], &out->move);
            return 2;   /* 2 = stale hit (move only) */
        }
        int d = TT_D[idx];
        out->score  = tt_score_read(TT_S[idx], ply);
        out->depth  = (d >> 8) & 0xFF;
        out->flag   = d & 3;
        out->static_eval = TT_E[idx];
        unpack_move(TT_M[idx], &out->move);
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
    /* Fast path: pass precomputed bb[12] and Zobrist hash directly so
     * nnue_eval_bb can (a) skip _build_bb_from_board and (b) use a
     * 2-slot hash cache to avoid re-projecting ext features for the
     * same position evaluated in the same search node. */
    return nnue_eval_bb(b->turn == COL_W ? 0 : 1, b->b, b->bb, b->hash);
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
 * `cmh0` = prev_to_sq[ply-1],  -1 if not available
 * `cmh1` = prev_to_sq[ply-2],  -1 if not available
 */
static int score_move(const Move *m, const Board *bd, int ply,
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
    if (killers[ply][0].from==mfr && killers[ply][0].to==mto &&
        (killers[ply][0].from||killers[ply][0].to)) return 900000;
    if (killers[ply][1].from==mfr && killers[ply][1].to==mto &&
        (killers[ply][1].from||killers[ply][1].to)) return 800000;
    if (prev_ft_val >= 0 && counter_move[prev_ft_val] == (mfr*64+mto)) return 780000;

    int ft = mfr*64 + mto;
    int h  = mv_history[ft];
    int c0 = (cmh0 >= 0) ? cont_hist[0][cmh0][ft] : 0;
    int c1 = (cmh1 >= 0) ? cont_hist[1][cmh1][ft] : 0;
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
static void sort_moves(Move *moves, int n, const Board *bd, int ply,
                       const Move *pv_move, int ok_sq, int prev_ft_val,
                       int cmh0, int cmh1) {
    for (int i = 0; i < n; i++)
        moves[i].score = score_move(&moves[i], bd, ply, pv_move, ok_sq,
                                    prev_ft_val, cmh0, cmh1);
    for (int i = 1; i < n; i++) {
        Move m = moves[i]; int j = i-1;
        while (j >= 0 && moves[j].score < m.score) { moves[j+1]=moves[j]; j--; }
        moves[j+1] = m;
    }
}

/* ── Quiescence search ───────────────────────────────────────── */
static int qsearch(Board *b, int alpha, int beta, int ply) {
    if (ply >= MAX_PLY-1) return eval_stm(b);
    if (s_nodes >= s_node_limit || time_up()) return eval_stm(b);
    s_nodes++; s_nodes_total++;

    /* ── TT probe in qsearch ─────────────────────────────────── */
    uint64_t qs_hash = b->hash;
    TTE qs_tte;
    int qs_tte_hit = tt_probe(qs_hash, ply, &qs_tte);
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
        sort_moves(moves, n, b, ply, NULL, ok_sq, -1, -1, -1);
        int best = -99999, legal = 0;
        Move best_move_qs = {0};
        for (int i = 0; i < n; i++) {
            board_make(b, &moves[i]);
            int prev_turn = b->turn ^ 24;
            if (board_is_attacked(b, prev_turn==COL_W ? b->wk : b->bk, b->turn)) {
                board_unmake(b); continue;
            }
            legal++;
            int sc = -qsearch(b, -beta, -alpha, ply+1);
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
        moves[i].score = score_move(&moves[i], b, ply, qs_pv, ok_sq, -1, -1, -1);
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
        int sc = -qsearch(b, -beta, -alpha, ply+1);
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

/* ── Alpha-Beta ──────────────────────────────────────────────── */
/* Multi-PV root exclusion — forward declarations (defined after search_reset) */
static _Thread_local Move s_excluded_root[MAX_MULTI_PV];
static _Thread_local int  s_excluded_root_n = 0;
static int  is_excluded_root(const Move *m);

/* Forward declaration */
static int alpha_beta(Board *b, int depth, int alpha, int beta,
                      Move *pv, int *pv_len, int ply, int in_check_hint);

static int alpha_beta(Board *b, int depth, int alpha, int beta,
                      Move *pv, int *pv_len, int ply, int in_check_hint) {
    if (ply >= MAX_PLY-1) { *pv_len=0; return eval_stm(b); }
    if (s_nodes >= s_node_limit || time_up()) {
        *pv_len=0;
        return depth<=0 ? eval_stm(b) : qsearch(b,alpha,beta,ply);
    }
    s_nodes++; s_nodes_total++;
    *pv_len = 0;

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

    /* ── Syzygy WDL probe (non-root) ───────────────────────────── */
    if (ply > 0 && depth >= g_tb_probe_depth) {
        int npieces = __builtin_popcountll(b->occ);
        if (npieces <= g_tb_probe_limit) {
            int wdl;
            if (syzygy_probe_wdl(b, &wdl)) {
                s_tb_hits++;
                int tb_score;
                switch (wdl) {
                    case 4: /* TB_WIN */
                        tb_score = 18000 - ply; break;
                    case 3: /* TB_CURSED_WIN (50-move draw) */
                        tb_score = 1; break;
                    case 2: /* TB_DRAW */
                        tb_score = 0; break;
                    case 1: /* TB_BLESSED_LOSS (50-move draw) */
                        tb_score = -1; break;
                    case 0: /* TB_LOSS */
                    default:
                        tb_score = -18000 + ply; break;
                }
                /* Use TB result for cutoffs only when clearly outside
                 * the alpha-beta window.  Do NOT adjust alpha/beta bounds
                 * — the bound tightening was causing the engine to search
                 * differently (artificially narrow windows) which hurt
                 * practical play at fast time controls.
                 *
                 * WIN:  if score >= beta, cut.  Otherwise continue.
                 * LOSS: if score <= alpha, cut. Otherwise continue.
                 * DRAW: only return 0 if alpha < 0 < beta (exact). */
                if (wdl == 4 || wdl == 3) {
                    if (tb_score >= beta) return tb_score;
                    /* WIN but below beta: let search continue normally */
                } else if (wdl == 0 || wdl == 1) {
                    if (tb_score <= alpha) return tb_score;
                    /* LOSS but above alpha: let search continue normally */
                } else {
                    /* DRAW: only return if 0 is an exact cutoff */
                    if (0 >= beta || 0 <= alpha) return 0;
                    /* Otherwise continue searching normally */
                }
            }
        }
    }

    /* TT probe */
    TTE tte; int tte_hit = 0;
    Move pv_move = {0};
    tte_hit = tt_probe(b->hash, ply, &tte);
    if (tte_hit) {
        pv_move = tte.move;   /* always use the stored move for ordering */
        /* Skip TT cutoffs at ply 0 when Multi-PV exclusion is active:
         * the stored score was computed with a different root move set. */
        if (tte_hit == 1 && tte.depth >= depth && !(ply == 0 && s_excluded_root_n > 0)) {
            /* Full hit — can use score for cutoff */
            if (tte.flag == TT_EXACT) return tte.score;
            if (tte.flag == TT_LOWER && tte.score >= beta) return tte.score;
            if (tte.flag == TT_UPPER && tte.score <= alpha) return tte.score;
        }
    }

    if (depth <= 0) return qsearch(b, alpha, beta, ply);

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

    /* Improving flag — uses raw_eval (uncorrected) for consistency across plies */
    int improving = 0;
    if (!in_check) {
        prev_static_eval[ply] = raw_eval;   /* store raw, not TT-corrected */
        if (ply >= 2 && prev_static_eval[ply-2] != TT_EVAL_NONE)
            improving = raw_eval > prev_static_eval[ply-2];
    } else {
        prev_static_eval[ply] = TT_EVAL_NONE;
    }

    /* Razoring */
    if (!in_check && !is_pv && depth==1 && static_eval+200 < alpha) {
        int qs = qsearch(b, alpha-1, alpha, ply);
        if (qs < alpha) return qs;
    }

    /* Reverse Futility Pruning — extended to depth 9 */
    if (!in_check && !is_pv && depth>=2 && depth<=9 && beta<18000 && static_eval<18000) {
        int rfp_margin = depth*90 - (improving ? 50 : 0);
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
        int R = 3 + depth / 3;
        if (R > 6) R = 6;
        if (static_eval - beta > 200) R += 1;
        /* Make null move */
        uint64_t save_hash = b->hash;
        int8_t   save_ep   = b->ep;
        int      save_turn = b->turn;
        int      save_hm   = b->hm;
        if (b->ep >= 0) { b->hash^=ZR_ep[b->ep&7]; b->ep=-1; }
        b->hash ^= ZR_side; b->turn ^= 24;
        b->hm++;
        /* Increment global hist so board_is_draw works.
         * Guard against overflow: if the table is full, skip the write
         * (repetition detection will be incomplete but won't corrupt memory). */
        int hist_pushed = 0;
        if (g_hist_len < HIST_SIZE) {
            g_hist[g_hist_len] = b->hash;
            g_hist_len++; hist_pushed = 1;
        }
        Move dummy_pv[MAX_PLY]; int dummy_len=0;
        int null_score = -alpha_beta(b, depth-1-R, -beta, -beta+1, dummy_pv, &dummy_len, ply+1, -1);
        if (hist_pushed) g_hist_len--;
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
        int pc_beta  = beta + 200;
        int pc_depth = depth - 4;   /* shallow probe: depth-4, min 1 */
        if (pc_depth < 1) pc_depth = 1;

        /* Generate captures + promotions only */
        Move pc_moves[MAX_MOVES];
        int  pc_n = board_gen_captures(b, pc_moves);
        /* Score and sort (reuse existing move scorer) */
        int pc_ok = b->turn == COL_W ? b->bk : b->wk;
        for (int pi = 0; pi < pc_n; pi++)
            pc_moves[pi].score = score_move(&pc_moves[pi], b, ply,
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
            int pc_sc = -qsearch(b, -pc_beta, -pc_beta+1, ply+1);
            if (pc_sc >= pc_beta) {
                Move pc_pv[MAX_PLY]; int pc_len = 0;
                pc_sc = -alpha_beta(b, pc_depth, -pc_beta, -pc_beta+1,
                                    pc_pv, &pc_len, ply+1, -1);
            }
            board_unmake(b);

            if (pc_sc >= pc_beta) return pc_sc;
        }
    }

    /* IIR */
    if (!pv_move.from && !pv_move.to && depth>=4 && !in_check && depth-1>=2) depth--;

    /* Singular extension */
    int sing_ext = 0;
    if (!in_check && depth>=7 && tte_hit && tte.depth>=depth-4 &&
        sing_from[ply]<0 && ply>0 && (tte.flag==TT_EXACT||tte.flag==TT_LOWER)) {
        int s_beta  = tte.score - depth*2; if (s_beta < -18000) s_beta = -18000;
        int s_depth = (depth>>1)-1; if (s_depth < 1) s_depth = 1;
        sing_from[ply] = (int8_t)pv_move.from;
        sing_to  [ply] = (int8_t)pv_move.to;
        Move sp_pv[MAX_PLY]; int sp_len=0;
        int ss = alpha_beta(b, s_depth, s_beta-1, s_beta, sp_pv, &sp_len, ply, -1);
        sing_from[ply] = -1; sing_to[ply] = -1;
        if (ss < s_beta) sing_ext = 1;
        else if (s_beta >= beta) return s_beta;
    }

    /* Move generation + scoring (lazy pick-best — no full sort) */
    Move moves[MAX_MOVES];
    int n = board_gen_moves(b, moves);
    int ok_sq = b->turn==COL_W ? b->bk : b->wk;
    int cur_prev_ft = ply > 0 ? prev_ft[ply-1] : -1;
    int cmh0 = ply >= 1 ? prev_to_sq[ply-1] : -1;   /* 1-ply CMH context */
    int cmh1 = ply >= 2 ? prev_to_sq[ply-2] : -1;   /* 2-ply CMH context */
    /* Score all moves upfront (same as before), then use pick-best (selection sort)
     * instead of a full insertion sort.  Nodes that cut early avoid sorting the
     * remaining N-k moves entirely. */
    const Move *pv_ptr = (pv_move.from||pv_move.to) ? &pv_move : NULL;
    for (int i = 0; i < n; i++)
        moves[i].score = score_move(&moves[i], b, ply, pv_ptr, ok_sq,
                                    cur_prev_ft, cmh0, cmh1);

    /* LMP limits — extended to depth 5 */
    static const int lmp_limit[8] = {0,10,18,26,36,48,62,78};
    int fut_adj = improving ? 0 : 50;
    static const int fut_base[9] = {0,150,300,450,600,750,900,1050,1200};

    int best = -99999, flag = TT_UPPER;
    Move best_move = {0};
    int legal_count = 0, quiet_count = 0;
    Move child_pv[MAX_PLY]; int child_len = 0;

    for (int i = 0; i < n; i++) {
        /* Pick-best: find the highest-scored remaining move and swap to position i */
        int best_idx = i;
        for (int j = i+1; j < n; j++)
            if (moves[j].score > moves[best_idx].score) best_idx = j;
        if (best_idx != i) {
            Move tmp_pb = moves[i];
            moves[i] = moves[best_idx];
            moves[best_idx] = tmp_pb;
        }

        Move *m = &moves[i];
        int mfr = m->from, mto = m->to;
        int is_capture = !!(b->b[mto] || m->epc);
        int is_promo   = !!m->prom;
        int is_castle  = !!m->castle;
        int is_quiet   = !is_capture && !is_promo && !is_castle;

        /* LMP */
        int is_killer = is_quiet && (
            (killers[ply][0].from==mfr&&killers[ply][0].to==mto&&(killers[ply][0].from||killers[ply][0].to)) ||
            (killers[ply][1].from==mfr&&killers[ply][1].to==mto&&(killers[ply][1].from||killers[ply][1].to)));
        if (!in_check && is_quiet && depth<=7 && i>0 && !is_killer) {
            quiet_count++;
            int lmp_lim = lmp_limit[depth<8?depth:7];
            if (!improving) lmp_lim = (lmp_lim + 1) / 2;
            if (quiet_count > lmp_lim) continue;

            /* History pruning: skip moves with very bad history at shallow depths.
             * Combined history = mv_history + CMH[0] + CMH[1]. */
            if (depth <= 4) {
                int ft_hp = mfr * 64 + mto;
                int ch_hp = mv_history[ft_hp];
                if (cmh0 >= 0) ch_hp += cont_hist[0][cmh0][ft_hp];
                if (cmh1 >= 0) ch_hp += cont_hist[1][cmh1][ft_hp];
                int hp_thresh = -4000 * depth;   /* -4000 at d1, -16000 at d4 */
                if (ch_hp < hp_thresh) continue;
            }
        }

        /* Skip singular move */
        if (sing_from[ply]>=0 && mfr==sing_from[ply] && mto==sing_to[ply]) continue;

        /* SEE pruning on losing captures.
         * score_move() already ran SEE and encoded the result:
         *   losing capture → score = 200000 + sv*2   (sv < 0)
         *   winning/equal  → score >= 1600000         (sv >= 0, never prune)
         * Recovering sv from the score avoids a second see() call entirely. */
        if (!in_check && is_capture && !is_promo && i>0 && depth>2 && !is_pv) {
            int ms = m->score;
            if (ms < 1600000) {
                int sv = (ms - 200000) / 2;
                int see_thresh = depth<=4 ? -80 : depth<=6 ? -120 : -160;
                if (sv < see_thresh) {
                    int tr=mto>>3,tc=mto&7,kr=ok_sq>>3,kc=ok_sq&7;
                    int dr=tr-kr;if(dr<0)dr=-dr;
                    int dc=tc-kc;if(dc<0)dc=-dc;
                    if ((dr>dc?dr:dc) > 1) continue;
                }
            }
        }

        board_make(b, m);
        /* #1 TT Prefetch: issue cache hint for the child's TT slot now,
         * so the line is in L1 by the time tt_probe fires inside the recursive call. */
        __builtin_prefetch(&TT_H[((b->hash ^ ZR_side) & TT_MASK) * TT_BUCKETS], 0, 1);
        prev_ft[ply]    = mfr*64 + mto;
        prev_to_sq[ply] = mto;

        /* Legality */
        int mover_col = b->turn ^ 24;
        int king_sq   = mover_col == COL_W ? b->wk : b->bk;
        if (board_is_attacked(b, king_sq, b->turn)) { board_unmake(b); continue; }

        /* Multi-PV: skip excluded root moves */
        if (ply == 0 && s_excluded_root_n > 0 && is_excluded_root(m)) {
            board_unmake(b); continue;
        }

        legal_count++;

        /* Gives check */
        int gives_check = 0;
        { uint8_t gpt=b->b[m->to]&7,gksq=b->turn==COL_W?b->wk:b->bk;
          int gdr=((m->to>>3)-(gksq>>3)); if(gdr<0)gdr=-gdr;
          int gdc=((m->to&7)-(gksq&7));   if(gdc<0)gdc=-gdc;
          if (gpt>=3||gpt==2||m->prom||(gdr>gdc?gdr:gdc)<=2)
              gives_check = board_in_check(b);
        }

        /* Futility pruning — extended to depth 8 */
        if (!in_check && is_quiet && !gives_check && depth>=1 && depth<=8 && legal_count>1) {
            int fd = depth < 9 ? depth : 8;
            if (static_eval + fut_base[fd] + fut_adj <= alpha) { board_unmake(b); continue; }
        }

        int check_ext = 0;
        if (in_check && depth==1) check_ext=1;
        if (sing_ext && pv_move.from==m->from && pv_move.to==m->to)
            check_ext = check_ext>1?check_ext:1;

        /* Pawn-on-7th extension: pawn about to promote is critical */
        if (!check_ext && !in_check) {
            uint8_t pc = b->b[m->to];
            if (pc && PC_TYPE(pc)==1) {  /* pawn */
                int to_rank = m->to >> 3;
                /* For white: rank 1 = row 1 (0-indexed from a8), for black: rank 6 */
                if ((PC_COLOR(pc)==COL_W && to_rank==1) ||
                    (PC_COLOR(pc)==COL_B && to_rank==6))
                    check_ext = 1;
            }
        }

        int sc;
        child_len = 0;
        if (legal_count == 1) {
            sc = -alpha_beta(b, depth-1+check_ext, -beta, -alpha, child_pv, &child_len, ply+1, gives_check);
        } else {
            int reduce = 0;
            if (legal_count>=3 && depth>=3 && !in_check && !gives_check && is_quiet && !is_killer) {
                int ld = depth<LMR_D?depth:LMR_D-1;
                int lm = legal_count<LMR_M?legal_count:LMR_M-1;
                reduce = lmr_tab[ld*LMR_M+lm];
                if (is_pv) reduce = reduce>0?reduce-1:0;
                if (!improving) reduce += 1;  /* reduce more when not improving */

                /* History-based LMR adjustment (Phase 7):
                 * Bad-history moves get more reduction, good-history less.
                 * combined_hist = mv_history + CMH[0] + CMH[1].
                 * Each is [-16384..16384], so combined is [-49152..49152]. */
                {
                    int ft_idx = mfr * 64 + mto;
                    int ch = mv_history[ft_idx];
                    if (cmh0 >= 0) ch += cont_hist[0][cmh0][ft_idx];
                    if (cmh1 >= 0) ch += cont_hist[1][cmh1][ft_idx];
                    /* Reduce more for bad history, less for good */
                    if (ch < -4000) reduce += 1;
                    if (ch < -8000) reduce += 1;
                    if (ch > 4000 && reduce > 0) reduce -= 1;
                }
                /* Don't reduce below 0 or into qs */
                if (reduce >= depth - 1) reduce = depth - 2;
                if (reduce < 0) reduce = 0;
            }
            /* LMR for losing captures (pre-computed score 100000..200000 = SEE < 0) */
            if (is_capture && !is_promo && depth>=3 && legal_count>=4 && !in_check) {
                if (m->score >= 100000 && m->score <= 200000) reduce = 1;
            }
            Move null_pv[MAX_PLY]; int null_len=0;
            sc = -alpha_beta(b, depth-1-reduce+check_ext, -alpha-1, -alpha, null_pv, &null_len, ply+1, gives_check);
            /* PVS re-search: if LMR reduced search fails high... */
            if (sc > alpha && reduce > 0) {
                /* First: full-depth ZWS (cheaper than full window) */
                sc = -alpha_beta(b, depth-1+check_ext, -alpha-1, -alpha, null_pv, &null_len, ply+1, gives_check);
            }
            if (sc > alpha && sc < beta) {
                /* Full window re-search only if ZWS confirmed fail-high
                 * and we're in a PV node (alpha < sc < beta) */
                sc = -alpha_beta(b, depth-1+check_ext, -beta, -alpha, child_pv, &child_len, ply+1, gives_check);
            }
        }
        board_unmake(b);

        if (sc > best) {
            best = sc; best_move = *m;
            if (sc > alpha) {
                alpha = sc; flag = TT_EXACT;
                /* Build PV */
                pv[0] = *m;
                memcpy(pv+1, child_pv, child_len * sizeof(Move));
                *pv_len = child_len + 1;
            }
        }
        if (alpha >= beta) {
            /* Beta cutoff — update killers, history, counter move, and both CMH slots */
            if (is_quiet) {
                /* Killers */
                int km0_set = killers[ply][0].from||killers[ply][0].to;
                if (!km0_set || killers[ply][0].from!=mfr || killers[ply][0].to!=mto) {
                    killers[ply][1] = killers[ply][0];
                    killers[ply][0] = *m;
                }
                /* Counter move */
                if (cur_prev_ft >= 0) counter_move[cur_prev_ft] = mfr*64+mto;

                /* History / CMH bonus: depth² (same formula for bonus and penalty
                 * so the table doesn't drift).  Clamp to ±16384 to leave headroom. */
                int bonus = depth*depth;
                if (bonus > 16384) bonus = 16384;

                /* ── Reward the move that caused cutoff ── */
                int ft = mfr*64+mto;
                {   int h = mv_history[ft] + bonus;
                    mv_history[ft] = h < 16384 ? h : 16384; }
                if (cmh0 >= 0) {
                    int c = cont_hist[0][cmh0][ft] + bonus;
                    cont_hist[0][cmh0][ft] = (int16_t)(c < 16384 ? c : 16384);
                }
                if (cmh1 >= 0) {
                    int c = cont_hist[1][cmh1][ft] + bonus;
                    cont_hist[1][cmh1][ft] = (int16_t)(c < 16384 ? c : 16384);
                }

                /* ── Penalise quiet moves searched before the cutoff ── */
                for (int j = 0; j < i; j++) {
                    Move *pm = &moves[j];
                    if (b->b[pm->to] || pm->epc || pm->prom) continue;  /* skip non-quiets */
                    int pft = pm->from*64 + pm->to;
                    {   int h = mv_history[pft] - bonus;
                        mv_history[pft] = h > -16384 ? h : -16384; }
                    if (cmh0 >= 0) {
                        int c = cont_hist[0][cmh0][pft] - bonus;
                        cont_hist[0][cmh0][pft] = (int16_t)(c > -16384 ? c : -16384);
                    }
                    if (cmh1 >= 0) {
                        int c = cont_hist[1][cmh1][pft] - bonus;
                        cont_hist[1][cmh1][pft] = (int16_t)(c > -16384 ? c : -16384);
                    }
                }
            }
            flag = TT_LOWER; break;
        }
    }

    if (!legal_count) return in_check ? (-19000+ply) : 0;
    if (best_move.from||best_move.to)
        tt_store(b->hash, best, depth, flag, &best_move, ply, raw_eval);
    return best;
}

/* ── Iterative deepening + aspiration ────────────────────────── */
void search_init(void) {
    /* LMR table */
    for (int d = 1; d < LMR_D; d++)
        for (int m = 1; m < LMR_M; m++) {
            double v = log((double)d) * log((double)m) / 1.5;
            int r = (int)v;
            if (r < 1) r = 1;
            if (r > d-1) r = d-1;
            lmr_tab[d*LMR_M+m] = (uint8_t)r;
        }
    /* MVV-LVA: (6-attacker) + victim*10 */
    for (int v = 1; v <= 5; v++)
        for (int a = 1; a <= 6; a++)
            MVV_LVA[v*7+a] = (6-a) + v*10;
    /* TT init */
    memset(TT_H, 0, sizeof(TT_H));
    for (int i = 0; i < TT_SIZE; i++) TT_E[i] = TT_EVAL_NONE;
}

void search_history_clear(void) {
    memset(mv_history,   0, sizeof(mv_history));
    memset(counter_move, 0, sizeof(counter_move));
    memset(cont_hist,    0, sizeof(cont_hist));
}

void search_reset(void) {
    memset(killers,      0, sizeof(killers));
    memset(sing_from,   -1, sizeof(sing_from));
    memset(sing_to,     -1, sizeof(sing_to));
    for (int i = 0; i < MAX_PLY; i++) {
        prev_ft[i] = -1;
        prev_to_sq[i] = -1;
        prev_static_eval[i] = TT_EVAL_NONE;
    }
    /* Age history tables: divide by 4 to preserve relative ordering
     * while preventing saturation across games/iterations.
     * Loop is ~4K iterations for mv_history and ~512K for cont_hist[2];
     * compiler will auto-vectorise with -O3 -march=native.            */
    for (int i = 0; i < 64*64; i++) mv_history[i] >>= 2;
    memset(counter_move, 0, sizeof(counter_move));
    /* Age both CMH slots (int16_t arithmetic right-shift → signed divide by 4) */
    for (int s = 0; s < 2; s++)
        for (int i = 0; i < 64; i++)
            for (int j = 0; j < 64*64; j++)
                cont_hist[s][i][j] >>= 2;
    g_undo_top = 0;
    g_hist_len = 0;
}

/* is_excluded_root — definition (forward-declared before alpha_beta) */
static int is_excluded_root(const Move *m) {
    for (int i = 0; i < s_excluded_root_n; i++)
        if (s_excluded_root[i].from == m->from && s_excluded_root[i].to == m->to &&
            s_excluded_root[i].prom == m->prom)
            return 1;
    return 0;
}

SearchResult search_best(Board *b, const SearchParams *p) {
    SearchResult res = {0};
    /* Only main thread increments TT generation — helpers share it */
    if (p->start_depth <= 1) TT_GEN = (TT_GEN+1) & 0xFFFF;
    nnue_reset();

    int md = p->max_depth;
    int n_pvs = p->multi_pv;
    if (n_pvs < 1) n_pvs = 1;
    if (n_pvs > MAX_MULTI_PV) n_pvs = MAX_MULTI_PV;

    /* Node limit */
    if (p->time_limit_ms > 0) {
        s_node_limit = p->node_limit > 0 ? p->node_limit : (long)2e9;
    } else if (p->node_limit > 0) {
        s_node_limit = p->node_limit;
    } else {
        s_node_limit = (long)2e9;
    }

    s_deadline_ms = p->time_limit_ms > 0 ? now_ms() + p->time_limit_ms : 0;
    s_time_up     = 0;
    s_stop_flag   = p->stop;  /* may be NULL */
    s_nodes       = 0;
    s_nodes_total = 0;
    s_tb_hits     = 0;

    search_reset();

    /* Rebuild NNUE accumulator */
    if (nnue_ready()) nnue_rebuild(b->b);

    /* Re-seed g_hist from board's game history */
    for (int i = 0; i < b->hist_len && g_hist_len < HIST_SIZE; i++) {
        g_hist[g_hist_len] = b->hist[i];
        g_hist_len++;
    }

    /* ── Root Syzygy TB probe ────────────────────────────────────── */
    /* Probe once at the root for TB positions (≤5 pieces, no castling).
     * This is the main way TB helps: it gives the PERFECT move immediately
     * without wasting search time.  In-tree probing is mostly disabled
     * (probe_depth=6) because its I/O overhead hurts more than it helps. */
    if (p->start_depth <= 1) {  /* only main thread, not SMP helpers */
        int npieces = __builtin_popcountll(b->occ);
        if (npieces >= 4 && npieces <= g_tb_probe_limit) {
            int tb_from, tb_to, tb_wdl, tb_dtz;
            if (syzygy_probe_root(b, &tb_from, &tb_to, &tb_wdl, &tb_dtz)) {
                s_tb_hits++;
                /* For WIN: return TB's best move immediately — it's the
                 * theoretically correct move and saves all search time. */
                if (tb_wdl == 4 || tb_wdl == 3) { /* TB_WIN or TB_CURSED_WIN */
                    res.best.from = (uint8_t)tb_from;
                    res.best.to   = (uint8_t)tb_to;
                    res.bests[0]  = res.best;
                    int tb_score  = (tb_wdl == 4) ? 18000 : 1;
                    res.score     = (b->turn == COL_W) ? tb_score : -tb_score;
                    res.scores[0] = res.score;
                    res.depth     = 1;
                    res.nodes     = 1;
                    res.tb_hits   = 1;
                    char uci[6]; move_to_uci(&res.best, uci);
                    snprintf(res.pv, 256, "%s", uci);
                    memcpy(res.pvs[0], res.pv, 256);
                    if (p->info_cb) {
                        p->info_cb(1, res.score, 1, uci, b->turn, 1);
                    }
                    return res;
                }
                /* For LOSS: also return TB move (minimize damage). */
                if (tb_wdl == 0 || tb_wdl == 1) {
                    res.best.from = (uint8_t)tb_from;
                    res.best.to   = (uint8_t)tb_to;
                    res.bests[0]  = res.best;
                    int tb_score  = (tb_wdl == 0) ? -18000 : -1;
                    res.score     = (b->turn == COL_W) ? tb_score : -tb_score;
                    res.scores[0] = res.score;
                    res.depth     = 1;
                    res.nodes     = 1;
                    res.tb_hits   = 1;
                    char uci[6]; move_to_uci(&res.best, uci);
                    snprintf(res.pv, 256, "%s", uci);
                    memcpy(res.pvs[0], res.pv, 256);
                    if (p->info_cb) {
                        p->info_cb(1, res.score, 1, uci, b->turn, 1);
                    }
                    return res;
                }
                /* For DRAW: let normal search run to find best practical move */
            }
        }
    }

    /* ── Multi-PV outer loop ─────────────────────────────────────── */
    s_excluded_root_n = 0;
    /* Starting depth for iterative deepening (Lazy SMP helpers start higher) */
    int sd = p->start_depth > 1 ? p->start_depth : 1;

    for (int mpv = 0; mpv < n_pvs && !s_time_up; mpv++) {
        Move best_move = {0}; int best_score = 0;
        Move pv[MAX_PLY]; int pv_len = 0;
        int prev_score_valid = 0;
        int prev_score = 0;

        for (int depth = sd; depth <= md && !s_time_up; depth++) {
            s_nodes = 0;
            Move iter_pv[MAX_PLY]; int iter_len = 0;
            int score;

            if (depth <= 2 || !prev_score_valid) {
                score = alpha_beta(b, depth, -99999, 99999, iter_pv, &iter_len, 0, -1);
            } else {
                /* Aspiration windows */
                int delta = 20, alpha2 = prev_score-delta, beta2 = prev_score+delta;
                int tries = 0; int exact = 0;
                while (tries < 6) {
                    tries++;
                    iter_len = 0;
                    score = alpha_beta(b, depth, alpha2, beta2, iter_pv, &iter_len, 0, -1);
                    if (s_time_up) break;
                    if      (score <= alpha2) { alpha2 -= delta; if(alpha2<-18000)alpha2=-18000; delta*=2; if(delta>500)delta=500; exact=0; }
                    else if (score >= beta2)  { beta2  += delta; if(beta2>18000)beta2=18000;   delta*=2; if(delta>500)delta=500; exact=0; }
                    else { exact=1; break; }
                    if (alpha2<=-18000 && beta2>=18000) break;
                }
                if (!exact && iter_len > 0) {}   /* use whatever we have */
            }

            if (s_time_up && !iter_len) break;
            if (iter_len > 0) {
                /* Check if the best move from this iteration is excluded */
                if (is_excluded_root(&iter_pv[0])) {
                    /* The root alpha-beta already skips excluded moves,
                     * but aspiration re-searches might not.  Accept it anyway. */
                }
                int update = !s_time_up || (best_move.from == 0 && best_move.to == 0);
                if (update) {
                    best_move = iter_pv[0];
                    best_score = score;
                    memcpy(pv, iter_pv, iter_len * sizeof(Move));
                    pv_len = iter_len;
                    prev_score = score; prev_score_valid = 1;
                    if (mpv == 0) res.depth = depth;

                    /* Emit info with multipv tag */
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
                        p->info_cb(depth, wb_score, s_nodes_total, pv_str, b->turn, mpv + 1);
                    }
                }
            }
            if (s_time_up) break;
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
            res.nodes = s_nodes_total;
            res.tb_hits = s_tb_hits;
            memcpy(res.pv, res.pvs[0], 256);
        }

        /* Exclude this best move from subsequent PV searches */
        if (best_move.from || best_move.to) {
            s_excluded_root[s_excluded_root_n++] = best_move;
        } else {
            break;  /* no legal move found, stop Multi-PV */
        }
    }

    res.num_pvs = s_excluded_root_n > 0 ? s_excluded_root_n : 1;
    s_excluded_root_n = 0;  /* clean up for next search */
    return res;
}

/* ── WASM sret wrapper ─────────────────────────────────────────────────────
 * Emscripten cannot call C functions that return structs by value from JS.
 * This wrapper takes an explicit result pointer as the first argument,
 * matching the sret calling convention Emscripten uses internally.
 * Exported as _search_best_sret and called from the JS worker.              */
void search_best_sret(SearchResult *out, Board *b, const SearchParams *p) {
    SearchResult tmp = search_best(b, p);
    /* Use memcpy instead of struct assignment to avoid potential
     * WASM sret ABI issues with 1376-byte struct copies. */
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
