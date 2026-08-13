/* tools/arena.h — public surface of tools/arena.c, for embedding.
 *
 * WHY THIS EXISTS: Appendix F.4's arena is meant to be reusable as the
 * fitness function of a future native GA/SPSA tuner for search
 * constants (see arena.c's header "EMBEDDABLE AS A LIBRARY"). That
 * tuner links arena.c compiled with -DARENA_NO_MAIN (which #ifndef's
 * out main() at the bottom of that file) and drives it directly:
 *
 *     arena_global_init();                 // once
 *     for (each candidate parameter set) {
 *         Config cfg = ...;                // net:/uci: players, movetime, games, ...
 *         ArenaReport rep;
 *         arena_run(&cfg, &openings, &rep); // one short match, in-process
 *         // read rep.pair[][], rep.rank[], rep.sprt_* directly —
 *         // no subprocess, no argv, no JSON/stdout parsing
 *     }
 *
 * *** CONTRACT: every type below MUST stay byte-identical to its
 * definition inside arena.c (which does NOT #include this file — see
 * that file's comment at each type for why: its internal definitions
 * are interleaved with the static helper functions that build them up
 * incrementally, in a specific compile-order-dependent sequence, and
 * a full single-source-of-truth split was judged not worth the
 * reorganisation risk for a v4.00 branch under active development).
 * This mirrors the SAME cross-language-contract discipline
 * tools/selfplay.c's SelfplaySample / train/dataset.py's SAMPLE_DTYPE
 * already use in this codebase — if you change a struct in arena.c,
 * update this file in the same commit. ***
 */
#pragma once
#include <stdint.h>
#include <stdio.h>
#include "nnue.h"

/* ── Player definition ─────────────────────────────────────────── */
typedef enum { PLAYER_NET, PLAYER_UCI } PlayerKind;

#define MAX_PLAYERS 16

typedef struct {
    PlayerKind kind;
    char       path[512];
    char       name[128];
    NnueNet   *net;              /* PLAYER_NET only; caller loads via nnue_net_load() */
} PlayerDef;

/* ── Opening pool (FEN list, from EPD or PGN-replayed-to-FEN) ─────
 * Build with load_openings() (declared below) or by hand (fens/n/cap,
 * see opening_pool_add()'s shape in arena.c if constructing one
 * programmatically — not exposed here since a tuner will typically
 * just call load_openings() once and reuse the pool across many
 * arena_run() calls). */
typedef struct {
    char **fens;
    int    n;
    int    cap;
} OpeningPool;

/* ── Match configuration ───────────────────────────────────────── */
typedef struct {
    int      games;            /* per pairing if >2 players, else total */
    int      threads;
    int      movetime_ms;
    long     nodes;
    int      max_plies;
    uint64_t seed;
    double   tt_mb;
    int      gauntlet;              /* 1 if a gauntlet candidate is set */
    char     gauntlet_spec[128];    /* raw CLI form; a tuner building Config by hand
                                      * should instead set gauntlet_idx directly and
                                      * leave this empty */
    int      gauntlet_idx;
    char     openings_path[512];
    int      opening_plies;
    int      sprt;
    double   elo0, elo1, alpha, beta;
    char     json_path[512];
    char     pgn_path[512];
    char     bin_path[512];   /* "" = no --bin training-sample output (net: players only, all sharing
                                * one weight file — see arena.c's ARENA_DEFAULT_BIN_PATH comment) */
    char     epd_path[512];   /* "" = no --epd training-position output (c2 opcode carries the actual
                                * mover's player name per row — safe for mixed net:/uci: matches) */
    int      uci_move_timeout_ms;
    PlayerDef players[MAX_PLAYERS];
    int      n_players;
} Config;

/* ── Results ────────────────────────────────────────────────────── */
typedef struct {
    long long w, d, l;   /* from player[first]'s perspective */
} PairResult;

typedef enum { SPRT_CONTINUE, SPRT_ACCEPT_H0, SPRT_ACCEPT_H1 } SprtVerdict;

typedef struct {
    long long w, d, l;                 /* pooled across every pairing this player took part in */
    double    pooled_elo, pooled_ci95; /* "field performance" rating — NOT a joint whole-history MLE */
} PlayerRankRow;

typedef struct {
    int         n_players;
    PairResult  pair[MAX_PLAYERS][MAX_PLAYERS];   /* [i][j] = i's W/D/L vs j */
    PlayerRankRow rank[MAX_PLAYERS];               /* indexed by player id (NOT pre-sorted) */

    int         sprt_applicable;   /* 1 iff cfg->sprt && n_players==2 */
    SprtVerdict sprt_verdict;
    double      sprt_llr, sprt_lo, sprt_hi;

    int         total_games;
    double      elapsed_sec;
} ArenaReport;

/* ── Public functions ──────────────────────────────────────────── */

/* One-time process-wide init (board_init/search_init/dummy g_tt).
 * Call exactly once, before the first arena_run(). */
void arena_global_init(void);

/* Play exactly the match described by `cfg` (task list derived from
 * cfg + `openings`), fill `*out` with every pairwise/ranking/SPRT
 * number. Safe to call repeatedly in one process (NOT concurrently
 * with itself) — each call resets all shared match state at entry.
 * Caller is responsible for cfg->players[i].net being already loaded
 * (nnue_net_load()) for every PLAYER_NET entry before calling this.
 *
 * *** ALSO REQUIRED, EASY TO MISS ***: search.c's eval_stm() gates on
 * the process-global nnue_ready() flag (nnue_net_ready(g_nnue_net)),
 * which only the legacy nnue_load()/nnue_load_from_mem() sets — NOT
 * nnue_net_load(). If the caller never touches g_nnue_net, every
 * PLAYER_NET search silently evaluates every position as 0 (confirmed
 * by running an actual match — see arena.c's main() for the one-line
 * fix and the full explanation). After loading at least one player's
 * net, set `if (!g_nnue_net) g_nnue_net = <that net>;` once — this
 * only flips the boolean gate; actual evaluation always reads each
 * player's own NnueAccum::net, so per-player correctness is unaffected. */
void arena_run(const Config *cfg, const OpeningPool *openings, ArenaReport *out);

/* .epd/.fen (one FEN per line) or .pgn (SAN-replayed to a final FEN,
 * first `opening_plies` plies of each game) opening loader. Returns 0
 * on success. Caller frees op->fens[i] then op->fens. */
int load_openings(const char *path, OpeningPool *op, int opening_plies);

/* Trinomial Elo difference + 95% CI — same formula as
 * tests/elo_calc.py's elo_difference(). */
void elo_diff_ci(long long w, long long d, long long l, double *elo, double *ci);

/* Expected score from an Elo value (standard logistic model). */
double score_from_elo(double elo);

/* GSPRT / Wald log-likelihood ratio (trinomial variance) — see
 * arena.c's header "SPRT" for the exact method and citations. */
SprtVerdict sprt_llr(long long w, long long d, long long l,
                      double elo0, double elo1, double alpha, double beta,
                      double *out_llr, double *out_lo, double *out_hi);
