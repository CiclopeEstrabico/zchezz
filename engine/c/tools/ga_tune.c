/* tools/ga_tune.c — native genetic-algorithm tuner for Zchezz's SEARCH
 * constants (implementation plan Appendix F.4 scope: "afinar constantes de
 * busca automaticamente" — the SPSA tuner mentioned there was never built;
 * this is a GA over the same target: search.c's pruning/reduction margins,
 * NOT the NNUE weights, which is train/train_nnue.py's job entirely).
 *
 * ── BUILD ─────────────────────────────────────────────────────────────
 *   From engine/build/:  mingw32-make ENGINE=v400 ga_tune
 *   (output: engine/build/ga_tune.exe — a shared tool binary, not a
 *   per-version artifact, exactly like arena.exe/selfplay.exe.) Direct
 *   gcc line (mirrors arena.c's own header, same flags/libs):
 *
 *     gcc -O3 -ffast-math -D_GNU_SOURCE -std=c11 -mavxvnni -mavx2 \
 *         -I../c/zchezz_v400 -DNO_TABLEBASES -DNO_BOOK -DARENA_NO_MAIN \
 *         -Wno-unused-variable -Wno-unused-but-set-variable \
 *         -Wno-maybe-uninitialized -Wno-misleading-indentation \
 *         -Wno-sign-compare -Wno-unused-function -Wno-parentheses \
 *         -o ga_tune.exe ../c/tools/ga_tune.c ../c/tools/arena.c \
 *         ../c/tools/opening_pool.c ../c/zchezz_v400/board.c \
 *         ../c/zchezz_v400/search.c ../c/zchezz_v400/nnue.c \
 *         -static -lm -pthread
 *
 * ── WHAT THIS TUNES, AND WHY THESE 13 ───────────────────────────────
 *
 *   search.c's alpha_beta()/iterative-deepening loop had its pruning and
 *   reduction margins as bare literals scattered through the function
 *   body (razoring's "+200", NMP's "3 + depth/3", the futility margin
 *   table, the LMR log-log divisor, ...). search.h/search.c now expose
 *   those as a `SearchTunables` struct read from a per-thread `g_tune`
 *   (see search.h's comment on that struct for the full contract) —
 *   THAT refactor is what makes this file possible without touching
 *   search.c's control flow at all; ga_tune.c only ever calls
 *   search_tunables_apply().
 *
 *   The 13 parameters below are every margin/reduction/divisor in
 *   search.c that (a) is a genuine engine-strength knob classical engines
 *   commonly SPSA/GA-tune (Stockfish's own tuning sessions target exactly
 *   this family: RFP/NMP/LMR/futility/aspiration constants) and (b)
 *   reduces to a small number of scalars. Two things were DELIBERATELY
 *   left out:
 *     - lmp_limit[8] (late-move-pruning move-count table): an 8-entry
 *       hand-shaped table with no clean 1-2 parameter formula (unlike
 *       the futility table, which collapses exactly to fut_mult*depth —
 *       see search.c's fut_base comment). Tuning it well needs either
 *       per-element SPSA or a GA genome 8 entries longer for one pruning
 *       rule; out of scope for a first version.
 *     - CONTEMPT (draw-aversion, 15cp): a style knob, not a strength
 *       knob in the same sense — changing it trades off draw rate for
 *       expected score against weaker/stronger opponents asymmetrically,
 *       which this tuner's fixed-baseline fitness function does not
 *       model correctly (see "FITNESS" below).
 *
 * ── WHY THIS IS ITS OWN GAME LOOP, NOT arena_run() ───────────────────
 *
 *   arena.h's Config/arena_run() is explicitly scoped to comparing two
 *   NNUE weight files (net:) or two engine executables (uci:) — there is
 *   no field anywhere in Config for "these two players use different
 *   SEARCH CONSTANTS with the SAME weights". Adding one would mean
 *   editing arena.c itself, which this branch's task explicitly rules
 *   out (another agent owns arena.c's --bin path right now). So per that
 *   file's own header ("EMBEDDABLE AS A LIBRARY"), this file links
 *   arena.c compiled with -DARENA_NO_MAIN and reuses ONLY the pieces that
 *   need no per-player search-constants awareness: arena_global_init()
 *   (board_init/search_init), load_openings()/OpeningPool (arena.h), and
 *   elo_diff_ci() (same trinomial Elo formula as tests/elo_calc.py, so
 *   these numbers are directly comparable to every other tool's).
 *
 *   Game-playing itself is NOT reused: this file's play_one_game() is a
 *   deliberately smaller cousin of arena.c's — net: vs net: ONLY (a GA
 *   tuning our own search never needs a uci: opponent), same NNUE
 *   weights on both sides (only g_tune differs), so no --bin/--epd/--pgn
 *   output, no UCI subprocess plumbing at all. zmalloc32/detect_cpus/
 *   xstrdup/the xorshift RNG are duplicated from arena.c — same
 *   "duplicated on purpose" tradeoff arena.c's own header documents for
 *   its relationship to selfplay.c (small, audited, unlikely to drift).
 *
 *   *** WHAT I WOULD HAVE WANTED TO CHANGE IN arena.c INSTEAD (not done,
 *   see task constraints): add a `SearchTunables tune;` field to
 *   PlayerDef, defaulted to search_tunables_defaults(), and one call to
 *   search_tunables_apply(&pi->def->tune) at the top of play_move_net()
 *   right before search_best(). That would let arena.exe itself run a
 *   tunables-only A/B (and SPRT-gate a GA winner against the previous
 *   incumbent using arena's OWN richer reporting/PGN/JSON), and this
 *   file could then shrink to "build Configs, call arena_run(), read
 *   ArenaReport" like the header comment on arena.h originally
 *   envisioned. Left as a follow-up since it requires editing arena.c. ***
 *
 * ── TT ISOLATION (same invariant as arena.c) ──────────────────────────
 *
 *   Candidate and baseline are adversaries under test, exactly like
 *   arena.c's two players — each gets its OWN TTable, created once per
 *   (worker thread), physically tt_clear()'d before every game. See
 *   CLAUDE.md CRITICAL INVARIANTS: "Arena ISOLATES a TTable per player".
 *   Both share ONE NnueNet (read-only, same weights — see "WHY THIS IS
 *   ITS OWN GAME LOOP" above) but each has its own NnueAccum.
 *
 * ── GA DESIGN ─────────────────────────────────────────────────────────
 *
 *   Genome: 13 doubles, one per SearchTunables field (int fields are
 *   rounded on decode — see PARAM[] table below for bounds/defaults).
 *   Population: CFG_POP_SIZE individuals, generation 0 = defaults +
 *   gaussian jitter (except CFG_ELITE_COUNT clones of the exact default,
 *   so the tuner can never do WORSE than known-good on generation 0).
 *
 *   Fitness: each individual plays CFG_GAMES_PER_EVAL games (antithetic
 *   opening pairing, exactly like arena.c's add_pairing_tasks — every
 *   opening played once with the candidate as White, once as Black,
 *   back to back) against ONE fixed reference genome for that whole
 *   generation: g_baseline. Fitness = (w + 0.5d) / n, the standard
 *   score fraction (0.5 = "as strong as baseline").
 *
 *   Selection: tournament selection, size CFG_TOURNAMENT_K, by fitness.
 *   Crossover: uniform, gene-by-gene, prob CFG_CROSSOVER_RATE per child
 *   (else the child is a straight clone of one tournament winner).
 *   Mutation: per gene, prob CFG_MUTATION_RATE, gaussian noise with
 *   sigma = CFG_MUTATION_SIGMA_FRAC * (max-min) for that gene, clamped
 *   back into [min,max]. Elitism: the top CFG_ELITE_COUNT individuals by
 *   fitness survive unmutated into the next generation.
 *
 *   Baseline ladder (CFG_UPDATE_BASELINE=1, default): after a
 *   generation's fitness is known, if the BEST individual's score
 *   against g_baseline exceeds CFG_BASELINE_PROMOTE_SCORE (default
 *   0.5 — "did at least as well as a coin flip"), g_baseline becomes
 *   that individual's genome for the next generation ("beat the
 *   champion" — same self-play-ladder idea as Appendix F.5's bootstrap
 *   promote/discard gate, just without SPRT — with CFG_GAMES_PER_EVAL
 *   this small the sample size is too thin for a real SPRT, hence a
 *   plain score threshold, not sprt_llr()). Set to 0 to tune everything
 *   against one fixed anchor (search_tunables_defaults()) instead.
 *
 * ── PERSISTENCE ───────────────────────────────────────────────────────
 *
 *   CFG_STATE_PATH: one plain-text file, rewritten (not appended) after
 *   EVERY generation — generation number, g_baseline's genome, then
 *   every individual's genome + fitness + w/d/l. --resume (CFG_RESUME)
 *   loads it back byte-for-byte and continues from generation+1, so a
 *   long run can be killed and restarted without losing progress. See
 *   save_state()/load_state() for the exact format (self-documenting,
 *   first line is a version tag).
 *
 *   CFG_BEST_OUT_PATH: the best individual EVER SEEN (tracked across
 *   every generation, not just the final one) written as a C header —
 *   literally the same #define names search.c's TUNE_*_DEFAULT block
 *   uses, so promoting a tuning result is "paste this file's body over
 *   that block". Also printed to stdout at the end of the run.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <pthread.h>
#include <stdatomic.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "board.h"
#include "search.h"
#include "nnue.h"
#include "arena.h"   /* OpeningPool, load_openings(), arena_global_init(), elo_diff_ci() —
                       * see header "WHY THIS IS ITS OWN GAME LOOP" for what is and is not reused */

/* ═══════════════════════════════════════════════════════════════════
 * ===== CONFIGURATION =====
 * CLAUDE.md rule 8: every CLI-reachable option has its default HERE,
 * named, with units. Running ga_tune with NO arguments does exactly
 * what these constants say; --show-config prints the resolved values
 * and exits without touching disk or playing a game.
 * ═══════════════════════════════════════════════════════════════════ */
#define GA_DEFAULT_NET_PATH          "../c/zchezz_v401/nnue_weights.bin"  /* shared weights, both sides */
#define GA_DEFAULT_OPENINGS_PATH     "../../openings/lines/2moves_LT_1000.pgn" /* "" = startpos only (see arena.c's OPENINGS warning — do not run empty for real tuning, rule 9 */
#define GA_DEFAULT_OPENING_PLIES     4          /* PGN opening file: plies replayed per game before recording FEN */
#define GA_DEFAULT_POP_SIZE          32         /* individuals per generation */
#define GA_DEFAULT_GENERATIONS       20         /* GA generations to run */
#define GA_DEFAULT_GAMES_PER_EVAL    8           /* games per individual per generation, vs g_baseline (antithetic pairs) */
#define GA_DEFAULT_MOVETIME_MS       10          /* per-move time budget, ms — deliberately tiny; a tuning run needs
                                                    * MANY games, not deep ones (same tradeoff Stockfish SPSA sessions
                                                    * make: short TC, huge game count) */
#define GA_DEFAULT_MAX_PLIES         200        /* game-length safety cap -> counted as draw */
#define GA_DEFAULT_THREADS           0          /* concurrent games; 0 = autodetect logical cores */
#define GA_DEFAULT_TT_MB             4.0        /* per-PlayerSlot TTable memory budget, MB (small: many short games) */
#define GA_DEFAULT_SEED              1          /* RNG seed — opening cycling AND all GA randomness */
#define GA_DEFAULT_ELITE_COUNT       2          /* top-N individuals copied unmutated into the next generation */
#define GA_DEFAULT_TOURNAMENT_K      3          /* tournament-selection group size */
#define GA_DEFAULT_MUTATION_RATE     0.15       /* probability ANY given gene mutates */
#define GA_DEFAULT_MUTATION_SIGMA    0.15       /* gaussian mutation sigma, as a FRACTION of that gene's (max-min) range */
#define GA_DEFAULT_CROSSOVER_RATE    0.70       /* probability a child is uniform-crossed (else a straight clone) */
#define GA_DEFAULT_UPDATE_BASELINE   1          /* 1 = best-of-generation replaces g_baseline when it beats it ("ladder") */
#define GA_DEFAULT_BASELINE_PROMOTE  0.50       /* score fraction (vs g_baseline) needed to promote, when the above is 1 */
#define GA_DEFAULT_STATE_PATH        "ga_tune_state.txt"      /* population+baseline checkpoint, rewritten every generation */
#define GA_DEFAULT_RESUME            0          /* 1 = load GA_STATE_PATH if present and continue past its generation */
#define GA_DEFAULT_BEST_OUT_PATH     "ga_best_params.h"        /* best-ever genome, as paste-back-ready C #defines */
/* ═════════════════════════════════════════════════════════════════ */

#define GA_STR(x)  #x
#define GA_XSTR(x) GA_STR(x)

typedef struct {
    char   net_path[512];
    char   openings_path[512];
    int    opening_plies;
    int    pop_size;
    int    generations;
    int    games_per_eval;
    int    movetime_ms;
    int    max_plies;
    int    threads;
    double tt_mb;
    uint64_t seed;
    int    elite_count;
    int    tournament_k;
    double mutation_rate;
    double mutation_sigma;
    double crossover_rate;
    int    update_baseline;
    double baseline_promote;
    char   state_path[512];
    int    resume;
    char   best_out_path[512];
} GaConfig;

static void ga_config_defaults(GaConfig *c) {
    memset(c, 0, sizeof(*c));
    strncpy(c->net_path, GA_DEFAULT_NET_PATH, sizeof(c->net_path) - 1);
    strncpy(c->openings_path, GA_DEFAULT_OPENINGS_PATH, sizeof(c->openings_path) - 1);
    c->opening_plies   = GA_DEFAULT_OPENING_PLIES;
    c->pop_size        = GA_DEFAULT_POP_SIZE;
    c->generations     = GA_DEFAULT_GENERATIONS;
    c->games_per_eval  = GA_DEFAULT_GAMES_PER_EVAL;
    c->movetime_ms     = GA_DEFAULT_MOVETIME_MS;
    c->max_plies       = GA_DEFAULT_MAX_PLIES;
    c->threads         = GA_DEFAULT_THREADS;
    c->tt_mb           = GA_DEFAULT_TT_MB;
    c->seed            = GA_DEFAULT_SEED;
    c->elite_count     = GA_DEFAULT_ELITE_COUNT;
    c->tournament_k    = GA_DEFAULT_TOURNAMENT_K;
    c->mutation_rate   = GA_DEFAULT_MUTATION_RATE;
    c->mutation_sigma  = GA_DEFAULT_MUTATION_SIGMA;
    c->crossover_rate  = GA_DEFAULT_CROSSOVER_RATE;
    c->update_baseline = GA_DEFAULT_UPDATE_BASELINE;
    c->baseline_promote= GA_DEFAULT_BASELINE_PROMOTE;
    strncpy(c->state_path, GA_DEFAULT_STATE_PATH, sizeof(c->state_path) - 1);
    c->resume          = GA_DEFAULT_RESUME;
    strncpy(c->best_out_path, GA_DEFAULT_BEST_OUT_PATH, sizeof(c->best_out_path) - 1);
}

/* ═══════════════════════════════════════════════════════════════════
 * Small helpers duplicated from arena.c — see header "WHY THIS IS ITS
 * OWN GAME LOOP", same "duplicated on purpose" tradeoff.
 * ═══════════════════════════════════════════════════════════════════ */
static void *zmalloc32(size_t bytes) {
    void *ptr = NULL;
#if defined(_WIN32)
    ptr = _aligned_malloc(bytes, 32);
#elif defined(__APPLE__) || defined(__linux__)
    if (posix_memalign(&ptr, 32, bytes) != 0) ptr = NULL;
#else
    ptr = malloc(bytes);
#endif
    return ptr;
}
static void zfree32(void *ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}
#if defined(_WIN32)
static int detect_cpus(void) {
    const char *e = getenv("NUMBER_OF_PROCESSORS");
    int n = e ? atoi(e) : 0;
    return n > 0 ? n : 4;
}
#else
static int detect_cpus(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 4;
}
#endif

/* xorshift64* RNG — one per worker/GA-thread purpose, seeded from
 * GaConfig.seed so a whole run is reproducible. Same algorithm as
 * arena.c's Rng (not the same instance — duplicated for the same
 * "small, audited, unlikely to drift" reason as zmalloc32 above). */
typedef struct { uint64_t s; } Rng;
static inline uint64_t rng_next_u64(Rng *r) {
    uint64_t x = r->s;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    r->s = x;
    return x * 0x2545F4914F6CDD1DULL;
}
static void rng_seed(Rng *r, uint64_t seed) {
    uint64_t s = seed ^ 0xD1B54A32D192ED03ULL;
    for (int i = 0; i < 4; i++) {
        s += 0x9E3779B97F4A7C15ULL;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        s = z;
    }
    r->s = s ? s : 0xD1B54A32D192ED03ULL;
}
static double rng_uniform(Rng *r) {   /* [0,1) */
    return (double)(rng_next_u64(r) >> 11) * (1.0 / 9007199254740992.0);
}
/* Box-Muller, one deviate per call (discards the paired second sample —
 * simplicity over throughput; this is called a few hundred times per
 * generation, not per node). */
static double rng_gaussian(Rng *r) {
    double u1 = rng_uniform(r), u2 = rng_uniform(r);
    if (u1 < 1e-12) u1 = 1e-12;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* ═══════════════════════════════════════════════════════════════════
 * Genome <-> SearchTunables — the 13 tuned parameters, their bounds,
 * and whether they decode to an int or a double field. See header
 * "WHAT THIS TUNES, AND WHY THESE 13" for the selection rationale.
 * ═══════════════════════════════════════════════════════════════════ */
#define N_PARAMS 13

typedef struct {
    const char *name;      /* matches search.h's SearchTunables field name */
    const char *macro;     /* matches search.c's TUNE_*_DEFAULT macro name, for the paste-back header */
    double      lo, hi;    /* mutation/clamp bounds */
    double      dflt;      /* search_tunables_defaults() value, as a double */
    int         is_int;    /* 1 = round to nearest int on decode; 0 = keep as double (lmr_divisor only) */
} ParamDesc;

static const ParamDesc PARAM[N_PARAMS] = {
    { "razor_margin",             "TUNE_RAZOR_MARGIN_DEFAULT",              50,  400, 200, 1 },
    { "rfp_mult",                 "TUNE_RFP_MULT_DEFAULT",                  30,  200,  90, 1 },
    { "rfp_improving_bonus",      "TUNE_RFP_IMPROVING_BONUS_DEFAULT",        0,  150,  50, 1 },
    { "nmp_base",                 "TUNE_NMP_BASE_DEFAULT",                   1,    5,   3, 1 },
    { "nmp_depth_div",            "TUNE_NMP_DEPTH_DIV_DEFAULT",              1,    6,   3, 1 },
    { "nmp_max_r",                "TUNE_NMP_MAX_R_DEFAULT",                  2,   10,   6, 1 },
    { "nmp_eval_bonus_threshold", "TUNE_NMP_EVAL_BONUS_THRESHOLD_DEFAULT",  50,  400, 200, 1 },
    { "probcut_margin",           "TUNE_PROBCUT_MARGIN_DEFAULT",            50,  400, 200, 1 },
    { "lmr_divisor",              "TUNE_LMR_DIVISOR_DEFAULT",              0.5,  3.0, 1.5, 0 },
    { "fut_mult",                 "TUNE_FUT_MULT_DEFAULT",                  50,  300, 150, 1 },
    { "fut_improving_adj",        "TUNE_FUT_IMPROVING_ADJ_DEFAULT",          0,  150,  50, 1 },
    { "asp_delta_init",           "TUNE_ASP_DELTA_INIT_DEFAULT",             5,   60,  20, 1 },
    { "asp_delta_max",            "TUNE_ASP_DELTA_MAX_DEFAULT",            100, 1000, 500, 1 },
};

typedef double Genome[N_PARAMS];

static void genome_defaults(Genome g) {
    for (int i = 0; i < N_PARAMS; i++) g[i] = PARAM[i].dflt;
}
static void genome_clamp(Genome g) {
    for (int i = 0; i < N_PARAMS; i++) {
        if (g[i] < PARAM[i].lo) g[i] = PARAM[i].lo;
        if (g[i] > PARAM[i].hi) g[i] = PARAM[i].hi;
    }
}
static void genome_to_tunables(const Genome g, SearchTunables *t) {
    /* Field order matches PARAM[]/N_PARAMS above exactly — kept in one
     * place (this function) so a mismatch is a one-spot fix, not a
     * silent index-shift bug scattered through the GA operators. */
    t->razor_margin              = (int)lround(g[0]);
    t->rfp_mult                  = (int)lround(g[1]);
    t->rfp_improving_bonus       = (int)lround(g[2]);
    t->nmp_base                  = (int)lround(g[3]);
    t->nmp_depth_div             = (int)lround(g[4]); if (t->nmp_depth_div < 1) t->nmp_depth_div = 1;
    t->nmp_max_r                 = (int)lround(g[5]);
    t->nmp_eval_bonus_threshold  = (int)lround(g[6]);
    t->probcut_margin            = (int)lround(g[7]);
    t->lmr_divisor                = g[8]; if (t->lmr_divisor < 0.1) t->lmr_divisor = 0.1;
    t->fut_mult                  = (int)lround(g[9]);
    t->fut_improving_adj         = (int)lround(g[10]);
    t->asp_delta_init            = (int)lround(g[11]);
    t->asp_delta_max             = (int)lround(g[12]);
}

/* ═══════════════════════════════════════════════════════════════════
 * Population
 * ═══════════════════════════════════════════════════════════════════ */
typedef struct {
    Genome genome;
    double fitness;         /* (w + 0.5d) / n vs g_baseline this generation; -1 = not yet evaluated */
    long long w, d, l;
} Individual;

static Individual *g_pop = NULL;      /* current generation, GA_DEFAULT_POP_SIZE entries */
static Genome      g_baseline;        /* this generation's fixed opponent — see "GA DESIGN" */
static Genome      g_best_ever;       /* best individual seen across the whole run */
static double      g_best_ever_fitness = -1.0;
static int         g_generation = 0;

/* ═══════════════════════════════════════════════════════════════════
 * PlayerSlot — one worker thread's private candidate+baseline search
 * resources. See header "TT ISOLATION" — tt/nnue/ss are never shared
 * between candidate and baseline, or between two worker threads.
 * ═══════════════════════════════════════════════════════════════════ */
typedef struct {
    NnueAccum   *nnue;
    SearchState *ss;
    TTable      *tt;
} PlayerSlot;

static PlayerSlot *slot_create(size_t tt_entries, const NnueNet *net) {
    PlayerSlot *p = (PlayerSlot *)calloc(1, sizeof(PlayerSlot));
    p->nnue = (NnueAccum *)zmalloc32(sizeof(NnueAccum));
    memset(p->nnue, 0, sizeof(NnueAccum));
    p->nnue->net = net;
    p->ss = search_state_new();
    p->tt = tt_create(tt_entries);
    return p;
}
static void slot_destroy(PlayerSlot *p) {
    if (!p) return;
    if (p->nnue) zfree32(p->nnue);
    if (p->ss)   search_state_free(p->ss);
    if (p->tt)   tt_destroy(p->tt);
    free(p);
}

/* Ask a PlayerSlot for its move at the current position, searching
 * under `tune` (applied to THIS thread's g_tune right before
 * search_best() — see search.h's SearchTunables comment). */
static int play_move(PlayerSlot *p, Board *b, const SearchTunables *tune,
                      int movetime_ms, Move *out) {
    search_tunables_apply(tune);
    SearchParams sp; memset(&sp, 0, sizeof(sp));
    sp.max_depth      = MAX_PLY - 1;
    sp.start_depth    = 0;
    sp.time_limit_ms  = movetime_ms;
    sp.node_limit     = 0;
    sp.multi_pv       = 1;
    sp.threads        = 1;     /* one OS thread per game slot, like arena.c */
    sp.stop           = NULL;
    sp.search_state   = p->ss;
    sp.info_cb        = NULL;
    sp.tt             = p->tt;
    sp.mpv_share_budget = 0;

    SearchResult res = search_best(b, &sp);
    if (res.bests[0].from == 0 && res.bests[0].to == 0) return 0;
    *out = res.bests[0];
    return 1;
}

typedef enum { OUT_DRAW = 0, OUT_WHITE = 1, OUT_BLACK = -1 } Outcome;

/* Plays one game, candidate (genome `cand_tune`) as White iff
 * `cand_white`, against baseline (`base_tune`). Same shape as arena.c's
 * play_one_game() minus every non-net:/non-training-output concern (see
 * header "WHY THIS IS ITS OWN GAME LOOP"). game_undo/game_undo_top: one
 * per-WORKER undo stack, reused across games (mirrors arena.c's
 * worker_fn — sized for the whole game plus the deepest search on top
 * of it, see that file's comment on the same allocation for why
 * STACK_SIZE alone is not enough). */
static Outcome play_one_game(PlayerSlot *cand, PlayerSlot *base, int cand_white,
                              const char *start_fen, const SearchTunables *cand_tune,
                              const SearchTunables *base_tune, int movetime_ms, int max_plies,
                              UndoFrame *game_undo, int *game_undo_top, long long *out_plies) {
    Board board;
    board_load_fen(&board, start_fen);
    *game_undo_top = 0;
    board.undo     = game_undo;
    board.undo_top = game_undo_top;

    tt_clear(cand->tt);
    tt_clear(base->tt);
    /* v4.02: wipe ordering state per game too — search_reset() no longer
     * ages history per move, so tables would otherwise persist across the
     * whole GA run (same reasoning as arena.c's per-game reset). */
    search_clear_ordering(cand->ss);
    search_clear_ordering(base->ss);

    PlayerSlot *white = cand_white ? cand : base;
    PlayerSlot *black = cand_white ? base : cand;
    const SearchTunables *white_tune = cand_white ? cand_tune : base_tune;
    const SearchTunables *black_tune = cand_white ? base_tune : cand_tune;

    Outcome result = OUT_DRAW;
    long long ply;
    for (ply = 0; ply < max_plies; ply++) {
        int dr = board_is_draw(&board);
        if (dr == 1 || dr == 2) { result = OUT_DRAW; break; }

        int white_to_move = (board.turn == COL_W);
        PlayerSlot *mover = white_to_move ? white : black;
        const SearchTunables *mover_tune = white_to_move ? white_tune : black_tune;

        board.nnue = mover->nnue;   /* rebind accumulator to the mover's own private state
                                      * (undo stack stays shared — see arena.c's
                                      * net_instance_bind_board() comment for why the
                                      * accumulator alone needs this and undo does not) */

        Move mv;
        int ok = play_move(mover, &board, mover_tune, movetime_ms, &mv);
        if (!ok) {
            result = board_in_check(&board) ? (white_to_move ? OUT_BLACK : OUT_WHITE) : OUT_DRAW;
            break;
        }
        board_make(&board, &mv);
    }
    if (ply >= max_plies) result = OUT_DRAW;
    *out_plies = ply;
    return result;
}

/* ═══════════════════════════════════════════════════════════════════
 * Task list — one task per (individual, game-within-eval), antithetic
 * pairing exactly like arena.c's add_pairing_tasks: every opening
 * played once with the candidate as White, once as Black, back to back.
 * ═══════════════════════════════════════════════════════════════════ */
typedef struct {
    int indiv_idx;
    int opening_idx;   /* -1 = startpos */
    int cand_white;
} GaTask;

static GaTask *g_tasks = NULL;
static int     g_n_tasks = 0;
static atomic_int g_next_task;
static const GaConfig   *g_cfg;
static const OpeningPool *g_openings;
static const NnueNet    *g_net;
static size_t             g_tt_entries;
static pthread_mutex_t    g_results_mtx = PTHREAD_MUTEX_INITIALIZER;

static void build_tasks(void) {
    int per_indiv = g_cfg->games_per_eval;
    int cap = g_cfg->pop_size * per_indiv;
    g_tasks = (GaTask *)malloc((size_t)cap * sizeof(GaTask));
    g_n_tasks = 0;
    int n_openings = g_openings->n;
    for (int i = 0; i < g_cfg->pop_size; i++) {
        int pairs = per_indiv / 2;
        for (int k = 0; k < pairs; k++) {
            int op = n_openings > 0 ? (k % n_openings) : -1;
            g_tasks[g_n_tasks++] = (GaTask){ i, op, 1 };
            g_tasks[g_n_tasks++] = (GaTask){ i, op, 0 };
        }
        if (per_indiv & 1) {
            int op = n_openings > 0 ? (pairs % n_openings) : -1;
            g_tasks[g_n_tasks++] = (GaTask){ i, op, 1 };
        }
    }
}

static const char *GA_STARTPOS_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

typedef struct { int id; } WorkerArg;

static void *worker_fn(void *arg) {
    WorkerArg *wa = (WorkerArg *)arg;
    PlayerSlot *cand = slot_create(g_tt_entries, g_net);
    PlayerSlot *base = slot_create(g_tt_entries, g_net);
    size_t undo_cap = (size_t)g_cfg->max_plies + MAX_PLY + 64;
    UndoFrame *game_undo = (UndoFrame *)malloc(undo_cap * sizeof(UndoFrame));
    int game_undo_top = 0;

    SearchTunables base_tune;
    genome_to_tunables(g_baseline, &base_tune);

    for (;;) {
        int idx = atomic_fetch_add(&g_next_task, 1);
        if (idx >= g_n_tasks) break;
        GaTask *t = &g_tasks[idx];

        SearchTunables cand_tune;
        genome_to_tunables(g_pop[t->indiv_idx].genome, &cand_tune);

        const char *fen = (t->opening_idx >= 0 && g_openings->n > 0)
            ? g_openings->fens[t->opening_idx] : GA_STARTPOS_FEN;

        long long plies = 0;
        Outcome o = play_one_game(cand, base, t->cand_white, fen, &cand_tune, &base_tune,
                                   g_cfg->movetime_ms, g_cfg->max_plies,
                                   game_undo, &game_undo_top, &plies);

        int cand_won  = (t->cand_white && o == OUT_WHITE) || (!t->cand_white && o == OUT_BLACK);
        int base_won  = (t->cand_white && o == OUT_BLACK) || (!t->cand_white && o == OUT_WHITE);
        pthread_mutex_lock(&g_results_mtx);
        if (o == OUT_DRAW) g_pop[t->indiv_idx].d++;
        else if (cand_won) g_pop[t->indiv_idx].w++;
        else if (base_won) g_pop[t->indiv_idx].l++;
        pthread_mutex_unlock(&g_results_mtx);
    }

    slot_destroy(cand);
    slot_destroy(base);
    free(game_undo);
    (void)wa;
    return NULL;
}

/* Plays every task for the current generation, filling g_pop[i].{w,d,l,fitness}. */
static void evaluate_generation(int threads) {
    for (int i = 0; i < g_cfg->pop_size; i++) { g_pop[i].w = g_pop[i].d = g_pop[i].l = 0; }
    build_tasks();
    atomic_store(&g_next_task, 0);

    pthread_t *tids = (pthread_t *)malloc((size_t)threads * sizeof(pthread_t));
    WorkerArg *was  = (WorkerArg *)malloc((size_t)threads * sizeof(WorkerArg));
    for (int i = 0; i < threads; i++) { was[i].id = i; pthread_create(&tids[i], NULL, worker_fn, &was[i]); }
    for (int i = 0; i < threads; i++) pthread_join(tids[i], NULL);
    free(tids); free(was);
    free(g_tasks); g_tasks = NULL;

    for (int i = 0; i < g_cfg->pop_size; i++) {
        long long n = g_pop[i].w + g_pop[i].d + g_pop[i].l;
        g_pop[i].fitness = n > 0 ? (g_pop[i].w + 0.5 * g_pop[i].d) / (double)n : 0.5;
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * GA operators
 * ═══════════════════════════════════════════════════════════════════ */
static int cmp_fitness_desc(const void *a, const void *b) {
    double fa = ((const Individual *)a)->fitness, fb = ((const Individual *)b)->fitness;
    return fa < fb ? 1 : (fa > fb ? -1 : 0);
}

static int tournament_select(Rng *rng, int k) {
    int best = -1; double best_fit = -1.0;
    for (int i = 0; i < k; i++) {
        int idx = (int)(rng_uniform(rng) * g_cfg->pop_size);
        if (idx >= g_cfg->pop_size) idx = g_cfg->pop_size - 1;
        if (g_pop[idx].fitness > best_fit) { best_fit = g_pop[idx].fitness; best = idx; }
    }
    return best;
}

static void crossover(Rng *rng, const Genome a, const Genome b, Genome out) {
    if (rng_uniform(rng) >= g_cfg->crossover_rate) { memcpy(out, a, sizeof(Genome)); return; }
    for (int i = 0; i < N_PARAMS; i++) out[i] = (rng_uniform(rng) < 0.5) ? a[i] : b[i];
}

static void mutate(Rng *rng, Genome g) {
    for (int i = 0; i < N_PARAMS; i++) {
        if (rng_uniform(rng) >= g_cfg->mutation_rate) continue;
        double sigma = g_cfg->mutation_sigma * (PARAM[i].hi - PARAM[i].lo);
        g[i] += rng_gaussian(rng) * sigma;
    }
    genome_clamp(g);
}

/* Builds the next generation IN PLACE (a scratch copy avoids reading a
 * parent that was already overwritten). Elitism: the top
 * cfg->elite_count individuals (by fitness, already sorted by the
 * caller) survive verbatim. */
static void evolve(Rng *rng) {
    Individual *next = (Individual *)malloc((size_t)g_cfg->pop_size * sizeof(Individual));
    int e = g_cfg->elite_count; if (e > g_cfg->pop_size) e = g_cfg->pop_size;
    for (int i = 0; i < e; i++) {
        memcpy(next[i].genome, g_pop[i].genome, sizeof(Genome));
        next[i].fitness = -1; next[i].w = next[i].d = next[i].l = 0;
    }
    for (int i = e; i < g_cfg->pop_size; i++) {
        int pa = tournament_select(rng, g_cfg->tournament_k);
        int pb = tournament_select(rng, g_cfg->tournament_k);
        crossover(rng, g_pop[pa].genome, g_pop[pb].genome, next[i].genome);
        mutate(rng, next[i].genome);
        next[i].fitness = -1; next[i].w = next[i].d = next[i].l = 0;
    }
    free(g_pop);
    g_pop = next;
}

/* ═══════════════════════════════════════════════════════════════════
 * Persistence — plain text, rewritten every generation. See header
 * "PERSISTENCE" for the resume contract.
 * ═══════════════════════════════════════════════════════════════════ */
static void save_state(const GaConfig *cfg) {
    FILE *f = fopen(cfg->state_path, "w");
    if (!f) { fprintf(stderr, "[ga_tune] WARNING: could not write state file '%s'\n", cfg->state_path); return; }
    fprintf(f, "GA_TUNE_STATE v1\n");
    fprintf(f, "generation %d\n", g_generation);
    fprintf(f, "baseline");
    for (int i = 0; i < N_PARAMS; i++) fprintf(f, " %.10g", g_baseline[i]);
    fprintf(f, "\n");
    fprintf(f, "best_ever_fitness %.10g\n", g_best_ever_fitness);
    fprintf(f, "best_ever");
    for (int i = 0; i < N_PARAMS; i++) fprintf(f, " %.10g", g_best_ever[i]);
    fprintf(f, "\n");
    fprintf(f, "population %d\n", cfg->pop_size);
    for (int i = 0; i < cfg->pop_size; i++) {
        fprintf(f, "individual %d fitness %.10g w %lld d %lld l %lld genome", i,
                g_pop[i].fitness, g_pop[i].w, g_pop[i].d, g_pop[i].l);
        for (int j = 0; j < N_PARAMS; j++) fprintf(f, " %.10g", g_pop[i].genome[j]);
        fprintf(f, "\n");
    }
    fclose(f);
}

/* Returns 1 if a state file was found and loaded (population resized to
 * match cfg->pop_size if the file has fewer/more, filling any gap with
 * fresh gaussian-jittered defaults), 0 if no file / parse failure (the
 * caller falls back to a fresh generation 0). */
static int load_state(const GaConfig *cfg, Rng *rng) {
    FILE *f = fopen(cfg->state_path, "r");
    if (!f) return 0;
    char tag[64];
    if (fscanf(f, "%63s", tag) != 1 || strcmp(tag, "GA_TUNE_STATE")) { fclose(f); return 0; }
    fscanf(f, "%*s"); /* version token "v1" */
    char kw[32];
    if (fscanf(f, "%31s %d", kw, &g_generation) != 2) { fclose(f); return 0; }
    fscanf(f, "%31s", kw); for (int i = 0; i < N_PARAMS; i++) fscanf(f, "%lf", &g_baseline[i]);
    fscanf(f, "%31s %lf", kw, &g_best_ever_fitness);
    fscanf(f, "%31s", kw); for (int i = 0; i < N_PARAMS; i++) fscanf(f, "%lf", &g_best_ever[i]);
    int saved_pop = 0;
    fscanf(f, "%31s %d", kw, &saved_pop);
    for (int i = 0; i < cfg->pop_size; i++) {
        if (i < saved_pop) {
            int idx; long long w, d, l;
            fscanf(f, "%31s %d %31s %lf %31s %lld %31s %lld %31s %lld %31s",
                   kw, &idx, kw, &g_pop[i].fitness, kw, &w, kw, &d, kw, &l, kw);
            g_pop[i].w = w; g_pop[i].d = d; g_pop[i].l = l;
            for (int j = 0; j < N_PARAMS; j++) fscanf(f, "%lf", &g_pop[i].genome[j]);
        } else {
            genome_defaults(g_pop[i].genome);
            mutate(rng, g_pop[i].genome);
            g_pop[i].fitness = -1; g_pop[i].w = g_pop[i].d = g_pop[i].l = 0;
        }
    }
    /* Any extra saved individuals beyond cfg->pop_size are simply not
     * read further — fscanf just stops; harmless, the file gets
     * truncated to the new pop_size on the next save_state(). */
    fclose(f);
    g_generation += 1;   /* resume STARTS the generation after the one that was saved */
    return 1;
}

/* Best-ever genome, written as a C header whose #define names match
 * search.c's TUNE_*_DEFAULT block verbatim — see PARAM[].macro. */
static void write_best_header(const GaConfig *cfg) {
    FILE *f = fopen(cfg->best_out_path, "w");
    if (!f) { fprintf(stderr, "[ga_tune] WARNING: could not write '%s'\n", cfg->best_out_path); return; }
    fprintf(f, "/* ga_tune.c output — best genome found, fitness %.4f vs its baseline at the\n"
               " * time it was recorded (see ga_tune_state.txt for the run this came from).\n"
               " * Paste this block over the TUNE_*_DEFAULT block near the top of search.c\n"
               " * (search for \"Tunable search constants — defaults\") to promote it. */\n",
               g_best_ever_fitness);
    SearchTunables t; genome_to_tunables(g_best_ever, &t);
    for (int i = 0; i < N_PARAMS; i++) {
        if (PARAM[i].is_int) {
            int v = (int)lround(g_best_ever[i]);
            fprintf(f, "#define %-40s %d\n", PARAM[i].macro, v);
        } else {
            fprintf(f, "#define %-40s %.4f\n", PARAM[i].macro, g_best_ever[i]);
        }
    }
    fclose(f);
    (void)t;
}

/* ═══════════════════════════════════════════════════════════════════
 * CLI
 * ═══════════════════════════════════════════════════════════════════ */
static void print_usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  Tunes zchezz_v400/search.c's pruning/reduction constants with a genetic\n"
        "  algorithm, playing candidate-vs-baseline games in-process (own NNUE\n"
        "  weights, own TTable per side — see this file's header comment).\n"
        "  Every option below has a default in the CONFIGURATION block at the top\n"
        "  of ga_tune.c; running with no arguments uses those defaults.\n\n"
        "  --net PATH              shared NNUE weights, both sides (default " GA_DEFAULT_NET_PATH ")\n"
        "  --openings FILE         .epd/.fen or .pgn opening file, \"\" = startpos only\n"
        "                          (default " GA_DEFAULT_OPENINGS_PATH ")\n"
        "  --opening-plies N       PGN opening replay depth (default " GA_XSTR(GA_DEFAULT_OPENING_PLIES) ")\n"
        "  --pop N                 population size (default " GA_XSTR(GA_DEFAULT_POP_SIZE) ")\n"
        "  --generations N         GA generations to run (default " GA_XSTR(GA_DEFAULT_GENERATIONS) ")\n"
        "  --games N               games per individual per generation (default " GA_XSTR(GA_DEFAULT_GAMES_PER_EVAL) ")\n"
        "  --movetime MS           per-move budget, ms (default " GA_XSTR(GA_DEFAULT_MOVETIME_MS) ")\n"
        "  --max-plies N           game-length safety cap (default " GA_XSTR(GA_DEFAULT_MAX_PLIES) ")\n"
        "  --threads N             concurrent games (default = logical cores)\n"
        "  --tt-mb MB              per-side TTable memory (default " GA_XSTR(GA_DEFAULT_TT_MB) ")\n"
        "  --seed N                RNG seed (default " GA_XSTR(GA_DEFAULT_SEED) ")\n"
        "  --elite N               elitism count (default " GA_XSTR(GA_DEFAULT_ELITE_COUNT) ")\n"
        "  --tournament-k N        tournament selection size (default " GA_XSTR(GA_DEFAULT_TOURNAMENT_K) ")\n"
        "  --mutation-rate F       per-gene mutation probability (default " GA_XSTR(GA_DEFAULT_MUTATION_RATE) ")\n"
        "  --mutation-sigma F      gaussian sigma, fraction of gene range (default " GA_XSTR(GA_DEFAULT_MUTATION_SIGMA) ")\n"
        "  --crossover-rate F      probability of uniform crossover (default " GA_XSTR(GA_DEFAULT_CROSSOVER_RATE) ")\n"
        "  --update-baseline 0|1   best-of-gen replaces baseline on promotion (default " GA_XSTR(GA_DEFAULT_UPDATE_BASELINE) ")\n"
        "  --baseline-promote F    score fraction needed to promote (default " GA_XSTR(GA_DEFAULT_BASELINE_PROMOTE) ")\n"
        "  --state PATH            checkpoint file (default " GA_DEFAULT_STATE_PATH ")\n"
        "  --resume                load --state if present and continue\n"
        "  --best-out PATH         best-genome C header output (default " GA_DEFAULT_BEST_OUT_PATH ")\n"
        "  --show-config           print resolved config and exit (no games played)\n"
        "  --help                  this text\n",
        argv0);
}

static void print_config(const GaConfig *c) {
    printf("net_path          = %s\n", c->net_path);
    printf("openings_path     = %s\n", c->openings_path);
    printf("opening_plies     = %d\n", c->opening_plies);
    printf("pop_size          = %d\n", c->pop_size);
    printf("generations       = %d\n", c->generations);
    printf("games_per_eval    = %d\n", c->games_per_eval);
    printf("movetime_ms       = %d\n", c->movetime_ms);
    printf("max_plies         = %d\n", c->max_plies);
    printf("threads           = %d (0 = autodetect -> %d)\n", c->threads, detect_cpus());
    printf("tt_mb             = %.2f\n", c->tt_mb);
    printf("seed              = %llu\n", (unsigned long long)c->seed);
    printf("elite_count       = %d\n", c->elite_count);
    printf("tournament_k      = %d\n", c->tournament_k);
    printf("mutation_rate     = %.4f\n", c->mutation_rate);
    printf("mutation_sigma    = %.4f\n", c->mutation_sigma);
    printf("crossover_rate    = %.4f\n", c->crossover_rate);
    printf("update_baseline   = %d\n", c->update_baseline);
    printf("baseline_promote  = %.4f\n", c->baseline_promote);
    printf("state_path        = %s\n", c->state_path);
    printf("resume            = %d\n", c->resume);
    printf("best_out_path     = %s\n", c->best_out_path);
    printf("--- tuned parameters (%d) ---\n", N_PARAMS);
    for (int i = 0; i < N_PARAMS; i++)
        printf("  %-26s [%g .. %g] default %g (%s)\n", PARAM[i].name, PARAM[i].lo, PARAM[i].hi,
               PARAM[i].dflt, PARAM[i].is_int ? "int" : "double");
}

static int parse_args(int argc, char **argv, GaConfig *c) {
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        #define NEXT() (++i < argc ? argv[i] : NULL)
        if      (!strcmp(a, "--help") || !strcmp(a, "-h")) { print_usage(argv[0]); return 1; }
        else if (!strcmp(a, "--show-config")) { print_config(c); return 1; }
        else if (!strcmp(a, "--net")) { const char *v = NEXT(); if (v) strncpy(c->net_path, v, sizeof(c->net_path)-1); }
        else if (!strcmp(a, "--openings")) { const char *v = NEXT(); if (v) strncpy(c->openings_path, v, sizeof(c->openings_path)-1); }
        else if (!strcmp(a, "--opening-plies")) { const char *v = NEXT(); if (v) c->opening_plies = atoi(v); }
        else if (!strcmp(a, "--pop")) { const char *v = NEXT(); if (v) c->pop_size = atoi(v); }
        else if (!strcmp(a, "--generations")) { const char *v = NEXT(); if (v) c->generations = atoi(v); }
        else if (!strcmp(a, "--games")) { const char *v = NEXT(); if (v) c->games_per_eval = atoi(v); }
        else if (!strcmp(a, "--movetime")) { const char *v = NEXT(); if (v) c->movetime_ms = atoi(v); }
        else if (!strcmp(a, "--max-plies")) { const char *v = NEXT(); if (v) c->max_plies = atoi(v); }
        else if (!strcmp(a, "--threads")) { const char *v = NEXT(); if (v) c->threads = atoi(v); }
        else if (!strcmp(a, "--tt-mb")) { const char *v = NEXT(); if (v) c->tt_mb = atof(v); }
        else if (!strcmp(a, "--seed")) { const char *v = NEXT(); if (v) c->seed = strtoull(v, NULL, 10); }
        else if (!strcmp(a, "--elite")) { const char *v = NEXT(); if (v) c->elite_count = atoi(v); }
        else if (!strcmp(a, "--tournament-k")) { const char *v = NEXT(); if (v) c->tournament_k = atoi(v); }
        else if (!strcmp(a, "--mutation-rate")) { const char *v = NEXT(); if (v) c->mutation_rate = atof(v); }
        else if (!strcmp(a, "--mutation-sigma")) { const char *v = NEXT(); if (v) c->mutation_sigma = atof(v); }
        else if (!strcmp(a, "--crossover-rate")) { const char *v = NEXT(); if (v) c->crossover_rate = atof(v); }
        else if (!strcmp(a, "--update-baseline")) { const char *v = NEXT(); if (v) c->update_baseline = atoi(v); }
        else if (!strcmp(a, "--baseline-promote")) { const char *v = NEXT(); if (v) c->baseline_promote = atof(v); }
        else if (!strcmp(a, "--state")) { const char *v = NEXT(); if (v) strncpy(c->state_path, v, sizeof(c->state_path)-1); }
        else if (!strcmp(a, "--resume")) { c->resume = 1; }
        else if (!strcmp(a, "--best-out")) { const char *v = NEXT(); if (v) strncpy(c->best_out_path, v, sizeof(c->best_out_path)-1); }
        else { fprintf(stderr, "[ga_tune] unknown option '%s' (see --help)\n", a); return 1; }
        #undef NEXT
    }
    return 0;
}

int main(int argc, char **argv) {
    GaConfig cfg; ga_config_defaults(&cfg);
    int early_exit = parse_args(argc, argv, &cfg);
    if (early_exit) return 0;

    if (cfg.threads <= 0) cfg.threads = detect_cpus();
    if (cfg.pop_size < 2) { fprintf(stderr, "[ga_tune] --pop must be >= 2\n"); return 1; }

    arena_global_init();   /* board_init()/search_init() + dummy g_tt — see arena.h */

    NnueNet *net = nnue_net_load(cfg.net_path);
    if (!net || !nnue_net_ready(net)) {
        fprintf(stderr, "[ga_tune] FATAL: could not load NNUE weights '%s'\n", cfg.net_path);
        return 1;
    }
    g_net = net;
    /* See arena.h's arena_run() comment "ALSO REQUIRED, EASY TO MISS":
     * eval_stm() gates on the process-global nnue_ready() flag, which
     * only the legacy nnue_load()/nnue_load_from_mem() set. Flip the
     * gate once; every search still reads its own NnueAccum::net. */
    if (!g_nnue_net) g_nnue_net = net;

    OpeningPool openings; memset(&openings, 0, sizeof(openings));
    if (cfg.openings_path[0]) {
        if (load_openings(cfg.openings_path, &openings, cfg.opening_plies) != 0)
            fprintf(stderr, "[ga_tune] WARNING: could not load openings '%s' — startpos only\n", cfg.openings_path);
        else
            fprintf(stderr, "[ga_tune] loaded %d opening(s) from %s\n", openings.n, cfg.openings_path);
    } else {
        fprintf(stderr, "[ga_tune] WARNING: no --openings given, every game starts at the standard "
                        "position (CLAUDE.md rule 9: this narrows the tuning distribution badly — "
                        "fine for a smoke test, not for a real run)\n");
    }
    g_openings = &openings;

    /* Round --tt-mb down to a valid TTable size, same formula as
     * arena.c's tt_entries_for_mb() (~26 bytes/logical entry). */
    { const double bytes_per_entry = 26.0;
      size_t want = (size_t)((cfg.tt_mb * 1024.0 * 1024.0) / bytes_per_entry);
      size_t n = TT_BUCKETS; while (n * 2 <= want) n *= 2;
      if (n < (size_t)TT_BUCKETS * 512) n = (size_t)TT_BUCKETS * 512;
      g_tt_entries = n;
    }
    g_cfg = &cfg;

    g_pop = (Individual *)malloc((size_t)cfg.pop_size * sizeof(Individual));
    Rng rng; rng_seed(&rng, cfg.seed);

    int resumed = 0;
    if (cfg.resume) resumed = load_state(&cfg, &rng);
    if (!resumed) {
        g_generation = 0;
        genome_defaults(g_baseline);
        genome_defaults(g_best_ever); g_best_ever_fitness = -1.0;
        int e = cfg.elite_count; if (e > cfg.pop_size) e = cfg.pop_size;
        for (int i = 0; i < cfg.pop_size; i++) {
            genome_defaults(g_pop[i].genome);
            if (i >= e) mutate(&rng, g_pop[i].genome);   /* generation 0: elites = exact defaults */
            g_pop[i].fitness = -1; g_pop[i].w = g_pop[i].d = g_pop[i].l = 0;
        }
    } else {
        fprintf(stderr, "[ga_tune] resumed from '%s' at generation %d\n", cfg.state_path, g_generation);
    }

    fprintf(stderr, "[ga_tune] pop=%d generations=%d games/eval=%d movetime=%dms threads=%d "
                    "tt=%.1fMB(%zu entries) openings=%d\n",
            cfg.pop_size, cfg.generations, cfg.games_per_eval, cfg.movetime_ms, cfg.threads,
            cfg.tt_mb, g_tt_entries, openings.n);

    time_t t0 = time(NULL);
    for (; g_generation < cfg.generations; g_generation++) {
        evaluate_generation(cfg.threads);
        qsort(g_pop, (size_t)cfg.pop_size, sizeof(Individual), cmp_fitness_desc);

        if (g_pop[0].fitness > g_best_ever_fitness) {
            g_best_ever_fitness = g_pop[0].fitness;
            memcpy(g_best_ever, g_pop[0].genome, sizeof(Genome));
        }

        double mean_fitness = 0.0;
        for (int i = 0; i < cfg.pop_size; i++) mean_fitness += g_pop[i].fitness;
        mean_fitness /= cfg.pop_size;
        fprintf(stderr, "[ga_tune] gen %3d  best_fitness=%.4f (w=%lld d=%lld l=%lld)  "
                        "mean_fitness=%.4f  best_ever=%.4f  elapsed=%lds\n",
                g_generation, g_pop[0].fitness, g_pop[0].w, g_pop[0].d, g_pop[0].l,
                mean_fitness, g_best_ever_fitness, (long)(time(NULL) - t0));

        if (cfg.update_baseline && g_pop[0].fitness >= cfg.baseline_promote) {
            memcpy(g_baseline, g_pop[0].genome, sizeof(Genome));
            fprintf(stderr, "[ga_tune] gen %3d  baseline PROMOTED (score %.4f >= %.4f)\n",
                    g_generation, g_pop[0].fitness, cfg.baseline_promote);
        }

        save_state(&cfg);
        write_best_header(&cfg);

        if (g_generation + 1 < cfg.generations) evolve(&rng);
    }

    printf("\n[ga_tune] done. best_ever_fitness=%.4f\n", g_best_ever_fitness);
    printf("[ga_tune] best genome:\n");
    for (int i = 0; i < N_PARAMS; i++) printf("  %-26s = %g\n", PARAM[i].name, g_best_ever[i]);
    printf("[ga_tune] paste-back header written to %s\n", cfg.best_out_path);
    printf("[ga_tune] resumable state written to %s\n", cfg.state_path);

    free(g_pop);
    for (int i = 0; i < openings.n; i++) free(openings.fens[i]);
    free(openings.fens);
    nnue_net_destroy(net);
    return 0;
}
