/* tools/arena.c — Zchezz v4.00 native A/B arena (implementation plan
 * Appendix F.4, gate mechanism for Appendix F.5's bootstrap loop).
 *
 * ── BUILD ─────────────────────────────────────────────────────────────
 *   This file lives in engine/c/tools/ — SHARED across engine versions,
 *   tracking the CURRENT engine API only (see README.md, "engine/c/tools/").
 *   From engine/build/:  mingw32-make ENGINE=v400 arena
 *   (output: engine/build/arena.exe — a shared tool binary, not a
 *   per-version artifact. See engine/build/Makefile for ENGINE=... to
 *   target a different engine/c/zchezz_vXXX/ folder.)
 *   (mirrors selfplay's flags; links board.c/search.c/nnue.c but NOT
 *   main.c/syzygy.c/tbprobe.c/book.c — see selfplay.c's header comment
 *   for why -DNO_TABLEBASES -DNO_BOOK is safe here), from engine/build/:
 *
 *     gcc -O3 -ffast-math -D_GNU_SOURCE -std=c11 -mavxvnni -mavx2 \
 *         -I../c/zchezz_v400 -DNO_TABLEBASES -DNO_BOOK \
 *         -Wno-unused-variable -Wno-unused-but-set-variable \
 *         -Wno-maybe-uninitialized -Wno-misleading-indentation \
 *         -Wno-sign-compare -Wno-unused-function -Wno-parentheses \
 *         -o arena.exe ../c/tools/arena.c ../c/zchezz_v400/board.c \
 *         ../c/zchezz_v400/search.c ../c/zchezz_v400/nnue.c \
 *         -static -lm -pthread
 *
 * ── WHY THIS EXISTS (Appendix F.4 / F.5) ────────────────────────────
 *
 *   Appendix F.5 closes the AlphaZero-style bootstrap loop: generate
 *   self-play data with gen_i -> train gen_{i+1} -> GATE gen_{i+1} vs
 *   gen_i via SPRT -> promote or discard -> repeat.  Step 3 (the gate)
 *   runs once per generation, exactly as often as data generation, so
 *   it has to be nearly as fast as generation itself.  Driving the gate
 *   through tests/run_tournament_quick.py (subprocess + UCI text
 *   protocol, one process pair per game) would make the gate the
 *   bottleneck of the whole loop.  This tool plays the gate's games
 *   in-process, the same way tools/selfplay.c plays self-play games
 *   in-process — see that file's header for the full "why a subprocess
 *   is wasteful here" argument, which applies identically.
 *
 *   Appendix F.4's own text scopes this tool narrowly to comparing two
 *   NNUE weight files and/or two sets of search constants *within the
 *   same code* (the in-process case) — real code-version comparisons
 *   (v3.14 vs v4.00, a git tag vs HEAD, vs Stockfish) were explicitly
 *   left to run_tournament.py's real-subprocess UCI path.  This
 *   implementation widens that scope on purpose: below, a "player" is
 *   one of TWO kinds, and a single match can mix them freely.
 *
 * ── TWO PLAYER KINDS (the central design decision of this file) ──────
 *
 *   1. `net:<path.nnu4>`  — IN-PROCESS player.  Same engine CODE
 *      (this binary's own board.c/search.c/nnue.c), a DIFFERENT set of
 *      NNUE weights loaded into its own NnueNet (nnue_net_load, v4.00
 *      Phase 0-follow-up — see nnue.h).  This is the fast path and the
 *      whole reason this tool exists: it is the SPRT gate of Appendix
 *      F.5 (gen_{i+1} vs gen_i), invoked once per bootstrap generation,
 *      and it has to run at close to self-play throughput.  No process
 *      spawn, no text protocol, no per-move syscall round trip — just
 *      search_best() called directly with a SearchParams pointed at
 *      this player's own TTable/SearchState/NnueAccum.
 *
 *   2. `uci:<path-to-exe>` — EXTERNAL UCI PROCESS player.  A real,
 *      independently-compiled engine executable, driven over stdin/
 *      stdout with the standard UCI text protocol (uci/isready/
 *      position/go/bestmove).  Slower — one process per concurrent game
 *      slot, one text round trip per move — but it is the ONLY honest
 *      way to compare different CODE, not just different weights: v3.14
 *      vs v4.00, a git tag vs a branch, or a third-party engine such as
 *      Stockfish.  tests/run_arena.py builds these from git refs (see
 *      that file) and hands arena.exe the resulting .exe paths.
 *
 *   A match may freely mix kinds — e.g. `net:gen6.nnu4` (candidate,
 *   in-process) vs `uci:zchezz_v314.exe` (external, different code) in
 *   the same run.  Internally both kinds implement the same tiny
 *   interface used by play_one_game() below: "give me your move for
 *   this position under this time/node budget" — see PlayerInstance
 *   and play_move_net()/play_move_uci().
 *
 * ── TT ISOLATION IS MANDATORY — THE OPPOSITE OF SELFPLAY ─────────────
 *
 *   *** READ THIS BEFORE "optimizing" this file to share a TTable. ***
 *
 *   tools/selfplay.c shares ONE TTable between both colors ON PURPOSE
 *   (default `!--separate-tt`): within a self-play game both colors are
 *   the SAME engine with the SAME weights, so sharing loses no
 *   correctness and buys real throughput (half the TT memory per
 *   worker, transpositions reused across colors).
 *
 *   An arena match is the OPPOSITE situation: the two players are
 *   ADVERSARIES under test — frequently different NNUE weights,
 *   sometimes entirely different engine code.  If they shared a TT,
 *   player A's search would silently read back scores, best moves and
 *   even mate distances computed by player B's evaluation function
 *   (or, in the uci: case, this wouldn't even be reachable, but for a
 *   net: vs net: match it absolutely would be).  That is not a
 *   performance shortcut, it is information leakage from one side of
 *   the A/B test into the other, and it would silently invalidate
 *   every result this tool produces — the SPRT gate could promote a
 *   WORSE network because it got to peek at the better one's search.
 *
 *   Therefore: every PlayerInstance (see below) that is `net:` kind
 *   gets its OWN TTable, created once per (worker thread x player)
 *   and physically tt_clear()'d before every single game — same
 *   physical-wipe discipline as selfplay.c's TT POLICY section, and
 *   for the same path-dependent-repetition-score reason documented
 *   there (not repeated in full here; see tools/selfplay.c).  `uci:`
 *   kind players are separate OS processes and therefore trivially
 *   TT-isolated already — that isolation is not something this file
 *   has to engineer, it falls out of using a real subprocess.
 *
 * ── CONCURRENCY MODEL ──────────────────────────────────────────────
 *
 *   `--threads N` = N concurrent GAMES (one worker OS thread per game
 *   in flight), NOT one thread per color — chess is strictly
 *   alternating, a single worker thread plays both colors of one game
 *   sequentially, exactly like selfplay.c's worker_fn.  Work is pulled
 *   from a shared atomic task counter (dynamic, same pattern as
 *   selfplay.c) over a precomputed task list (see build_task_list()).
 *
 *   Each worker thread lazily builds its OWN PlayerInstance for every
 *   PlayerDef it ends up needing (cached in a per-worker array indexed
 *   by player id, created on first use, reused for the rest of that
 *   worker's games).  Net effect: with T threads and a net: vs net:
 *   match, you get T independent NnueAccum/TTable/SearchState pairs
 *   PER PLAYER (2*T total) — deliberately, so no two concurrent games
 *   ever share a TTable, whether or not they involve the "same" player.
 *   With a uci: player, this also means T independent OS processes of
 *   that exe are spawned (one per concurrent game slot) — the process
 *   itself is the isolation boundary there, so nothing else changes.
 *
 * ── ANTITHETIC (COLOR-SWAPPED) PAIRING ────────────────────────────────
 *
 *   Per the task's zquoridor recon (tools/arena/arena.cpp there, and
 *   its GA tuner's `antithetic()`): every opening is played TWICE, once
 *   with each player as White, back to back, before moving to the next
 *   opening.  This cancels first-move/opening-choice advantage far more
 *   effectively than N independent random-color games at the same N —
 *   it is the standard variance-reduction trick used by cutechess-cli,
 *   Fishtest, and (per the recon doc) zquoridor's own arena and SPSA
 *   harness.  See build_task_list() below.
 *
 * ── OPENINGS ──────────────────────────────────────────────────────────
 *
 *   `--openings FILE`: .epd/.fen files are read one FEN per line
 *   (first 4 whitespace-separated tokens = board/stm/castling/ep,
 *   exactly the convention tests/run_tournament.py's OpeningIndex
 *   already uses — halfmove/fullmove default to "0 1" if omitted).
 *   .pgn files are replayed ply-by-ply with a minimal SAN resolver
 *   (see pgn_load_openings()) for the first `--opening-plies` plies of
 *   each game, and the resulting FEN is recorded — this tool never
 *   plays PGN move text directly, it only uses PGN files to DERIVE a
 *   list of opening FENs once at startup, on a scratch Board, before
 *   any worker starts.  Without `--openings`, every game starts from
 *   the standard position and this tool says so loudly: with only one
 *   opening, only the antithetic pairing still helps — variance across
 *   pairs will be much higher than with a real opening book.
 *
 * ── SPRT (Appendix F.5's promote/discard gate) ────────────────────────
 *
 *   `--sprt --elo0 E0 --elo1 E1 [--alpha A] [--beta B]` computes the
 *   standard GSPRT log-likelihood ratio (Wald's SPRT under a normal
 *   approximation of the mean score, trinomial variance) — the same
 *   method cutechess-cli/Fishtest use, and the same trinomial variance
 *   formula as tests/elo_calc.py's elo_difference() (`w*(1-p)^2 +
 *   d*(0.5-p)^2 + l*(0-p)^2`), so the numbers this tool reports are
 *   directly comparable to elo_calc.py's own output. See sprt_llr()
 *   below for the exact formula and citations. SPRT only applies to a
 *   straight 2-player match (the Appendix F.5 gate is always exactly
 *   candidate-vs-baseline); with more players this tool still plays
 *   every requested game and reports pairwise Elo, it just does not
 *   attempt a sequential test that was never well-defined for >2 arms.
 *
 * ── NO STOCKFISH-ANCHORED ABSOLUTE ELO HERE — ON PURPOSE ──────────────
 *
 *   An earlier draft of this file added a `uci:<path>@<elo>` anchor
 *   marker and absolute-Elo math (Stockfish at a calibrated UCI_Elo,
 *   same as tests/run_tournament.py's ANCHORS mechanism). That was
 *   deliberately removed: this native arena's whole reason to exist
 *   is comparing two VERSIONS of OUR OWN engine quickly (Appendix F.4/
 *   F.5's SPRT gate) — benchmarking against an external reference at
 *   a calibrated strength is a different job with a different cadence
 *   (run occasionally, not once per bootstrap generation), and
 *   tests/run_tournament.py already does it well. Duplicating that
 *   machinery here would be two places that can drift out of sync for
 *   no throughput benefit (an anchor match is inherently uci:-only,
 *   i.e. always paying subprocess/text-protocol overhead — the native
 *   fast path buys nothing there). `uci:` players are still fully
 *   supported (see "TWO PLAYER KINDS" above) — comparing two engine
 *   VERSIONS via uci: (v3.14 vs v4.00, a git tag vs a branch) is
 *   exactly the intended use; only the known-strength-anchor/absolute-
 *   Elo bookkeeping was cut. Use tests/run_tournament.py for that.
 *
 * ── MULTI-ENGINE MINI-TOURNAMENT / RANKING ────────────────────────────
 *
 *   With >2 `--player` entries this tool plays a full round-robin (or,
 *   with `--gauntlet <player>`, one designated player against every
 *   other player only) and reports, in addition to the per-pairing
 *   W/D/L already described above, two more things per player:
 *     1. A full cross-table (see print_cross_table()).
 *     2. A POOLED "field performance" Elo: that player's total W/D/L
 *        across EVERY pairing it took part in, fed through the same
 *        trinomial elo_diff_ci() as a single aggregate match "against
 *        the field". This is a first-order performance-rating
 *        approximation (same idea as a tournament performance rating),
 *        NOT a joint whole-history/Bayeselo-style MLE across all N
 *        players simultaneously — a true joint MLE would need to solve
 *        for all players' ratings at once so that transitive
 *        information (A beat B, B beat C => something about A vs C)
 *        is used even for pairs that never played each other. That is
 *        a real algorithm (Bayeselo/WHR) this tool does not implement;
 *        pooled-vs-field is a reasonable, honestly-labelled substitute
 *        for a fast native tool, and is reported as such (both in the
 *        text report and the `"rating_method"` JSON field).
 *
 * ── EMBEDDABLE AS A LIBRARY (for a future native GA/SPSA tuner) ──────
 *
 *   arena_global_init() (board_init/search_init, once per process) and
 *   arena_run() (play exactly the match described by one Config,
 *   return one ArenaReport) are ordinary functions, not buried inside
 *   main() — a future tuner (Appendix F.4's SPSA/GA harness) can link
 *   this file compiled with -DARENA_NO_MAIN (see the bottom of this
 *   file) and drive it in a loop: mutate a Config's movetime/players/
 *   search constants, call arena_run(), read back w/d/l + elo + sprt
 *   verdict from the returned ArenaReport, repeat — no subprocess,
 *   no argv, no stdout parsing. See tools/arena.h for the public
 *   surface this exposes.
 *
 * ── WHAT WAS DUPLICATED VS FACTORED OUT ───────────────────────────────
 *
 *   selfplay.c's header comment already flags this file as the reason
 *   play_one_game() was written in a reusable shape — but selfplay's
 *   actual play_one_game() bakes in temperature-sampling move choice
 *   and a single shared-or-per-color TT policy that does not fit an
 *   adversarial A/B match (this file's move choice is always "best
 *   move found", and its TT policy is unconditionally per-player, see
 *   above).  Rather than force both tools through one parameterized
 *   function and risk a knob leaking the wrong default into either
 *   one, this file duplicates the small pieces that are truly
 *   identical (zmalloc32/zfree32 aligned alloc, detect_cpus(), the
 *   board_load_fen() global-rebind fix — see worker_reset_board() in
 *   both files, same bug, same one-line fix) and writes its own
 *   play_one_game() shaped for two distinct adversaries.  This mirrors
 *   selfplay.c's own stated policy (it duplicates zmalloc32 from
 *   main.c/nnue.c rather than factor a shared header on a v4.00 branch
 *   that hasn't shipped) — same tradeoff, same rationale.
 *
 *   UPDATE (opening-book capability audit): san_resolve() and
 *   move_to_san() WERE duplicated 1:1 between this file and
 *   selfplay.c's new opening-book support (selfplay.c needs the exact
 *   same SAN<->Move machinery for PGN opening replay and its own --pgn
 *   output). A byte-identical duplicate of working, already-audited
 *   SAN parsing code is exactly the kind of "second parser that can
 *   drift and get its own bugs" this project avoids — so those two
 *   functions were factored out into tools/opening_pool.c/.h (this
 *   file now #includes opening_pool.h and calls them unchanged). The
 *   rest of this file's "duplicated on purpose" reasoning above still
 *   applies to everything else listed (zmalloc32/detect_cpus/the
 *   board_load_fen rebind fix, and play_one_game() itself).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "sample.h"
#include <time.h>
#include <stdint.h>
#include <ctype.h>
#include <pthread.h>
#include <stdatomic.h>
#include <sys/stat.h>

#include "board.h"
#include "search.h"
#include "nnue.h"
#include "opening_pool.h"   /* san_resolve()/move_to_san() — shared with selfplay.c, see that header */

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#include <fcntl.h>
#endif

/* ═══════════════════════════════════════════════════════════════════
 * ===== CONFIGURATION =====
 *
 * Project-wide convention (see tests/run_tournament.py and friends):
 * EVERY option settable on the command line must ALSO have a named,
 * documented default HERE, and the CLI parser (config_defaults(),
 * further down) must read its defaults FROM these constants — never
 * duplicate a literal. If you change a default, change it here; the
 * usage text (print_usage()) also pulls these in via the XSTR() macro
 * below so the --help output and the actual default can never drift
 * apart either.
 * ═══════════════════════════════════════════════════════════════════ */
#define ARENA_DEFAULT_GAMES            20        /* games per pairing (>2 players) or total (2 players) */
#define ARENA_DEFAULT_THREADS          0         /* concurrent games; 0 = autodetect logical cores */
#define ARENA_DEFAULT_MOVETIME_MS      100       /* per-move time budget, ms; 0 = use ARENA_DEFAULT_NODES instead */
#define ARENA_DEFAULT_NODES            0         /* per-move node budget, used only when movetime==0 */
#define ARENA_DEFAULT_MAX_PLIES        300       /* game-length safety cap -> counted as a draw */
#define ARENA_DEFAULT_SEED             1         /* opening-cycling RNG seed */
#define ARENA_DEFAULT_TT_MB            8.0       /* per-PlayerInstance TTable memory budget, MB */
#define ARENA_DEFAULT_OPENING_PLIES    8         /* PGN opening file: plies replayed per game before recording its FEN */
#define ARENA_DEFAULT_ELO0             0.0       /* SPRT null hypothesis Elo */
#define ARENA_DEFAULT_ELO1             5.0       /* SPRT alt hypothesis Elo */
#define ARENA_DEFAULT_ALPHA            0.05      /* SPRT false-accept-H1 rate */
#define ARENA_DEFAULT_BETA             0.05      /* SPRT false-accept-H0 rate */
#define ARENA_DEFAULT_UCI_TIMEOUT_MS   15000     /* max wait for a uci: player's "bestmove" line */
#define ARENA_DEFAULT_SPRT_ENABLED     0         /* 1 = --sprt is on by default (it is not) */
#define ARENA_DEFAULT_OPENINGS_PATH    ""        /* "" = no opening file -> every game starts at startpos */
#define ARENA_DEFAULT_JSON_PATH        ""        /* "" = no --json output */
#define ARENA_DEFAULT_PGN_PATH         ""        /* "" = no --pgn output */
#define ARENA_DEFAULT_BIN_PATH         ""        /* "" = no --bin training-sample output (net: players only,
                                                    * AND every net: player must share one weight file — see
                                                    * the --bin gate in arena_run() for why: a mixed-net match
                                                    * would blend two different evaluators' opinions into one
                                                    * unlabeled eval_cp field, which sample.h has no room to
                                                    * disambiguate) */
#define ARENA_DEFAULT_EPD_PATH         ""        /* "" = no --epd training-position output. Unlike --bin, EPD's
                                                    * c2 opcode carries a free-text evaluator label per position,
                                                    * so --epd has NO same-net restriction: a mixed net:/uci:
                                                    * or net-A vs net-B match is fine, each row's c2 records the
                                                    * ACTUAL mover's player name (see write_epd_rows()'s header) */
#define ARENA_DEFAULT_GAUNTLET_SPEC    ""        /* "" = round-robin (no gauntlet candidate set) */

/* ── PLAYERS config block ──────────────────────────────────────────────
 * --player is repeatable, so it cannot sit next to a scalar ARENA_DEFAULT_*
 * constant the way every other flag above does — this is its config-block
 * equivalent (CLAUDE.md rule 8: every CLI-reachable input, including
 * repeated/list-valued ones, must be fully controllable by editing this
 * file). Populate this array to define a match by EDITING THE FILE and
 * running arena.exe with no --player flags at all; passing --player on the
 * command line still works exactly as before and OVERRIDES this list in
 * its entirety (the first --player on the CLI clears these defaults — see
 * parse_args()), so nothing here changes behavior for existing invocations
 * that already pass --player. `name` "" auto-generates "P<idx>:<basename>",
 * identical to a CLI-supplied spec (see parse_player_spec()). */
typedef struct {
    const char *kind;   /* "net" or "uci" */
    const char *path;
    const char *name;   /* "" = auto-generate from path, like the CLI does */
} ArenaDefaultPlayer;
static const ArenaDefaultPlayer ARENA_DEFAULT_PLAYERS[] = {
    { "net", "../c/zchezz_v400/nnue_weights.bin", "" },
    { "net", "../c/zchezz_v400/nnue_weights.bin", "" },
};
#define ARENA_DEFAULT_PLAYERS_COUNT ((int)(sizeof(ARENA_DEFAULT_PLAYERS) / sizeof(ARENA_DEFAULT_PLAYERS[0])))

#define ARENA_STR(x)  #x
#define ARENA_XSTR(x) ARENA_STR(x)   /* expands x before stringizing, so macros show their VALUE not their name */

/* ═══════════════════════════════════════════════════════════════════
 * Small helpers duplicated from selfplay.c — see header comment "WHAT
 * WAS DUPLICATED VS FACTORED OUT".
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

static char *xstrdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = (char *)malloc(n);
    if (p) memcpy(p, s, n);
    return p;
}

/* ═══════════════════════════════════════════════════════════════════
 * RNG — xorshift64*, one per worker (used only for --openings cycling
 * order if the caller wants a shuffled task list; deterministic and
 * reseeded from --seed so a run is reproducible).
 * ═══════════════════════════════════════════════════════════════════ */
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

/* ═══════════════════════════════════════════════════════════════════
 * Player definition. See tools/arena.h for the byte-identical public
 * copy of this type.
 * ═══════════════════════════════════════════════════════════════════ */
typedef enum { PLAYER_NET, PLAYER_UCI } PlayerKind;

#define MAX_PLAYERS 16

typedef struct {
    PlayerKind kind;
    char       path[512];
    char       name[128];       /* display name (basename, deduplicated) */
    NnueNet   *net;              /* PLAYER_NET only, loaded once, shared read-only */
} PlayerDef;

/* ═══════════════════════════════════════════════════════════════════
 * UCI subprocess plumbing — spawn an engine .exe, talk stdin/stdout,
 * line-buffered, with a background reader thread feeding a small
 * mutex-protected queue so the game loop can poll with a timeout
 * (mirrors the nanosleep-poll idiom selfplay.c's progress reporter
 * already uses in this codebase, instead of relying on
 * pthread_cond_timedwait's absolute-time semantics, which vary more
 * across the MinGW/winpthreads vs glibc pthread implementations this
 * project builds against on Windows vs Linux).
 * ═══════════════════════════════════════════════════════════════════ */
#define UCI_LINE_MAX   4096
#define UCI_QUEUE_CAP  256

typedef struct {
    char  *lines[UCI_QUEUE_CAP];
    int    head, tail, len;
    pthread_mutex_t mtx;
    volatile int alive;      /* 0 once the process/pipe is known dead */
#if defined(_WIN32)
    HANDLE hProcess;
    HANDLE hIn_write;   /* our end: write engine's stdin  */
    HANDLE hOut_read;   /* our end: read engine's stdout  */
#else
    pid_t  pid;
    int    fd_in_write;
    int    fd_out_read;
#endif
    pthread_t reader;
} UciProc;

/* Serializes the whole "create pipes -> mark our own ends
 * non-inheritable -> CreateProcess/fork -> close the child-side pipe
 * ends" sequence in uci_proc_start() across worker threads.
 *
 * Why this is required, not just nice-to-have: CreateProcess(...,
 * bInheritHandles=TRUE, ...) inherits EVERY currently-inheritable
 * handle open in this process at the moment it runs -- not just the
 * ones named in STARTUPINFO. CreatePipe() hands back BOTH ends of a
 * pipe as inheritable; uci_proc_start() only strips the inherit flag
 * from ITS OWN ends (p->hIn_write / p->hOut_read) via
 * SetHandleInformation, and only closes the child-side ends
 * (child_stdin_read / child_stdout_write) after CreateProcess returns.
 * Between "CreatePipe" and "SetHandleInformation" on one thread, those
 * handles are wide open and inheritable process-wide.
 *
 * With --threads > 1, multiple worker threads call uci_proc_start()
 * concurrently (arena spawns up to 2 * threads engine subprocesses at
 * once). Thread A's CreateProcess() can therefore run in the gap
 * before thread B has stripped the inherit flag on its own pipe
 * handles, silently duplicating B's pipe handles into A's freshly
 * spawned child. That child now (unknowingly) holds an extra open
 * handle to a pipe it has nothing to do with. Windows anonymous pipes
 * only signal EOF to a reader once EVERY write handle is closed, so a
 * leaked handle keeps some OTHER engine's stdout pipe (or the arena's
 * own stdin-write pipe) artificially "alive" even after the intended
 * process dies -- the reader thread's ReadFile() (or a WriteFile() on
 * a full stdin pipe the real reader stopped draining) then blocks with
 * NO timeout, hanging that worker thread forever. This reproduced
 * exactly: --threads 1 never hung, --threads 4/6 hung within the first
 * few games, always at process-spawn time (see arena.c git history /
 * CLAUDE.md-adjacent investigation notes for the instrumented repro).
 *
 * The standard, Microsoft-documented fix for multi-threaded spawners
 * that use inheritable handles is exactly this: hold one process-wide
 * lock across the whole create-handles -> spawn -> close-child-ends
 * sequence so no two spawns can interleave. Spawning is rare (once per
 * PlayerInstance per worker, plus the occasional respawn) so the
 * serialization costs nothing on the hot path (move exchange is
 * unaffected -- only uci_proc_start() takes this lock). */
static pthread_mutex_t g_spawn_mtx = PTHREAD_MUTEX_INITIALIZER;

static void uciq_push(UciProc *p, const char *line) {
    pthread_mutex_lock(&p->mtx);
    if (p->len < UCI_QUEUE_CAP) {
        p->lines[p->tail] = xstrdup(line);
        p->tail = (p->tail + 1) % UCI_QUEUE_CAP;
        p->len++;
    }
    /* else: queue full (should never happen at UCI info-line volumes
     * for one engine talking to one client) — drop rather than block
     * the reader thread forever. */
    pthread_mutex_unlock(&p->mtx);
}

#if defined(_WIN32)
static void *uci_reader_thread(void *arg) {
    UciProc *p = (UciProc *)arg;
    char buf[UCI_LINE_MAX];
    size_t buflen = 0;
    for (;;) {
        char chunk[1024];
        DWORD got = 0;
        BOOL ok = ReadFile(p->hOut_read, chunk, sizeof(chunk), &got, NULL);
        if (!ok || got == 0) break;
        for (DWORD i = 0; i < got; i++) {
            char c = chunk[i];
            if (c == '\n') {
                if (buflen > 0 && buf[buflen - 1] == '\r') buflen--;
                buf[buflen] = 0;
                uciq_push(p, buf);
                buflen = 0;
            } else if (buflen + 1 < sizeof(buf)) {
                buf[buflen++] = c;
            }
        }
    }
    pthread_mutex_lock(&p->mtx);
    p->alive = 0;
    pthread_mutex_unlock(&p->mtx);
    return NULL;
}
#else
static void *uci_reader_thread(void *arg) {
    UciProc *p = (UciProc *)arg;
    FILE *f = fdopen(p->fd_out_read, "r");
    char buf[UCI_LINE_MAX];
    if (f) {
        while (fgets(buf, sizeof(buf), f)) {
            size_t n = strlen(buf);
            while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r')) buf[--n] = 0;
            uciq_push(p, buf);
        }
        fclose(f);
    }
    pthread_mutex_lock(&p->mtx);
    p->alive = 0;
    pthread_mutex_unlock(&p->mtx);
    return NULL;
}
#endif

/* Directory part of `path`, written into `out` (size cap). Mirrors
 * tests/run_tournament.py's `cwd=engine_dir` — spawning a uci: player
 * with its OWN directory as the working directory matters for our own
 * zchezz.exe (relative "nnue_weights.bin" / "tablebases/" lookups) and
 * is harmless for Stockfish (which needs no relative resources). */
static void dirname_of(const char *path, char *out, size_t cap) {
    const char *s1 = strrchr(path, '/');
    const char *s2 = strrchr(path, '\\');
    const char *s = s1 > s2 ? s1 : s2;
    if (!s) { strncpy(out, ".", cap - 1); out[cap-1] = 0; return; }
    size_t n = (size_t)(s - path);
    if (n >= cap) n = cap - 1;
    memcpy(out, path, n);
    out[n] = 0;
    if (n == 0) { out[0] = '.'; out[1] = 0; }
}

/* Returns a freshly spawned UciProc, or NULL on failure (bad path,
 * spawn failure). Caller owns the returned pointer (uci_proc_stop()
 * frees it). Spawned with its working directory set to exe_path's own
 * folder — see dirname_of() comment. */
static UciProc *uci_proc_start(const char *exe_path) {
    UciProc *p = (UciProc *)calloc(1, sizeof(UciProc));
    if (!p) return NULL;
    pthread_mutex_init(&p->mtx, NULL);
    p->alive = 1;

    /* See g_spawn_mtx's header comment: the whole create-handles ->
     * spawn -> close-child-ends sequence below must run with no other
     * thread's uci_proc_start() interleaved, or CreateProcess()/fork()
     * can silently leak an in-flight pipe handle from another thread
     * into this child (or vice versa), producing a pipe that never
     * sees a real EOF and a worker thread that hangs forever with no
     * timeout anywhere in the picture. */
    pthread_mutex_lock(&g_spawn_mtx);

#if defined(_WIN32)
    SECURITY_ATTRIBUTES sa; memset(&sa, 0, sizeof(sa));
    sa.nLength = sizeof(sa); sa.bInheritHandle = TRUE;

    HANDLE child_stdin_read = NULL, child_stdout_write = NULL;
    if (!CreatePipe(&child_stdin_read, &p->hIn_write, &sa, 0)) {
        pthread_mutex_unlock(&g_spawn_mtx); free(p); return NULL;
    }
    if (!CreatePipe(&p->hOut_read, &child_stdout_write, &sa, 0)) {
        CloseHandle(child_stdin_read); CloseHandle(p->hIn_write);
        pthread_mutex_unlock(&g_spawn_mtx); free(p); return NULL;
    }
    /* Our own ends of each pipe must NOT be inherited by the child. */
    SetHandleInformation(p->hIn_write, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(p->hOut_read, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOA si; memset(&si, 0, sizeof(si));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput  = child_stdin_read;
    si.hStdOutput = child_stdout_write;
    si.hStdError  = child_stdout_write;   /* fold stderr into the same
                                            * stream — a UCI engine that
                                            * prints diagnostics to
                                            * stderr would otherwise
                                            * silently vanish; the reader
                                            * simply ignores any line
                                            * that isn't a token we're
                                            * waiting for. */
    PROCESS_INFORMATION pi; memset(&pi, 0, sizeof(pi));

    char cmdline[1024];
    snprintf(cmdline, sizeof(cmdline), "\"%s\"", exe_path);
    char cwd[512]; dirname_of(exe_path, cwd, sizeof(cwd));
    BOOL ok = CreateProcessA(NULL, cmdline, NULL, NULL, TRUE,
                              CREATE_NO_WINDOW, NULL, cwd, &si, &pi);
    CloseHandle(child_stdin_read);
    CloseHandle(child_stdout_write);
    if (!ok) {
        CloseHandle(p->hIn_write); CloseHandle(p->hOut_read);
        pthread_mutex_unlock(&g_spawn_mtx); free(p); return NULL;
    }
    p->hProcess = pi.hProcess;
    CloseHandle(pi.hThread);
    pthread_mutex_unlock(&g_spawn_mtx);
#else
    int in_pipe[2], out_pipe[2];
    if (pipe(in_pipe) != 0) { pthread_mutex_unlock(&g_spawn_mtx); free(p); return NULL; }
    if (pipe(out_pipe) != 0) {
        close(in_pipe[0]); close(in_pipe[1]);
        pthread_mutex_unlock(&g_spawn_mtx); free(p); return NULL;
    }
    pid_t pid = fork();
    if (pid < 0) {
        close(in_pipe[0]); close(in_pipe[1]);
        close(out_pipe[0]); close(out_pipe[1]);
        pthread_mutex_unlock(&g_spawn_mtx); free(p); return NULL;
    }
    if (pid == 0) {
        dup2(in_pipe[0], STDIN_FILENO);
        dup2(out_pipe[1], STDOUT_FILENO);
        dup2(out_pipe[1], STDERR_FILENO);
        close(in_pipe[0]); close(in_pipe[1]);
        close(out_pipe[0]); close(out_pipe[1]);
        char cwd[512]; dirname_of(exe_path, cwd, sizeof(cwd));
        if (chdir(cwd) != 0) { /* fall back to launching from current cwd */ }
        execlp(exe_path, exe_path, (char *)NULL);
        _exit(127);   /* execlp failed */
    }
    /* Parent: still holding g_spawn_mtx until the child-side fds below
     * are closed — see header comment. */
    close(in_pipe[0]);
    close(out_pipe[1]);
    p->pid = pid;
    p->fd_in_write = in_pipe[1];
    p->fd_out_read = out_pipe[0];
    pthread_mutex_unlock(&g_spawn_mtx);
#endif
    pthread_create(&p->reader, NULL, uci_reader_thread, p);
    return p;
}

static void uci_proc_send(UciProc *p, const char *line) {
    if (!p || !p->alive) return;
    char buf[UCI_LINE_MAX];
    int n = snprintf(buf, sizeof(buf), "%s\n", line);
    if (n <= 0) return;
#if defined(_WIN32)
    DWORD written = 0;
    WriteFile(p->hIn_write, buf, (DWORD)n, &written, NULL);
#else
    ssize_t r = write(p->fd_in_write, buf, (size_t)n);
    (void)r;
#endif
}

/* Poll for the next queued line, waiting up to timeout_ms.
 * Returns 1 (line copied into out, up to cap), 0 (timeout), or
 * -1 (process/pipe closed, no more lines will ever arrive). */
static int uci_proc_readline(UciProc *p, char *out, size_t cap, int timeout_ms) {
    int waited = 0;
    for (;;) {
        pthread_mutex_lock(&p->mtx);
        if (p->len > 0) {
            char *line = p->lines[p->head];
            p->head = (p->head + 1) % UCI_QUEUE_CAP;
            p->len--;
            pthread_mutex_unlock(&p->mtx);
            strncpy(out, line, cap - 1);
            out[cap - 1] = 0;
            free(line);
            return 1;
        }
        int alive = p->alive;
        pthread_mutex_unlock(&p->mtx);
        if (!alive) return -1;
        if (waited >= timeout_ms) return 0;
        struct timespec ts = {0, 5 * 1000 * 1000};   /* 5ms poll */
        nanosleep(&ts, NULL);
        waited += 5;
    }
}

/* Read lines, discarding anything that doesn't start with `prefix`,
 * until one does (returned in out, WITHOUT the prefix stripped) or the
 * timeout/EOF is hit. Returns 1/0/-1 like uci_proc_readline(). */
static int uci_proc_wait_for(UciProc *p, const char *prefix, char *out, size_t cap, int timeout_ms) {
    long long start = (long long)time(NULL) * 1000;
    for (;;) {
        long long now = (long long)time(NULL) * 1000;
        int remaining = timeout_ms - (int)(now - start);
        if (remaining < 0) remaining = 0;
        int rc = uci_proc_readline(p, out, cap, remaining > 200 ? 200 : (remaining > 0 ? remaining : 1));
        if (rc == 1) {
            if (!strncmp(out, prefix, strlen(prefix))) return 1;
            continue;
        }
        if (rc == -1) return -1;
        if ((int)(((long long)time(NULL) * 1000) - start) >= timeout_ms) return 0;
    }
}

static void uci_proc_stop(UciProc *p) {
    if (!p) return;
    uci_proc_send(p, "quit");
#if defined(_WIN32)
    WaitForSingleObject(p->hProcess, 2000);
    TerminateProcess(p->hProcess, 0);   /* no-op if it already exited */
    CloseHandle(p->hIn_write);
    CloseHandle(p->hOut_read);
    CloseHandle(p->hProcess);
#else
    int status; (void)status;
    struct timespec ts = {2, 0}; nanosleep(&ts, NULL);
    kill(p->pid, SIGKILL);
    waitpid(p->pid, &status, 0);
    close(p->fd_in_write);
    close(p->fd_out_read);
#endif
    pthread_join(p->reader, NULL);
    pthread_mutex_lock(&p->mtx);
    for (int i = 0; i < p->len; i++) free(p->lines[(p->head + i) % UCI_QUEUE_CAP]);
    pthread_mutex_unlock(&p->mtx);
    pthread_mutex_destroy(&p->mtx);
    free(p);
}

/* Full UCI handshake: uci/uciok, isready/readyok. No `setoption`
 * traffic (no anchor/known-strength mechanism here — see header "NO
 * STOCKFISH-ANCHORED ABSOLUTE ELO HERE"; a uci: player is driven with
 * whatever strength it starts up at). Returns 1 on success. */
static int uci_proc_handshake(UciProc *p, int timeout_ms) {
    char line[UCI_LINE_MAX];
    uci_proc_send(p, "uci");
    if (uci_proc_wait_for(p, "uciok", line, sizeof(line), timeout_ms) != 1) return 0;
    uci_proc_send(p, "isready");
    if (uci_proc_wait_for(p, "readyok", line, sizeof(line), timeout_ms) != 1) return 0;
    return 1;
}

/* ═══════════════════════════════════════════════════════════════════
 * Player definitions and per-worker instances
 * ═══════════════════════════════════════════════════════════════════ */
/* One player's live resources for ONE worker thread. See header comment
 * "TT ISOLATION IS MANDATORY" — tt/nnue/ss are never shared with any
 * other PlayerInstance, whether that's the opponent in the SAME game or
 * the SAME player's instance on a DIFFERENT worker thread. */
typedef struct {
    const PlayerDef *def;
    /* PLAYER_NET */
    NnueAccum   *nnue;
    SearchState *ss;
    TTable      *tt;
    UndoFrame   *undo;
    int          undo_top;
    /* PLAYER_UCI */
    UciProc     *proc;
    int          uci_dead;      /* set after one failed respawn attempt */
} PlayerInstance;

static const char *basename_of(const char *path) {
    const char *s1 = strrchr(path, '/');
    const char *s2 = strrchr(path, '\\');
    const char *s = s1 > s2 ? s1 : s2;
    return s ? s + 1 : path;
}

static PlayerInstance *player_instance_create(const PlayerDef *def, size_t tt_entries) {
    PlayerInstance *pi = (PlayerInstance *)calloc(1, sizeof(PlayerInstance));
    pi->def = def;
    if (def->kind == PLAYER_NET) {
        pi->undo = (UndoFrame *)malloc(STACK_SIZE * sizeof(UndoFrame));
        pi->nnue = (NnueAccum *)zmalloc32(sizeof(NnueAccum));
        memset(pi->nnue, 0, sizeof(NnueAccum));
        pi->nnue->net = def->net;
        pi->ss  = search_state_new();
        pi->tt  = tt_create(tt_entries);
    } else {
        pi->proc = uci_proc_start(def->path);
        if (!pi->proc || !uci_proc_handshake(pi->proc, 10000)) {
            fprintf(stderr, "[arena] WARNING: uci player '%s' failed to start/handshake\n", def->path);
            pi->uci_dead = 1;
        }
    }
    return pi;
}

/* A uci_dead PlayerInstance stays dead FOREVER unless something calls
 * this: play_move_uci()'s own respawn attempt (see play_one_game())
 * only retries once, mid-game, on a live-but-misbehaving process. Once
 * that retry also fails, uci_dead=1 is permanent and every subsequent
 * ply for that (worker, player) pair — including every future GAME
 * this worker plays for that player — silently auto-forfeits via
 * play_move_uci()'s very first guard, with no further log line. A
 * transient startup hiccup (see g_spawn_mtx's header comment for one
 * real cause of exactly this) would otherwise take one worker's copy
 * of one player out of the match for good, quietly wrecking that
 * worker's share of the cross-table.
 *
 * Called once at the start of EVERY game (see play_one_game()) so a
 * dead instance gets a fresh, independent chance each game instead of
 * being written off after its first bad game. Only touches instances
 * that are actually dead — a live instance is untouched (no extra
 * spawn cost on the hot path). */
static void uci_instance_revive_if_dead(PlayerInstance *pi) {
    if (!pi || pi->def->kind != PLAYER_UCI || !pi->uci_dead) return;
    if (pi->proc) { uci_proc_stop(pi->proc); pi->proc = NULL; }
    pi->proc = uci_proc_start(pi->def->path);
    if (pi->proc && uci_proc_handshake(pi->proc, 10000)) {
        pi->uci_dead = 0;
    } else {
        fprintf(stderr, "[arena] WARNING: uci player '%s' failed to start/handshake on revive attempt — "
                        "forfeiting this worker's remaining games with it too\n", pi->def->path);
    }
}

static void player_instance_destroy(PlayerInstance *pi) {
    if (!pi) return;
    if (pi->undo) free(pi->undo);
    if (pi->nnue) zfree32(pi->nnue);
    if (pi->ss)   search_state_free(pi->ss);
    if (pi->tt)   tt_destroy(pi->tt);
    if (pi->proc) uci_proc_stop(pi->proc);
    free(pi);
}

/* Rebind a PLAYER_NET instance's Board to this instance's own undo/nnue
 * after board_load_fen() — same bug/fix as selfplay.c's
 * worker_reset_board(): board_load_fen() unconditionally rebinds
 * Board::undo/Board::nnue to the process-global g_undo/g_nnue_accum. */
/* Bind a net: player's PRIVATE state to the shared game board.
 *
 * ONLY the NNUE accumulator is swapped. The undo stack is deliberately
 * NOT touched, and this is the whole point of the function's current
 * shape — the previous version did:
 *
 *     pi->undo_top = 0;              // <-- reset the game's move history
 *     b->undo     = pi->undo;        // <-- and swap to this player's stack
 *
 * which broke two things at once:
 *
 *   1. `pi->undo_top = 0` on EVERY ply threw away the board's move
 *      history, so board_is_draw()'s repetition and fifty-move detection
 *      had nothing to look at. Games could not be drawn by repetition,
 *      and threefold positions were played on as if novel.
 *   2. The two players own separate undo arrays, so consecutive game
 *      moves landed in alternating stacks with the depth counter jumping
 *      between them. When a depth ran past STACK_SIZE, board_make() hit
 *      its `[BUG] undo overflow` guard and RETURNED WITHOUT MAKING THE
 *      MOVE — the game then continued from a board that silently did not
 *      match what either engine believed. That is what produced the
 *      overflow spam and the "failed to respond" respawns in a v400 vs
 *      v314 match.
 *
 * The game board now owns ONE undo stack for the whole game (allocated in
 * the worker, see game_undo below). Searches push and pop symmetrically
 * on top of it and always return to the same depth, so sharing it between
 * two adversaries leaks nothing — unlike the TT, which stays per player.
 *
 * The accumulator does NOT need the same treatment: search_best() calls
 * nnue_rebuild() on entry, so each player's accumulator is re-seeded from
 * the current board at the start of its own search. */
static void net_instance_bind_board(PlayerInstance *pi, Board *b) {
    b->nnue = pi->nnue;
}

/* ═══════════════════════════════════════════════════════════════════
 * Opening pool — EPD (one FEN per line) or PGN (SAN-replayed to a
 * final FEN). See header comment "OPENINGS".
 * ═══════════════════════════════════════════════════════════════════ */
typedef struct {
    char **fens;
    int    n;
    int    cap;
} OpeningPool;

static void opening_pool_add(OpeningPool *op, const char *fen) {
    if (op->n >= op->cap) {
        op->cap = op->cap ? op->cap * 2 : 64;
        op->fens = (char **)realloc(op->fens, op->cap * sizeof(char *));
    }
    op->fens[op->n++] = xstrdup(fen);
}

static int load_epd_openings(const char *path, OpeningPool *op) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        char *s = line;
        while (*s == ' ' || *s == '\t') s++;
        if (*s == '#' || *s == 0 || *s == '\n' || *s == '\r') continue;
        /* Count whitespace-separated tokens; need >= 4 (board stm castling ep). */
        char tmp[1024]; strncpy(tmp, s, sizeof(tmp) - 1); tmp[sizeof(tmp) - 1] = 0;
        char *tok[8]; int ntok = 0;
        char *save = NULL;
        char *t = strtok_r(tmp, " \t\r\n", &save);
        while (t && ntok < 8) { tok[ntok++] = t; t = strtok_r(NULL, " \t\r\n", &save); }
        if (ntok < 4) continue;
        char fen[512];
        if (ntok >= 6)
            snprintf(fen, sizeof(fen), "%s %s %s %s %s %s", tok[0], tok[1], tok[2], tok[3], tok[4], tok[5]);
        else
            snprintf(fen, sizeof(fen), "%s %s %s %s 0 1", tok[0], tok[1], tok[2], tok[3]);
        opening_pool_add(op, fen);
    }
    fclose(f);
    return 0;
}

/* san_resolve() moved to tools/opening_pool.c (shared with selfplay.c
 * — see opening_pool.h). Declared there, used here unchanged. */

/* Replay up to `opening_plies` plies of PGN mainline SAN starting from
 * the standard position on a throwaway Board, appending the resulting
 * FEN after each successfully-parsed game to `op`. Games are separated
 * by a blank line following their movetext (standard PGN). Tag lines
 * ("[Event ...]") and result tokens (1-0/0-1/1/2-1/2/*) are skipped. */
static int load_pgn_openings(const char *path, OpeningPool *op, int opening_plies) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;

    Board scratch;
    UndoFrame *undo = (UndoFrame *)malloc(STACK_SIZE * sizeof(UndoFrame));
    NnueAccum *nnue = (NnueAccum *)zmalloc32(sizeof(NnueAccum));
    memset(nnue, 0, sizeof(NnueAccum));

    static const char *STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    int in_game = 0, plies = 0, games = 0;
    char line[4096];

    board_load_fen(&scratch, STARTPOS);
    int undo_top = 0;
    scratch.undo = undo; scratch.undo_top = &undo_top; scratch.nnue = nnue;

    while (fgets(line, sizeof(line), f)) {
        char *s = line;
        while (*s == ' ' || *s == '\t') s++;
        if (s[0] == '[') continue;
        int blank = 1;
        for (char *c = s; *c; c++) if (!isspace((unsigned char)*c)) { blank = 0; break; }
        if (blank) {
            if (in_game) {
                if (plies > 0) {
                    char fen[512]; board_to_fen(&scratch, fen);
                    opening_pool_add(op, fen);
                    games++;
                }
                in_game = 0; plies = 0;
                board_load_fen(&scratch, STARTPOS);
                undo_top = 0; scratch.undo = undo; scratch.undo_top = &undo_top; scratch.nnue = nnue;
            }
            continue;
        }
        in_game = 1;
        char tmp[4096]; strncpy(tmp, s, sizeof(tmp) - 1); tmp[sizeof(tmp) - 1] = 0;
        char *save = NULL;
        char *tokraw = strtok_r(tmp, " \t\r\n", &save);
        while (tokraw && plies < opening_plies) {
            char *tok = tokraw;
            /* Strip a leading "N." or "N..." move-number prefix, which
             * may be glued to the move itself ("1.b3") or standalone
             * ("1." then a separate "b3" token). */
            char *dot = strrchr(tok, '.');
            if (dot) {
                int all_digits = 1;
                for (char *c = tok; c < dot; c++) if (!isdigit((unsigned char)*c)) { all_digits = 0; break; }
                if (all_digits && dot != tok) {
                    tok = dot + 1;
                    if (*tok == 0) { tokraw = strtok_r(NULL, " \t\r\n", &save); continue; }
                }
            }
            if (!strcmp(tok, "1-0") || !strcmp(tok, "0-1") ||
                !strcmp(tok, "1/2-1/2") || !strcmp(tok, "*")) break;

            Move mv;
            if (!san_resolve(&scratch, tok, &mv)) break;
            board_make(&scratch, &mv);
            plies++;
            tokraw = strtok_r(NULL, " \t\r\n", &save);
        }
    }
    if (in_game && plies > 0) {
        char fen[512]; board_to_fen(&scratch, fen);
        opening_pool_add(op, fen);
        games++;
    }
    fclose(f);
    free(undo);
    zfree32(nnue);
    fprintf(stderr, "[arena] loaded %d opening(s) from %s (%d plies each, max)\n", games, path, opening_plies);
    return 0;
}

static int has_suffix(const char *s, const char *suf) {
    size_t ls = strlen(s), lsuf = strlen(suf);
    return ls >= lsuf && !strcasecmp(s + ls - lsuf, suf);
}

/* Context passed through opening_walk_files() callbacks when the path
 * given to load_openings() turns out to be a directory. */
typedef struct { OpeningPool *op; int opening_plies; int loaded; } WalkOpenCtx;

static void load_openings_walk_cb(const char *filepath, void *ud) {
    WalkOpenCtx *wc = (WalkOpenCtx *)ud;
    if (has_suffix(filepath, ".pgn")) {
        if (load_pgn_openings(filepath, wc->op, wc->opening_plies) == 0) wc->loaded++;
    } else if (has_suffix(filepath, ".epd") || has_suffix(filepath, ".fen")) {
        if (load_epd_openings(filepath, wc->op) == 0) wc->loaded++;
    }
    /* other extensions (e.g. .idx cache files) are silently skipped */
}

int load_openings(const char *path, OpeningPool *op, int opening_plies) {
    /* Single file — dispatch by extension (original behaviour). */
    if (has_suffix(path, ".pgn")) return load_pgn_openings(path, op, opening_plies);
    if (has_suffix(path, ".epd") || has_suffix(path, ".fen")) return load_epd_openings(path, op);

    /* Check whether the path is a directory; if so, walk it recursively
     * loading every .pgn/.epd/.fen file found.  opening_walk_files() is
     * declared in opening_pool.h and implemented in opening_pool.c —
     * already linked into arena.exe via the Makefile's arena target. */
    struct stat st;
    if (stat(path, &st) == 0 && (st.st_mode & S_IFDIR)) {
        WalkOpenCtx wc = { op, opening_plies, 0 };
        int rc = opening_walk_files(path, load_openings_walk_cb, &wc);
        if (rc < 0) return -1;  /* stat() failed inside the walker */
        if (wc.loaded == 0) {
            fprintf(stderr, "[arena] directory '%s' contained no .pgn/.epd/.fen files\n", path);
            return -1;
        }
        return 0;
    }

    /* Unknown extension and not a directory: fall back to EPD parser
     * (treats the file as one FEN per line, same as before). */
    return load_epd_openings(path, op);
}

/* ═══════════════════════════════════════════════════════════════════
 * Match configuration and task list
 * ═══════════════════════════════════════════════════════════════════ */
typedef struct {
    int      games;            /* per pairing if >2 players, else total */
    int      threads;
    int      movetime_ms;
    long     nodes;
    int      max_plies;
    uint64_t seed;
    double   tt_mb;
    int      gauntlet;              /* 1 if --gauntlet was given */
    char     gauntlet_spec[128];    /* raw arg: numeric index OR name/path substring */
    int      gauntlet_idx;          /* resolved by resolve_gauntlet() after all --player parsed */
    char     openings_path[512];
    int      opening_plies;
    int      sprt;
    double   elo0, elo1, alpha, beta;
    char     json_path[512];
    char     pgn_path[512];
    char     bin_path[512];   /* "" = no --bin output */
    char     epd_path[512];   /* "" = no --epd output */
    int      uci_move_timeout_ms;
    PlayerDef players[MAX_PLAYERS];
    int      n_players;
} Config;

/* Every field here is read from an ARENA_DEFAULT_* constant declared
 * in the CONFIGURATION block at the top of this file — see that
 * block's comment for why (project-wide convention: CLI default must
 * literally BE the documented config constant, never a second copy
 * of the same literal). `--threads 0` (explicit) still means
 * "autodetect", same as leaving it unset — detect_cpus() is only
 * invoked here because ARENA_DEFAULT_THREADS is itself 0. */
static void config_defaults(Config *c) {
    memset(c, 0, sizeof(*c));
    c->games        = ARENA_DEFAULT_GAMES;
    c->threads      = ARENA_DEFAULT_THREADS > 0 ? ARENA_DEFAULT_THREADS : detect_cpus();
    c->movetime_ms  = ARENA_DEFAULT_MOVETIME_MS;
    c->nodes        = ARENA_DEFAULT_NODES;
    c->max_plies    = ARENA_DEFAULT_MAX_PLIES;
    c->seed         = ARENA_DEFAULT_SEED;
    c->tt_mb        = ARENA_DEFAULT_TT_MB;
    c->opening_plies = ARENA_DEFAULT_OPENING_PLIES;
    c->elo0 = ARENA_DEFAULT_ELO0; c->elo1 = ARENA_DEFAULT_ELO1;
    c->alpha = ARENA_DEFAULT_ALPHA; c->beta = ARENA_DEFAULT_BETA;
    c->uci_move_timeout_ms = ARENA_DEFAULT_UCI_TIMEOUT_MS;
    c->sprt = ARENA_DEFAULT_SPRT_ENABLED;
    strncpy(c->openings_path, ARENA_DEFAULT_OPENINGS_PATH, sizeof(c->openings_path) - 1);
    strncpy(c->json_path, ARENA_DEFAULT_JSON_PATH, sizeof(c->json_path) - 1);
    strncpy(c->pgn_path, ARENA_DEFAULT_PGN_PATH, sizeof(c->pgn_path) - 1);
    strncpy(c->bin_path, ARENA_DEFAULT_BIN_PATH, sizeof(c->bin_path) - 1);
    strncpy(c->epd_path, ARENA_DEFAULT_EPD_PATH, sizeof(c->epd_path) - 1);
    strncpy(c->gauntlet_spec, ARENA_DEFAULT_GAUNTLET_SPEC, sizeof(c->gauntlet_spec) - 1);
}

/* Case-insensitive substring search (no strcasestr dependency — not
 * available on every libc this project targets). */
static const char *ci_strstr(const char *hay, const char *needle) {
    size_t nlen = strlen(needle);
    if (nlen == 0) return hay;
    for (const char *h = hay; *h; h++) {
        size_t i = 0;
        while (i < nlen && h[i] && tolower((unsigned char)h[i]) == tolower((unsigned char)needle[i])) i++;
        if (i == nlen) return h;
    }
    return NULL;
}

/* Resolves cfg->gauntlet_spec (numeric player index, or a case-
 * insensitive substring of a player's name/path) to cfg->gauntlet_idx.
 * Called once, after every --player has been parsed. Returns 0 on
 * success, -1 if the spec matches no player. */
static int resolve_gauntlet(Config *cfg) {
    if (!cfg->gauntlet) return 0;
    char *endp = NULL;
    long idx = strtol(cfg->gauntlet_spec, &endp, 10);
    if (endp && *endp == 0 && endp != cfg->gauntlet_spec && idx >= 0 && idx < cfg->n_players) {
        cfg->gauntlet_idx = (int)idx;
        return 0;
    }
    for (int i = 0; i < cfg->n_players; i++) {
        if (ci_strstr(cfg->players[i].name, cfg->gauntlet_spec) ||
            ci_strstr(cfg->players[i].path, cfg->gauntlet_spec)) {
            cfg->gauntlet_idx = i;
            return 0;
        }
    }
    fprintf(stderr, "[arena] --gauntlet '%s' matches no player (by index or name/path substring)\n",
            cfg->gauntlet_spec);
    return -1;
}

/* One unit of work: play ONE game, playerA as White iff `a_white`. */
typedef struct {
    int pairA, pairB;    /* indices into cfg->players */
    int opening_idx;     /* -1 = startpos */
    int a_white;
} GameTask;

typedef struct {
    GameTask *tasks;
    int       n;
    int       cap;
} TaskList;

static void tasklist_add(TaskList *tl, int a, int b, int op, int a_white) {
    if (tl->n >= tl->cap) {
        tl->cap = tl->cap ? tl->cap * 2 : 256;
        tl->tasks = (GameTask *)realloc(tl->tasks, tl->cap * sizeof(GameTask));
    }
    GameTask *t = &tl->tasks[tl->n++];
    t->pairA = a; t->pairB = b; t->opening_idx = op; t->a_white = a_white;
}

/* Build the antithetic task list for one pairing (a,b): `games` games,
 * each opening played once with `a` as White and once as Black, back
 * to back (see header comment "ANTITHETIC PAIRING"). If `games` is odd
 * the last game is unpaired (still logged, flagged in the summary). */
static void add_pairing_tasks(TaskList *tl, int a, int b, int games, int n_openings) {
    int pairs = games / 2;
    for (int i = 0; i < pairs; i++) {
        int op = n_openings > 0 ? (i % n_openings) : -1;
        tasklist_add(tl, a, b, op, 1);
        tasklist_add(tl, a, b, op, 0);
    }
    if (games & 1) {
        int op = n_openings > 0 ? (pairs % n_openings) : -1;
        tasklist_add(tl, a, b, op, 1);
    }
}

static TaskList build_task_list(const Config *cfg, int n_openings) {
    TaskList tl; memset(&tl, 0, sizeof(tl));
    if (cfg->gauntlet) {
        for (int i = 0; i < cfg->n_players; i++)
            if (i != cfg->gauntlet_idx) add_pairing_tasks(&tl, cfg->gauntlet_idx, i, cfg->games, n_openings);
    } else if (cfg->n_players == 2) {
        add_pairing_tasks(&tl, 0, 1, cfg->games, n_openings);
    } else {
        for (int i = 0; i < cfg->n_players; i++)
            for (int j = i + 1; j < cfg->n_players; j++)
                add_pairing_tasks(&tl, i, j, cfg->games, n_openings);
    }
    return tl;
}

/* Round a requested TT memory budget (MB) down to a valid TTable size:
 * a power of two multiple of TT_BUCKETS. ~26 bytes/logical entry. */
static size_t tt_entries_for_mb(double mb) {
    const double bytes_per_entry = 26.0;
    size_t want = (size_t)((mb * 1024.0 * 1024.0) / bytes_per_entry);
    size_t n = TT_BUCKETS;
    while (n * 2 <= want) n *= 2;
    if (n < (size_t)(TT_BUCKETS * 512)) n = (size_t)TT_BUCKETS * 512;
    return n;
}

/* ═══════════════════════════════════════════════════════════════════
 * Pairwise results + Elo (matches tests/elo_calc.py's trinomial model
 * exactly — see that file's docstring for the derivation).
 * ═══════════════════════════════════════════════════════════════════ */
typedef struct {
    long long w, d, l;   /* from player[first]'s perspective */
} PairResult;

static PairResult g_results[MAX_PLAYERS][MAX_PLAYERS];
static pthread_mutex_t g_results_mtx = PTHREAD_MUTEX_INITIALIZER;

static void record_result(int a, int b, int a_white, int outcome /* 1=white win,-1=black win,0=draw */) {
    int a_won  = (a_white && outcome == 1) || (!a_white && outcome == -1);
    int b_won  = (a_white && outcome == -1) || (!a_white && outcome == 1);
    pthread_mutex_lock(&g_results_mtx);
    if (outcome == 0) { g_results[a][b].d++; g_results[b][a].d++; }
    else if (a_won)   { g_results[a][b].w++; g_results[b][a].l++; }
    else if (b_won)   { g_results[a][b].l++; g_results[b][a].w++; }
    pthread_mutex_unlock(&g_results_mtx);
}

/* Trinomial Elo diff + 95% CI — verbatim port of elo_calc.py's
 * elo_difference() so numbers from this tool and from Python tooling
 * (tests/run_tournament*.py) are directly comparable. */
void elo_diff_ci(long long w, long long d, long long l, double *elo, double *ci) {
    long long n = w + d + l;
    if (n == 0) { *elo = 0; *ci = INFINITY; return; }
    double p = (w + 0.5 * d) / (double)n;
    if (p <= 0.0) { *elo = -INFINITY; *ci = INFINITY; return; }
    if (p >= 1.0) { *elo = INFINITY; *ci = INFINITY; return; }
    double e = -400.0 * log10(1.0 / p - 1.0);
    double wf = w / (double)n, df = d / (double)n, lf = l / (double)n;
    double var = wf * (1.0 - p) * (1.0 - p) + df * (0.5 - p) * (0.5 - p) + lf * (0.0 - p) * (0.0 - p);
    double se_p = sqrt(var / n);
    double d_elo_dp = 400.0 / (log(10.0) * p * (1.0 - p));
    double se_elo = fabs(d_elo_dp) * se_p;
    *elo = e; *ci = 1.96 * se_elo;
}

/* ── SPRT: GSPRT / Wald log-likelihood ratio under a normal
 * approximation of the mean score, trinomial variance — the same
 * method used by cutechess-cli's Sprt.cpp and Fishtest. `var` is
 * exactly elo_calc.py's per-game trinomial variance (see
 * elo_diff_ci() above), so this is directly consistent with this
 * tool's own Elo/CI numbers, not a separately-tuned formula. ──────── */
typedef enum { SPRT_CONTINUE, SPRT_ACCEPT_H0, SPRT_ACCEPT_H1 } SprtVerdict;

double score_from_elo(double elo) { return 1.0 / (1.0 + pow(10.0, -elo / 400.0)); }

SprtVerdict sprt_llr(long long w, long long d, long long l,
                      double elo0, double elo1, double alpha, double beta,
                      double *out_llr, double *out_lo, double *out_hi) {
    long long n = w + d + l;
    double lo = log(beta / (1.0 - alpha));
    double hi = log((1.0 - beta) / alpha);
    *out_lo = lo; *out_hi = hi;
    if (n < 10) { *out_llr = 0.0; return SPRT_CONTINUE; }   /* too few games for a stable variance estimate */

    double p = (w + 0.5 * d) / (double)n;
    double wf = w / (double)n, df = d / (double)n, lf = l / (double)n;
    double var = wf * (1.0 - p) * (1.0 - p) + df * (0.5 - p) * (0.5 - p) + lf * (0.0 - p) * (0.0 - p);
    if (var < 1e-9) var = 1e-9;

    double mu0 = score_from_elo(elo0), mu1 = score_from_elo(elo1);
    double llr = (double)n * (mu1 - mu0) * (p - (mu0 + mu1) / 2.0) / var;
    *out_llr = llr;
    if (llr <= lo) return SPRT_ACCEPT_H0;
    if (llr >= hi) return SPRT_ACCEPT_H1;
    return SPRT_CONTINUE;
}

/* ═══════════════════════════════════════════════════════════════════
 * The game loop — one worker thread, two adversary PlayerInstances.
 * ═══════════════════════════════════════════════════════════════════ */
typedef enum { OUT_DRAW = 0, OUT_WHITE = 1, OUT_BLACK = -1 } Outcome;

/* Ask a PLAYER_NET instance for its move at the current position.
 * `out_score_cp`, when non-NULL, receives the search's STM-relative
 * score — this is what --bin records as eval_cp, and it is only
 * available for net: players (arena does not parse score lines from
 * external UCI engines; see the --bin restriction in main()). */
static int play_move_net(PlayerInstance *pi, Board *b, const Config *cfg, Move *out,
                          int *out_score_cp) {
    SearchParams sp; memset(&sp, 0, sizeof(sp));
    sp.max_depth = MAX_PLY - 1;
    sp.start_depth = 0;
    sp.time_limit_ms = cfg->movetime_ms;
    sp.node_limit = cfg->nodes;
    sp.multi_pv = 1;
    sp.threads = 1;      /* one OS thread per game slot — see header "CONCURRENCY MODEL" */
    sp.stop = NULL;
    sp.search_state = pi->ss;
    sp.info_cb = NULL;
    sp.tt = pi->tt;
    sp.mpv_share_budget = 0;

    SearchResult res = search_best(b, &sp);
    if (res.bests[0].from == 0 && res.bests[0].to == 0) return 0;   /* no legal move */
    *out = res.bests[0];
    if (out_score_cp) *out_score_cp = res.score;
    return 1;
}

/* Ask a PLAYER_UCI instance for its move. `moves_uci` is the
 * space-separated UCI move list from the opening FEN to the current
 * position (UCI engines are given "position fen X moves ...", not a
 * fresh FEN every ply, so their own internal repetition/history
 * tracking works exactly as it would for a real game).
 *
 * Returns 1 with `*out` filled in on success.
 *
 * Returns 0 on any GENUINE protocol failure (no "bestmove" line within
 * cfg->uci_move_timeout_ms, or the pipe/process died) — the caller
 * treats this as "the engine is broken" and attempts a respawn (see
 * "CONCURRENCY MODEL").
 *
 * `*out_no_legal_move` (when non-NULL) is set to 1 for the OTHER,
 * entirely healthy way this returns 0: the engine cleanly answered
 * "I have no legal move here" — "bestmove (none)" or the UCI null-move
 * sentinel "bestmove 0000" (zchezz's own move_to_uci() prints exactly
 * "0000" for a zeroed Move; see main.c). This is the normal way a
 * conforming UCI engine reports checkmate/stalemate, and it is ALSO
 * what a healthy zchezz process sends if search_best() can't complete
 * even one depth within the time budget under heavy host contention —
 * neither case means the process is broken, so the caller must NOT
 * spend a respawn attempt (and forfeit-through-that-attempt) on it.
 * Before this distinction existed, arena's parser did not recognize
 * "0000" at all, fell through to move_from_uci() (which correctly
 * rejects it as unparseable), and the caller could not tell that
 * apart from a real hang — every "0000" reply killed and respawned a
 * perfectly healthy engine process. That was empirically the dominant
 * source of "failed to respond — respawning" spam in a v400 vs v314
 * match at --threads > 1 (see repro notes near g_spawn_mtx above for
 * the other, unrelated hang this file also had). */
static int play_move_uci(PlayerInstance *pi, Board *b, const char *start_fen,
                          const char *moves_uci, const Config *cfg, Move *out,
                          int *out_no_legal_move) {
    if (out_no_legal_move) *out_no_legal_move = 0;
    if (pi->uci_dead || !pi->proc) return 0;
    char cmd[8192];
    if (moves_uci[0])
        snprintf(cmd, sizeof(cmd), "position fen %s moves %s", start_fen, moves_uci);
    else
        snprintf(cmd, sizeof(cmd), "position fen %s", start_fen);
    uci_proc_send(pi->proc, cmd);

    if (cfg->movetime_ms > 0)
        snprintf(cmd, sizeof(cmd), "go movetime %d", cfg->movetime_ms);
    else
        snprintf(cmd, sizeof(cmd), "go nodes %ld", cfg->nodes);
    uci_proc_send(pi->proc, cmd);

    char line[UCI_LINE_MAX];
    int rc = uci_proc_wait_for(pi->proc, "bestmove", line, sizeof(line), cfg->uci_move_timeout_ms);
    if (rc != 1) return 0;   /* genuine protocol failure/timeout */

    char mvstr[16] = {0};
    sscanf(line, "bestmove %15s", mvstr);
    if (!mvstr[0] || !strcmp(mvstr, "(none)") || !strcmp(mvstr, "0000")) {
        if (out_no_legal_move) *out_no_legal_move = 1;
        return 0;
    }
    return move_from_uci(b, mvstr, out);
}

/* SAN move text for `m`, played from position `b` (BEFORE the move —
 * disambiguation needs the other legal moves of the same piece type
 * at THIS position). Does NOT append a check/mate suffix — the caller
 * appends '+'/'#' itself after board_make(), since that requires the
 * POST-move position (see write_pgn_game()'s only caller, below).
 * Only invoked when --pgn is set (see play_one_game()'s `pgn_movetext`
 * parameter) — this is deliberately NOT on the hot path otherwise,
 * since board_gen_moves() for disambiguation is real extra cost per
 * move that a plain net:/uci: A/B match should never pay. */
/* move_to_san() moved to tools/opening_pool.c (shared with selfplay.c
 * — see opening_pool.h). Declared there, used here unchanged. */

/* One EPD row, captured pre-move during play_one_game() — same shape as
 * tools/selfplay.c's EpdRow, PLUS an `evaluator` field selfplay.c does
 * not need. WHY: selfplay.c is one engine playing itself, so a constant
 * "Zchezz-native" string in c2 is accurate for every row. arena.c is not
 * — a game can be net: vs uci:, or net: vs a DIFFERENT net: (a different
 * v3.14/v4.00 build or weight file). If every row silently said one
 * fixed string, two different evaluators' opinions would be recorded
 * under one undifferentiated label — exactly the mixed-evaluator
 * mislabeling CLAUDE.md rule 10's c2 opcode exists to prevent. So each
 * row here captures the ACTUAL mover's player name (cfg->players[i].name,
 * user-assigned via --player NAME=... or defaulted to the path/exe
 * basename — see parse_player_spec()), decided at the moment the row is
 * captured, not fixed for the whole game. */
typedef struct {
    char fen[128];
    int  cp_white;      /* WHITE-relative eval; 0 for uci: movers (arena
                          * does not parse score lines from external
                          * engines — see play_move_uci()'s header) */
    char evaluator[128]; /* the mover's player name for THIS row */
} EpdRow;

/* Plays one game between instA (White iff a_white) and instB, starting
 * from `start_fen`. Returns the outcome; White/Black-relative, exactly
 * like tools/selfplay.c's GameOutcome (kept as an int here for the
 * simpler two-player-adversary shape — no per-position sample buffer
 * to fill, arena only needs the final result).
 *
 * `pgn_movetext`/`pgn_cap`: if pgn_movetext is non-NULL, standard PGN
 * movetext ("1. e4 e5 2. Nf3 ..." with '+'/'#' suffixes) is built up
 * into it as the game is played — see move_to_san() above and header
 * comment "PGN OUTPUT". Pass NULL to skip this entirely (the default
 * — see --pgn in print_usage()), which avoids the extra
 * board_gen_moves() disambiguation cost on every move.
 *
 * `epd_rows`/`epd_rows_cap`/`n_epd_rows`: if epd_rows is non-NULL, one
 * EpdRow is appended per ply (pre-move position) — see EpdRow's header
 * comment above for why this, unlike --bin, has no same-engine
 * restriction. Pass NULL to skip entirely (the default). */
static Outcome play_one_game(PlayerInstance *instA, PlayerInstance *instB, int a_white,
                              const char *start_fen, const Config *cfg,
                              long long *out_plies, char *pgn_movetext, size_t pgn_cap,
                              SelfplaySample *samples, size_t samples_cap, size_t *n_samples,
                              EpdRow *epd_rows, size_t epd_rows_cap, size_t *n_epd_rows,
                              UndoFrame *game_undo, int *game_undo_top) {
    Board board;
    board_load_fen(&board, start_fen);
    /* board_load_fen() rebinds undo/nnue to the process globals (g_undo /
     * g_nnue_accum). Point the board at THIS GAME's undo stack, once, and
     * never reset it again for the rest of the game — the full move
     * history is what board_is_draw() needs for repetition detection. */
    *game_undo_top   = 0;
    board.undo       = game_undo;
    board.undo_top   = game_undo_top;

    PlayerInstance *white = a_white ? instA : instB;
    PlayerInstance *black = a_white ? instB : instA;

    /* TT clear before every game for EVERY net: participant — see
     * header comment "TT ISOLATION IS MANDATORY" for why this must be
     * a physical wipe, not a generation bump (identical reasoning to
     * selfplay.c's TT POLICY section: repetition/draw scores are
     * path-dependent and the TT does not know which game wrote them). */
    if (instA->def->kind == PLAYER_NET) tt_clear(instA->tt);
    if (instB->def->kind == PLAYER_NET) tt_clear(instB->tt);
    /* Give a previously-dead uci: instance a fresh chance every game —
     * see uci_instance_revive_if_dead()'s header comment. No-op for
     * live instances. */
    uci_instance_revive_if_dead(instA);
    uci_instance_revive_if_dead(instB);
    if (instA->def->kind == PLAYER_UCI && instA->proc && !instA->uci_dead) uci_proc_send(instA->proc, "ucinewgame");
    if (instB->def->kind == PLAYER_UCI && instB->proc && !instB->uci_dead) uci_proc_send(instB->proc, "ucinewgame");

    char moves_uci[8192] = {0};
    size_t moves_len = 0;
    size_t pgn_len = 0;
    if (pgn_movetext && pgn_cap) pgn_movetext[0] = 0;

    if (n_samples) *n_samples = 0;
    if (n_epd_rows) *n_epd_rows = 0;

    Outcome result = OUT_DRAW;
    long long ply;
    for (ply = 0; ply < cfg->max_plies; ply++) {
        int dr = board_is_draw(&board);
        if (dr == 1 || dr == 2) { result = OUT_DRAW; break; }

        int white_to_move = (board.turn == COL_W);
        PlayerInstance *mover = white_to_move ? white : black;

        Move mv;
        int ok;
        int move_score_cp = 0;
        if (mover->def->kind == PLAYER_NET) {
            net_instance_bind_board(mover, &board);
            ok = play_move_net(mover, &board, cfg, &mv, &move_score_cp);
        } else {
            int no_legal_move = 0;
            ok = play_move_uci(mover, &board, start_fen, moves_uci, cfg, &mv, &no_legal_move);
            /* Only a GENUINE protocol failure (no_legal_move == 0: no
             * "bestmove" in time, or the process/pipe died) burns a
             * respawn attempt. A clean "bestmove (none)"/"0000" is a
             * healthy engine answer (see play_move_uci()'s header
             * comment) — nothing to recover from, so don't kill a
             * perfectly good process over it. */
            if (!ok && !no_legal_move && !mover->uci_dead) {
                /* One respawn attempt on protocol failure (crash/hang) —
                 * see header "CONCURRENCY MODEL". */
                fprintf(stderr, "[arena] uci player '%s' failed to respond — respawning\n", mover->def->path);
                uci_proc_stop(mover->proc);
                mover->proc = uci_proc_start(mover->def->path);
                if (mover->proc && uci_proc_handshake(mover->proc, 10000)) {
                    ok = play_move_uci(mover, &board, start_fen, moves_uci, cfg, &mv, &no_legal_move);
                }
                if (!ok && !no_legal_move) mover->uci_dead = 1;
            }
        }

        if (!ok) {
            /* No legal move (checkmate/stalemate) OR a dead uci player
             * (forfeit from whatever position it left the board in).
             * board_in_check() is always authoritative here — it reads
             * arena's own board, not anything the engine claimed — so
             * this distinction is correct for both net: and uci: kinds
             * uniformly, including a clean uci: "(none)"/"0000" answer. */
            if (!board_in_check(&board)) {
                result = OUT_DRAW;
            } else {
                result = white_to_move ? OUT_BLACK : OUT_WHITE;   /* mover loses */
            }
            break;
        }

        if (mover->def->kind == PLAYER_UCI) {
            char uci[8]; move_to_uci(&mv, uci);
            size_t ul = strlen(uci);
            if (moves_len + ul + 2 < sizeof(moves_uci)) {
                if (moves_len) moves_uci[moves_len++] = ' ';
                memcpy(moves_uci + moves_len, uci, ul);
                moves_len += ul;
                moves_uci[moves_len] = 0;
            }
        }

        char san_buf[16] = {0};
        if (pgn_movetext) move_to_san(&board, &mv, san_buf, sizeof(san_buf));

        /* --bin: record the position BEFORE the move, exactly like
         * tools/selfplay.c does. eval_cp must be STM-relative, but
         * SearchResult.score is WHITE-relative — flip for Black, or the
         * training target is sign-inverted for half the samples. */
        if (samples && n_samples && *n_samples < samples_cap) {
            SelfplaySample *s = &samples[(*n_samples)++];
            memset(s, 0, sizeof(*s));
            memcpy(s->board, board.b, 64);
            s->stm         = white_to_move ? 0 : 1;
            s->rule50      = (uint8_t)board.hm;
            s->castling    = (uint8_t)board.ca;
            s->ep_file     = (uint8_t)(board.ep >= 0 ? (board.ep & 7) : 8);
            s->eval_cp     = (int16_t)(white_to_move ? move_score_cp : -move_score_cp);
            s->move_played = sample_pack_move(mv.from, mv.to, mv.prom, mv.epc);
            s->game_result = 0;   /* filled in the second pass below */
        }

        /* --epd: record the position BEFORE the move, same as --bin above,
         * but with the mover's identity attached per-row (see EpdRow's
         * header comment) — this is what makes --epd safe for mixed
         * net:/uci: or net-A-vs-net-B games where --bin is refused. */
        if (epd_rows && n_epd_rows && *n_epd_rows < epd_rows_cap) {
            EpdRow *row = &epd_rows[(*n_epd_rows)++];
            board_to_fen(&board, row->fen);
            row->cp_white = (mover->def->kind == PLAYER_NET)
                ? (white_to_move ? move_score_cp : -move_score_cp)
                : 0;   /* uci: movers: arena never parses external score lines */
            strncpy(row->evaluator, mover->def->name, sizeof(row->evaluator) - 1);
            row->evaluator[sizeof(row->evaluator) - 1] = 0;
        }

        board_make(&board, &mv);

        if (pgn_movetext) {
            /* Check/mate suffix needs the POST-move position — see
             * move_to_san()'s header comment for why this can't be
             * folded into that function. board_gen_moves() here is
             * trusted to return LEGAL moves (same assumption
             * san_resolve()/move_from_uci() already make elsewhere in
             * this file), so "in check AND zero moves" == checkmate. */
            Move tmp_legal[MAX_MOVES];
            int check_now = board_in_check(&board);
            int has_moves = board_gen_moves(&board, tmp_legal) > 0;
            size_t sl = strlen(san_buf);
            if (check_now && sl + 1 < sizeof(san_buf)) {
                san_buf[sl++] = has_moves ? '+' : '#';
                san_buf[sl] = 0;
            }
            char tok[32];
            int tn = white_to_move
                ? snprintf(tok, sizeof(tok), "%lld. %s ", ply / 2 + 1, san_buf)
                : snprintf(tok, sizeof(tok), "%s ", san_buf);
            if (tn > 0 && pgn_len + (size_t)tn < pgn_cap) {
                memcpy(pgn_movetext + pgn_len, tok, (size_t)tn);
                pgn_len += (size_t)tn;
                pgn_movetext[pgn_len] = 0;
            }
        }
    }
    if (ply >= cfg->max_plies) result = OUT_DRAW;

    /* Second pass — game_result is relative to the side to move IN THAT
     * POSITION, which only becomes knowable once the game has ended.
     * Same contract as tools/selfplay.c. */
    if (samples && n_samples) {
        for (size_t i = 0; i < *n_samples; i++) {
            int8_t gr;
            if (result == OUT_DRAW) gr = 0;
            else {
                int sample_is_white = (samples[i].stm == 0);
                int white_won = (result == OUT_WHITE);
                gr = (sample_is_white == white_won) ? 1 : -1;
            }
            samples[i].game_result = gr;
        }
    }

    *out_plies = ply;
    return result;
}

/* ═══════════════════════════════════════════════════════════════════
 * Worker thread — pulls GameTasks from the shared atomic index,
 * lazily creates/caches one PlayerInstance per PlayerDef it needs.
 * ═══════════════════════════════════════════════════════════════════ */
static atomic_int   g_next_task;
static atomic_llong g_games_done;
static TaskList     g_tasks;
static const Config *g_cfg;
static const OpeningPool *g_openings;
static size_t        g_tt_entries;

/* ── PGN OUTPUT (--pgn) ──────────────────────────────────────────
 * A single shared FILE*, mutex-guarded, one fwrite+fflush per FINISHED
 * GAME (never per move) — same "accumulate a whole game, then take
 * the lock once" discipline tools/selfplay.c's sample-writing already
 * uses, for the same reason (the lock is essentially uncontended at
 * game granularity; per-move locking would not be). Opened fresh
 * ("w", not append) at the start of arena_run() if cfg->pgn_path[0],
 * so each CLI invocation produces its own complete PGN file — same
 * "overwrite, don't accumulate across unrelated runs" behavior as
 * --json. g_pgn_round is a simple per-process game counter used for
 * the PGN [Round] tag (cosmetic only). */
static FILE            *g_pgn_out = NULL;
/* --bin: packed training samples, same format tools/selfplay.c writes
 * (sample.h / train/dataset.py SAMPLE_DTYPE). Append mode, like
 * selfplay.c, so several arena runs can accumulate into one dataset. */
static FILE            *g_bin_out = NULL;
/* --epd: FEN + c0/c1/c2/c3 opcode rows, same textual convention as
 * tools/selfplay.c's --epd (see write_epd_rows()'s header below for the
 * one difference: c2 here is per-row, not a constant). Append mode, like
 * --bin, so several arena runs accumulate into one dataset. */
static FILE            *g_epd_out = NULL;
static pthread_mutex_t  g_bin_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t  g_pgn_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t  g_epd_mutex = PTHREAD_MUTEX_INITIALIZER;
static atomic_int        g_pgn_round;

static const char *ARENA_STARTPOS_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

static void write_pgn_game(const Config *cfg, const GameTask *t, Outcome o,
                            const char *start_fen, const char *movetext) {
    if (!g_pgn_out) return;
    const char *white_name = t->a_white ? cfg->players[t->pairA].name : cfg->players[t->pairB].name;
    const char *black_name = t->a_white ? cfg->players[t->pairB].name : cfg->players[t->pairA].name;
    const char *result = o == OUT_WHITE ? "1-0" : o == OUT_BLACK ? "0-1" : "1/2-1/2";
    int is_startpos = !strcmp(start_fen, ARENA_STARTPOS_FEN);
    int round = atomic_fetch_add(&g_pgn_round, 1) + 1;

    char block[8192];
    int n = snprintf(block, sizeof(block),
        "[Event \"Zchezz native arena\"]\n[Site \"?\"]\n[Date \"????.??.??\"]\n[Round \"%d\"]\n"
        "[White \"%s\"]\n[Black \"%s\"]\n[Result \"%s\"]\n",
        round, white_name, black_name, result);
    if (!is_startpos && n > 0 && (size_t)n < sizeof(block))
        n += snprintf(block + n, sizeof(block) - (size_t)n, "[SetUp \"1\"]\n[FEN \"%s\"]\n", start_fen);
    if (n > 0 && (size_t)n < sizeof(block))
        n += snprintf(block + n, sizeof(block) - (size_t)n, "\n%s%s\n\n", movetext ? movetext : "", result);
    if (n <= 0) return;
    if ((size_t)n > sizeof(block)) n = (int)sizeof(block);   /* truncated by snprintf — still a valid, if short, PGN block */

    pthread_mutex_lock(&g_pgn_mutex);
    fwrite(block, 1, (size_t)n, g_pgn_out);
    fflush(g_pgn_out);
    pthread_mutex_unlock(&g_pgn_mutex);
}

/* Writes one finished game's worth of EPD rows to g_epd_out, one
 * fwrite+fflush per GAME (accumulate-then-flush, same discipline as
 * write_pgn_game() above and tools/selfplay.c's --epd). Convention:
 * "<fen> c0 \"<result>\"; c1 \"<white-cp>\"; c2 \"<evaluator>\"; c3 \"<row index>\";"
 * — byte-for-byte the same opcode SET tools/selfplay.c and
 * tests/run_selfplay.py use, EXCEPT c2: those two always write a
 * constant ("Zchezz-native"/"book") because self-play has exactly one
 * evaluator per game. arena games can be net: vs uci:, or net-A vs
 * net-B, so c2 here is `row->evaluator` — the ACTUAL mover's player
 * name captured per-row in play_one_game() (see EpdRow's header
 * comment). Downstream consumers that want a same-evaluator-only subset
 * can filter on c2 exactly as they would filter selfplay data on c2's
 * "book" vs real-eval distinction. */
static void write_epd_rows(const EpdRow *rows, size_t n_rows, Outcome o, char *scratch, size_t scratch_cap) {
    if (!g_epd_out || n_rows == 0) return;
    const char *res_str = o == OUT_WHITE ? "1-0" : o == OUT_BLACK ? "0-1" : "1/2-1/2";
    size_t len = 0;
    for (size_t i = 0; i < n_rows; i++) {
        int wrote = snprintf(scratch + len, scratch_cap - len,
            "%s c0 \"%s\"; c1 \"%d\"; c2 \"%s\"; c3 \"%zu\";\n",
            rows[i].fen, res_str, rows[i].cp_white, rows[i].evaluator, i);
        if (wrote < 0 || len + (size_t)wrote >= scratch_cap) break;   /* truncate, don't overflow */
        len += (size_t)wrote;
    }
    if (len == 0) return;
    pthread_mutex_lock(&g_epd_mutex);
    fwrite(scratch, 1, len, g_epd_out);
    fflush(g_epd_out);
    pthread_mutex_unlock(&g_epd_mutex);
}

typedef struct {
    int id;
    PlayerInstance *inst[MAX_PLAYERS];   /* lazily created, indexed by player id */
} Worker;

static PlayerInstance *worker_get_instance(Worker *w, int player_idx) {
    if (!w->inst[player_idx])
        w->inst[player_idx] = player_instance_create(&g_cfg->players[player_idx], g_tt_entries);
    return w->inst[player_idx];
}

static void *worker_fn(void *arg) {
    Worker *w = (Worker *)arg;
    char pgn_movetext[4096];
    /* Per-worker --bin sample buffer, one game's worth. Allocated once,
     * not per game — max_plies is the hard upper bound on samples. */
    /* ONE undo stack per WORKER, reused across its games. Sized for the
     * whole game plus the deepest search that can sit on top of it:
     * max_plies game moves are never unmade, and a search pushes up to
     * MAX_PLY more. STACK_SIZE (512) is the search-only bound and is NOT
     * enough for a long game — that mismatch is what the old code hid by
     * resetting undo_top every ply. */
    const size_t game_undo_cap = (size_t)g_cfg->max_plies + MAX_PLY + 64;
    UndoFrame *game_undo = (UndoFrame *)malloc(game_undo_cap * sizeof(UndoFrame));
    if (!game_undo) { fprintf(stderr, "[arena] OOM allocating game undo stack\n"); exit(1); }
    int game_undo_top = 0;

    SelfplaySample *bin_buf = NULL;
    if (g_bin_out) {
        bin_buf = (SelfplaySample *)malloc((size_t)g_cfg->max_plies * sizeof(SelfplaySample));
        if (!bin_buf) { fprintf(stderr, "[arena] OOM allocating --bin buffer\n"); exit(1); }
    }
    /* Per-worker --epd row buffer + text scratch, one game's worth each.
     * Mirrors bin_buf above; sized the same way (max_plies is the hard
     * upper bound on rows/samples per game either way). */
    EpdRow *epd_buf = NULL;
    char   *epd_scratch = NULL;
    size_t  epd_scratch_cap = 0;
    if (g_epd_out) {
        epd_buf = (EpdRow *)malloc((size_t)g_cfg->max_plies * sizeof(EpdRow));
        if (!epd_buf) { fprintf(stderr, "[arena] OOM allocating --epd row buffer\n"); exit(1); }
        epd_scratch_cap = (size_t)g_cfg->max_plies * 256;   /* ~fen+opcodes per row, generous */
        epd_scratch = (char *)malloc(epd_scratch_cap);
        if (!epd_scratch) { fprintf(stderr, "[arena] OOM allocating --epd text scratch\n"); exit(1); }
    }
    for (;;) {
        int idx = atomic_fetch_add(&g_next_task, 1);
        if (idx >= g_tasks.n) break;
        GameTask *t = &g_tasks.tasks[idx];

        PlayerInstance *ia = worker_get_instance(w, t->pairA);
        PlayerInstance *ib = worker_get_instance(w, t->pairB);

        const char *fen = (t->opening_idx >= 0 && g_openings->n > 0)
            ? g_openings->fens[t->opening_idx]
            : ARENA_STARTPOS_FEN;

        long long plies = 0;
        size_t n_samp = 0;
        size_t n_epd = 0;
        Outcome o = play_one_game(ia, ib, t->a_white, fen, g_cfg, &plies,
                                   g_pgn_out ? pgn_movetext : NULL,
                                   g_pgn_out ? sizeof(pgn_movetext) : 0,
                                   bin_buf,
                                   bin_buf ? (size_t)g_cfg->max_plies : 0,
                                   bin_buf ? &n_samp : NULL,
                                   epd_buf,
                                   epd_buf ? (size_t)g_cfg->max_plies : 0,
                                   epd_buf ? &n_epd : NULL,
                                   game_undo, &game_undo_top);
        record_result(t->pairA, t->pairB, t->a_white, (int)o);
        if (g_pgn_out) write_pgn_game(g_cfg, t, o, fen, pgn_movetext);
        if (g_bin_out && n_samp) {
            pthread_mutex_lock(&g_bin_mutex);
            fwrite(bin_buf, sizeof(SelfplaySample), n_samp, g_bin_out);
            fflush(g_bin_out);
            pthread_mutex_unlock(&g_bin_mutex);
        }
        if (g_epd_out && n_epd) write_epd_rows(epd_buf, n_epd, o, epd_scratch, epd_scratch_cap);
        atomic_fetch_add(&g_games_done, 1);
    }
    free(bin_buf);
    free(epd_buf);
    free(epd_scratch);
    free(game_undo);
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════════
 * ArenaReport — the fully-computed, self-contained result of one
 * arena_run() call. print_report()/JSON emission below are pure
 * FORMATTERS over this struct — all the actual math (pairwise Elo,
 * anchor-relative absolute Elo, pooled field-performance Elo, SPRT)
 * happens once, inside arena_run(), so a programmatic caller (a
 * future GA/SPSA tuner, see header "EMBEDDABLE AS A LIBRARY") gets
 * every number it needs directly from this struct without re-deriving
 * anything or scraping stdout/JSON text.
 * ═══════════════════════════════════════════════════════════════════ */
typedef struct {
    long long w, d, l;                 /* pooled across every pairing this player took part in */
    double    pooled_elo, pooled_ci95; /* "field performance" rating — see header for the caveat
                                         * that this is NOT a joint whole-history MLE */
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

/* Pooled "vs the field" W/D/L for one player, across every pairing it
 * took part in — see header "MULTI-ENGINE MINI-TOURNAMENT / RANKING"
 * for why this is a first-order approximation, not a joint MLE. */
static void compute_pooled(const Config *cfg, int player_idx, long long *w, long long *d, long long *l) {
    *w = *d = *l = 0;
    for (int j = 0; j < cfg->n_players; j++) {
        if (j == player_idx) continue;
        *w += g_results[player_idx][j].w;
        *d += g_results[player_idx][j].d;
        *l += g_results[player_idx][j].l;
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * arena_run() — play exactly the match described by `cfg` (task list
 * derived from cfg + `openings`), then fill `*out` with every number
 * a caller could want. This is the function a future tuner calls in a
 * loop (see header "EMBEDDABLE AS A LIBRARY") — it does not touch
 * argv, stdout formatting, or JSON; main() below is just its first
 * caller. Resets all shared match state (g_results, task counters) at
 * entry, so repeated calls in one process are safe as long as they
 * are NOT run concurrently with each other (nothing in this codebase
 * needs that — a tuner calls arena_run() to completion, reads *out,
 * then calls it again for the next candidate).
 * ═══════════════════════════════════════════════════════════════════ */
void arena_run(const Config *cfg, const OpeningPool *openings, ArenaReport *out) {
    memset(out, 0, sizeof(*out));
    out->n_players = cfg->n_players;

    g_tt_entries = tt_entries_for_mb(cfg->tt_mb);
    g_tasks = build_task_list(cfg, openings->n);
    g_cfg = cfg;
    g_openings = openings;
    atomic_init(&g_next_task, 0);
    atomic_init(&g_games_done, 0);
    atomic_init(&g_pgn_round, 0);
    memset(g_results, 0, sizeof(g_results));

    /* Fresh PGN file per arena_run() call (overwrite, not append —
     * see write_pgn_game()'s header comment for why). NULL if --pgn
     * wasn't given: worker_fn()/write_pgn_game() both no-op on that,
     * and play_one_game() is called with pgn_movetext=NULL so it never
     * pays the SAN-generation cost either. */
    /* --bin refuses uci: players on purpose. eval_cp must come from the
     * search that CHOSE the move; arena never parses score lines from an
     * external engine, so a uci: mover would contribute eval_cp=0 and
     * silently poison the K-blend target for every one of its positions.
     * Failing loudly beats producing a subtly wrong dataset.
     *
     * --bin ALSO refuses a net-vs-DIFFERENT-net match (two net: players
     * with different weight paths). sample.h's SelfplaySample has no
     * evaluator-identity field (unlike EPD's free-text c2, see
     * write_epd_rows()'s header) — a mixed-net match would silently
     * blend two different evaluators' opinions into one eval_cp column,
     * exactly the mislabeled-training-data failure mode CLAUDE.md rule 10
     * warns about. --bin is therefore restricted to same-net (or single-
     * player self-play-shaped) matches; use --epd for anything mixed. */
    if (cfg->bin_path[0]) {
        for (int i = 0; i < cfg->n_players; i++) {
            if (cfg->players[i].kind != PLAYER_NET) {
                fprintf(stderr, "[arena] --bin requires every player to be net: "
                                "(player %d is uci:). eval_cp is only available from an\n"
                                "        in-process search; recording 0 for external engines would\n"
                                "        corrupt the training target. Drop --bin or use net: players.\n", i);
                exit(1);
            }
            if (i > 0 && strcmp(cfg->players[i].path, cfg->players[0].path) != 0) {
                fprintf(stderr, "[arena] --bin requires every net: player to use the SAME weight file\n"
                                "        (player 0: '%s', player %d: '%s'). Mixing evaluators would\n"
                                "        blend two different engines' opinions under one eval_cp label —\n"
                                "        sample.h has no per-row evaluator field to disambiguate them\n"
                                "        (EPD's c2 opcode does; use --epd for a mixed-net/uci match).\n",
                                cfg->players[0].path, i, cfg->players[i].path);
                exit(1);
            }
        }
        /* sample_open_bin_append() (sample.h) writes/verifies the file's
         * SampleFileHeader — engine build + NNUE weight-file fingerprint
         * for every eval_cp that follows. Every player here is already
         * net: on the SAME weight path (checked just above), so
         * cfg->players[0].path is the one true source for this run. This
         * is the provenance layer the --bin same-net gate's comment
         * above points at: it is what would eventually let that gate
         * relax from "same weight PATH string" to "same weight CONTENT"
         * — not done in this change, the gate stays exactly as strict. */
        g_bin_out = sample_open_bin_append(cfg->bin_path, SAMPLE_ENGINE_VERSION, cfg->players[0].path);
        if (!g_bin_out) {
            /* Hard failure, unlike --pgn's warn-and-continue: the caller
             * asked for training data, and finishing the match while
             * producing none is worse than not starting. */
            fprintf(stderr, "[arena] cannot open --bin file '%s'\n", cfg->bin_path);
            exit(1);
        }
    }

    g_pgn_out = cfg->pgn_path[0] ? fopen(cfg->pgn_path, "w") : NULL;
    if (cfg->pgn_path[0] && !g_pgn_out)
        fprintf(stderr, "[arena] WARNING: could not open --pgn output file '%s' — continuing without PGN output\n",
                cfg->pgn_path);

    /* --epd: no same-net/no-uci restriction — see EpdRow's header
     * comment and write_epd_rows() for why it's safe for any player mix.
     * Append mode (like --bin), warn-and-continue on open failure (like
     * --pgn) since EPD is a diagnostic/training convenience, not the
     * sole point of the run. */
    g_epd_out = cfg->epd_path[0] ? fopen(cfg->epd_path, "ab") : NULL;
    if (cfg->epd_path[0] && !g_epd_out)
        fprintf(stderr, "[arena] WARNING: could not open --epd output file '%s' — continuing without EPD output\n",
                cfg->epd_path);

    int n_threads = cfg->threads > 0 ? cfg->threads : 1;
    pthread_t *tids = (pthread_t *)malloc(n_threads * sizeof(pthread_t));
    Worker    *ws   = (Worker *)calloc((size_t)n_threads, sizeof(Worker));
    long long t0 = (long long)time(NULL);
    for (int i = 0; i < n_threads; i++) {
        ws[i].id = i;
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, 8 * 1024 * 1024);
        pthread_create(&tids[i], &attr, worker_fn, &ws[i]);
        pthread_attr_destroy(&attr);
    }

    long long last_done = -1;
    for (;;) {
        struct timespec ts = {0, 500 * 1000 * 1000};
        nanosleep(&ts, NULL);
        long long done = atomic_load(&g_games_done);
        if (done != last_done) {
            fprintf(stderr, "\r[arena] games %lld/%d   ", done, g_tasks.n);
            fflush(stderr);
            last_done = done;
        }
        if (done >= g_tasks.n) break;
    }
    fprintf(stderr, "\n");
    for (int i = 0; i < n_threads; i++) pthread_join(tids[i], NULL);
    out->elapsed_sec = (double)((long long)time(NULL) - t0);
    out->total_games = g_tasks.n;

    for (int i = 0; i < n_threads; i++)
        for (int j = 0; j < MAX_PLAYERS; j++)
            if (ws[i].inst[j]) player_instance_destroy(ws[i].inst[j]);
    free(ws);
    free(tids);
    free(g_tasks.tasks);
    g_tasks.tasks = NULL; g_tasks.n = 0; g_tasks.cap = 0;

    if (g_pgn_out) { fclose(g_pgn_out); g_pgn_out = NULL; }
    if (g_bin_out) { fclose(g_bin_out); g_bin_out = NULL; }
    if (g_epd_out) { fclose(g_epd_out); g_epd_out = NULL; }

    /* ── Fill the report ── */
    memcpy(out->pair, g_results, sizeof(g_results));
    for (int i = 0; i < cfg->n_players; i++) {
        long long w, d, l;
        compute_pooled(cfg, i, &w, &d, &l);
        out->rank[i].w = w; out->rank[i].d = d; out->rank[i].l = l;
        elo_diff_ci(w, d, l, &out->rank[i].pooled_elo, &out->rank[i].pooled_ci95);
    }

    if (cfg->sprt && cfg->n_players == 2) {
        out->sprt_applicable = 1;
        PairResult r = out->pair[0][1];
        out->sprt_verdict = sprt_llr(r.w, r.d, r.l, cfg->elo0, cfg->elo1, cfg->alpha, cfg->beta,
                                      &out->sprt_llr, &out->sprt_lo, &out->sprt_hi);
    }
}

/* One-time process-wide init (LMR/Zobrist/magic tables, dummy g_tt —
 * see comment at its call site). Call exactly once, before the first
 * arena_run(). Safe to call from a tuner's own main() too. */
void arena_global_init(void) {
    board_init();
    search_init();
    /* Same g_tt dummy-allocation trick as selfplay.c — this tool never
     * uses the process-wide default TTable (every SearchParams.tt is
     * set explicitly per PlayerInstance), so avoid wasting the ~104MB
     * native default allocation search_init() would otherwise make. */
    if (!g_tt) g_tt = tt_create(TT_BUCKETS * 512);
}

/* ═══════════════════════════════════════════════════════════════════
 * CLI
 * ═══════════════════════════════════════════════════════════════════ */
/* Every "(default ...)" below is pulled from the SAME ARENA_DEFAULT_*
 * constant config_defaults() reads (via ARENA_XSTR()) — see the
 * CONFIGURATION block at the top of this file. This text cannot say
 * anything different from what the program actually does. */
static void print_usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s --player net:PATH.nnu4 --player uci:PATH.exe [options]\n"
        "  --player SPEC          net:<weights.nnu4> or uci:<exe path> (repeat, 2..%d players).\n"
        "                         Omit entirely to use the ARENA_DEFAULT_PLAYERS config\n"
        "                         block at the top of this file instead (edit-the-file\n"
        "                         workflow); any --player on the CLI overrides it.\n"
        "                         No known-strength anchor mode here — that's\n"
        "                         tests/run_tournament.py's job, see arena.c header.\n"
        "  --games N              games per pairing if >2 players, else total (default " ARENA_XSTR(ARENA_DEFAULT_GAMES) ")\n"
        "  --threads N            concurrent games (default = logical cores; 0 also means that)\n"
        "  --movetime MS          per-move budget in ms (default " ARENA_XSTR(ARENA_DEFAULT_MOVETIME_MS) "; 0 = use --nodes)\n"
        "  --nodes N              per-move node budget (used when --movetime 0; default " ARENA_XSTR(ARENA_DEFAULT_NODES) ")\n"
        "  --max-plies N          game-length safety cap, counted as draw (default " ARENA_XSTR(ARENA_DEFAULT_MAX_PLIES) ")\n"
        "  --openings FILE        .epd/.fen or .pgn opening file (default: startpos only)\n"
        "  --opening-plies N      PGN opening-file replay depth per opening (default " ARENA_XSTR(ARENA_DEFAULT_OPENING_PLIES) ")\n"
        "  --seed N               RNG seed, opening cycling order (default " ARENA_XSTR(ARENA_DEFAULT_SEED) ")\n"
        "  --tt-mb MB             per-instance TTable budget in MB (default " ARENA_XSTR(ARENA_DEFAULT_TT_MB) ")\n"
        "  --gauntlet PLAYER      PLAYER (0-based index or name/path substring) vs every\n"
        "                         other player, instead of a full round-robin (default: off)\n"
        "  --sprt                 enable SPRT verdict, 2-player matches only (default: off)\n"
        "  --elo0 X --elo1 Y      SPRT null/alt hypothesis Elo (default " ARENA_XSTR(ARENA_DEFAULT_ELO0) " / " ARENA_XSTR(ARENA_DEFAULT_ELO1) ")\n"
        "  --alpha A --beta B     SPRT error rates (default " ARENA_XSTR(ARENA_DEFAULT_ALPHA) " / " ARENA_XSTR(ARENA_DEFAULT_BETA) ")\n"
        "  --uci-timeout MS       max wait for a uci: player's bestmove (default " ARENA_XSTR(ARENA_DEFAULT_UCI_TIMEOUT_MS) ")\n"
        "  --json FILE            write machine-readable results to FILE (default: no JSON output)\n"
        "  --pgn FILE             write all games as standard PGN to FILE, GUI-loadable\n"
        "                         (default: no PGN output — this is the ONLY output that\n"
        "                         costs extra CPU: SAN + check/mate detection per move)\n"
        "  --bin FILE             append packed training samples (sample.h SelfplaySample /\n"
        "                         train/dataset.py SAMPLE_DTYPE) to FILE. REQUIRES every\n"
        "                         --player to be net: AND share the SAME weight file — a\n"
        "                         mixed evaluator match would blend two engines' opinions\n"
        "                         into one unlabeled eval_cp column (default: no --bin output)\n"
        "  --epd FILE             append FEN + c0/c1/c2/c3 opcode rows to FILE (same textual\n"
        "                         convention as tools/selfplay.c --epd). No same-net/no-uci\n"
        "                         restriction — c2 records the ACTUAL mover's player name per\n"
        "                         row, so mixed net:/uci: or net-A-vs-net-B matches are fine\n"
        "                         (default: no --epd output). --bin/--pgn/--epd are additive,\n"
        "                         combine freely — never either/or\n",
        argv0, MAX_PLAYERS);
}

/* Parses "net:<path>" / "uci:<path>". No anchor/'@elo' syntax — see
 * header "NO STOCKFISH-ANCHORED ABSOLUTE ELO HERE". */
static int parse_player_spec(const char *spec, PlayerDef *out, int idx) {
    memset(out, 0, sizeof(*out));
    PlayerKind kind;
    const char *rest;
    if (!strncmp(spec, "net:", 4)) { kind = PLAYER_NET; rest = spec + 4; }
    else if (!strncmp(spec, "uci:", 4)) { kind = PLAYER_UCI; rest = spec + 4; }
    else {
        fprintf(stderr, "[arena] player spec must start with 'net:' or 'uci:': %s\n", spec);
        return -1;
    }
    out->kind = kind;
    strncpy(out->path, rest, sizeof(out->path) - 1);
    snprintf(out->name, sizeof(out->name), "P%d:%s", idx, basename_of(out->path));
    return 0;
}

static int parse_args(int argc, char **argv, Config *cfg) {
    config_defaults(cfg);
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        #define NEED_ARG() if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a); return -1; }
        if (!strcmp(a, "--player")) {
            NEED_ARG();
            if (cfg->n_players >= MAX_PLAYERS) { fprintf(stderr, "too many --player entries\n"); return -1; }
            if (parse_player_spec(argv[++i], &cfg->players[cfg->n_players], cfg->n_players) != 0) return -1;
            cfg->n_players++;
        }
        else if (!strcmp(a, "--games"))         { NEED_ARG(); cfg->games = atoi(argv[++i]); }
        else if (!strcmp(a, "--threads"))       { NEED_ARG(); cfg->threads = atoi(argv[++i]); }
        else if (!strcmp(a, "--movetime"))      { NEED_ARG(); cfg->movetime_ms = atoi(argv[++i]); }
        else if (!strcmp(a, "--nodes"))         { NEED_ARG(); cfg->nodes = atol(argv[++i]); }
        else if (!strcmp(a, "--max-plies"))     { NEED_ARG(); cfg->max_plies = atoi(argv[++i]); }
        else if (!strcmp(a, "--openings"))      { NEED_ARG(); strncpy(cfg->openings_path, argv[++i], sizeof(cfg->openings_path) - 1); }
        else if (!strcmp(a, "--opening-plies")) { NEED_ARG(); cfg->opening_plies = atoi(argv[++i]); }
        else if (!strcmp(a, "--seed"))          { NEED_ARG(); cfg->seed = (uint64_t)strtoull(argv[++i], NULL, 10); }
        else if (!strcmp(a, "--tt-mb"))         { NEED_ARG(); cfg->tt_mb = atof(argv[++i]); }
        else if (!strcmp(a, "--gauntlet"))      { NEED_ARG(); cfg->gauntlet = 1; strncpy(cfg->gauntlet_spec, argv[++i], sizeof(cfg->gauntlet_spec) - 1); }
        else if (!strcmp(a, "--sprt"))          { cfg->sprt = 1; }
        else if (!strcmp(a, "--elo0"))          { NEED_ARG(); cfg->elo0 = atof(argv[++i]); }
        else if (!strcmp(a, "--elo1"))          { NEED_ARG(); cfg->elo1 = atof(argv[++i]); }
        else if (!strcmp(a, "--alpha"))         { NEED_ARG(); cfg->alpha = atof(argv[++i]); }
        else if (!strcmp(a, "--beta"))          { NEED_ARG(); cfg->beta = atof(argv[++i]); }
        else if (!strcmp(a, "--uci-timeout"))   { NEED_ARG(); cfg->uci_move_timeout_ms = atoi(argv[++i]); }
        else if (!strcmp(a, "--json"))          { NEED_ARG(); strncpy(cfg->json_path, argv[++i], sizeof(cfg->json_path) - 1); }
        else if (!strcmp(a, "--pgn"))           { NEED_ARG(); strncpy(cfg->pgn_path, argv[++i], sizeof(cfg->pgn_path) - 1); }
        else if (!strcmp(a, "--bin"))           { NEED_ARG(); strncpy(cfg->bin_path, argv[++i], sizeof(cfg->bin_path) - 1); }
        else if (!strcmp(a, "--epd"))           { NEED_ARG(); strncpy(cfg->epd_path, argv[++i], sizeof(cfg->epd_path) - 1); }
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { print_usage(argv[0]); exit(0); }
        else { fprintf(stderr, "unknown argument: %s\n", a); print_usage(argv[0]); return -1; }
        #undef NEED_ARG
    }
    /* No --player given on the CLI at all -> fall back to the
     * ARENA_DEFAULT_PLAYERS config block (see that block's comment).
     * A CLI --player (even just one) always overrides this entirely. */
    if (cfg->n_players == 0) {
        for (int i = 0; i < ARENA_DEFAULT_PLAYERS_COUNT && cfg->n_players < MAX_PLAYERS; i++) {
            const ArenaDefaultPlayer *dp = &ARENA_DEFAULT_PLAYERS[i];
            char spec[600];
            snprintf(spec, sizeof(spec), "%s:%s", dp->kind, dp->path);
            if (parse_player_spec(spec, &cfg->players[cfg->n_players], cfg->n_players) != 0) return -1;
            if (dp->name[0]) strncpy(cfg->players[cfg->n_players].name, dp->name,
                                      sizeof(cfg->players[cfg->n_players].name) - 1);
            cfg->n_players++;
        }
        if (cfg->n_players > 0)
            fprintf(stderr, "[arena] no --player given -- using %d player(s) from the "
                            "ARENA_DEFAULT_PLAYERS config block (see arena.c)\n", cfg->n_players);
    }
    if (cfg->n_players < 2) { fprintf(stderr, "error: need at least 2 --player entries\n"); print_usage(argv[0]); return -1; }
    if (cfg->games <= 0) { fprintf(stderr, "error: --games must be > 0\n"); return -1; }
    /* 0 means autodetect, matching ARENA_DEFAULT_THREADS's comment and
     * selfplay.c's identical rule — see that file for why clamping to 1
     * here is a silent single-threaded run rather than a safe default. */
    if (cfg->threads <= 0) cfg->threads = detect_cpus();
    if (cfg->max_plies < 1) cfg->max_plies = 1;
    if (cfg->sprt && cfg->n_players != 2) {
        fprintf(stderr, "error: --sprt requires exactly 2 --player entries\n");
        return -1;
    }
    if (resolve_gauntlet(cfg) != 0) return -1;
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * Reporting — pure formatters over an already-computed ArenaReport
 * (see arena_run() above for where the numbers come from).
 * ═══════════════════════════════════════════════════════════════════ */
static void print_cross_table(const Config *cfg, const ArenaReport *rep) {
    printf("[arena] cross-table (row's result vs column, W-D-L):\n");
    printf("[arena] %-28s", "");
    for (int j = 0; j < cfg->n_players; j++) printf(" %14.14s", cfg->players[j].name);
    printf("\n");
    for (int i = 0; i < cfg->n_players; i++) {
        printf("[arena] %-28s", cfg->players[i].name);
        for (int j = 0; j < cfg->n_players; j++) {
            if (i == j) { printf(" %14s", "--"); continue; }
            PairResult r = rep->pair[i][j];
            if (r.w + r.d + r.l == 0) { printf(" %14s", "."); continue; }
            char cell[16]; snprintf(cell, sizeof(cell), "%lld-%lld-%lld", r.w, r.d, r.l);
            printf(" %14s", cell);
        }
        printf("\n");
    }
}

/* Ranking sorted by pooled_elo descending — a scratch index array is
 * sorted rather than reordering `rep->rank` itself, so the report
 * struct's player-id indexing stays stable for programmatic callers. */
static void print_ranking(const Config *cfg, const ArenaReport *rep) {
    int order[MAX_PLAYERS];
    for (int i = 0; i < cfg->n_players; i++) order[i] = i;
    for (int i = 0; i < cfg->n_players; i++)
        for (int j = i + 1; j < cfg->n_players; j++)
            if (rep->rank[order[j]].pooled_elo > rep->rank[order[i]].pooled_elo) {
                int t = order[i]; order[i] = order[j]; order[j] = t;
            }
    printf("[arena] ranking (pooled field-performance Elo — see header comment "
           "'MULTI-ENGINE MINI-TOURNAMENT / RANKING' for what this is and isn't):\n");
    for (int k = 0; k < cfg->n_players; k++) {
        int i = order[k];
        const PlayerRankRow *rr = &rep->rank[i];
        printf("[arena]   #%d %-24s +%-4lld =%-4lld -%-4lld  pooled Elo: %+7.1f +/- %5.1f\n",
            k + 1, cfg->players[i].name, rr->w, rr->d, rr->l, rr->pooled_elo, rr->pooled_ci95);
    }
}

/* Format a double for JSON output.
 *
 * WHY THIS EXISTS: elo_diff_ci() legitimately returns +/-infinity for a
 * shutout (a 4-0 pairing has no finite Elo estimate) and the CI is
 * infinite with it.  Printing those with "%.2f" emits the bare tokens
 * `inf` / `-inf` / `nan`, which are NOT valid JSON — every standard
 * parser rejects the whole file.  Since --json is what the bootstrap
 * loop (Appendix F.5) reads to decide whether to promote a generation, a
 * shutout — the most decisive result there is — would break exactly the
 * automation it is meant to drive.
 *
 * JSON has no infinity literal, so non-finite values become `null`;
 * consumers must treat null as "undefined / off the scale", which is
 * what it means. Returns a pointer to `buf` for use inline in fprintf. */
static const char *json_num(char *buf, size_t cap, double v) {
    if (!isfinite(v)) { snprintf(buf, cap, "null"); return buf; }
    snprintf(buf, cap, "%.2f", v);
    return buf;
}

static void print_report(const Config *cfg, const ArenaReport *rep, FILE *json_out) {
    printf("\n[arena] ══════════════════════════════════════════════ RESULTS "
           "(%d games, %.0fs)\n", rep->total_games, rep->elapsed_sec);

    if (json_out) {
        fprintf(json_out, "{\"rating_method\":\"pairwise trinomial Elo (elo_calc.py-compatible) + "
                "pooled vs-field performance rating (NOT a joint whole-history MLE); "
                "no absolute/anchor Elo here — see tests/run_tournament.py for that\",");
        fprintf(json_out, "\"total_games\":%d,\"elapsed_sec\":%.1f,", rep->total_games, rep->elapsed_sec);
        fprintf(json_out, "\"players\":[");
        for (int i = 0; i < cfg->n_players; i++)
            fprintf(json_out, "%s{\"id\":%d,\"kind\":\"%s\",\"path\":\"%s\",\"name\":\"%s\"}",
                i ? "," : "", i, cfg->players[i].kind == PLAYER_NET ? "net" : "uci",
                cfg->players[i].path, cfg->players[i].name);
        fprintf(json_out, "]");
    }

    if (json_out) fprintf(json_out, ",\"pairings\":[");
    int first_pair = 1;
    for (int i = 0; i < cfg->n_players; i++) {
        for (int j = i + 1; j < cfg->n_players; j++) {
            PairResult r = rep->pair[i][j];
            long long n = r.w + r.d + r.l;
            if (n == 0) continue;
            double elo, ci;
            elo_diff_ci(r.w, r.d, r.l, &elo, &ci);
            printf("[arena] %-24s vs %-24s : +%lld =%lld -%lld (%lld games)  Elo: %+.1f +/- %.1f\n",
                cfg->players[i].name, cfg->players[j].name, r.w, r.d, r.l, n, elo, ci);
            if (json_out) {
                char b1[32], b2[32];
                fprintf(json_out, "%s{\"a\":%d,\"b\":%d,\"w\":%lld,\"d\":%lld,\"l\":%lld,\"elo\":%s,\"ci95\":%s}",
                    first_pair ? "" : ",", i, j, r.w, r.d, r.l,
                    json_num(b1, sizeof b1, elo), json_num(b2, sizeof b2, ci));
                first_pair = 0;
            }
        }
    }
    if (json_out) fprintf(json_out, "]");

    if (cfg->n_players > 2) {
        printf("\n");
        print_cross_table(cfg, rep);
    }
    printf("\n");
    print_ranking(cfg, rep);
    if (json_out) {
        fprintf(json_out, ",\"ranking\":[");
        for (int i = 0; i < cfg->n_players; i++) {
            const PlayerRankRow *rr = &rep->rank[i];
            char b1[32], b2[32];
            fprintf(json_out, "%s{\"id\":%d,\"w\":%lld,\"d\":%lld,\"l\":%lld,"
                "\"pooled_elo\":%s,\"pooled_ci95\":%s}",
                i ? "," : "", i, rr->w, rr->d, rr->l,
                json_num(b1, sizeof b1, rr->pooled_elo), json_num(b2, sizeof b2, rr->pooled_ci95));
        }
        fprintf(json_out, "]");
    }

    if (rep->sprt_applicable) {
        const char *vstr = rep->sprt_verdict == SPRT_ACCEPT_H1 ? "ACCEPT_H1 (promote)"
                          : rep->sprt_verdict == SPRT_ACCEPT_H0 ? "ACCEPT_H0 (discard)"
                          : "CONTINUE (inconclusive at this sample size)";
        printf("\n[arena] SPRT elo0=%.1f elo1=%.1f alpha=%.3f beta=%.3f : LLR=%.3f  bounds=[%.3f, %.3f]  verdict=%s\n",
            cfg->elo0, cfg->elo1, cfg->alpha, cfg->beta, rep->sprt_llr, rep->sprt_lo, rep->sprt_hi, vstr);
        char bl[32];
        if (json_out)
            fprintf(json_out, ",\"sprt\":{\"elo0\":%.2f,\"elo1\":%.2f,\"alpha\":%.3f,\"beta\":%.3f,"
                "\"llr\":%s,\"lo\":%.4f,\"hi\":%.4f,\"verdict\":\"%s\"}",
                cfg->elo0, cfg->elo1, cfg->alpha, cfg->beta, json_num(bl, sizeof bl, rep->sprt_llr), rep->sprt_lo, rep->sprt_hi,
                rep->sprt_verdict == SPRT_ACCEPT_H1 ? "ACCEPT_H1" : rep->sprt_verdict == SPRT_ACCEPT_H0 ? "ACCEPT_H0" : "CONTINUE");
    }
    if (json_out) fprintf(json_out, "}\n");
}

#ifndef ARENA_NO_MAIN
/* ═══════════════════════════════════════════════════════════════════
 * main — CLI entry point. Everything reusable lives in arena_run()/
 * arena_global_init() above; main() is just argv parsing, one-time
 * resource loading (NNUE nets, openings), and output formatting — see
 * header "EMBEDDABLE AS A LIBRARY" for how a future tuner reuses the
 * former without the latter (compile with -DARENA_NO_MAIN).
 * ═══════════════════════════════════════════════════════════════════ */
int main(int argc, char **argv) {
    Config cfg;
    if (parse_args(argc, argv, &cfg) != 0) return 1;

    arena_global_init();

    for (int i = 0; i < cfg.n_players; i++) {
        if (cfg.players[i].kind == PLAYER_NET) {
            cfg.players[i].net = nnue_net_load(cfg.players[i].path);
            if (!cfg.players[i].net || !nnue_net_ready(cfg.players[i].net)) {
                fprintf(stderr, "[arena] failed to load NNUE weights '%s'\n", cfg.players[i].path);
                return 1;
            }
            /* *** REQUIRED, NON-OBVIOUS ***: eval_stm() (search.c) does
             * NOT consult the per-instance net bound to the Board it was
             * given — it gates purely on the PROCESS-GLOBAL nnue_ready()
             * flag, i.e. nnue_net_ready(g_nnue_net), which only the
             * legacy nnue_load()/nnue_load_from_mem() ever set. This
             * tool deliberately never calls that legacy loader (it would
             * imply a single process-wide net, defeating the entire
             * point of per-player NnueNet — see header "TWO PLAYER
             * KINDS"), so without this line g_nnue_net stays NULL,
             * nnue_ready() stays false FOREVER, and eval_stm() silently
             * returns 0 for every net: player's every search — found by
             * running an actual net: vs net: match during verification
             * (search.c prints "[NNUE] eval called but weights not
             * loaded" exactly once, then keeps returning 0 silently).
             * The fix does NOT make every player share one net: it only
             * flips the boolean gate true; nnue_eval_bb() below always
             * reads the per-instance na->net (bound in
             * player_instance_create()), so evaluation quality is still
             * fully per-player-correct — this just satisfies the gate. */
            if (!g_nnue_net) g_nnue_net = cfg.players[i].net;
        }
    }

    OpeningPool openings; memset(&openings, 0, sizeof(openings));
    if (cfg.openings_path[0]) {
        if (load_openings(cfg.openings_path, &openings, cfg.opening_plies) != 0) {
            fprintf(stderr, "[arena] failed to load openings from '%s'\n", cfg.openings_path);
            return 1;
        }
        fprintf(stderr, "[arena] %d opening position(s) loaded from %s\n", openings.n, cfg.openings_path);
    } else {
        fprintf(stderr,
            "[arena] WARNING: no --openings given — every game starts from the "
            "standard position. Antithetic color-swap pairing still cancels "
            "first-move advantage, but variance across game PAIRS will be much "
            "higher than with a real opening book. Strongly recommended: "
            "--openings openings/lines/8moves_v3.pgn (or any file under openings/lines/).\n");
    }

    fprintf(stderr,
        "[arena] players=%d games/pairing=%d threads=%d movetime=%dms nodes=%ld "
        "max_plies=%d tt-mb=%.1f seed=%llu%s%s\n",
        cfg.n_players, cfg.games, cfg.threads, cfg.movetime_ms, cfg.nodes,
        cfg.max_plies, cfg.tt_mb, (unsigned long long)cfg.seed,
        cfg.sprt ? " sprt=on" : "", cfg.gauntlet ? " gauntlet=on" : "");
    for (int i = 0; i < cfg.n_players; i++)
        fprintf(stderr, "[arena]   %s = %s:%s\n", cfg.players[i].name,
            cfg.players[i].kind == PLAYER_NET ? "net" : "uci", cfg.players[i].path);
    if (cfg.gauntlet)
        fprintf(stderr, "[arena]   gauntlet candidate: %s\n", cfg.players[cfg.gauntlet_idx].name);

    ArenaReport report;
    arena_run(&cfg, &openings, &report);
    fprintf(stderr, "[arena] DONE %d games in %.0fs\n", report.total_games, report.elapsed_sec);

    FILE *json_out = cfg.json_path[0] ? fopen(cfg.json_path, "w") : NULL;
    print_report(&cfg, &report, json_out);
    if (json_out) fclose(json_out);

    for (int i = 0; i < openings.n; i++) free(openings.fens[i]);
    free(openings.fens);
    for (int i = 0; i < cfg.n_players; i++)
        if (cfg.players[i].net) nnue_net_destroy(cfg.players[i].net);
    return 0;
}
#endif /* ARENA_NO_MAIN */
