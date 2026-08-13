/* tools/selfplay.c — Zchezz v4.00 native, in-process, multithreaded
 * self-play data generator (implementation plan Appendix F.2 / F.3.0 /
 * F.3 Estagio 1).
 *
 * ── BUILD ─────────────────────────────────────────────────────────────
 *   This file lives in engine/c/tools/ — SHARED across engine versions,
 *   tracking the CURRENT engine API only (see README.md's engine/c/tools/ file inventory).
 *   From engine/build/:  mingw32-make ENGINE=v400 selfplay
 *   (output: engine/build/selfplay.exe — a shared tool binary, not a
 *   per-version artifact. See engine/build/Makefile for ENGINE=... to
 *   target a different engine/c/zchezz_vXXX/ folder.)
 *   (or manually — mirrors main.c's own flags, minus main.c itself,
 *   plus -DNO_TABLEBASES -DNO_BOOK so syzygy.c/tbprobe.c/book.c don't
 *   need to be linked; their headers stub every call out to a no-op),
 *   from engine/build/:
 *
 *     gcc -O3 -ffast-math -D_GNU_SOURCE -std=c11 -mavxvnni -mavx2 \
 *         -I../c/zchezz_v400 -DNO_TABLEBASES -DNO_BOOK \
 *         -Wno-unused-variable -Wno-unused-but-set-variable \
 *         -Wno-maybe-uninitialized -Wno-misleading-indentation \
 *         -Wno-sign-compare -Wno-unused-function -Wno-parentheses \
 *         -o selfplay.exe ../c/tools/selfplay.c ../c/zchezz_v400/board.c \
 *         ../c/zchezz_v400/search.c ../c/zchezz_v400/nnue.c \
 *         -static -lm -pthread
 *
 * ── WHY THIS EXISTS (motivation, see Appendix F.2 of
 *    zchezz_v400_implementation_plan.md) ──────────────────────────────
 *
 *   The OLD path (tests/run_selfplay.py) spawns 16 zchezz.exe PROCESSES,
 *   drives each one over the UCI TEXT protocol (one "position ... / go
 *   movetime N" round-trip per move, parsing "bestmove ..." back), writes
 *   a full PGN per game, re-parses that PGN, hands every position to a
 *   SEPARATE Stockfish process for labelling, and finally serializes to
 *   Parquet via pandas.  Every one of those steps exists only because the
 *   old pipeline talks to the engine over a text protocol meant for GUIs,
 *   not because self-play data generation actually needs any of it.
 *
 *   This tool collapses the whole chain: N worker THREADS in ONE process,
 *   each calling search_best() directly (no subprocess, no UCI text, no
 *   PGN), each search's own MultiPV root scores decide BOTH the move
 *   played (Appendix F.3.0 temperature sampling) AND the eval_cp label
 *   (free — it's a number the search already computed, not a second
 *   Stockfish pass), and each finished game is packed straight into the
 *   75-byte SAMPLE_DTYPE record train/dataset.py already defines
 *   and written to a flat .bin file.  No PGN, no EPD, no Parquet, no
 *   Stockfish anywhere in this file.
 *
 * ── THREAD-SAFETY AUDIT (required before trusting N independent
 *    concurrent searches in one process — the engine was built for ONE
 *    search at a time with Lazy SMP HELPERS, not N fully independent
 *    games) ─────────────────────────────────────────────────────────────
 *
 *   Audited: search.c (SearchState, TT, alpha_beta/qsearch local state),
 *   board.c (Board, UndoFrame stack, board_load_fen), nnue.c (weight
 *   arrays, NnueAccum).  Findings:
 *
 *   1. SearchState (search.h) is ALREADY a clean per-instance struct —
 *      v4.00 removed the old global `ss` pointer race (see search.c's own
 *      comment on that fix).  Every mutable per-search field (killers,
 *      history, counters, tb_hits, excluded_root[]) lives inside it.  As
 *      long as every worker thread gets its OWN SearchState (via
 *      search_state_new(), exactly like main.c's helper_thread_fn does),
 *      there is no sharing here.  NOT A BUG — already correct, just had
 *      to be used correctly (see worker setup below).
 *
 *   2. TTable (search.h) is ALREADY a per-instance struct (v4.00) with
 *      its own tt_create()/tt_destroy()/tt_clear().  g_tt is only the
 *      *process-wide default* the UCI engine happens to use; nothing
 *      forces two SearchParams to point at the same TTable.  Each worker
 *      here gets its own TTable(s) via tt_create() (see "TT POLICY"
 *      below for the shared-vs-separate-per-color question, which is a
 *      correctness question about GAME semantics, not thread safety).
 *
 *   3. NnueAccum (nnue.h) — weight matrices (_nnL1WT etc. in nnue.c) are
 *      global but loaded ONCE via nnue_load() before any thread starts,
 *      then only ever READ.  Safe.  The *mutable* per-search accumulator
 *      state lives entirely in NnueAccum, which is NOT global — it's
 *      reached through Board::nnue, a per-Board pointer.  Each worker
 *      heap-allocates its own NnueAccum with zmalloc32 (32-byte aligned,
 *      required for the AVX2 code path), same as main.c's
 *      helper_thread_fn.  NOT A BUG — already correct, just had to be
 *      used correctly.
 *
 *   4. board.c's undo stack: g_undo[STACK_SIZE]/g_undo_top are ONE
 *      process-global array+index, but Board never touches them
 *      directly — it goes through Board::undo / Board::undo_top, two
 *      pointers that can be rebound to ANY buffer.  main.c's
 *      helper_thread_fn mallocs a private UndoFrame stack per helper and
 *      rebinds those two pointers.  Same pattern used here.  NOT A BUG.
 *
 *   5. *** THE ACTUAL BUG FOUND, AND WHY IT MATTERS HERE SPECIFICALLY ***
 *      board_load_fen() (board.c) does:
 *          memset(b, 0, sizeof(*b));
 *          ...
 *          board_bind_undo_global(b);   // b->undo = g_undo (GLOBAL!)
 *          board_bind_nnue_global(b);   // b->nnue = &g_nnue_accum (GLOBAL!)
 *      This is CORRECT for the UCI engine's single position command on
 *      the main thread (that's what "global" means there — there is only
 *      one live search at a time), and helper_thread_fn never calls
 *      board_load_fen at all, so the existing engine never hits this.
 *      But this tool calls board_load_fen() ONCE PER GAME, ON EVERY
 *      WORKER THREAD, to reset each worker's Board to the start position.
 *      If left unpatched, EVERY worker's Board::undo/Board::nnue would
 *      silently be rebound to the SAME g_undo[]/g_nnue_accum globals
 *      after every single board_load_fen() call — i.e. every worker
 *      thread racing on the same undo stack and the same NNUE
 *      accumulator, corrupting both immediately under any concurrency.
 *      This is exactly the class of bug the task asked to hunt for: a
 *      function that LOOKS like it only touches its own Board argument
 *      but actually reaches into process-global state.
 *      FIX (see worker_reset_board() below): after every
 *      board_load_fen() call, explicitly rebind
 *      b->undo/b->undo_top/b->nnue back to this worker's own heap
 *      allocations.  board_make/board_unmake never touch these pointers
 *      themselves (verified — only board_load_fen does), so one rebind
 *      immediately after each board_load_fen() call is sufficient.
 *
 *   6. Minor, deliberately NOT fixed: eval_stm() (search.c) has a
 *      `static int warned` guarding a one-time stderr message when NNUE
 *      weights aren't loaded.  Benign race (worst case: the message
 *      prints more than once under concurrency) — not a correctness bug
 *      for training-data output, and this tool always loads NNUE weights
 *      before starting workers anyway, so the path is dead here.
 *
 *   7. g_tb_probe_depth/g_tb_probe_limit (search.c) are process-global
 *      ints, but this tool never writes them (no UCI "setoption" path
 *      exists here) and compiles with -DNO_TABLEBASES anyway, which
 *      stubs every syzygy_* call to a no-op regardless of their values.
 *      Read-only in practice for this tool. Not a bug.
 *
 *   8. nnue_load()/board_init()/search_init() themselves are NOT
 *      thread-safe (they write global tables: weight arrays, magic
 *      bitboard tables, Zobrist tables, lmr_tab/MVV_LVA) — but they are
 *      each called EXACTLY ONCE, from main(), BEFORE any worker thread
 *      is created (see main() below).  After that point they are only
 *      ever read.  Not a bug, just a sequencing requirement.
 *
 *   Net result: one real bug (item 5), fixed by an explicit rebind after
 *   every board_load_fen() call.  Everything else in the engine was
 *   already correctly per-instance/per-thread; it just had never been
 *   exercised by N *independent* concurrent searches before (Lazy SMP
 *   helpers all search the SAME position, so board_load_fen() is never
 *   called more than once per position there).
 *
 * ── TT POLICY (Appendix F.2 — the "shared TT between colors" question,
 *    a correctness question about GAME semantics, distinct from the
 *    thread-safety audit above) ────────────────────────────────────────
 *
 *   Within ONE self-play game, White and Black are the SAME engine with
 *   the SAME weights — sharing one TTable between the two colors loses
 *   no correctness (no "opponent model" leaks the way it would in a real
 *   A/B arena between two DIFFERENT engines/versions), it halves TT
 *   memory per worker, and it lets one color's search reuse the other
 *   color's transpositions for a modest search speedup.  This is the
 *   DEFAULT (`--separate-tt` to opt out).
 *
 *   `tt_clear()` — a PHYSICAL memset, not `tt_new_generation()` (a
 *   logical counter bump) — is mandatory before EVERY game, even in
 *   shared mode, for a specific reason: repetition/draw scores stored in
 *   the TT are PATH-DEPENDENT (board_is_draw() counts how many times the
 *   current Zobrist hash appears in THIS game's b->hist[]), but a TT
 *   entry is indexed ONLY by Zobrist hash — it carries no memory of
 *   which game produced it.  If a position P was scored "draw by
 *   repetition" in game G1 (because it repeated three times in G1's
 *   specific move history) and the SAME hash P is reached in game G2
 *   WITHOUT repeating (different game, different history), a stale TT
 *   entry would hand back G1's draw score as if it were a real
 *   evaluation of G2's position — silently biasing G2's search (and,
 *   over many games, silently inflating this generator's draw rate).
 *   `tt_new_generation()` (a `gen` counter bump, see tt_probe() in
 *   search.c) does NOT fix this: it is designed to age out entries
 *   across UCI "ucinewgame" boundaries where only ONE game is ever live
 *   at a time. Within a single self-play game, BOTH colors write to the
 *   SAME TT at the SAME generation while the SAME game is in progress —
 *   there is no "previous generation" to distinguish from; the
 *   contaminated entry and the new game share one generation.  Only a
 *   physical wipe removes the stale path-dependent score.  (The existing
 *   UCI engine never hits this because tests/run_selfplay.py sends
 *   "ucinewgame" between games AND runs one game per PROCESS — a fresh
 *   process has a zeroed TT already; this tool reuses one process and
 *   one TTable across thousands of games per worker, so it has to do the
 *   wipe itself.)
 *
 * ── TEMPERATURE MOVE SELECTION (Appendix F.3.0 / F.3 Estagio 1) ────────
 *
 *   Instead of always playing best_move (argmax), sample among the top-N
 *   root moves using search_best()'s EXISTING MultiPV machinery
 *   (n_pvs = --multipv candidates) with a softmax over the STM-relative
 *   root scores.
 *
 *   SHARED MULTIPV TIME BUDGET (sp.mpv_share_budget = 1, set below):
 *   search_best()'s default MultiPV behavior gives EACH of the --multipv
 *   candidate lines its OWN full movetime_ms budget (so interactive
 *   analysis in the UCI/browser UI gets a full-depth search per line).
 *   That is the right behavior for a human staring at 4-5 analysis
 *   lines, but it is a straight throughput tax here: this generator only
 *   ever needs the ROOT SCORE of each candidate to feed the softmax
 *   above, not a full independent search per candidate, yet at
 *   --multipv 4 the naive per-line-budget behavior means one MOVE costs
 *   4x one ordinary search — and move-selection throughput is exactly
 *   what gates the whole self-play bootstrapping loop. Setting
 *   sp.mpv_share_budget = 1 makes search_best() divide movetime_ms by
 *   n_pvs instead, so all --multipv lines together cost about the same
 *   as ONE full-budget search (see search.c's search_best() for the
 *   design rationale and the guarantee that every line still completes
 *   at least depth 2 before its slice can run out, so the softmax above
 *   never sees a garbage/unsearched score). This does NOT affect the UCI
 *   engine's interactive MultiPV — that path leaves mpv_share_budget at
 *   its default 0 and is unchanged.
 *
 *   The formula:
 *
 *       P(move_i) = exp(score_i / T) / sum_j exp(score_j / T)
 *
 *   Raw centipawn scores fed straight into exp() saturate almost
 *   immediately (exp(300) overflows a double long before T gets
 *   interesting), so scores are first divided by --temp-scale (a
 *   centipawn constant, default 100 — i.e. "100cp of advantage is worth
 *   1.0 softmax-units before temperature is applied") and ONLY THEN
 *   divided by T:
 *
 *       adjusted_i = score_i / temp_scale
 *       P(move_i)  = exp(adjusted_i / T) / sum_j exp(adjusted_j / T)
 *
 *   with the usual max-subtraction for numerical stability. Two-phase
 *   schedule (AlphaZero-style step function, per the plan): T = T0
 *   (--temperature, default 1.0) for the first --temp-plies plies
 *   (default 24), then T = T1 (--temp-final, default 0.05 — a tiny but
 *   nonzero temperature, effectively-but-not-exactly argmax) for the
 *   rest of the game.  Every worker owns its own RNG (xorshift64*),
 *   reseeded PER GAME from a deterministic function of (--seed,
 *   game_index) — NOT of (--seed, thread_index) — because dynamic work
 *   stealing means which thread plays game #42 is nondeterministic
 *   across runs; only the game index is a stable identity.
 *
 * ── SAMPLE FORMAT (BINDING CONTRACT — see train/dataset.py's
 *    SAMPLE_DTYPE, which this struct must match byte-for-byte) ─────────
 *
 *   uint8  board[64]     mailbox pieces, Zchezz encoding, zsq 0=a8..63=h1
 *   uint8  stm            0=white, 1=black
 *   uint8  rule50
 *   uint8  castling       bitmask (CA_WK|CA_WQ|CA_BK|CA_BQ from board.h)
 *   uint8  ep_file        0..7, 8=none
 *   int16  eval_cp        STM-relative score from the search that CHOSE
 *                          the move recorded in this sample (free — it's
 *                          the winning candidate's own MultiPV score,
 *                          not a second eval pass)
 *   int8   game_result    +1/0/-1 from the perspective of the side to
 *                          move IN THIS POSITION (filled in a SECOND
 *                          pass over the game's samples once the real
 *                          result is known — see finish_game() below)
 *   uint16 move_played    packed move, LOW 16 BITS of search.c's own
 *                          pack_move() bit layout (see pack_move16()
 *                          below for why only 16 of its 20 bits fit and
 *                          what that costs)
 *   uint16 _pad           always zero
 *
 *   Total: 75 bytes/record, no implicit padding (matches dataset.py's
 *   `np.dtype(..., align=False)` — this file is compiled with
 *   `#pragma pack(push,1)` for the exact same reason).
 *
 * ── GAME TERMINATION ─────────────────────────────────────────────────
 *
 *   Checkmate/stalemate: detected via search_best()'s own root result —
 *   when NO legal move exists, search_best() returns best={0,0} (see
 *   search.c's iterative-deepening loop: "no legal move found" only
 *   leaves best_move.from==0 && best_move.to==0).  board_in_check()
 *   then distinguishes checkmate (opponent wins) from stalemate (draw).
 *   Insufficient material / 50-move / threefold repetition: all detected
 *   by the engine's OWN board_is_draw() (board.c) — NOT reimplemented
 *   here.  A hard --max-plies cap (default 400) treats any game that
 *   runs past it as a draw (matches the plan's default and zquoridor's
 *   analogous safety cutoff).
 *
 * ── WHAT WAS DUPLICATED VS FACTORED OUT ─────────────────────────────
 *
 *   main.c owns main() and therefore cannot be linked into this tool.
 *   The two small helpers this file needed from main.c (zmalloc32/
 *   zfree32 — 32-byte aligned alloc for NnueAccum, and the
 *   pthread_create staggered worker-pool pattern) are small enough that
 *   DUPLICATING them here (as main.c itself already duplicates zmalloc32
 *   from nnue.c — see main.c's own comment on that) was simpler and
 *   safer than factoring a new shared header out of a working, tested
 *   main.c on a v4.00 branch that hasn't shipped yet. `play_one_game()`
 *   IS written as a clean, reusable, config-driven function (see
 *   worker_fn() below) specifically because the task called out that a
 *   future native A/B arena will want the identical game-loop shape with
 *   a DIFFERENT TT policy (two independent engines => always
 *   --separate-tt-equivalent, never shared) — swapping that policy is a
 *   one-line change in an arena tool that reuses this file's
 *   play_one_game(), worker_reset_board(), and SelfplaySample machinery.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <pthread.h>
#include <stdatomic.h>

#include "board.h"
#include "search.h"
#include "nnue.h"
#include "opening_pool.h"   /* opening-book support (san_resolve/move_to_san/
                             * OpeningIndex) — shared with arena.c, see that
                             * header's comment for why this isn't a second
                             * copy of the same SAN parser. */

/* ═══════════════════════════════════════════════════════════════════
 * On-disk sample record — SEE HEADER COMMENT: THIS IS A CROSS-LANGUAGE
 * CONTRACT WITH train/dataset.py's SAMPLE_DTYPE. Do not reorder
 * or resize any field without updating that file's dtype in lockstep.
 * ═══════════════════════════════════════════════════════════════════ */
#include "sample.h"

/* ── Packed move, 16 bits (NOT search.c's own pack_move(), which needs
 * 20 bits: from[0:5] to[6:11] prom[12:14] epc[15] castle[16:19]).
 * SAMPLE_DTYPE's move_played field is uint16, so this duplicates ONLY
 * the low 16 bits of that exact same bit layout — from/to/prom/epc
 * round-trip exactly, but the 4 castle bits (which search.c's TT packing
 * needs 32 bits to keep) are dropped.  This is not a loss for consumers:
 * a castling move is always the king moving two files on its home rank
 * (from/to alone identify it unambiguously; a future policy-head reader
 * can special-case e1g1/e1c1/e8g8/e8c8 without the explicit flag if it
 * ever needs to distinguish "king move" from "castle" — this generator
 * currently records move_played only for a possible FUTURE policy head,
 * per Appendix F.3 Estagio 2, and does not itself decode it). If exact
 * parity with search.c's 20-bit pack_move is ever required, widen
 * SAMPLE_DTYPE's move_played to uint32 instead of silently reusing the
 * bits this field drops today. */
static inline uint16_t pack_move16(const Move *m) {
    return sample_pack_move(m->from, m->to, m->prom, m->epc);
}

static inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* ═══════════════════════════════════════════════════════════════════
 * Aligned alloc — duplicated from main.c (which itself duplicates it
 * from nnue.c's static copy) — see header comment "WHAT WAS DUPLICATED".
 * Needed for NnueAccum, which nnue.c's AVX2 kernels require 32-byte
 * aligned. ═══════════════════════════════════════════════════════════ */
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

/* ── Portable logical-CPU-count detection (no <windows.h> — avoids
 * pulling in ~1MB of macro surface into a translation unit that already
 * includes board.h/search.h/nnue.h; NUMBER_OF_PROCESSORS is set by
 * every Windows cmd/PowerShell/Explorer-launched process). ─────────── */
#if defined(_WIN32)
static int detect_cpus(void) {
    const char *e = getenv("NUMBER_OF_PROCESSORS");
    int n = e ? atoi(e) : 0;
    return n > 0 ? n : 4;
}
#else
#include <unistd.h>
static int detect_cpus(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 4;
}
#endif

/* ═══════════════════════════════════════════════════════════════════
 * RNG — xorshift64*, one instance per worker, reseeded PER GAME from
 * (--seed, game_index) so runs are reproducible regardless of which
 * thread happens to claim which game (dynamic work-stealing means that
 * assignment is NOT deterministic run-to-run — only the game index is).
 * ═══════════════════════════════════════════════════════════════════ */
typedef struct { uint64_t s; } Rng;

static inline uint64_t rng_next_u64(Rng *r) {
    uint64_t x = r->s;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    r->s = x;
    return x * 0x2545F4914F6CDD1DULL;
}
/* Uniform double in [0, 1), 53 bits of precision. */
static inline double rng_next_double(Rng *r) {
    return (double)(rng_next_u64(r) >> 11) * (1.0 / 9007199254740992.0);
}
/* Deterministic per-game seed: splitmix64 warm-up over
 * (global_seed XOR game_idx*golden-ratio-odd-constant) to decorrelate
 * nearby game indices (xorshift64* is sensitive to low-entropy seeds,
 * e.g. seed=0 or seed=1 would otherwise start every stream almost
 * identically for small game indices). */
static void rng_seed_for_game(Rng *r, uint64_t global_seed, uint64_t game_idx) {
    uint64_t s = global_seed ^ (game_idx * 0x9E3779B97F4A7C15ULL + 0xD1B54A32D192ED03ULL);
    for (int i = 0; i < 4; i++) {
        s += 0x9E3779B97F4A7C15ULL;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        s = z;
    }
    r->s = s ? s : 0xD1B54A32D192ED03ULL;   /* xorshift64* needs a nonzero state */
}

/* ═══════════════════════════════════════════════════════════════════
 * ===== CONFIGURATION =====
 * Defaults for every CLI option — CLAUDE.md rule 8: everything
 * reachable from the command line must ALSO be settable from this
 * documented block, and the CLI default must literally BE the constant
 * below (config_defaults() only ever reads from here — never a second
 * copy of the same literal). See tools/arena.c's own CONFIGURATION
 * block for the same convention applied there.
 * ═══════════════════════════════════════════════════════════════════ */
#define SP_DEFAULT_GAMES              100          /* number of self-play games */
#define SP_DEFAULT_THREADS              0          /* worker threads; 0 = autodetect logical cores */
#define SP_DEFAULT_MOVETIME_MS        100          /* per-move time budget, ms; 0 = use --nodes/--depth instead */
#define SP_DEFAULT_NODES                0          /* per-move node budget; used only when movetime==0 and depth==0 */
#define SP_DEFAULT_DEPTH                0          /* per-move depth budget, plies; 0 = use movetime/nodes instead (parity gap fix: matches run_selfplay.py's tc_mode="depth") */
#define SP_DEFAULT_MULTIPV              4          /* root candidates sampled for temperature move choice */
#define SP_DEFAULT_TEMPERATURE        1.0          /* softmax T0 for the first --temp-plies plies of SEARCH (opening-forced plies don't count, see resolve_opening_for_game()) */
#define SP_DEFAULT_TEMP_SCALE        100.0         /* centipawns per softmax unit before applying T */
#define SP_DEFAULT_TEMP_PLIES          24          /* search plies using T0 before switching to --temp-final */
#define SP_DEFAULT_TEMP_FINAL        0.05          /* softmax T1 after --temp-plies (near-argmax, not exactly) */
#define SP_DEFAULT_MAX_PLIES          400          /* game-length safety cap (opening + search plies) -> counted as draw */
#define SP_DEFAULT_SEED                 1          /* RNG base seed, per-game deterministic */
#define SP_DEFAULT_SEPARATE_TT          0          /* 0 = shared TT between colors within a game (default) */
#define SP_DEFAULT_TT_MB              8.0          /* per-TTable memory budget, MB */
#define SP_DEFAULT_NNUE_PATH  "nnue_weights.bin"   /* NNUE weights file */
#define SP_DEFAULT_OUT_PATH            ""          /* "" = --out is required, no default */
#define SP_DEFAULT_PGN_PATH            ""          /* "" = no --pgn output (rule 9: .bin and .pgn are not either/or) */
#define SP_DEFAULT_EPD_PATH            ""          /* "" = no --epd output (rule 9: .bin/.pgn/.epd are all additive, not either/or) */
#define SP_DEFAULT_SAVE_OPENING_IN_EPD  1          /* 1 = include forced opening-phase plies in the EPD output (default ON,
                                                     * matches tests/run_selfplay.py's SAVE_OPENING_IN_EPD=True — EPD's
                                                     * inclusion default is intentionally the OPPOSITE of the .bin's
                                                     * SP_DEFAULT_SAVE_OPENING_SAMPLES=0, see that constant's comment) */
#define SP_DEFAULT_OPENINGS_PATH       ""          /* "" = no opening corpus indexed -> book/all modes degrade to random */
#define SP_DEFAULT_OPENING_MODE   "book"            /* "book" | "random" | "all" — see tests/run_selfplay.py's OPENING_MODE */
#define SP_DEFAULT_RANDOM_PLIES         6          /* plies of random legal opening moves (random/all modes; matches run_selfplay.py's RANDOM_PLIES) */
#define SP_DEFAULT_BOOK_PORTION      0.97          /* "all" mode: fraction of (paired) games using a book opening vs random plies (matches run_selfplay.py's BOOK_PORTION) */
#define SP_DEFAULT_SAME_OPENING_TWICE   1          /* 1 = paired games (2k,2k+1) share one opening -> variance reduction; matches run_selfplay.py's SAME_OPENING_TWICE default (ON) */
#define SP_DEFAULT_SAVE_OPENING_SAMPLES 0          /* 0 = exclude forced opening-phase plies from the .bin (they were not chosen by search, see header "OPENING SUPPORT") */
/* ═══════════════════════════════════════════════════════════════════ */

#define SP_STR(x)  #x
#define SP_XSTR(x) SP_STR(x)   /* expand x before stringizing, so --help shows the VALUE not the macro name */

typedef enum { OPEN_MODE_BOOK = 0, OPEN_MODE_RANDOM = 1, OPEN_MODE_ALL = 2 } OpenMode;

static int open_mode_from_str(const char *s, OpenMode *out) {
    if (!strcmp(s, "book"))   { *out = OPEN_MODE_BOOK;   return 1; }
    if (!strcmp(s, "random")) { *out = OPEN_MODE_RANDOM; return 1; }
    if (!strcmp(s, "all"))    { *out = OPEN_MODE_ALL;    return 1; }
    return 0;
}

typedef struct {
    int      games;
    int      threads;
    int      movetime_ms;     /* 0 = use nodes/depth instead */
    long     nodes;           /* 0 = unlimited (only meaningful if movetime_ms==0 and depth==0) */
    int      depth;           /* 0 = use movetime/nodes instead */
    int      multipv;         /* MultiPV candidates to sample among */
    double   temperature;     /* T0 */
    double   temp_scale;      /* cp scale divisor before softmax/T   */
    int      temp_plies;      /* plies using T0 before switching to T1 */
    double   temp_final;      /* T1 */
    int      max_plies;       /* hard game-length cap -> draw */
    uint64_t seed;
    int      separate_tt;     /* 0 = shared TT between colors (default) */
    double   tt_mb;           /* per-TTable size budget, megabytes */
    char     nnue_path[512];
    char     out_path[512];
    char     pgn_path[512];         /* "" = no --pgn output */
    char     epd_path[512];         /* "" = no --epd output */
    int      save_opening_in_epd;   /* 1 = include forced opening plies in the EPD output (default ON) */
    char     openings_path[512];    /* "" = no opening corpus -> every game starts at standard startpos */
    OpenMode opening_mode;
    int      random_plies;
    double   book_portion;          /* "all" mode only */
    int      same_opening_twice;
    int      save_opening_samples;  /* 1 = include forced opening-phase plies in the .bin (off by default) */
} Config;

static void config_defaults(Config *c) {
    memset(c, 0, sizeof(*c));
    c->games               = SP_DEFAULT_GAMES;
    c->threads             = SP_DEFAULT_THREADS > 0 ? SP_DEFAULT_THREADS : detect_cpus();
    c->movetime_ms         = SP_DEFAULT_MOVETIME_MS;
    c->nodes                = SP_DEFAULT_NODES;
    c->depth                = SP_DEFAULT_DEPTH;
    c->multipv              = SP_DEFAULT_MULTIPV;
    c->temperature          = SP_DEFAULT_TEMPERATURE;
    c->temp_scale           = SP_DEFAULT_TEMP_SCALE;
    c->temp_plies           = SP_DEFAULT_TEMP_PLIES;
    c->temp_final           = SP_DEFAULT_TEMP_FINAL;
    c->max_plies            = SP_DEFAULT_MAX_PLIES;
    c->seed                 = SP_DEFAULT_SEED;
    c->separate_tt          = SP_DEFAULT_SEPARATE_TT;
    c->tt_mb                = SP_DEFAULT_TT_MB;
    strncpy(c->nnue_path, SP_DEFAULT_NNUE_PATH, sizeof(c->nnue_path) - 1);
    strncpy(c->out_path, SP_DEFAULT_OUT_PATH, sizeof(c->out_path) - 1);
    strncpy(c->pgn_path, SP_DEFAULT_PGN_PATH, sizeof(c->pgn_path) - 1);
    strncpy(c->epd_path, SP_DEFAULT_EPD_PATH, sizeof(c->epd_path) - 1);
    c->save_opening_in_epd  = SP_DEFAULT_SAVE_OPENING_IN_EPD;
    strncpy(c->openings_path, SP_DEFAULT_OPENINGS_PATH, sizeof(c->openings_path) - 1);
    open_mode_from_str(SP_DEFAULT_OPENING_MODE, &c->opening_mode);
    c->random_plies         = SP_DEFAULT_RANDOM_PLIES;
    c->book_portion         = SP_DEFAULT_BOOK_PORTION;
    c->same_opening_twice   = SP_DEFAULT_SAME_OPENING_TWICE;
    c->save_opening_samples = SP_DEFAULT_SAVE_OPENING_SAMPLES;
}

static void print_usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s --out FILE.bin [options]\n"
        "  --games N            number of self-play games (default " SP_XSTR(SP_DEFAULT_GAMES) ")\n"
        "  --threads N          worker threads (default = logical cores)\n"
        "  --movetime MS        per-move time budget in ms (default " SP_XSTR(SP_DEFAULT_MOVETIME_MS) "; 0 = use --nodes/--depth)\n"
        "  --nodes N            per-move node budget (used when --movetime 0 and --depth 0)\n"
        "  --depth N            per-move depth budget, plies (default off; overrides --movetime/--nodes when > 0)\n"
        "  --multipv N          root candidates to sample among (default " SP_XSTR(SP_DEFAULT_MULTIPV) ")\n"
        "  --temperature T0     softmax temperature for the first --temp-plies SEARCH plies (default " SP_XSTR(SP_DEFAULT_TEMPERATURE) ")\n"
        "  --temp-scale CP      centipawns per softmax unit before applying T (default " SP_XSTR(SP_DEFAULT_TEMP_SCALE) ")\n"
        "  --temp-plies N       plies using T0 before switching to --temp-final (default " SP_XSTR(SP_DEFAULT_TEMP_PLIES) ")\n"
        "  --temp-final T1      softmax temperature after --temp-plies (default " SP_XSTR(SP_DEFAULT_TEMP_FINAL) ", ~argmax)\n"
        "  --max-plies N        game-length safety cap, treated as draw (default " SP_XSTR(SP_DEFAULT_MAX_PLIES) ")\n"
        "  --seed N             RNG base seed, per-game deterministic (default " SP_XSTR(SP_DEFAULT_SEED) ")\n"
        "  --separate-tt        give each color its own TTable (default: shared within a game)\n"
        "  --tt-mb MB           per-TTable memory budget in MB (default " SP_XSTR(SP_DEFAULT_TT_MB) ")\n"
        "  --nnue PATH          NNUE weights file (default " SP_DEFAULT_NNUE_PATH ")\n"
        "  --out PATH           output .bin file (required, append mode)\n"
        "  --pgn PATH           write all games as standard PGN to PATH (default: no PGN output; rule 9 — not either/or with --out)\n"
        "  --epd PATH           write every position as EPD (FEN + c0/c1/c2/c3 eval opcodes) to PATH\n"
        "                       (default: no EPD output; rule 9 — not either/or with --out/--pgn;\n"
        "                       same c0/c1/c2/c3 convention as tests/run_selfplay.py's EPD export)\n"
        "  --save-opening-in-epd     include forced opening plies in the EPD output (default: ON)\n"
        "  --no-save-opening-in-epd  exclude forced opening plies from the EPD output\n"
        "  --openings PATH      opening corpus: a .pgn/.epd file OR a directory walked recursively\n"
        "                       (e.g. openings/ for openings/lines/*.pgn + openings/positions/*.epd)\n"
        "                       (default: none -> book/all opening-mode degrades to random)\n"
        "  --opening-mode MODE  book | random | all (default " SP_DEFAULT_OPENING_MODE ") — see tests/run_selfplay.py's OPENING_MODE\n"
        "  --random-plies N     random legal opening plies for random/all modes (default " SP_XSTR(SP_DEFAULT_RANDOM_PLIES) ")\n"
        "  --book-portion F     \"all\" mode: fraction of games using a book opening (default " SP_XSTR(SP_DEFAULT_BOOK_PORTION) ")\n"
        "  --same-opening-twice     pair games (2k,2k+1) on the same opening (default: ON, matches run_selfplay.py)\n"
        "  --no-same-opening-twice  disable pairing -> every game gets its own opening draw\n"
        "  --save-opening-samples   include forced opening-phase plies in the .bin (default: excluded, see header)\n",
        argv0);
}

static int parse_args(int argc, char **argv, Config *cfg) {
    config_defaults(cfg);
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        #define NEXT_INT()    atoi(argv[++i])
        #define NEXT_LONG()   atol(argv[++i])
        #define NEXT_DOUBLE() atof(argv[++i])
        #define NEXT_STR()    argv[++i]
        #define NEED_ARG()    if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a); return -1; }
        if (!strcmp(a, "--games"))            { NEED_ARG(); cfg->games = NEXT_INT(); }
        else if (!strcmp(a, "--threads"))     { NEED_ARG(); cfg->threads = NEXT_INT(); }
        else if (!strcmp(a, "--movetime"))    { NEED_ARG(); cfg->movetime_ms = NEXT_INT(); }
        else if (!strcmp(a, "--nodes"))       { NEED_ARG(); cfg->nodes = NEXT_LONG(); }
        else if (!strcmp(a, "--depth"))       { NEED_ARG(); cfg->depth = NEXT_INT(); }
        else if (!strcmp(a, "--multipv"))     { NEED_ARG(); cfg->multipv = NEXT_INT(); }
        else if (!strcmp(a, "--temperature")) { NEED_ARG(); cfg->temperature = NEXT_DOUBLE(); }
        else if (!strcmp(a, "--temp-scale"))  { NEED_ARG(); cfg->temp_scale = NEXT_DOUBLE(); }
        else if (!strcmp(a, "--temp-plies"))  { NEED_ARG(); cfg->temp_plies = NEXT_INT(); }
        else if (!strcmp(a, "--temp-final"))  { NEED_ARG(); cfg->temp_final = NEXT_DOUBLE(); }
        else if (!strcmp(a, "--max-plies"))   { NEED_ARG(); cfg->max_plies = NEXT_INT(); }
        else if (!strcmp(a, "--seed"))        { NEED_ARG(); cfg->seed = (uint64_t)strtoull(NEXT_STR(), NULL, 10); }
        else if (!strcmp(a, "--separate-tt")) { cfg->separate_tt = 1; }
        else if (!strcmp(a, "--tt-mb"))       { NEED_ARG(); cfg->tt_mb = NEXT_DOUBLE(); }
        else if (!strcmp(a, "--nnue"))        { NEED_ARG(); strncpy(cfg->nnue_path, NEXT_STR(), sizeof(cfg->nnue_path) - 1); }
        else if (!strcmp(a, "--out"))         { NEED_ARG(); strncpy(cfg->out_path, NEXT_STR(), sizeof(cfg->out_path) - 1); }
        else if (!strcmp(a, "--pgn"))         { NEED_ARG(); strncpy(cfg->pgn_path, NEXT_STR(), sizeof(cfg->pgn_path) - 1); }
        else if (!strcmp(a, "--epd"))         { NEED_ARG(); strncpy(cfg->epd_path, NEXT_STR(), sizeof(cfg->epd_path) - 1); }
        else if (!strcmp(a, "--save-opening-in-epd"))    { cfg->save_opening_in_epd = 1; }
        else if (!strcmp(a, "--no-save-opening-in-epd")) { cfg->save_opening_in_epd = 0; }
        else if (!strcmp(a, "--openings"))    { NEED_ARG(); strncpy(cfg->openings_path, NEXT_STR(), sizeof(cfg->openings_path) - 1); }
        else if (!strcmp(a, "--opening-mode")) {
            NEED_ARG();
            const char *v = NEXT_STR();
            if (!open_mode_from_str(v, &cfg->opening_mode)) {
                fprintf(stderr, "error: --opening-mode must be book|random|all, got '%s'\n", v);
                return -1;
            }
        }
        else if (!strcmp(a, "--random-plies"))         { NEED_ARG(); cfg->random_plies = NEXT_INT(); }
        else if (!strcmp(a, "--book-portion"))         { NEED_ARG(); cfg->book_portion = NEXT_DOUBLE(); }
        else if (!strcmp(a, "--same-opening-twice"))    { cfg->same_opening_twice = 1; }
        else if (!strcmp(a, "--no-same-opening-twice")) { cfg->same_opening_twice = 0; }
        else if (!strcmp(a, "--save-opening-samples"))  { cfg->save_opening_samples = 1; }
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { print_usage(argv[0]); exit(0); }
        else { fprintf(stderr, "unknown argument: %s\n", a); print_usage(argv[0]); return -1; }
        #undef NEXT_INT
        #undef NEXT_LONG
        #undef NEXT_DOUBLE
        #undef NEXT_STR
        #undef NEED_ARG
    }
    if (!cfg->out_path[0]) { fprintf(stderr, "error: --out is required\n"); print_usage(argv[0]); return -1; }
    if (cfg->games <= 0)   { fprintf(stderr, "error: --games must be > 0\n"); return -1; }
    if (cfg->threads <= 0) cfg->threads = 1;
    if (cfg->multipv < 1)  cfg->multipv = 1;
    if (cfg->multipv > MAX_MULTI_PV) cfg->multipv = MAX_MULTI_PV;
    if (cfg->max_plies < 1) cfg->max_plies = 1;
    if (cfg->depth < 0) cfg->depth = 0;
    if (cfg->random_plies < 0) cfg->random_plies = 0;
    if (cfg->random_plies > OPENING_MAX_FORCED_MOVES) cfg->random_plies = OPENING_MAX_FORCED_MOVES;
    if (cfg->book_portion < 0.0) cfg->book_portion = 0.0;
    if (cfg->book_portion > 1.0) cfg->book_portion = 1.0;
    return 0;
}

/* Round a requested TT memory budget (MB) down to a valid TTable size:
 * a power of two multiple of TT_BUCKETS.  ~26 bytes/logical entry
 * (8+4+4+2+4+4 for H/S/D/G/M/E — see TTable in search.h). */
static size_t tt_entries_for_mb(double mb) {
    const double bytes_per_entry = 26.0;
    size_t want = (size_t)((mb * 1024.0 * 1024.0) / bytes_per_entry);
    size_t n = TT_BUCKETS;
    while (n * 2 <= want) n *= 2;
    if (n < (size_t)(TT_BUCKETS * 512)) n = (size_t)TT_BUCKETS * 512; /* sane floor */
    return n;
}

/* ═══════════════════════════════════════════════════════════════════
 * Shared (process-wide) run state — atomics only, no locks needed for
 * these; the output file gets its own mutex separately.
 * ═══════════════════════════════════════════════════════════════════ */
static atomic_int       g_next_game;
static atomic_llong     g_games_done;
static atomic_llong     g_positions_written;
static pthread_mutex_t  g_out_mutex = PTHREAD_MUTEX_INITIALIZER;
static FILE            *g_out = NULL;
/* --pgn output (rule 9: .bin and .pgn are not either/or).  Guarded by
 * its own mutex so a PGN write never serialises behind a .bin flush. */
static pthread_mutex_t  g_pgn_mutex = PTHREAD_MUTEX_INITIALIZER;
static FILE            *g_pgn = NULL;
/* --epd output (rule 9: .bin/.pgn/.epd are all additive, never either/or).
 * Own mutex for the same reason g_pgn has one — an EPD write never
 * serialises behind a .bin or PGN flush. */
static pthread_mutex_t  g_epd_mutex = PTHREAD_MUTEX_INITIALIZER;
static FILE            *g_epd = NULL;

/* ═══════════════════════════════════════════════════════════════════
 * Per-worker context — everything one thread needs to play games with
 * NO sharing except: (a) NNUE weight arrays (global, read-only after
 * load), (b) magic/Zobrist/leaper tables (global, read-only after
 * board_init()), (c) the output FILE* (write-only, mutex-guarded).
 * ═══════════════════════════════════════════════════════════════════ */
static const char *SP_STARTPOS_FEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/* One EPD row, captured pre-move during play_one_game(), same in spirit as
 * a SelfplaySample but independent of it: --save-opening-in-epd and
 * --save-opening-samples are separate flags with DIFFERENT defaults (see
 * SP_DEFAULT_SAVE_OPENING_IN_EPD's comment), so the EPD row list and the
 * .bin sample buffer can legitimately disagree on which plies they cover. */
typedef struct {
    char fen[128];
    int  cp_white;    /* WHITE-relative eval, matching tests/run_selfplay.py's
                        * c1 convention exactly (that script converts the
                        * engine's own STM-relative score back to White's
                        * POV before writing it) — 0 for forced/opening plies */
    int  is_forced;    /* forced opening ply vs a real search decision */
} EpdRow;

typedef struct {
    int            id;
    const Config  *cfg;
    Board          board;
    UndoFrame     *undo;
    int            undo_top;
    NnueAccum     *nnue;
    SearchState   *ss;
    TTable        *tt_shared;   /* used when !cfg->separate_tt */
    TTable        *tt_w;        /* used when cfg->separate_tt  */
    TTable        *tt_b;
    Rng            rng;
    SelfplaySample *buf;        /* per-game sample buffer, reused across games */
    size_t          buf_cap;
    /* Opening-book support (see "OPENING SUPPORT" header comment):
     * a second, throwaway Board+undo+nnue used ONLY to replay a PGN
     * opening's SAN mainline / generate random legal opening plies
     * BEFORE the real game board is touched — needs its own undo/nnue
     * for the exact same reason w->board does (board_load_fen()'s
     * global-rebind bug, finding #5 of the thread-safety audit; this
     * scratch board calls board_make() too, so it is just as exposed). */
    Board          scratch_board;
    UndoFrame     *scratch_undo;
    int            scratch_undo_top;
    NnueAccum     *scratch_nnue;
    /* PGN output (--pgn): reused per-game movetext buffer. */
    char          *pgn_buf;
    size_t         pgn_cap;
    /* EPD output (--epd): reused per-game row buffer (raw rows, filled
     * during the ply loop) and a reused text buffer (final "FEN c0 ...;"
     * lines, built once the game's outcome is known — see the second-pass
     * comment in play_one_game()). */
    EpdRow        *epd_rows;
    size_t         epd_rows_cap;
    size_t         epd_count;
    char          *epd_text;
    size_t         epd_text_cap;
    /* Aggregate stats for the final report */
    long long games_played;
    long long positions;
    long long wins_white, wins_black, draws;
} WorkerCtx;

/* Reset a worker's Board to `fen` (or the standard start position if
 * fen==NULL) AND fix the global-rebinding bug in board_load_fen()
 * described in the header comment (finding #5 of the thread-safety
 * audit) — this is the single most important correctness line in this
 * file. Every board_load_fen() call anywhere in this tool MUST be
 * immediately followed by this. */
static void worker_reset_board(WorkerCtx *w, const char *fen) {
    board_load_fen(&w->board, fen ? fen : SP_STARTPOS_FEN);
    /* board_load_fen() just memset the whole Board to 0 and rebound
     * undo/nnue to the GLOBAL g_undo/g_nnue_accum — undo that here. */
    w->undo_top = 0;
    w->board.undo     = w->undo;
    w->board.undo_top = &w->undo_top;
    w->board.nnue     = w->nnue;
}

/* Same rebind fix, for the worker's SCRATCH board (opening resolution
 * only — never touched by search). */
static void worker_reset_scratch(WorkerCtx *w) {
    board_load_fen(&w->scratch_board, SP_STARTPOS_FEN);
    w->scratch_undo_top = 0;
    w->scratch_board.undo     = w->scratch_undo;
    w->scratch_board.undo_top = &w->scratch_undo_top;
    w->scratch_board.nnue     = w->scratch_nnue;
}

/* Pseudo-legal -> legal move filter (board_gen_moves() is pseudo-legal
 * only, same as everywhere else in this engine — see search.c's own
 * qsearch()/alpha_beta() loops, which apply the identical
 * board_make()+board_is_attacked()+board_unmake() pattern duplicated
 * here). Needed for --opening-mode random/all: tests/run_selfplay.py's
 * random_opening() samples from python-chess's board.legal_moves,
 * which IS fully legal, so this tool must filter the same way to match
 * that semantics rather than occasionally playing a move that leaves
 * its own king in check. */
static int gen_legal_moves(Board *b, Move *out) {
    Move pseudo[MAX_MOVES];
    int np = board_gen_moves(b, pseudo);
    int n = 0;
    for (int i = 0; i < np; i++) {
        board_make(b, &pseudo[i]);
        int mover_col = b->turn ^ 24;
        int king_sq   = mover_col == COL_W ? b->wk : b->bk;
        int illegal   = board_is_attacked(b, king_sq, b->turn);
        board_unmake(b);
        if (!illegal) out[n++] = pseudo[i];
    }
    return n;
}

/* Ensure the per-game sample buffer can hold at least cfg->max_plies
 * records (one sample can be produced per ply at most). Allocated once
 * per worker, reused (not freed/realloc'd) across games. */
static void worker_ensure_buf(WorkerCtx *w) {
    size_t need = (size_t)w->cfg->max_plies + 1;
    if (w->buf_cap >= need) return;
    SelfplaySample *nb = (SelfplaySample *)realloc(w->buf, need * sizeof(SelfplaySample));
    if (!nb) { fprintf(stderr, "[worker %d] OOM growing sample buffer\n", w->id); exit(1); }
    w->buf = nb;
    w->buf_cap = need;
}

/* Same growth pattern as worker_ensure_buf(), for the EPD row buffer.
 * Capacity need is the same upper bound (one row can be produced per ply
 * at most, forced or searched — see play_one_game()'s row-append site). */
static void worker_ensure_epd_buf(WorkerCtx *w) {
    size_t need = (size_t)w->cfg->max_plies + 1;
    if (w->epd_rows_cap >= need) return;
    EpdRow *nb = (EpdRow *)realloc(w->epd_rows, need * sizeof(EpdRow));
    if (!nb) { fprintf(stderr, "[worker %d] OOM growing EPD row buffer\n", w->id); exit(1); }
    w->epd_rows = nb;
    w->epd_rows_cap = need;
}

/* ═══════════════════════════════════════════════════════════════════
 * OPENING SUPPORT (capability audit fix — see this file's header
 * comment's original omission: every game used to start from the
 * identical standard position).
 *
 *   g_opening_index is built once in main() (before any worker starts,
 *   like nnue_load()/board_init() — see audit finding #8) from
 *   --openings via opening_index_build() (tools/opening_pool.c),
 *   mirroring tests/run_selfplay.py's OpeningIndex: byte offsets only,
 *   the multi-hundred-thousand-entry corpus is never loaded into RAM.
 *
 *   resolve_opening_for_game() below decides, per game, whether that
 *   game starts from a book entry or N random legal plies, and — for
 *   --same-opening-twice (default ON, matches run_selfplay.py) — makes
 *   sure paired games (2k, 2k+1) get the IDENTICAL book choice. This
 *   mirrors run_selfplay.py's own semantics exactly:
 *     - "book": every game samples one entry from the corpus.
 *     - "random": every game plays --random-plies random legal moves
 *       from the standard start position.
 *     - "all": each (pair of) game(s) independently rolls book vs
 *       random with P(book) = --book-portion (matches BOOK_PORTION).
 *   Per run_selfplay.py's play_game(): when the pick is "book", BOTH
 *   games of a same-opening-twice pair replay the SAME index (the
 *   corpus entry is fetched once per PAIR); when the pick is "random",
 *   random_opening() is called FRESH per game even inside a pair (see
 *   that function's call site in worker_loop() — it is not memoized
 *   across the pair). This file reproduces that asymmetry exactly: the
 *   book-vs-random ROLL and the book INDEX are drawn from an
 *   opening_rng seeded by (--seed, pair_idx) so both pair members agree,
 *   but the actual random-ply MOVES are drawn from w->rng, which is
 *   seeded per GAME (rng_seed_for_game(), unchanged, called earlier in
 *   play_one_game()) — so two paired "random" games still diverge.
 * ═══════════════════════════════════════════════════════════════════ */
static OpeningIndex g_opening_index;   /* built once in main(), read-only after that */

typedef struct {
    int   use_fen;                        /* board starts from a custom FEN (EPD book entry) */
    char  fen[128];
    int   n_forced;                        /* forced moves to replay from the standard start
                                             * (PGN book entry OR random plies) before search begins */
    Move  forced[OPENING_MAX_FORCED_MOVES];
} OpeningChoice;

static void resolve_opening_for_game(WorkerCtx *w, uint64_t game_idx, OpeningChoice *out) {
    const Config *cfg = w->cfg;
    memset(out, 0, sizeof(*out));

    OpenMode mode = cfg->opening_mode;
    if ((mode == OPEN_MODE_BOOK || mode == OPEN_MODE_ALL) && g_opening_index.n_entries == 0)
        mode = OPEN_MODE_RANDOM;   /* no corpus indexed -> degrade gracefully, see --openings default */

    /* Pair-seeded RNG: decides the book-vs-random roll (all mode) and
     * the book index, so both members of a --same-opening-twice pair
     * agree on both (see header comment above). Seeded distinctly from
     * w->rng (game-indexed, used for random-ply moves and temperature
     * sampling) via an XOR salt so the two streams never collide. */
    uint64_t pair_idx = cfg->same_opening_twice ? (game_idx / 2) : game_idx;
    Rng orng;
    rng_seed_for_game(&orng, cfg->seed ^ 0x9E3779B97F4A7C15ULL, pair_idx);

    int use_book;
    if (mode == OPEN_MODE_BOOK)        use_book = 1;
    else if (mode == OPEN_MODE_RANDOM) use_book = 0;
    else                                use_book = (rng_next_double(&orng) < cfg->book_portion);

    if (use_book) {
        size_t pick = (size_t)(rng_next_double(&orng) * (double)g_opening_index.n_entries);
        if (pick >= g_opening_index.n_entries) pick = g_opening_index.n_entries - 1;

        worker_reset_scratch(w);
        OpeningPick p;
        if (opening_index_fetch(&g_opening_index, pick, &w->scratch_board, &p)) {
            if (p.has_fen) {
                out->use_fen = 1;
                strncpy(out->fen, p.fen, sizeof(out->fen) - 1);
            } else if (p.n_moves > 0) {
                out->n_forced = p.n_moves;
                memcpy(out->forced, p.moves, sizeof(Move) * (size_t)p.n_moves);
            }
            return;
        }
        /* Corpus entry failed to parse (should not happen against a
         * clean corpus, but corpora are external data) — fall through
         * to random plies rather than silently starting at startpos,
         * so a handful of bad lines don't quietly collapse diversity. */
    }

    int n = cfg->random_plies;
    if (n > OPENING_MAX_FORCED_MOVES) n = OPENING_MAX_FORCED_MOVES;
    if (n > 0) {
        worker_reset_scratch(w);
        for (int i = 0; i < n; i++) {
            Move legal[MAX_MOVES];
            int cnt = gen_legal_moves(&w->scratch_board, legal);
            if (cnt <= 0) break;   /* checkmate/stalemate inside the random walk — vanishingly rare, just stop */
            int idx = (int)(rng_next_double(&w->rng) * cnt);
            if (idx >= cnt) idx = cnt - 1;
            out->forced[out->n_forced++] = legal[idx];
            board_make(&w->scratch_board, &legal[idx]);
        }
    }
}

/* Softmax-with-temperature move selection over MultiPV root candidates.
 * `stm_scores_cp` are STM-relative centipawn scores (already flipped
 * from SearchResult's White-relative scores[] by the caller), length
 * n (n == res.num_pvs, clamped to cfg->multipv). Returns the chosen
 * candidate index. See header comment "TEMPERATURE MOVE SELECTION" for
 * the exact formula and rationale. */
static int select_by_temperature(const int *stm_scores_cp, int n, double T,
                                  double temp_scale, Rng *rng) {
    if (n <= 1) return 0;
    if (T < 1e-6) T = 1e-6;   /* guard against div-by-zero; T this small is de facto argmax anyway */

    double z[MAX_MULTI_PV];
    double zmax = -1e300;
    for (int i = 0; i < n; i++) {
        z[i] = (stm_scores_cp[i] / temp_scale) / T;
        if (z[i] > zmax) zmax = z[i];
    }
    double p[MAX_MULTI_PV];
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        p[i] = exp(z[i] - zmax);
        sum += p[i];
    }
    double roll = rng_next_double(rng) * sum;
    double cum = 0.0;
    for (int i = 0; i < n; i++) {
        cum += p[i];
        if (roll < cum) return i;
    }
    return n - 1;   /* floating-point rounding fallback */
}

/* ═══════════════════════════════════════════════════════════════════
 * play_one_game — the reusable game-loop this file's header comment
 * promises a future native A/B arena can reuse.  Plays one game to
 * completion (checkmate / stalemate / draw by board_is_draw() / the
 * --max-plies cap), fills w->buf[0..count) with samples (game_result
 * left at 0 — see finish_game() for the second pass), and returns the
 * sample count. `game_idx` seeds this game's RNG deterministically.
 *
 * Opening phase (capability audit fix): plies 0..oc.n_forced-1 are
 * REPLAYED, not searched (a book PGN entry's mainline, or random legal
 * plies — see resolve_opening_for_game()). Those plies are excluded
 * from the recorded .bin samples by default (cfg->save_opening_samples
 * off) because they were not chosen by search — see this file's
 * SAMPLE FORMAT header comment: eval_cp is documented as "the score
 * from the search that CHOSE the move," which is simply untrue for a
 * forced opening ply, and the fixed .bin record format (a binding
 * contract with train/dataset.py, NOT to be changed here) has no field
 * to flag a sample as "unchosen" instead. Excluding by default is the
 * correct default; --save-opening-samples is available for callers who
 * want the extra positions anyway and understand eval_cp will read 0
 * for them. The temperature schedule (--temp-plies) counts SEARCH
 * plies only, i.e. `ply - oc.n_forced` once forced plies are behind
 * us — the opening phase itself has no engine "decision" to temper.
 * ═══════════════════════════════════════════════════════════════════ */
typedef enum { RES_DRAW = 0, RES_WHITE_WINS = 1, RES_BLACK_WINS = 2 } GameOutcome;

static size_t play_one_game(WorkerCtx *w, uint64_t game_idx, GameOutcome *outcome,
                             char *pgn_movetext, size_t pgn_cap, char *start_fen_out,
                             char *epd_text, size_t epd_text_cap) {
    const Config *cfg = w->cfg;
    rng_seed_for_game(&w->rng, cfg->seed, game_idx);
    worker_ensure_buf(w);
    if (cfg->epd_path[0]) worker_ensure_epd_buf(w);
    w->epd_count = 0;

    OpeningChoice oc;
    resolve_opening_for_game(w, game_idx, &oc);
    worker_reset_board(w, oc.use_fen ? oc.fen : NULL);
    if (start_fen_out) strncpy(start_fen_out, oc.use_fen ? oc.fen : SP_STARTPOS_FEN, 127);

    size_t pgn_len = 0;
    if (pgn_movetext && pgn_cap) pgn_movetext[0] = 0;
    if (epd_text && epd_text_cap) epd_text[0] = 0;

    /* TT policy (Appendix F.2): physical clear before EVERY game, even
     * in shared mode — see header comment "TT POLICY" for why a logical
     * generation bump is not sufficient here. */
    if (cfg->separate_tt) {
        tt_clear(w->tt_w);
        tt_clear(w->tt_b);
    } else {
        tt_clear(w->tt_shared);
    }

    size_t n = 0;
    GameOutcome result = RES_DRAW;
    int ply;
    for (ply = 0; ply < cfg->max_plies; ply++) {
        int dr = board_is_draw(&w->board);
        if (dr == 1 || dr == 2) { result = RES_DRAW; break; }  /* 50-move/material, or 3-fold */

        int white_to_move = (w->board.turn == COL_W);
        int is_forced = (ply < oc.n_forced);

        Move mv;
        int chosen_eval_cp = 0;
        int have_move = 1;

        if (is_forced) {
            mv = oc.forced[ply];
        } else {
            double T = ((ply - oc.n_forced) < cfg->temp_plies) ? cfg->temperature : cfg->temp_final;

            SearchParams sp;
            memset(&sp, 0, sizeof(sp));
            sp.start_depth   = 0;
            if (cfg->depth > 0) {
                /* --depth: per-move depth budget, unbounded time/nodes —
                 * mirrors main.c's UCI "go depth N" handling exactly
                 * (parity gap the capability audit flagged: this tool
                 * had movetime/nodes but no depth mode). */
                sp.max_depth     = cfg->depth;
                sp.time_limit_ms = 0;
                sp.node_limit    = 0;
            } else {
                sp.max_depth     = MAX_PLY - 1;   /* same cap main.c uses for movetime/nodes-only "go" */
                sp.time_limit_ms = cfg->movetime_ms;
                sp.node_limit    = cfg->nodes;
            }
            sp.multi_pv      = cfg->multipv;
            sp.threads       = 1;             /* CRITICAL: no nested Lazy SMP — this worker
                                                * thread IS the search thread; spawning helper
                                                * threads from inside search_best() itself is
                                                * neither implemented (search_best doesn't spawn
                                                * anything — main.c's search_thread_fn does that
                                                * externally) nor wanted here: N workers x M
                                                * nested helpers would oversubscribe the machine
                                                * and break the "1 OS thread per logical search"
                                                * assumption this whole tool is built on. */
            sp.stop          = NULL;
            sp.search_state  = w->ss;
            sp.info_cb       = NULL;
            sp.tt            = cfg->separate_tt ? (white_to_move ? w->tt_w : w->tt_b)
                                                 : w->tt_shared;
            sp.mpv_share_budget = 1;  /* v4.00: only need root scores for the
                                        * softmax below, not N full-depth lines —
                                        * see header comment "SHARED MULTIPV TIME
                                        * BUDGET" for why this matters here. */

            SearchResult res = search_best(&w->board, &sp);

            if (res.bests[0].from == 0 && res.bests[0].to == 0) {
                have_move = 0;
            } else {
                int num_cand = res.num_pvs;
                if (num_cand < 1) num_cand = 1;
                if (num_cand > cfg->multipv) num_cand = cfg->multipv;

                int stm_scores[MAX_MULTI_PV];
                for (int i = 0; i < num_cand; i++)
                    stm_scores[i] = white_to_move ? res.scores[i] : -res.scores[i];

                int chosen = select_by_temperature(stm_scores, num_cand, T, cfg->temp_scale, &w->rng);
                mv = res.bests[chosen];
                chosen_eval_cp = stm_scores[chosen];
            }
        }

        if (!have_move) {
            /* No legal move: checkmate or stalemate (see header comment
             * "GAME TERMINATION" — search_best()'s own root loop already
             * detected this; not reimplemented here). */
            if (board_in_check(&w->board)) {
                result = white_to_move ? RES_BLACK_WINS : RES_WHITE_WINS;
            } else {
                result = RES_DRAW;
            }
            break;
        }

        /* EPD row capture — independent inclusion policy from the .bin
         * sample below (see EpdRow's own comment): --save-opening-in-epd
         * (default ON) vs --save-opening-samples (default OFF). Captured
         * from the PRE-move board, same as the sample below. */
        if (cfg->epd_path[0] && (!is_forced || cfg->save_opening_in_epd)) {
            worker_ensure_epd_buf(w);
            EpdRow *row = &w->epd_rows[w->epd_count++];
            board_to_fen(&w->board, row->fen);
            row->is_forced = is_forced;
            /* c1 is WHITE-relative (matches tests/run_selfplay.py's own
             * convention exactly — chosen_eval_cp here is STM-relative). */
            row->cp_white = is_forced ? 0 : (white_to_move ? chosen_eval_cp : -chosen_eval_cp);
        }

        if (!is_forced || cfg->save_opening_samples) {
            SelfplaySample *rec = &w->buf[n++];
            memcpy(rec->board, w->board.b, 64);
            rec->stm         = white_to_move ? 0 : 1;
            rec->rule50      = w->board.hm;
            rec->castling    = w->board.ca;
            rec->ep_file     = (w->board.ep < 0) ? 8 : (uint8_t)(w->board.ep & 7);
            rec->eval_cp     = is_forced ? 0 : (int16_t)clampi(chosen_eval_cp, -32000, 32000);
            rec->game_result = 0;   /* filled below once the outcome is known */
            rec->move_played = pack_move16(&mv);
            rec->_pad        = 0;
        }

        char san_buf[16] = {0};
        if (pgn_movetext) move_to_san(&w->board, &mv, san_buf, sizeof(san_buf));

        board_make(&w->board, &mv);

        if (pgn_movetext) {
            Move tmp_legal[MAX_MOVES];
            int check_now = board_in_check(&w->board);
            int has_moves = board_gen_moves(&w->board, tmp_legal) > 0;
            size_t sl = strlen(san_buf);
            if (check_now && sl + 1 < sizeof(san_buf)) {
                san_buf[sl++] = has_moves ? '+' : '#';
                san_buf[sl] = 0;
            }
            char tok[32];
            int tn = white_to_move
                ? snprintf(tok, sizeof(tok), "%d. %s ", ply / 2 + 1, san_buf)
                : snprintf(tok, sizeof(tok), "%s ", san_buf);
            if (tn > 0 && pgn_len + (size_t)tn < pgn_cap) {
                memcpy(pgn_movetext + pgn_len, tok, (size_t)tn);
                pgn_len += (size_t)tn;
                pgn_movetext[pgn_len] = 0;
            }
        }
    }
    if (ply >= cfg->max_plies) result = RES_DRAW;   /* safety cap reached */

    /* Second pass (required — see header comment "What to record per
     * position"): game_result is STM-of-that-position-relative, which
     * can only be known once the final outcome is known. */
    for (size_t i = 0; i < n; i++) {
        int8_t gr;
        if (result == RES_DRAW) gr = 0;
        else {
            int sample_is_white = (w->buf[i].stm == 0);
            int white_won = (result == RES_WHITE_WINS);
            gr = (sample_is_white == white_won) ? 1 : -1;
        }
        w->buf[i].game_result = gr;
    }

    /* Build the EPD text now that the final result is known — c0 is the
     * GAME-level PGN-style result string, same for every row, which is
     * why this has to wait until after the loop (mirrors the .bin second
     * pass above). Convention: "<fen> c0 \"<result>\"; c1 \"<white-cp>\";
     * c2 \"<evaluator>\"; c3 \"<row index>\";" — byte-for-byte the same
     * opcode set tests/run_selfplay.py's write_epd()/play_game() emit, so
     * EPD files from both tools are interchangeable training data. */
    if (epd_text && epd_text_cap && w->epd_count > 0) {
        const char *res_str = (result == RES_WHITE_WINS) ? "1-0"
                             : (result == RES_BLACK_WINS) ? "0-1" : "1/2-1/2";
        size_t len = 0;
        for (size_t i = 0; i < w->epd_count; i++) {
            const EpdRow *row = &w->epd_rows[i];
            const char *evaluator = row->is_forced ? "book" : "Zchezz-native";
            int wrote = snprintf(epd_text + len, epd_text_cap - len,
                "%s c0 \"%s\"; c1 \"%d\"; c2 \"%s\"; c3 \"%zu\";\n",
                row->fen, res_str, row->cp_white, evaluator, i);
            if (wrote < 0 || len + (size_t)wrote >= epd_text_cap) break;  /* truncate, don't overflow */
            len += (size_t)wrote;
        }
    }

    *outcome = result;
    return n;
}

/* ═══════════════════════════════════════════════════════════════════
 * worker_fn — thread entry point: claim games from the shared atomic
 * counter until exhausted, playing each to completion and flushing its
 * samples under g_out_mutex (per-thread accumulate-then-flush, per the
 * task's "preferred: less contention" guidance — one lock/fwrite/fflush
 * per GAME, not per sample).
 * ═══════════════════════════════════════════════════════════════════ */
static void *worker_fn(void *arg) {
    WorkerCtx *w = (WorkerCtx *)arg;
    const Config *cfg = w->cfg;

    /* PGN movetext buffer: allocated once per worker, only when --pgn is
     * active.  8 KB comfortably holds max_plies SAN tokens; play_one_game
     * truncates rather than overflowing if a game somehow exceeds it. */
    if (cfg->pgn_path[0] && !w->pgn_buf) {
        w->pgn_cap = 8192;
        w->pgn_buf = (char *)malloc(w->pgn_cap);
        if (!w->pgn_buf) { fprintf(stderr, "[worker %d] OOM allocating PGN buffer\n", w->id); exit(1); }
    }

    /* EPD text buffer: allocated once per worker, only when --epd is
     * active.  64 KB comfortably holds --max-plies FEN+opcode lines
     * (~110 bytes/line x 400 plies default is ~44 KB); play_one_game
     * truncates rather than overflowing if a game somehow exceeds it. */
    if (cfg->epd_path[0] && !w->epd_text) {
        w->epd_text_cap = 65536;
        w->epd_text = (char *)malloc(w->epd_text_cap);
        if (!w->epd_text) { fprintf(stderr, "[worker %d] OOM allocating EPD text buffer\n", w->id); exit(1); }
    }

    for (;;) {
        int g = atomic_fetch_add(&g_next_game, 1);
        if (g >= cfg->games) break;

        GameOutcome outcome;
        char start_fen[128];
        start_fen[0] = 0;
        size_t n = play_one_game(w, (uint64_t)g, &outcome,
                                 w->pgn_buf, w->pgn_cap, start_fen,
                                 w->epd_text, w->epd_text_cap);

        if (n > 0) {
            pthread_mutex_lock(&g_out_mutex);
            fwrite(w->buf, sizeof(SelfplaySample), n, g_out);
            fflush(g_out);
            pthread_mutex_unlock(&g_out_mutex);
        }

        if (g_pgn && w->pgn_buf) {
            const char *res = (outcome == RES_WHITE_WINS) ? "1-0"
                            : (outcome == RES_BLACK_WINS) ? "0-1" : "1/2-1/2";
            /* [FEN]/[SetUp] only when the game did NOT start from the
             * standard position — a PGN reader must otherwise assume
             * startpos, and emitting a redundant FEN confuses some GUIs. */
            int from_fen = (start_fen[0] && strcmp(start_fen, SP_STARTPOS_FEN) != 0);
            pthread_mutex_lock(&g_pgn_mutex);
            fprintf(g_pgn,
                    "[Event \"Zchezz selfplay\"]\n[Site \"native\"]\n"
                    "[Round \"%d\"]\n[White \"Zchezz\"]\n[Black \"Zchezz\"]\n"
                    "[Result \"%s\"]\n", g + 1, res);
            if (from_fen)
                fprintf(g_pgn, "[SetUp \"1\"]\n[FEN \"%s\"]\n", start_fen);
            fprintf(g_pgn, "\n%s %s\n\n", w->pgn_buf, res);
            fflush(g_pgn);
            pthread_mutex_unlock(&g_pgn_mutex);
        }

        if (g_epd && w->epd_text && w->epd_text[0]) {
            pthread_mutex_lock(&g_epd_mutex);
            fputs(w->epd_text, g_epd);
            fflush(g_epd);
            pthread_mutex_unlock(&g_epd_mutex);
        }

        w->games_played++;
        w->positions += (long long)n;
        if      (outcome == RES_WHITE_WINS) w->wins_white++;
        else if (outcome == RES_BLACK_WINS) w->wins_black++;
        else                                 w->draws++;

        atomic_fetch_add(&g_games_done, 1);
        atomic_fetch_add(&g_positions_written, (long long)n);
    }
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════════
 * main — one-time global init (NOT thread-safe, must run before any
 * worker starts — see audit finding #8), worker pool spawn/join,
 * progress reporting, final summary.
 * ═══════════════════════════════════════════════════════════════════ */
int main(int argc, char **argv) {
    Config cfg;
    if (parse_args(argc, argv, &cfg) != 0) return 1;

    board_init();
    if (nnue_load(cfg.nnue_path) != 0) {
        fprintf(stderr, "[selfplay] failed to load NNUE weights from '%s'\n", cfg.nnue_path);
        return 1;
    }
    /* Pre-seed g_tt with a tiny dummy allocation so search_init()'s
     * `if (!g_tt) g_tt = tt_create(TT_SIZE)` guard skips allocating the
     * process-wide default (up to ~104 MB native) — this tool never
     * uses g_tt (every SearchParams.tt is set explicitly per worker
     * below), so that would be pure waste. */
    g_tt = tt_create(TT_BUCKETS * 512);
    search_init();

    g_out = fopen(cfg.out_path, "ab");
    if (!g_out) {
        fprintf(stderr, "[selfplay] failed to open output file '%s'\n", cfg.out_path);
        return 1;
    }

    /* Build the opening index ONCE, before any worker starts.  It stores
     * byte offsets only (never the positions themselves) — the corpus is
     * ~850k entries and loading it into RAM would dwarf everything else
     * this tool allocates.  Read-only afterwards, so all workers share it
     * without a lock. */
    if (cfg.openings_path[0]) {
        /* NOTE the convention: opening_index_build() returns 0 on SUCCESS
         * and -1 only if `root` does not exist — a successful build with
         * zero files found still returns 0, so the "did we get anything"
         * question is answered by n_entries, not by the return code. */
        int rc = opening_index_build(&g_opening_index, cfg.openings_path);
        if (rc != 0 || g_opening_index.n_entries == 0) {
            fprintf(stderr, "[selfplay] WARNING: no openings indexed from '%s' — "
                            "falling back to --random-plies for every game\n",
                    cfg.openings_path);
        } else {
            fprintf(stderr, "[selfplay] openings: %zu entries indexed from '%s'\n",
                    g_opening_index.n_entries, cfg.openings_path);
        }
    } else if (cfg.opening_mode != OPEN_MODE_RANDOM) {
        fprintf(stderr, "[selfplay] NOTE: --opening-mode=%s but no --openings given; "
                        "every game will use --random-plies instead\n",
                cfg.opening_mode == OPEN_MODE_BOOK ? "book" : "all");
    }

    /* --pgn/--epd are additive, never a replacement for the .bin (rule 9). */
    if (cfg.pgn_path[0]) {
        g_pgn = fopen(cfg.pgn_path, "a");
        if (!g_pgn) {
            fprintf(stderr, "[selfplay] failed to open PGN file '%s'\n", cfg.pgn_path);
            fclose(g_out);
            return 1;
        }
    }
    if (cfg.epd_path[0]) {
        g_epd = fopen(cfg.epd_path, "a");
        if (!g_epd) {
            fprintf(stderr, "[selfplay] failed to open EPD file '%s'\n", cfg.epd_path);
            fclose(g_out);
            if (g_pgn) fclose(g_pgn);
            return 1;
        }
    }

    size_t tt_entries = tt_entries_for_mb(cfg.tt_mb);
    fprintf(stderr,
        "[selfplay] games=%d threads=%d movetime=%dms nodes=%ld multipv=%d "
        "temp=%.3g->%.3g@%dplies scale=%.0fcp max_plies=%d seed=%llu "
        "tt=%s(%zu entries/table, ~%.1fMB) nnue=%s out=%s pgn=%s epd=%s\n",
        cfg.games, cfg.threads, cfg.movetime_ms, cfg.nodes, cfg.multipv,
        cfg.temperature, cfg.temp_final, cfg.temp_plies, cfg.temp_scale, cfg.max_plies,
        (unsigned long long)cfg.seed,
        cfg.separate_tt ? "separate" : "shared", tt_entries,
        (tt_entries * 26.0) / (1024.0 * 1024.0),
        cfg.nnue_path, cfg.out_path,
        cfg.pgn_path[0] ? cfg.pgn_path : "(none)",
        cfg.epd_path[0] ? cfg.epd_path : "(none)");

    atomic_init(&g_next_game, 0);
    atomic_init(&g_games_done, 0);
    atomic_init(&g_positions_written, 0);

    int n_threads = cfg.threads;
    pthread_t   *tids = (pthread_t *)malloc(n_threads * sizeof(pthread_t));
    WorkerCtx   *ws   = (WorkerCtx *)calloc((size_t)n_threads, sizeof(WorkerCtx));
    if (!tids || !ws) { fprintf(stderr, "[selfplay] OOM allocating worker pool\n"); return 1; }

    for (int i = 0; i < n_threads; i++) {
        WorkerCtx *w = &ws[i];
        w->id  = i;
        w->cfg = &cfg;
        w->undo = (UndoFrame *)malloc(STACK_SIZE * sizeof(UndoFrame));
        w->undo_top = 0;
        w->nnue = (NnueAccum *)zmalloc32(sizeof(NnueAccum));
        if (w->nnue) {
            memset(w->nnue, 0, sizeof(NnueAccum));
            /* v4.00 Phase 0-follow-up: bind this worker's accumulator to
             * the net loaded by nnue_load() above (g_nnue_net). Every
             * accumulator needs a bound net or nnue_eval* silently
             * returns 0 for it — see nnue.h's NnueAccum::net comment. */
            w->nnue->net = g_nnue_net;
        }
        /* Scratch board's OWN undo stack + accumulator.  These are not a
         * luxury: resolve_opening_for_game() runs BEFORE the real board is
         * reset, and it calls board_make()/board_unmake() to replay a book
         * line and to filter legal moves.  Without its own undo stack the
         * scratch board would either dereference NULL (crash) or, worse,
         * share w->undo and silently corrupt the real game's undo stack.
         * The accumulator is never evaluated here (openings are replayed,
         * not searched) but board_make() pushes to it regardless. */
        w->scratch_undo = (UndoFrame *)malloc(STACK_SIZE * sizeof(UndoFrame));
        w->scratch_undo_top = 0;
        w->scratch_nnue = (NnueAccum *)zmalloc32(sizeof(NnueAccum));
        if (w->scratch_nnue) {
            memset(w->scratch_nnue, 0, sizeof(NnueAccum));
            w->scratch_nnue->net = g_nnue_net;
        }
        w->ss = search_state_new();
        if (cfg.separate_tt) {
            w->tt_w = tt_create(tt_entries);
            w->tt_b = tt_create(tt_entries);
        } else {
            w->tt_shared = tt_create(tt_entries);
        }
        if (!w->undo || !w->nnue || !w->scratch_undo || !w->scratch_nnue || !w->ss ||
            (cfg.separate_tt ? (!w->tt_w || !w->tt_b) : !w->tt_shared)) {
            fprintf(stderr, "[selfplay] OOM setting up worker %d\n", i);
            return 1;
        }
        w->buf = NULL;
        w->buf_cap = 0;

        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, 8 * 1024 * 1024);  /* alpha_beta's deep recursion needs this (see main.c) */
        pthread_create(&tids[i], &attr, worker_fn, w);
        pthread_attr_destroy(&attr);
    }

    /* Progress reporting: poll the shared atomics from the main thread
     * every ~500ms until every game has been counted done. */
    long long t0 = (long long)time(NULL);
    long long last_done = -1;
    for (;;) {
        struct timespec ts = {0, 500 * 1000 * 1000};
        nanosleep(&ts, NULL);
        long long done = atomic_load(&g_games_done);
        long long pos  = atomic_load(&g_positions_written);
        long long now  = (long long)time(NULL);
        double secs = (double)(now - t0);
        double gps  = secs > 0 ? (double)done / secs : 0.0;
        if (done != last_done) {
            fprintf(stderr, "\r[selfplay] games %lld/%d  positions %lld  %.2f games/sec   ",
                    done, cfg.games, pos, gps);
            fflush(stderr);
            last_done = done;
        }
        if (done >= cfg.games) break;
    }
    fprintf(stderr, "\n");

    for (int i = 0; i < n_threads; i++) pthread_join(tids[i], NULL);
    fclose(g_out);
    if (g_pgn) fclose(g_pgn);
    if (g_epd) fclose(g_epd);

    long long tot_games = 0, tot_pos = 0, tot_ww = 0, tot_wb = 0, tot_dr = 0;
    for (int i = 0; i < n_threads; i++) {
        tot_games += ws[i].games_played;
        tot_pos   += ws[i].positions;
        tot_ww    += ws[i].wins_white;
        tot_wb    += ws[i].wins_black;
        tot_dr    += ws[i].draws;
    }
    long long total_secs = (long long)time(NULL) - t0;
    fprintf(stderr,
        "[selfplay] DONE games=%lld positions=%lld white_wins=%lld black_wins=%lld draws=%lld "
        "elapsed=%llds (%.2f games/sec)\n",
        tot_games, tot_pos, tot_ww, tot_wb, tot_dr, total_secs,
        total_secs > 0 ? (double)tot_games / (double)total_secs : 0.0);

    for (int i = 0; i < n_threads; i++) {
        WorkerCtx *w = &ws[i];
        if (w->buf) free(w->buf);
        if (w->epd_rows) free(w->epd_rows);
        if (w->epd_text) free(w->epd_text);
        if (w->pgn_buf) free(w->pgn_buf);
        if (w->undo) free(w->undo);
        if (w->nnue) zfree32(w->nnue);
        if (w->ss) search_state_free(w->ss);
        if (w->tt_shared) tt_destroy(w->tt_shared);
        if (w->tt_w) tt_destroy(w->tt_w);
        if (w->tt_b) tt_destroy(w->tt_b);
    }
    free(ws);
    free(tids);
    return 0;
}
