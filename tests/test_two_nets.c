/* test_two_nets.c — proof that NnueNet is truly per-instance (v4.00
 * Phase 0-follow-up).
 *
 * WHY this test exists: before this change, nnue.c held the weight
 * matrices as file-scope statics (_nnL1WT, _nnL1B, ...) — a process
 * could load exactly one network, ever. The native A/B arena needs
 * TWO different generations' weights alive in the SAME process at the
 * SAME time (gen_{i+1} vs gen_i). This test is the direct proof that
 * now works:
 *
 *   1. Load net A and net B (two different random NNU4 files) into two
 *      independent NnueNet objects in one process.
 *   2. Build two NnueAccum, one bound to each net, for the SAME board
 *      position.
 *   3. Show eval(A) != eval(B) — the nets are actually independent, not
 *      aliases of the same storage.
 *   4. Show eval(A) via the new per-instance API is bit-identical to
 *      what the OLD/legacy global API (nnue_load) computes when net A
 *      is the only net loaded in an otherwise-fresh process — i.e. the
 *      per-instance refactor did not change any arithmetic, it just
 *      stopped sharing storage.
 *
 * Linked directly against nnue.c (not the full engine) so this stays a
 * fast, dependency-free unit test. g_nnue_accum, which nnue.c expects
 * to find at link time (board.c normally provides it), is defined here
 * instead — exactly like nnue.c's own NNUE_TEST harness does.
 *
 * Build:
 *   gcc -O3 -mavx2 -std=c11 -I <engine_dir> -o test_two_nets.exe \
 *       test_two_nets.c <engine_dir>/nnue.c -lm
 *
 * Usage:
 *   test_two_nets.exe <net_a.bin> <net_b.bin>
 */
#include "nnue.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* nnue.c wants this symbol (normally defined in board.c). */
NnueAccum g_nnue_accum;

/* Piece constants — must match board.h / nnue.c's PC_COLOR/PC_TYPE. */
#define COL_W 8
#define COL_B 16
#define WP  9
#define WN 10
#define WB 11
#define WR 12
#define WQ 13
#define WK 14
#define BP 17
#define BN 18
#define BB 19
#define BR 20
#define BQ 21
#define BK 22

/* Standard opening position, Zchezz mailbox order (a8=0 .. h1=63). */
static void build_startpos(uint8_t board[64]) {
    static const uint8_t row8[8] = { BR, BN, BB, BQ, BK, BB, BN, BR };
    static const uint8_t row1[8] = { WR, WN, WB, WQ, WK, WB, WN, WR };
    memset(board, 0, 64);
    for (int f = 0; f < 8; f++) {
        board[f]        = row8[f];
        board[8 + f]     = BP;
        board[48 + f]    = WP;
        board[56 + f]    = row1[f];
    }
}

/* A second, structurally different position (a middlegame-ish setup)
 * so the two-net divergence isn't just tested on one board. */
static void build_midgame(uint8_t board[64]) {
    /* r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4 */
    static const uint8_t row[8*8] = {
        BR,0,BB,BQ,BK,BB,0,BR,
        BP,BP,BP,BP,0,BP,BP,BP,
        0,0,BN,0,0,BN,0,0,
        0,0,0,0,BP,0,0,0,
        0,0,WB,0,WP,0,0,0,
        0,0,0,0,0,WN,0,0,
        WP,WP,WP,WP,0,WP,WP,WP,
        WR,WN,WB,WQ,WK,0,0,WR,
    };
    memcpy(board, row, 64);
}

static int fail_count = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); fail_count++; } \
    else fprintf(stderr, "OK:   %s\n", msg); \
} while (0)

int main(int argc, char **argv) {
    const char *path_a = argc > 1 ? argv[1] : "random_nnu4.bin";
    const char *path_b = argc > 2 ? argv[2] : "random_nnu4_seed777.bin";

    /* ── Part 1: two independent NnueNet instances, one process ──── */
    NnueNet *netA = nnue_net_load(path_a);
    NnueNet *netB = nnue_net_load(path_b);
    CHECK(netA != NULL, "net A loaded");
    CHECK(netB != NULL, "net B loaded");
    CHECK(nnue_net_ready(netA), "net A ready");
    CHECK(nnue_net_ready(netB), "net B ready");
    CHECK(netA != netB, "net A and net B are distinct allocations");

    uint8_t board_start[64], board_mid[64];
    build_startpos(board_start);
    build_midgame(board_mid);

    NnueAccum accA, accB;
    memset(&accA, 0, sizeof accA);
    memset(&accB, 0, sizeof accB);
    accA.net = netA;
    accB.net = netB;

    nnue_rebuild(&accA, board_start);
    nnue_rebuild(&accB, board_start);
    int evalA_start = nnue_eval(&accA, 0, board_start);
    int evalB_start = nnue_eval(&accB, 0, board_start);
    fprintf(stderr, "startpos: eval(netA)=%d  eval(netB)=%d\n", evalA_start, evalB_start);
    CHECK(evalA_start != evalB_start,
          "startpos: net A and net B disagree (proves independent weight storage)");

    nnue_rebuild(&accA, board_mid);
    nnue_rebuild(&accB, board_mid);
    int evalA_mid = nnue_eval(&accA, 0, board_mid);
    int evalB_mid = nnue_eval(&accB, 0, board_mid);
    fprintf(stderr, "midgame:  eval(netA)=%d  eval(netB)=%d\n", evalA_mid, evalB_mid);
    CHECK(evalA_mid != evalB_mid,
          "midgame: net A and net B disagree (proves independent weight storage)");

    /* Sanity: re-evaluating net A after having loaded/evaluated net B
     * must be unaffected by net B's existence — no shared statics. */
    nnue_rebuild(&accA, board_start);
    int evalA_start_again = nnue_eval(&accA, 0, board_start);
    CHECK(evalA_start_again == evalA_start,
          "net A's startpos eval is unaffected by net B's presence");

    /* ── Part 2: per-instance API matches the legacy global API ───── */
    /* Load net A a SECOND time, this time through the legacy global
     * path (nnue_load), exactly as the single-net UCI engine always
     * has. If the per-instance refactor changed no arithmetic, this
     * must produce bit-identical evals to accA above, on both boards. */
    int r = nnue_load(path_a);
    CHECK(r == 0, "legacy nnue_load(net A) succeeded");
    CHECK(nnue_ready(), "legacy nnue_ready() true after nnue_load");

    nnue_reset(&g_nnue_accum);
    nnue_rebuild(&g_nnue_accum, board_start);
    int legacy_start = nnue_eval(&g_nnue_accum, 0, board_start);
    fprintf(stderr, "startpos: eval(legacy global, net A weights)=%d\n", legacy_start);
    CHECK(legacy_start == evalA_start,
          "legacy global API == per-instance API for the same net A weights (startpos)");

    nnue_reset(&g_nnue_accum);
    nnue_rebuild(&g_nnue_accum, board_mid);
    int legacy_mid = nnue_eval(&g_nnue_accum, 0, board_mid);
    fprintf(stderr, "midgame:  eval(legacy global, net A weights)=%d\n", legacy_mid);
    CHECK(legacy_mid == evalA_mid,
          "legacy global API == per-instance API for the same net A weights (midgame)");

    nnue_net_destroy(netA);
    nnue_net_destroy(netB);

    fprintf(stderr, "\n%s (%d failure%s)\n",
            fail_count == 0 ? "ALL PASSED" : "SOME FAILED",
            fail_count, fail_count == 1 ? "" : "s");
    return fail_count == 0 ? 0 : 1;
}
