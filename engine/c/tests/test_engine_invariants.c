/* Native correctness invariants for the current Zchezz engine API.
 *
 * This harness tests state restoration and representation consistency without
 * going through UCI. It intentionally links board.c and nnue.c directly.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "board.h"
#include "nnue.h"

static int failures = 0;

#define CHECK(cond, ...) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: "); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
        failures++; \
    } \
} while (0)

typedef struct {
    uint8_t b[64];
    uint64_t bb[12];
    uint64_t occ, occ_w, occ_b;
    int turn;
    uint8_t ca;
    int8_t ep;
    uint8_t hm;
    uint16_t fm;
    uint8_t wk, bk;
    int castled_w, castled_b;
    uint64_t hash;
    uint64_t hist[HIST_SIZE];
    int hist_len;
} Snapshot;

static int bb_index(uint8_t piece) {
    switch (piece) {
        case WP: return 0; case WN: return 1; case WB: return 2;
        case WR: return 3; case WQ: return 4; case WK: return 5;
        case BP: return 6; case BN: return 7; case BB: return 8;
        case BR: return 9; case BQ: return 10; case BK: return 11;
        default: return -1;
    }
}

static Snapshot snapshot(const Board *b) {
    Snapshot s;
    memcpy(s.b, b->b, sizeof(s.b));
    memcpy(s.bb, b->bb, sizeof(s.bb));
    s.occ = b->occ; s.occ_w = b->occ_w; s.occ_b = b->occ_b;
    s.turn = b->turn; s.ca = b->ca; s.ep = b->ep; s.hm = b->hm; s.fm = b->fm;
    s.wk = b->wk; s.bk = b->bk; s.castled_w = b->castled_w; s.castled_b = b->castled_b;
    s.hash = b->hash; s.hist_len = b->hist_len;
    memcpy(s.hist, b->hist, sizeof(s.hist));
    return s;
}

static int snapshot_equal(const Snapshot *a, const Board *b) {
    if (memcmp(a->b, b->b, sizeof(a->b)) != 0) return 0;
    if (memcmp(a->bb, b->bb, sizeof(a->bb)) != 0) return 0;
    if (a->occ != b->occ || a->occ_w != b->occ_w || a->occ_b != b->occ_b) return 0;
    if (a->turn != b->turn || a->ca != b->ca || a->ep != b->ep || a->hm != b->hm || a->fm != b->fm) return 0;
    if (a->wk != b->wk || a->bk != b->bk || a->castled_w != b->castled_w || a->castled_b != b->castled_b) return 0;
    if (a->hash != b->hash || a->hist_len != b->hist_len) return 0;
    /* Entries beyond hist_len are scratch history, not logical position state. */
    if (a->hist_len > 0 && memcmp(a->hist, b->hist, (size_t)a->hist_len * sizeof(uint64_t)) != 0) return 0;
    return 1;
}

static void check_consistency(const Board *b, const char *context) {
    uint64_t expected_bb[12] = {0};
    for (int sq = 0; sq < 64; sq++) {
        int bi = bb_index(b->b[sq]);
        if (bi >= 0) expected_bb[bi] |= (uint64_t)1 << sq;
    }
    CHECK(memcmp(expected_bb, b->bb, sizeof(expected_bb)) == 0, "%s: mailbox and bitboards differ", context);

    uint64_t ow = 0, ob = 0;
    for (int i = 0; i < 6; i++) ow |= b->bb[i];
    for (int i = 6; i < 12; i++) ob |= b->bb[i];
    CHECK(ow == b->occ_w, "%s: occ_w differs", context);
    CHECK(ob == b->occ_b, "%s: occ_b differs", context);
    CHECK((ow | ob) == b->occ, "%s: occ differs", context);
    CHECK((ow & ob) == 0, "%s: white/black occupancy overlap", context);
    CHECK(b->wk < 64 && b->b[b->wk] == WK, "%s: white king square invalid", context);
    CHECK(b->bk < 64 && b->b[b->bk] == BK, "%s: black king square invalid", context);

    uint64_t recomputed = board_compute_hash(b->b, b->turn, b->ca, b->ep);
    CHECK(recomputed == b->hash, "%s: incremental hash differs from recomputed hash", context);
}

static int compare_nnue_full_rebuild(Board *b, const char *context) {
    if (!nnue_ready()) return 1;
    int stm = b->turn == COL_B;
    int incremental = nnue_eval(b->nnue, stm, b->b);

    NnueAccum *fresh = (NnueAccum *)calloc(1, sizeof(NnueAccum));
    if (!fresh) {
        CHECK(0, "%s: cannot allocate fresh NNUE accumulator", context);
        return 0;
    }
    fresh->net = g_nnue_net;
    fresh->acc_dirty = 1;
    nnue_rebuild(fresh, b->b);
    int rebuilt = nnue_eval(fresh, stm, b->b);
    CHECK(incremental == rebuilt, "%s: incremental NNUE %d != rebuilt %d", context, incremental, rebuilt);
    free(fresh);
    return incremental == rebuilt;
}

static void run_sequence(const char *name, const char *fen, const char *moves[], int count) {
    Board b;
    g_undo_top = 0;
    CHECK(board_load_fen(&b, fen) == 0, "%s: FEN load failed", name);
    if (nnue_ready()) nnue_rebuild(b.nnue, b.b);
    check_consistency(&b, name);
    compare_nnue_full_rebuild(&b, name);
    Snapshot original = snapshot(&b);

    for (int i = 0; i < count; i++) {
        Move move;
        int before_failures = failures;
        CHECK(move_from_uci(&b, moves[i], &move), "%s: move %s not generated", name, moves[i]);
        if (failures != before_failures) return;
        board_make(&b, &move);
        char context[128];
        snprintf(context, sizeof(context), "%s after %s", name, moves[i]);
        check_consistency(&b, context);
        compare_nnue_full_rebuild(&b, context);
    }

    for (int i = count - 1; i >= 0; i--) board_unmake(&b);
    CHECK(g_undo_top == 0, "%s: undo stack did not return to zero", name);
    CHECK(snapshot_equal(&original, &b), "%s: make/unmake did not restore complete snapshot", name);
    check_consistency(&b, name);
}

static void test_nnue_feature_contracts(void) {
    CHECK(nnue_feature_index(WP, 48, 1, 0) == 8,
          "white-POV WP a2 feature index changed");
    CHECK(nnue_feature_index(BP, 8, 0, 0) == 8,
          "black-POV BP a7 feature index changed");
    CHECK(nnue_feature_index(WK, 60, 1, 0) == -1,
          "king must not be an NNUE feature");
    CHECK(nnue_king_bucket_w(60) == 1,
          "white king e1 bucket mapping changed");
    CHECK(nnue_king_bucket_b(4) == 1,
          "black king e8 bucket mapping changed");
}

static void test_draw_contracts(void) {
    Board b;
    g_undo_top = 0;
    board_load_fen(&b, "8/8/8/8/8/8/4K3/4k3 w - - 100 1");
    CHECK(board_is_draw(&b) == 1, "50-move rule not detected at hm=100");
    board_load_fen(&b, "8/8/8/8/8/8/4K3/4k3 w - - 0 1");
    CHECK(board_is_draw(&b) == 1, "K vs K insufficient material not detected");
    board_load_fen(&b, "8/8/8/8/8/8/4K3/3Nk3 w - - 0 1");
    CHECK(board_is_draw(&b) == 1, "KN vs K insufficient material not detected");
}

int main(int argc, char **argv) {
    board_init();
    if (argc > 1) {
        if (nnue_load(argv[1]) != 0) {
            fprintf(stderr, "WARN: NNUE file could not be loaded; NNUE equivalence checks skipped\n");
        }
    } else {
        fprintf(stderr, "WARN: no NNUE path supplied; NNUE equivalence checks skipped\n");
    }

    const char *opening[] = {"e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1"};
    run_sequence("opening+castle", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", opening, 9);

    const char *ep[] = {"e5d6"};
    run_sequence("en-passant", "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", ep, 1);

    const char *promotion[] = {"a7a8q"};
    run_sequence("promotion", "4k3/P7/8/8/8/8/8/4K3 w - - 0 1", promotion, 1);

    test_draw_contracts();
    test_nnue_feature_contracts();

    if (failures) {
        fprintf(stderr, "FAILED: %d invariant(s)\n", failures);
        return 1;
    }
    printf("PASS: native engine invariants\n");
    return 0;
}

