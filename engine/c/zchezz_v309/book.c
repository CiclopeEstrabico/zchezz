/* book.c — Polyglot opening book reader
 *
 * Implements the standard Polyglot .bin format:
 *   - 16-byte entries: 8B hash, 2B move, 2B weight, 2B learn, 2B padding
 *   - Entries sorted by hash for binary search
 *   - Uses Polyglot-standard Zobrist keys (different from Zchezz internal keys)
 *
 * Square mapping:
 *   Zchezz:   a8=0, h8=7, a1=56, h1=63  (rank 8 first)
 *   Polyglot: a1=0, h1=7, a8=56, h8=63  (rank 1 first)
 *   Conversion: poly_sq = zchezz_sq ^ 56
 */

#ifndef NO_BOOK

#include "book.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ── Polyglot entry ──────────────────────────────────────────── */
typedef struct {
    uint64_t key;
    uint16_t move;
    uint16_t weight;
    uint32_t learn;   /* 2B learn + 2B padding */
} PolyEntry;

static PolyEntry *g_book = NULL;
static int         g_book_n = 0;
static int         g_rng_seeded = 0;

/* ── Read big-endian uint64 from buffer ──────────────────────── */
static uint64_t read_be64(const uint8_t *p) {
    return ((uint64_t)p[0]<<56) | ((uint64_t)p[1]<<48) |
           ((uint64_t)p[2]<<40) | ((uint64_t)p[3]<<32) |
           ((uint64_t)p[4]<<24) | ((uint64_t)p[5]<<16) |
           ((uint64_t)p[6]<< 8) | ((uint64_t)p[7]);
}

static uint16_t read_be16(const uint8_t *p) {
    return (uint16_t)((p[0]<<8) | p[1]);
}

/* ══════════════════════════════════════════════════════════════
 *  POLYGLOT ZOBRIST KEYS
 *
 *  Standard 781 random 64-bit values used by ALL Polyglot-compatible
 *  tools.  Hardcoded from the official PolyGlot source (Fabien Letouzey).
 *
 *  Layout: poly_random64[781]
 *    [  0..767] = piece keys: 12 piece_types × 64 squares
 *                 piece order: BP BN BB BR BQ BK WP WN WB WR WQ WK
 *                 square order: a1=0, b1=1, ..., h8=63
 *    [768..771] = castling: wk, wq, bk, bq
 *    [772..779] = en passant file: a..h
 *    [780]      = turn (XOR when white to move)
 * ══════════════════════════════════════════════════════════════ */

#include "poly_keys.h"


/* ── Compute Polyglot hash from Board ────────────────────────── */
/*
 * Polyglot piece type mapping:
 *   BlackPawn=0, BlackKnight=1, BlackBishop=2, BlackRook=3, BlackQueen=4, BlackKing=5
 *   WhitePawn=6, WhiteKnight=7, WhiteBishop=8, WhiteRook=9, WhiteQueen=10, WhiteKing=11
 *
 * Zchezz piece encoding: WP=9 WN=10 WB=11 WR=12 WQ=13 WK=14 BP=17 BN=18 BB=19 BR=20 BQ=21 BK=22
 * PC_TYPE: P=1 N=2 B=3 R=4 Q=5 K=6
 * PC_COLOR: W=8 B=16
 */
static int zchezz_to_poly_piece(uint8_t p) {
    if (!p) return -1;
    int type  = p & 7;   /* 1=P 2=N 3=B 4=R 5=Q 6=K */
    int color = p & 24;  /* 8=W 16=B */
    /* Polyglot piece index = (type-1)*2 + color_pivot
     * color_pivot: 0=Black, 1=White
     * Result: BP=0 WP=1 BN=2 WN=3 BB=4 WB=5 BR=6 WR=7 BQ=8 WQ=9 BK=10 WK=11 */
    int color_pivot = (color == COL_W) ? 1 : 0;
    return (type - 1) * 2 + color_pivot;
}

static uint64_t polyglot_hash(const Board *b) {

    uint64_t h = 0;

    /* Pieces: for each piece on the board, XOR the corresponding key.
     * Polyglot square: row = rank (0=rank1, 7=rank8), col = file (0=a, 7=h)
     * Zchezz square: sq 0=a8, 1=b8, ..., 56=a1, 57=b1, ..., 63=h1
     * Conversion: poly_sq = (zchezz_sq ^ 56)
     *   but Polyglot index = piece_type * 64 + poly_sq
     *   where poly_sq = row * 8 + col, row = 7 - (zsq >> 3), col = zsq & 7
     *   Equivalently: poly_sq = zsq ^ 56  (flip rank bits) */
    for (int zsq = 0; zsq < 64; zsq++) {
        uint8_t p = b->b[zsq];
        if (!p) continue;
        int poly_piece = zchezz_to_poly_piece(p);
        if (poly_piece < 0) continue;
        int poly_sq = zsq ^ 56;
        h ^= poly_random64[poly_piece * 64 + poly_sq];
    }

    /* Castling */
    if (b->ca & CA_WK) h ^= poly_random64[768];
    if (b->ca & CA_WQ) h ^= poly_random64[769];
    if (b->ca & CA_BK) h ^= poly_random64[770];
    if (b->ca & CA_BQ) h ^= poly_random64[771];

    /* En passant: Polyglot only includes EP if a pawn can actually capture */
    if (b->ep >= 0) {
        int ep_file = b->ep & 7;
        int ep_rank = b->ep >> 3;
        int can_capture = 0;

        if (b->turn == COL_W) {
            /* White pawns that can capture on ep square (ep is on rank 2 = row 2 in Zchezz) */
            /* Pawns are one rank below ep: ep_rank + 1 */
            int left  = (ep_rank + 1) * 8 + (ep_file - 1);
            int right = (ep_rank + 1) * 8 + (ep_file + 1);
            if (ep_file > 0 && left < 64 && b->b[left] == WP) can_capture = 1;
            if (ep_file < 7 && right < 64 && b->b[right] == WP) can_capture = 1;
        } else {
            /* Black pawns that can capture on ep square */
            int left  = (ep_rank - 1) * 8 + (ep_file - 1);
            int right = (ep_rank - 1) * 8 + (ep_file + 1);
            if (ep_file > 0 && left >= 0 && b->b[left] == BP) can_capture = 1;
            if (ep_file < 7 && right >= 0 && b->b[right] == BP) can_capture = 1;
        }

        if (can_capture) {
            h ^= poly_random64[772 + ep_file];
        }
    }

    /* Turn: Polyglot XORs the turn key when WHITE to move */
    if (b->turn == COL_W) h ^= poly_random64[780];

    return h;
}

/* ── Open book ───────────────────────────────────────────────── */
int book_open(const char *path) {
    book_close();
    if (!path || !path[0]) return 0;

    FILE *f = fopen(path, "rb");
    if (!f) return 0;

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (size < 16 || size % 16 != 0) {
        fclose(f);
        return 0;
    }

    int n = (int)(size / 16);
    uint8_t *raw = (uint8_t *)malloc(size);
    if (!raw) { fclose(f); return 0; }

    if ((long)fread(raw, 1, size, f) != size) {
        free(raw); fclose(f); return 0;
    }
    fclose(f);

    g_book = (PolyEntry *)malloc(n * sizeof(PolyEntry));
    if (!g_book) { free(raw); return 0; }

    for (int i = 0; i < n; i++) {
        const uint8_t *e = raw + i * 16;
        g_book[i].key    = read_be64(e);
        g_book[i].move   = read_be16(e + 8);
        g_book[i].weight = read_be16(e + 10);
        g_book[i].learn  = (uint32_t)read_be16(e + 12) << 16 | read_be16(e + 14);
    }
    free(raw);
    g_book_n = n;

    return n;
}

/* ── Close book ──────────────────────────────────────────────── */
void book_close(void) {
    if (g_book) { free(g_book); g_book = NULL; }
    g_book_n = 0;
}

int book_is_loaded(void) {
    return g_book != NULL && g_book_n > 0;
}

/* ── Binary search for first entry with given key ────────────── */
static int book_find_first(uint64_t key) {
    int lo = 0, hi = g_book_n;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (g_book[mid].key < key) lo = mid + 1;
        else hi = mid;
    }
    if (lo < g_book_n && g_book[lo].key == key) return lo;
    return -1;
}

/* ── Convert Polyglot move to Zchezz Move ────────────────────── */
/*
 * Polyglot move encoding (16 bits):
 *   bits  0..2  : to_file   (0=a, 7=h)
 *   bits  3..5  : to_row    (0=rank1, 7=rank8)
 *   bits  6..8  : from_file (0=a, 7=h)
 *   bits  9..11 : from_row  (0=rank1, 7=rank8)
 *   bits 12..14 : promotion (0=none, 1=knight, 2=bishop, 3=rook, 4=queen)
 */
static int poly_move_to_zchezz(uint16_t pmove, const Board *b, Move *out) {
    int to_file   = pmove & 7;
    int to_row    = (pmove >> 3) & 7;
    int from_file = (pmove >> 6) & 7;
    int from_row  = (pmove >> 9) & 7;
    int promo     = (pmove >> 12) & 7;

    /* Polyglot square to Zchezz square: poly = row*8+col, zchezz = poly ^ 56 */
    int poly_from = from_row * 8 + from_file;
    int poly_to   = to_row * 8 + to_file;
    int z_from = poly_from ^ 56;
    int z_to   = poly_to ^ 56;

    /* Build Move struct */
    memset(out, 0, sizeof(Move));
    out->from = (uint8_t)z_from;
    out->to   = (uint8_t)z_to;

    /* Promotion: Polyglot 1=N 2=B 3=R 4=Q → Zchezz 2=N 3=B 4=R 5=Q */
    if (promo >= 1 && promo <= 4) {
        out->prom = (uint8_t)(promo + 1);
    }

    /* Detect castling: king moves 2+ squares horizontally */
    uint8_t piece = b->b[z_from];
    if (piece && PC_TYPE(piece) == 6) {  /* King */
        int df = (z_to & 7) - (z_from & 7);
        if (df == 2 || df == -2) {
            /* Castle — determine which type */
            if (PC_COLOR(piece) == COL_W) {
                out->castle = (df > 0) ? 1 : 2;  /* 1=WK, 2=WQ */
            } else {
                out->castle = (df > 0) ? 3 : 4;  /* 3=BK, 4=BQ */
            }
            /* Adjust 'to' square: Polyglot may encode castling differently.
             * Standard: e1g1 (KS) or e1c1 (QS) — matches Zchezz UCI convention. */
        }
    }

    /* Detect en passant */
    if (piece && PC_TYPE(piece) == 1 && z_to == b->ep && b->ep >= 0) {
        out->epc = 1;
    }

    /* Validate: check that there's actually a piece to move */
    if (!piece) return 0;

    return 1;
}

/* ── Probe book ──────────────────────────────────────────────── */
int book_probe(const Board *b, Move *out) {
    if (!g_book || g_book_n == 0) return 0;

    uint64_t key = polyglot_hash(b);
    int idx = book_find_first(key);
    if (idx < 0) return 0;

    /* Collect all entries with this key */
    uint16_t moves[64];
    uint16_t weights[64];
    int count = 0;
    uint32_t total_weight = 0;

    for (int i = idx; i < g_book_n && g_book[i].key == key && count < 64; i++) {
        if (g_book[i].weight > 0) {
            moves[count]   = g_book[i].move;
            weights[count] = g_book[i].weight;
            total_weight  += g_book[i].weight;
            count++;
        }
    }

    if (count == 0) return 0;

    /* Weighted random selection */
    if (!g_rng_seeded) { srand((unsigned)time(NULL)); g_rng_seeded = 1; }

    uint32_t r = (uint32_t)(rand() % (int)total_weight);
    uint32_t cumul = 0;
    int pick = 0;
    for (int i = 0; i < count; i++) {
        cumul += weights[i];
        if (r < cumul) { pick = i; break; }
    }

    return poly_move_to_zchezz(moves[pick], b, out);
}

#endif /* NO_BOOK */
