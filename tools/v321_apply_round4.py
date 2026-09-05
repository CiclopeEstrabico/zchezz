#!/usr/bin/env python3
"""Apply one v3.21 tree-preserving speed candidate for A/B Elo testing."""
from pathlib import Path
import argparse

VARIANTS = ["nnue_l1_prefetch", "attack_order"]
p = argparse.ArgumentParser()
p.add_argument("variant", choices=VARIANTS)
p.add_argument("--root", default="engine/c/zchezz_v321cand")
a = p.parse_args()
root = Path(a.root)


def replace_exact(path: Path, old: str, new: str, expected: int, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    n = text.count(old)
    if n != expected:
        raise SystemExit(f"{label}: expected {expected} matches, found {n}")
    path.write_text(text.replace(old, new), encoding="utf-8")


if a.variant == "nnue_l1_prefetch":
    path = root / "nnue.c"
    old = """    const int16_t *bRow = _nnL1WT + (coB*64 + pt*64 + sq  ) * NN_L1_OUT;
#ifdef __AVX2__
    for (int o = 0; o < NN_L1_OUT; o += 16) {
"""
    new = """    const int16_t *bRow = _nnL1WT + (coB*64 + pt*64 + sq  ) * NN_L1_OUT;
#ifdef __AVX2__
    /* v3.21 speed candidate: bring both 512-byte feature rows into L1
     * before the accumulator update. Semantics and arithmetic are unchanged. */
    _mm_prefetch((const char*)wRow, _MM_HINT_T0);
    _mm_prefetch((const char*)bRow, _MM_HINT_T0);
    for (int o = 0; o < NN_L1_OUT; o += 16) {
"""
    replace_exact(path, old, new, 2, a.variant)

elif a.variant == "attack_order":
    path = root / "board.c"
    old = """    if (by == COL_W) {
        /* White pawn attacks: bitboard shift (Phase 2) */
        if (bpawn_attacks_bb(sq_bb) & b->bb[0]) return 1;  /* sq attacked by WP */
        if (NATK[sq] & b->bb[1])  return 1;   /* white knight */
        if (bish_attacks(sq,occ) & (b->bb[2]|b->bb[4])) return 1;  /* WB|WQ */
        if (rook_attacks(sq,occ) & (b->bb[3]|b->bb[4])) return 1;  /* WR|WQ */
        if (KATK[sq] & b->bb[5])  return 1;   /* white king */
    } else {
        /* Black pawn attacks: bitboard shift (Phase 2) */
        if (wpawn_attacks_bb(sq_bb) & b->bb[6]) return 1;  /* sq attacked by BP */
        if (NATK[sq] & b->bb[7])  return 1;
        if (bish_attacks(sq,occ) & (b->bb[8]|b->bb[10])) return 1; /* BB|BQ */
        if (rook_attacks(sq,occ) & (b->bb[9]|b->bb[10])) return 1; /* BR|BQ */
        if (KATK[sq] & b->bb[11]) return 1;
    }
"""
    new = """    if (by == COL_W) {
        /* Cheap leapers first; skip magic lookups when no relevant slider exists. */
        if (bpawn_attacks_bb(sq_bb) & b->bb[0]) return 1;
        if (NATK[sq] & b->bb[1]) return 1;
        if (KATK[sq] & b->bb[5]) return 1;
        uint64_t bq = b->bb[2] | b->bb[4];
        uint64_t rq = b->bb[3] | b->bb[4];
        if (bq && (bish_attacks(sq,occ) & bq)) return 1;
        if (rq && (rook_attacks(sq,occ) & rq)) return 1;
    } else {
        if (wpawn_attacks_bb(sq_bb) & b->bb[6]) return 1;
        if (NATK[sq] & b->bb[7]) return 1;
        if (KATK[sq] & b->bb[11]) return 1;
        uint64_t bq = b->bb[8] | b->bb[10];
        uint64_t rq = b->bb[9] | b->bb[10];
        if (bq && (bish_attacks(sq,occ) & bq)) return 1;
        if (rq && (rook_attacks(sq,occ) & rq)) return 1;
    }
"""
    replace_exact(path, old, new, 1, a.variant)

print(f"applied {a.variant}")
