#!/usr/bin/env python3
"""Apply final v3.21 confirmation bundles to a disposable engine copy."""
from pathlib import Path
import argparse, math

VARIANTS = ["strong_no_qs", "strong_bundle", "strong_plus_prefetch"]
p=argparse.ArgumentParser(); p.add_argument('variant', choices=VARIANTS); p.add_argument('--root',default='engine/c/zchezz_v321cand'); a=p.parse_args()
root=Path(a.root)

def repl(path, old, new, n=1):
    path=Path(path); s=path.read_text(encoding='utf-8'); c=s.count(old)
    if c!=n: raise SystemExit(f'{a.variant}: expected {n} matches for {old!r}, got {c}')
    path.write_text(s.replace(old,new,n),encoding='utf-8')

search=root/'search.c'; board=root/'board.c'; nnue=root/'nnue.c'

# Search bundle: the only interaction bundle with positive timed + fixed-node signs.
repl(search,'if (static_eval - beta > 200) R += 1;','if (static_eval - beta > 134) R += 1;')
repl(search,'double v = log((double)d) * log((double)m) / 1.5;','double v = log((double)d) * log((double)m) / 1.35;')
repl(search,'static const int lmp_limit[8] = {0,10,18,26,36,48,62,78};','static const int lmp_limit[8] = {0,12,22,32,44,58,74,94};')
repl(search,'if (!pv_move.from && !pv_move.to && depth>=4 && !in_check && depth-1>=2) depth--;','if (!pv_move.from && !pv_move.to && depth>=3 && !in_check && depth-1>=2) depth--;')

# Tree-preserving board attack ordering.
old_attack='''    if (by == COL_W) {\n        /* White pawn attacks: bitboard shift (Phase 2) */\n        if (bpawn_attacks_bb(sq_bb) & b->bb[0]) return 1;  /* sq attacked by WP */\n        if (NATK[sq] & b->bb[1])  return 1;   /* white knight */\n        if (bish_attacks(sq,occ) & (b->bb[2]|b->bb[4])) return 1;  /* WB|WQ */\n        if (rook_attacks(sq,occ) & (b->bb[3]|b->bb[4])) return 1;  /* WR|WQ */\n        if (KATK[sq] & b->bb[5])  return 1;   /* white king */\n    } else {\n        /* Black pawn attacks: bitboard shift (Phase 2) */\n        if (wpawn_attacks_bb(sq_bb) & b->bb[6]) return 1;  /* sq attacked by BP */\n        if (NATK[sq] & b->bb[7])  return 1;\n        if (bish_attacks(sq,occ) & (b->bb[8]|b->bb[10])) return 1; /* BB|BQ */\n        if (rook_attacks(sq,occ) & (b->bb[9]|b->bb[10])) return 1; /* BR|BQ */\n        if (KATK[sq] & b->bb[11]) return 1;\n    }\n'''
new_attack='''    if (by == COL_W) {\n        if (bpawn_attacks_bb(sq_bb) & b->bb[0]) return 1;\n        if (NATK[sq] & b->bb[1]) return 1;\n        if (KATK[sq] & b->bb[5]) return 1;\n        uint64_t bq = b->bb[2] | b->bb[4], rq = b->bb[3] | b->bb[4];\n        if (bq && (bish_attacks(sq,occ) & bq)) return 1;\n        if (rq && (rook_attacks(sq,occ) & rq)) return 1;\n    } else {\n        if (wpawn_attacks_bb(sq_bb) & b->bb[6]) return 1;\n        if (NATK[sq] & b->bb[7]) return 1;\n        if (KATK[sq] & b->bb[11]) return 1;\n        uint64_t bq = b->bb[8] | b->bb[10], rq = b->bb[9] | b->bb[10];\n        if (bq && (bish_attacks(sq,occ) & bq)) return 1;\n        if (rq && (rook_attacks(sq,occ) & rq)) return 1;\n    }\n'''
repl(board,old_attack,new_attack)

if a.variant in ('strong_bundle','strong_plus_prefetch'):
    old='''        if (sc > qs_best) { qs_best = sc; best_move_qs = moves[i]; }\n        if (sc >= beta) {\n            return beta;\n        }\n'''
    new='''        if (sc > qs_best) { qs_best = sc; best_move_qs = moves[i]; }\n        if (sc >= beta) {\n            if (!ss->time_up) tt_store(b->hash, qs_best, 0, TT_LOWER, &best_move_qs, ply, stand);\n            return beta;\n        }\n'''
    repl(search,old,new)
    old='''    /* Don't store non-cutoff qsearch results — they pollute the TT\n     * with depth-0 entries that displace more valuable deeper entries */\n\n    return alpha;\n'''
    new='''    /* Cache searched-capture improvements, but never pure stand-pat entries. */\n    if (qs_best > stand && !ss->time_up) {\n        int from_move = best_move_qs.from || best_move_qs.to;\n        tt_store(b->hash, qs_best, 0,\n                 (from_move && qs_best > qs_orig_alpha) ? TT_EXACT : TT_UPPER,\n                 from_move ? &best_move_qs : NULL, ply, stand);\n    }\n\n    return alpha;\n'''
    repl(search,old,new)

if a.variant=='strong_plus_prefetch':
    old='''    const int16_t *bRow = _nnL1WT + (coB*64 + pt*64 + sq  ) * NN_L1_OUT;\n#ifdef __AVX2__\n    for (int o = 0; o < NN_L1_OUT; o += 16) {\n'''
    new='''    const int16_t *bRow = _nnL1WT + (coB*64 + pt*64 + sq  ) * NN_L1_OUT;\n#ifdef __AVX2__\n    _mm_prefetch((const char*)wRow, _MM_HINT_T0);\n    _mm_prefetch((const char*)bRow, _MM_HINT_T0);\n    for (int o = 0; o < NN_L1_OUT; o += 16) {\n'''
    repl(nnue,old,new,2)
print('applied',a.variant)
