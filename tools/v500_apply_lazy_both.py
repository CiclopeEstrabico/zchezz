#!/usr/bin/env python3
"""Add a common-path fused materializer for both clean HalfKP perspectives.

The promoted v5 lazy accumulator currently walks the same ancestor chain twice
at each evaluated node: once for White POV and once for Black POV.  In the
normal case both perspectives have identical validity state, so replay both in
one pass and let _acc_add/sub_piece update both rows together.  Dirty king-
bucket crossings and asymmetric states retain the existing per-perspective
fallback, so semantics are unchanged.
"""
from pathlib import Path
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--root', default='engine/c/zchezz_v500_lazyboth')
a = ap.parse_args()
p = Path(a.root) / 'nnue.c'
s = p.read_text(encoding='utf-8')

needle = '''static inline void _ensure_lazy_perspective(NnueAccum *na, int white_pov) {
    int p=na->acc_ptr;
    uint8_t *valid = white_pov ? na->acc_valid_w : na->acc_valid_b;
    uint8_t *dirty = white_pov ? na->dirty_w_stack : na->dirty_b_stack;
    if (valid[p] || dirty[p]) return;

    int q=p;
    while (q > 0 && !valid[q] && !dirty[q]) q--;
    /* A dirty ancestor can only propagate to p until an eval refreshes it.
     * Therefore clean p implies that q is a valid materialized ancestor. */
    if (!valid[q]) return;

    const int16_t *L1WT=na->net->L1WT;
    for (int d=q+1; d<=p; d++) {
        int bw=na->bucket_w_stack[d], bb=na->bucket_b_stack[d];
        if (white_pov)
            memcpy(na->acc_stack_w[d],na->acc_stack_w[d-1],NN_L1_OUT*sizeof(int16_t));
        else
            memcpy(na->acc_stack_b[d],na->acc_stack_b[d-1],NN_L1_OUT*sizeof(int16_t));
        for (int k=0;k<na->delta_n[d];k++) {
            uint8_t pc=na->delta_piece[d][k]; int sq=na->delta_sq[d][k];
            int add=na->delta_sign[d][k] > 0;
            if (add)
                _acc_add_piece(na->acc_stack_w[d],na->acc_stack_b[d],L1WT,pc,sq,bw,bb,white_pov,!white_pov);
            else
                _acc_sub_piece(na->acc_stack_w[d],na->acc_stack_b[d],L1WT,pc,sq,bw,bb,white_pov,!white_pov);
        }
        valid[d]=1;
    }
}
'''
if s.count(needle) != 1:
    raise SystemExit(f'expected one lazy helper, found {s.count(needle)}')

both = needle + r'''

/* Fast common path: after a normal eval, White and Black validity advance
 * together.  When both current perspectives are clean and equally valid,
 * walk the ancestor chain once, copy both 512-wide accumulators together,
 * and replay every piece delta once with updW=updB=1.  Any asymmetric or
 * dirty state falls back to _ensure_lazy_perspective in the caller. */
static inline int _ensure_lazy_both_clean(NnueAccum *na) {
    int p=na->acc_ptr;
    if (na->dirty_w_stack[p] || na->dirty_b_stack[p]) return 0;
    if (na->acc_valid_w[p] && na->acc_valid_b[p]) return 1;
    if (na->acc_valid_w[p] != na->acc_valid_b[p]) return 0;

    int q=p;
    while (q > 0 && !na->acc_valid_w[q] && !na->acc_valid_b[q] &&
           !na->dirty_w_stack[q] && !na->dirty_b_stack[q]) q--;
    if (!na->acc_valid_w[q] || !na->acc_valid_b[q] ||
        na->dirty_w_stack[q] || na->dirty_b_stack[q]) return 0;

    /* A clean descendant cannot legally have a dirty frame in this span;
     * check first so the function never partially materializes then falls
     * back. */
    for (int d=q+1; d<=p; d++)
        if (na->dirty_w_stack[d] || na->dirty_b_stack[d]) return 0;

    const int16_t *L1WT=na->net->L1WT;
    for (int d=q+1; d<=p; d++) {
        memcpy(na->acc_stack_w[d],na->acc_stack_w[d-1],NN_L1_OUT*sizeof(int16_t));
        memcpy(na->acc_stack_b[d],na->acc_stack_b[d-1],NN_L1_OUT*sizeof(int16_t));
        int bw=na->bucket_w_stack[d], bb=na->bucket_b_stack[d];
        for (int k=0;k<na->delta_n[d];k++) {
            uint8_t pc=na->delta_piece[d][k];
            int sq=na->delta_sq[d][k];
            if (na->delta_sign[d][k] > 0)
                _acc_add_piece(na->acc_stack_w[d],na->acc_stack_b[d],L1WT,pc,sq,bw,bb,1,1);
            else
                _acc_sub_piece(na->acc_stack_w[d],na->acc_stack_b[d],L1WT,pc,sq,bw,bb,1,1);
        }
        na->acc_valid_w[d]=1;
        na->acc_valid_b[d]=1;
    }
    return 1;
}
'''
s = s.replace(needle, both)

old_forward = '''    /* Step 0a: materialize clean lazy chains only if this node evaluates. */
    if (!na->dirty_w_stack[ptr]) _ensure_lazy_perspective(na, 1);
    if (!na->dirty_b_stack[ptr]) _ensure_lazy_perspective(na, 0);
    /* Step 0b: a king-bucket crossing is still rebuilt exactly from the
'''
new_forward = '''    /* Step 0a: fuse the overwhelmingly common case where both clean
     * perspectives share the same lazy chain.  Preserve the original helper
     * as the exact fallback for dirty/asymmetric states. */
    if (!na->dirty_w_stack[ptr] && !na->dirty_b_stack[ptr]) {
        if (!_ensure_lazy_both_clean(na)) {
            _ensure_lazy_perspective(na, 1);
            _ensure_lazy_perspective(na, 0);
        }
    } else {
        if (!na->dirty_w_stack[ptr]) _ensure_lazy_perspective(na, 1);
        if (!na->dirty_b_stack[ptr]) _ensure_lazy_perspective(na, 0);
    }
    /* Step 0b: a king-bucket crossing is still rebuilt exactly from the
'''
if s.count(old_forward) != 1:
    raise SystemExit(f'expected one forward lazy block, found {s.count(old_forward)}')
s = s.replace(old_forward, new_forward)
p.write_text(s, encoding='utf-8')
print('applied fused dual-perspective lazy materializer')
