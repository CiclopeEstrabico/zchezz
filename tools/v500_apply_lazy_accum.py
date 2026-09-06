#!/usr/bin/env python3
"""Materialize a lazy HalfKP accumulator candidate for Zchezz v5.

board_make records only the piece delta and bucket/dirty metadata.  A clean
perspective is materialized from the nearest valid ancestor only when the
forward pass actually needs it.  King-bucket crossings retain the existing
v5 semantics: that perspective is rebuilt directly from the current board.
No feature, weight, quantization, or search semantic is changed.
"""
from pathlib import Path
import argparse

ap=argparse.ArgumentParser(); ap.add_argument('--root',default='engine/c/zchezz_v500_lazy'); a=ap.parse_args()
root=Path(a.root)

def rep(path, old, new, n, label):
    s=path.read_text(encoding='utf-8'); c=s.count(old)
    if c != n: raise SystemExit(f'{label}: expected {n}, found {c}')
    path.write_text(s.replace(old,new),encoding='utf-8')

h=root/'nnue.h'
old='''    int      acc_ptr;
    int16_t  acc_w[NN_L1_OUT]                     __attribute__((aligned(32)));
    int16_t  acc_b[NN_L1_OUT]                     __attribute__((aligned(32)));
    uint8_t  bucket_w_stack[NN_ACC_STACK];
'''
new='''    int      acc_ptr;
    /* Lazy materialization metadata.  Each move needs at most four
     * non-king feature operations (castle rook: 2; EP/promotion/capture: <=3). */
    uint8_t  acc_valid_w[NN_ACC_STACK];
    uint8_t  acc_valid_b[NN_ACC_STACK];
    uint8_t  delta_n[NN_ACC_STACK];
    uint8_t  delta_piece[NN_ACC_STACK][4];
    uint8_t  delta_sq[NN_ACC_STACK][4];
    int8_t   delta_sign[NN_ACC_STACK][4];
    int16_t  acc_w[NN_L1_OUT]                     __attribute__((aligned(32)));
    int16_t  acc_b[NN_L1_OUT]                     __attribute__((aligned(32)));
    uint8_t  bucket_w_stack[NN_ACC_STACK];
'''
rep(h,old,new,1,'header')

c=root/'nnue.c'
old='''void nnue_reset(NnueAccum *na) {
    na->acc_dirty = 1;
    na->acc_ptr   = 0;
}'''
new='''void nnue_reset(NnueAccum *na) {
    na->acc_dirty = 1;
    na->acc_ptr   = 0;
    memset(na->acc_valid_w, 0, sizeof(na->acc_valid_w));
    memset(na->acc_valid_b, 0, sizeof(na->acc_valid_b));
}'''
rep(c,old,new,1,'reset')

old='''    na->dirty_w_stack[0]  = 0;
    na->dirty_b_stack[0]  = 0;
}'''
new='''    na->dirty_w_stack[0]  = 0;
    na->dirty_b_stack[0]  = 0;
    memset(na->acc_valid_w, 0, sizeof(na->acc_valid_w));
    memset(na->acc_valid_b, 0, sizeof(na->acc_valid_b));
    na->acc_valid_w[0] = 1;
    na->acc_valid_b[0] = 1;
    na->delta_n[0] = 0;
}'''
rep(c,old,new,1,'rebuild-valid')

# A bucket refresh is a full exact materialization of that perspective.
old='''        na->bucket_w_stack[ptr] = (uint8_t)bw;
        na->dirty_w_stack[ptr]  = 0;
    } else {'''
new='''        na->bucket_w_stack[ptr] = (uint8_t)bw;
        na->dirty_w_stack[ptr]  = 0;
        na->acc_valid_w[ptr]    = 1;
    } else {'''
rep(c,old,new,1,'refresh-w')
old='''        na->bucket_b_stack[ptr] = (uint8_t)bb;
        na->dirty_b_stack[ptr]  = 0;
    }
}'''
new='''        na->bucket_b_stack[ptr] = (uint8_t)bb;
        na->dirty_b_stack[ptr]  = 0;
        na->acc_valid_b[ptr]    = 1;
    }
}'''
rep(c,old,new,1,'refresh-b')

s=c.read_text(encoding='utf-8')
start=s.index('void nnue_push_na(NnueAccum *na, const uint8_t *board, const NNMove *m) {')
end=s.index('\n/* Pop restores the parent frame.', start)
oldfunc=s[start:end]
newfunc=r'''void nnue_push_na(NnueAccum *na, const uint8_t *board, const NNMove *m) {
    if (!na->net) { na->acc_dirty = 1; return; }

    int src = na->acc_ptr, dst = src + 1;
    if (dst >= NN_ACC_STACK) { na->acc_dirty = 1; return; }
    if (na->acc_dirty) { nnue_rebuild(na, board); src = 0; dst = 1; }

    int bw = na->bucket_w_stack[src], bb = na->bucket_b_stack[src];
    int dw = na->dirty_w_stack[src],  db = na->dirty_b_stack[src];
    int n = 0;
#define LDOP(pc_, sq_, sign_) do { \
        uint8_t _pc=(uint8_t)(pc_); \
        if (_pc && PC_TYPE(_pc) != PT_KING && n < 4) { \
            na->delta_piece[dst][n]=_pc; na->delta_sq[dst][n]=(uint8_t)(sq_); \
            na->delta_sign[dst][n]=(int8_t)(sign_); n++; \
        } \
    } while (0)

    if (m->castle) {
        const int *cs = _castle_sq[m->castle];
        int kf=cs[0], kt=cs[1], rf=cs[2], rt=cs[3];
        uint8_t rook=board[rf];
        LDOP(rook,rf,-1); LDOP(rook,rt,+1);
        if (PC_COLOR(board[kf]) == COL_W) {
            int nb=nnue_king_bucket_w(kt); if (nb != bw) { dw=1; bw=nb; }
        } else {
            int nb=nnue_king_bucket_b(kt); if (nb != bb) { db=1; bb=nb; }
        }
    } else {
        int f=m->from_sq, to=m->to_sq;
        uint8_t p=board[f], cap=board[to];
        if (PC_TYPE(p) == PT_KING) {
            if (cap) LDOP(cap,to,-1);
            if (PC_COLOR(p) == COL_W) {
                int nb=nnue_king_bucket_w(to); if (nb != bw) { dw=1; bw=nb; }
            } else {
                int nb=nnue_king_bucket_b(to); if (nb != bb) { db=1; bb=nb; }
            }
        } else {
            LDOP(p,f,-1);
            if (cap) LDOP(cap,to,-1);
            if (m->is_epc) {
                int e=(PC_COLOR(p)==COL_W)?to+8:to-8;
                if (board[e]) LDOP(board[e],e,-1);
            }
            uint8_t land=m->prom?(uint8_t)(PC_COLOR(p)|m->prom):p;
            LDOP(land,to,+1);
        }
    }
#undef LDOP

    na->delta_n[dst]=(uint8_t)n;
    na->bucket_w_stack[dst]=(uint8_t)bw;
    na->bucket_b_stack[dst]=(uint8_t)bb;
    na->dirty_w_stack[dst]=(uint8_t)dw;
    na->dirty_b_stack[dst]=(uint8_t)db;
    na->acc_valid_w[dst]=0;
    na->acc_valid_b[dst]=0;
    na->acc_ptr=dst;
}

/* Materialize one clean perspective from the nearest already-materialized
 * ancestor.  If a bucket crossing is pending, the existing refresh path
 * rebuilds from the current board instead and this helper intentionally
 * does nothing. */
static inline void _ensure_lazy_perspective(NnueAccum *na, int white_pov) {
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
s=s[:start]+newfunc+s[end:]
c.write_text(s,encoding='utf-8')

# Ensure both clean perspectives immediately before the existing dirty/bucket
# refresh logic reads current-frame accumulator memory.
s=c.read_text(encoding='utf-8')
old='''    int ptr = na->acc_ptr;
    /* Step 0: lazy bucket refresh (rare: only after a king move that
     * crossed a bucket border). */
    if (na->dirty_w_stack[ptr]) _refresh_perspective(na, board, 1);
    if (na->dirty_b_stack[ptr]) _refresh_perspective(na, board, 0);'''
new='''    int ptr = na->acc_ptr;
    /* Step 0a: materialize clean lazy chains only if this node evaluates. */
    if (!na->dirty_w_stack[ptr]) _ensure_lazy_perspective(na, 1);
    if (!na->dirty_b_stack[ptr]) _ensure_lazy_perspective(na, 0);
    /* Step 0b: a king-bucket crossing is still rebuilt exactly from the
     * current board, preserving the original v5 bucket semantics. */
    if (na->dirty_w_stack[ptr]) _refresh_perspective(na, board, 1);
    if (na->dirty_b_stack[ptr]) _refresh_perspective(na, board, 0);'''
rep(c,old,new,1,'forward-ensure')
print('applied lazy HalfKP accumulator candidate')
