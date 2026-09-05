#!/usr/bin/env python3
"""Apply lazy NNUE accumulator materialization candidate for v3.22."""
from pathlib import Path
import argparse
p=argparse.ArgumentParser();p.add_argument("--root",default="engine/c/zchezz_v322cand");a=p.parse_args();root=Path(a.root)

def rep(path,old,new,n,label):
 s=path.read_text(encoding='utf-8');c=s.count(old)
 if c!=n:raise SystemExit(f'{label}: expected {n}, found {c}')
 path.write_text(s.replace(old,new),encoding='utf-8')

h=root/'nnue.h'
old='''    int      acc_ptr;
    int16_t  acc_w[NN_L1_OUT]                __attribute__((aligned(32)));
'''
new='''    int      acc_ptr;
    uint8_t  acc_valid[NN_ACC_STACK];
    uint8_t  delta_n[NN_ACC_STACK];
    uint8_t  delta_piece[NN_ACC_STACK][4];
    uint8_t  delta_sq[NN_ACC_STACK][4];
    int8_t   delta_sign[NN_ACC_STACK][4];
    int16_t  acc_w[NN_L1_OUT]                __attribute__((aligned(32)));
'''
rep(h,old,new,1,'header')

c=root/'nnue.c'
old='''    na->acc_ptr   = 0;
    na->ext_dirty[0] = 1;
'''
new='''    na->acc_ptr   = 0;
    memset(na->acc_valid, 0, sizeof(na->acc_valid));
    na->ext_dirty[0] = 1;
'''
rep(c,old,new,1,'reset')
old='''    memcpy(na->acc_stack_w[0], dW, NN_L1_OUT*sizeof(int16_t));
    memcpy(na->acc_stack_b[0], dB, NN_L1_OUT*sizeof(int16_t));
    /* Seed extra-feature arrays'''
new='''    memcpy(na->acc_stack_w[0], dW, NN_L1_OUT*sizeof(int16_t));
    memcpy(na->acc_stack_b[0], dB, NN_L1_OUT*sizeof(int16_t));
    memset(na->acc_valid, 0, sizeof(na->acc_valid));
    na->acc_valid[0] = 1;
    /* Seed extra-feature arrays'''
rep(c,old,new,1,'rebuild-valid')

start=c.read_text(encoding='utf-8')
oldfunc='''void nnue_push_na(NnueAccum *na, const uint8_t *board, const NNMove *m) {
    int src = na->acc_ptr, dst = src + 1;
    if (dst >= NN_ACC_STACK) { na->acc_dirty = 1; return; }
    if (na->acc_dirty) { nnue_rebuild(na, board); na->acc_dirty = 0; src = 0; dst = 1; }
    memcpy(na->acc_stack_w[dst], na->acc_stack_w[src], NN_L1_OUT*sizeof(int16_t));
    memcpy(na->acc_stack_b[dst], na->acc_stack_b[src], NN_L1_OUT*sizeof(int16_t));
    int16_t *cW = na->acc_stack_w[dst], *cB = na->acc_stack_b[dst];
    if (m->castle) {
        const int *sq = _castle_sq[m->castle];
        _acc_sub_piece(cW, cB, board[sq[0]], sq[0]);
        _acc_add_piece(cW, cB, board[sq[0]], sq[1]);
        _acc_sub_piece(cW, cB, board[sq[2]], sq[2]);
        _acc_add_piece(cW, cB, board[sq[2]], sq[3]);
    } else {
        int f = m->from_sq, to = m->to_sq;
        uint8_t p = board[f], cap = board[to];
        _acc_sub_piece(cW, cB, p, f);
        if (cap) _acc_sub_piece(cW, cB, cap, to);
        if (m->is_epc) {
            int epsq = (PC_COLOR(p) == COL_W) ? to + 8 : to - 8;
            if (board[epsq]) _acc_sub_piece(cW, cB, board[epsq], epsq);
        }
        uint8_t landing = m->prom ? (uint8_t)(PC_COLOR(p) | m->prom) : p;
        _acc_add_piece(cW, cB, landing, to);
    }
    na->acc_ptr = dst;
    na->ext_dirty[0] = 1;
    na->ext_dirty[1] = 1;
}
'''
newfunc='''void nnue_push_na(NnueAccum *na, const uint8_t *board, const NNMove *m) {
    int src=na->acc_ptr, dst=src+1;
    if(dst>=NN_ACC_STACK){na->acc_dirty=1;return;}
    if(na->acc_dirty){nnue_rebuild(na,board);na->acc_dirty=0;src=0;dst=1;}
    int n=0;
#define LDOP(pc,sqv,sgn) do { if((pc) && n<4){na->delta_piece[dst][n]=(uint8_t)(pc);na->delta_sq[dst][n]=(uint8_t)(sqv);na->delta_sign[dst][n]=(int8_t)(sgn);n++;} } while(0)
    if(m->castle){
        const int *sq=_castle_sq[m->castle];
        LDOP(board[sq[0]],sq[0],-1); LDOP(board[sq[0]],sq[1],+1);
        LDOP(board[sq[2]],sq[2],-1); LDOP(board[sq[2]],sq[3],+1);
    } else {
        int f=m->from_sq,to=m->to_sq; uint8_t pc=board[f],cap=board[to];
        LDOP(pc,f,-1); if(cap)LDOP(cap,to,-1);
        if(m->is_epc){int e=(PC_COLOR(pc)==COL_W)?to+8:to-8;if(board[e])LDOP(board[e],e,-1);}
        uint8_t land=m->prom?(uint8_t)(PC_COLOR(pc)|m->prom):pc; LDOP(land,to,+1);
    }
#undef LDOP
    na->delta_n[dst]=(uint8_t)n; na->acc_valid[dst]=0; na->acc_ptr=dst;
    na->ext_dirty[0]=1;na->ext_dirty[1]=1;
}

/* Materialize only when evaluation actually needs the child accumulator.
 * TT/draw/TB cutoffs can therefore make/unmake without touching 1 KB of HM
 * accumulator state.  Chains of unmaterialized check-evasion nodes are
 * resolved from the nearest valid ancestor in order. */
static inline void _ensure_acc322(NnueAccum *na) {
    int p=na->acc_ptr; if(na->acc_valid[p]) return;
    int q=p; while(q>0 && !na->acc_valid[q]) q--;
    for(int d=q+1;d<=p;d++){
        memcpy(na->acc_stack_w[d],na->acc_stack_w[d-1],NN_L1_OUT*sizeof(int16_t));
        memcpy(na->acc_stack_b[d],na->acc_stack_b[d-1],NN_L1_OUT*sizeof(int16_t));
        int16_t *w=na->acc_stack_w[d],*b=na->acc_stack_b[d];
        for(int k=0;k<na->delta_n[d];k++){
            uint8_t pc=na->delta_piece[d][k];int sq=na->delta_sq[d][k];
            if(na->delta_sign[d][k]>0)_acc_add_piece(w,b,pc,sq);else _acc_sub_piece(w,b,pc,sq);
        }
        na->acc_valid[d]=1;
    }
}
'''
if start.count(oldfunc)!=1: raise SystemExit(f'push function matches={start.count(oldfunc)}')
c.write_text(start.replace(oldfunc,newfunc),encoding='utf-8')

# Both forward paths must materialize before reading acc_stack.
s=c.read_text(encoding='utf-8')
old='''    if (na->acc_dirty) return 0;  /* v3.14: per-thread dirty flag */

    /* Step 1: compute extra feature values'''
new='''    if (na->acc_dirty) return 0;  /* v3.14: per-thread dirty flag */
    _ensure_acc322(na);

    /* Step 1: compute extra feature values'''
if s.count(old)!=1: raise SystemExit(f'eval ensure matches={s.count(old)}')
s=s.replace(old,new,1)
old2='''    if (na->acc_dirty) return 0;  /* v3.14: per-thread dirty flag */

    /* Cache key: mix hash'''
new2='''    if (na->acc_dirty) return 0;  /* v3.14: per-thread dirty flag */
    _ensure_acc322(na);

    /* Cache key: mix hash'''
if s.count(old2)!=1: raise SystemExit(f'eval_bb ensure matches={s.count(old2)}')
c.write_text(s.replace(old2,new2,1),encoding='utf-8')
print('applied lazy_acc')
