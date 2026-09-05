#!/usr/bin/env python3
"""Apply second-wave v3.22 NNU3 hot-path candidates."""
from pathlib import Path
import argparse

VARIANTS = ["fused_quiet_push", "feature_state_cache16"]
p=argparse.ArgumentParser(); p.add_argument("variant", choices=VARIANTS); p.add_argument("--root", default="engine/c/zchezz_v322cand")
a=p.parse_args(); root=Path(a.root)

def rex(path, old, new, n, label):
    s=path.read_text(encoding="utf-8"); c=s.count(old)
    if c!=n: raise SystemExit(f"{label}: expected {n}, found {c}")
    path.write_text(s.replace(old,new), encoding="utf-8")

if a.variant == "fused_quiet_push":
    path=root/"nnue.c"
    marker="""/* ── Extra-feature (31 endgame features) computation ────────────── */"""
    helper=r'''/* v3.22 candidate: fuse accumulator copy + quiet piece delta.
 * Common non-capture/non-promotion quiets become one pass over the child
 * accumulator instead of memcpy + subtract pass + add pass. */
static inline void _acc_copy_quiet_move(int16_t *dstW, int16_t *dstB,
                                        const int16_t *srcW, const int16_t *srcB,
                                        uint8_t p, int from, int to) {
    int pt=piece_type_idx(p); if(pt<0){
        memcpy(dstW,srcW,NN_L1_OUT*sizeof(int16_t));
        memcpy(dstB,srcB,NN_L1_OUT*sizeof(int16_t)); return;
    }
    int isW=(PC_COLOR(p)==COL_W), coW=isW?0:6, coB=isW?6:0;
    const int16_t *fw=_nnL1WT+(coW*64+pt*64+(from^56))*NN_L1_OUT;
    const int16_t *tw=_nnL1WT+(coW*64+pt*64+(to^56))*NN_L1_OUT;
    const int16_t *fb=_nnL1WT+(coB*64+pt*64+from)*NN_L1_OUT;
    const int16_t *tb=_nnL1WT+(coB*64+pt*64+to)*NN_L1_OUT;
#ifdef __AVX2__
    _mm_prefetch((const char*)fw,_MM_HINT_T0); _mm_prefetch((const char*)tw,_MM_HINT_T0);
    _mm_prefetch((const char*)fb,_MM_HINT_T0); _mm_prefetch((const char*)tb,_MM_HINT_T0);
    for(int o=0;o<NN_L1_OUT;o+=16){
        __m256i w=_mm256_load_si256((const __m256i*)(srcW+o));
        __m256i b=_mm256_load_si256((const __m256i*)(srcB+o));
        w=_mm256_add_epi16(_mm256_sub_epi16(w,_mm256_load_si256((const __m256i*)(fw+o))),
                           _mm256_load_si256((const __m256i*)(tw+o)));
        b=_mm256_add_epi16(_mm256_sub_epi16(b,_mm256_load_si256((const __m256i*)(fb+o))),
                           _mm256_load_si256((const __m256i*)(tb+o)));
        _mm256_store_si256((__m256i*)(dstW+o),w); _mm256_store_si256((__m256i*)(dstB+o),b);
    }
#else
    for(int o=0;o<NN_L1_OUT;o++){
        dstW[o]=(int16_t)(srcW[o]-fw[o]+tw[o]);
        dstB[o]=(int16_t)(srcB[o]-fb[o]+tb[o]);
    }
#endif
}

'''
    rex(path, marker, helper+marker, 1, a.variant)
    old="""    memcpy(na->acc_stack_w[dst], na->acc_stack_w[src], NN_L1_OUT*sizeof(int16_t));
    memcpy(na->acc_stack_b[dst], na->acc_stack_b[src], NN_L1_OUT*sizeof(int16_t));
    int16_t *cW = na->acc_stack_w[dst], *cB = na->acc_stack_b[dst];
    if (m->castle) {
"""
    new="""    int16_t *cW = na->acc_stack_w[dst], *cB = na->acc_stack_b[dst];
    int f0=m->from_sq, t0=m->to_sq;
    uint8_t p0=board[f0], cap0=board[t0];
    if (!m->castle && !cap0 && !m->is_epc && !m->prom) {
        _acc_copy_quiet_move(cW,cB,na->acc_stack_w[src],na->acc_stack_b[src],p0,f0,t0);
    } else {
        memcpy(cW, na->acc_stack_w[src], NN_L1_OUT*sizeof(int16_t));
        memcpy(cB, na->acc_stack_b[src], NN_L1_OUT*sizeof(int16_t));
    }
    if (m->castle) {
"""
    rex(path, old, new, 1, a.variant)
    # Avoid applying the normal quiet delta a second time; special cases retain old logic.
    old2="""    } else {
        int f = m->from_sq, to = m->to_sq;
        uint8_t p = board[f], cap = board[to];
        _acc_sub_piece(cW, cB, p, f);
"""
    new2="""    } else if (cap0 || m->is_epc || m->prom) {
        int f = m->from_sq, to = m->to_sq;
        uint8_t p = board[f], cap = board[to];
        _acc_sub_piece(cW, cB, p, f);
"""
    rex(path, old2, new2, 1, a.variant)

elif a.variant == "feature_state_cache16":
    hp=root/"nnue.h"
    rex(hp,"#define EXT_CACHE_SLOTS 4","#define EXT_CACHE_SLOTS 16",1,a.variant)
    old="""    uint64_t cache_key[EXT_CACHE_SLOTS];
    int16_t  cache_buf[EXT_CACHE_SLOTS][NN_L1_OUT] __attribute__((aligned(32)));
"""
    new="""    uint64_t cache_key[EXT_CACHE_SLOTS];
    uint32_t cache_aux[EXT_CACHE_SLOTS];
    int16_t  cache_buf[EXT_CACHE_SLOTS][NN_L1_OUT] __attribute__((aligned(32)));
"""
    rex(hp,old,new,1,a.variant)

    cp=root/"nnue.c"
    marker="""/* Per-thread ext cache (v3.13).
 * Uses NnueAccum's cache_key/cache_buf instead of global statics."""
    helper=r'''/* Coarse NN_EXTRA state.  The 31 manual features depend only on piece
 * counts, passed-pawn files, side-to-move perspective, and king distance.
 * Cache this state rather than the full Zobrist position so ordinary piece
 * manoeuvres can reuse the expensive 31x256 projection. */
typedef struct { int cw[6], cb[6]; uint8_t pw, pb, kd; } ExtraState322;
static void _extra_state322(ExtraState322 *s, const uint64_t bb[12]) {
    if(!_extra_masks_init) _init_extra_masks();
    for(int t=0;t<6;t++){s->cw[t]=__builtin_popcountll(bb[t]);s->cb[t]=__builtin_popcountll(bb[t+6]);}
    s->pw=s->pb=0; uint64_t wp=bb[0],bp=bb[6],tmp=wp;
    while(tmp){int q=__builtin_ctzll(tmp);tmp&=tmp-1;if(!(_pp_span_w[q]&bp))s->pw|=(uint8_t)(1u<<(q&7));}
    tmp=bp; while(tmp){int q=__builtin_ctzll(tmp);tmp&=tmp-1;if(!(_pp_span_b[q]&wp))s->pb|=(uint8_t)(1u<<(q&7));}
    if(bb[5]&&bb[11]){int w=__builtin_ctzll(bb[5]),b=__builtin_ctzll(bb[11]);int df=(w&7)-(b&7);if(df<0)df=-df;int dr=(w>>3)-(b>>3);if(dr<0)dr=-dr;s->kd=(uint8_t)(df>dr?df:dr);} else s->kd=0;
}
static uint64_t _extra_key322(const ExtraState322 *s) {
    uint64_t k=0; int sh=0;
    for(int i=0;i<6;i++,sh+=5) k|=((uint64_t)(s->cw[i]&31))<<sh;
    for(int i=0;i<6;i++,sh+=5) k|=((uint64_t)(s->cb[i]&31))<<sh;
    return k;
}
static void _extra_feat322(float *f,const ExtraState322 *s,int stm){
    static const float MC[6]={8.f,2.f,2.f,2.f,1.f,1.f}, MV[6]={1.f,3.f,3.f,5.f,9.f,0.f};
    const int *a=stm==0?s->cw:s->cb,*o=stm==0?s->cb:s->cw;
    for(int i=0;i<6;i++){f[i]=a[i]/MC[i];f[6+i]=o[i]/MC[i];}
    float mat=0.f;for(int i=0;i<6;i++)mat+=(s->cw[i]+s->cb[i])*MV[i];f[12]=mat/78.f;f[13]=1.f;
    uint8_t ap=stm==0?s->pw:s->pb,op=stm==0?s->pb:s->pw;
    for(int i=0;i<8;i++){f[14+i]=(ap>>i)&1;f[22+i]=(op>>i)&1;}
    f[30]=(float)s->kd/7.f;
}

'''
    rex(cp,marker,helper+marker,1,a.variant)
    oldblock="""    /* Cache key: mix hash with stm so White/Black perspectives are separate slots */
    uint64_t key = board_hash ^ ((uint64_t)stm * 0x9e3779b97f4a7c15ULL);
    int slot = (int)(key & (EXT_CACHE_SLOTS - 1));

    const int16_t *ext;
    if (na->cache_key[slot] != key) {
        /* Cache miss — compute extra features using precomputed bb[], project */
        float feat[NN_EXTRA];
        _compute_extra_feat_bb(feat, bb, stm);
        int16_t *buf = na->cache_buf[slot];
        memset(buf, 0, NN_L1_OUT * sizeof(int16_t));
        _project_feat_full(buf, feat);
        na->cache_key[slot] = key;
        ext = buf;
    } else {
        /* Cache hit — reuse projected buffer */
        ext = na->cache_buf[slot];
    }
"""
    newblock="""    ExtraState322 es; _extra_state322(&es, bb);
    uint64_t key=_extra_key322(&es);
    uint32_t aux=(uint32_t)es.pw | ((uint32_t)es.pb<<8) | ((uint32_t)es.kd<<16) | ((uint32_t)stm<<20);
    int slot=(int)((key ^ ((uint64_t)aux*0x9e3779b97f4a7c15ULL)) & (EXT_CACHE_SLOTS-1));
    const int16_t *ext;
    if (na->cache_key[slot] != key || na->cache_aux[slot] != aux) {
        float feat[NN_EXTRA]; _extra_feat322(feat,&es,stm);
        int16_t *buf=na->cache_buf[slot]; _project_feat_full(buf,feat);
        na->cache_key[slot]=key; na->cache_aux[slot]=aux; ext=buf;
    } else ext=na->cache_buf[slot];
"""
    rex(cp,oldblock,newblock,1,a.variant)
    # aux is irrelevant when key mismatches; zero it for deterministic state anyway.
    oldreset="""    memset(na->cache_key, 0, sizeof(na->cache_key));"""
    newreset="""    memset(na->cache_key, 0, sizeof(na->cache_key));
    memset(na->cache_aux, 0, sizeof(na->cache_aux));"""
    rex(cp,oldreset,newreset,1,a.variant)

print(f"applied {a.variant}")
