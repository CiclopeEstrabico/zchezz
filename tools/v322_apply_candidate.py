#!/usr/bin/env python3
"""Apply one isolated v3.22 NPS/search candidate to a copied v3.21 engine."""
from pathlib import Path
import argparse

VARIANTS = [
    "quiet_see",
    "premake_futility",
    "extcache16",
    "extcache64",
    "extra_fixedpoint",
]

p = argparse.ArgumentParser()
p.add_argument("variant", choices=VARIANTS)
p.add_argument("--root", default="engine/c/zchezz_v322cand")
a = p.parse_args()
root = Path(a.root)


def replace_exact(path: Path, old: str, new: str, expected: int, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    n = text.count(old)
    if n != expected:
        raise SystemExit(f"{label}: expected {expected} matches, found {n}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def ensure_quiet_direct_check(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "static inline int quiet_direct_check" in text:
        return
    marker = """    return gained[0] - sc;\n}\n\n\n/* ── Move scoring + sorting"""
    helper = """    return gained[0] - sc;\n}\n\n/* Cheap direct-check test ported from v4.02.  It intentionally detects\n * direct checks only; discovered checks remain outside this cheap prune gate. */\nstatic inline int quiet_direct_check(const Board *bd, int from, int to) {\n    int ksq = bd->turn == COL_W ? bd->bk : bd->wk;\n    uint64_t occ = bd->occ & ~((uint64_t)1 << from);\n    switch (PC_TYPE(bd->b[from])) {\n        case 1: return (bd->turn == COL_W ? wpawn_attacks_bb((uint64_t)1 << to)\n                                          : bpawn_attacks_bb((uint64_t)1 << to)) >> ksq & 1;\n        case 2: return (NATK[to] >> ksq) & 1;\n        case 3: return (bish_attacks(to, occ) >> ksq) & 1;\n        case 4: case 5:\n            return ((rook_attacks(to, occ) | bish_attacks(to, occ)) >> ksq) & 1;\n        default: return 0;\n    }\n}\n\n\n/* ── Move scoring + sorting"""
    if marker not in text:
        raise SystemExit("quiet_direct_check insertion marker not found")
    path.write_text(text.replace(marker, helper, 1), encoding="utf-8")


if a.variant == "quiet_see":
    path = root / "search.c"
    replace_exact(
        path,
        """static int see_board(const Board *bd, int from, int to, int is_epc) {\n    if (!bd->b[to] && !is_epc) return 0;\n\n    int gained[32];""",
        """static int see_board(const Board *bd, int from, int to, int is_epc) {\n    /* v3.22 candidate: empty target means a zero-valued capture, allowing\n     * SEE to measure whether a quiet move simply hangs the moved piece. */\n    int gained[32];""",
        1,
        a.variant,
    )
    ensure_quiet_direct_check(path)
    marker = """                /* Skip singular move */\n                if (ss->sing_from[ply]>=0 && mfr==ss->sing_from[ply] && mto==ss->sing_to[ply]) continue;\n\n                board_make(b, m);"""
    new = """                /* v4.02 quiet SEE pruning: historical result was ~24% fewer\n                 * nodes with a small positive Elo signal. Killers, counters and\n                 * direct checks are exempt. */\n                if (!in_check && !is_pv && legal_count > 0 && depth <= 3 &&\n                    !is_killer &&\n                    !(cur_prev_ft >= 0 && ss->counter_move[cur_prev_ft] == (mfr*64+mto)) &&\n                    !quiet_direct_check(b, mfr, mto)) {\n                    if (see_board(b, mfr, mto, 0) < -(depth * 60)) continue;\n                }\n\n                /* Skip singular move */\n                if (ss->sing_from[ply]>=0 && mfr==ss->sing_from[ply] && mto==ss->sing_to[ply]) continue;\n\n                board_make(b, m);"""
    replace_exact(path, marker, new, 1, a.variant)

elif a.variant == "premake_futility":
    path = root / "search.c"
    ensure_quiet_direct_check(path)
    old = """                /* Skip singular move */\n                if (ss->sing_from[ply]>=0 && mfr==ss->sing_from[ply] && mto==ss->sing_to[ply]) continue;\n\n                board_make(b, m);"""
    new = """                /* Skip singular move */\n                if (ss->sing_from[ply]>=0 && mfr==ss->sing_from[ply] && mto==ss->sing_to[ply]) continue;\n\n                /* v4.02 pre-make futility: prune hopeless non-checking quiets\n                 * before paying board_make + NNUE push + legality + unmake. */\n                if (!in_check && is_quiet && depth>=1 && depth<=8 && legal_count>0 &&\n                    !quiet_direct_check(b, mfr, mto)) {\n                    int fd = depth < 9 ? depth : 8;\n                    if (static_eval + fut_base[fd] + fut_adj <= alpha) continue;\n                }\n\n                board_make(b, m);"""
    replace_exact(path, old, new, 1, a.variant)
    old_post = """                /* Futility pruning */\n                if (!in_check && is_quiet && !gives_check && depth>=1 && depth<=8 && legal_count>1) {\n                    int fd = depth < 9 ? depth : 8;\n                    if (static_eval + fut_base[fd] + fut_adj <= alpha) { board_unmake(b); continue; }\n                }\n\n"""
    replace_exact(path, old_post, "", 1, a.variant)

elif a.variant in ("extcache16", "extcache64"):
    path = root / "nnue.h"
    slots = "16" if a.variant == "extcache16" else "64"
    replace_exact(path, "#define EXT_CACHE_SLOTS 4", f"#define EXT_CACHE_SLOTS {slots}", 1, a.variant)

elif a.variant == "extra_fixedpoint":
    path = root / "nnue.c"
    text = path.read_text(encoding="utf-8")
    marker = """/* Per-thread ext cache (v3.13).\n * Uses NnueAccum's cache_key/cache_buf instead of global statics."""
    if marker not in text:
        raise SystemExit("extra_fixedpoint insertion marker not found")
    helper = r'''/* v3.22 candidate: integer/fixed-point form of the 31 extra features.
 * The existing projection quantizes every float as (int16_t)(feat*256).
 * Computing that quantized value directly removes float divides/multiplies on
 * every ext-cache miss while preserving the exact values for this feature set. */
static void _compute_extra_feat16_bb(int16_t feat[NN_EXTRA], const uint64_t bb[12], int stm) {
    static const int MAXCNT[6] = {8,2,2,2,1,1};
    static const int MATVAL[6] = {1,3,3,5,9,0};
    if (!_extra_masks_init) _init_extra_masks();
    int cw[6], cb[6];
    for (int t=0; t<6; t++) { cw[t]=__builtin_popcountll(bb[t]); cb[t]=__builtin_popcountll(bb[t+6]); }
    int *sc = stm==0 ? cw : cb, *oc = stm==0 ? cb : cw;
    for (int i=0;i<6;i++) feat[i]   = (int16_t)((sc[i]*256)/MAXCNT[i]);
    for (int i=0;i<6;i++) feat[6+i] = (int16_t)((oc[i]*256)/MAXCNT[i]);
    int mat=0; for (int i=0;i<6;i++) mat += (cw[i]+cb[i])*MATVAL[i];
    feat[12]=(int16_t)((mat*256)/78); feat[13]=256;
    for (int f=0;f<8;f++) { feat[14+f]=0; feat[22+f]=0; }
    uint64_t wp=bb[0], bp=bb[6], tmp;
    if (stm==0) {
        tmp=wp; while(tmp){int sq=__builtin_ctzll(tmp);tmp&=tmp-1;if(!(_pp_span_w[sq]&bp))feat[14+(sq&7)]=256;}
        tmp=bp; while(tmp){int sq=__builtin_ctzll(tmp);tmp&=tmp-1;if(!(_pp_span_b[sq]&wp))feat[22+(sq&7)]=256;}
    } else {
        tmp=bp; while(tmp){int sq=__builtin_ctzll(tmp);tmp&=tmp-1;if(!(_pp_span_b[sq]&wp))feat[14+(sq&7)]=256;}
        tmp=wp; while(tmp){int sq=__builtin_ctzll(tmp);tmp&=tmp-1;if(!(_pp_span_w[sq]&bp))feat[22+(sq&7)]=256;}
    }
    int wk=bb[5]?__builtin_ctzll(bb[5]):-1, bk=bb[11]?__builtin_ctzll(bb[11]):-1;
    if (wk>=0 && bk>=0) {
        int df=(wk&7)-(bk&7); if(df<0)df=-df; int dr=(wk>>3)-(bk>>3); if(dr<0)dr=-dr;
        int d=df>dr?df:dr; feat[30]=(int16_t)((d*256)/7);
    } else feat[30]=0;
}

static void _project_feat_full16(int16_t *out, const int16_t feat[NN_EXTRA]) {
    memset(out, 0, NN_L1_OUT*sizeof(int16_t));
    for (int j=0;j<NN_EXTRA;j++) if (feat[j]) _project_feat_add(out, j, feat[j]);
}

'''
    text = text.replace(marker, helper + marker, 1)
    old = """        float feat[NN_EXTRA];\n        _compute_extra_feat_bb(feat, bb, stm);\n        int16_t *buf = na->cache_buf[slot];\n        memset(buf, 0, NN_L1_OUT * sizeof(int16_t));\n        _project_feat_full(buf, feat);"""
    new = """        int16_t feat16[NN_EXTRA];\n        _compute_extra_feat16_bb(feat16, bb, stm);\n        int16_t *buf = na->cache_buf[slot];\n        _project_feat_full16(buf, feat16);"""
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"extra_fixedpoint cache-miss block: expected 1, found {n}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")

print(f"applied {a.variant}")
