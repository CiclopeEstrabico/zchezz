#!/usr/bin/env python3
"""Apply third-wave v3.22 board hot-path candidates."""
from pathlib import Path
import argparse
p=argparse.ArgumentParser();p.add_argument("variant",choices=["incremental_occ"]);p.add_argument("--root",default="engine/c/zchezz_v322cand");a=p.parse_args()
path=Path(a.root)/"board.c";s=path.read_text(encoding="utf-8")
old='''    /* Incremental occupancy update (Phase 4 v212B):
     * Instead of rebuilding from 12 ORs, compute occ from the old saved value.
     * The bb[] array has already been updated, so we can derive occ cheaply
     * by XOR-ing the from/to/capture bits. But it's even safer and simpler
     * to just compute from the 6 bb per side (only 5+5 ORs vs 12). */
    b->occ_w = b->bb[0]|b->bb[1]|b->bb[2]|b->bb[3]|b->bb[4]|b->bb[5];
    b->occ_b = b->bb[6]|b->bb[7]|b->bb[8]|b->bb[9]|b->bb[10]|b->bb[11];
    b->occ   = b->occ_w | b->occ_b;
'''
new='''    /* v3.22 candidate: true incremental occupancy update.  UndoFrame already
     * carries the exact parent occupancies, so update only the squares touched
     * by the move instead of OR-reducing all 12 piece bitboards every make. */
    uint64_t own = col==COL_W ? uf->occ_w : uf->occ_b;
    uint64_t opp = col==COL_W ? uf->occ_b : uf->occ_w;
    if (m->castle && m->castle<=4) {
        int kf=CASTLE_SQ[m->castle][0], kt=CASTLE_SQ[m->castle][1];
        int rf=CASTLE_SQ[m->castle][2], rt=CASTLE_SQ[m->castle][3];
        own &= ~(((uint64_t)1<<kf)|((uint64_t)1<<rf));
        own |=  (((uint64_t)1<<kt)|((uint64_t)1<<rt));
    } else {
        own &= ~((uint64_t)1<<f);
        own |=  ((uint64_t)1<<to);
        if (cap) opp &= ~((uint64_t)1<<to);
        if (m->epc) {
            int epsq=col==COL_W ? to+8 : to-8;
            opp &= ~((uint64_t)1<<epsq);
        }
    }
    if(col==COL_W){b->occ_w=own;b->occ_b=opp;}else{b->occ_b=own;b->occ_w=opp;}
    b->occ=b->occ_w|b->occ_b;
'''
if s.count(old)!=1: raise SystemExit(f"occupancy block matches={s.count(old)}")
path.write_text(s.replace(old,new),encoding="utf-8")
print("applied incremental_occ")
