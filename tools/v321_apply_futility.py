#!/usr/bin/env python3
"""Apply the v3.21 pre-make futility candidate to a disposable source copy."""
from pathlib import Path
import argparse

p = argparse.ArgumentParser()
p.add_argument("--root", default="engine/c/zchezz_v321cand")
a = p.parse_args()
path = Path(a.root) / "search.c"
text = path.read_text(encoding="utf-8")

anchor = "\n\n/* ── Move scoring + sorting ──────────────────────────────────── */"
helper = r'''

/* v3.21 candidate: cheap direct-check detector for pre-make futility.
 * It deliberately protects direct checks without paying board_make/unmake.
 * Discovered checks are not detected; that behavioral delta is why this
 * optimization must pass an Elo gate rather than being assumed neutral. */
static inline int quiet_direct_check(const Board *bd, int from, int to) {
    int ksq = bd->turn == COL_W ? bd->bk : bd->wk;
    uint64_t occ = bd->occ & ~((uint64_t)1 << from);
    switch (PC_TYPE(bd->b[from])) {
        case 1: return (bd->turn == COL_W ? wpawn_attacks_bb((uint64_t)1 << to)
                                          : bpawn_attacks_bb((uint64_t)1 << to)) >> ksq & 1;
        case 2: return (NATK[to] >> ksq) & 1;
        case 3: return (bish_attacks(to, occ) >> ksq) & 1;
        case 4:
        case 5: return ((rook_attacks(to, occ) | bish_attacks(to, occ)) >> ksq) & 1;
        default: return 0;
    }
}
'''
if text.count(anchor) != 1:
    raise SystemExit(f"helper anchor expected 1 match, found {text.count(anchor)}")
text = text.replace(anchor, helper + anchor, 1)

old = '''                /* Skip singular move */
                if (ss->sing_from[ply]>=0 && mfr==ss->sing_from[ply] && mto==ss->sing_to[ply]) continue;

                board_make(b, m);
'''
new = '''                /* Skip singular move */
                if (ss->sing_from[ply]>=0 && mfr==ss->sing_from[ply] && mto==ss->sing_to[ply]) continue;

                /* v3.21 candidate: same futility margin, but before board_make.
                 * legal_count>0 corresponds to the old post-make legal_count>1.
                 * Direct checks are exempted through quiet_direct_check(). */
                if (!in_check && is_quiet && depth>=1 && depth<=8 && legal_count>0 &&
                    !quiet_direct_check(b, mfr, mto)) {
                    int fd = depth < 9 ? depth : 8;
                    if (static_eval + fut_base[fd] + fut_adj <= alpha) continue;
                }

                board_make(b, m);
'''
if text.count(old) != 1:
    raise SystemExit(f"pre-make insertion expected 1 match, found {text.count(old)}")
text = text.replace(old, new, 1)

old = '''                /* Futility pruning */
                if (!in_check && is_quiet && !gives_check && depth>=1 && depth<=8 && legal_count>1) {
                    int fd = depth < 9 ? depth : 8;
                    if (static_eval + fut_base[fd] + fut_adj <= alpha) { board_unmake(b); continue; }
                }

'''
if text.count(old) != 1:
    raise SystemExit(f"old futility block expected 1 match, found {text.count(old)}")
text = text.replace(old, "", 1)
path.write_text(text, encoding="utf-8")
print("applied futility_pre_make candidate")
