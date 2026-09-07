#!/usr/bin/env python3
from pathlib import Path

p = Path('engine/c/zchezz_v323/search.c')
s = p.read_text(encoding='utf-8')

def rep(old, new):
    global s
    n = s.count(old)
    if n != 1:
        raise SystemExit(f'anchor count={n}: {old}')
    s = s.replace(old, new, 1)

# Excluded singular verification must not probe the normal-position TT.
rep('    tte_hit = tt_probe(b->hash, ply, &tte);',
    '    if (ss->sing_from[ply] < 0) tte_hit = tt_probe(b->hash, ply, &tte);')

# No tablebase result/caching while the best move is artificially excluded.
rep('    if (ply > 0 && !is_pv_early && b->hm == 0) {',
    '    if (ss->sing_from[ply] < 0 && ply > 0 && !is_pv_early && b->hm == 0) {')

# Null move must not stand in for searching the legal alternatives.
rep('    if (!in_check && !is_pv && depth>=3 && ply>0 && not_endgame && static_eval>=beta) {',
    '    if (ss->sing_from[ply] < 0 && !in_check && !is_pv && depth>=3 && ply>0 && not_endgame && static_eval>=beta) {')

# ProbCut must not bypass the excluded-move verification.
rep('    if (!in_check && !is_pv && depth >= 5 && beta < 18000 && ply > 0) {',
    '    if (ss->sing_from[ply] < 0 && !in_check && !is_pv && depth >= 5 && beta < 18000 && ply > 0) {')

# Preserve the requested verification depth. With TT disabled an IIR reduction
# here would make the singular test artificially shallow.
rep('    if (!pv_move.from && !pv_move.to && depth>=3 && !in_check && depth-1>=2) depth--;',
    '    if (ss->sing_from[ply] < 0 && !pv_move.from && !pv_move.to && depth>=3 && !in_check && depth-1>=2) depth--;')

# Never write an excluded-search result under the normal position hash.
rep('    if ((best_move.from||best_move.to) && !ss->time_up)\n        tt_store(b->hash, best, depth, flag, &best_move, ply, raw_eval);',
    '    if ((best_move.from||best_move.to) && !ss->time_up && ss->sing_from[ply] < 0)\n        tt_store(b->hash, best, depth, flag, &best_move, ply, raw_eval);')

p.write_text(s, encoding='utf-8')
print('v3.24 isolated singular verification patch applied')
