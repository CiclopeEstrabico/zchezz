#!/usr/bin/env python3
"""Apply one disposable v3.21 round-5 candidate."""
from pathlib import Path
import argparse

VARIANTS = [
    "nmp_lmr",
    "nmp_lmp_looser",
    "nmp_lmr_lmp_looser",
    "aspiration_16",
    "aspiration_28",
    "iir_depth5",
    "iir_depth3",
]
p=argparse.ArgumentParser(); p.add_argument('variant',choices=VARIANTS); p.add_argument('--root',default='engine/c/zchezz_v321cand'); a=p.parse_args()
path=Path(a.root)/'search.c'

def repl(old,new,n=1):
    s=path.read_text(encoding='utf-8'); c=s.count(old)
    if c!=n: raise SystemExit(f'{a.variant}: expected {n} matches for {old!r}, got {c}')
    path.write_text(s.replace(old,new,n),encoding='utf-8')

def nmp(): repl('if (static_eval - beta > 200) R += 1;','if (static_eval - beta > 134) R += 1;')
def lmr(): repl('double v = log((double)d) * log((double)m) / 1.5;','double v = log((double)d) * log((double)m) / 1.35;')
def lmp(): repl('static const int lmp_limit[8] = {0,10,18,26,36,48,62,78};','static const int lmp_limit[8] = {0,12,22,32,44,58,74,94};')

if a.variant=='nmp_lmr': nmp(); lmr()
elif a.variant=='nmp_lmp_looser': nmp(); lmp()
elif a.variant=='nmp_lmr_lmp_looser': nmp(); lmr(); lmp()
elif a.variant=='aspiration_16': repl('int delta = 20, alpha2 = prev_score-delta, beta2 = prev_score+delta;','int delta = 16, alpha2 = prev_score-delta, beta2 = prev_score+delta;')
elif a.variant=='aspiration_28': repl('int delta = 20, alpha2 = prev_score-delta, beta2 = prev_score+delta;','int delta = 28, alpha2 = prev_score-delta, beta2 = prev_score+delta;')
elif a.variant=='iir_depth5': repl('if (!pv_move.from && !pv_move.to && depth>=4 && !in_check && depth-1>=2) depth--;','if (!pv_move.from && !pv_move.to && depth>=5 && !in_check && depth-1>=2) depth--;')
elif a.variant=='iir_depth3': repl('if (!pv_move.from && !pv_move.to && depth>=4 && !in_check && depth-1>=2) depth--;','if (!pv_move.from && !pv_move.to && depth>=3 && !in_check && depth-1>=2) depth--;')
print('applied',a.variant)
