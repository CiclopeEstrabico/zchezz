#!/usr/bin/env python3
from pathlib import Path
import argparse, subprocess, sys
p=argparse.ArgumentParser();p.add_argument('variant',choices=['d3x30','d3x45','d3x75','d4x60']);p.add_argument('--root',default='engine/c/zchezz_v322cand');a=p.parse_args()
subprocess.run([sys.executable,'tools/v322_apply_candidate.py','quiet_see','--root',a.root],check=True)
path=Path(a.root)/'search.c';s=path.read_text(encoding='utf-8')
mult={'d3x30':30,'d3x45':45,'d3x75':75,'d4x60':60}[a.variant]
if s.count('see_board(b, mfr, mto, 0) < -(depth * 60)')!=1: raise SystemExit('SEE threshold marker mismatch')
s=s.replace('see_board(b, mfr, mto, 0) < -(depth * 60)',f'see_board(b, mfr, mto, 0) < -(depth * {mult})',1)
if a.variant=='d4x60':
    marker='!in_check && !is_pv && legal_count > 0 && depth <= 3 &&'
    if s.count(marker)!=1: raise SystemExit('depth marker mismatch')
    s=s.replace(marker,'!in_check && !is_pv && legal_count > 0 && depth <= 4 &&',1)
path.write_text(s,encoding='utf-8')
print('applied',a.variant)
