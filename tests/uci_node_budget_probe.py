#!/usr/bin/env python3
"""Measure wall time for identical UCI fixed-node searches.

Unlike movetime probes, this removes iterative-deepening reporting granularity:
every engine receives the same node budget on the same positions.  The primary
metric is go->bestmove wall time.
"""
from __future__ import annotations
import argparse,json,random,statistics,subprocess,time
from pathlib import Path
import chess


def send(p, s):
    p.stdin.write(s+'\n'); p.stdin.flush()

def read_until(p,prefix,timeout=10):
    deadline=time.monotonic()+timeout; lines=[]
    while time.monotonic()<deadline:
        line=p.stdout.readline()
        if line=='': raise RuntimeError('engine pipe closed')
        line=line.strip(); lines.append(line)
        if line.startswith(prefix): return lines
    raise TimeoutError(prefix)

def start(path,hash_mb):
    p=subprocess.Popen([path],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,bufsize=1)
    send(p,'uci'); read_until(p,'uciok')
    send(p,f'setoption name Hash value {hash_mb}')
    send(p,'setoption name Threads value 1')
    send(p,'setoption name OwnBook value false')
    send(p,'isready'); read_until(p,'readyok')
    return p

def positions(seed,count):
    r=random.Random(seed); out=[]; seen=set()
    while len(out)<count:
        b=chess.Board()
        for _ in range(r.randint(8,54)):
            lm=list(b.legal_moves)
            if not lm: break
            b.push(r.choice(lm))
            if b.is_game_over(claim_draw=True): break
        if b.is_game_over(claim_draw=True) or b.legal_moves.count()<2: continue
        f=b.fen()
        if f in seen: continue
        seen.add(f); out.append(f)
    return out

def probe(p,fen,nodes):
    send(p,'ucinewgame'); send(p,'isready'); read_until(p,'readyok')
    send(p,f'position fen {fen}')
    t0=time.perf_counter_ns(); send(p,f'go nodes {nodes}')
    lines=read_until(p,'bestmove',30)
    wall_ms=(time.perf_counter_ns()-t0)/1e6
    bm=next(x for x in reversed(lines) if x.startswith('bestmove')).split()
    best=bm[1] if len(bm)>1 else ''
    return {'wall_ms':wall_ms,'bestmove':best}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--engine',action='append',required=True,help='NAME=PATH')
    ap.add_argument('--budgets',default='10000,25000,50000,100000,200000,400000')
    ap.add_argument('--positions',type=int,default=64)
    ap.add_argument('--seed',type=int,default=806300)
    ap.add_argument('--hash-mb',type=int,default=16)
    ap.add_argument('--json',required=True)
    a=ap.parse_args(); specs=[x.split('=',1) for x in a.engine]; budgets=[int(x) for x in a.budgets.split(',')]
    fens=positions(a.seed,a.positions); procs={n:start(p,a.hash_mb) for n,p in specs}
    out={'seed':a.seed,'positions':a.positions,'budgets':{},'ratios':{}}
    try:
        for budget in budgets:
            rows={n:[] for n,_ in specs}
            for i,fen in enumerate(fens):
                order=specs[i%len(specs):]+specs[:i%len(specs)]
                for n,_ in order: rows[n].append(probe(procs[n],fen,budget))
            summary={}
            for n,_ in specs:
                vals=[x['wall_ms'] for x in rows[n]]
                summary[n]={'median_wall_ms':statistics.median(vals),'mean_wall_ms':statistics.mean(vals)}
            out['budgets'][str(budget)]={'summary':summary,'rows':rows}
            base=specs[0][0]; bw=summary[base]['median_wall_ms']
            for n,_ in specs[1:]:
                # >1 means base engine is faster (opponent takes more wall time).
                out['ratios'][f'{base}_speed_over_{n}_{budget}n']=summary[n]['median_wall_ms']/bw
            print(budget, json.dumps(summary), flush=True)
    finally:
        for p in procs.values():
            try: send(p,'quit'); p.wait(timeout=2)
            except Exception: p.kill()
    Path(a.json).write_text(json.dumps(out,indent=2))
    print(json.dumps(out['ratios'],indent=2))
if __name__=='__main__': main()
