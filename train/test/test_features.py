import numpy as np
start_pos = [
    20,18,19,21,22,19,18,20,
    17,17,17,17,17,17,17,17,
     0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0,
     9, 9, 9, 9, 9, 9, 9, 9,
    12,10,11,13,14,11,10,12,
]
def pc_type(p): return p & 7
def pc_color(p): return p & 24

for sq in range(64):
    p = start_pos[sq]
    if not p: continue
    t = pc_type(p) - 1
    pySq = sq ^ 56
    c = 0 if pc_color(p) == 8 else 1
    idx = c*64 + t*64 + pySq
    print(f"sq={sq} p={p} pt={t} color={pc_color(p)} => idx={idx}")
