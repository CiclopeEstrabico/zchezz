import json, numpy as np

CKPT = r"c:\nnue_checkpoints\checkpoints\nnue_qat_v188H_epoch52_2026-04-26_00-10-36.json"
with open(CKPT, "r") as f:
    state = json.load(f)

w = state['weights']
L1W = np.array(w['l1.weight'])
L1B = np.array(w['l1.bias'])
L2W = np.array(w['l2.weight'])
L2B = np.array(w['l2.bias'])
L3W = np.array(w['l3.weight'])
L3B = np.array(w['l3.bias'])

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

feat = np.zeros(799, dtype=np.float32)

def pc_type(p): return p & 7
def pc_color(p): return p & 24

for sq in range(64):
    p = start_pos[sq]
    if not p: continue
    t = pc_type(p) - 1
    # White perspective half-mirror: sq ^ 56
    pySq = sq ^ 56
    c = 0 if pc_color(p) == 8 else 6
    idx = c*64 + t*64 + pySq
    feat[idx] = 1.0

# Extras
# The exact values from the C engine for start_pos are:
feat[768] = 8.0/8.0
feat[769] = 2.0/2.0
feat[770] = 2.0/2.0
feat[771] = 2.0/2.0
feat[772] = 1.0/1.0
feat[773] = 1.0/1.0

feat[774] = 8.0/8.0
feat[775] = 2.0/2.0
feat[776] = 2.0/2.0
feat[777] = 2.0/2.0
feat[778] = 1.0/1.0
feat[779] = 1.0/1.0

feat[780] = 78.0 / 78.0
feat[781] = 1.0

feat[782:790] = 1.0 # file pawns
feat[790:798] = 1.0 # file pawns
feat[798] = 1.0 # king distance


# Inference
x = feat
x = np.dot(L1W, x) + L1B
x = np.clip(x, 0.0, 1.0)
print("Float L1 sum:", np.sum(x))
print("Float L1 sum * 255:", np.sum(x) * 255)
x = np.dot(L2W, x) + L2B
x = np.clip(x, 0.0, 1.0)
print("Float L2 sum:", np.sum(x))
print("Float L2 sum * 64:", np.sum(x) * 64)
x = np.dot(L3W, x) + L3B
print("Float L3 raw:", x[0])
cp = x[0] * 320.0
print("Float Evaluation:", cp)
