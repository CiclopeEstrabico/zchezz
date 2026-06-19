import json, numpy as np

CKPT = r"c:\nnue_checkpoints\checkpoints\nnue_qat_v188H_epoch52_2026-04-26_00-10-36.json"
with open(CKPT, "r") as f:
    state = json.load(f)

w = state['weights']
QA = 255.0
QB = 64.0

L1W = np.round(np.array(w['l1.weight']) * QA).astype(np.int32)
L1B = np.round(np.array(w['l1.bias']) * QA).astype(np.int32)

L2W = np.round(np.array(w['l2.weight']) * QB).astype(np.int32)
L2B = np.round(np.array(w['l2.bias']) * QA * QB).astype(np.int32)

L3W = np.round(np.array(w['l3.weight']) * QB).astype(np.int32)
L3B = np.array(w['l3.bias']) # Float!

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
    pySq = sq ^ 56
    c = 0 if pc_color(p) == 8 else 6
    idx = c*64 + t*64 + pySq
    feat[idx] = 1.0

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
feat[782:790] = 0.0 # MATCH C ENGINE BLOCKED PAWNS
feat[790:798] = 0.0 # MATCH C ENGINE BLOCKED PAWNS
feat[798] = 1.0

# Simulate exact C engine L1 accumulation
acc_hm = np.zeros(256, dtype=np.int32)
# HM pieces
for i in range(768):
    if feat[i] == 1.0:
        acc_hm += L1W[:, i]

acc_ext = np.zeros(256, dtype=np.int32)
# Ext features
for i in range(768, 799):
    if feat[i] > 0.0:
        fj = int(feat[i] * 256.0)
        # C engine: out += (fj * row) / 256 (arithmetic shift)
        prod = fj * L1W[:, i]
        acc_ext += np.floor_divide(prod, 256)

print("Python HM max:", np.max(acc_hm))
print("Python Ext max:", np.max(acc_ext))
print("Python Sum max:", np.max(acc_hm + acc_ext))
print("Python Sum min:", np.min(acc_hm + acc_ext))

acc = acc_hm + acc_ext + L1B
relu1 = np.clip(acc, 0, 255)
print("Python Quantized L1 sum:", np.sum(relu1))

# Simulate L2
acc2 = np.dot(L2W, relu1) + L2B
relu2 = np.clip(acc2 >> 8, 0, 64)
print("Python Quantized L2 sum:", np.sum(relu2))

# Simulate L3
out = np.sum(L3W * relu2)
print("Python Quantized L3 raw sum:", out)
cp = (out * (320.0 / 4096.0)) + (L3B[0] * 320.0)
print("Python Quantized Eval:", cp)
