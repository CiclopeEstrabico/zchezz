"""
convert_nnue3.py  —  QAT (int16/int8)
Converte checkpoint JSON (arquitetura 799→256→64→1) → nnue_weights.bin (NNU3 format).

Mudanças em relação à versão anterior (NNU2):
  • Magic: NNU3
  • Header extra: 4x float32 scale params: [QA, QB, SHIFT, OUT_SCALE]
  • Tipos dos tensores:
    L1W_T: int16
    L1B:   int32
    L2W:   int8  (Transposed)
    L2B:   int32
    L3W:   int8
    L3B:   float32
"""

import json, struct, numpy as np, pathlib, os

CKPT_DIR = pathlib.Path('c:/nnue_checkpoints/checkpoints')
DST = pathlib.Path('c:/nnue_checkpoints/nnue_weights.bin')

# Dimensões Phase 1
L1_IN  = 799   # 768 HM + 31 features manuais
L1_OUT = 256
L2_IN  = 256
L2_OUT = 64
L3_IN  = 64

# Quantization parameters (from mixtrain3.py)
QA = 255.0
QB = 64.0
SHIFT = 8.0
# After the >>SHIFT in nnue_eval, relu2 is in scale QB (not QA).
# L3 accumulates relu2[i]*L3W[i] → scale QB*QB = 4096.
# Therefore OUT_SCALE = 320 / (QB*QB), NOT 320/(QA*QB).
OUT_SCALE = 320.0 / (QB * QB)

def get_latest_checkpoint():
    all_jsons = [f for f in os.listdir(CKPT_DIR) if f.startswith('nnue_') and f.endswith('.json')]
    if not all_jsons:
        return None
    latest = max(all_jsons, key=lambda f: os.path.getmtime(CKPT_DIR / f))
    return CKPT_DIR / latest

def load(path):
    with open(path) as f:
        return json.load(f)

def main():
    src_path = get_latest_checkpoint()
    if not src_path:
        print("No checkpoint found.")
        return

    print(f"Loading {src_path} …")
    ck    = load(src_path)
    
    # Check if new format or old format keys
    if "weights" in ck:
        w = ck["weights"]
    else:
        keys = [k for k in ck if k not in ('epoch', 'avg_loss', 'lr', 'timestamp', 'dataset', 'arch', 'qa', 'qb', 'relu_clip', 'qat')]
        w = {k: ck[k] for k in keys}

    epoch = int(ck.get("epoch", 0))

    # Valida que o checkpoint é Phase1
    arch = ck.get("arch", {})
    if arch:
        assert arch.get("input") == L1_IN,  f"Esperava input={L1_IN}, checkpoint tem {arch.get('input')}"
        assert arch.get("h2")    == L2_OUT,  f"Esperava h2={L2_OUT}, checkpoint tem {arch.get('h2')}"

    # Mapping from v187H keys to v188H keys if needed
    mapping = {
        'net.0.weight': 'l1.weight', 'net.0.bias': 'l1.bias',
        'net.2.weight': 'l2.weight', 'net.2.bias': 'l2.bias',
        'net.4.weight': 'l3.weight', 'net.4.bias': 'l3.bias',
    }
    w = {mapping.get(k, k): v for k, v in w.items()}

    # ── Extrai tensores ──────────────────────────────────────────────────────
    L1W = np.array(w["l1.weight"], dtype=np.float32)  # [256, 799]
    L1B = np.array(w["l1.bias"],   dtype=np.float32)  # [256]
    L2W = np.array(w["l2.weight"], dtype=np.float32)  # [64, 256]
    L2B = np.array(w["l2.bias"],   dtype=np.float32)  # [64]
    L3W = np.array(w["l3.weight"], dtype=np.float32)  # [1, 64]
    L3B = np.array(w["l3.bias"],   dtype=np.float32)  # [1]

    # Quantize — with overflow reporting (Bug 8 fix)
    def _quant16(arr, scale, name):
        q = np.round(arr * scale)
        n_ov = int(np.sum(np.abs(q) > 32767))
        if n_ov:
            print(f"  WARNING: {n_ov} {name} values overflow int16 and will be clipped!")
        return np.clip(q, -32767, 32767).astype(np.int16)

    def _quant8(arr, scale, name):
        q = np.round(arr * scale)
        n_ov = int(np.sum(np.abs(q) > 127))
        if n_ov:
            print(f"  WARNING: {n_ov} {name} values overflow int8 and will be clipped!")
        return np.clip(q, -127, 127).astype(np.int8)

    L1W_q = _quant16(L1W,            QA,      'L1W')
    L1B_q = np.round(L1B * QA).astype(np.int32)
    L2W_q = _quant8 (L2W,            QB,      'L2W')
    L2B_q = np.round(L2B * QA * QB).astype(np.int32)
    # Bug 7 fix: flatten L3W [1,64] → [64] explicitly before quantizing
    L3W_q = _quant8 (L3W.flatten(),  QB,      'L3W')
    L3B_q = L3B.astype(np.float32)   # kept as float32

    # Transpõe L1W: [L1_OUT, L1_IN] → [L1_IN, L1_OUT]
    L1W_T = np.ascontiguousarray(L1W_q.T)   # [799, 256] int16

    # Transpõe L2W: [L2_OUT, L2_IN] → [L2_IN, L2_OUT]
    L2W_T = np.ascontiguousarray(L2W_q.T) # [256, 64] int8

    # ── Escreve binário ──────────────────────────────────────────────────────
    with open(DST, "wb") as f:
        f.write(b"NNU3")
        f.write(struct.pack("<I", epoch))
        for d in (L1_IN, L1_OUT, L2_IN, L2_OUT, L3_IN):
            f.write(struct.pack("<I", d))
        
        # Scale params
        f.write(struct.pack("<ffff", QA, QB, SHIFT, OUT_SCALE))

        f.write(L1W_T.tobytes())
        f.write(L1B_q.tobytes())
        f.write(L2W_T.tobytes())
        f.write(L2B_q.tobytes())
        f.write(L3W_q.tobytes())
        f.write(L3B_q.tobytes())

    expected = 4 + 4 + 20 + 16 + \
               (L1_IN * L1_OUT * 2) + \
               (L1_OUT * 4) + \
               (L2_IN * L2_OUT * 1) + \
               (L2_OUT * 4) + \
               (L3_IN * 1) + \
               (1 * 4)
               
    actual   = DST.stat().st_size
    ok = "OK" if actual == expected else "FAIL SIZE MISMATCH"
    print(f"Written -> {DST}  ({actual:,} bytes, expected {expected:,})  {ok}")
    print(f"  epoch={epoch}  avg_loss={ck.get('avg_loss','?')}  lr={ck.get('lr','?')}")

if __name__ == "__main__":
    main()
