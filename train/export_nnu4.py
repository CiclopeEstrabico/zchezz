"""
train/export_nnu4.py — checkpoint -> NNU4 binary converter (Zchezz v4.00)

Converts a train_nnue.py checkpoint (torch.save'd dict, see train_nnue.py's
save_checkpoint()) into the `nnue_weights.bin` NNU4 binary format that
engine/c/zchezz_v400/nnue.c's `_load_weights()` reads. The exact byte
layout below was copied from that function's own doc comment (nnue.c,
around line 211) — READ THAT COMMENT before touching this file; it is the
authoritative spec, this script's only job is to satisfy it.

    "NNU4"                       4 B          magic
    epoch                        uint32
    dims[5]                      L1_IN, L1_OUT, L2_IN, L2_OUT, L3_IN  (uint32 x5)
    scales[4]                    QA, QB, SHIFT, OUT_SCALE             (float32 x4)
    L1W  [2560][512] int16       feature-major (row i = feature i's 512 weights)
    L1B  [512]       int32
    L2W  [32][1024]  int8        output-major, NO transpose
    L2B  [32]        int32
    L3W  [32]        int8
    L3B                          float32 (raw, never quantized)

── THE ONE PLACE THIS SCRIPT DELIBERATELY DIVERGES FROM THE IMPLEMENTATION
   PLAN'S PSEUDOCODE (PARTE 6) — READ THIS BEFORE "FIXING" THE L1 TRANSPOSE ──

PARTE 6 of zchezz_v400_implementation_plan.md writes:

    L1W = np.array(w["l1.weight"], dtype=np.float32)   # [512, 2560]
    ...
    L1W_T = np.ascontiguousarray(L1W_q.T)   # [2560, 512] int16

That pseudocode assumes `l1.weight` came from an `nn.Linear(2560, 512)`,
whose weight tensor is `[out_features, in_features] = [512, 2560]` and
therefore genuinely needs transposing to reach nnue.c's feature-major
`[2560][512]` layout.

model.py's NNUE class does NOT use nn.Linear for L1 — it uses
`nn.EmbeddingBag(2560, 512, mode='sum')` instead, specifically so training
can consume sparse active-feature indices instead of a dense 2560-wide
one-hot vector (see model.py's docstring for the full rationale: at most
~30 active features out of 2560, a dense matmul wastes >98% of the work).
An EmbeddingBag's weight tensor has shape `[num_embeddings, embedding_dim]
= [2560, 512]` — i.e. it is ALREADY stored one contiguous 512-wide row per
feature index, which is EXACTLY the feature-major layout nnue.c wants.

So: **this script does NOT transpose L1W.** Transposing it here would
silently swap the file's [2560][512] layout for a [512][2560]-shaped
tensor's data reinterpreted as [2560][512] — every row nnue.c reads would
be 512 elements taken from the wrong place, and the engine would produce
garbage evaluations that still happen to "run" (no crash, no dimension
mismatch — just wrong numbers). This is exactly the kind of bug the
plan's own APPENDIX D calls "the engine plays against itself."

If you ever change model.py back to an nn.Linear-based L1 (e.g. for a
future architecture that no longer needs sparse indices), you MUST restore
the `.T` here, and this comment block will be wrong — update it.

L2 is unaffected by this: `nn.Linear(1024, 32).weight` is already
`[32, 1024]` = `[out, in]`, which is precisely the "output-major, NO
transpose" layout the file format wants — this one line of PARTE 6's
pseudocode is correct as written and is used verbatim below.

── Quantization ────────────────────────────────────────────────────────

Scales and per-tensor quantization rules are APPENDIX E of the plan,
section E.2, reproduced in the table below (also in model.py's docstring,
since QAT training must fake-quantize at the SAME scales this script uses
for real, or the clamp_weights_() safety margin becomes meaningless):

    L1W : int16, scale QA=255                        (±32767)
    L1B : int32, scale QA=255
    L2W : int8,  scale QB=64                          (±127), NO transpose
    L2B : int32, scale QA_EFF*QB = 254*64 = 16256      (QA*QB=16320 also acceptable, ~0.39% error)
    L3W : int8,  scale QB=64
    L3B : float32, NOT quantized
    OUT_SCALE = 320.0 / (QB*QB) = 320/4096 = 0.078125

Overflow-warning quantizers (APPENDIX E.7): any weight whose rounded
quantized value would not fit in the target int range is clipped AND
reported by name+count, since with clamp_weights_() applied every training
step this should never actually trigger — if it does, that is a real bug
(clamp_weights_ was skipped, or QA/QB drifted between training and
conversion) and deserves a loud warning, not a silent clip.
"""

from __future__ import annotations

import argparse
import os
import struct
from pathlib import Path

import numpy as np
import torch

# ── Architecture dims (must match engine/c/zchezz_v400/nnue.h exactly) ────
L1_IN = 2560
L1_OUT = 512
L2_IN = 1024
L2_OUT = 32
L3_IN = 32

# ── Quantization constants (must match model.py and nnue.h) ───────────────
QA = 255.0
QB = 64.0
SHIFT = 8.0
QA_EFF = 254.0   # (QA*QA) // 256 — see model.py's SCReLU docstring
OUT_SCALE = 320.0 / (QB * QB)   # 320 / 4096 = 0.078125

MAGIC = b"NNU4"

# ═══════════════ CONFIGURATION ═══════════════
# Everything reachable from the CLI (see build_arg_parser below) also lives
# here — edit these to change the defaults without typing flags every time.
CKPT = None                                             # explicit checkpoint .pt path; None = auto-pick newest in CKPT_DIR
CKPT_DIR = Path("C:/nnue_checkpoints/checkpoints_v400")  # directory searched for nnue_v400_*.pt when --ckpt is not given
DST_PATH = Path("C:/nnue_checkpoints/nnue_weights_v400.bin")  # output NNU4 binary path
# ═══════════════════════════════════════════


def _quant16(arr: np.ndarray, scale: float, name: str) -> np.ndarray:
    q = np.round(arr * scale)
    n_ov = int(np.sum(np.abs(q) > 32767))
    if n_ov > 0:
        print(f"  WARNING: {n_ov} {name} values overflow int16 and will be clipped!")
        print(f"    max_abs={np.abs(arr).max():.4f}, limit={32767 / scale:.4f}")
    return np.clip(q, -32767, 32767).astype(np.int16)


def _quant8(arr: np.ndarray, scale: float, name: str) -> np.ndarray:
    q = np.round(arr * scale)
    n_ov = int(np.sum(np.abs(q) > 127))
    if n_ov > 0:
        print(f"  WARNING: {n_ov} {name} values overflow int8 and will be clipped!")
        print(f"    max_abs={np.abs(arr).max():.4f}, limit={127 / scale:.4f}")
    return np.clip(q, -127, 127).astype(np.int8)


def _quant_bias_int32(arr: np.ndarray, scale: float) -> np.ndarray:
    return np.round(arr * scale).astype(np.int32)


def find_latest_checkpoint(ckpt_dir: Path) -> Path | None:
    candidates = sorted(
        (p for p in ckpt_dir.glob("nnue_v400_*.pt")),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def convert(ckpt_path: Path, dst_path: Path) -> None:
    print(f"Loading {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    w = ckpt["weights"] if "weights" in ckpt else ckpt
    epoch = int(ckpt.get("epoch", 0))

    arch = ckpt.get("arch", {})
    if arch:
        assert arch.get("input") == L1_IN, f"expected input={L1_IN}, checkpoint has {arch.get('input')}"
        assert arch.get("h1") == L1_OUT, f"expected h1={L1_OUT}, checkpoint has {arch.get('h1')}"
        assert arch.get("concat") == L2_IN, f"expected concat={L2_IN}, checkpoint has {arch.get('concat')}"
        assert arch.get("h2") == L2_OUT, f"expected h2={L2_OUT}, checkpoint has {arch.get('h2')}"
        assert arch.get("encoding") == "halfkp_4bucket", (
            f"checkpoint encoding={arch.get('encoding')!r} is not 'halfkp_4bucket' "
            f"— refusing to convert a non-HalfKP checkpoint with the v400 converter."
        )

    def _t(key: str) -> np.ndarray:
        return w[key].detach().cpu().numpy().astype(np.float32)

    # model.py's NNUE uses nn.EmbeddingBag for L1: weight is ALREADY
    # [2560, 512] (num_embeddings, embedding_dim) — see module docstring,
    # "THE ONE PLACE THIS SCRIPT DELIBERATELY DIVERGES..." above.
    L1W = _t("l1.weight")          # [2560, 512]  <- NOT transposed, see above
    L1B = _t("l1_bias")            # [512]        <- plain nn.Parameter, not "l1.bias"
    L2W = _t("l2.weight")          # [32, 1024]   output-major already
    L2B = _t("l2.bias")            # [32]
    L3W = _t("l3.weight")          # [1, 32]
    L3B = _t("l3.bias")            # [1]

    assert L1W.shape == (L1_IN, L1_OUT), f"L1W shape {L1W.shape} != {(L1_IN, L1_OUT)} — did model.py change L1 back to nn.Linear? See this file's docstring."
    assert L1B.shape == (L1_OUT,)
    assert L2W.shape == (L2_OUT, L2_IN)
    assert L2B.shape == (L2_OUT,)
    assert L3W.flatten().shape == (L3_IN,)
    assert L3B.flatten().shape == (1,)

    print("Quantizing ...")
    L1W_q = _quant16(L1W, QA, "L1W")                       # [2560, 512] int16, feature-major already
    L1B_q = _quant_bias_int32(L1B, QA)                     # [512] int32
    L2W_q = _quant8(L2W, QB, "L2W")                        # [32, 1024] int8, output-major already
    L2B_q = _quant_bias_int32(L2B, QA_EFF * QB)             # [32] int32, scale 16256
    L3W_q = _quant8(L3W.flatten(), QB, "L3W")               # [32] int8
    L3B_f = float(L3B.flatten()[0])                         # raw float32, never quantized

    L1W_out = np.ascontiguousarray(L1W_q)   # already [2560][512] feature-major, no .T
    L2W_out = np.ascontiguousarray(L2W_q)   # already [32][1024] output-major, no .T

    print(f"Writing {dst_path} ...")
    with open(dst_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", epoch))
        f.write(struct.pack("<5I", L1_IN, L1_OUT, L2_IN, L2_OUT, L3_IN))
        f.write(struct.pack("<4f", QA, QB, SHIFT, OUT_SCALE))
        f.write(L1W_out.tobytes())      # [2560][512] int16
        f.write(L1B_q.tobytes())        # [512] int32
        f.write(L2W_out.tobytes())      # [32][1024] int8
        f.write(L2B_q.tobytes())        # [32] int32
        f.write(L3W_q.tobytes())        # [32] int8
        f.write(struct.pack("<f", L3B_f))

    expected = (
        4 + 4 + 20 + 16
        + (L1_IN * L1_OUT * 2)
        + (L1_OUT * 4)
        + (L2_OUT * L2_IN * 1)
        + (L2_OUT * 4)
        + (L3_IN * 1)
        + 4
    )
    actual = dst_path.stat().st_size
    status = "OK" if actual == expected else "FAIL SIZE MISMATCH"
    print(f"Written -> {dst_path}  ({actual:,} bytes, expected {expected:,})  {status}")
    print(f"  epoch={epoch}  avg_loss={ckpt.get('avg_loss', '?')}  val_loss={ckpt.get('val_loss', '?')}  lr={ckpt.get('lr', '?')}")
    if actual != expected:
        raise SystemExit(1)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert a v400 train_nnue.py checkpoint to NNU4 binary.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", type=Path, default=CKPT,
                    help="Checkpoint .pt path. Defaults to the newest nnue_v400_*.pt in --ckpt-dir.")
    p.add_argument("--ckpt-dir", type=Path, default=CKPT_DIR,
                    help="Directory searched for the newest checkpoint when --ckpt is not given.")
    p.add_argument("--dst", type=Path, default=DST_PATH,
                    help="Output NNU4 binary path.")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    ckpt_path = args.ckpt or find_latest_checkpoint(args.ckpt_dir)
    if ckpt_path is None:
        raise SystemExit(f"No checkpoint found in {args.ckpt_dir} and --ckpt not given.")
    convert(ckpt_path, args.dst)
