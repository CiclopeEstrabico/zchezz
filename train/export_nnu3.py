"""Export a v3.22 NNU3 PyTorch checkpoint to the engine binary format.

The emitted file is byte-compatible with engine/c/zchezz_v322/nnue.c:
NNU3, 799->256->64->1, int16 L1, int8 L2/L3, 426864 bytes.
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
import struct

import numpy as np
import torch

from model_nnu3 import (
    INPUT_DIM, HIDDEN1, HIDDEN2, QA, QB, assert_compatible_arch,
)

SHIFT = 8.0
OUT_SCALE = 320.0 / (QB * QB)
EXPECTED_SIZE = 426_864


def newest_checkpoint(path: str) -> Path:
    p = Path(path)
    if p.is_file():
        return p
    matches = [Path(x) for x in glob.glob(str(p / "*.pt"))]
    if not matches:
        raise FileNotFoundError(f"no .pt checkpoints in {p}")
    return max(matches, key=lambda x: x.stat().st_mtime_ns)


def _state_dict(ckpt: object) -> tuple[dict, dict | None, int]:
    if not isinstance(ckpt, dict):
        raise ValueError("checkpoint must be a dict")
    arch = ckpt.get("arch")
    epoch = int(ckpt.get("epoch", 0))
    for key in ("model", "state_dict", "model_state_dict"):
        state = ckpt.get(key)
        if isinstance(state, dict):
            return state, arch, epoch
    # Permit a bare state_dict only when all model tensors are present.
    if all(k in ckpt for k in ("l1.weight", "l1.bias", "l2.weight", "l2.bias", "l3.weight", "l3.bias")):
        return ckpt, arch, epoch
    raise ValueError("checkpoint has no model/state_dict/model_state_dict")


def _q16(a: np.ndarray, scale: float, name: str) -> np.ndarray:
    q = np.rint(a * scale)
    n = int((np.abs(q) > 32767).sum())
    if n:
        print(f"WARNING: clipping {n} overflowing {name} int16 values")
    return np.clip(q, -32767, 32767).astype("<i2")


def _q8(a: np.ndarray, scale: float, name: str) -> np.ndarray:
    q = np.rint(a * scale)
    n = int((np.abs(q) > 127).sum())
    if n:
        print(f"WARNING: clipping {n} overflowing {name} int8 values")
    return np.clip(q, -127, 127).astype("i1")


def export_checkpoint(checkpoint: Path, output: Path) -> None:
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state, arch, epoch = _state_dict(raw)
    assert_compatible_arch(arch)

    def arr(name: str) -> np.ndarray:
        value = state[name]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    l1w = arr("l1.weight")
    l1b = arr("l1.bias")
    l2w = arr("l2.weight")
    l2b = arr("l2.bias")
    l3w = arr("l3.weight")
    l3b = arr("l3.bias")

    expected_shapes = {
        "l1.weight": (HIDDEN1, INPUT_DIM),
        "l1.bias": (HIDDEN1,),
        "l2.weight": (HIDDEN2, HIDDEN1),
        "l2.bias": (HIDDEN2,),
        "l3.weight": (1, HIDDEN2),
        "l3.bias": (1,),
    }
    actual = {
        "l1.weight": l1w.shape, "l1.bias": l1b.shape,
        "l2.weight": l2w.shape, "l2.bias": l2b.shape,
        "l3.weight": l3w.shape, "l3.bias": l3b.shape,
    }
    for name, shape in expected_shapes.items():
        if actual[name] != shape:
            raise ValueError(f"{name}: shape {actual[name]} != {shape}")

    # File layout matches the historical NNU3 exporter and current C loader.
    l1w_t = np.ascontiguousarray(_q16(l1w, QA, "L1W").T)       # [799,256]
    l1b_q = np.rint(l1b * QA).astype("<i4")
    l2w_t = np.ascontiguousarray(_q8(l2w, QB, "L2W").T)         # [256,64]
    l2b_q = np.rint(l2b * QA * QB).astype("<i4")
    l3w_q = np.ascontiguousarray(_q8(l3w.reshape(-1), QB, "L3W"))
    l3b_f = np.asarray(l3b, dtype="<f4")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        f.write(b"NNU3")
        f.write(struct.pack("<I", epoch))
        for d in (INPUT_DIM, HIDDEN1, HIDDEN1, HIDDEN2, HIDDEN2):
            f.write(struct.pack("<I", d))
        f.write(struct.pack("<ffff", float(QA), float(QB), SHIFT, OUT_SCALE))
        f.write(l1w_t.tobytes())
        f.write(l1b_q.tobytes())
        f.write(l2w_t.tobytes())
        f.write(l2b_q.tobytes())
        f.write(l3w_q.tobytes())
        f.write(l3b_f.tobytes())

    size = output.stat().st_size
    if size != EXPECTED_SIZE:
        output.unlink(missing_ok=True)
        raise ValueError(f"NNU3 export size {size} != {EXPECTED_SIZE}")
    print(f"NNU3 export: {checkpoint} -> {output} ({size:,} bytes, epoch={epoch})")


def main() -> None:
    p = argparse.ArgumentParser(description="Export v3.22 NNU3 checkpoint")
    p.add_argument("--checkpoint", default="checkpoints/v322",
                   help=".pt file or directory; newest .pt is used for a directory")
    p.add_argument("--output", default="artifacts/nnu3/nnue_weights.bin",
                   help="output file; intentionally does not overwrite the engine by default")
    args = p.parse_args()
    export_checkpoint(newest_checkpoint(args.checkpoint), Path(args.output))


if __name__ == "__main__":
    main()
