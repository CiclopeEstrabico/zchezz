#!/usr/bin/env python3
"""Convert an NNU4 engine weight file back into a PyTorch transfer checkpoint.

v5.00 uses this as a warm start: generation 0 fine-tunes the already useful
v4.03/v5.00 HalfKP network instead of discarding it and starting randomly.

This is the exact inverse of train/export_nnu4.py for the fixed NNU4 format:
  L1W int16 / QA
  L1B int32 / QA
  L2W int8  / QB
  L2B int32 / (QA_EFF * QB)
  L3W int8  / QB
  L3B float32 unchanged

A no-training import -> export round trip is required to be byte-identical.
The CI workflow v500-nnu4-roundtrip.yml enforces that invariant.
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

import numpy as np
import torch

L1_IN = 2560
L1_OUT = 512
L2_IN = 1024
L2_OUT = 32
L3_IN = 32
QA = 255.0
QB = 64.0
SHIFT = 8.0
OUT_SCALE = 320.0 / (QB * QB)


def read_nnu4(path: Path) -> tuple[dict[str, torch.Tensor], dict]:
    data = path.read_bytes()
    off = 0
    if data[:4] != b"NNU4":
        raise ValueError(f"{path}: bad magic; expected NNU4")
    off += 4

    (epoch,) = struct.unpack_from("<I", data, off)
    off += 4
    dims = struct.unpack_from("<5I", data, off)
    off += 20
    want = (L1_IN, L1_OUT, L2_IN, L2_OUT, L3_IN)
    if tuple(dims) != want:
        raise ValueError(f"{path}: dims={dims}, expected={want}")

    qa, qb, shift, out_scale = struct.unpack_from("<4f", data, off)
    off += 16
    expected_scales = (("QA", qa, QA), ("QB", qb, QB),
                       ("SHIFT", shift, SHIFT), ("OUT_SCALE", out_scale, OUT_SCALE))
    for name, got, expected in expected_scales:
        if not math.isclose(got, expected, rel_tol=0.0, abs_tol=1e-7):
            raise ValueError(f"{path}: {name}={got}, expected fixed NNU4 value {expected}")

    qa_eff = int((qa * qa) // (1 << int(shift)))
    if qa_eff != 254:
        raise ValueError(f"{path}: derived QA_EFF={qa_eff}, expected 254")

    def take(dtype: str, count: int) -> np.ndarray:
        nonlocal off
        dt = np.dtype(dtype).newbyteorder("<")
        nbytes = dt.itemsize * count
        if off + nbytes > len(data):
            raise ValueError(f"{path}: truncated at offset {off}, need {nbytes} more bytes")
        arr = np.frombuffer(data, dtype=dt, count=count, offset=off).copy()
        off += nbytes
        return arr

    l1w = take("i2", L1_IN * L1_OUT).reshape(L1_IN, L1_OUT).astype(np.float32) / qa
    l1b = take("i4", L1_OUT).astype(np.float32) / qa
    l2w = take("i1", L2_OUT * L2_IN).reshape(L2_OUT, L2_IN).astype(np.float32) / qb
    l2b = take("i4", L2_OUT).astype(np.float32) / float(qa_eff * qb)
    l3w = take("i1", L3_IN).reshape(1, L3_IN).astype(np.float32) / qb
    (l3b,) = struct.unpack_from("<f", data, off)
    off += 4

    if off != len(data):
        raise ValueError(f"{path}: trailing bytes ({len(data)-off}) after expected NNU4 payload")

    weights = {
        "l1.weight": torch.from_numpy(l1w),
        "l1_bias": torch.from_numpy(l1b),
        "l2.weight": torch.from_numpy(l2w),
        "l2.bias": torch.from_numpy(l2b),
        "l3.weight": torch.from_numpy(l3w),
        "l3.bias": torch.tensor([l3b], dtype=torch.float32),
    }
    meta = {
        "source_epoch": int(epoch),
        "qa": float(qa),
        "qb": float(qb),
        "shift": float(shift),
        "out_scale": float(out_scale),
        "qa_eff": int(qa_eff),
    }
    return weights, meta


def convert(src: Path, dst: Path) -> None:
    weights, meta = read_nnu4(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        # Preserve the source epoch so a pure import/export is byte-identical.
        # train_nnue.py still starts transfer learning at epoch 0 because the
        # dataset tag below intentionally differs from the target dataset tag.
        "epoch": meta["source_epoch"],
        "dataset": "v500_bootstrap_nnu4",
        "avg_loss": None,
        "val_loss": None,
        "lr": None,
        "timestamp": "bootstrap",
        "arch": {
            "input": L1_IN,
            "h1": L1_OUT,
            "concat": L2_IN,
            "h2": L2_OUT,
            "encoding": "halfkp_4bucket",
        },
        "qat": True,
        "qa": meta["qa"],
        "qb": meta["qb"],
        "weights": weights,
        "bootstrap": {"source": str(src), **meta},
    }
    torch.save(ckpt, dst)
    print(f"[bootstrap] {src} -> {dst}")
    print(
        f"[bootstrap] source_epoch={meta['source_epoch']} "
        f"QA={meta['qa']:g} QB={meta['qb']:g} OUT_SCALE={meta['out_scale']:g}"
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Convert NNU4 weights to a v5.00 PyTorch warm-start checkpoint")
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    args = p.parse_args()
    convert(args.src, args.dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
