#!/usr/bin/env python3
"""Create an exact-function compressed v5 NNU4 checkpoint.

The accepted 512x32 network has only a small set of H1 channels with non-zero
quantized outgoing L2 weights.  This utility copies every such live channel,
pads the requested H1 width with quantized-dead baseline channels, and keeps the
original 32-unit H2/L3 path unchanged.  No new residual capacity is added and
no training is performed.

The resulting network is intended to be bit/search equivalent at fixed depth
while reducing accumulator and dense-layer work.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train"
if str(TRAIN) not in sys.path:
    sys.path.insert(0, str(TRAIN))

import import_nnu4 as imp
import v500_arch_sweep as arch

BASE_H1 = 512
BASE_H2 = 32
QB = 64.0


def quantized_importance(base: dict[str, torch.Tensor]) -> torch.Tensor:
    w2q = torch.round(base["l2.weight"].float() * QB).to(torch.int32)
    return w2q[:, :BASE_H1].abs().sum(0) + w2q[:, BASE_H1:].abs().sum(0)


def compress(base_nnu4: Path, dst: Path, h1: int, seed: int) -> None:
    if h1 < 1 or h1 > BASE_H1:
        raise ValueError(f"h1 must be in [1,{BASE_H1}]")

    base, meta = imp.read_nnu4(base_nnu4)
    importance = quantized_importance(base)
    live = torch.nonzero(importance > 0, as_tuple=False).flatten()
    if len(live) > h1:
        raise ValueError(f"H1={h1} cannot fit {len(live)} quantized-live baseline channels")

    live_sorted = live[torch.argsort(importance[live], descending=True)]
    dead = torch.nonzero(importance == 0, as_tuple=False).flatten()
    selected = torch.cat((live_sorted, dead[: h1 - len(live_sorted)]))

    m, _, _ = arch._configure_arch(h1, BASE_H2)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    net = m.NNUE()
    state = net.state_dict()

    state["l1.weight"].copy_(base["l1.weight"][:, selected])
    state["l1_bias"].copy_(base["l1_bias"][selected])

    cols = torch.cat((selected, selected + BASE_H1))
    state["l2.weight"].copy_(base["l2.weight"][:, cols])
    state["l2.bias"].copy_(base["l2.bias"])
    state["l3.weight"].copy_(base["l3.weight"])
    state["l3.bias"].copy_(base["l3.bias"])

    dst.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": int(meta["source_epoch"]),
        "dataset": f"v500_exact_compress_{h1}x{BASE_H2}",
        "avg_loss": None,
        "val_loss": None,
        "lr": None,
        "timestamp": "exact-compress",
        "arch": {
            "input": 2560,
            "h1": h1,
            "concat": 2 * h1,
            "h2": BASE_H2,
            "encoding": "halfkp_4bucket",
        },
        "qat": True,
        "qa": meta["qa"],
        "qb": meta["qb"],
        "weights": state,
        "exact_compress": {
            "source": str(base_nnu4),
            "selected_original_h1": selected.tolist(),
            "live_original_h1": live_sorted.tolist(),
            "n_live": int(len(live_sorted)),
            **meta,
        },
    }
    torch.save(ckpt, dst)
    print(f"[exact-compress] {BASE_H1}x{BASE_H2} -> {h1}x{BASE_H2}")
    print(f"[exact-compress] quantized-live H1 copied: {len(live_sorted)}/{h1}")
    print(f"[exact-compress] padded quantized-dead H1: {h1-len(live_sorted)}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-nnu4", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--h1", type=int, required=True)
    p.add_argument("--seed", type=int, default=200809)
    a = p.parse_args()
    compress(a.base_nnu4, a.dst, a.h1, a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
