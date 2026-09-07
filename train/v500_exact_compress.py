#!/usr/bin/env python3
"""Create an exact-function compressed v5 NNU4 checkpoint.

The accepted 512x32 network is far sparser after integer quantization than its
nominal architecture suggests.  A channel is removable when there is no
quantized path from that channel to the scalar output:

* H2 is observable iff its quantized L3 coefficient is non-zero.
* H1 is observable iff it has a non-zero quantized L2 edge (in either
  perspective half) to at least one observable H2 unit.

This utility copies every observable path, pads requested widths with
quantized-unobservable baseline channels when necessary, and performs no
training.  Removing a path whose integer weight is zero cannot change the
runtime NNU4 evaluation.

For compatibility with existing experiments, --h2 defaults to the original 32.
The native AVX2 engine currently prefers H1 multiples of 16 and H2 multiples of
4; the compressor itself does not impose those kernel-specific constraints.
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


def quantized_structure(
    base: dict[str, torch.Tensor], qb: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return observable H1 importance, observable H1 indices and live H2."""
    q2 = torch.round(base["l2.weight"].float() * qb).to(torch.int32)
    q3 = torch.round(base["l3.weight"].float() * qb).to(torch.int32).flatten()

    live_h2 = torch.nonzero(q3 != 0, as_tuple=False).flatten()
    if len(live_h2) == 0:
        # A constant network has no observable H1 path.  Keep shapes valid and
        # let padding choose arbitrary original channels.
        importance = torch.zeros(BASE_H1, dtype=torch.int64)
    else:
        q2_obs = q2[live_h2]
        importance = (
            q2_obs[:, :BASE_H1].abs().sum(0)
            + q2_obs[:, BASE_H1:].abs().sum(0)
        )
    live_h1 = torch.nonzero(importance > 0, as_tuple=False).flatten()
    return importance, live_h1, live_h2


def compress(base_nnu4: Path, dst: Path, h1: int, h2: int, seed: int) -> None:
    if h1 < 1 or h1 > BASE_H1:
        raise ValueError(f"h1 must be in [1,{BASE_H1}]")
    if h2 < 1 or h2 > BASE_H2:
        raise ValueError(f"h2 must be in [1,{BASE_H2}]")

    base, meta = imp.read_nnu4(base_nnu4)
    qb = float(meta["qb"])
    importance, live_h1, live_h2 = quantized_structure(base, qb)

    if len(live_h1) > h1:
        raise ValueError(
            f"H1={h1} cannot fit {len(live_h1)} transitively observable baseline channels"
        )
    if len(live_h2) > h2:
        raise ValueError(
            f"H2={h2} cannot fit {len(live_h2)} quantized-live baseline units"
        )

    # H1: preserve all paths that can reach an observable H2.  Sorting live
    # channels by importance is irrelevant to the function but deterministic;
    # padding channels are guaranteed to have zero quantized edges to every
    # live H2 and therefore cannot affect the scalar output.
    live_h1_sorted = live_h1[torch.argsort(importance[live_h1], descending=True)]
    dead_h1 = torch.nonzero(importance == 0, as_tuple=False).flatten()
    selected_h1 = torch.cat((live_h1_sorted, dead_h1[: h1 - len(live_h1_sorted)]))

    # H2: retain every non-zero quantized L3 path.  If the original width is
    # requested, preserve original ordering to keep the legacy 512x32/48x32
    # experiments maximally comparable.  Otherwise append zero-L3 baseline
    # units only as SIMD-width padding.
    dead_h2_mask = torch.ones(BASE_H2, dtype=torch.bool)
    dead_h2_mask[live_h2] = False
    dead_h2 = torch.nonzero(dead_h2_mask, as_tuple=False).flatten()
    if h2 == BASE_H2:
        selected_h2 = torch.arange(BASE_H2, dtype=torch.long)
    else:
        selected_h2 = torch.cat((live_h2, dead_h2[: h2 - len(live_h2)]))

    m, _, _ = arch._configure_arch(h1, h2)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    net = m.NNUE()
    state = net.state_dict()

    state["l1.weight"].copy_(base["l1.weight"][:, selected_h1])
    state["l1_bias"].copy_(base["l1_bias"][selected_h1])

    cols = torch.cat((selected_h1, selected_h1 + BASE_H1))
    state["l2.weight"].copy_(base["l2.weight"][selected_h2][:, cols])
    state["l2.bias"].copy_(base["l2.bias"][selected_h2])
    state["l3.weight"].copy_(base["l3.weight"][:, selected_h2])
    state["l3.bias"].copy_(base["l3.bias"])

    dst.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": int(meta["source_epoch"]),
        "dataset": f"v500_exact_compress_{h1}x{h2}",
        "avg_loss": None,
        "val_loss": None,
        "lr": None,
        "timestamp": "exact-compress",
        "arch": {
            "input": 2560,
            "h1": h1,
            "concat": 2 * h1,
            "h2": h2,
            "encoding": "halfkp_4bucket",
        },
        "qat": True,
        "qa": meta["qa"],
        "qb": meta["qb"],
        "weights": state,
        "exact_compress": {
            "source": str(base_nnu4),
            "selected_original_h1": selected_h1.tolist(),
            "live_original_h1": live_h1_sorted.tolist(),
            "n_live_h1": int(len(live_h1_sorted)),
            "selected_original_h2": selected_h2.tolist(),
            "live_original_h2": live_h2.tolist(),
            "n_live_h2": int(len(live_h2)),
            **meta,
        },
    }
    torch.save(ckpt, dst)
    print(f"[exact-compress] {BASE_H1}x{BASE_H2} -> {h1}x{h2}")
    print(
        f"[exact-compress] transitively observable H1 copied: "
        f"{len(live_h1_sorted)}/{h1}; padding={h1-len(live_h1_sorted)}"
    )
    print(
        f"[exact-compress] quantized-live H2 copied: "
        f"{len(live_h2)}/{h2}; padding={h2-len(live_h2)}"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-nnu4", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--h1", type=int, required=True)
    p.add_argument("--h2", type=int, default=BASE_H2)
    p.add_argument("--seed", type=int, default=200809)
    a = p.parse_args()
    compress(a.base_nnu4, a.dst, a.h1, a.h2, a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
