#!/usr/bin/env python3
"""Residual capacity-growth trainer for Zchezz v5 HalfKP.

The accepted 512x32 NNU4 is severely under-utilized after quantization: only
39/512 H1 channels have any live path to L2. Ordinary transfer learning at a
small global LR does not reliably wake quantized-zero channels.

This module grows a smaller/faster target architecture while preserving the
accepted baseline function at initialization:

* all live baseline H1 channels are copied;
* dead/extra H1 channels are deliberately initialized on the quantization
  grid, so they are alive in the exported integer net;
* the first 32 H2 units + L3 path are the frozen accepted baseline path;
* new H2 units are initialized in identical pairs with opposite +/-1 L3
  weights. Their initial contributions cancel exactly, but gradients flow
  through the pair immediately and break the symmetry;
* during residual stage, gradient hooks freeze the accepted path and train
  only new H1 channels plus new H2/L3 capacity. Weight decay must be zero in
  this stage so frozen parameters remain bit-stable.

This is a capacity experiment, not a search change. HalfKP, king buckets,
SCReLU, dual perspective, lazy/full accumulator, and manual-feature policy are
unchanged.
"""
from __future__ import annotations

import argparse
import os
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
QA = 255.0
QB = 64.0


def _baseline_importance(base: dict[str, torch.Tensor]) -> torch.Tensor:
    w2q = torch.round(base["l2.weight"].float() * QB).to(torch.int32)
    return w2q[:, :BASE_H1].abs().sum(0) + w2q[:, BASE_H1:].abs().sum(0)


def residual_bootstrap(base_nnu4: Path, dst: Path, h1: int, h2: int, seed: int) -> None:
    if h1 < 64:
        raise ValueError("residual growth expects h1 >= 64 so all 39 live baseline channels fit")
    if h1 > BASE_H1:
        raise ValueError("this experiment only compresses/reuses the accepted H1")
    if h2 <= BASE_H2 or (h2 - BASE_H2) % 2:
        raise ValueError("h2 must be >32 and add an even number of residual units")

    base, meta = imp.read_nnu4(base_nnu4)
    m, _, _ = arch._configure_arch(h1, h2)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    net = m.NNUE()
    state = net.state_dict()

    importance = _baseline_importance(base)
    live_orig = torch.nonzero(importance > 0, as_tuple=False).flatten()
    if len(live_orig) > h1:
        raise ValueError(f"target H1={h1} cannot fit {len(live_orig)} live baseline channels")

    # Put live channels first in descending importance, then fill remaining
    # target slots from dead original channels. This makes live/new masks
    # explicit and deterministic rather than depending on topk tie order.
    live_sorted = live_orig[torch.argsort(importance[live_orig], descending=True)]
    dead_orig = torch.nonzero(importance == 0, as_tuple=False).flatten()
    selected = torch.cat((live_sorted, dead_orig[: h1 - len(live_sorted)]))
    n_live = len(live_sorted)
    assert len(selected) == h1

    # Start from copied baseline channels. Dead originals are all-zero in the
    # accepted quantized NNU4, then deliberately revived below.
    state["l1.weight"].copy_(base["l1.weight"][:, selected])
    state["l1_bias"].copy_(base["l1_bias"][selected])

    gen = torch.Generator().manual_seed(seed ^ 0x5A17)
    n_new = h1 - n_live
    if n_new:
        # Quantization-grid initialization. +/-[2..8]/QA survives NNU4 export.
        q = torch.randint(-8, 9, (state["l1.weight"].shape[0], n_new), generator=gen)
        q[q == 0] = 2
        state["l1.weight"][:, n_live:].copy_(q.float() / QA)
        # Keep SCReLU in an active, non-saturated region for new channels.
        state["l1_bias"][n_live:].fill_(64.0 / QA)

    # Frozen baseline H2/L3 path, remapped to target H1 columns.
    cols = torch.cat((selected, selected + BASE_H1))
    state["l2.weight"].zero_()
    state["l2.bias"].zero_()
    state["l3.weight"].zero_()
    state["l2.weight"][:BASE_H2].copy_(base["l2.weight"][:, cols])
    state["l2.bias"][:BASE_H2].copy_(base["l2.bias"])
    state["l3.weight"][:, :BASE_H2].copy_(base["l3.weight"])
    state["l3.bias"].copy_(base["l3.bias"])

    # Residual H2 units are paired: same quantized L2 row/bias, opposite
    # quantized L3 weights. Their integer contribution is exactly zero at
    # initialization, but non-zero +/- L3 weights give L2 gradients on the
    # first backward pass. After one optimizer step the pair can diverge.
    for j in range(BASE_H2, h2, 2):
        qw = torch.randint(-2, 3, (2 * h1,), generator=gen)
        # Ensure the revived H1 block has real connections.
        if n_new:
            dead_cols = torch.cat((
                torch.arange(n_live, h1),
                torch.arange(h1 + n_live, 2 * h1),
            ))
            z = qw[dead_cols] == 0
            qw[dead_cols[z]] = 1
        row = qw.float() / QB
        state["l2.weight"][j].copy_(row)
        state["l2.weight"][j + 1].copy_(row)
        state["l2.bias"][j].fill_(0.25)
        state["l2.bias"][j + 1].fill_(0.25)
        state["l3.weight"][0, j] = 1.0 / QB
        state["l3.weight"][0, j + 1] = -1.0 / QB

    dst.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": int(meta["source_epoch"]),
        "dataset": f"v500_residual_bootstrap_{h1}x{h2}",
        "avg_loss": None, "val_loss": None, "lr": None,
        "timestamp": "residual-bootstrap",
        "arch": {"input": 2560, "h1": h1, "concat": 2*h1, "h2": h2,
                 "encoding": "halfkp_4bucket"},
        "qat": True, "qa": meta["qa"], "qb": meta["qb"],
        "weights": state,
        "residual_growth": {
            "source": str(base_nnu4),
            "selected_original_h1": selected.tolist(),
            "live_original_h1": live_sorted.tolist(),
            "n_live": n_live,
            "n_revived": n_new,
            "base_h2": BASE_H2,
            **meta,
        },
    }
    torch.save(ckpt, dst)
    print(f"[residual] bootstrap {BASE_H1}x{BASE_H2} -> {h1}x{h2}")
    print(f"[residual] copied live H1={n_live}, revived H1={n_new}, residual H2={h2-BASE_H2}")
    print("[residual] new H2 path is cancel-paired: initial output contribution = 0")


def _install_residual_model(h1: int, h2: int, n_live: int):
    m, tn, _ = arch._configure_arch(h1, h2)
    Base = m.NNUE

    class ResidualNNUE(Base):
        def __init__(self):
            super().__init__()
            # Old H1 live columns frozen; revived columns train at full LR.
            l1mask = torch.zeros_like(self.l1.weight)
            l1mask[:, n_live:] = 1.0
            self.l1.weight.register_hook(lambda g, mask=l1mask: g * mask.to(g.device))
            b1mask = torch.zeros_like(self.l1_bias); b1mask[n_live:] = 1.0
            self.l1_bias.register_hook(lambda g, mask=b1mask: g * mask.to(g.device))

            # Freeze original 32 H2 rows completely. Residual rows can learn
            # from both copied-live and revived H1 channels.
            w2mask = torch.zeros_like(self.l2.weight); w2mask[BASE_H2:] = 1.0
            self.l2.weight.register_hook(lambda g, mask=w2mask: g * mask.to(g.device))
            b2mask = torch.zeros_like(self.l2.bias); b2mask[BASE_H2:] = 1.0
            self.l2.bias.register_hook(lambda g, mask=b2mask: g * mask.to(g.device))

            # Freeze accepted L3 weights and bias; train only residual outputs.
            w3mask = torch.zeros_like(self.l3.weight); w3mask[:, BASE_H2:] = 1.0
            self.l3.weight.register_hook(lambda g, mask=w3mask: g * mask.to(g.device))
            self.l3.bias.register_hook(lambda g: torch.zeros_like(g))

    tn.NNUE = ResidualNNUE
    return tn


def train_residual(args: argparse.Namespace) -> None:
    ckpt = torch.load(args.bootstrap, map_location="cpu", weights_only=False)
    rg = ckpt.get("residual_growth") or {}
    n_live = int(rg.get("n_live", -1))
    if n_live <= 0:
        raise ValueError("bootstrap lacks residual_growth.n_live")
    tn = _install_residual_model(args.h1, args.h2, n_live)

    seed=int(args.seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    ns=tn.build_arg_parser().parse_args([])
    ns.ckpt_dir=str(args.ckpt_dir)
    ns.dataset_name=f"v500_residual_{args.h1}x{args.h2}_{args.dataset_tag}"
    ns.epochs=int(args.epochs)
    ns.batch_size=int(args.batch_size)
    ns.lr=float(args.lr)
    ns.transfer_lr=float(args.lr)
    # Critical: Adam weight decay would move frozen parameters even when
    # gradient hooks return zero. Residual stage therefore uses wd=0.
    ns.weight_decay=0.0
    ns.workers=int(args.workers)
    ns.device=args.device
    ns.val_every=1
    ns.resample_each_epoch=False
    ns.encode_cache=True
    ns.show_config=False
    ns.checkpoint_source=str(args.bootstrap)
    ns.sources=[tn.SourceSpec(kind="parquet", path=str(args.source), target_col="cp",
        lam=float(args.lam), train_pct=1.0, val_frac=float(args.val_frac),
        name=f"v500_residual_teacher_lam{float(args.lam):.2f}", pct_mode="sample-rows")]
    print(f"[residual] TRAIN {args.h1}x{args.h2}: only revived H1 + residual H2/L3 are trainable")
    print(f"[residual] n_live frozen={n_live}; residual lr={args.lr:.2e}; wd=0; lam={args.lam}")
    tn.train(ns)


def main() -> int:
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="cmd",required=True)
    b=sub.add_parser("bootstrap")
    b.add_argument("--base-nnu4",type=Path,required=True); b.add_argument("--dst",type=Path,required=True)
    b.add_argument("--h1",type=int,required=True); b.add_argument("--h2",type=int,required=True)
    b.add_argument("--seed",type=int,default=200809)
    t=sub.add_parser("train")
    t.add_argument("--bootstrap",type=Path,required=True); t.add_argument("--ckpt-dir",type=Path,required=True)
    t.add_argument("--source",type=Path,required=True); t.add_argument("--h1",type=int,required=True); t.add_argument("--h2",type=int,required=True)
    t.add_argument("--epochs",type=int,default=12); t.add_argument("--batch-size",type=int,default=2048)
    t.add_argument("--lr",type=float,default=3e-4); t.add_argument("--workers",type=int,default=4)
    t.add_argument("--device",default="cpu"); t.add_argument("--lam",type=float,default=0.15)
    t.add_argument("--val-frac",type=float,default=0.02); t.add_argument("--seed",type=int,default=200809)
    t.add_argument("--dataset-tag",default="stage1")
    a=p.parse_args()
    if a.cmd=="bootstrap": residual_bootstrap(a.base_nnu4,a.dst,a.h1,a.h2,a.seed)
    else: train_residual(a)
    return 0

if __name__=="__main__": raise SystemExit(main())
