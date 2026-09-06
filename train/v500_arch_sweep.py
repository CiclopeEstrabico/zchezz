#!/usr/bin/env python3
"""Architecture sweep harness for Zchezz v5 HalfKP NNUE.

Keeps the feature set/search fixed and varies only H1/H2.  The accepted
v5 NNU4 (512x32) is the source of truth for warm starts.

Design goals:
- 512x64 / 512x128 start functionally identical to the accepted 512x32 net:
  the old 32 L2 units and their L3 weights are copied exactly; new L2 units
  are live/random, but their L3 weights start at zero.
- 384/256 H1 variants keep the most important baseline channels, ranked by
  absolute outgoing L2 weight across BOTH STM and opponent halves.
- No change to HalfKP buckets, accumulator semantics, search, or manual
  features.
"""
from __future__ import annotations

import argparse
import os
import random
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train"
if str(TRAIN) not in sys.path:
    sys.path.insert(0, str(TRAIN))

BASE_H1 = 512
BASE_H2 = 32


def _configure_arch(h1: int, h2: int):
    if h1 <= 0 or h2 <= 0:
        raise ValueError("h1/h2 must be positive")
    import model as m
    import train_nnue as tn
    import export_nnu4 as ex

    m.HIDDEN1 = int(h1)
    m.CONCAT_DIM = int(h1) * 2
    m.HIDDEN2 = int(h2)
    tn.ARCH_DICT = {
        "input": 2560,
        "h1": int(h1),
        "concat": int(h1) * 2,
        "h2": int(h2),
        "encoding": "halfkp_4bucket",
    }
    ex.L1_OUT = int(h1)
    ex.L2_IN = int(h1) * 2
    ex.L2_OUT = int(h2)
    ex.L3_IN = int(h2)
    return m, tn, ex


def _select_h1(base: dict[str, torch.Tensor], h1: int) -> tuple[torch.Tensor, float]:
    if h1 > BASE_H1:
        raise ValueError(f"H1 expansion above {BASE_H1} is not supported by this sweep")
    if h1 == BASE_H1:
        return torch.arange(BASE_H1, dtype=torch.long), 1.0

    # Each H1 channel appears once in STM and once in opponent half of L2.
    w2 = base["l2.weight"].float()
    importance = w2[:, :BASE_H1].abs().sum(0) + w2[:, BASE_H1:].abs().sum(0)
    top = torch.topk(importance, k=h1, largest=True, sorted=False).indices
    top = torch.sort(top).values
    retained = float(importance[top].sum() / importance.sum().clamp_min(1e-12))
    return top, retained


def bootstrap(base_nnu4: Path, dst: Path, h1: int, h2: int, seed: int) -> None:
    import import_nnu4 as imp

    base, meta = imp.read_nnu4(base_nnu4)
    m, _, _ = _configure_arch(h1, h2)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    net = m.NNUE()
    state = net.state_dict()

    idx, retained = _select_h1(base, h1)
    cols = torch.cat((idx, idx + BASE_H1))

    state["l1.weight"].copy_(base["l1.weight"][:, idx])
    state["l1_bias"].copy_(base["l1_bias"][idx])

    # Copy the complete legacy H2 path. New H2 rows retain PyTorch's random
    # initialization so they have non-zero activations, but are initially
    # invisible because their L3 weights are exactly zero.
    n_old_h2 = min(BASE_H2, h2)
    state["l2.weight"][:n_old_h2].copy_(base["l2.weight"][:n_old_h2, :][:, cols])
    state["l2.bias"][:n_old_h2].copy_(base["l2.bias"][:n_old_h2])
    state["l3.weight"].zero_()
    state["l3.weight"][:, :n_old_h2].copy_(base["l3.weight"][:, :n_old_h2])
    state["l3.bias"].copy_(base["l3.bias"])

    dst.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": int(meta["source_epoch"]),
        "dataset": f"v500_arch_bootstrap_{h1}x{h2}",
        "avg_loss": None,
        "val_loss": None,
        "lr": None,
        "timestamp": "arch-bootstrap",
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
        "bootstrap": {
            "source": str(base_nnu4),
            "source_h1": BASE_H1,
            "source_h2": BASE_H2,
            "retained_h1_outgoing_importance": retained,
            "selected_h1": idx.tolist(),
            **meta,
        },
    }
    torch.save(ckpt, dst)
    print(f"[arch] bootstrap {BASE_H1}x{BASE_H2} -> {h1}x{h2}: {dst}")
    print(f"[arch] retained H1 outgoing importance: {retained:.6f}")
    if h1 == BASE_H1 and h2 >= BASE_H2:
        print("[arch] initialization preserves the complete baseline output path exactly")


def export_checkpoint(ckpt: Path, dst: Path, h1: int, h2: int) -> None:
    _, _, ex = _configure_arch(h1, h2)
    dst.parent.mkdir(parents=True, exist_ok=True)
    ex.convert(ckpt, dst)


def train_candidate(args: argparse.Namespace) -> None:
    m, tn, _ = _configure_arch(args.h1, args.h2)
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ns = tn.build_arg_parser().parse_args([])
    ns.ckpt_dir = str(args.ckpt_dir)
    ns.dataset_name = f"v500_arch_{args.h1}x{args.h2}_{args.dataset_tag}"
    ns.epochs = int(args.epochs)
    ns.batch_size = int(args.batch_size)
    ns.lr = float(args.lr)
    ns.transfer_lr = float(args.transfer_lr)
    ns.weight_decay = float(args.weight_decay)
    ns.workers = int(args.workers)
    ns.device = args.device
    ns.val_every = 1
    ns.resample_each_epoch = False
    ns.encode_cache = True
    ns.show_config = False
    ns.checkpoint_source = str(args.bootstrap)
    ns.sources = [tn.SourceSpec(
        kind="parquet",
        path=str(args.source),
        target_col="cp",
        lam=float(args.lam),
        train_pct=1.0,
        val_frac=float(args.val_frac),
        name=f"v500_arch_source_lam{float(args.lam):.2f}",
        pct_mode="sample-rows",
    )]
    print(f"[arch] train {args.h1}x{args.h2} epochs={ns.epochs} transfer_lr={ns.transfer_lr:.2e} lam={args.lam}")
    tn.train(ns)


def materialize_engine(base_engine: Path, dst_engine: Path, weights: Path, h1: int, h2: int) -> None:
    if dst_engine.exists():
        shutil.rmtree(dst_engine)
    shutil.copytree(base_engine, dst_engine)
    header = dst_engine / "nnue.h"
    text = header.read_text(encoding="utf-8")

    replacements = {
        "NN_L1_OUT": h1,
        "NN_L2_IN": 2 * h1,
        "NN_L2_OUT": h2,
        "NN_L3_IN": h2,
    }
    for name, value in replacements.items():
        pattern = rf"(^#define\s+{re.escape(name)}\s+)\d+"
        text, n = re.subn(pattern, rf"\g<1>{value}", text, count=1, flags=re.MULTILINE)
        if n != 1:
            raise RuntimeError(f"could not patch {name} in {header}")
    header.write_text(text, encoding="utf-8")
    shutil.copy2(weights, dst_engine / "nnue_weights.bin")
    print(f"[arch] materialized engine {dst_engine.name}: H1={h1} H2={h2}")


def _latest_ckpt(path: Path) -> Path:
    xs = sorted(path.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    if not xs:
        raise SystemExit(f"no checkpoints found in {path}")
    return xs[-1]


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bootstrap")
    b.add_argument("--base-nnu4", type=Path, required=True)
    b.add_argument("--dst", type=Path, required=True)
    b.add_argument("--h1", type=int, required=True)
    b.add_argument("--h2", type=int, required=True)
    b.add_argument("--seed", type=int, default=200809)

    e = sub.add_parser("export")
    e.add_argument("--ckpt", type=Path)
    e.add_argument("--ckpt-dir", type=Path)
    e.add_argument("--dst", type=Path, required=True)
    e.add_argument("--h1", type=int, required=True)
    e.add_argument("--h2", type=int, required=True)

    t = sub.add_parser("train")
    t.add_argument("--bootstrap", type=Path, required=True)
    t.add_argument("--ckpt-dir", type=Path, required=True)
    t.add_argument("--source", type=Path, required=True)
    t.add_argument("--h1", type=int, required=True)
    t.add_argument("--h2", type=int, required=True)
    t.add_argument("--epochs", type=int, default=6)
    t.add_argument("--batch-size", type=int, default=2048)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--transfer-lr", type=float, default=8e-6)
    t.add_argument("--weight-decay", type=float, default=1e-4)
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--device", default="cpu")
    t.add_argument("--lam", type=float, default=0.15)
    t.add_argument("--val-frac", type=float, default=0.02)
    t.add_argument("--seed", type=int, default=200809)
    t.add_argument("--dataset-tag", default="probe")

    m = sub.add_parser("materialize-engine")
    m.add_argument("--base-engine", type=Path, required=True)
    m.add_argument("--dst-engine", type=Path, required=True)
    m.add_argument("--weights", type=Path, required=True)
    m.add_argument("--h1", type=int, required=True)
    m.add_argument("--h2", type=int, required=True)

    args = p.parse_args()
    if args.cmd == "bootstrap":
        bootstrap(args.base_nnu4, args.dst, args.h1, args.h2, args.seed)
    elif args.cmd == "export":
        ckpt = args.ckpt or _latest_ckpt(args.ckpt_dir)
        export_checkpoint(ckpt, args.dst, args.h1, args.h2)
    elif args.cmd == "train":
        train_candidate(args)
    elif args.cmd == "materialize-engine":
        materialize_engine(args.base_engine, args.dst_engine, args.weights, args.h1, args.h2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
