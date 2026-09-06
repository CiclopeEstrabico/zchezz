#!/usr/bin/env python3
"""Staged anti-collapse training for the v5 HalfKP residual network.

The accepted HalfKP lineage quantizes to only 39 live H1 channels.  This
experiment separates representation growth from deployment quantization:

  float stage: residual parameters train with NO fake quantization, wd=0,
               and constant LR;
  QAT stage  : same residual masks, deployment fake quantization restored,
               wd=0, constant lower LR.

The accepted 32-unit output path remains frozen in both stages.  Every epoch
is checkpointed by train_nnue.py so the workflow can export/audit/gate each
checkpoint instead of assuming the final epoch is best.
"""
from __future__ import annotations
import argparse, random, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train"
if str(TRAIN) not in sys.path:
    sys.path.insert(0, str(TRAIN))

import v500_arch_sweep as arch
import v500_residual_growth as residual


class _ConstantScheduler:
    """Drop-in scheduler used only by this experiment; LR never decays."""
    def __init__(self, optimizer, *args, **kwargs):
        self.optimizer = optimizer
    def step(self):
        return None


def _install_model(h1: int, h2: int, n_live: int, qat: bool):
    tn = residual._install_residual_model(h1, h2, n_live)
    QATResidual = tn.NNUE
    if qat:
        tn.NNUE = QATResidual
    else:
        class FloatResidual(QATResidual):
            def forward(self, stm_idx, stm_off, opp_idx, opp_off):
                # Same real-valued architecture, but no weight/bias/activation
                # rounding. Gradient masks installed by QATResidual.__init__
                # still freeze the accepted path exactly.
                h_stm = self._l1_perspective(stm_idx, stm_off, self.l1.weight, self.l1_bias)
                h_opp = self._l1_perspective(opp_idx, opp_off, self.l1.weight, self.l1_bias)
                h = torch.cat([h_stm, h_opp], dim=1)
                h2v = self.act2(F.linear(h, self.l2.weight, self.l2.bias))
                return torch.sigmoid(F.linear(h2v, self.l3.weight, self.l3.bias)).squeeze(1)
        tn.NNUE = FloatResidual

    # train_nnue constructs CosineAnnealingLR internally.  For this controlled
    # experiment we intentionally remove that confound: each stage has one LR.
    tn.torch.optim.lr_scheduler.CosineAnnealingLR = _ConstantScheduler
    return tn


def train_stage(a: argparse.Namespace) -> None:
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    rg = ckpt.get("residual_growth") or {}
    n_live = int(rg.get("n_live", -1))
    if n_live <= 0:
        # A later stage checkpoint no longer necessarily carries custom
        # top-level metadata; allow the caller to provide the known bootstrap
        # value explicitly.
        if a.n_live <= 0:
            raise ValueError("checkpoint lacks residual_growth.n_live; pass --n-live")
        n_live = int(a.n_live)

    tn = _install_model(a.h1, a.h2, n_live, a.qat)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    ns = tn.build_arg_parser().parse_args([])
    ns.ckpt_dir = str(a.ckpt_dir)
    ns.dataset_name = f"v500_anticollapse_{a.h1}x{a.h2}_{a.tag}"
    ns.epochs = int(a.epochs)
    ns.batch_size = int(a.batch_size)
    ns.lr = float(a.lr)
    ns.transfer_lr = float(a.lr)
    ns.weight_decay = 0.0
    ns.workers = int(a.workers)
    ns.device = a.device
    ns.val_every = 1
    ns.resample_each_epoch = False
    ns.encode_cache = True
    ns.show_config = False
    ns.checkpoint_source = str(a.checkpoint)
    ns.sources = [tn.SourceSpec(
        kind="parquet", path=str(a.source), target_col="cp",
        lam=float(a.lam), train_pct=1.0, val_frac=float(a.val_frac),
        name=f"v500_anticollapse_teacher_lam{float(a.lam):.2f}",
        pct_mode="sample-rows",
    )]
    mode = "QAT" if a.qat else "FLOAT"
    print(f"[anticollapse] {mode} stage {a.h1}x{a.h2}: epochs={a.epochs} lr={a.lr:.2e} wd=0 constant-LR n_live_frozen={n_live}")
    tn.train(ns)


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train-stage")
    t.add_argument("--checkpoint", type=Path, required=True)
    t.add_argument("--ckpt-dir", type=Path, required=True)
    t.add_argument("--source", type=Path, required=True)
    t.add_argument("--h1", type=int, default=256)
    t.add_argument("--h2", type=int, default=128)
    t.add_argument("--epochs", type=int, default=4)
    t.add_argument("--batch-size", type=int, default=2048)
    t.add_argument("--lr", type=float, required=True)
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--device", default="cpu")
    t.add_argument("--lam", type=float, default=0.15)
    t.add_argument("--val-frac", type=float, default=0.02)
    t.add_argument("--seed", type=int, default=200809)
    t.add_argument("--tag", required=True)
    t.add_argument("--n-live", type=int, default=39)
    t.add_argument("--qat", action=argparse.BooleanOptionalAction, default=False)
    a = p.parse_args()
    if a.cmd == "train-stage":
        train_stage(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
