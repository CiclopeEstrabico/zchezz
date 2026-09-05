"""Train the v3.22 NNU3 799->256->64->1 network on the modern data catalog.

This is the NNU3 companion to train_nnue.py. It deliberately reuses the
v4.03-era source catalog and label contract instead of carrying a second list
of legacy dataset names:

    parquet: fen + cp and/or result
    bin:     native Zchezz self-play shards from train/dataset.py

The network/encoding/export format remain NNU3-specific. NNU4 checkpoints are
rejected rather than partially loaded into an incompatible architecture.

Examples:
    python train/train_nnu3.py --list-sources
    python train/train_nnu3.py --epochs 30 --dataset-name nnu3_v322_run1
    python train/train_nnu3.py --source kind=parquet,path=data/foo,lam=.25,train_pct=.5,mode=sample-files
    python train/train_nnu3.py --source kind=bin,path=data/selfplay/*.bin,k=.75
    python train/train_nnu3.py --self-test
"""
from __future__ import annotations

import argparse
import glob
import math
import os
from pathlib import Path
import random
import time
import zlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:  # python -m train.train_nnu3
    from . import dataset
    from . import train_nnue as infra
    from .encoding_nnu3 import encode_positions, stm_targets, smoke_test as encoding_smoke_test
    from .model_nnu3 import NNUE, clamp_weights_, architecture_dict, assert_compatible_arch
except ImportError:  # python train/train_nnu3.py
    import dataset
    import train_nnue as infra
    from encoding_nnu3 import encode_positions, stm_targets, smoke_test as encoding_smoke_test
    from model_nnu3 import NNUE, clamp_weights_, architecture_dict, assert_compatible_arch


# Configuration defaults. Dataset NAMES/LABEL RULES live in train_nnue.py and
# are intentionally not duplicated here.
CKPT_DIR = "checkpoints/v322"
DATASET_NAME = "nnu3_v322"
EPOCHS = 100
BATCH_SIZE = 16_384
MAX_POSITIONS_CHUNK = 250_000
PARQUET_CHUNK_ROWS = 100_000
LR = 1e-3
TRANSFER_LR = 3e-5
WEIGHT_DECAY = 1e-4
WORKERS = os.cpu_count() or 4
DEVICE = "auto"
SEED = 3222026
VAL_EVERY = 1
MAX_VAL_POSITIONS = 250_000
RESAMPLE_EACH_EPOCH = True


def _parse_source(spec: str) -> infra.SourceSpec:
    kv = infra._parse_kv_spec(spec)
    kind = kv.get("kind")
    if kind not in ("parquet", "bin"):
        raise argparse.ArgumentTypeError("--source kind must be parquet or bin")
    path = kv.get("path") or kv.get("glob")
    if not path:
        raise argparse.ArgumentTypeError("--source requires path= or glob=")
    mode = kv.get("mode", kv.get("pct_mode", "sample-rows"))
    return infra.SourceSpec(
        kind=kind,
        path=path,
        target_col=kv.get("target_col", kv.get("col", "cp")),
        k=float(kv.get("k", 1.0)),
        lam=float(kv.get("lam", 0.0)),
        train_pct=float(kv.get("train_pct", kv.get("pct", 1.0))),
        val_frac=float(kv.get("val_frac", 0.02)),
        name=kv.get("name", ""),
        pct_mode=mode,
        suffix=kv.get("suffix", ".parquet"),
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Zchezz v3.22 NNU3 trainer using the modern v4.03 data/source layer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source", action="append", type=_parse_source, dest="sources", default=[],
                   metavar="kind=parquet|bin,path=...,pct=1,mode=sample-rows,lam=0,k=1",
                   help="Repeat to override the shared DATASETS/BIN_DATASETS catalog.")
    p.add_argument("--ckpt-dir", default=CKPT_DIR)
    p.add_argument("--checkpoint-source", "--resume-from", default="auto",
                   help="auto/new/path-to-file/path-to-dir; incompatible NNU4 checkpoints are rejected")
    p.add_argument("--dataset-name", default=DATASET_NAME)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--max-positions-chunk", type=int, default=MAX_POSITIONS_CHUNK)
    p.add_argument("--parquet-chunk-rows", type=int, default=PARQUET_CHUNK_ROWS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--transfer-lr", type=float, default=TRANSFER_LR)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--workers", type=int, default=WORKERS)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default=DEVICE)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--val-every", type=int, default=VAL_EVERY)
    p.add_argument("--max-val-positions", type=int, default=MAX_VAL_POSITIONS)
    p.add_argument("--resample-each-epoch", action=argparse.BooleanOptionalAction,
                   default=RESAMPLE_EACH_EPOCH)
    p.add_argument("--list-sources", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p


def _device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _source_salt(source: infra.SourceSpec) -> int:
    return zlib.crc32(source.name.encode("utf-8")) & 0xFFFFFFFF


def _parquet_files(source: infra.SourceSpec) -> list[str]:
    path = source.path
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, f"*{source.suffix}")))
    else:
        files = sorted(glob.glob(path))
        if not files and os.path.isfile(path):
            files = [path]
    if not files:
        raise FileNotFoundError(f"{source.name}: no parquet files matched {path!r}")
    return files


def _choose_train_files(files: list[str], source: infra.SourceSpec, epoch_seed: int) -> set[str]:
    if source.pct_mode != "sample-files" or source.train_pct >= 1.0:
        return set(files)
    n = max(1, min(len(files), round(len(files) * source.train_pct)))
    rng = random.Random(epoch_seed ^ _source_salt(source))
    return set(rng.sample(files, n))


def _legal_mask(fens: list[str]) -> np.ndarray:
    return np.fromiter((infra._fen_is_trainable(fen) for fen in fens),
                       dtype=np.bool_, count=len(fens))


def iter_parquet_source(source: infra.SourceSpec, split: str, epoch: int,
                        args: argparse.Namespace):
    """Yield dense NNU3 (X,y) batches from one parquet source.

    Validation membership is generated from a fixed per-file RNG, so it never
    moves between epochs. Training row/file sampling uses a separate epoch RNG
    and therefore can resample without leaking validation rows into training.
    """
    import pyarrow.parquet as pq

    files = _parquet_files(source)
    train_seed = args.seed + (epoch if args.resample_each_epoch else 0)
    chosen = _choose_train_files(files, source, train_seed)
    source_salt = _source_salt(source)

    for file_i, fpath in enumerate(files):
        if split == "train" and source.pct_mode == "sample-files" and fpath not in chosen:
            continue
        file_salt = zlib.crc32(os.path.basename(fpath).encode("utf-8")) & 0xFFFFFFFF
        split_rng = np.random.default_rng(args.seed ^ source_salt ^ file_salt)
        row_rng = np.random.default_rng(train_seed ^ source_salt ^ file_salt ^ 0x9E3779B9)
        pf = pq.ParquetFile(fpath)
        for batch in pf.iter_batches(batch_size=min(args.parquet_chunk_rows, args.max_positions_chunk)):
            df = batch.to_pandas()
            if "fen" not in df.columns:
                raise ValueError(f"{fpath}: missing required 'fen' column")
            n = len(df)
            val_mask = split_rng.random(n) < source.val_frac
            if split == "val":
                keep = val_mask
            else:
                keep = ~val_mask
                if source.pct_mode == "sample-rows" and source.train_pct < 1.0:
                    keep &= row_rng.random(n) < source.train_pct
            if not keep.any():
                continue
            df = df.loc[keep].reset_index(drop=True)
            fens = df["fen"].astype(str).tolist()
            legal = _legal_mask(fens)
            if not legal.all():
                df = df.loc[legal].reset_index(drop=True)
                fens = [fen for fen, ok in zip(fens, legal) if ok]
            if not fens:
                continue
            y_white = infra.blend_target(df, source).astype(np.float32)
            x, is_black = encode_positions(fens, workers=args.workers)
            y = stm_targets(y_white, is_black)
            yield x, y


def _bin_indices(source: infra.SourceSpec, n: int, split: str, epoch: int,
                 args: argparse.Namespace) -> np.ndarray:
    salt = _source_salt(source)
    split_rng = np.random.default_rng(args.seed ^ salt)
    perm = split_rng.permutation(n)
    n_val = max(1, int(round(n * source.val_frac))) if n > 1 else 0
    idx = perm[:n_val] if split == "val" else perm[n_val:]
    if split == "train" and source.train_pct < 1.0:
        seed = args.seed + (epoch if args.resample_each_epoch else 0)
        rng = np.random.default_rng(seed ^ salt ^ 0xA5A5A5A5)
        idx = idx[rng.random(len(idx)) < source.train_pct]
    return np.sort(idx)


def iter_bin_source(source: infra.SourceSpec, split: str, epoch: int,
                    args: argparse.Namespace):
    ds = dataset.MultiShardSelfplay.from_glob(source.path)
    idx = _bin_indices(source, len(ds), split, epoch, args)
    step = min(args.parquet_chunk_rows, args.max_positions_chunk)
    for start in range(0, len(idx), step):
        records = ds.get_batch(idx[start:start + step])
        fens, y = dataset.records_to_fens_and_targets(records, k=source.k)
        legal = _legal_mask(fens)
        if not legal.all():
            records = records[legal]
            if len(records) == 0:
                continue
            fens, y = dataset.records_to_fens_and_targets(records, k=source.k)
        # dataset.wl_target is already STM-relative. The encoder normalizes the
        # board to STM perspective, so do NOT flip y a second time.
        x, _ = encode_positions(fens, workers=args.workers)
        yield x, y.astype(np.float32)


def iter_sources(sources: list[infra.SourceSpec], split: str, epoch: int,
                 args: argparse.Namespace):
    order = list(range(len(sources)))
    random.Random(args.seed + epoch + (0 if split == "train" else 1_000_003)).shuffle(order)
    generators = []
    for i in order:
        src = sources[i]
        gen = (iter_parquet_source(src, split, epoch, args) if src.kind == "parquet"
               else iter_bin_source(src, split, epoch, args))
        generators.append((src.name, iter(gen)))

    # Round-robin across sources so a huge first dataset does not monopolize an epoch.
    active = generators
    while active:
        next_active = []
        for name, gen in active:
            try:
                yield name, next(gen)
                next_active.append((name, gen))
            except StopIteration:
                pass
        active = next_active


def _resolve_checkpoint(source: str, ckpt_dir: str) -> Path | None:
    token = str(source or "").strip()
    if token.lower() in ("", "new", "scratch", "false", "none"):
        return None
    p = Path(ckpt_dir if token.lower() == "auto" else token)
    if p.is_file():
        return p
    if p.is_dir():
        pts = list(p.glob("*.pt"))
        return max(pts, key=lambda x: x.stat().st_mtime_ns) if pts else None
    if token.lower() == "auto":
        return None
    raise FileNotFoundError(f"checkpoint source does not exist: {source}")


def _checkpoint_state(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"{path}: checkpoint is not a dict")
    assert_compatible_arch(ckpt.get("arch"))
    return ckpt


def _load_model(args: argparse.Namespace, device: torch.device):
    model = NNUE().to(device)
    ckpt_path = _resolve_checkpoint(args.checkpoint_source, args.ckpt_dir)
    start_epoch = 0
    lr = args.lr
    optimizer_state = None
    if ckpt_path is not None:
        ckpt = _checkpoint_state(ckpt_path)
        state = ckpt.get("model") or ckpt.get("state_dict") or ckpt.get("model_state_dict")
        if not isinstance(state, dict):
            raise ValueError(f"{ckpt_path}: no model state")
        model.load_state_dict(state, strict=True)
        if ckpt.get("dataset_name") == args.dataset_name:
            start_epoch = int(ckpt.get("epoch", 0))
            optimizer_state = ckpt.get("optimizer")
            lr = float(ckpt.get("lr", args.lr))
            print(f"resume: {ckpt_path} at epoch {start_epoch}, lr={lr:.3e}")
        else:
            lr = args.transfer_lr
            print(f"weight transfer: {ckpt_path}; dataset tag changed -> lr={lr:.3e}")
    else:
        print("checkpoint: new NNU3 model")
    return model, start_epoch, lr, optimizer_state


def _train_chunk(model: NNUE, optimizer: torch.optim.Optimizer, loss_fn: nn.Module,
                 x_np: np.ndarray, y_np: np.ndarray, batch_size: int,
                 device: torch.device, rng: np.random.Generator) -> tuple[float, float, int]:
    n = len(y_np)
    if n == 0:
        return 0.0, 0.0, 0
    order = rng.permutation(n)
    loss_sum = mae_sum = 0.0
    seen = 0
    model.train()
    for start in range(0, n, batch_size):
        ii = order[start:start + batch_size]
        x = torch.from_numpy(x_np[ii]).to(device=device, dtype=torch.float32, non_blocking=True)
        y = torch.from_numpy(y_np[ii]).to(device=device, dtype=torch.float32, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = loss_fn(pred, y)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite NNU3 training loss")
        loss.backward()
        optimizer.step()
        clamp_weights_(model)
        count = len(ii)
        loss_sum += float(loss.detach()) * count
        mae_sum += float(torch.mean(torch.abs(pred.detach() - y))) * count
        seen += count
    return loss_sum, mae_sum, seen


def _validate(model: NNUE, sources: list[infra.SourceSpec], args: argparse.Namespace,
              device: torch.device, epoch: int) -> tuple[float, float, int]:
    loss_fn = nn.BCELoss(reduction="mean")
    loss_sum = mae_sum = 0.0
    seen = 0
    model.eval()
    with torch.no_grad():
        for _, (x_np, y_np) in iter_sources(sources, "val", epoch, args):
            if args.max_val_positions and seen >= args.max_val_positions:
                break
            if args.max_val_positions:
                remain = args.max_val_positions - seen
                x_np, y_np = x_np[:remain], y_np[:remain]
            for start in range(0, len(y_np), args.batch_size):
                x = torch.from_numpy(x_np[start:start + args.batch_size]).to(device=device, dtype=torch.float32)
                y = torch.from_numpy(y_np[start:start + args.batch_size]).to(device=device, dtype=torch.float32)
                pred = model(x)
                loss = loss_fn(pred, y)
                count = len(y)
                loss_sum += float(loss) * count
                mae_sum += float(torch.mean(torch.abs(pred - y))) * count
                seen += count
    return (loss_sum / max(1, seen), mae_sum / max(1, seen), seen)


def _save_checkpoint(args: argparse.Namespace, model: NNUE, optimizer: torch.optim.Optimizer,
                     epoch: int, metrics: dict) -> Path:
    outdir = Path(args.ckpt_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"nnu3_v322_epoch{epoch:03d}.pt"
    torch.save({
        "format": "zchezz_nnu3_checkpoint_v1",
        "arch": architecture_dict(),
        "dataset_name": args.dataset_name,
        "epoch": epoch,
        "lr": float(optimizer.param_groups[0]["lr"]),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
        "seed": args.seed,
    }, out)
    return out


def _print_sources(sources: list[infra.SourceSpec]) -> None:
    print("kind      pct   mode          lam/k  name -> path")
    for s in sources:
        mix = f"lam={s.lam:g}" if s.kind == "parquet" else f"k={s.k:g}"
        print(f"{s.kind:7s} {s.train_pct:5.2f} {s.pct_mode:13s} {mix:8s} {s.name} -> {s.path}")


def self_test() -> None:
    encoding_smoke_test()
    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
        "8/8/8/3k4/8/3K4/4P3/8 w - - 0 1",
        "8/8/3k4/8/3K4/8/4p3/8 b - - 0 1",
    ]
    x, mask = encode_positions(fens, workers=1)
    y = stm_targets(np.asarray([0.5, 0.5, 0.7, 0.3], dtype=np.float32), mask)
    model = NNUE()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
    pred = model(torch.from_numpy(x))
    loss = nn.BCELoss()(pred, torch.from_numpy(y))
    loss.backward()
    opt.step()
    clamp_weights_(model)
    assert torch.isfinite(loss)
    assert pred.shape == (4,)
    print(f"OK: NNU3 train self-test loss={float(loss):.6f}, arch={architecture_dict()}")


def main() -> None:
    args = build_parser().parse_args()
    if args.self_test:
        self_test()
        return

    sources = args.sources or infra.sources_from_config()
    if args.list_sources:
        _print_sources(sources)
        return
    if not sources:
        raise SystemExit("no active sources: enable shared DATASETS/BIN_DATASETS or pass --source")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _device(args.device)
    print(f"device={device}  architecture=799->256->64->1 NNU3")
    _print_sources(sources)

    model, start_epoch, lr, optimizer_state = _load_model(args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    loss_fn = nn.BCELoss(reduction="mean")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=min(1e-6, lr * 0.1)
    )

    for epoch0 in range(start_epoch, start_epoch + args.epochs):
        epoch = epoch0 + 1
        t0 = time.time()
        loss_sum = mae_sum = 0.0
        seen = 0
        chunks = 0
        rng = np.random.default_rng(args.seed + epoch)
        for source_name, (x_np, y_np) in iter_sources(sources, "train", epoch, args):
            l, m, n = _train_chunk(model, optimizer, loss_fn, x_np, y_np,
                                   args.batch_size, device, rng)
            loss_sum += l
            mae_sum += m
            seen += n
            chunks += 1
            if chunks % 20 == 0:
                print(f"epoch {epoch:03d}  chunks={chunks:4d}  rows={seen:,}  "
                      f"loss={loss_sum/max(1,seen):.6f}  source={source_name}")
        if seen == 0:
            raise SystemExit("training epoch produced zero positions")

        metrics = {
            "train_loss": loss_sum / seen,
            "train_mae": mae_sum / seen,
            "train_rows": seen,
        }
        if args.val_every > 0 and epoch % args.val_every == 0:
            vl, vm, vn = _validate(model, sources, args, device, epoch)
            metrics.update(val_loss=vl, val_mae=vm, val_rows=vn)
        scheduler.step()
        out = _save_checkpoint(args, model, optimizer, epoch, metrics)
        print(f"epoch {epoch:03d}  train={metrics['train_loss']:.6f}/{metrics['train_mae']:.6f}  "
              f"val={metrics.get('val_loss', float('nan')):.6f}/{metrics.get('val_mae', float('nan')):.6f}  "
              f"rows={seen:,}  lr={optimizer.param_groups[0]['lr']:.3e}  "
              f"sec={time.time()-t0:.1f}  ckpt={out}")


if __name__ == "__main__":
    main()
