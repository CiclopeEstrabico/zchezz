#!/usr/bin/env python3
"""Zchezz v5.00 full-accumulator training loop.

This is the v5-specific orchestration layer. It intentionally does NOT use
LC0 as the main source and does NOT reintroduce the 31 hand-made features.
The loop is:

  1. Generate positions with zchezz_v500 native self-play.
  2. Re-label a sampled set with the stronger v3.14 engine.
  3. Train HalfKP/full-accumulator on teacher cp + real result.
  4. Export NNU4 candidate weights.
  5. Gate the candidate against v3.14 before promotion.

Why teacher distillation here?
The v4.03 experiments showed that simply adding more LC0 outcome data did
not increase Elo. v3.14 is already the concrete strength target we need to
recover, so it is a useful teacher: v5 sees positions from its OWN search
distribution while learning the value surface of the stronger engine.
Real results remain a separate signal and are blended only at training time.

Default result blend by generation:
    g0: 0.15 result + 0.85 v3.14 teacher
    g1: 0.20 result + 0.80 v3.14 teacher
    g2: 0.25 result + 0.75 v3.14 teacher
    g3+:0.30 result + 0.70 v3.14 teacher

The blend is deliberately modest: pure teacher imitation cannot exceed the
teacher easily, but pure TD(1) was noisy in the LC0 experiments. Later
generations add more outcome signal while keeping the stronger teacher as
an anchor.

Commands:
    python train/v500_pipeline.py plan --generation 0
    python train/v500_pipeline.py generate --generation 0
    python train/v500_pipeline.py label --generation 0
    python train/v500_pipeline.py train --generation 0
    python train/v500_pipeline.py export --generation 0
    python train/v500_pipeline.py gate --generation 0
    python train/v500_pipeline.py cycle --generation 0

The pipeline never auto-promotes a candidate weight file. A candidate must
win its Elo gate first; promotion remains an explicit action.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "train"
ENGINE = "v500"
BASELINE = "v314"
ENGINE_DIR = ROOT / "engine/c/zchezz_v500"
BASELINE_DIR = ROOT / "engine/c/zchezz_v314"
BUILD_DIR = ROOT / "engine/build"
SELFPLAY_ROOT = ROOT / "data/v500_selfplay"
TEACHER_ROOT = ROOT / "data/v500_teacher_v314"
CKPT_ROOT = ROOT / "checkpoints/v500"
ARTIFACT_ROOT = ROOT / "artifacts/v500-training"
OPENING_FOLDER = ROOT / "openings/lines"

# Generation defaults. These are quality-oriented, not CI-smoke values.
GAMES = 20_000
SELFPLAY_SHARDS = 4
CONCURRENCY = max(1, (os.cpu_count() or 4) - 1)
MOVETIME_MS = 50
MULTIPV = 4
TEMPERATURE = 0.80
TEMP_PLIES = 16
TEMP_FINAL = 0.0          # exact argmax after the diverse opening phase
TEMP_ARGMAX_EPS = 0.0
MAX_PLIES = 320
TT_MB = 16.0
BOOK_PORTION = 0.97
RANDOM_PLIES = 6
SEED = 500_000

TEACHER_NODES = 10_000
TEACHER_MAX_ROWS = 100_000
TEACHER_WORKERS = max(1, min(8, (os.cpu_count() or 4) // 2))
TEACHER_SHARD_ROWS = 10_000

EPOCHS = 24
BATCH_SIZE = 65_536
LR = 5e-4
TRANSFER_LR = 3e-5
WEIGHT_DECAY = 1e-4
VAL_EVERY = 1
REPLAY_GENERATIONS = 3

GATE_GAMES = 256
GATE_MOVETIME_MS = 100
GATE_NODES = 50_000


def result_blend(generation: int) -> float:
    return min(0.30, 0.15 + 0.05 * max(0, generation))


def _make_program() -> str:
    for name in ("mingw32-make", "make"):
        if shutil.which(name):
            return name
    raise SystemExit("GNU make not found (tried mingw32-make and make)")


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(str(x) for x in cmd))
    subprocess.run([str(x) for x in cmd], cwd=str(cwd) if cwd else None, check=True)


def gen_dir(g: int) -> Path:
    return SELFPLAY_ROOT / f"gen{g:02d}"


def teacher_dir(g: int) -> Path:
    return TEACHER_ROOT / f"gen{g:02d}"


def ckpt_dir(g: int) -> Path:
    return CKPT_ROOT / f"gen{g:02d}"


def artifact_dir(g: int) -> Path:
    return ARTIFACT_ROOT / f"gen{g:02d}"


def latest_checkpoint(path: Path) -> Path:
    files = list(path.glob("*.pt"))
    if not files:
        raise SystemExit(f"no checkpoint found in {path}")
    return max(files, key=lambda p: p.stat().st_mtime)


def build_selfplay() -> Path:
    """Build the native generator against zchezz_v500.

    SAMPLE_ENGINE_VERSION is passed explicitly because sample.h is shared by
    all historical engine directories; provenance must say 5.00 for v5 data.
    """
    make = _make_program()
    run([
        make, "-C", str(BUILD_DIR), "ENGINE=v500",
        "ARCH_FLAGS=-mavx2 -DSAMPLE_ENGINE_VERSION=500",
        "selfplay",
    ])
    exe = BUILD_DIR / "selfplay.exe"
    if not exe.is_file():
        raise SystemExit(f"selfplay build did not produce {exe}")
    return exe


def build_engine(version: str) -> Path:
    make = _make_program()
    run([make, "-C", str(BUILD_DIR), f"ENGINE={version}", "ARCH_FLAGS=-mavx2", "native"])
    exe = ROOT / f"engine/c/zchezz_{version}/zchezz.exe"
    if not exe.is_file():
        raise SystemExit(f"engine build did not produce {exe}")
    return exe


def generate(args: argparse.Namespace) -> None:
    exe = build_selfplay()
    out_dir = gen_dir(args.generation)
    out_dir.mkdir(parents=True, exist_ok=True)
    games_per_shard = (args.games + args.shards - 1) // args.shards

    remaining = args.games
    for shard in range(args.shards):
        n = min(games_per_shard, remaining)
        if n <= 0:
            break
        dst = out_dir / f"selfplay_{shard:03d}.bin"
        cmd = [
            str(exe),
            "--games", str(n),
            "--threads", str(args.threads),
            "--movetime", str(args.movetime),
            "--multipv", str(args.multipv),
            "--temperature", str(args.temperature),
            "--temp-scale", "100",
            "--temp-plies", str(args.temp_plies),
            "--temp-final", str(args.temp_final),
            "--temp-argmax-eps", str(args.temp_argmax_eps),
            "--max-plies", str(args.max_plies),
            "--seed", str(args.seed + shard),
            "--tt-mb", str(args.tt_mb),
            "--nnue", str(ENGINE_DIR / "nnue_weights.bin"),
            "--out", str(dst),
            "--openings", str(OPENING_FOLDER),
            "--opening-mode", "book+random",
            "--random-plies", str(args.random_plies),
            "--book-portion", str(args.book_portion),
            "--same-opening-twice",
        ]
        run(cmd, cwd=ROOT)
        remaining -= n


def label(args: argparse.Namespace) -> None:
    teacher = build_engine(BASELINE)
    src = str(gen_dir(args.generation) / "*.bin")
    dst = teacher_dir(args.generation)
    dst.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable, str(TRAIN_DIR / "labeling/label_with_teacher.py"),
        "--input", src,
        "--output", str(dst),
        "--teacher", str(teacher),
        "--nodes", str(args.teacher_nodes),
        "--max-rows", str(args.teacher_rows),
        "--workers", str(args.teacher_workers),
        "--shard-rows", str(args.teacher_shard_rows),
        "--seed", str(args.seed + 314),
        "--generation", str(args.generation),
    ], cwd=ROOT)


def train_generation(args: argparse.Namespace) -> None:
    """Call the canonical trainer with explicit SourceSpec objects.

    This intentionally bypasses train_nnue.py's --source string parser:
    SourceSpec has a `lam` field, but the current parser fails to propagate
    a supplied lam and would silently fall back to 0.0. Constructing the
    objects here makes the v5 result/teacher blend unambiguous.
    """
    if str(TRAIN_DIR) not in sys.path:
        sys.path.insert(0, str(TRAIN_DIR))
    import train_nnue as tn

    ns = tn.build_arg_parser().parse_args([])
    ns.ckpt_dir = str(ckpt_dir(args.generation))
    ns.dataset_name = f"v500_g{args.generation:02d}_teacher314"
    ns.epochs = args.epochs
    ns.batch_size = args.batch_size
    ns.lr = args.lr
    ns.transfer_lr = args.transfer_lr
    ns.weight_decay = args.weight_decay
    ns.workers = args.train_workers
    ns.device = args.device
    ns.val_every = args.val_every
    ns.resample_each_epoch = True
    ns.encode_cache = True
    ns.show_config = False

    if args.generation > 0 and ckpt_dir(args.generation - 1).exists():
        ns.checkpoint_source = str(ckpt_dir(args.generation - 1))
    else:
        ns.checkpoint_source = "new"

    lam = result_blend(args.generation) if args.result_blend is None else args.result_blend
    ns.sources = []
    oldest = max(0, args.generation - args.replay_generations + 1)
    for g in range(args.generation, oldest - 1, -1):
        src = teacher_dir(g)
        if not src.exists() or not list(src.glob("*.parquet")):
            if g == args.generation:
                raise SystemExit(f"teacher-labelled dataset missing: {src}")
            continue
        age = args.generation - g
        pct = 1.0 if age == 0 else max(0.25, 0.5 ** age)
        ns.sources.append(tn.SourceSpec(
            kind="parquet",
            path=str(src),
            target_col="cp",
            lam=float(lam),
            train_pct=float(pct),
            val_frac=0.02,
            name=f"v500_teacher314_g{g:02d}",
            pct_mode="sample-files",
        ))

    ckpt_dir(args.generation).mkdir(parents=True, exist_ok=True)
    print(f"[v500] training generation={args.generation} result_blend={lam:.2f} sources={len(ns.sources)}")
    for s in ns.sources:
        print(f"[v500]   {s.name}: pct={s.train_pct:.2f} lam={s.lam:.2f} path={s.path}")
    tn.train(ns)


def export_generation(args: argparse.Namespace) -> Path:
    ckpt = latest_checkpoint(ckpt_dir(args.generation))
    out_dir = artifact_dir(args.generation)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "nnue_weights.bin"
    run([
        sys.executable, str(TRAIN_DIR / "export_nnu4.py"),
        "--ckpt", str(ckpt),
        "--dst", str(dst),
    ], cwd=ROOT)
    print(f"[v500] candidate weights: {dst}")
    return dst


def gate(args: argparse.Namespace) -> None:
    weights = artifact_dir(args.generation) / "nnue_weights.bin"
    if not weights.is_file():
        weights = export_generation(args)

    tag = f"v500cand{args.generation:02d}"
    cand_dir = ROOT / f"engine/c/zchezz_{tag}"
    if cand_dir.exists():
        shutil.rmtree(cand_dir)
    shutil.copytree(ENGINE_DIR, cand_dir)
    shutil.copy2(weights, cand_dir / "nnue_weights.bin")

    try:
        candidate = build_engine(tag)
        baseline = build_engine(BASELINE)
        make = _make_program()
        run([make, "-C", str(BUILD_DIR), "ENGINE=v500", "ARCH_FLAGS=-mavx2", "arena"])

        out = artifact_dir(args.generation)
        out.mkdir(parents=True, exist_ok=True)
        common = [
            sys.executable, str(ROOT / "tests/run_arena.py"),
            "--player", f"uci:{candidate}",
            "--player", f"uci:{baseline}",
            "--threads", str(args.gate_threads),
            "--max-plies", "300",
            "--openings", str(OPENING_FOLDER),
            "--opening-plies", "0",
            "--tt-mb", "16",
        ]
        run(common + [
            "--games", str(args.gate_games), "--movetime", "0", "--nodes", str(args.gate_nodes),
            "--seed", str(args.seed + 900),
            "--json", str(out / "candidate_vs_v314_nodes.json"),
            "--pgn", str(out / "candidate_vs_v314_nodes.pgn"),
        ], cwd=ROOT)
        run(common + [
            "--games", str(args.gate_games), "--movetime", str(args.gate_movetime), "--nodes", "0",
            "--seed", str(args.seed + 901),
            "--json", str(out / "candidate_vs_v314_time.json"),
            "--pgn", str(out / "candidate_vs_v314_time.pgn"),
        ], cwd=ROOT)
    finally:
        if cand_dir.exists():
            shutil.rmtree(cand_dir)


def show_plan(args: argparse.Namespace) -> None:
    lam = result_blend(args.generation) if args.result_blend is None else args.result_blend
    print("Zchezz v5.00 training plan")
    print(f"  generation       : {args.generation}")
    print(f"  architecture     : HalfKP-4Bucket, full accumulator, current H2=32")
    print(f"  manual features  : 0 (the 31 legacy features stay out)")
    print(f"  selfplay         : {args.games:,} games, {args.movetime} ms/move, T={args.temperature} for {args.temp_plies} plies then argmax")
    print(f"  teacher          : v3.14 @ {args.teacher_nodes:,} nodes, sample {args.teacher_rows:,} positions")
    print(f"  target           : {lam:.2f} result + {1.0-lam:.2f} teacher WDL")
    print(f"  replay           : current + up to {args.replay_generations-1} previous generations")
    print(f"  training         : {args.epochs} epochs, lr={args.lr:g}, transfer_lr={args.transfer_lr:g}")
    print(f"  gate             : {args.gate_games} games @ {args.gate_nodes:,} nodes AND {args.gate_movetime} ms vs v3.14")
    print("  promotion        : never automatic")


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--generation", type=int, default=0)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--games", type=int, default=GAMES)
    p.add_argument("--shards", type=int, default=SELFPLAY_SHARDS)
    p.add_argument("--threads", type=int, default=CONCURRENCY)
    p.add_argument("--movetime", type=int, default=MOVETIME_MS)
    p.add_argument("--multipv", type=int, default=MULTIPV)
    p.add_argument("--temperature", type=float, default=TEMPERATURE)
    p.add_argument("--temp-plies", type=int, default=TEMP_PLIES)
    p.add_argument("--temp-final", type=float, default=TEMP_FINAL)
    p.add_argument("--temp-argmax-eps", type=float, default=TEMP_ARGMAX_EPS)
    p.add_argument("--max-plies", type=int, default=MAX_PLIES)
    p.add_argument("--tt-mb", type=float, default=TT_MB)
    p.add_argument("--book-portion", type=float, default=BOOK_PORTION)
    p.add_argument("--random-plies", type=int, default=RANDOM_PLIES)
    p.add_argument("--teacher-nodes", type=int, default=TEACHER_NODES)
    p.add_argument("--teacher-rows", type=int, default=TEACHER_MAX_ROWS)
    p.add_argument("--teacher-workers", type=int, default=TEACHER_WORKERS)
    p.add_argument("--teacher-shard-rows", type=int, default=TEACHER_SHARD_ROWS)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--transfer-lr", type=float, default=TRANSFER_LR)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--train-workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--val-every", type=int, default=VAL_EVERY)
    p.add_argument("--replay-generations", type=int, default=REPLAY_GENERATIONS)
    p.add_argument("--result-blend", type=float, default=None,
                   help="override generation schedule; fraction of real result in target")
    p.add_argument("--gate-games", type=int, default=GATE_GAMES)
    p.add_argument("--gate-nodes", type=int, default=GATE_NODES)
    p.add_argument("--gate-movetime", type=int, default=GATE_MOVETIME_MS)
    p.add_argument("--gate-threads", type=int, default=max(1, min(8, os.cpu_count() or 4)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Zchezz v5.00 full-accumulator train/gate pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "generate", "label", "train", "export", "gate", "cycle"):
        sp = sub.add_parser(name)
        add_common(sp)

    args = parser.parse_args()
    if args.command == "plan":
        show_plan(args)
    elif args.command == "generate":
        generate(args)
    elif args.command == "label":
        label(args)
    elif args.command == "train":
        train_generation(args)
    elif args.command == "export":
        export_generation(args)
    elif args.command == "gate":
        gate(args)
    elif args.command == "cycle":
        generate(args)
        label(args)
        train_generation(args)
        export_generation(args)
        gate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
