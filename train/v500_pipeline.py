#!/usr/bin/env python3
"""Zchezz v5.00 full-accumulator training and Elo-gating pipeline.

Design constraints:
- HalfKP-4Bucket/full accumulator stays in place.
- The 31 legacy hand-made features stay OUT.
- Positions come from v5 self-play, so training follows the student's real
  search distribution.
- Values come primarily from the stronger v3.14 teacher, blended with the
  actual game result.
- Generation 0 WARM-STARTS from the current v5 NNU4 weights. It never throws
  away the existing network and starts randomly unless explicitly forced by
  using train_nnue.py outside this pipeline.
- Recent generations are replayed to reduce forgetting.
- Candidates are never auto-promoted: they must survive fixed-node and
  fixed-time Elo gates against v3.14.

Default target blend:
  g0  15% result + 85% v3.14 teacher
  g1  20% result + 80% v3.14 teacher
  g2  25% result + 75% v3.14 teacher
  g3+ 30% result + 70% v3.14 teacher

Commands:
  python train/v500_pipeline.py plan --generation 0
  python train/v500_pipeline.py generate --generation 0
  python train/v500_pipeline.py label --generation 0
  python train/v500_pipeline.py train --generation 0
  python train/v500_pipeline.py export --generation 0
  python train/v500_pipeline.py gate --generation 0
  python train/v500_pipeline.py cycle --generation 0
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "train"
ENGINE_DIR = ROOT / "engine/c/zchezz_v500"
BUILD_DIR = ROOT / "engine/build"
SELFPLAY_ROOT = ROOT / "data/v500_selfplay"
TEACHER_ROOT = ROOT / "data/v500_teacher_v314"
CKPT_ROOT = ROOT / "checkpoints/v500"
ARTIFACT_ROOT = ROOT / "artifacts/v500-training"
OPENING_FOLDER = ROOT / "openings/lines"
OPENING_EXTS = {".epd", ".pgn"}
BASELINE = "v314"

GAMES = 20_000
SELFPLAY_SHARDS = 4
CONCURRENCY = max(1, (os.cpu_count() or 4) - 1)
MOVETIME_MS = 50
MULTIPV = 4
TEMPERATURE = 0.80
TEMP_PLIES = 16
TEMP_FINAL = 0.0
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
TEACHER_TIMEOUT = 30.0

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
GATE_OPENINGS = 384


def result_blend(generation: int) -> float:
    return min(0.30, 0.15 + 0.05 * max(0, generation))


def gen_dir(g: int) -> Path:
    return SELFPLAY_ROOT / f"gen{g:02d}"


def teacher_dir(g: int) -> Path:
    return TEACHER_ROOT / f"gen{g:02d}"


def ckpt_dir(g: int) -> Path:
    return CKPT_ROOT / f"gen{g:02d}"


def artifact_dir(g: int) -> Path:
    return ARTIFACT_ROOT / f"gen{g:02d}"


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=str(cwd) if cwd else None, check=True)


def _make_program() -> str:
    for name in ("mingw32-make", "make"):
        if shutil.which(name):
            return name
    raise SystemExit("GNU make not found (tried mingw32-make and make)")


def latest_checkpoint(path: Path) -> Path:
    files = list(path.glob("*.pt"))
    if not files:
        raise SystemExit(f"no checkpoint found in {path}")
    return max(files, key=lambda p: p.stat().st_mtime)


def _opening_corpus_available() -> bool:
    if not OPENING_FOLDER.is_dir():
        return False
    return any(p.is_file() and p.suffix.lower() in OPENING_EXTS
               for p in OPENING_FOLDER.rglob("*"))


def _selfplay_opening_args(args: argparse.Namespace) -> list[str]:
    if _opening_corpus_available():
        return [
            "--openings", str(OPENING_FOLDER),
            "--opening-mode", "all",
            "--book-portion", str(args.book_portion),
            "--random-plies", str(args.random_plies),
            "--same-opening-twice",
        ]
    print(f"[v500] opening corpus not found at {OPENING_FOLDER}; using random plies", flush=True)
    return [
        "--opening-mode", "random",
        "--random-plies", str(args.random_plies),
        "--same-opening-twice",
    ]


def _make_gate_openings(path: Path, seed: int, count: int = GATE_OPENINGS) -> Path:
    import chess

    rng = random.Random(seed)
    rows: list[str] = []
    seen: set[str] = set()
    while len(rows) < count:
        b = chess.Board()
        ok = True
        for _ in range(rng.randint(8, 18)):
            moves = list(b.legal_moves)
            if not moves:
                ok = False
                break
            quiet = [m for m in moves if not b.is_capture(m)]
            b.push(rng.choice(quiet if quiet and rng.random() < 0.75 else moves))
            if b.is_game_over(claim_draw=True):
                ok = False
                break
        if not ok:
            continue
        key = " ".join(b.fen().split()[:4])
        if key not in seen:
            seen.add(key)
            rows.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def build_selfplay() -> Path:
    make = _make_program()
    run([
        make, "-C", str(BUILD_DIR), "ENGINE=v500",
        "ARCH_FLAGS=-mavx2 -DSAMPLE_ENGINE_VERSION=500", "selfplay",
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
    out = gen_dir(args.generation)
    out.mkdir(parents=True, exist_ok=True)
    per_shard = (args.games + args.shards - 1) // args.shards
    opening_args = _selfplay_opening_args(args)
    remaining = args.games
    for shard in range(args.shards):
        n = min(per_shard, remaining)
        if n <= 0:
            break
        cmd = [
            str(exe), "--games", str(n), "--threads", str(args.threads),
            "--movetime", str(args.movetime), "--multipv", str(args.multipv),
            "--temperature", str(args.temperature), "--temp-scale", "100",
            "--temp-plies", str(args.temp_plies), "--temp-final", str(args.temp_final),
            "--temp-argmax-eps", str(args.temp_argmax_eps),
            "--max-plies", str(args.max_plies), "--seed", str(args.seed + shard),
            "--tt-mb", str(args.tt_mb),
            "--nnue", str(ENGINE_DIR / "nnue_weights.bin"),
            "--out", str(out / f"selfplay_{shard:03d}.bin"),
            *opening_args,
        ]
        run(cmd, cwd=ROOT)
        remaining -= n


def label(args: argparse.Namespace) -> None:
    teacher = build_engine(BASELINE)
    dst = teacher_dir(args.generation)
    dst.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable, str(TRAIN_DIR / "labeling/label_with_teacher.py"),
        "--input", str(gen_dir(args.generation) / "*.bin"),
        "--output", str(dst), "--teacher", str(teacher),
        "--nodes", str(args.teacher_nodes), "--max-rows", str(args.teacher_rows),
        "--workers", str(args.teacher_workers),
        "--shard-rows", str(args.teacher_shard_rows),
        "--timeout", str(args.teacher_timeout), "--seed", str(args.seed + 314),
        "--generation", str(args.generation),
    ], cwd=ROOT)


def _bootstrap_checkpoint() -> Path:
    """Materialize current engine NNU4 weights as a PyTorch transfer checkpoint."""
    src = ENGINE_DIR / "nnue_weights.bin"
    if not src.is_file():
        raise SystemExit(f"cannot warm-start: current v5 weights missing: {src}")
    dst = CKPT_ROOT / "bootstrap" / "v500_current_nnu4.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable, str(TRAIN_DIR / "import_nnu4.py"),
        "--src", str(src), "--dst", str(dst),
    ], cwd=ROOT)
    return dst


def _resolve_training_checkpoint(generation: int) -> str:
    """Resume current generation, transfer previous generation, else bootstrap NNU4."""
    current = ckpt_dir(generation)
    if current.exists() and list(current.glob("*.pt")):
        return str(current)
    if generation > 0:
        previous = ckpt_dir(generation - 1)
        if previous.exists() and list(previous.glob("*.pt")):
            return str(previous)
    return str(_bootstrap_checkpoint())


def train_generation(args: argparse.Namespace) -> None:
    """Fine-tune HalfKP using v3.14 teacher labels plus real game outcomes.

    SourceSpec objects are constructed directly because the generic
    train_nnue.py --source parser currently does not propagate its `lam`
    field. This keeps the requested teacher/result blend explicit.
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
    ns.checkpoint_source = _resolve_training_checkpoint(args.generation)

    lam = result_blend(args.generation) if args.result_blend is None else args.result_blend
    if not 0.0 <= lam <= 1.0:
        raise SystemExit(f"--result-blend must be in [0,1], got {lam}")

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
            kind="parquet", path=str(src), target_col="cp", lam=float(lam),
            train_pct=float(pct), val_frac=0.02,
            name=f"v500_teacher314_g{g:02d}", pct_mode="sample-files",
        ))

    ckpt_dir(args.generation).mkdir(parents=True, exist_ok=True)
    print(f"[v500] checkpoint source: {ns.checkpoint_source}")
    print(f"[v500] generation={args.generation} target={lam:.2f} result + {1-lam:.2f} teacher")
    for src in ns.sources:
        print(f"[v500] source {src.name}: pct={src.train_pct:.2f} lam={src.lam:.2f}")
    tn.train(ns)


def export_generation(args: argparse.Namespace) -> Path:
    ckpt = latest_checkpoint(ckpt_dir(args.generation))
    out = artifact_dir(args.generation)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / "nnue_weights.bin"
    run([
        sys.executable, str(TRAIN_DIR / "export_nnu4.py"),
        "--ckpt", str(ckpt), "--dst", str(dst),
    ], cwd=ROOT)
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
        openings = _make_gate_openings(out / "gate_openings.epd", args.seed + 731)
        common = [
            sys.executable, str(ROOT / "tests/run_arena.py"),
            "--player", f"uci:{candidate}", "--player", f"uci:{baseline}",
            "--threads", str(args.gate_threads), "--max-plies", "300",
            "--openings", str(openings), "--opening-plies", "0", "--tt-mb", "16",
        ]
        run(common + [
            "--games", str(args.gate_games), "--movetime", "0",
            "--nodes", str(args.gate_nodes), "--seed", str(args.seed + 900),
            "--json", str(out / "candidate_vs_v314_nodes.json"),
            "--pgn", str(out / "candidate_vs_v314_nodes.pgn"),
        ], cwd=ROOT)
        run(common + [
            "--games", str(args.gate_games), "--movetime", str(args.gate_movetime),
            "--nodes", "0", "--seed", str(args.seed + 901),
            "--json", str(out / "candidate_vs_v314_time.json"),
            "--pgn", str(out / "candidate_vs_v314_time.pgn"),
        ], cwd=ROOT)
    finally:
        if cand_dir.exists():
            shutil.rmtree(cand_dir)


def show_plan(args: argparse.Namespace) -> None:
    lam = result_blend(args.generation) if args.result_blend is None else args.result_blend
    openings = "97% local book + random" if _opening_corpus_available() else "random plies (no tracked corpus)"
    print("Zchezz v5.00 training plan")
    print(f"  generation       : {args.generation}")
    print("  architecture     : HalfKP-4Bucket, full accumulator, H2=32")
    print("  manual features  : 0")
    print("  initialization   : current v5 NNU4 weights (warm start), then previous generation")
    print(f"  selfplay         : {args.games:,} games @ {args.movetime} ms, MultiPV={args.multipv}")
    print(f"  openings         : {openings}")
    print(f"  teacher          : v3.14 @ {args.teacher_nodes:,} nodes, {args.teacher_rows:,} labels")
    print(f"  target           : {lam:.2f} result + {1-lam:.2f} teacher")
    print(f"  replay           : {args.replay_generations} generations")
    print(f"  training         : {args.epochs} epochs, lr={args.lr:g}, transfer_lr={args.transfer_lr:g}")
    print(f"  gate             : {args.gate_games} games @ {args.gate_nodes:,} nodes + {args.gate_movetime} ms vs v3.14")
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
    p.add_argument("--teacher-timeout", type=float, default=TEACHER_TIMEOUT)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--transfer-lr", type=float, default=TRANSFER_LR)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--train-workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--val-every", type=int, default=VAL_EVERY)
    p.add_argument("--replay-generations", type=int, default=REPLAY_GENERATIONS)
    p.add_argument("--result-blend", type=float, default=None)
    p.add_argument("--gate-games", type=int, default=GATE_GAMES)
    p.add_argument("--gate-nodes", type=int, default=GATE_NODES)
    p.add_argument("--gate-movetime", type=int, default=GATE_MOVETIME_MS)
    p.add_argument("--gate-threads", type=int, default=max(1, min(8, os.cpu_count() or 4)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Zchezz v5.00 full-accumulator train/gate pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "generate", "label", "train", "export", "gate", "cycle"):
        add_common(sub.add_parser(name))
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
    else:
        generate(args)
        label(args)
        train_generation(args)
        export_generation(args)
        gate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
