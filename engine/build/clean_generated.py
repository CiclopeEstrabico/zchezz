#!/usr/bin/env python3
"""Delete only known generated build outputs.

This script intentionally does not traverse or delete datasets, tablebases,
openings, checkpoints, or other ignored local resources.
"""
from __future__ import annotations

import argparse
from pathlib import Path

LOCAL_GENERATED = [
    "test_engine_invariants.exe",
    "arena.exe",
    "selfplay.exe",
    "ga_tune.exe",
]

ENGINE_GENERATED = [
    "zchezz.exe",
    "zchezz_debug.exe",
    "zchezz_sanitize.exe",
    "zchezz_wasm.js",
    "zchezz_wasm.wasm",
    "zchezz_bundle.html",
]

def remove_file(path: Path) -> None:
    if path.is_file() or path.is_symlink():
        path.unlink()
        print(f"removed {path}")

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-dir", required=True)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    engine = (here / args.engine_dir).resolve()
    for name in LOCAL_GENERATED:
        remove_file(here / name)
    for name in ENGINE_GENERATED:
        remove_file(engine / name)
    for path in here.glob("*.o"):
        remove_file(path)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
