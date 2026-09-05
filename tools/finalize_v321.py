#!/usr/bin/env python3
"""Materialize a confirmed v3.21 bundle and stamp the engine version.

This script is intentionally deterministic: it applies exactly one of the
already-tested final bundles to engine/c/zchezz_v321 and then changes only the
v3.20 identity strings to v3.21.
"""
from pathlib import Path
import argparse
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("variant", choices=["strong_no_qs", "strong_bundle", "strong_plus_prefetch"])
p.add_argument("--root", default="engine/c/zchezz_v321")
a = p.parse_args()
root = Path(a.root)

subprocess.run([
    sys.executable,
    "tools/v321_apply_final_bundle.py",
    a.variant,
    "--root", str(root),
], check=True)

main = root / "main.c"
s = main.read_text(encoding="utf-8")n = s.count('#define ENGINE_VERSION "3.20"')
if n != 1:
    raise SystemExit(f"expected one v3.20 ENGINE_VERSION, got {n}")
s = s.replace('#define ENGINE_VERSION "3.20"', '#define ENGINE_VERSION "3.21"', 1)
s = s.replace('/* main.c — Zchezz v3.20 UCI engine', '/* main.c — Zchezz v3.21 UCI engine', 1)
main.write_text(s, encoding="utf-8")
print(f"materialized Zchezz v3.21: {a.variant}")
