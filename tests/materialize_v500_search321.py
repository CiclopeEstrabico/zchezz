#!/usr/bin/env python3
"""Create a temporary v5 engine carrying only the proven v3.21 search bundle.

The experiment is intentionally narrow: evaluation, NNUE, TT layout and all
other v5 code stay unchanged.  We port the three search-policy differences that
are present in current v3.23 but absent from the v5 line:
  * no-TT-move depth reduction starts at depth 3 instead of 4;
  * wider LMP table from the confirmed v3.21 bundle;
  * LMR divisor 1.35 instead of 1.5.

Use on a benchmark/materialized copy, never on the accepted baseline directly.
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"{label}: expected exactly one source match, found {n}")
    return text.replace(old, new)


def materialize(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    p = dst / "search.c"
    s = p.read_text(encoding="utf-8")

    s = replace_once(
        s,
        "#define TUNE_LMR_DIVISOR_DEFAULT              1.5",
        "#define TUNE_LMR_DIVISOR_DEFAULT             1.35",
        "LMR divisor",
    )
    s = replace_once(
        s,
        "static const int lmp_limit[8] = {0,10,18,26,36,48,62,78};",
        "static const int lmp_limit[8] = {0,12,22,32,44,58,74,94};",
        "LMP table",
    )
    s = replace_once(
        s,
        "if (!pv_move.from && !pv_move.to && depth>=4 && !in_check && depth-1>=2) depth--;",
        "if (!pv_move.from && !pv_move.to && depth>=3 && !in_check && depth-1>=2) depth--;",
        "no-TT reduction",
    )

    p.write_text(s, encoding="utf-8")
    print(f"materialized v3.21 search bundle: {src} -> {dst}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    a = ap.parse_args()
    materialize(a.src, a.dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
