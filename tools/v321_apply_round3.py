#!/usr/bin/env python3
"""Apply one disposable v3.21 round-3 candidate for A/B screening."""
from pathlib import Path
import argparse

VARIANTS = [
    "qsee_relaxed",
    "qsee_strict",
    "main_see_relaxed",
    "main_see_strict",
    "lmp_stricter",
    "lmp_looser",
    "history_stricter",
    "history_looser",
]

p = argparse.ArgumentParser()
p.add_argument("variant", choices=VARIANTS)
p.add_argument("--root", default="engine/c/zchezz_v321cand")
a = p.parse_args()
path = Path(a.root) / "search.c"


def replace_exact(old: str, new: str, expected: int, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    n = text.count(old)
    if n != expected:
        raise SystemExit(f"{label}: expected {expected} matches, found {n}")
    path.write_text(text.replace(old, new), encoding="utf-8")


if a.variant == "qsee_relaxed":
    replace_exact(
        "if (sv < 0) { moves[i].score = -99999; continue; }",
        "if (sv < -50) { moves[i].score = -99999; continue; }",
        1, a.variant,
    )

elif a.variant == "qsee_strict":
    replace_exact(
        "if (sv < 0) { moves[i].score = -99999; continue; }",
        "if (sv < 30) { moves[i].score = -99999; continue; }",
        1, a.variant,
    )

elif a.variant == "main_see_relaxed":
    replace_exact(
        "int see_thresh = depth<=4 ? -80 : depth<=6 ? -120 : -160;",
        "int see_thresh = depth<=4 ? -40 : depth<=6 ? -80 : -120;",
        2, a.variant,
    )

elif a.variant == "main_see_strict":
    replace_exact(
        "int see_thresh = depth<=4 ? -80 : depth<=6 ? -120 : -160;",
        "int see_thresh = depth<=4 ? -120 : depth<=6 ? -180 : -240;",
        2, a.variant,
    )

elif a.variant == "lmp_stricter":
    replace_exact(
        "static const int lmp_limit[8] = {0,10,18,26,36,48,62,78};",
        "static const int lmp_limit[8] = {0,8,14,21,29,38,50,62};",
        1, a.variant,
    )

elif a.variant == "lmp_looser":
    replace_exact(
        "static const int lmp_limit[8] = {0,10,18,26,36,48,62,78};",
        "static const int lmp_limit[8] = {0,12,22,32,44,58,74,94};",
        1, a.variant,
    )

elif a.variant == "history_stricter":
    replace_exact(
        "int hp_thresh = -4000 * depth;",
        "int hp_thresh = -3000 * depth;",
        1, a.variant,
    )

elif a.variant == "history_looser":
    replace_exact(
        "int hp_thresh = -4000 * depth;",
        "int hp_thresh = -5000 * depth;",
        1, a.variant,
    )

print(f"applied {a.variant}")
