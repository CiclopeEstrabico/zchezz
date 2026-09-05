#!/usr/bin/env python3
"""Retarget inherited v4.03 infrastructure defaults to the active v3.22 line.

This intentionally changes only configuration/default wiring. It does not alter
engine search or NNUE code. The v4.x native net-vs-net fast paths remain NNU4
specific; NNU3 engines use the UCI paths in run_arena/run_selfplay/tournament.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace(path: str, old: str, new: str, *, required: bool = True) -> int:
    p = ROOT / path
    text = p.read_text(encoding="utf-8")
    n = text.count(old)
    if required and n == 0:
        raise SystemExit(f"{path}: expected text not found: {old!r}")
    if n:
        p.write_text(text.replace(old, new), encoding="utf-8")
        print(f"{path}: {n} replacement(s): {old!r} -> {new!r}")
    return n


# Python test/arena wrappers: v3.22 is now the default code-under-test.
replace("tests/run_arena.py", "zchezz_v402", "zchezz_v322")
replace("tests/run_arena.py", '"gcc", "-O3", "-ffast-math", "-D_GNU_SOURCE", "-std=c11", "-mavxvnni", "-mavx2",',
        '"gcc", "-O3", "-ffast-math", "-D_GNU_SOURCE", "-std=c11", "-mavx2",')
replace("tests/run_arena.py", "net:<path.nnu4>", "net:<weights-file>", required=False)

replace("tests/run_selfplay.py", r"engine\c\zchezz_v401\zchezz.exe", r"engine\c\zchezz_v322\zchezz.exe")
replace("tests/run_selfplay.py", "Zchezz-v401", "Zchezz-v322")

replace("tests/run_tournament.py", r"engine\c\zchezz_v402\zchezz.exe", r"engine\c\zchezz_v322\zchezz.exe")
replace("tests/run_tournament.py", "Zchezz-v401", "Zchezz-v322")

# Quick regression preset: current candidate versus the long-lived v3.14 baseline.
replace("tests/run_tournament_quick.py", r"engine\c\zchezz_v401\zchezz.exe", r"engine\c\zchezz_v322\zchezz.exe")
replace("tests/run_tournament_quick.py", '"label":    "v401-1T"', '"label":    "v322-1T"')
replace("tests/run_tournament_quick.py", r"engine\c\zchezz_v400\zchezz.exe", r"engine\c\zchezz_v314\zchezz.exe")
replace("tests/run_tournament_quick.py", '"label":    "v400-1T"', '"label":    "v314-1T"')
replace("tests/run_tournament_quick.py", "Default: v4.00 (engine under test) vs v3.14 (previous stable baseline)",
        "Default: v3.22 (engine under test) vs v3.14 (long-lived stable baseline)")
replace("tests/run_tournament_quick.py", "The two engine folders under engine/c/ are v4.00 and v3.14, which is the\npair this preset compares.",
        "The default engine pair is v3.22 and v3.14; CLI overrides can select any two builds.")
replace("tests/run_tournament_quick.py", "#  CONFIGURATION — v4.01 vs v4.00 sanity check (1T, 100 games)",
        "#  CONFIGURATION — v3.22 vs v3.14 sanity check (1T)")
replace("tests/run_tournament_quick.py", "# ── v4.01 (trained) vs v4.00 (placeholder weights) sanity check ───────────────",
        "# ── v3.22 candidate vs v3.14 stable baseline ───────────────────────────────")

# Native selfplay remains useful for NNU4. For NNU3, make the limitation explicit
# and retarget its source defaults only when the active engine has the native-net API.
# The general v3.22 selfplay path is tests/run_selfplay.py (persistent UCI workers).
replace("tests/run_selfplay_native.py", "make ENGINE=v402 selfplay", "make ENGINE=v403 selfplay")
replace("tests/run_selfplay_native.py", 'ENGINE_DIR = os.path.join(REPO_ROOT, "engine", "c", "zchezz_v402")',
        'ENGINE_DIR = os.path.join(REPO_ROOT, "engine", "c", "zchezz_v403")')
replace("tests/run_selfplay_native.py", '"-mavxvnni", "-mavx2",', '"-mavx2",')
replace("tests/run_selfplay_native.py", 'os.path.join("..", "c", "zchezz_v402")',
        'os.path.join("..", "c", "zchezz_v403")')
replace("tests/run_selfplay_native.py", "v4.02: absolute path", "NNU4 host: absolute path", required=False)

# Termux default follows the active release line.
replace("engine/build/build_termux.sh", 'VERSION="${1:-v401}"', 'VERSION="${1:-v322}"')

# Canonical test runner must use the explicit active-engine marker, not numeric max.
replace("tests/run_tests.py", "    latest_version,\n", "    active_version,\n")
replace("tests/run_tests.py", "version = args.version or latest_version()", "version = args.version or active_version()")

# Guardrail: no stale candidate default should survive in the main entry points.
checks = {
    "tests/run_arena.py": ["zchezz_v402"],
    "tests/run_selfplay.py": [r"engine\c\zchezz_v401\zchezz.exe", "Zchezz-v401"],
    "tests/run_tournament.py": [r"engine\c\zchezz_v402\zchezz.exe", "Zchezz-v401"],
    "engine/build/build_termux.sh": ['VERSION="${1:-v401}"'],
}
for path, needles in checks.items():
    text = (ROOT / path).read_text(encoding="utf-8")
    for needle in needles:
        if needle in text:
            raise SystemExit(f"{path}: stale default survived: {needle!r}")

print("v3.22 infrastructure retarget complete")
