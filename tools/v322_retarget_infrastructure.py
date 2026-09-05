#!/usr/bin/env python3
"""Retarget inherited infrastructure defaults to the active v3.22 line.

Only configuration/default wiring changes here. The v4.x native net-vs-net
fast paths require the NNU4 NnueNet API and therefore stay compiled against
v403. NNU3 engines such as v322 use architecture-neutral UCI paths for
arena/tournament and tests/run_selfplay.py's persistent UCI workers.

The script is intentionally idempotent: CI may run it after the defaults have
already been materialized into the branch.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace(path: str, old: str, new: str, *, required: bool = True) -> int:
    p = ROOT / path
    text = p.read_text(encoding="utf-8")
    n = text.count(old)
    if n == 0:
        if new in text:
            print(f"{path}: already materialized: {new!r}")
            return 0
        if required:
            raise SystemExit(f"{path}: expected old/new text not found: {old!r} / {new!r}")
        return 0
    p.write_text(text.replace(old, new), encoding="utf-8")
    print(f"{path}: {n} replacement(s): {old!r} -> {new!r}")
    return n


# Arena wrapper: HEAD/default players are v322, but the shared native arena
# binary remains hosted by v403 because arena.c's in-process `net:` player uses
# the NNU4-only NnueNet API. UCI players are architecture-neutral.
replace("tests/run_arena.py",
        'ENGINE_DIR_FOR_HEAD = os.path.join(REPO_ROOT, "engine", "c", "zchezz_v402")',
        'ENGINE_DIR_FOR_HEAD = os.path.join(REPO_ROOT, "engine", "c", "zchezz_v322")')
replace("tests/run_arena.py", r'uci:engine\c\zchezz_v402\zchezz.exe',
        'uci:engine/c/zchezz_v322/zchezz.exe')
replace("tests/run_arena.py", '"gcc", "-O3", "-ffast-math", "-D_GNU_SOURCE", "-std=c11", "-mavxvnni", "-mavx2",',
        '"gcc", "-O3", "-ffast-math", "-D_GNU_SOURCE", "-std=c11", "-mavx2",')
# Native arena host/fallback: upgrade old v402 references to the current NNU4 host.
replace("tests/run_arena.py", '"-I" + os.path.join("..", "c", "zchezz_v402")',
        '"-I" + os.path.join("..", "c", "zchezz_v403")')
replace("tests/run_arena.py", 'os.path.join("..", "c", "zchezz_v402", "board.c")',
        'os.path.join("..", "c", "zchezz_v403", "board.c")')
replace("tests/run_arena.py", 'os.path.join("..", "c", "zchezz_v402", "search.c")',
        'os.path.join("..", "c", "zchezz_v403", "search.c")')
replace("tests/run_arena.py", 'os.path.join("..", "c", "zchezz_v402", "nnue.c")',
        'os.path.join("..", "c", "zchezz_v403", "nnue.c")')
replace("tests/run_arena.py", '["make", "ENGINE=v402", "arena"]', '["make", "ENGINE=v403", "arena"]')
replace("tests/run_arena.py", '["mingw32-make", "ENGINE=v402", "arena"]', '["mingw32-make", "ENGINE=v403", "arena"]')
replace("tests/run_arena.py", "net:<path.nnu4>", "net:<weights-file>", required=False)
replace("tests/run_arena.py", r'RESULTS_DIR              = r"tests\arena_results"',
        'RESULTS_DIR              = "tests/arena_results"')
replace("tests/run_arena.py", r'OPENING_FOLDER           = r"openings\lines"',
        'OPENING_FOLDER           = "openings/lines"')
replace("tests/run_arena.py", "own engine/c/zchezz_v400/ (the current v4.00 dev folder",
        "own engine/c/zchezz_v322/ (the active v3.22 folder", required=False)

# Persistent UCI self-play is architecture-neutral and becomes the v322 default.
replace("tests/run_selfplay.py", r"engine\c\zchezz_v401\zchezz.exe", "engine/c/zchezz_v322/zchezz.exe")
replace("tests/run_selfplay.py", "Zchezz-v401", "Zchezz-v322")
replace("tests/run_selfplay.py", "self.path      = os.path.abspath(path)",
        'self.path      = os.path.abspath(path.replace("\\\\", os.sep))')
replace("tests/run_selfplay.py",
        '    subprocess.run(["powershell", "-Command", cmd], capture_output=True)',
        '    if os.name == "nt":\n        subprocess.run(["powershell", "-Command", cmd], capture_output=True)')
replace("tests/run_selfplay.py", r'RESULTS_DIR         = r"tests\selfplay_results"',
        'RESULTS_DIR         = "tests/selfplay_results"')
replace("tests/run_selfplay.py", r'OPENING_FOLDER      = r"openings\lines"',
        'OPENING_FOLDER      = "openings/lines"')

# Full tournament defaults to v322 vs the stable v314 reference. Normalize
# inherited Windows-style paths so the same config works on Linux CI.
replace("tests/run_tournament.py", r"engine\c\zchezz_v402\zchezz.exe", "engine/c/zchezz_v322/zchezz.exe")
replace("tests/run_tournament.py", "Zchezz-v401", "Zchezz-v322")
replace("tests/run_tournament.py", r"engine\c\zchezz_v314\zchezz.exe", "engine/c/zchezz_v314/zchezz.exe")
replace("tests/run_tournament.py", "self.path     = os.path.abspath(path)",
        'self.path     = os.path.abspath(path.replace("\\\\", os.sep))')
replace("tests/run_tournament.py", r'OPENING_FOLDER       = r"openings\lines"',
        'OPENING_FOLDER       = "openings/lines"')
replace("tests/run_tournament.py", r'RESULTS_DIR          = r"tests\complete_results"',
        'RESULTS_DIR          = "tests/complete_results"')
replace("tests/run_tournament.py", "currently\n  v4.00 vs v3.14", "currently\n  v3.22 vs v3.14", required=False)

# Quick regression preset: current candidate versus long-lived v3.14 baseline.
replace("tests/run_tournament_quick.py", r"engine\c\zchezz_v401\zchezz.exe", "engine/c/zchezz_v322/zchezz.exe")
replace("tests/run_tournament_quick.py", '"label":    "v401-1T"', '"label":    "v322-1T"')
replace("tests/run_tournament_quick.py", r"engine\c\zchezz_v400\zchezz.exe", "engine/c/zchezz_v314/zchezz.exe")
replace("tests/run_tournament_quick.py", '"label":    "v400-1T"', '"label":    "v314-1T"')
replace("tests/run_tournament_quick.py", "self.path     = os.path.abspath(path)",
        'self.path     = os.path.abspath(path.replace("\\\\", os.sep))')
replace("tests/run_tournament_quick.py", r'OPENING_FOLDER       = r"openings\lines"',
        'OPENING_FOLDER       = "openings/lines"')
replace("tests/run_tournament_quick.py", r'RESULTS_DIR          = r"tests\quick_results"',
        'RESULTS_DIR          = "tests/quick_results"')
replace("tests/run_tournament_quick.py", "Default: v4.00 (engine under test) vs v3.14 (previous stable baseline)",
        "Default: v3.22 (engine under test) vs v3.14 (long-lived stable baseline)")
replace("tests/run_tournament_quick.py", "The two engine folders under engine/c/ are v4.00 and v3.14, which is the\npair this preset compares.",
        "The default engine pair is v3.22 and v3.14; CLI overrides can select any two builds.")
replace("tests/run_tournament_quick.py", "#  CONFIGURATION — v4.01 vs v4.00 sanity check (1T, 100 games)",
        "#  CONFIGURATION — v3.22 vs v3.14 sanity check (1T)")
replace("tests/run_tournament_quick.py", "# ── v4.01 (trained) vs v4.00 (placeholder weights) sanity check ───────────────",
        "# ── v3.22 candidate vs v3.14 stable baseline ───────────────────────────────")

# Native selfplay remains the NNU4 fast path. Keep it on the current NNU4 host
# and remove mandatory VNNI so the helper can build on ordinary AVX2 machines.
replace("tests/run_selfplay_native.py", "make ENGINE=v402 selfplay", "make ENGINE=v403 selfplay")
replace("tests/run_selfplay_native.py", 'ENGINE_DIR = os.path.join(REPO_ROOT, "engine", "c", "zchezz_v402")',
        'ENGINE_DIR = os.path.join(REPO_ROOT, "engine", "c", "zchezz_v403")')
replace("tests/run_selfplay_native.py", '"-mavxvnni", "-mavx2",', '"-mavx2",')
replace("tests/run_selfplay_native.py", 'os.path.join("..", "c", "zchezz_v402")',
        'os.path.join("..", "c", "zchezz_v403")')
replace("tests/run_selfplay_native.py", "v4.02: absolute path", "NNU4 host: absolute path", required=False)

# Termux and canonical test runner follow the explicit active engine marker.
replace("engine/build/build_termux.sh", 'VERSION="${1:-v401}"', 'VERSION="${1:-v322}"')
replace("tests/run_tests.py", "    latest_version,\n", "    active_version,\n")
replace("tests/run_tests.py", "version = args.version or latest_version()", "version = args.version or active_version()")

# Guardrails: stale candidate/default paths must not survive. v403 references
# are permitted only in native NNU4 tool-host wiring.
checks = {
    "tests/run_arena.py": [r'OPENING_FOLDER           = r"openings\lines"'],
    "tests/run_selfplay.py": [r"engine\c\zchezz_v401\zchezz.exe", "Zchezz-v401", r'OPENING_FOLDER      = r"openings\lines"'],
    "tests/run_tournament.py": [r"engine\c\zchezz_v402\zchezz.exe", "Zchezz-v401", r'OPENING_FOLDER       = r"openings\lines"'],
    "tests/run_tournament_quick.py": [r"engine\c\zchezz_v401\zchezz.exe", r"engine\c\zchezz_v400\zchezz.exe", r'OPENING_FOLDER       = r"openings\lines"'],
    "engine/build/build_termux.sh": ['VERSION="${1:-v401}"'],
}
for path, needles in checks.items():
    text = (ROOT / path).read_text(encoding="utf-8")
    for needle in needles:
        if needle in text:
            raise SystemExit(f"{path}: stale default survived: {needle!r}")

arena = (ROOT / "tests/run_arena.py").read_text(encoding="utf-8")
for needle in ("ENGINE_DIR_FOR_HEAD = os.path.join(REPO_ROOT, \"engine\", \"c\", \"zchezz_v322\")",
               '["make", "ENGINE=v403", "arena"]',
               'OPENING_FOLDER           = "openings/lines"'):
    if needle not in arena:
        raise SystemExit(f"tests/run_arena.py: expected architecture/path routing missing: {needle}")

print("v3.22 infrastructure retarget complete")
