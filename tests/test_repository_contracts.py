"""Executable repository conventions that must not drift silently."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_core_files_exist():
    required = [
        "AGENTS.md", "CLAUDE.md", "pyproject.toml",
        "docs/testing.md", "docs/build.md", "docs/engine-contracts.md",
        "tests/run_tests.py", "utils/repo_paths.py", "tools/check_repo.py",
    ]
    missing = [path for path in required if not (ROOT / path).is_file()]
    assert not missing, f"required repository files are missing: {missing}"


def test_v403_core_has_expected_sources():
    root = ROOT / "engine" / "c" / "zchezz_v403"
    required = ["main.c", "board.c", "board.h", "search.c", "search.h", "nnue.c", "nnue.h", "syzygy.c", "tbprobe.c", "book.c"]
    missing = [name for name in required if not (root / name).is_file()]
    assert not missing, f"v403 core files are missing: {missing}"


def test_no_absolute_c_zchezz_in_active_python():
    offenders = []
    for top in ("tests", "train", "tools", "utils"):
        base = ROOT / top
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="replace")
            if re.search(r"(?i)c:\\\\" + "zchezz", text):
                offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"hard-coded Windows repository roots: {offenders}"


def test_piece_sets_are_complete_when_present():
    roots = [ROOT / "pieces", ROOT / "engine" / "build" / "pieces"]
    expected = {f"{color}{piece}.svg" for color in "wb" for piece in "KQRBNP"}
    for root in roots:
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if child.is_dir():
                present = {path.name for path in child.glob("*.svg")}
                assert expected <= present, f"{child.relative_to(ROOT)} missing {sorted(expected - present)}"


def test_shared_makefile_has_no_stale_default():
    makefile = ROOT / "engine" / "build" / "Makefile"
    if not makefile.is_file():
        return
    text = makefile.read_text(encoding="utf-8")
    assert not re.search(r"^ENGINE\s*\?=\s*v402\b", text, re.MULTILINE), "shared Makefile still defaults to v402 on the v403 branch"

