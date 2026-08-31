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
    missing = [p for p in required if not (ROOT / p).is_file()]
    assert not missing, f"required repository files are missing: {missing}"

def test_v403_core_has_expected_sources():
    root = ROOT / "engine" / "c" / "zchezz_v403"
    required = [
        "main.c", "board.c", "board.h", "search.c", "search.h",
        "nnue.c", "nnue.h", "syzygy.c", "syzygy.h", "book.c", "book.h",
    ]
    missing = [n for n in required if not (root / n).is_file()]
    assert not missing, f"v403 core files are missing: {missing}"
    tb_c = (root / "tbprobe.c").is_file()
    tb_h = (root / "tbprobe.h").is_file()
    assert tb_c == tb_h, "tbprobe.c and tbprobe.h must either both exist or both be absent"

def test_shared_makefile_supports_clean_checkout_without_fathom():
    text = (ROOT / "engine" / "build" / "Makefile").read_text(encoding="utf-8")
    assert "TB_C" in text and "TB_H" in text
    assert "-DNO_TABLEBASES" in text
    assert "require-tablebases" in text

def test_no_absolute_c_zchezz_in_active_python():
    offenders = []
    pattern = re.compile(r"(?i)c:[\\/]+zchezz")
    for top in ("tests", "train", "tools", "utils"):
        base = ROOT / top
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if path.resolve() == Path(__file__).resolve():
                continue
            if pattern.search(path.read_text(encoding="utf-8", errors="replace")):
                offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"hard-coded Windows repository roots: {offenders}"

def test_piece_sets_are_complete_when_present():
    roots = [ROOT / "pieces", ROOT / "engine" / "build" / "pieces"]
    expected = {f"{c}{p}.svg" for c in "wb" for p in "KQRBNP"}
    for root in roots:
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if child.is_dir():
                present = {p.name for p in child.glob("*.svg")}
                assert expected <= present, f"{child.relative_to(ROOT)} missing {sorted(expected - present)}"

def test_shared_makefile_has_no_stale_default():
    text = (ROOT / "engine" / "build" / "Makefile").read_text(encoding="utf-8")
    assert not re.search(r"^ENGINE\s*\?=\s*v402\b", text, re.MULTILINE)

def test_makefile_respects_caller_compiler():
    text = (ROOT / "engine" / "build" / "Makefile").read_text(encoding="utf-8")
    assert re.search(r"^CC\s*\?=\s*gcc\b", text, re.MULTILINE)
