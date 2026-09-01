"""Executable repository conventions that must not drift silently."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "utils"))

from repo_policy import (  # noqa: E402
    LEGACY_ABSOLUTE_ROOT_ALLOWLIST,
    absolute_root_files,
    new_absolute_root_files,
)

def test_core_files_exist():
    required = [
        "AGENTS.md", "CLAUDE.md", "pyproject.toml",
        "docs/testing.md", "docs/build.md", "docs/engine-contracts.md",
        "docs/regression-testing.md", "docs/release-process.md",
        "tests/run_tests.py", "utils/repo_paths.py", "utils/repo_policy.py",
        "tools/check_repo.py",
        "engine/build/zchezz_wasm.html",
        "engine/build/bundle_shared.py",
        "tools/promote_wasm_template.py",
    ]
    missing = [path for path in required if not (ROOT / path).is_file()]
    assert not missing, f"required repository files are missing: {missing}"

def test_v403_core_has_expected_tracked_sources():
    root = ROOT / "engine" / "c" / "zchezz_v403"
    required = [
        "main.c", "board.c", "board.h", "search.c", "search.h",
        "nnue.c", "nnue.h", "syzygy.c", "syzygy.h", "book.c", "book.h",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    assert not missing, f"v403 core files are missing: {missing}"

def test_fathom_local_files_are_a_complete_pair():
    root = ROOT / "engine" / "c" / "zchezz_v403"
    tb_c = (root / "tbprobe.c").is_file()
    tb_h = (root / "tbprobe.h").is_file()
    assert tb_c == tb_h, (
        "Fathom integration is incomplete: tbprobe.c and tbprobe.h must "
        "either both exist or both be absent"
    )

def test_shared_makefile_supports_both_fathom_modes():
    text = (ROOT / "engine" / "build" / "Makefile").read_text(encoding="utf-8")
    assert "TB_C" in text and "TB_H" in text
    assert "-DNO_TABLEBASES" in text
    assert "require-tablebases" in text
    assert "HAVE_FATHOM" in text

def test_no_new_absolute_c_zchezz_in_active_python():
    offenders = sorted(new_absolute_root_files(ROOT))
    assert not offenders, (
        "new hard-coded Windows repository roots are not allowed: "
        f"{offenders}. Use repo-relative paths/repo_paths.py."
    )

def test_absolute_root_debt_is_explicit():
    active = absolute_root_files(ROOT)
    unregistered = active - LEGACY_ABSOLUTE_ROOT_ALLOWLIST
    assert not unregistered
    missing_allowlisted_paths = [
        rel for rel in sorted(LEGACY_ABSOLUTE_ROOT_ALLOWLIST)
        if not (ROOT / rel).is_file()
    ]
    assert not missing_allowlisted_paths, (
        "legacy path-debt allowlist references missing files: "
        f"{missing_allowlisted_paths}"
    )

def test_piece_sets_are_complete_when_present():
    roots = [ROOT / "pieces", ROOT / "engine" / "build" / "pieces"]
    expected = {f"{color}{piece}.svg" for color in "wb" for piece in "KQRBNP"}
    for root in roots:
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if child.is_dir():
                present = {path.name for path in child.glob("*.svg")}
                assert expected <= present, (
                    f"{child.relative_to(ROOT)} missing {sorted(expected - present)}"
                )

def test_shared_makefile_has_active_v403_default():
    text = (ROOT / "engine" / "build" / "Makefile").read_text(encoding="utf-8")
    assert re.search(r"^ENGINE\s*\?=\s*v403\b", text, re.MULTILINE)
    assert not re.search(r"^ENGINE\s*\?=\s*v402\b", text, re.MULTILINE)

def test_makefile_respects_caller_compiler():
    text = (ROOT / "engine" / "build" / "Makefile").read_text(encoding="utf-8")
    assert "ifeq ($(origin CC),default)" in text
    assert re.search(r"^CC\s*=\s*gcc\b", text, re.MULTILINE)

def test_cleanup_is_delegated_to_safe_script():
    text = (ROOT / "engine" / "build" / "Makefile").read_text(encoding="utf-8")
    assert "clean_generated.py" in text
    assert (ROOT / "engine" / "build" / "clean_generated.py").is_file()
