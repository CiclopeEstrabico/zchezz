"""Executable repository conventions that must not drift silently."""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Transitional debt register.
#
# These files pre-date the repository-professionalization work and may still
# contain machine-local C:\Zchezz defaults. Keeping the list explicit prevents
# new hard-coded roots from being introduced while allowing the old scripts to
# be migrated deliberately, one at a time, with behavior-preserving tests.
#
# Remove an entry as soon as that file is migrated to repo-relative paths.
LEGACY_ABSOLUTE_ROOT_ALLOWLIST = {
    "tests/debug_game.py",
    "tests/debug_lc0_abs.py",
    "tests/debug_lc0_bits.py",
    "tests/debug_lc0_layout.py",
    "tests/debug_lc0_polarity.py",
    "tests/debug_lc0_polarity2.py",
    "tests/debug_lc0_stm.py",
    "tests/run_tournament.py",
    "tests/test_move_parsing.py",
    "tests/test_uci.py",
    "train/export_nnu4.py",
    "train/make_pt_from_nnu4.py",
    "train/train_nnue.py",
    "train/labeling/import_lc0.py",
}


def test_core_files_exist():
    required = [
        "AGENTS.md",
        "CLAUDE.md",
        "pyproject.toml",
        "docs/testing.md",
        "docs/build.md",
        "docs/engine-contracts.md",
        "tests/run_tests.py",
        "utils/repo_paths.py",
        "tools/check_repo.py",
    ]
    missing = [path for path in required if not (ROOT / path).is_file()]
    assert not missing, f"required repository files are missing: {missing}"


def test_v403_core_has_expected_sources():
    """Require tracked v403 core and validate optional Fathom as a pair."""
    root = ROOT / "engine" / "c" / "zchezz_v403"
    required = [
        "main.c",
        "board.c",
        "board.h",
        "search.c",
        "search.h",
        "nnue.c",
        "nnue.h",
        "syzygy.c",
        "syzygy.h",
        "book.c",
        "book.h",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    assert not missing, f"v403 core files are missing: {missing}"

    tb_c = (root / "tbprobe.c").is_file()
    tb_h = (root / "tbprobe.h").is_file()
    assert tb_c == tb_h, (
        "Fathom integration is incomplete: tbprobe.c and tbprobe.h must "
        "either both exist or both be absent"
    )


def test_shared_makefile_supports_clean_checkout_without_fathom():
    text = (ROOT / "engine" / "build" / "Makefile").read_text(encoding="utf-8")
    assert "TB_C" in text and "TB_H" in text
    assert "-DNO_TABLEBASES" in text
    assert "require-tablebases" in text


def _runtime_string_literals(path: Path) -> list[str]:
    """Return Python string literals, excluding comments and docstrings.

    A regex over raw source incorrectly flags comments and examples. Parsing
    the AST makes this contract about executable/configuration strings instead.
    """
    source = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        # Syntax validity belongs to the normal Python sanity checks. Avoid
        # hiding a syntax error behind this path-policy test.
        return []

    docstring_nodes: set[int] = set()

    def mark_docstring(node: ast.AST) -> None:
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            docstring_nodes.add(id(body[0].value))

    mark_docstring(tree)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            mark_docstring(node)

    values: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstring_nodes
        ):
            values.append(node.value)
    return values


def test_no_new_absolute_c_zchezz_in_active_python():
    """Freeze existing absolute-root debt and forbid any new occurrences."""
    pattern = re.compile(r"(?i)c:[\\/]+zchezz")
    offenders: list[str] = []

    for top in ("tests", "train", "tools", "utils"):
        base = ROOT / top
        if not base.exists():
            continue

        for path in base.rglob("*.py"):
            rel = path.relative_to(ROOT).as_posix()

            # The policy test itself necessarily contains the matching regex.
            if path.resolve() == Path(__file__).resolve():
                continue

            has_runtime_root = any(
                pattern.search(value) for value in _runtime_string_literals(path)
            )
            if has_runtime_root and rel not in LEGACY_ABSOLUTE_ROOT_ALLOWLIST:
                offenders.append(rel)

    assert not offenders, (
        "new hard-coded Windows repository roots are not allowed: "
        f"{sorted(offenders)}. Use repo-relative paths/repo_paths.py, or add "
        "a temporary allowlist entry only when preserving a verified legacy "
        "workflow is necessary."
    )


def test_legacy_absolute_root_allowlist_does_not_grow_stale():
    """Every allowlisted entry must still name a real Python file.

    This catches renamed/deleted files and keeps the debt register honest.
    It intentionally does not require every entry to still contain a hard-coded
    root: once a file is migrated, remove its entry in the same change.
    """
    missing = [
        rel for rel in sorted(LEGACY_ABSOLUTE_ROOT_ALLOWLIST)
        if not (ROOT / rel).is_file()
    ]
    assert not missing, (
        "absolute-root legacy allowlist references missing files: "
        f"{missing}"
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
                    f"{child.relative_to(ROOT)} missing "
                    f"{sorted(expected - present)}"
                )


def test_shared_makefile_has_no_stale_default():
    makefile = ROOT / "engine" / "build" / "Makefile"
    if not makefile.is_file():
        return

    text = makefile.read_text(encoding="utf-8")
    assert not re.search(
        r"^ENGINE\s*\?=\s*v402\b",
        text,
        re.MULTILINE,
    ), "shared Makefile still defaults to v402 on the v403 branch"


def test_makefile_respects_caller_compiler():
    text = (ROOT / "engine" / "build" / "Makefile").read_text(encoding="utf-8")
    assert re.search(
        r"^CC\s*\?=\s*gcc\b",
        text,
        re.MULTILINE,
    ), "Makefile must use CC ?= gcc so CI can actually exercise clang"
