"""Repository policy helpers shared by checks and tests."""
from __future__ import annotations

import ast
import re
from pathlib import Path

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

ABSOLUTE_ZCHEZZ_RE = re.compile(r"(?i)c:[\\/]+zchezz")

def runtime_string_literals(path: Path) -> list[str]:
    """Return string literals that are not module/function/class docstrings."""
    source = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(source, filename=str(path))
    docstrings: set[int] = set()

    def mark(node: ast.AST) -> None:
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            docstrings.add(id(body[0].value))

    mark(tree)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            mark(node)

    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]

def absolute_root_files(root: Path) -> set[str]:
    offenders: set[str] = set()
    for top in ("tests", "train", "tools", "utils"):
        base = root / top
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            try:
                values = runtime_string_literals(path)
            except SyntaxError:
                continue
            if any(ABSOLUTE_ZCHEZZ_RE.search(value) for value in values):
                offenders.add(path.relative_to(root).as_posix())
    return offenders

def new_absolute_root_files(root: Path) -> set[str]:
    return absolute_root_files(root) - LEGACY_ABSOLUTE_ROOT_ALLOWLIST
