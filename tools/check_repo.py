#!/usr/bin/env python3
"""Fast repository pre-flight check. No compilation and no chess games."""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "utils"))
from repo_paths import available_versions  # noqa: E402

ERRORS: list[str] = []
WARNINGS: list[str] = []


def error(message: str) -> None:
    ERRORS.append(message)


def warning(message: str) -> None:
    WARNINGS.append(message)


def check_required() -> None:
    for name in ("AGENTS.md", "CLAUDE.md", "docs/testing.md", "tests/run_tests.py", "utils/repo_paths.py"):
        if not (ROOT / name).is_file():
            error(f"missing required file: {name}")
    if not available_versions():
        error("no engine/c/zchezz_v* directories found")


def check_agent_mirrors() -> None:
    a = ROOT / "AGENTS.md"
    c = ROOT / "CLAUDE.md"
    if a.is_file() and c.is_file():
        abody = "\n".join(a.read_text(encoding="utf-8").splitlines()[1:]).strip()
        cbody = "\n".join(c.read_text(encoding="utf-8").splitlines()[1:]).strip()
        if abody != cbody:
            error("AGENTS.md and CLAUDE.md rule bodies differ")
    sa = ROOT / ".agents" / "skills" / "writing-rules" / "SKILL.md"
    sc = ROOT / ".claude" / "skills" / "writing-rules" / "SKILL.md"
    if sa.is_file() and sc.is_file() and sa.read_bytes() != sc.read_bytes():
        error("writing-rules skill copies differ")


def check_python_syntax() -> None:
    for top in ("tests", "tools", "utils"):
        base = ROOT / top
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            try:
                ast.parse(path.read_text(encoding="utf-8", errors="strict"), filename=str(path))
            except Exception as exc:
                error(f"Python syntax/encoding error in {path.relative_to(ROOT)}: {exc}")


def check_hardcoded_roots() -> None:
    pattern = re.compile(r"(?i)c:\\\\" + "zchezz")
    for top in ("tests", "tools", "utils", "train"):
        base = ROOT / top
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if pattern.search(path.read_text(encoding="utf-8", errors="replace")):
                error(f"hard-coded Windows repository root in {path.relative_to(ROOT)}")


def check_piece_sets() -> None:
    expected = {f"{c}{p}.svg" for c in "wb" for p in "KQRBNP"}
    for root in (ROOT / "pieces", ROOT / "engine" / "build" / "pieces"):
        if not root.is_dir():
            continue
        for folder in root.iterdir():
            if not folder.is_dir():
                continue
            present = {p.name for p in folder.glob("*.svg")}
            missing = sorted(expected - present)
            if missing:
                error(f"{folder.relative_to(ROOT)} missing piece SVGs: {missing}")


def main() -> int:
    check_required()
    check_agent_mirrors()
    check_python_syntax()
    check_hardcoded_roots()
    check_piece_sets()
    for item in WARNINGS:
        print(f"WARN: {item}")
    for item in ERRORS:
        print(f"ERROR: {item}")
    if ERRORS:
        print(f"FAILED: {len(ERRORS)} repository issue(s)")
        return 1
    print("PASS: repository pre-flight checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

