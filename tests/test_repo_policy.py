"""Unit tests for repository path-policy helpers."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "utils"))

from repo_policy import runtime_string_literals  # noqa: E402

def test_comments_and_docstrings_do_not_count_as_runtime_literals(tmp_path):
    local_root = "C:" + "\\" + "Zchezz"
    path = tmp_path / "sample.py"
    path.write_text(
        f'"""{local_root} in a module docstring."""\n'
        f'# {local_root} in a comment\n'
        'VALUE = "relative/path"\n',
        encoding="utf-8",
    )
    values = runtime_string_literals(path)
    assert values == ["relative/path"]

def test_executable_string_literal_is_returned(tmp_path):
    local_root = "C:" + "\\" + "Zchezz"
    path = tmp_path / "sample.py"
    path.write_text(f'ROOT = r"{local_root}"\n', encoding="utf-8")
    values = runtime_string_literals(path)
    assert any("Zchezz" in value for value in values)
