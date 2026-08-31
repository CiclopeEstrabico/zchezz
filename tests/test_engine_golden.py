"""Small deterministic golden contracts independent of Elo/performance noise."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "utils"))
from repo_paths import engine_executable, latest_version  # noqa: E402

GOLDEN = json.loads((ROOT / "tests" / "data" / "golden_engine.json").read_text(encoding="utf-8"))


def test_golden_schema_is_supported():
    assert GOLDEN["schema"] == 1


def test_required_uci_option_inventory_when_engine_is_built():
    exe = engine_executable(latest_version())
    if not exe.is_file():
        pytest.skip(f"engine binary not built: {exe}")
    proc = subprocess.run([str(exe)], cwd=exe.parent, input="uci\nquit\n", capture_output=True, text=True, timeout=10)
    assert proc.returncode == 0
    output = proc.stdout
    missing = [name for name in GOLDEN["required_uci_options"] if f"option name {name}" not in output]
    assert not missing, f"UCI option contract changed: missing {missing}"

