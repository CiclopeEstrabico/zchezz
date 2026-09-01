"""Protect the shared WebAssembly build contract."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "utils"))
from repo_paths import build_root, wasm_source  # noqa: E402


def _runner():
    spec = importlib.util.spec_from_file_location(
        "zchezz_web_runner", ROOT / "tests" / "run_tests.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_shared_wasm_template_exists():
    path = build_root() / "zchezz_wasm.html"
    assert path.is_file(), (
        "run `python tools/promote_wasm_template.py` and commit the result"
    )


def test_wasm_source_is_shared():
    expected = build_root() / "zchezz_wasm.html"
    assert wasm_source("v401") == expected
    assert wasm_source("v403") == expected


def test_makefile_exports_frontend_reset_symbol():
    text = (ROOT / "engine" / "build" / "Makefile").read_text(encoding="utf-8")
    assert '"_nnue_reset_global"' in text
    assert 'WASM_TEMPLATE = zchezz_wasm.html' in text
    assert "bundle_shared.py" in text


def test_browser_e2e_targets_generated_bundle():
    module = _runner()
    command = next(
        step.command
        for step in module.profiles("v403", "v402")["web"]
        if step.name == "browser e2e"
    )
    assert "--html" in command
    assert "zchezz_bundle.html" in command
