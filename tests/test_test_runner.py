"""Validate test-profile wiring without launching engines."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("zchezz_run_tests", ROOT / "tests" / "run_tests.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

def _steps(profile: str):
    return MODULE.profiles("v403", "v402")[profile]

def _step(profile: str, name: str):
    return next(step for step in _steps(profile) if step.name == name)

def test_smoke_builds_before_black_box_tests():
    names = [step.name for step in _steps("smoke")]
    assert names.index("native build") < names.index("perft smoke")
    assert names.index("native build") < names.index("UCI smoke")

def test_smoke_perft_is_reduced_depth():
    command = _step("smoke", "perft smoke").command
    assert "--max-depth" in command
    index = command.index("--max-depth")
    assert command[index + 1] == str(MODULE.SMOKE_PERFT_MAX_DEPTH)

def test_full_has_one_full_perft_and_no_web_checks():
    names = [step.name for step in _steps("full")]
    assert names.count("perft full") == 1
    assert "browser static" not in names
    assert "book contracts" not in names
    assert "bundle" not in names

def test_web_builds_before_generated_artifact_checks():
    names = [step.name for step in _steps("web")]
    assert names.index("bundle") < names.index("browser static")
    assert names.index("bundle") < names.index("book contracts")
    assert names.index("bundle") < names.index("browser e2e")

def test_regression_uses_selected_candidate_and_baseline():
    command = _step("regression", "quick H2H").command
    text = " ".join(command)
    assert "zchezz_v403" in text
    assert "zchezz_v402" in text
    assert "--engine-a" in command and "--engine-b" in command
    assert "--threads" in command

def test_release_executes_sanitized_binary_after_build():
    names = [step.name for step in _steps("release")]
    assert names.index("ASan+UBSan build") < names.index("ASan+UBSan UCI smoke")
    assert any("zchezz_sanitize.exe" in p for p in _step("release", "ASan+UBSan UCI smoke").command)

def test_native_profiles_use_platform_make_program(monkeypatch):
    monkeypatch.setattr(MODULE.os, "name", "nt")
    assert MODULE.make_program() == "mingw32-make"

def test_skip_is_explicit_for_missing_web_prerequisite(tmp_path, monkeypatch):
    step = MODULE.Step(
        "example",
        ("does-not-matter",),
        when_file="definitely/missing.file",
        skip_reason="example prerequisite",
    )
    reason = MODULE.prerequisite_skip(step)
    assert reason and "missing file" in reason and "example prerequisite" in reason


def test_web_uses_shared_template_and_generated_bundle():
    bundle_step = _step("web", "bundle")
    assert bundle_step.when_file == "engine/build/zchezz_wasm.html"
    browser = _step("web", "browser e2e")
    assert "--html" in browser.command
    assert "zchezz_bundle.html" in browser.command
