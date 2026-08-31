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


def _step(profile: str, name: str):
    steps = MODULE.profiles("v403", "v402")[profile]
    return next(step for step in steps if step.name == name)


def test_smoke_builds_candidate_before_black_box_tests():
    names = [step.name for step in MODULE.profiles("v403", "v402")["smoke"]]
    assert names.index("native build") < names.index("perft smoke")
    assert names.index("native build") < names.index("UCI smoke")


def test_regression_uses_selected_candidate_and_baseline():
    command = _step("regression", "quick H2H").command
    text = " ".join(command)
    assert "zchezz_v403" in text
    assert "zchezz_v402" in text
    assert "--engine-a" in command and "--engine-b" in command
    assert "--threads" in command


def test_web_profile_passes_explicit_version_to_browser():
    command = _step("web", "browser e2e").command
    assert "--version" in command
    assert "v403" in command
    assert "--headless" in command


def test_release_executes_sanitized_binary_after_build():
    names = [step.name for step in MODULE.profiles("v403", "v402")["release"]]
    assert names.index("ASan+UBSan build") < names.index("ASan+UBSan UCI smoke")
    command = _step("release", "ASan+UBSan UCI smoke").command
    assert any("zchezz_sanitize.exe" in part for part in command)


def test_native_profiles_use_the_platform_make_program(monkeypatch):
    monkeypatch.setattr(MODULE.os, "name", "nt")
    assert MODULE.make_program() == "mingw32-make"
