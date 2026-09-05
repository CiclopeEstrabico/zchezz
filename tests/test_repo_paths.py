"""Unit tests for canonical repository path/version logic."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("zchezz_repo_paths", ROOT / "utils" / "repo_paths.py")
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def test_normalize_version():
    assert MOD.normalize_version("403") == "v403"
    assert MOD.normalize_version("v403") == "v403"
    assert MOD.normalize_version("zchezz_v403") == "v403"
    assert MOD.normalize_version(403) == "v403"


def test_numeric_ordering_not_lexicographic(tmp_path, monkeypatch):
    for name in ("zchezz_v99", "zchezz_v314", "zchezz_v403", "not_an_engine"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(MOD, "engine_root", lambda: tmp_path)
    assert MOD.available_versions() == ["v99", "v314", "v403"]
    assert MOD.latest_version() == "v403"
    assert MOD.previous_version("v403") == "v314"


def test_previous_version_uses_actual_available_versions(tmp_path, monkeypatch):
    for name in ("zchezz_v314", "zchezz_v401", "zchezz_v402", "zchezz_v403"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(MOD, "engine_root", lambda: tmp_path)
    assert MOD.previous_version("v403") == "v402"
    assert MOD.previous_version("v402") == "v401"


def test_active_version_prefers_environment(tmp_path, monkeypatch):
    for name in ("zchezz_v314", "zchezz_v322", "zchezz_v403"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(MOD, "engine_root", lambda: tmp_path)
    monkeypatch.setenv("ZCHEZZ_ENGINE", "322")
    assert MOD.active_version() == "v322"


def test_active_version_uses_repository_marker(tmp_path, monkeypatch):
    engine_c = tmp_path / "engine" / "c"
    engine_c.mkdir(parents=True)
    for name in ("zchezz_v314", "zchezz_v322", "zchezz_v403"):
        (engine_c / name).mkdir()
    (tmp_path / "engine" / "ACTIVE_ENGINE").write_text("v322\n", encoding="utf-8")
    monkeypatch.delenv("ZCHEZZ_ENGINE", raising=False)
    monkeypatch.setattr(MOD, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(MOD, "engine_root", lambda: engine_c)
    assert MOD.active_version() == "v322"


def test_active_version_rejects_stale_marker(tmp_path, monkeypatch):
    engine_c = tmp_path / "engine" / "c"
    engine_c.mkdir(parents=True)
    (engine_c / "zchezz_v403").mkdir()
    (tmp_path / "engine" / "ACTIVE_ENGINE").write_text("v322\n", encoding="utf-8")
    monkeypatch.delenv("ZCHEZZ_ENGINE", raising=False)
    monkeypatch.setattr(MOD, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(MOD, "engine_root", lambda: engine_c)
    try:
        MOD.active_version()
    except FileNotFoundError as exc:
        assert "v322" in str(exc)
    else:
        raise AssertionError("stale active-engine marker must fail")
