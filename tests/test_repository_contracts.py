from __future__ import annotations

import re
from pathlib import Path

from repo_policy import LEGACY_ABSOLUTE_ROOT_ALLOWLIST, absolute_root_files

ROOT = Path(__file__).resolve().parents[1]


def test_no_tracked_generated_outputs_in_source_tree():
    offenders = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if any(part.startswith(".") for part in path.relative_to(ROOT).parts):
            continue
        if rel.startswith("artifacts/"):
            continue
        if path.suffix.lower() in {".exe", ".o", ".obj", ".so", ".dll", ".dylib", ".pyc"}:
            offenders.append(rel)
    assert not offenders, f"generated binary/object outputs tracked in source tree: {offenders}"


def test_no_python_cache_directories():
    offenders = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob("__pycache__") if p.is_dir()]
    assert not offenders, f"python cache directories present: {offenders}"


def test_active_engine_marker_points_to_existing_version():
    marker = ROOT / "engine" / "ACTIVE_ENGINE"
    assert marker.is_file()
    version = marker.read_text(encoding="utf-8").strip()
    assert re.fullmatch(r"v\d+", version)
    assert (ROOT / "engine" / "c" / f"zchezz_{version}").is_dir()


def test_no_unregistered_absolute_machine_paths():
    offenders = sorted(absolute_root_files(ROOT) - LEGACY_ABSOLUTE_ROOT_ALLOWLIST)
    assert not offenders, (
        "unregistered machine-local absolute paths: "
        f"{offenders}. Use repo-relative paths/repo_paths.py."
    )


def test_absolute_root_debt_is_explicit():
    active = absolute_root_files(ROOT)
    unregistered = active - LEGACY_ABSOLUTE_ROOT_ALLOWLIST
    assert not unregistered


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


def test_shared_makefile_matches_active_engine_marker():
    active = (ROOT / "engine" / "ACTIVE_ENGINE").read_text(encoding="utf-8").strip()
    text = (ROOT / "engine" / "build" / "Makefile").read_text(encoding="utf-8")
    match = re.search(r"^ENGINE\s*\?=\s*(v\d+)\b", text, re.MULTILINE)
    assert match, "shared Makefile must define an ENGINE ?= vXXX default"
    assert match.group(1) == active, (
        f"Makefile default {match.group(1)} must match engine/ACTIVE_ENGINE {active}"
    )


def test_makefile_respects_caller_compiler():
    text = (ROOT / "engine" / "build" / "Makefile").read_text(encoding="utf-8")
    assert "ifeq ($(origin CC),default)" in text
    assert re.search(r"^CC\s*=\s*gcc\b", text, re.MULTILINE)


def test_cleanup_is_delegated_to_safe_script():
    text = (ROOT / "engine" / "build" / "Makefile").read_text(encoding="utf-8")
    assert "clean_generated.py" in text
    assert (ROOT / "engine" / "build" / "clean_generated.py").is_file()
