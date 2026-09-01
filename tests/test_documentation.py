"""Verify objective documentation facts against the repository."""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

ACTIVE_DOCS = {
    "architecture.md", "build.md", "engine-contracts.md", "generated-artifacts.md",
    "nnue.md", "regression-testing.md", "release-process.md", "repository-layout.md",
    "syzygy.md", "testing.md", "wasm.md",
}

def _markdown_files():
    yield ROOT / "AGENTS.md"
    yield ROOT / "CLAUDE.md"
    for name in ACTIVE_DOCS:
        yield DOCS / name

def _runner_module():
    spec = importlib.util.spec_from_file_location("zchezz_docs_runner", ROOT / "tests" / "run_tests.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def test_documented_repo_paths_exist():
    missing = set()
    pattern = re.compile(r"`((?:docs|tests|tools|utils|engine|train)/[A-Za-z0-9_./*-]+)`")
    for document in _markdown_files():
        if not document.is_file():
            continue
        text = document.read_text(encoding="utf-8", errors="replace")
        for raw in pattern.findall(text):
            if "*" in raw or raw.endswith("/") or "vXXX" in raw:
                continue
            path = ROOT / raw
            if not path.exists():
                missing.add((document.name, raw))
    assert not missing, "documented literal repository paths do not exist: " + repr(sorted(missing))

def test_required_testing_commands_are_documented():
    text = (DOCS / "testing.md").read_text(encoding="utf-8")
    for command in (
        "python tests/run_tests.py smoke",
        "python tests/run_tests.py full",
        "python tests/run_tests.py web",
        "python tests/run_tests.py regression",
        "python tests/run_tests.py release",
    ):
        assert command in text

def test_every_runner_gate_is_named_in_testing_catalog():
    module = _runner_module()
    names = {
        step.name
        for profile in module.profiles("v403", "v402").values()
        for step in profile
    }
    text = (DOCS / "testing.md").read_text(encoding="utf-8")
    missing = sorted(name for name in names if f"Gate: {name}" not in text)
    assert not missing, f"runner gates missing from docs/testing.md: {missing}"

def test_agents_do_not_describe_v402_as_current_candidate():
    for name in ("AGENTS.md", "CLAUDE.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "Active development candidate: `engine/c/zchezz_v403/`" in text
        assert "Current engine version: `engine/c/zchezz_v402/`" not in text

def test_engine_contract_ids_are_unique():
    text = (DOCS / "engine-contracts.md").read_text(encoding="utf-8")
    ids = re.findall(r"\b(?:CORE|UCI|NNUE|SMP|WEB|REG|DOC|REL)-\d{2}\b", text)
    assert ids
    assert len(ids) == len(set(ids)), "duplicate contract identifiers"
