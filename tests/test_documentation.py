"""Verify objective documentation facts against the repository."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ACTIVE_DOCS = {
    "architecture.md", "build.md", "engine-contracts.md", "generated-artifacts.md",
    "nnue.md", "regression-testing.md", "release-process.md", "repository-layout.md",
    "syzygy.md", "testing.md", "wasm.md",
}


def _markdown_files():
    yield from [ROOT / "AGENTS.md", ROOT / "CLAUDE.md"]
    for name in ACTIVE_DOCS:
        yield DOCS / name


def test_documented_repo_paths_exist():
    # Only explicit backtick paths that look like repository files are checked.
    missing = set()
    pattern = re.compile(r"`((?:docs|tests|tools|utils|engine|train|openings)/[A-Za-z0-9_./*-]+)`")
    for document in _markdown_files():
        if not document.is_file():
            continue
        text = document.read_text(encoding="utf-8", errors="replace")
        for raw in pattern.findall(text):
            if "*" in raw or raw.endswith("/"):
                continue
            path = ROOT / raw
            # Version examples such as zchezz_vXXX are descriptive, not literal paths.
            if "vXXX" in raw:
                continue
            if not path.exists():
                missing.add((document.name, raw))
    assert not missing, "documented paths do not exist: " + repr(sorted(missing))


def test_required_testing_commands_are_documented():
    text = (DOCS / "testing.md").read_text(encoding="utf-8")
    for command in (
        "python tests/run_tests.py smoke",
        "python tests/run_tests.py full",
        "python tests/run_tests.py web",
        "python tests/run_tests.py release",
    ):
        assert command in text


def test_engine_contract_ids_are_unique():
    text = (DOCS / "engine-contracts.md").read_text(encoding="utf-8")
    ids = re.findall(r"\b(?:CORE|UCI|NNUE|SMP|WEB|REG|DOC|REL)-\d{2}\b", text)
    assert ids
    assert len(ids) == len(set(ids)), "duplicate contract identifiers"
