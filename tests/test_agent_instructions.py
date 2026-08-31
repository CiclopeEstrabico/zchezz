"""Protect cross-agent working rules from silent drift."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _body(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[1:]).strip()


def test_agent_files_exist():
    assert (ROOT / "AGENTS.md").is_file()
    assert (ROOT / "CLAUDE.md").is_file()


def test_agent_rule_bodies_match():
    assert _body(ROOT / "AGENTS.md") == _body(ROOT / "CLAUDE.md"), (
        "AGENTS.md and CLAUDE.md diverged. Keep only the title line different."
    )


def test_agent_sections_are_not_duplicated():
    for name in ("AGENTS.md", "CLAUDE.md"):
        headings = [line for line in (ROOT / name).read_text(encoding="utf-8").splitlines() if line.startswith("## ")]
        repeated = sorted({heading for heading in headings if headings.count(heading) > 1})
        assert not repeated, f"{name} repeats sections: {repeated}"


def test_writing_skill_is_mirrored():
    a = ROOT / ".agents" / "skills" / "writing-rules" / "SKILL.md"
    c = ROOT / ".claude" / "skills" / "writing-rules" / "SKILL.md"
    assert a.is_file() and c.is_file()
    assert a.read_bytes() == c.read_bytes(), "writing-rules skill copies diverged"

