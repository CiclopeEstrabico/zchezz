"""Unit tests for fast repository-quality tools."""
from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EPD = _load("zchezz_check_epd", ROOT / "tools" / "check_epd.py")
NNUE = _load("zchezz_check_nnue", ROOT / "tools" / "check_nnue.py")


def test_epd_accepts_valid_fen_prefix():
    ok, reason = EPD.valid_fen_prefix("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - bm e4;")
    assert ok, reason


def test_epd_rejects_bad_rank_width():
    ok, reason = EPD.valid_fen_prefix("7/8/8/8/8/8/8/8 w - -")
    assert not ok
    assert "rank expands" in reason


def test_epd_scan_detects_duplicate(tmp_path):
    path = tmp_path / "suite.epd"
    line = "8/8/8/8/8/8/4K3/4k3 w - -"
    path.write_text(line + "\n" + line + "\n", encoding="utf-8")
    count, failures = EPD.scan(path)
    assert count == 2
    assert any("duplicate position" in item for item in failures)


def _write_minimal_nnu4(path: Path):
    dims = (2560, 512, 1024, 32, 32)
    header = b"NNU4" + struct.pack("<I", 1) + struct.pack("<5I", *dims) + struct.pack("<4f", 255.0, 64.0, 8.0, 1.0)
    payload_size = 2560 * 512 * 2 + 512 * 4 + 32 * 1024 + 32 * 4 + 32 + 4
    path.write_bytes(header + bytes(payload_size))


def test_nnue_inspector_accepts_structurally_valid_nnu4(tmp_path):
    path = tmp_path / "weights.bin"
    _write_minimal_nnu4(path)
    info = NNUE.inspect(path)
    assert info["dims"] == (2560, 512, 1024, 32, 32)
    assert info["bytes"] == path.stat().st_size


def test_nnue_inspector_rejects_bad_magic(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"BAD!" + bytes(100))
    try:
        NNUE.inspect(path)
    except ValueError as exc:
        assert "bad magic" in str(exc)
    else:
        raise AssertionError("bad magic was accepted")

