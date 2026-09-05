#!/usr/bin/env python3
"""Validate a Zchezz NNUE artifact without running a tournament."""
from __future__ import annotations

import argparse
import hashlib
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "utils"))
from repo_paths import active_version, nnue_weights  # noqa: E402

NNU3_DIMS = (799, 256, 256, 64, 64)
NNU3_SIZE = 426_864
NNU4_DIMS = (2560, 512, 1024, 32, 32)


def _header(data: bytes) -> tuple[bytes, int, tuple[int, ...], tuple[float, ...]]:
    if len(data) < 44:
        raise ValueError(f"file is too small: {len(data)} bytes")
    magic = data[:4]
    epoch = struct.unpack_from("<I", data, 4)[0]
    dims = struct.unpack_from("<5I", data, 8)
    scales = struct.unpack_from("<4f", data, 28)
    return magic, epoch, dims, scales


def _validate_nnu3(data: bytes, dims: tuple[int, ...]) -> None:
    if dims != NNU3_DIMS:
        raise ValueError(f"NNU3 dimension mismatch: {dims} != {NNU3_DIMS}")
    if len(data) != NNU3_SIZE:
        raise ValueError(f"NNU3 size mismatch: {len(data)} != {NNU3_SIZE}")


def _validate_nnu4(data: bytes, dims: tuple[int, ...]) -> None:
    if dims != NNU4_DIMS:
        raise ValueError(f"NNU4 dimension mismatch: {dims} != {NNU4_DIMS}")
    l1 = 2560 * 512 * 2
    l1b = 512 * 4
    l2 = 32 * 1024
    l2b = 32 * 4
    l3 = 32
    expected_min = 44 + l1 + l1b + l2 + l2b + l3 + 4
    if len(data) < expected_min:
        raise ValueError(f"truncated NNU4: {len(data)} < {expected_min}")


def inspect(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    magic, epoch, dims, scales = _header(data)
    if magic == b"NNU3":
        _validate_nnu3(data, dims)
    elif magic == b"NNU4":
        _validate_nnu4(data, dims)
    else:
        raise ValueError(f"unsupported NNUE magic {magic!r}; expected NNU3 or NNU4")
    return {
        "path": str(path),
        "format": magic.decode("ascii"),
        "bytes": len(data),
        "epoch": epoch,
        "dims": dims,
        "scales": scales,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="")
    parser.add_argument("--version", default="")
    args = parser.parse_args()
    path = Path(args.path) if args.path else nnue_weights(args.version or active_version())
    try:
        result = inspect(path)
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 1
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
