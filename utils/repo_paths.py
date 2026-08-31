"""Canonical repository and engine path handling for Zchezz."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

_VERSION_RE = re.compile(r"^zchezz_v(?P<number>\d+)$")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def engine_root() -> Path:
    return repo_root() / "engine" / "c"


def build_root() -> Path:
    return repo_root() / "engine" / "build"


def artifacts_root() -> Path:
    value = os.environ.get("ZCHEZZ_ARTIFACTS_DIR")
    return Path(value).expanduser().resolve() if value else repo_root() / "artifacts"


def tablebase_root() -> Path:
    value = os.environ.get("ZCHEZZ_SYZYGY_PATH")
    return Path(value).expanduser().resolve() if value else repo_root() / "tablebases"


def openings_root() -> Path:
    return repo_root() / "openings"


def normalize_version(version: str | int) -> str:
    if isinstance(version, int):
        return f"v{version}"
    text = str(version).strip().lower()
    if text.startswith("zchezz_"):
        text = text[len("zchezz_"):]
    number = text[1:] if text.startswith("v") else text
    if not number.isdigit():
        raise ValueError(f"invalid engine version: {version!r}")
    return f"v{int(number)}"


def version_number(version: str | int) -> int:
    return int(normalize_version(version)[1:])


def available_versions() -> list[str]:
    root = engine_root()
    if not root.is_dir():
        return []
    versions = [
        f"v{int(match.group('number'))}"
        for path in root.iterdir()
        if path.is_dir() and (match := _VERSION_RE.match(path.name))
    ]
    return sorted(versions, key=version_number)


def latest_version() -> str:
    versions = available_versions()
    if not versions:
        raise FileNotFoundError(f"no zchezz_v* directory under {engine_root()}")
    return versions[-1]


def previous_version(version: str | int | None = None) -> str:
    current = version_number(latest_version() if version is None else version)
    older = [item for item in available_versions() if version_number(item) < current]
    if not older:
        raise FileNotFoundError(f"no engine version older than v{current}")
    return older[-1]


def engine_dir(version: str | int | None = None) -> Path:
    resolved = latest_version() if version is None else normalize_version(version)
    return engine_root() / f"zchezz_{resolved}"


def engine_executable(version: str | int | None = None, *, require: bool = False) -> Path:
    path = engine_dir(version) / ("zchezz.exe" if os.name == "nt" else "zchezz")
    if not path.exists() and os.name != "nt":
        fallback = engine_dir(version) / "zchezz.exe"
        if fallback.exists():
            path = fallback
    if require and not path.is_file():
        raise FileNotFoundError(f"engine executable not found: {path}")
    return path


def nnue_weights(version: str | int | None = None, *, require: bool = False) -> Path:
    path = engine_dir(version) / "nnue_weights.bin"
    if require and not path.is_file():
        raise FileNotFoundError(f"NNUE weights not found: {path}")
    return path


def wasm_source(version: str | int | None = None) -> Path:
    return engine_dir(version) / "zchezz_wasm.html"


def wasm_bundle(version: str | int | None = None) -> Path:
    return engine_dir(version) / "zchezz_bundle.html"


def first_existing(paths: Iterable[Path]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def default_opening_book() -> Path | None:
    return first_existing((
        openings_root() / "book.bin",
        repo_root() / "utils" / "book.bin",
        repo_root() / "utils" / "OpeningBook.bin",
    ))


def artifact_dir(category: str, run_id: str | None = None) -> Path:
    path = artifacts_root() / category
    if run_id:
        path /= run_id
    path.mkdir(parents=True, exist_ok=True)
    return path
