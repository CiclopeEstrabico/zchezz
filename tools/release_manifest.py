#!/usr/bin/env python3
"""Create a machine-readable Zchezz release provenance manifest."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "utils"))

from repo_paths import artifact_dir, engine_dir, engine_executable, latest_version, nnue_weights  # noqa: E402

def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def command_text(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            command,
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
        ).strip()
    except Exception:
        return None

def latest_test_summaries() -> dict[str, dict[str, object]]:
    root = ROOT / "artifacts" / "tests"
    found: dict[str, tuple[float, Path, dict[str, object]]] = {}
    if not root.is_dir():
        return {}
    for path in root.glob("*/summary.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            profile = str(payload.get("profile", ""))
            if not profile:
                continue
            stamp = path.stat().st_mtime
            if profile not in found or stamp > found[profile][0]:
                found[profile] = (stamp, path, payload)
        except Exception:
            continue

    result: dict[str, dict[str, object]] = {}
    for profile, (_, path, payload) in found.items():
        result[profile] = {
            "path": str(path.relative_to(ROOT)),
            "version": payload.get("version"),
            "baseline": payload.get("baseline"),
            "passed": payload.get("passed"),
            "coverage_complete": payload.get("coverage_complete"),
            "counts": payload.get("counts"),
        }
    return result

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="")
    parser.add_argument("--regression-run", default="")
    args = parser.parse_args()

    version = args.version or latest_version()
    now = dt.datetime.now(dt.timezone.utc)
    edir = engine_dir(version)
    status = command_text(["git", "status", "--porcelain"])

    payload = {
        "schema": 2,
        "created_utc": now.isoformat(),
        "version": version,
        "git_sha": command_text(["git", "rev-parse", "HEAD"]),
        "git_branch": command_text(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(status),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "compiler": command_text(["gcc", "--version"]),
        "engine_path": str(engine_executable(version)),
        "engine_sha256": sha256(engine_executable(version)),
        "nnue_path": str(nnue_weights(version)),
        "nnue_sha256": sha256(nnue_weights(version)),
        "fathom": {
            "tbprobe_c": (edir / "tbprobe.c").is_file(),
            "tbprobe_h": (edir / "tbprobe.h").is_file(),
        },
        "latest_test_summaries": latest_test_summaries(),
        "regression_run": args.regression_run or None,
    }

    out = artifact_dir("releases") / f"{version}-{now.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(out.relative_to(ROOT))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
