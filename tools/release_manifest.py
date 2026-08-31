#!/usr/bin/env python3
"""Create a machine-readable release provenance manifest."""
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
from repo_paths import artifact_dir, engine_executable, latest_version, nnue_weights  # noqa: E402


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def command_text(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="")
    parser.add_argument("--regression-run", default="")
    args = parser.parse_args()
    version = args.version or latest_version()
    now = dt.datetime.now(dt.timezone.utc)
    payload = {
        "schema": 1,
        "created_utc": now.isoformat(),
        "version": version,
        "git_sha": command_text(["git", "rev-parse", "HEAD"]),
        "git_branch": command_text(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "compiler": command_text(["gcc", "--version"]),
        "engine_path": str(engine_executable(version)),
        "engine_sha256": sha256(engine_executable(version)),
        "nnue_path": str(nnue_weights(version)),
        "nnue_sha256": sha256(nnue_weights(version)),
        "regression_run": args.regression_run or None,
    }
    out = artifact_dir("releases") / f"{version}-{now.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(out.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

