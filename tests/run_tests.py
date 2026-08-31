#!/usr/bin/env python3
"""Run canonical Zchezz deterministic test profiles."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# ═══════════════ CONFIGURATION ═══════════════
DEFAULT_PROFILE = "smoke"  # default test profile
DEFAULT_VERSION = ""       # empty selects the latest numeric version
DEFAULT_BASELINE = ""      # empty selects the preceding numeric version
# ═════════════════════════════════════════════

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "utils"))
from repo_paths import artifact_dir, engine_dir, engine_executable, latest_version, previous_version  # noqa: E402


@dataclass(frozen=True)
class Step:
    name: str
    command: tuple[str, ...]
    required: bool = True
    when_file: str | None = None


@dataclass
class Result:
    name: str
    command: list[str]
    status: str
    returncode: int | None
    seconds: float
    required: bool
    log: str


def _python(*args: str) -> tuple[str, ...]:
    return (sys.executable, *args)


def make_program() -> str:
    return "mingw32-make" if os.name == "nt" else "make"


def profiles(version: str, baseline: str) -> dict[str, list[Step]]:
    candidate_exe = str(engine_executable(version))
    baseline_exe = str(engine_executable(baseline))
    sanitizer_exe = str(engine_dir(version) / "zchezz_sanitize.exe")
    make = make_program()
    smoke = [
        Step("repository contracts", _python("tools/check_repo.py")),
        Step("agent instructions", _python("-m", "pytest", "tests/test_agent_instructions.py", "-q")),
        Step("repository contracts pytest", _python("-m", "pytest", "tests/test_repository_contracts.py", "-q")),
        Step("native build", (make, "-C", "engine/build", f"ENGINE={version}", "native")),
        Step("NNUE artifact", _python("tools/check_nnue.py", "--version", version)),
        Step("C invariants", (make, "-C", "engine/build", f"ENGINE={version}", "test-c"), when_file="engine/c/tests/test_engine_invariants.c"),
        Step("perft smoke", _python("tests/test_perft.py", "--version", version)),
        Step("UCI smoke", _python("tests/test_uci_extended.py", "--version", version, "--only", "T1", "--only", "T2")),
        Step("golden engine contracts", _python("-m", "pytest", "tests/test_engine_golden.py", "-q")),
    ]
    full = smoke + [
        Step("perft full", _python("tests/test_perft.py", "--version", version)),
        Step("UCI extended", _python("tests/test_uci_extended.py", "--version", version)),
        Step("documentation contracts", _python("-m", "pytest", "tests/test_documentation.py", "-q")),
        Step("book tests", _python("tests/test_book.py"), required=False, when_file="tests/test_book.py"),
        Step("browser static", _python("tests/test_browser_html.py"), required=False, when_file="tests/test_browser_html.py"),
    ]
    web = [
        Step("bundle", (make, "-C", "engine/build", f"ENGINE={version}", "bundle")),
        Step("browser static", _python("tests/test_browser_html.py"), when_file="tests/test_browser_html.py"),
        Step("browser e2e", _python("tests/test_browser.py", "--version", version, "--headless"), when_file="tests/test_browser.py"),
    ]
    regression = [Step("quick H2H", _python("tests/run_tournament_quick.py", "--engine-a", candidate_exe, "--label-a", version, "--engine-b", baseline_exe, "--label-b", baseline, "--threads", "1"), when_file="tests/run_tournament_quick.py")]
    release = full + web + [
        Step("ASan+UBSan build", (make, "-C", "engine/build", f"ENGINE={version}", "sanitize")),
        Step("ASan+UBSan UCI smoke", _python("tests/test_uci_extended.py", "--exe", sanitizer_exe, "--only", "T1", "--only", "T2")),
        Step("release manifest", _python("tools/release_manifest.py", "--version", version)),
    ]
    return {"smoke": smoke, "full": full, "web": web, "regression": regression, "release": release}


def run_step(step: Step, log_dir: Path, env: dict[str, str]) -> Result:
    if step.when_file and not (ROOT / step.when_file).exists():
        return Result(step.name, list(step.command), "SKIP", None, 0.0, step.required, "")
    start = time.monotonic()
    proc = subprocess.run(step.command, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    seconds = time.monotonic() - start
    log = proc.stdout + ("\n" if proc.stdout and proc.stderr else "") + proc.stderr
    log_path = log_dir / (step.name.lower().replace(" ", "_").replace("+", "plus") + ".log")
    log_path.write_text(log, encoding="utf-8")
    status = "PASS" if proc.returncode == 0 else "FAIL"
    if proc.returncode != 0 and not step.required:
        status = "WARN"
    return Result(step.name, list(step.command), status, proc.returncode, seconds, step.required, str(log_path.relative_to(ROOT)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", nargs="?", default=DEFAULT_PROFILE, choices=["smoke", "full", "web", "regression", "release"])
    parser.add_argument("--version", default=DEFAULT_VERSION, help="candidate engine version")
    parser.add_argument("--baseline", default=DEFAULT_BASELINE, help="regression baseline")
    parser.add_argument("--list", action="store_true", help="print the selected profile")
    parser.add_argument("--keep-going", action="store_true", help="continue after a required failure")
    args = parser.parse_args()
    version = args.version or latest_version()
    baseline = args.baseline or previous_version(version)
    selected = profiles(version, baseline)[args.profile]
    if args.list:
        print(f"candidate: {version}")
        print(f"baseline:  {baseline}")
        for step in selected:
            print(f"{'REQ' if step.required else 'OPT'}  {step.name:28}  {shlex.join(step.command)}")
        return 0
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = artifact_dir("tests", f"{args.profile}-{stamp}")
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({"ZCHEZZ_TEST_PROFILE": args.profile, "ZCHEZZ_ENGINE_VERSION": version, "ZCHEZZ_BASELINE_VERSION": baseline})
    results: list[Result] = []
    print(f"Zchezz test profile: {args.profile}  candidate: {version}  baseline: {baseline}")
    for index, step in enumerate(selected, 1):
        print(f"[{index:02d}/{len(selected):02d}] {step.name} ...", flush=True)
        result = run_step(step, logs, env)
        results.append(result)
        print(f"  {result.status} ({result.seconds:.1f}s)")
        if result.status == "FAIL" and result.required and not args.keep_going:
            break
    failed = [result for result in results if result.status == "FAIL" and result.required]
    payload = {"profile": args.profile, "version": version, "baseline": baseline, "engine": str(engine_executable(version)), "results": [asdict(result) for result in results], "passed": not failed}
    (out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [f"profile: {args.profile}", f"version: {version}", f"baseline: {baseline}", f"passed: {not failed}", ""]
    lines.extend(f"{result.status:5} {result.seconds:8.1f}s  {result.name}" for result in results)
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report: {out.relative_to(ROOT)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
