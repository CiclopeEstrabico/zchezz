#!/usr/bin/env python3
"""Run canonical Zchezz test profiles and write machine-readable evidence."""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# ═══════════════ CONFIGURATION ═══════════════
DEFAULT_PROFILE = "smoke"
DEFAULT_VERSION = ""
DEFAULT_BASELINE = ""
SMOKE_PERFT_MAX_DEPTH = 3
# ═════════════════════════════════════════════

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "utils"))

from repo_paths import (  # noqa: E402
    active_version,
    artifact_dir,
    engine_dir,
    engine_executable,
    previous_version,
    wasm_bundle,
    wasm_source,
)

@dataclass(frozen=True)
class Step:
    name: str
    command: tuple[str, ...]
    required: bool = True
    when_file: str | None = None
    when_command: str | None = None
    when_module: str | None = None
    os_names: tuple[str, ...] | None = None
    skip_reason: str = ""

@dataclass
class Result:
    name: str
    command: list[str]
    status: str
    returncode: int | None
    seconds: float
    required: bool
    log: str
    note: str = ""

def _python(*args: str) -> tuple[str, ...]:
    return (sys.executable, *args)

def make_program() -> str:
    return "mingw32-make" if os.name == "nt" else "make"

def _path_arg(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)

def profiles(version: str, baseline: str) -> dict[str, list[Step]]:
    candidate_exe = str(engine_executable(version))
    baseline_exe = str(engine_executable(baseline))
    sanitizer_exe = str(engine_dir(version) / "zchezz_sanitize.exe")
    template = _path_arg(wasm_source(version))
    bundle = _path_arg(wasm_bundle(version))
    make = make_program()

    smoke = [
        Step("repository contracts", _python("tools/check_repo.py")),
        Step("agent instructions", _python("-m", "pytest", "tests/test_agent_instructions.py", "-q")),
        Step("repository contracts pytest", _python("-m", "pytest", "tests/test_repository_contracts.py", "-q")),
        Step("native build", (make, "-C", "engine/build", f"ENGINE={version}", "native")),
        Step("NNUE artifact", _python("tools/check_nnue.py", "--version", version)),
        Step("C invariants", (make, "-C", "engine/build", f"ENGINE={version}", "test-c")),
        Step(
            "perft smoke",
            _python("tests/test_perft.py", "--version", version, "--max-depth", str(SMOKE_PERFT_MAX_DEPTH)),
        ),
        Step(
            "UCI smoke",
            _python("tests/test_uci_extended.py", "--version", version, "--only", "T1", "--only", "T2"),
        ),
        Step("golden engine contracts", _python("-m", "pytest", "tests/test_engine_golden.py", "-q")),
    ]

    full = smoke + [
        Step("perft full", _python("tests/test_perft.py", "--version", version)),
        Step("UCI extended", _python("tests/test_uci_extended.py", "--version", version)),
        Step("documentation contracts", _python("-m", "pytest", "tests/test_documentation.py", "-q")),
        Step(
            "infrastructure unit tests",
            _python(
                "-m", "pytest",
                "tests/test_repo_paths.py",
                "tests/test_quality_tools.py",
                "tests/test_test_runner.py",
                "tests/test_repo_policy.py",
                "tests/test_wasm_wiring.py",
                "-q",
            ),
        ),
    ]

    web = [
        Step(
            "bundle",
            (make, "-C", "engine/build", f"ENGINE={version}", "bundle"),
            when_file=template,
            when_command="emcc",
            skip_reason="requires Emscripten and engine/build/zchezz_wasm.html",
        ),
        Step(
            "browser static",
            _python("tests/test_browser_html.py", "--html", bundle),
            when_file=bundle,
            skip_reason="requires a successfully generated bundle",
        ),
        Step(
            "book contracts",
            _python("tests/test_book.py", "--html", bundle, "--evaluate", "false", "--write-cleaned", "false"),
            when_file=bundle,
            skip_reason="requires a successfully generated bundle",
        ),
        Step(
            "browser e2e",
            _python(
                "tests/test_browser.py",
                "--version", version,
                "--html", "zchezz_bundle.html",
                "--headless",
            ),
            when_file=bundle,
            when_module="playwright",
            skip_reason="requires the generated bundle and Playwright",
        ),
    ]

    regression = [
        Step(
            "quick H2H",
            _python(
                "tests/run_tournament_quick.py",
                "--engine-a", candidate_exe,
                "--label-a", version,
                "--engine-b", baseline_exe,
                "--label-b", baseline,
                "--threads", "1",
            ),
            when_file="tests/run_tournament_quick.py",
        )
    ]

    release = full + web + [
        Step(
            "ASan+UBSan build",
            (make, "-C", "engine/build", f"ENGINE={version}", "sanitize"),
            os_names=("posix",),
            skip_reason="sanitizer release gate is executed on supported POSIX toolchains",
        ),
        Step(
            "ASan+UBSan UCI smoke",
            _python(
                "tests/test_uci_extended.py",
                "--exe", sanitizer_exe,
                "--only", "T1",
                "--only", "T2",
            ),
            when_file=_path_arg(Path(sanitizer_exe)),
            os_names=("posix",),
            skip_reason="requires a successfully built POSIX sanitizer executable",
        ),
        Step("release manifest", _python("tools/release_manifest.py", "--version", version)),
    ]

    return {
        "smoke": smoke,
        "full": full,
        "web": web,
        "regression": regression,
        "release": release,
    }

def prerequisite_skip(step: Step) -> str | None:
    reasons: list[str] = []
    if step.os_names is not None and os.name not in step.os_names:
        reasons.append(f"os.name={os.name!r} not in {step.os_names}")
    if step.when_file and not (ROOT / step.when_file).is_file():
        reasons.append(f"missing file: {step.when_file}")
    if step.when_command and shutil.which(step.when_command) is None:
        reasons.append(f"missing command: {step.when_command}")
    if step.when_module and importlib.util.find_spec(step.when_module) is None:
        reasons.append(f"missing Python module: {step.when_module}")
    if not reasons:
        return None
    suffix = f"; {step.skip_reason}" if step.skip_reason else ""
    return "; ".join(reasons) + suffix

def run_step(step: Step, log_dir: Path, env: dict[str, str]) -> Result:
    why = prerequisite_skip(step)
    if why:
        return Result(step.name, list(step.command), "SKIP", None, 0.0, step.required, "", why)

    start = time.monotonic()
    try:
        proc = subprocess.run(
            step.command,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        seconds = time.monotonic() - start
        log = proc.stdout + ("\n" if proc.stdout and proc.stderr else "") + proc.stderr
        status = "PASS" if proc.returncode == 0 else ("FAIL" if step.required else "WARN")
        returncode = proc.returncode
    except FileNotFoundError as exc:
        seconds = time.monotonic() - start
        log = f"{exc}\n"
        status = "FAIL" if step.required else "WARN"
        returncode = 127

    log_path = log_dir / (
        step.name.lower().replace(" ", "_").replace("+", "plus").replace("/", "_") + ".log"
    )
    log_path.write_text(log, encoding="utf-8")
    return Result(
        step.name,
        list(step.command),
        status,
        returncode,
        seconds,
        step.required,
        str(log_path.relative_to(ROOT)),
    )

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "profile",
        nargs="?",
        default=DEFAULT_PROFILE,
        choices=["smoke", "full", "web", "regression", "release"],
    )
    parser.add_argument("--version", default=DEFAULT_VERSION, help="candidate engine version")
    parser.add_argument("--baseline", default=DEFAULT_BASELINE, help="comparison engine version")
    parser.add_argument("--list", action="store_true", help="print the selected profile")
    parser.add_argument("--keep-going", action="store_true", help="continue after a required failure")
    args = parser.parse_args()

    version = args.version or active_version()
    baseline = args.baseline or previous_version(version)
    selected = profiles(version, baseline)[args.profile]

    if args.list:
        print(f"candidate: {version}")
        print(f"baseline:  {baseline}")
        for step in selected:
            prereq = prerequisite_skip(step)
            state = "SKIP" if prereq else ("REQ" if step.required else "OPT")
            print(f"{state:4}  {step.name:28}  {shlex.join(step.command)}")
            if prereq:
                print(f"      reason: {prereq}")
        return 0

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = artifact_dir("tests", f"{args.profile}-{stamp}")
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({
        "ZCHEZZ_TEST_PROFILE": args.profile,
        "ZCHEZZ_ENGINE_VERSION": version,
        "ZCHEZZ_BASELINE_VERSION": baseline,
    })

    results: list[Result] = []
    print(f"Zchezz test profile: {args.profile}  candidate: {version}  baseline: {baseline}")

    for index, step in enumerate(selected, 1):
        print(f"[{index:02d}/{len(selected):02d}] {step.name} ...", flush=True)
        result = run_step(step, logs, env)
        results.append(result)
        suffix = f" — {result.note}" if result.note else ""
        print(f"  {result.status} ({result.seconds:.1f}s){suffix}")
        if result.status == "FAIL" and result.required and not args.keep_going:
            break

    failed = [r for r in results if r.status == "FAIL" and r.required]
    skipped = [r for r in results if r.status == "SKIP"]
    warned = [r for r in results if r.status == "WARN"]

    payload = {
        "schema": 2,
        "profile": args.profile,
        "version": version,
        "baseline": baseline,
        "engine": str(engine_executable(version)),
        "results": [asdict(r) for r in results],
        "counts": {
            "pass": sum(r.status == "PASS" for r in results),
            "fail": sum(r.status == "FAIL" for r in results),
            "warn": len(warned),
            "skip": len(skipped),
        },
        "passed": not failed,
        "coverage_complete": not skipped,
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"profile: {args.profile}",
        f"version: {version}",
        f"baseline: {baseline}",
        f"passed: {not failed}",
        f"coverage_complete: {not skipped}",
        "",
    ]
    for result in results:
        note = f" — {result.note}" if result.note else ""
        lines.append(f"{result.status:5} {result.seconds:8.1f}s  {result.name}{note}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Report: {out.relative_to(ROOT)}")
    if skipped:
        print(f"Coverage note: {len(skipped)} step(s) skipped; SKIP is not validation evidence.")
    return 1 if failed else 0

if __name__ == "__main__":
    raise SystemExit(main())