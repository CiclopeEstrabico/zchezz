#!/usr/bin/env python3
"""Bundle a per-version Zchezz WASM build using the shared HTML template."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLER = HERE / "bundle.py"
TEMP_NAME = ".zchezz_wasm_template.html"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True)
    parser.add_argument("--engine-dir", required=True)
    args = parser.parse_args()

    template = Path(args.template)
    if not template.is_absolute():
        template = (HERE / template).resolve()

    engine_dir = Path(args.engine_dir)
    if not engine_dir.is_absolute():
        engine_dir = (HERE / engine_dir).resolve()

    required = [
        template,
        BUNDLER,
        engine_dir / "zchezz_wasm.js",
        engine_dir / "zchezz_wasm.wasm",
        engine_dir / "nnue_weights.bin",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        print("ERROR: bundle prerequisites missing:")
        for item in missing:
            print(f"  {item}")
        return 1

    temp = engine_dir / TEMP_NAME
    output = engine_dir / "zchezz_bundle.html"

    try:
        shutil.copyfile(template, temp)
        command = [
            sys.executable,
            str(BUNDLER),
            str(temp),
            str(engine_dir / "zchezz_wasm.js"),
            str(engine_dir / "zchezz_wasm.wasm"),
            str(engine_dir / "nnue_weights.bin"),
        ]
        proc = subprocess.run(command, cwd=HERE)
        if proc.returncode != 0:
            return proc.returncode
        if not output.is_file():
            print(f"ERROR: bundler returned success but did not create {output}")
            return 1
        print(f"PASS: generated {output}")
        return 0
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
