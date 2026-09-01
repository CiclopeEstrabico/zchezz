#!/usr/bin/env python3
"""Static validation of a generated Zchezz browser bundle."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "utils"))
from repo_paths import latest_version, wasm_bundle  # noqa: E402

HTML_PATH = ""

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", default=HTML_PATH)
    args = parser.parse_args()

    path = Path(args.html) if args.html else wasm_bundle(latest_version())
    if not path.is_file():
        print(f"SKIP: browser bundle not found: {path}")
        return 0

    html = path.read_text(encoding="utf-8", errors="replace")
    checks = {
        "non-trivial HTML": len(html) > 10_000,
        "engine bootstrap": "ZchezzEngine" in html or "startEngine" in html,
        "embedded WASM marker": "_WASM_B64" in html or "base64" in html.lower(),
        "embedded NNUE marker": "_WEIGHTS_B64" in html or "nnue" in html.lower(),
        "no unresolved obvious placeholder": not re.search(r"__(?:WASM|WEIGHTS|ENGINE_JS)__", html),
    }

    failures = [name for name, ok in checks.items() if not ok]
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}: {name}")

    if failures:
        return 1
    print("PASS: static browser bundle contract")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
