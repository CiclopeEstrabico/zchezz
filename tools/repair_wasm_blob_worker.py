#!/usr/bin/env python3
"""Repair Zchezz browser workers so Emscripten can resolve WASM from a Blob URL."""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "engine" / "build" / "zchezz_wasm.html"
BUNDLER = ROOT / "engine" / "build" / "bundle.py"

HTML_OLD = (
    "async function init(wb,wts){\n"
    "  Mod=await ZchezzEngine({noInitialRun:true,wasmBinary:wb,"
    "print:function(){},printErr:function(){}});"
)

HTML_NEW = (
    "async function init(wb,wts){\n"
    "  const wasmBlobUrl=URL.createObjectURL(new Blob([wb],"
    "{type:'application/wasm'}));\n"
    "  try {\n"
    "    Mod=await ZchezzEngine({\n"
    "      noInitialRun:true,\n"
    "      wasmBinary:wb,\n"
    "      locateFile:function(path,prefix){\n"
    "        if(path.endsWith('.wasm')) return wasmBlobUrl;\n"
    "        if(prefix && !prefix.startsWith('blob:')) return prefix+path;\n"
    "        return path;\n"
    "      },\n"
    "      print:function(){},\n"
    "      printErr:function(){}\n"
    "    });\n"
    "  } finally {\n"
    "    URL.revokeObjectURL(wasmBlobUrl);\n"
    "  }"
)

BUNDLE_OLD = (
    "async function init(wb,wts){{\n"
    "  Mod=await ZchezzEngine({{noInitialRun:true,wasmBinary:wb,"
    "print:function(){{}},printErr:function(){{}}}});"
)

BUNDLE_NEW = (
    "async function init(wb,wts){{\n"
    "  const wasmBlobUrl=URL.createObjectURL(new Blob([wb],"
    "{{type:'application/wasm'}}));\n"
    "  try {{\n"
    "    Mod=await ZchezzEngine({{\n"
    "      noInitialRun:true,\n"
    "      wasmBinary:wb,\n"
    "      locateFile:function(path,prefix){{\n"
    "        if(path.endsWith('.wasm')) return wasmBlobUrl;\n"
    "        if(prefix && !prefix.startsWith('blob:')) return prefix+path;\n"
    "        return path;\n"
    "      }},\n"
    "      print:function(){{}},\n"
    "      printErr:function(){{}}\n"
    "    }});\n"
    "  }} finally {{\n"
    "    URL.revokeObjectURL(wasmBlobUrl);\n"
    "  }}"
)

def has_contract(text: str) -> bool:
    return (
        "URL.createObjectURL(new Blob([wb]" in text
        and "locateFile:function(path,prefix)" in text
        and "if(path.endsWith('.wasm')) return wasmBlobUrl;" in text
        and "URL.revokeObjectURL(wasmBlobUrl);" in text
    )

def patch_file(path: Path, old: str, new: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing file: {path.relative_to(ROOT)}")

    text = path.read_text(encoding="utf-8")
    if has_contract(text):
        print(f"PASS: {path.relative_to(ROOT)} already repaired")
        return

    if old not in text:
        raise RuntimeError(
            f"{path.relative_to(ROOT)} does not contain the known legacy "
            "worker initialization pattern; refusing a blind edit"
        )

    updated = text.replace(old, new, 1)
    if not has_contract(updated):
        raise RuntimeError(
            f"repaired {path.relative_to(ROOT)} does not satisfy the contract"
        )
    path.write_text(updated, encoding="utf-8", newline="\n")
    print(f"UPDATED: {path.relative_to(ROOT)}")

def check_file(path: Path) -> bool:
    if not path.is_file():
        print(f"FAIL: missing {path.relative_to(ROOT)}")
        return False
    ok = has_contract(path.read_text(encoding="utf-8"))
    print(f"{'PASS' if ok else 'FAIL'}: {path.relative_to(ROOT)} Blob-WASM contract")
    return ok

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check:
        a = check_file(TEMPLATE)
        b = check_file(BUNDLER)
        return 0 if a and b else 1

    try:
        patch_file(TEMPLATE, HTML_OLD, HTML_NEW)
        patch_file(BUNDLER, BUNDLE_OLD, BUNDLE_NEW)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    print("PASS: Blob-worker WASM URL handling repaired")
    print("Rebuild the web bundle before rerunning browser E2E.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
