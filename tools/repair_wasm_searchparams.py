#!/usr/bin/env python3
"""Repair the v403 WebAssembly SearchParams ABI in browser-side code.

The current C SearchParams layout is 44 bytes in wasm32:
  0  max_depth
  4  start_depth
  8  time_limit_ms
 12  node_limit
 16  multi_pv
 20  threads
 24  stop pointer
 28  search_state pointer
 32  info_cb pointer
 36  tt pointer
 40  mpv_share_budget

Historical browser code allocates only 32 bytes. search_best_sret sanitizes the
new pointer fields, but copying a 44-byte C struct from a 32-byte allocation is
still an out-of-allocation read. This migration makes the JS allocation match
the actual ABI and passes startDepth through the bundled worker.

The tool is idempotent and refuses to silently succeed if the expected browser
patterns are absent.
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    ROOT / "engine" / "build" / "bundle.py",
    ROOT / "engine" / "build" / "zchezz_wasm.html",
)

SP_SIZE = 44


def replace_once_or_existing(text: str, old: str, new: str, label: str) -> tuple[str, bool]:
    if new in text:
        return text, False
    if old not in text:
        raise RuntimeError(f"cannot find expected {label} pattern")
    return text.replace(old, new, 1), True


def repair_text(path: Path, text: str) -> tuple[str, list[str]]:
    changes: list[str] = []

    old = "const pp=Mod._malloc(32);"
    new = f"const SP_SZ={SP_SIZE};\\n  const pp=Mod._malloc(SP_SZ);"
    text, changed = replace_once_or_existing(text, old, new, "SearchParams allocation")
    if changed:
        changes.append("SearchParams allocation 32 -> 44")

    old = "Mod.HEAPU8.fill(0,pp,pp+32);"
    new = "Mod.HEAPU8.fill(0,pp,pp+SP_SZ);"
    text, changed = replace_once_or_existing(text, old, new, "SearchParams zero-fill")
    if changed:
        changes.append("SearchParams zero-fill uses SP_SZ")

    old = "new DataView(Mod.HEAPU8.buffer,pp,32);"
    new = "new DataView(Mod.HEAPU8.buffer,pp,SP_SZ);"
    text, changed = replace_once_or_existing(text, old, new, "SearchParams DataView")
    if changed:
        changes.append("SearchParams DataView uses SP_SZ")

    # Ensure the new fields are explicitly initialized rather than relying only
    # on the zero fill. This also documents the ABI in generated JS.
    marker = "pdv.setInt32(28,0,true);"
    extra = (
        "pdv.setInt32(28,0,true);\\n"
        "  pdv.setInt32(32,0,true);  // info_cb = NULL\\n"
        "  pdv.setInt32(36,0,true);  // tt = NULL (search uses g_tt)\\n"
        "  pdv.setInt32(40,0,true);  // mpv_share_budget = 0"
    )
    if "pdv.setInt32(40,0,true);" not in text:
        if marker not in text:
            raise RuntimeError("cannot find SearchParams offset-28 initialization")
        text = text.replace(marker, extra, 1)
        changes.append("explicitly initialize offsets 32/36/40")

    # bundle.py historically discarded startDepth even though the shared source
    # template already knew about it.
    if path.name == "bundle.py":
        old_call = (
            "doSearch(msg.fen,msg.moves||'',msg.depth||9,msg.id,"
            "msg.timeLimitMs||0,msg.multiPv||1)"
        )
        new_call = (
            "doSearch(msg.fen,msg.moves||'',msg.depth||9,msg.id,"
            "msg.timeLimitMs||0,msg.multiPv||1,msg.startDepth||0)"
        )
        if new_call not in text:
            if old_call not in text:
                raise RuntimeError("cannot find bundled-worker doSearch call")
            text = text.replace(old_call, new_call, 1)
            changes.append("pass startDepth into bundled worker")

        old_sig = "function doSearch(fen,moves,depth,id,timeLimitMs,multiPv)"
        new_sig = "function doSearch(fen,moves,depth,id,timeLimitMs,multiPv,startDepth)"
        if new_sig not in text:
            if old_sig not in text:
                raise RuntimeError("cannot find bundled-worker doSearch signature")
            text = text.replace(old_sig, new_sig, 1)
            changes.append("accept startDepth in bundled worker")

        old_depth = "pdv.setInt32(4,0,true);"
        new_depth = "pdv.setInt32(4,startDepth||0,true);"
        if new_depth not in text:
            if old_depth not in text:
                raise RuntimeError("cannot find bundled-worker start_depth initialization")
            text = text.replace(old_depth, new_depth, 1)
            changes.append("write startDepth at SearchParams offset 4")

    # Update stale explanatory text where present.
    text = text.replace(
        "SearchParams v305: max_depth(4)+start_depth(4)+time_limit_ms(4)+node_limit(4)+multi_pv(4)+threads(4)+stop(4)+info_cb(4)=32 bytes",
        "SearchParams v403 wasm32: 44 bytes; see docs/wasm.md for the field layout",
    )
    return text, changes


def validate(path: Path, text: str) -> list[str]:
    errors: list[str] = []
    if "const SP_SZ=44;" not in text:
        errors.append("SP_SZ=44 missing")
    if "Mod._malloc(SP_SZ)" not in text:
        errors.append("SearchParams allocation does not use SP_SZ")
    if "pp+SP_SZ" not in text:
        errors.append("SearchParams zero-fill does not use SP_SZ")
    if "pp,SP_SZ" not in text:
        errors.append("SearchParams DataView does not use SP_SZ")
    for offset in (32, 36, 40):
        if f"pdv.setInt32({offset},0,true);" not in text:
            errors.append(f"offset {offset} is not explicitly initialized")
    if path.name == "bundle.py":
        if "msg.startDepth||0" not in text:
            errors.append("bundle worker does not pass startDepth")
        if "pdv.setInt32(4,startDepth||0,true);" not in text:
            errors.append("bundle worker does not write startDepth")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    failed = False
    for path in FILES:
        if not path.is_file():
            print(f"ERROR: missing {path.relative_to(ROOT)}")
            failed = True
            continue

        text = path.read_text(encoding="utf-8")
        if args.check:
            errors = validate(path, text)
            if errors:
                failed = True
                print(f"FAIL: {path.relative_to(ROOT)}")
                for error in errors:
                    print(f"  {error}")
            else:
                print(f"PASS: {path.relative_to(ROOT)} SearchParams ABI")
            continue

        try:
            repaired, changes = repair_text(path, text)
        except RuntimeError as exc:
            failed = True
            print(f"ERROR: {path.relative_to(ROOT)}: {exc}")
            continue

        errors = validate(path, repaired)
        if errors:
            failed = True
            print(f"ERROR: repaired {path.relative_to(ROOT)} still invalid:")
            for error in errors:
                print(f"  {error}")
            continue

        if repaired != text:
            path.write_text(repaired, encoding="utf-8", newline="\n")
            print(f"UPDATED: {path.relative_to(ROOT)}")
            for change in changes:
                print(f"  - {change}")
        else:
            print(f"PASS: {path.relative_to(ROOT)} already repaired")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
