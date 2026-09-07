#!/usr/bin/env python3
"""Materialize a temporary engine copy with correct UCI `go nodes` semantics.

This is deliberately benchmark-side only. It lets us compare historical and
current engines at a true whole-search node budget without changing an accepted
engine baseline.

Two independent bugs are repaired in the temporary copy:
  1) nodes-only UCI searches must not inherit DEFAULT_DEPTH=8;
  2) the node budget must use nodes_total (whole search), not the per-ID-depth
     nodes counter, and hitting it must mark the search interrupted so iterative
     deepening cannot publish a post-budget pseudo-iteration.

The patch supports both the legacy v3.x search shape and the v5 search shape
(with stop_guard). It intentionally validates the source shape before writing.
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def patch(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    main = dst / "main.c"
    text = main.read_text(encoding="utf-8")
    needle = """    if (infinite || ponder) {
        p.max_depth     = MAX_PLY - 1;
        p.time_limit_ms = 0;
"""
    repl = """    /* UCI nodes-only search must not inherit DEFAULT_DEPTH=8. Keep an
     * explicitly supplied depth as an additional cap; arena sends nodes only. */
    if (p.node_limit > 0 && !strstr(line, "depth") && movetime <= 0 &&
        !infinite && !ponder && mate <= 0 && wtime <= 0 && btime <= 0) {
        p.max_depth = MAX_PLY - 1;
    }

    if (infinite || ponder) {
        p.max_depth     = MAX_PLY - 1;
        p.time_limit_ms = 0;
"""
    if text.count(needle) != 1:
        raise RuntimeError(
            f"unexpected cmd_go shape in {main}: marker count={text.count(needle)}"
        )
    main.write_text(text.replace(needle, repl), encoding="utf-8")

    search = dst / "search.c"
    text = search.read_text(encoding="utf-8")

    # v3.x checks ss->nodes (reset each ID depth); v5 already checks
    # ss->nodes_total. Match either source form, but require exactly qsearch +
    # alpha_beta so a future code-shape change fails loudly instead of silently
    # producing a bogus benchmark.
    guard_re = re.compile(
        r"if \(ss->nodes(?:_total)? >= ss->node_limit \|\| time_up\(ss\)\)"
    )
    matches = list(guard_re.finditer(text))
    if len(matches) != 2:
        raise RuntimeError(
            f"expected two node-limit guards in {search}, found {len(matches)}"
        )

    if "int    stop_guard;" in text:
        new_guard = (
            "if ((!ss->stop_guard && ss->nodes_total >= ss->node_limit && "
            "(ss->time_up = 1)) || time_up(ss))"
        )
    else:
        new_guard = (
            "if ((ss->nodes_total >= ss->node_limit && (ss->time_up = 1)) "
            "|| time_up(ss))"
        )

    text, n = guard_re.subn(new_guard, text)
    if n != 2:
        raise RuntimeError(f"failed to patch both node guards in {search}: {n}")
    search.write_text(text, encoding="utf-8")

    print(f"patched UCI node semantics: {src} -> {dst}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    a = ap.parse_args()
    patch(a.src, a.dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
