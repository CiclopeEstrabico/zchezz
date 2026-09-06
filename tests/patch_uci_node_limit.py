#!/usr/bin/env python3
"""Materialize a temporary engine copy with correct UCI `go nodes` semantics.

This is deliberately a benchmark-side patch first.  It lets us invalidate or
confirm historical "fixed-node" Elo results without changing an accepted
engine baseline.  Once validated, the same two changes can be promoted into
the engine sources:
  1) a nodes-only UCI search must not inherit DEFAULT_DEPTH=8;
  2) reaching the whole-search node budget must mark the current iteration
     interrupted (`time_up=1`) so iterative deepening stops and cannot publish
     a post-budget pseudo-iteration.
"""
from __future__ import annotations
import argparse, shutil
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
    repl = """    /* UCI nodes-only search must not inherit DEFAULT_DEPTH=8.  Keep an
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
        raise RuntimeError(f"unexpected cmd_go shape in {main}: marker count={text.count(needle)}")
    main.write_text(text.replace(needle, repl), encoding="utf-8")

    search = dst / "search.c"
    text = search.read_text(encoding="utf-8")
    old = "if (ss->nodes_total >= ss->node_limit || time_up(ss))"
    new = "if ((!ss->stop_guard && ss->nodes_total >= ss->node_limit && (ss->time_up = 1)) || time_up(ss))"
    count = text.count(old)
    if count != 2:
        raise RuntimeError(f"expected two node-limit guards in {search}, found {count}")
    search.write_text(text.replace(old, new), encoding="utf-8")
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
