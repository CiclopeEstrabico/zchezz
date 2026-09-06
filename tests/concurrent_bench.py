#!/usr/bin/env python3
"""Measure fixed-depth engine throughput as process concurrency increases."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

NODES_RE = re.compile(r"(?i)Total nodes\s*:\s*([0-9,]+)")


def one(exe: str, depth: int) -> dict:
    t0 = time.perf_counter_ns()
    p = subprocess.run(
        [exe, "bench", str(depth)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
        timeout=300,
    )
    wall = (time.perf_counter_ns() - t0) / 1e9
    m = NODES_RE.findall(p.stdout)
    if not m:
        raise RuntimeError(p.stdout[-2000:])
    nodes = int(m[-1].replace(",", ""))
    return {"nodes": nodes, "wall_s": wall, "nps": nodes / wall}


def batch(exe: str, depth: int, concurrency: int) -> dict:
    t0 = time.perf_counter_ns()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        runs = list(pool.map(lambda _: one(exe, depth), range(concurrency)))
    batch_wall = (time.perf_counter_ns() - t0) / 1e9
    total_nodes = sum(r["nodes"] for r in runs)
    return {
        "batch_wall_s": batch_wall,
        "aggregate_nps": total_nodes / batch_wall,
        "median_process_nps": statistics.median(r["nps"] for r in runs),
        "runs": runs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", action="append", required=True, help="NAME=PATH")
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--concurrency", default="1,2,4")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    engines = [s.split("=", 1) for s in args.engine]
    concs = [int(x) for x in args.concurrency.split(",")]
    out = {"depth": args.depth, "repeats": args.repeats, "engines": {}}

    for name, exe in engines:
        out["engines"][name] = {}
        for c in concs:
            batches = []
            for _ in range(args.repeats):
                b = batch(exe, args.depth, c)
                batches.append(b)
                print(name, "concurrency", c, "aggregate_nps", round(b["aggregate_nps"]), flush=True)
            out["engines"][name][str(c)] = {
                "median_aggregate_nps": statistics.median(b["aggregate_nps"] for b in batches),
                "median_process_nps": statistics.median(b["median_process_nps"] for b in batches),
                "batches": batches,
            }

    base = engines[0][0]
    out["ratios"] = {}
    for c in concs:
        key = str(c)
        b = out["engines"][base][key]["median_process_nps"]
        for name, _ in engines[1:]:
            n = out["engines"][name][key]["median_process_nps"]
            out["ratios"][f"{base}_over_{name}_process_nps_c{c}"] = b / n

    Path(args.json).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["ratios"], indent=2))


if __name__ == "__main__":
    main()
