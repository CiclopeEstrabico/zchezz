#!/usr/bin/env python3
"""make_random_nnu4.py — emit a random-weight NNU4 binary for accumulator
verification testing (no trained weights exist yet for v4.00).

Byte layout MUST match `_load_weights` in engine/c/zchezz_v400/nnue.c:

    "NNU4"                       4 B  magic
    epoch                        uint32
    dims[5]                      L1_IN, L1_OUT, L2_IN, L2_OUT, L3_IN  (uint32 x5)
    scales[4]                    QA, QB, SHIFT, OUT_SCALE             (float32 x4)
    L1W  [2560][512] int16       feature-major
    L1B  [512]       int32
    L2W  [32][1024]  int8        output-major (no transpose)
    L2B  [32]        int32
    L3W  [32]        int8
    L3B                          float32

Weights are small random ints chosen to produce varied, non-saturated
evals through the SCReLU/ClippedReLU pipeline:
    L1W  ~ int16 in [-300, 300]
    L1B  ~ int32 in [-100, 100]   (modest bias, same domain as accum)
    L2W  ~ int8  in [-40, 40]
    L2B  ~ int32 in [-500, 500]
    L3W  ~ int8  in [-40, 40]
    L3B  ~ float32 in [-1, 1]

This file is a TEST FIXTURE ONLY — it is not a trained network and must
never be treated as one.
"""
import argparse
import struct
import sys
import numpy as np

# Architecture constants — must match nnue.h
NN_L1_IN, NN_L1_OUT = 2560, 512
NN_L2_IN, NN_L2_OUT = 1024, 32
NN_L3_IN = 32

NN_QA, NN_QB, NN_SHIFT = 255, 64, 8
OUT_SCALE = 1.0 / 64.0   # arbitrary plausible scale; keeps l3_sum*scale in cp range

SEED = 400  # default seed -> reproducible fixture (random_nnu4.bin)
OUT_PATH = "random_nnu4.bin"  # default output path when out_path is omitted


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_path", nargs="?", default=OUT_PATH,
                     help="Output .bin path (default %(default)s).")
    ap.add_argument("--seed", type=int, default=SEED,
                     help="RNG seed (default %(default)s). Use a different "
                          "seed to produce a second, independent fixture "
                          "net — e.g. for the two-net A/B test.")
    args = ap.parse_args()
    out_path = args.out_path
    rng = np.random.default_rng(args.seed)

    l1w = rng.integers(-300, 301, size=(NN_L1_IN, NN_L1_OUT), dtype=np.int32).astype(np.int16)
    l1b = rng.integers(-100, 101, size=NN_L1_OUT, dtype=np.int32)

    l2w = rng.integers(-40, 41, size=(NN_L2_OUT, NN_L2_IN), dtype=np.int32).astype(np.int8)
    l2b = rng.integers(-500, 501, size=NN_L2_OUT, dtype=np.int32)

    l3w = rng.integers(-40, 41, size=NN_L3_IN, dtype=np.int32).astype(np.int8)
    l3b = np.float32(rng.uniform(-1.0, 1.0))

    with open(out_path, "wb") as f:
        f.write(b"NNU4")
        f.write(struct.pack("<I", 0))  # epoch
        f.write(struct.pack("<5I", NN_L1_IN, NN_L1_OUT, NN_L2_IN, NN_L2_OUT, NN_L3_IN))
        f.write(struct.pack("<4f", float(NN_QA), float(NN_QB), float(NN_SHIFT), OUT_SCALE))
        f.write(l1w.tobytes())              # [2560][512] int16, feature-major (C order matches)
        f.write(l1b.astype(np.int32).tobytes())
        f.write(l2w.tobytes())              # [32][1024] int8, output-major (C order matches)
        f.write(l2b.astype(np.int32).tobytes())
        f.write(l3w.tobytes())
        f.write(struct.pack("<f", l3b))

    import os
    print(f"Wrote {out_path} ({os.path.getsize(out_path):,} bytes)")


if __name__ == "__main__":
    main()
