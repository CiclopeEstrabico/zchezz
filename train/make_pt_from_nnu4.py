"""
train/make_pt_from_nnu4.py — NNU4 binary -> train_nnue.py checkpoint (.pt)

Inverse of export_nnu4.py. Exists because the ORIGINAL v402 training
checkpoint is not on disk anymore (only its exported nnue_weights.bin),
but warm-starting ("transfer") from the v402 net requires a loadable .pt.
Dequantization uses the SAME scales export_nnu4.py wrote (see its header):
L1W/L1B by QA=255, L2W/L3W by QB=64, L2B by QA_EFF*QB=16256, L3B raw
float. The round-trip is exact to within float rounding (quantize(
dequantize(q)) == q), verified by --verify below.

USAGE
    python train/make_pt_from_nnu4.py                     # CONFIGURATION block
    python train/make_pt_from_nnu4.py --verify            # round-trip check
"""

import argparse
import struct
from pathlib import Path

import numpy as np
import torch

# ── Architecture dims (must match nnue.h / export_nnu4.py / model.py) ─────
L1_IN, L1_OUT, L2_IN, L2_OUT, L3_IN = 2560, 512, 1024, 32, 32
QA, QB, SHIFT = 255.0, 64.0, 8.0
QA_EFF = 254.0
OUT_SCALE = 320.0 / (QB * QB)
MAGIC = b"NNU4"

# ═══════════════ CONFIGURATION ═══════════════
SRC_NNU4 = Path("C:/Zchezz/engine/c/zchezz_v402/nnue_weights.bin")  # input NNU4 binary
DST_PT   = Path("C:/Zchezz/checkpoints/v402/v402_from_nnu4.pt")     # output .pt path
TAG      = "v402_restored"                                          # dataset-name stored in ckpt
EPOCH    = 0                                                        # epoch number to record
# ═════════════════════════════════════════════


def read_nnu4(path: Path) -> dict:
    blob = path.read_bytes()
    off = 0
    def take(fmt):
        nonlocal off
        vals = struct.unpack_from(fmt, blob, off)
        off += struct.calcsize(fmt)
        return vals
    assert blob[:4] == MAGIC, "not an NNU4 file"
    take("<4s")
    (epoch,) = take("<I")
    dims = take("<5I")
    scales = take("<4f")
    l1_in, l1_out, l2_in, l2_out, l3_in = dims

    # straightforward sequential reads:
    def arr(dtype, count):
        nonlocal off
        a = np.frombuffer(blob, dtype=dtype, count=count, offset=off)
        off += np.dtype(dtype).itemsize * count
        return a
    w = {
        "l1.weight": arr("<i2", l1_in * l1_out).reshape(l1_in, l1_out).astype(np.float32) / QA,
        "l1_bias":   arr("<i4", l1_out).astype(np.float32) / QA,
        "l2.weight": arr("i1", l2_out * l2_in).reshape(l2_out, l2_in).astype(np.float32) / QB,
        "l2.bias":   arr("<i4", l2_out).astype(np.float32) / (QA_EFF * QB),
        "l3.weight": arr("i1", l3_in).reshape(1, l3_in).astype(np.float32) / QB,
        "l3.bias":   arr("<f4", 1).astype(np.float32),
    }
    assert off == len(blob), f"trailing bytes: {len(blob) - off}"
    return {"epoch": epoch, "dims": dims, "scales": scales, "weights": w}


def build_ckpt(parsed: dict, tag: str, epoch_override: int | None) -> dict:
    w_t = {k: torch.from_numpy(v.copy()) for k, v in parsed["weights"].items()}
    return {
        "arch": {"input": L1_IN, "h1": L1_OUT, "concat": L2_IN, "h2": L2_OUT,
                 "encoding": "halfkp_4bucket"},
        "weights": w_t,
        "epoch": parsed["epoch"] if epoch_override is None else epoch_override,
        "dataset_name": tag,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Rebuild a trainable .pt from an NNU4 binary")
    p.add_argument("--src", type=Path, default=SRC_NNU4)
    p.add_argument("--dst", type=Path, default=DST_PT)
    p.add_argument("--tag", default=TAG)
    p.add_argument("--verify", action="store_true",
                   help="after writing, re-export the rebuilt weights through "
                        "export_nnu4.convert() and byte-compare against --src")
    a = p.parse_args()

    parsed = read_nnu4(a.src)
    print(f"read {a.src}: dims={parsed['dims']} scales={parsed['scales']}")
    ckpt = build_ckpt(parsed, a.tag, None)
    a.dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, a.dst)
    print(f"wrote {a.dst} ({a.dst.stat().st_size:,} bytes)")

    if a.verify:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        import export_nnu4
        tmp = a.dst.with_suffix(".roundtrip.bin")
        export_nnu4.convert(a.dst, tmp)
        same = tmp.read_bytes() == a.src.read_bytes()
        print(f"round-trip byte-identical: {same}")
        tmp.unlink()
        if not same:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
