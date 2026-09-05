"""NNU3 799->256->64->1 QAT model used by the v3.22 engine line.

This is the v3.14-family architecture, kept separate from model.py's NNU4
HalfKP-4Bucket network. Quantization constants and fake-quant behavior are
compatible with engine/c/zchezz_v322/nnue.c and export_nnu3.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_MAIN = 768
INPUT_EXTRA = 31
INPUT_DIM = INPUT_MAIN + INPUT_EXTRA
HIDDEN1 = 256
HIDDEN2 = 64
QA = 255.0
QB = 64.0
RELU_CLIP = 1.0
ENCODING = "nnu3_hm_768_plus_31"


def _fake_quant(t: torch.Tensor, scale: float, qmax: float) -> torch.Tensor:
    limit = qmax / scale
    q = (t.clamp(-limit, limit) * scale).round() / scale
    return t + (q - t).detach()


def fake_quant_int16(t: torch.Tensor, scale: float = QA) -> torch.Tensor:
    return _fake_quant(t, scale, 32767.0)


def fake_quant_int8(t: torch.Tensor, scale: float = QB) -> torch.Tensor:
    return _fake_quant(t, scale, 127.0)


def fake_quant_bias_int32(t: torch.Tensor, scale: float) -> torch.Tensor:
    q = (t * scale).round() / scale
    return t + (q - t).detach()


class NNUE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(INPUT_DIM, HIDDEN1)
        self.l2 = nn.Linear(HIDDEN1, HIDDEN2)
        self.l3 = nn.Linear(HIDDEN2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1 = fake_quant_int16(self.l1.weight, QA)
        b1 = fake_quant_bias_int32(self.l1.bias, QA)
        h1 = F.linear(x, w1, b1).clamp(0.0, RELU_CLIP)
        h1q = (h1 * QA).round().clamp(0.0, QA) / QA
        h1 = h1 + (h1q - h1).detach()

        w2 = fake_quant_int8(self.l2.weight, QB)
        b2 = fake_quant_bias_int32(self.l2.bias, QA * QB)
        h2 = F.linear(h1, w2, b2).clamp(0.0, RELU_CLIP)
        h2q = (h2 * QB).round().clamp(0.0, QB) / QB
        h2 = h2 + (h2q - h2).detach()

        w3 = fake_quant_int8(self.l3.weight, QB)
        return torch.sigmoid(F.linear(h2, w3, self.l3.bias)).squeeze(1)


def clamp_weights_(model: NNUE) -> None:
    with torch.no_grad():
        lim1 = 32767.0 / QA
        model.l1.weight.clamp_(-lim1, lim1)
        model.l1.bias.clamp_(-lim1, lim1)
        lim8 = 127.0 / QB
        model.l2.weight.clamp_(-lim8, lim8)
        model.l3.weight.clamp_(-lim8, lim8)


def architecture_dict() -> dict[str, int | str]:
    return {
        "input": INPUT_DIM,
        "h1": HIDDEN1,
        "h2": HIDDEN2,
        "encoding": ENCODING,
        "format": "NNU3",
    }


def assert_compatible_arch(arch: dict | None) -> None:
    if not arch:
        raise ValueError("checkpoint has no architecture metadata")
    expected = architecture_dict()
    for key in ("input", "h1", "h2", "encoding"):
        if arch.get(key) != expected[key]:
            raise ValueError(
                f"incompatible checkpoint architecture: {key}={arch.get(key)!r}, "
                f"expected {expected[key]!r}"
            )
