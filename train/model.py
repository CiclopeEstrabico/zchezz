"""
train/model.py — HalfKP-4Bucket NNUE model + QAT helpers (Zchezz v4.00)

Architecture (must match engine/c/zchezz_v400/nnue.h and PARTE 5 of
zchezz_v400_implementation_plan.md):

    Features  : HalfKP-4Bucket, 2560 per perspective (encoding.py)
    L1        : 2560 -> 512, SHARED between both perspectives
    Act L1    : SCReLU  —  c = clamp(x, 0, 1); out = c*c
    Concat    : [L1(stm) 512 | L1(opp) 512] = 1024   (STM ALWAYS FIRST)
    L2        : 1024 -> 32, ClippedReLU [0, 1]
    L3        : 32   -> 1,  sigmoid -> WDL probability, STM-relative

Every weight tensor is QAT-fake-quantized on every forward pass with the
straight-through estimator (STE) pattern ported verbatim from v3.14's
train_nnue.py (see zchezz_train_recon.md), with SCALES updated per APPENDIX E
of the v400 plan (the QA_EFF=254 business below is new in v400 — v3.14 used
QA for both the L1 weight scale AND the L1 activation fake-quant scale,
which is wrong for v400 because SCReLU's output range is [0, (255*255)>>8]
= [0, 254], not [0, 255]).

── Why nn.EmbeddingBag instead of nn.Linear for L1 ─────────────────────────

A HalfKP position has at most 30 active features out of 2560 per
perspective (see encoding.py's docstring on dense-vs-sparse). Running a
dense nn.Linear(2560, 512) forward means multiplying by ~2530 zero columns
per sample for nothing. nn.EmbeddingBag(2560, 512, mode='sum') solves this
exactly: given the flat indices of active features and their per-sample
offsets (the same "offsets" bagging protocol EmbeddingBag expects), it
computes sum_i weight[i, :] over the active indices — mathematically
identical to `x_onehot(2560) @ W^T` for an nn.Linear-style W of shape
[512, 2560], but without ever materializing the dense one-hot vector or
looping over the 2530 zero features per sample.

*** THIS HAS A CONSEQUENCE THAT MUST BE PRESERVED ALL THE WAY TO
    export_nnu4.py: *** an EmbeddingBag's weight tensor has shape
    [num_embeddings, embedding_dim] = [2560, 512] — i.e. it is ALREADY
    stored "feature-major" (one contiguous 512-wide row per feature index),
    which is exactly the on-disk layout nnue.c's loader expects for L1
    ("L1W [2560][512] int16 feature-major", see nnue.c's `_load_weights`
    doc comment). The v3.14-derived plan pseudocode in PARTE 6 of the
    implementation plan assumes an nn.Linear-shaped `l1.weight` of
    [512, 2560] and therefore transposes it before writing the NNU4 file.
    Because this model uses EmbeddingBag, `l1.weight` here is ALREADY
    [2560, 512] — export_nnu4.py in this v400 tree does NOT transpose L1.
    This is flagged again, loudly, in export_nnu4.py's own docstring.

The math is unaffected: EmbeddingBag(mode='sum') applied to the active-
feature indices of position p computes
    acc_p = sum_{i active} W[i, :]
which is the same vector nn.Linear(x_onehot, W.T) would produce, since
W.T's *columns* i are the ones being selected/summed for a one-hot x — and
W.T's column i is exactly EmbeddingBag's row i (`W[i, :]`).

── Bias handling ────────────────────────────────────────────────────────

torch.nn.EmbeddingBag has no bias parameter. L1's bias is therefore a
plain nn.Parameter(512,), fake-quantized exactly like an nn.Linear bias
would be, and added manually after the bag-sum. Both perspectives share
the SAME bias (same as they share the same weight matrix) — there is only
ever one L1 (weight, bias) pair in this model.

── Public surface ───────────────────────────────────────────────────────

  SCReLU, ClippedReLU             — activation modules
  fake_quant_int16/int8/bias_int32 — QAT STE helpers, ported from v3.14
  clamp_weights_(model)           — post-optimizer-step hard clamp
  NNUE                            — the model itself
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ═══════════════ CONFIGURATION (architecture / quantization constants) ═══════════════
# Library module — no CLI. These MUST match engine/c/zchezz_v400/nnue.h and
# train/export_nnu4.py exactly; a drift here silently produces a checkpoint
# that trains fine but evaluates as garbage once exported.
QA = 255           # L1 weight/bias scale, and int16 range for L1W
QB = 64             # L2/L3 weight scale, int8 range
NN_SHIFT = 8        # informational only here — the >>8 in the C integer
                    # forward pass is implicit in the QA_EFF definition below,
                    # the float QAT forward never needs an explicit shift
                    # (see APPENDIX E.1: the float path operates on true
                    # real-valued activations, the shift only exists to
                    # rescale the *integer* accumulator in nnue.c).
QA_EFF = (QA * QA) >> NN_SHIFT   # = 254 — effective scale of SCReLU's output
assert QA_EFF == 254

INPUT_DIM = 2560    # per-perspective HalfKP-4Bucket feature count
HIDDEN1 = 512       # L1 output width, per perspective (shared weights)
CONCAT_DIM = HIDDEN1 * 2   # 1024, [stm | opp]
HIDDEN2 = 32        # L2 output width
# ════════════════════════════════════════════════════════════════════════════════════


# ── QAT fake-quant helpers (straight-through estimator) ───────────────────
# Ported verbatim from v3.14's train_nnue.py (see zchezz_train_recon.md,
# lines 344-358) — architecture-agnostic, only the *scales* passed in
# differ for v400 (see APPENDIX E of the plan).

def fake_quant_int16(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """Simulate rounding `tensor * scale` to an int16 and dividing back,
    with a straight-through gradient (backward pass sees the identity)."""
    limit = 32767.0 / scale
    x_clamp = tensor.clamp(-limit, limit)
    x_q = (x_clamp * scale).round() / scale
    return tensor + (x_q - tensor).detach()


def fake_quant_int8(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """Same as fake_quant_int16 but for the int8 range (L2W, L3W)."""
    limit = 127.0 / scale
    x_clamp = tensor.clamp(-limit, limit)
    x_q = (x_clamp * scale).round() / scale
    return tensor + (x_q - tensor).detach()


def fake_quant_bias_int32(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """Fake-quant for bias tensors: int32 range is large enough in
    practice that no clamp is applied, only round+STE."""
    x_q = (tensor * scale).round() / scale
    return tensor + (x_q - tensor).detach()


class SCReLU(nn.Module):
    """Squared Clipped ReLU: c = clamp(x, 0, 1); out = c * c.

    This is v400's L1 activation (replacing v3.14's ClippedReLU on L1).
    APPENDIX E.1 of the plan gives the exact fake-quant scale to apply to
    its output: QA_EFF = 254 = (QA*QA) >> 8, NOT QA=255 — the squared
    activation's natural output range after quantization is [0, 254], and
    quantizing it at scale 255 would waste a code point and silently
    mismatch the integer C forward pass in nnue.c (which uses `(c*c)>>8`
    and therefore also lands in [0, 254]).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.clamp(0.0, 1.0)
        return c * c


class ClippedReLU(nn.Module):
    """out = clamp(x, 0, clip). L2's activation (unchanged from v3.14)."""

    def __init__(self, clip: float = 1.0):
        super().__init__()
        self.clip = clip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(0.0, self.clip)

    def extra_repr(self) -> str:
        return f"clip={self.clip}"


class NNUE(nn.Module):
    """HalfKP-4Bucket dual-perspective NNUE, QAT-trained in float32.

    forward() accepts SPARSE feature batches in the EmbeddingBag "offsets"
    protocol (see encoding.py::encode_positions) for both perspectives.
    Both perspectives are pushed through the SAME L1 (self.l1_weight /
    self.l1_bias) — there is only one L1 layer in this model, applied
    twice.
    """

    def __init__(self) -> None:
        super().__init__()

        # L1: nn.EmbeddingBag stores its weight as [num_embeddings=2560,
        # embedding_dim=512] — i.e. ALREADY the "feature-major" layout
        # nnue.c wants on disk (see module docstring). mode='sum' turns
        # "look up these active indices and sum their rows" into exactly
        # the accumulator-add semantics of a real NNUE.
        self.l1 = nn.EmbeddingBag(INPUT_DIM, HIDDEN1, mode="sum")
        self.l1_bias = nn.Parameter(torch.zeros(HIDDEN1))
        self.act1 = SCReLU()

        self.l2 = nn.Linear(CONCAT_DIM, HIDDEN2)
        self.act2 = ClippedReLU(1.0)

        self.l3 = nn.Linear(HIDDEN2, 1)

        # Standard NNUE-friendly init: small uniform range keeps the
        # SCReLU/ClippedReLU activations from saturating at step 0.
        nn.init.uniform_(self.l1.weight, -0.05, 0.05)
        nn.init.zeros_(self.l1_bias)

    def _l1_perspective(self, indices: torch.Tensor, offsets: torch.Tensor,
                         w1: torch.Tensor, b1: torch.Tensor) -> torch.Tensor:
        acc = F.embedding_bag(indices, w1, offsets, mode="sum")
        acc = acc + b1
        return self.act1(acc)

    def forward(self, stm_idx: torch.Tensor, stm_off: torch.Tensor,
                opp_idx: torch.Tensor, opp_off: torch.Tensor) -> torch.Tensor:
        """
        stm_idx/stm_off : flat int32/int64 EmbeddingBag inputs for the
                           side-to-move's perspective (see encoding.py).
        opp_idx/opp_off : same, for the opponent's perspective.

        Returns: (batch,) float tensor in [0, 1], the STM-relative WDL
        probability estimate (sigmoid of L3's output).
        """
        # ── L1 (shared weights, fake-quantized once, used twice) ────────
        w1 = fake_quant_int16(self.l1.weight, QA)
        b1 = fake_quant_bias_int32(self.l1_bias, float(QA))

        h_stm = self._l1_perspective(stm_idx, stm_off, w1, b1)
        h_opp = self._l1_perspective(opp_idx, opp_off, w1, b1)

        # Fake-quant SCReLU's output at QA_EFF (254), NOT QA (255) —
        # see SCReLU's docstring / APPENDIX E.1.
        h_stm_q = (h_stm * QA_EFF).round().clamp(0, QA_EFF) / QA_EFF
        h_stm = h_stm + (h_stm_q - h_stm).detach()
        h_opp_q = (h_opp * QA_EFF).round().clamp(0, QA_EFF) / QA_EFF
        h_opp = h_opp + (h_opp_q - h_opp).detach()

        # ── Concat: STM ALWAYS FIRST ─────────────────────────────────
        h = torch.cat([h_stm, h_opp], dim=1)   # (batch, 1024)

        # ── L2 ────────────────────────────────────────────────────────
        w2 = fake_quant_int8(self.l2.weight, QB)
        # APPENDIX E.2: L2 bias scale is QA_EFF*QB (254*64=16256); QA*QB
        # (255*64=16320) is documented as an acceptable alternative
        # (~0.39% error) but QA_EFF*QB is the exact match to the integer
        # C forward pass, so it is what we use here.
        b2 = fake_quant_bias_int32(self.l2.bias, float(QA_EFF * QB))
        h2 = self.act2(F.linear(h, w2, b2))

        h2_q = (h2 * QB).round().clamp(0, QB) / QB
        h2 = h2 + (h2_q - h2).detach()

        # ── L3 (bias NOT quantized — kept float32 end to end) ───────────
        w3 = fake_quant_int8(self.l3.weight, QB)
        out = torch.sigmoid(F.linear(h2, w3, self.l3.bias))
        return out.squeeze(1)

    def forward_dense(self, x_stm: torch.Tensor, x_opp: torch.Tensor) -> torch.Tensor:
        """Reference dense forward pass — takes (batch, 2560) float/uint8
        one-hot-ish tensors instead of sparse indices. Mathematically
        identical to forward() for the same positions; used only for
        parity testing (small batches, encoding.py's dense path) since it
        is O(batch * 2560 * 512) instead of O(batch * 30 * 512).
        """
        w1 = fake_quant_int16(self.l1.weight, QA)          # [2560, 512]
        b1 = fake_quant_bias_int32(self.l1_bias, float(QA))

        # EmbeddingBag weight is [2560, 512] == W^T of an nn.Linear-style
        # [512, 2560] matrix, so F.linear(x, w1.t(), b1) reproduces the
        # exact same accumulator as the sparse EmbeddingBag path.
        h_stm = self.act1(F.linear(x_stm.float(), w1.t(), b1))
        h_opp = self.act1(F.linear(x_opp.float(), w1.t(), b1))

        h_stm_q = (h_stm * QA_EFF).round().clamp(0, QA_EFF) / QA_EFF
        h_stm = h_stm + (h_stm_q - h_stm).detach()
        h_opp_q = (h_opp * QA_EFF).round().clamp(0, QA_EFF) / QA_EFF
        h_opp = h_opp + (h_opp_q - h_opp).detach()

        h = torch.cat([h_stm, h_opp], dim=1)

        w2 = fake_quant_int8(self.l2.weight, QB)
        b2 = fake_quant_bias_int32(self.l2.bias, float(QA_EFF * QB))
        h2 = self.act2(F.linear(h, w2, b2))

        h2_q = (h2 * QB).round().clamp(0, QB) / QB
        h2 = h2 + (h2_q - h2).detach()

        w3 = fake_quant_int8(self.l3.weight, QB)
        out = torch.sigmoid(F.linear(h2, w3, self.l3.bias))
        return out.squeeze(1)


def clamp_weights_(model: NNUE) -> None:
    """In-place hard clamp of all quantized weight/bias tensors to their
    representable int range at the FIXED target scale. Call this after
    every optimizer step (see train_nnue.py's training loop) — this is what
    makes export_nnu4.py's post-hoc quantization safe: with this clamp
    applied every step, `_quant16`/`_quant8`'s overflow-warning path
    should never actually trigger.

    L2 bias / L3 bias are intentionally NOT clamped: they are int32/float32
    respectively, with enough dynamic range that clamping them is not
    necessary (matches v3.14's clamp_weights_ exactly — see
    zchezz_train_recon.md lines 375-384).
    """
    with torch.no_grad():
        lim1 = 32767.0 / QA
        model.l1.weight.clamp_(-lim1, lim1)
        model.l1_bias.clamp_(-lim1, lim1)

        lim2 = 127.0 / QB
        model.l2.weight.clamp_(-lim2, lim2)
        model.l3.weight.clamp_(-lim2, lim2)


if __name__ == "__main__":
    # Tiny smoke test: random sparse batch through both forward paths,
    # check they agree (within float rounding) and that the model runs.
    import numpy as np
    from encoding import encode_positions, encode_positions_dense

    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    ]
    stm_idx, stm_off, opp_idx, opp_off = encode_positions(fens)
    x_stm, x_opp = encode_positions_dense(fens)

    model = NNUE()
    model.eval()
    with torch.no_grad():
        out_sparse = model(
            torch.from_numpy(stm_idx).long(), torch.from_numpy(stm_off),
            torch.from_numpy(opp_idx).long(), torch.from_numpy(opp_off),
        )
        out_dense = model.forward_dense(
            torch.from_numpy(x_stm), torch.from_numpy(x_opp)
        )

    print("sparse:", out_sparse.tolist())
    print("dense :", out_dense.tolist())
    max_diff = (out_sparse - out_dense).abs().max().item()
    print(f"max |sparse-dense| diff = {max_diff:.3e}")
    assert max_diff < 1e-5, "sparse and dense forward passes disagree"
    print("OK: sparse/dense forward parity")
