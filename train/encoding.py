"""
train/encoding.py — HalfKP-4Bucket feature encoding (Zchezz v4.00)

This module is the SINGLE SOURCE OF TRUTH, on the Python side, for how a
chess.Board is turned into NNUE input features. It is imported by
train_nnue.py (training), export_nnu4.py (indirectly, via checkpoints
produced using this encoding) and check_parity.py (cross-checked against
the C implementation in engine/c/zchezz_v400/nnue.c).

Read engine/c/zchezz_v400/nnue.h in full before touching this file. Every
invariant below is copied from that header's big comment block and from
APPENDIX D of zchezz_v400_implementation_plan.md — nnue.h wins if the two
ever disagree (they should not; if you find a disagreement, fix this file,
never the other way around, and flag it loudly).

── Coordinate convention (the #1 way to silently ship a broken net) ──────

  python-chess square      : 0 = a1 .. 63 = h8.  This IS the White-POV
                              square (Zchezz calls it sq_w).
  Black-POV square          : sq_b = sq_w ^ 56  (vertical mirror: file
                              unchanged, rank flipped: e.g. e2(12)->e7(52)).

  This differs from Zchezz's own mailbox convention (zsq, 0=a8..63=h1,
  i.e. zsq = sq_w ^ 56) — encoding.py works entirely in python-chess
  coordinates (sq_w) and never touches zsq. The C engine converts zsq to
  sq_w via `zsq ^ 56` before computing bucket/feature index (see nnue.c
  `_king_bucket`/`nnue_feature_index`), so the two sides agree once both
  are expressed in the same POV-square space. This is exactly the mapping
  documented in nnue.h's "Coordinate and feature conventions" section.

── King bucket ────────────────────────────────────────────────────────────

  king_bucket(s) for a POV square s (0..63):
      bit 0 = (s % 8) >= 4      -> kingside half of the board
      bit 1 = (s // 8) >= 4     -> "far" half in this POV (ranks 5-8)

  bucket_white = king_bucket(white_king_sq)          # sq already White-POV
  bucket_black = king_bucket(black_king_sq ^ 56)      # mirror to Black-POV

  This one-line asymmetry (white: no flip: black: flip) is intentional and
  matches nnue.c's nnue_king_bucket_w/b exactly:
      nnue_king_bucket_w(wk_zsq) = _king_bucket(wk_zsq ^ 56) = _king_bucket(sq_w)
      nnue_king_bucket_b(bk_zsq) = _king_bucket(bk_zsq)      = _king_bucket(sq_w ^ 56)
  (because zsq = sq_w ^ 56, so wk_zsq ^ 56 == sq_w, and bk_zsq == sq_w ^ 56
  for the black king's own sq_w). Don't "simplify" this by flipping both
  sides the same way — it is asymmetric on purpose.

── Feature index ───────────────────────────────────────────────────────────

  Relative piece plane (the KING IS NEVER A FEATURE):
      ally   P,N,B,R,Q -> 0,1,2,3,4
      enemy  P,N,B,R,Q -> 5,6,7,8,9

  feature = bucket * 640 + rel * 64 + pov_square        (0 .. 2559)

  pov_square for a perspective is the piece's square already expressed in
  that perspective's POV (sq_w for White's perspective, sq_w ^ 56 for
  Black's perspective) — NOT the same as the king square used for the
  bucket lookup (obviously; every piece has its own square).

── Two perspectives per position, STM always first downstream ────────────

  x_stm = halfkp_features(board, pov_is_white = (board.turn == chess.WHITE))
  x_opp = halfkp_features(board, pov_is_white = (board.turn != chess.WHITE))

  model.forward(x_stm, x_opp) concatenates [h(x_stm) | h(x_opp)] with STM
  ALWAYS FIRST. Swapping the order silently flips the sign of every
  evaluation the net has ever seen during training vs what the engine will
  ask of it at inference time — this is the single most dangerous mistake
  to reintroduce while refactoring this file.

── Dense vs sparse representation ──────────────────────────────────────────

  A HalfKP-4Bucket position has at most 30 active features per perspective
  (30 non-king pieces on the board, in the extreme case of promotions
  aside; a legal start-of-game-ish position has at most 30: 16+16-2 kings).
  A dense (N, 2560) uint8 array costs 2.5 KB/position/perspective — for
  100M+ positions (the kind of dataset size train_nnue.py's docstring
  targets) that is 250+ GB just for one perspective's dense encoding, which
  does not fit on disk let alone in RAM.

  The SPARSE representation — the list of active feature indices per
  position — costs at most 30 * 4 bytes = 120 bytes/position/perspective,
  about 20x smaller, and is what nn.EmbeddingBag(mode='sum') / index_add
  are built to consume directly: EmbeddingBag(2560, 512, mode='sum') has a
  weight matrix of shape [2560, 512] and, given a flat index tensor plus a
  parallel offsets tensor (the standard "offsets" bagging protocol used by
  PyTorch's own recommendex API), computes exactly
      sum_{i in active features of position p} weight[i, :]
  which is mathematically identical to the dense formulation
      x_onehot(2560) @ W^T   where W is an nn.Linear-style [512, 2560]
  matrix and W^T == the EmbeddingBag weight — see model.py's docstring for
  why this also happens to sidestep a transpose at export time.

  `encode_positions()` therefore defaults to the SPARSE (indices, offsets)
  representation. The DENSE path (`halfkp_features`, `encode_positions_dense`)
  is kept because it is easier to eyeball/unit-test and is what
  check_parity.py and small ad-hoc scripts use.

── Dependencies ────────────────────────────────────────────────────────────

  Deliberately dependency-light: only `python-chess` and `numpy`. Both
  train_nnue.py (training, needs speed + multiprocessing-friendliness) and
  check_parity.py (needs to import this in isolation, no torch) import
  this module, so it must never import torch or anything training-loop
  specific.
"""

from __future__ import annotations

import chess
import numpy as np

# ── Architecture constants (mirror nnue.h) ─────────────────────────────
NN_KING_BUCKETS = 4
NN_FEAT_PER_BUCKET = 640          # 10 relative piece planes x 64 squares
NN_FEAT_IN = 2560                 # NN_KING_BUCKETS * NN_FEAT_PER_BUCKET
# Upper bound on active features per perspective.
#
# A LEGAL chess position has at most 30 non-king pieces (32 - 2 kings), and
# this constant used to be 30 for that reason. That was wrong in practice:
# the historical parquet datasets in data/ contain synthetic stress
# positions that are not legal chess — observed examples include a board
# with 47 non-king pieces (ranks packed with rooks) and several with 31-32.
# v3.14's DENSE encoder silently tolerated them (it just set bits in a
# 768-wide array); this sparse encoder wrote past the end of a fixed
# 30-element buffer and raised IndexError instead.
#
# The bound is now "every square except the two kings", which no position
# representable on a 64-square board can exceed. Positions that exceed the
# LEGAL limit of 30 are still garbage and should not be trained on — they
# are filtered upstream in train_nnue.py, which reports how many it drops.
# This constant only guarantees the encoder cannot crash on them.
MAX_ACTIVE_FEATURES = 62          # 64 squares - 2 kings
LEGAL_MAX_ACTIVE_FEATURES = 30    # 32 pieces - 2 kings, for validity checks

# Relative piece-type index. King is deliberately absent (index -1 below).
# python-chess piece types: PAWN=1, KNIGHT=2, BISHOP=3, ROOK=4, QUEEN=5, KING=6
_PIECE_TYPE_TO_REL = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    # chess.KING intentionally not mapped: never a feature.
}


def king_bucket(pov_square: int) -> int:
    """King bucket for a king already expressed in POV-square coordinates.

    `pov_square` must already be POV-adjusted by the caller: pass the raw
    python-chess square for White's perspective, and `sq ^ 56` for Black's
    perspective. See module docstring — this asymmetry is intentional and
    must match nnue.c's `_king_bucket` composed with `nnue_king_bucket_w/b`.
    """
    file_ = pov_square & 7
    rank_ = pov_square >> 3
    return (1 if file_ >= 4 else 0) | (2 if rank_ >= 4 else 0)


def halfkp_features(board: "chess.Board", pov_is_white: bool) -> np.ndarray:
    """Dense (2560,) uint8 HalfKP-4Bucket feature vector for one perspective.

    Returns an all-zero vector if the perspective's king is missing from
    the board (should not happen for legal positions, but callers building
    synthetic/partial boards may hit this — fail soft, not with an
    exception, matching the reference implementation in the plan).
    """
    feats = np.zeros(NN_FEAT_IN, dtype=np.uint8)

    king_color = chess.WHITE if pov_is_white else chess.BLACK
    king_sq = board.king(king_color)
    if king_sq is None:
        return feats

    pov_king_sq = king_sq if pov_is_white else (king_sq ^ 56)
    bucket = king_bucket(pov_king_sq)
    base = bucket * NN_FEAT_PER_BUCKET

    for sq, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        rel_type = _PIECE_TYPE_TO_REL.get(piece.piece_type)
        if rel_type is None:
            continue

        is_ally = (piece.color == king_color)
        color_offset = 0 if is_ally else 5
        pov_sq = sq if pov_is_white else (sq ^ 56)

        feat_idx = base + (color_offset + rel_type) * 64 + pov_sq
        feats[feat_idx] = 1

    return feats


def halfkp_active_indices(board: "chess.Board", pov_is_white: bool) -> np.ndarray:
    """Sparse encoding: int32 array of the (<=30) active feature indices
    for one perspective, in arbitrary order (order does not matter for
    EmbeddingBag(mode='sum') / index_add, both of which are order-
    invariant reductions).

    This is the fast path used by encode_positions() by default, and the
    one model.py's NNUE.forward() consumes.
    """
    king_color = chess.WHITE if pov_is_white else chess.BLACK
    king_sq = board.king(king_color)
    if king_sq is None:
        return np.zeros(0, dtype=np.int32)

    pov_king_sq = king_sq if pov_is_white else (king_sq ^ 56)
    bucket = king_bucket(pov_king_sq)
    base = bucket * NN_FEAT_PER_BUCKET

    indices = np.empty(MAX_ACTIVE_FEATURES, dtype=np.int32)
    n = 0
    for sq, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        rel_type = _PIECE_TYPE_TO_REL.get(piece.piece_type)
        if rel_type is None:
            continue

        is_ally = (piece.color == king_color)
        color_offset = 0 if is_ally else 5
        pov_sq = sq if pov_is_white else (sq ^ 56)

        indices[n] = base + (color_offset + rel_type) * 64 + pov_sq
        n += 1

    return indices[:n].copy()


def king_buckets_of(board: "chess.Board") -> tuple[int, int]:
    """Convenience: (bucket_white, bucket_black) for a position, using the
    same asymmetric POV rule as everywhere else in this file."""
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    bw = king_bucket(wk) if wk is not None else -1
    bb = king_bucket(bk ^ 56) if bk is not None else -1
    return bw, bb


# ── Batch encoders ──────────────────────────────────────────────────────

def encode_positions_dense(fens: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Dense reference encoder: (X_stm, X_opp), each (N, 2560) uint8.

    O(N * 2560) memory — only use for small batches (tests, parity
    checking, tiny debugging runs). train_nnue.py's real training path uses
    encode_positions() (sparse) instead.
    """
    n = len(fens)
    x_stm = np.zeros((n, NN_FEAT_IN), dtype=np.uint8)
    x_opp = np.zeros((n, NN_FEAT_IN), dtype=np.uint8)

    for i, fen in enumerate(fens):
        board = chess.Board(fen)
        stm_is_white = board.turn == chess.WHITE
        x_stm[i] = halfkp_features(board, pov_is_white=stm_is_white)
        x_opp[i] = halfkp_features(board, pov_is_white=not stm_is_white)

    return x_stm, x_opp


def _bag(indices_per_position: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Pack a list of variable-length int32 index arrays into the
    EmbeddingBag "offsets" protocol: a single flat concatenated index
    array plus an offsets array of length N where offsets[i] is the start
    of position i's slice (offsets[0] == 0 always).

    This mirrors torch.nn.EmbeddingBag's own expected input format
    exactly, so model.py can pass these arrays straight to
    F.embedding_bag without any further reshaping.
    """
    lengths = np.fromiter((len(a) for a in indices_per_position),
                           dtype=np.int64, count=len(indices_per_position))
    offsets = np.zeros(len(indices_per_position), dtype=np.int64)
    if len(lengths) > 0:
        offsets[1:] = np.cumsum(lengths)[:-1]
    if indices_per_position:
        flat = np.concatenate(indices_per_position).astype(np.int32, copy=False)
    else:
        flat = np.zeros(0, dtype=np.int32)
    return flat, offsets


def encode_positions(fens: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sparse batch encoder (the DEFAULT path).

    Returns (stm_indices, stm_offsets, opp_indices, opp_offsets):
      stm_indices : int32 (total_active_stm,)  flat concatenated feature ids
      stm_offsets : int64 (N,)                 start offset of position i
      opp_indices : int32 (total_active_opp,)
      opp_offsets : int64 (N,)

    Feed directly to torch.nn.functional.embedding_bag(indices, weight,
    offsets, mode='sum') — see model.py.
    """
    stm_lists: list[np.ndarray] = []
    opp_lists: list[np.ndarray] = []

    for fen in fens:
        board = chess.Board(fen)
        stm_is_white = board.turn == chess.WHITE
        stm_lists.append(halfkp_active_indices(board, pov_is_white=stm_is_white))
        opp_lists.append(halfkp_active_indices(board, pov_is_white=not stm_is_white))

    stm_idx, stm_off = _bag(stm_lists)
    opp_idx, opp_off = _bag(opp_lists)
    return stm_idx, stm_off, opp_idx, opp_off


def encode_mailbox_batch(boards: np.ndarray, stms: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized HalfKP-4Bucket sparse encoder straight from Zchezz's own
    packed mailbox board array — NO FEN string, NO python-chess, NO
    per-position Python loop over 64 squares.

    This exists to remove a wasteful round-trip in the `.bin` training
    path: `dataset.py`'s SAMPLE_DTYPE records already carry a structured
    64-byte mailbox `board[]` (see dataset.py's module docstring), and the
    old path built a FEN STRING from it (`records_to_fens_and_targets`)
    only to have `chess.Board(fen)` + `piece_map()` parse that string back
    into per-square data one square at a time. Array -> string -> array.
    This function goes straight from array to sparse feature indices.

    Inputs
    ------
    boards : (N, 64) uint8 array, Zchezz mailbox convention: zsq 0=a8 ..
             63=h1 (row-major, rank8 first), piece codes 0=empty,
             WP=9..WK=14 (COL_W=8 + type 1..6), BP=17..BK=22
             (COL_B=16 + type 1..6). Exactly `SAMPLE_DTYPE["board"]`.
    stms   : (N,) uint8 array, 0 = white to move, 1 = black to move.
             Exactly `SAMPLE_DTYPE["stm"]`.

    Returns (stm_indices, stm_offsets, opp_indices, opp_offsets), the SAME
    structure and EmbeddingBag "offsets" protocol as encode_positions() —
    order is always [stm, opp] (see module docstring: swapping this order
    flips the sign of every evaluation the net has ever seen).

    ── Coordinate derivation (must match nnue.h byte for byte) ───────────

    encoding.py's other functions work in python-chess sq_w coordinates
    (a1=0), where sq_w = zsq ^ 56 (see module docstring). Substituting that
    identity into halfkp_features/halfkp_active_indices's formulas and
    simplifying collapses to two clean per-perspective rules expressed
    directly in zsq (no intermediate FEN, no sq_w variable needed):

        white-POV square of a zsq  = zsq ^ 56   (== sq_w)
        black-POV square of a zsq  = zsq         (== sq_w ^ 56, i.e. sq_b)

        bucket_white = king_bucket(wk_zsq ^ 56)
        bucket_black = king_bucket(bk_zsq)

    which is exactly nnue.h's "Coordinate and feature conventions" table
    (and the asymmetric white-no-flip/black-flip king-bucket rule
    documented at the top of this file) — this function is a direct
    transcription of that table into batched numpy, not a re-derivation.

    Implementation notes
    ---------------------
    Fully vectorized over the batch: every per-square computation (piece
    color/type extraction, ally/enemy classification, feature-index
    arithmetic) is one elementwise numpy expression over the whole (N, 64)
    array. The only two loop-shaped operations are the O(N) `_pack` cumsum
    (building the offsets array — not a per-position loop over squares)
    and the O(1) `np.where` perspective swap. Matches `halfkp_active_indices`
    index-for-index (verified by tests/test_mailbox_encoder.py against the
    existing FEN+python-chess path over real self-play `.bin` records) —
    king-missing rows fail soft to zero active features for that
    perspective, exactly like `halfkp_features`/`halfkp_active_indices` do
    for a partial board.
    """
    boards = np.asarray(boards)
    stms = np.asarray(stms)
    if boards.ndim != 2 or boards.shape[1] != 64:
        raise ValueError(f"encode_mailbox_batch: boards must be (N, 64), got {boards.shape}")
    n = boards.shape[0]
    if stms.shape[0] != n:
        raise ValueError(
            f"encode_mailbox_batch: boards has {n} rows but stms has {stms.shape[0]}")

    codes = boards.astype(np.int32)
    color = codes & 24            # 8=white, 16=black, 0=empty
    ptype = codes & 7             # 1..6 (P,N,B,R,Q,K), 0=empty
    occupied = codes != 0
    is_king = ptype == 6

    zsq_idx = np.arange(64, dtype=np.int32)   # column j IS the zsq of that column
    sq_w = zsq_idx ^ 56                        # white-POV square per column

    def _perspective(color_bit: int, pov_is_white: bool):
        king_mask = is_king & (color == color_bit)
        has_king = king_mask.any(axis=1)
        # argmax finds the (only, for a legal position) True entry; rows
        # with no king at all get has_king=False and are zeroed via `valid`
        # below regardless of what argmax returns for them.
        king_zsq = np.argmax(king_mask, axis=1).astype(np.int32)
        king_pov = (king_zsq ^ 56) if pov_is_white else king_zsq
        bucket = (((king_pov & 7) >= 4).astype(np.int32)
                  | (((king_pov >> 3) >= 4).astype(np.int32) << 1))
        base = bucket * NN_FEAT_PER_BUCKET                       # (N,)

        is_ally = color == color_bit                              # (N, 64)
        color_offset = np.where(is_ally, 0, 5)
        rel_type = ptype - 1   # 0..4 for P..Q; garbage for king/empty cells,
                                # but those are excluded by `valid` below.

        pov_sq = sq_w if pov_is_white else zsq_idx                # (64,)
        feat = base[:, None] + (color_offset + rel_type) * 64 + pov_sq[None, :]

        valid = occupied & ~is_king & has_king[:, None]
        return feat.astype(np.int32), valid

    white_feat, white_valid = _perspective(8, True)
    black_feat, black_valid = _perspective(16, False)

    is_white_stm = (stms == 0)[:, None]   # (N, 1), broadcasts against (N, 64)
    stm_feat = np.where(is_white_stm, white_feat, black_feat)
    stm_valid = np.where(is_white_stm, white_valid, black_valid)
    opp_feat = np.where(is_white_stm, black_feat, white_feat)
    opp_valid = np.where(is_white_stm, black_valid, white_valid)

    def _pack(feat: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        counts = valid.sum(axis=1).astype(np.int64)
        offsets = np.zeros(n, dtype=np.int64)
        if n > 1:
            offsets[1:] = np.cumsum(counts)[:-1]
        # Boolean indexing of a 2D array visits rows in order (C/row-major),
        # so `flat` is already grouped per position in the same order as
        # `offsets` expects — this IS the EmbeddingBag "offsets" protocol,
        # no extra reshape/sort needed.
        flat = feat[valid].astype(np.int32)
        return flat, offsets

    stm_idx, stm_off = _pack(stm_feat, stm_valid)
    opp_idx, opp_off = _pack(opp_feat, opp_valid)
    return stm_idx, stm_off, opp_idx, opp_off


def encode_position_indices(board: "chess.Board") -> tuple[np.ndarray, np.ndarray]:
    """Single-position convenience wrapper: (stm_indices, opp_indices),
    each a variable-length int32 array. Used by check_parity.py."""
    stm_is_white = board.turn == chess.WHITE
    stm = halfkp_active_indices(board, pov_is_white=stm_is_white)
    opp = halfkp_active_indices(board, pov_is_white=not stm_is_white)
    return stm, opp


if __name__ == "__main__":
    # Tiny smoke test: start position, both perspectives.
    b = chess.Board()
    xs, xo = encode_position_indices(b)
    print(f"start position: {len(xs)} active STM features, {len(xo)} active OPP features")
    assert len(xs) == 30 and len(xo) == 30, "start position must have 30 non-king pieces"
    bw, bb = king_buckets_of(b)
    print(f"king buckets: white={bw} black={bb}")
    dstm, dopp = halfkp_features(b, True), halfkp_features(b, False)
    assert dstm.sum() == 30 and dopp.sum() == 30
    assert set(np.nonzero(dstm)[0].tolist()) == set(xs.tolist())
    assert set(np.nonzero(dopp)[0].tolist()) == set(xo.tolist())
    print("OK: dense/sparse encodings agree")
