"""Feature encoder for the v3.22 NNU3 network.

Input is always normalized to side-to-move perspective. For black-to-move
FENs python-chess mirror() makes the moving side White; this exactly matches
v3.22 nnue.c's acc_w/acc_b perspective mapping. The output has 799 floats:
768 piece-square HM features followed by the 31 handcrafted features used by
NNU3. Targets supplied in White-relative form must be flipped with the
returned ``is_black`` mask.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os

import chess
import numpy as np

INPUT_MAIN = 768
INPUT_EXTRA = 31
INPUT_DIM = 799
PIECE_MAP = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}
_MAXCNT = np.asarray([8, 2, 2, 2, 1, 1], dtype=np.float32)
_MATVAL = np.asarray([1, 3, 3, 5, 9, 0], dtype=np.float32)


def encode_chunk(fens: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Encode FENs into dense NNU3 inputs plus original black-to-move mask."""
    n = len(fens)
    out = np.zeros((n, INPUT_DIM), dtype=np.float32)
    is_black = np.zeros(n, dtype=np.bool_)

    for i, fen in enumerate(fens):
        board = chess.Board(fen)
        black = board.turn == chess.BLACK
        is_black[i] = black
        if black:
            board = board.mirror()

        pieces = board.piece_map()
        cnt_w = np.zeros(6, dtype=np.float32)
        cnt_b = np.zeros(6, dtype=np.float32)

        for sq, piece in pieces.items():
            pt = PIECE_MAP[piece.piece_type]
            color_offset = 0 if piece.color == chess.WHITE else 6
            out[i, (color_offset + pt) * 64 + sq] = 1.0
            if piece.color == chess.WHITE:
                cnt_w[pt] += 1.0
            else:
                cnt_b[pt] += 1.0

        extra = out[i, INPUT_MAIN:]
        extra[0:6] = cnt_w / _MAXCNT
        extra[6:12] = cnt_b / _MAXCNT
        extra[12] = float(np.dot(cnt_w + cnt_b, _MATVAL)) / 78.0
        extra[13] = 1.0

        wp = board.pieces(chess.PAWN, chess.WHITE)
        bp = board.pieces(chess.PAWN, chess.BLACK)

        for sq in wp:
            file_ = chess.square_file(sq)
            rank_ = chess.square_rank(sq)
            blocked = False
            for f2 in range(max(0, file_ - 1), min(8, file_ + 2)):
                for r2 in range(rank_ + 1, 8):
                    if chess.square(f2, r2) in bp:
                        blocked = True
                        break
                if blocked:
                    break
            if not blocked:
                extra[14 + file_] = 1.0

        for sq in bp:
            file_ = chess.square_file(sq)
            rank_ = chess.square_rank(sq)
            blocked = False
            for f2 in range(max(0, file_ - 1), min(8, file_ + 2)):
                for r2 in range(0, rank_):
                    if chess.square(f2, r2) in wp:
                        blocked = True
                        break
                if blocked:
                    break
            if not blocked:
                extra[22 + file_] = 1.0

        wk = board.king(chess.WHITE)
        bk = board.king(chess.BLACK)
        if wk is not None and bk is not None:
            extra[30] = max(
                abs(chess.square_file(wk) - chess.square_file(bk)),
                abs(chess.square_rank(wk) - chess.square_rank(bk)),
            ) / 7.0

    return out, is_black


def encode_positions(fens: list[str], workers: int | None = None
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Parallel wrapper with the same deterministic row order as input."""
    if not fens:
        return (
            np.zeros((0, INPUT_DIM), dtype=np.float32),
            np.zeros(0, dtype=np.bool_),
        )
    workers = workers or (os.cpu_count() or 1)
    workers = max(1, min(int(workers), len(fens)))
    if workers == 1 or len(fens) < 512:
        return encode_chunk(fens)

    size = (len(fens) + workers - 1) // workers
    chunks = [fens[i:i + size] for i in range(0, len(fens), size)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(encode_chunk, chunks))
    return (
        np.concatenate([x for x, _ in results], axis=0),
        np.concatenate([m for _, m in results], axis=0),
    )


def stm_targets(white_relative: np.ndarray, is_black: np.ndarray) -> np.ndarray:
    """Convert normalized White-relative 0..1 labels to STM-relative labels."""
    y = np.asarray(white_relative, dtype=np.float32).copy()
    y[is_black] = 1.0 - y[is_black]
    return y


def smoke_test() -> None:
    # Mirrored positions must encode identically after STM normalization.
    w = chess.Board()
    b = w.mirror()
    b.turn = chess.BLACK
    x, mask = encode_positions([w.fen(), b.fen()], workers=1)
    assert x.shape == (2, INPUT_DIM)
    assert mask.tolist() == [False, True]
    assert np.array_equal(x[0], x[1]), "STM normalization/mirroring drift"
    assert int(np.count_nonzero(x[0, :INPUT_MAIN])) == 32
    assert x[0, INPUT_MAIN + 13] == 1.0


if __name__ == "__main__":
    smoke_test()
    print("OK: NNU3 encoding smoke test passed")
