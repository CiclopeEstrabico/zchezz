"""
train/dataset.py — packed self-play `.bin` reader (Zchezz v4.00)

This module reads the binary sample format that the (not-yet-written)
native selfplay tool (`tools/selfplay.c`, F.2/F.3 of
zchezz_v400_implementation_plan.md) will emit. It is modeled directly on
zquoridor's `training/read_selfplay.py` (see zquoridor_recon.md, section
"Binary sample format & numpy reader"): a numpy structured dtype read via
`np.memmap`, with lazy multi-shard indexing so a dataset bigger than RAM
can be trained on without ever loading it whole.

── THE RECORD LAYOUT IS A CONTRACT ─────────────────────────────────────────

The dtype below is not just "how this file happens to read bytes" — it is
the contract the (future) C writer in tools/selfplay.c MUST match
byte-for-byte, in this exact field order, with this exact packing (no
padding beyond the explicit `_pad` field). If the C struct and this dtype
ever drift, every read silently reinterprets garbage as data instead of
raising an error — this is precisely the class of bug zquoridor's recon
flags about `NUM_FEATURES` drifting between train/quantize scripts (see
"Quantization & parity check" in zquoridor_recon.md): always validate
against the artifact's actual on-disk byte count, not just a duplicated
constant.

    uint8  board[64]     mailbox pieces, Zchezz encoding (0=empty, WP=9..BK=22),
                          sq 0=a8 .. 63=h1 (Zchezz's own mailbox convention,
                          NOT python-chess's; see encoding.py's docstring for
                          the zsq <-> python-chess-sq relationship: sq_w = zsq ^ 56)
    uint8  stm            0 = white, 1 = black
    uint8  rule50
    uint8  castling        bitmask (bit layout is the C engine's own castling
                            rights encoding — not reinterpreted here, only
                            round-tripped/ignored by the current training code)
    uint8  ep_file          0..7, 8 = none
    int16  eval_cp          STM-relative eval from the search that chose the move
    int8   game_result      +1/0/-1 from the point of view of the side to move
                             in THIS position (NOT the side that eventually won —
                             see wl_target() below for how this is combined with
                             eval_cp)
    uint16 move_played      packed move (Zchezz's own pack_move() format from
                             search.c — opaque to this module, kept for future
                             policy-head training, F.3 Estagio 2 of the plan)
    uint16 _pad             explicit padding to keep the record a round 76
                             bytes and everything after `board` naturally
                             aligned; always zero, ignored on read

Total record size: 64 + 1+1+1+1 + 2 + 1 + 2 + 2 = 75 bytes... wait, see
SAMPLE_DTYPE.itemsize below for the authoritative value (numpy applies its
own alignment unless `align=False`, we build the dtype with an explicit
field list and no implicit padding beyond `_pad`, but ALWAYS trust
`SAMPLE_DTYPE.itemsize`, not hand arithmetic, when writing the C struct —
this is exactly the mistake this docstring is warning against making).

── wl_target: the K-blend value-head training label ────────────────────────

Ported from zquoridor's `training/train_nnue.py::to_chunk_tensors` (see
zquoridor_recon.md, "wl_target blend (k)"), adapted to Zchezz's existing
cp<->probability convention (APPENDIX D.2 of the implementation plan,
which is also what `to_wdl()` in the legacy train/train_nnue.py already
does): `sigmoid(cp / 320.0)`.

    result_prob = (game_result + 1) / 2        # {-1,0,1} -> {0, 0.5, 1}
    ev_prob     = sigmoid(eval_cp / 320.0)      # the net's own recorded eval
    wl_target   = K * result_prob + (1 - K) * ev_prob

`K` is a PER-DATASET parameter (per shard-directory / per generation, not
a single global), exactly as in zquoridor: K=1.0 (the default) trains pure
TD(1) against the real game result and ignores eval_cp entirely; K<1.0
bootstraps from the engine's own recorded evaluation as well. See APPENDIX
F.3.0 of the implementation plan for the full rationale (why K should
start near 1.0 in early bootstrap generations and can be lowered later).

Unlike zquoridor's Quoridor engine (which has a dedicated WL head and
therefore stores `nnue_eval` already as a probability), Zchezz's engine
produces a centipawn-like score, so the cp->probability conversion step is
required here and is NOT a no-op (see zquoridor_recon.md's explicit note
that this constant "would need to be added" for a classical/NNUE-hybrid
engine like Zchezz's — this module is where it lives).

── Multi-shard, bigger-than-RAM streaming ───────────────────────────────────

`MultiShardSelfplay` keeps every shard file `np.memmap`-backed (zero-copy)
and only densifies (materializes an actual in-RAM numpy array) the exact
rows requested via `__getitem__`/`get_batch`, using `np.searchsorted` over
cumulative shard-length offsets to map a global index to (shard, local
index) — the same trick as zquoridor's `MultiFileSelfPlay`. This is what
lets train_nnue.py shuffle over a global index permutation without ever
loading more than one training chunk's worth of raw records into RAM at a
time.

════════════════════════════════════════════════════════════════════════
 LABEL CONVENTION (CLAUDE.md rule 10) — result / cp / wdl
════════════════════════════════════════════════════════════════════════

 | Name     | Meaning                                | Range / frame          |
 |----------|----------------------------------------|------------------------|
 | result   | the REAL game outcome                  | 0.0 / 0.5 / 1.0, WHITE |
 |          |                                        | 0 = Black won, 1 = White|
 | cp       | evaluation in centipawns               | int, WHITE-relative    |
 | wdl      | sigmoid(cp / 320) — a FUNCTION OF cp,  | 0..1, WHITE-relative   |
 |          | NOT an outcome                         |                        |
 | target   | lam*result + (1-lam)*wdl               | 0..1, computed at      |
 |          |                                        | TRAINING time only     |

 * All three are WHITE-relative, so ONE flip (x -> 1-x) converts the whole
   set to STM-relative downstream.
 * result and wdl share the SAME 0..1 scale because the target is a CONVEX
   combination. A result on a -1..1 scale would leave the sigmoid's range
   at lam=1 and break the BCE loss.
 * `wdl` is DERIVABLE from `cp`, so the two can rot apart. Whenever `cp`
   exists it wins and wdl is recomputed from it; a stored wdl is used only
   when cp is missing, and a disagreement is reported.
 * NEVER bake the blend into a dataset. lam is per-dataset, set in the
   DATASETS block at the top of train/train_nnue.py, and is precisely the
   knob you anneal across bootstrap generations.
 * 320 is the same constant as nnue.c's `_nnL3B * 320.0f` output scale.
   Changing one without the others silently rescales everything.
════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass

import numpy as np

# ── Provenance header — SEE MODULE DOCSTRING: THIS MIRRORS sample.h's
# SampleFileHeader BYTE-FOR-BYTE ──────────────────────────────────────────
#
# A .bin file written by the current tools/selfplay.c / tools/arena.c
# starts with this fixed-size header before any SAMPLE_DTYPE record: which
# engine build and which NNUE weight file produced every eval_cp that
# follows (see sample.h's "PROVENANCE HEADER" section for the full
# rationale — file-level, not per-row, because arena.c's --bin gate
# already enforces one evaluator per file).
#
# Format is detected by MAGIC, never guessed from file size: a file whose
# first 8 bytes equal SAMPLE_FILE_MAGIC is format v2 (has this header);
# anything else — including every .bin written before this header existed
# — is format v1 (legacy, headerless), and reads back with provenance
# "unknown" rather than being misinterpreted as v2 fields.
SAMPLE_FILE_MAGIC = b"ZCHZSMP2"

HEADER_DTYPE = np.dtype([
    ("magic", "S8"),
    ("header_size", "<u4"),        # bytes to skip before the first record —
                                    # ALWAYS trust this over EXPECTED header
                                    # size, so the header can grow later
                                    # (sample.h's `reserved` field) without
                                    # breaking this reader.
    ("engine_version", "<u4"),     # major*100+minor, e.g. 400 = "4.00"
    ("weight_fingerprint", "<u8"), # FNV-1a 64 over the raw NNUE weight bytes
    ("weight_path", "S128"),       # informational only, NUL-padded/truncated
    ("reserved", "u1", (32,)),
], align=False)

EXPECTED_HEADER_SIZE = 184
assert HEADER_DTYPE.itemsize == EXPECTED_HEADER_SIZE, (
    f"HEADER_DTYPE.itemsize={HEADER_DTYPE.itemsize} != {EXPECTED_HEADER_SIZE}; "
    "this drifted from sample.h's SampleFileHeader — update both in lockstep."
)


@dataclass
class Provenance:
    """Where a .bin shard's eval_cp column came from.

    `format_version` is 1 for legacy headerless files (every field below
    reads "unknown"/None) or 2 for files carrying a SampleFileHeader.
    """
    format_version: int
    engine_version: str            # e.g. "4.00", or "unknown"
    weight_fingerprint: int | None  # FNV-1a 64, or None if unknown
    weight_path: str               # "" if unknown

    @property
    def known(self) -> bool:
        return self.format_version >= 2


UNKNOWN_PROVENANCE = Provenance(
    format_version=1, engine_version="unknown", weight_fingerprint=None, weight_path="",
)


def read_bin_header(path: str) -> tuple[int, Provenance]:
    """Detects and reads a .bin file's provenance header, if any.

    Returns (record_start_offset, Provenance) — `record_start_offset` is
    how many bytes to skip before the first SAMPLE_DTYPE record starts
    (0 for legacy files, `header.header_size` for v2 files — NOT a
    hardcoded constant, so a future header grown via sample.h's
    `reserved` bytes is still skipped correctly by an older copy of this
    function, as long as it still trusts the field instead of the
    dtype's own itemsize).

    Detection is by MAGIC only, never by file size or guessing: a file
    too short to even hold a header, or one that starts with something
    other than SAMPLE_FILE_MAGIC, is treated as legacy (offset 0,
    unknown provenance) — exactly the "existing headerless .bin files
    must still load" backward-compatibility requirement.
    """
    size = os.path.getsize(path)
    if size < HEADER_DTYPE.itemsize:
        return 0, UNKNOWN_PROVENANCE

    with open(path, "rb") as f:
        raw = f.read(HEADER_DTYPE.itemsize)
    if raw[:8] != SAMPLE_FILE_MAGIC:
        return 0, UNKNOWN_PROVENANCE

    hdr = np.frombuffer(raw, dtype=HEADER_DTYPE, count=1)[0]
    header_size = int(hdr["header_size"])
    if header_size < HEADER_DTYPE.itemsize:
        # A corrupt/impossible header_size — safer to treat as unreadable
        # provenance than to risk skipping too few bytes and misreading
        # the first record's leading bytes as header fields.
        return 0, UNKNOWN_PROVENANCE

    engine_version_num = int(hdr["engine_version"])
    engine_version = f"{engine_version_num // 100}.{engine_version_num % 100:02d}"
    weight_path = bytes(hdr["weight_path"]).split(b"\x00", 1)[0].decode("utf-8", "replace")

    return header_size, Provenance(
        format_version=2,
        engine_version=engine_version,
        weight_fingerprint=int(hdr["weight_fingerprint"]),
        weight_path=weight_path,
    )

# ── Record dtype — SEE MODULE DOCSTRING: THIS IS A CROSS-LANGUAGE CONTRACT ──
SAMPLE_DTYPE = np.dtype([
    ("board", "u1", (64,)),   # mailbox pieces, Zchezz encoding, zsq 0=a8..63=h1
    ("stm", "u1"),             # 0=white, 1=black
    ("rule50", "u1"),
    ("castling", "u1"),        # bitmask, opaque to this module
    ("ep_file", "u1"),         # 0..7, 8=none
    ("eval_cp", "<i2"),        # STM-relative, from the search that chose the move
    ("game_result", "i1"),     # +1/0/-1, STM-of-this-position relative
    ("move_played", "<u2"),    # packed move (search.c pack_move format)
    ("_pad", "<u2"),           # explicit, always zero
], align=False)

# 64 + 1+1+1+1 + 2 + 1 + 2 + 2 = 75 -> numpy with align=False packs tightly,
# no hidden padding is inserted beyond the explicit `_pad` field.
EXPECTED_ITEMSIZE = 75
assert SAMPLE_DTYPE.itemsize == EXPECTED_ITEMSIZE, (
    f"SAMPLE_DTYPE.itemsize={SAMPLE_DTYPE.itemsize} != {EXPECTED_ITEMSIZE}; "
    "the on-disk record layout changed — update tools/selfplay.c's "
    "writer struct to match, or fix this dtype, before trusting any .bin file."
)

CP_TO_PROB_TEMPERATURE = 320.0   # cp -> probability scale; must match nnue.c's output scale


def wl_target(eval_cp: np.ndarray, game_result: np.ndarray, k: float) -> np.ndarray:
    """K-blend value-head training label (APPENDIX D.2 / F.3.0).

        wl_target = k * result_prob + (1 - k) * ev_prob
        result_prob = (game_result + 1) / 2
        ev_prob     = sigmoid(eval_cp / 320.0)

    `eval_cp`/`game_result` are already STM-relative for the position they
    describe (see SAMPLE_DTYPE docstring) — no perspective flip is needed
    here, unlike the legacy parquet path in train_nnue.py which has to flip
    a white-relative label for black-to-move FENs.

    `k` is a python float, broadcast over the whole batch — callers that
    need a per-source k (mixing several datasets with different k in one
    training batch) should call this once per source and concatenate the
    results, exactly as zquoridor's `k_by_source` array does.
    """
    eval_cp = eval_cp.astype(np.float32)
    game_result = game_result.astype(np.float32)

    result_prob = (game_result + 1.0) / 2.0
    ev_prob = 1.0 / (1.0 + np.exp(-eval_cp / CP_TO_PROB_TEMPERATURE))

    k = np.float32(k)
    return (k * result_prob + (1.0 - k) * ev_prob).astype(np.float32)


@dataclass
class _ShardInfo:
    path: str
    n_rows: int
    cum_start: int   # global index of this shard's first row
    provenance: Provenance


class MultiShardSelfplay:
    """Lazy, memmap-backed concatenation of many `.bin` shards.

    Usage:
        ds = MultiShardSelfplay.from_glob("C:/nnue_checkpoints/selfplay/gen5/*.bin")
        len(ds)                       # total records across all shards
        ds[1234]                      # single structured record (materializes 1 row)
        ds.get_batch(idx_array)       # structured array of len(idx_array) rows,
                                       # grouped per-shard internally for efficiency
    """

    def __init__(self, shard_paths: list[str]):
        if not shard_paths:
            raise ValueError("MultiShardSelfplay: no shard paths given")

        self._mmaps: list[np.memmap] = []
        self._shards: list[_ShardInfo] = []
        cum = 0
        for path in shard_paths:
            header_offset, provenance = read_bin_header(path)
            size = os.path.getsize(path) - header_offset
            if size % SAMPLE_DTYPE.itemsize != 0:
                raise ValueError(
                    f"{path}: record region size {size} (file size minus "
                    f"{header_offset}-byte header) is not a multiple of the "
                    f"record size {SAMPLE_DTYPE.itemsize} — file is truncated, "
                    f"corrupt, or was written with a different SAMPLE_DTYPE "
                    f"than this module's."
                )
            n_rows = size // SAMPLE_DTYPE.itemsize
            if n_rows == 0:
                continue
            mm = np.memmap(path, dtype=SAMPLE_DTYPE, mode="r", shape=(n_rows,), offset=header_offset)
            self._mmaps.append(mm)
            self._shards.append(_ShardInfo(path=path, n_rows=n_rows, cum_start=cum, provenance=provenance))
            cum += n_rows

        self._total = cum
        self._cum_starts = np.fromiter(
            (s.cum_start for s in self._shards), dtype=np.int64, count=len(self._shards)
        )

    @classmethod
    def from_glob(cls, pattern: str) -> "MultiShardSelfplay":
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No shards matched glob pattern: {pattern}")
        return cls(paths)

    def __len__(self) -> int:
        return self._total

    def _locate(self, global_idx: int) -> tuple[int, int]:
        """Map a global row index to (shard_index, local_row_index)."""
        shard_i = int(np.searchsorted(self._cum_starts, global_idx, side="right")) - 1
        local_i = global_idx - self._shards[shard_i].cum_start
        return shard_i, local_i

    def __getitem__(self, global_idx: int) -> np.void:
        shard_i, local_i = self._locate(global_idx)
        return self._mmaps[shard_i][local_i]

    def get_batch(self, global_indices: np.ndarray) -> np.ndarray:
        """Materialize an arbitrary (possibly unsorted, possibly spanning
        many shards) set of global indices into one dense structured
        array of shape (len(global_indices),).

        Groups requested indices per-shard so each shard's memmap is
        fancy-indexed once, not once per row — this is the same batching
        trick zquoridor's `MultiFileSelfPlay.__getitem__` uses.
        """
        global_indices = np.asarray(global_indices, dtype=np.int64)
        shard_of = np.searchsorted(self._cum_starts, global_indices, side="right") - 1

        out = np.empty(len(global_indices), dtype=SAMPLE_DTYPE)
        for shard_i in np.unique(shard_of):
            mask = shard_of == shard_i
            local_idx = global_indices[mask] - self._shards[int(shard_i)].cum_start
            out[mask] = self._mmaps[int(shard_i)][local_idx]

        return out

    def shard_summary(self) -> list[tuple[str, int]]:
        return [(s.path, s.n_rows) for s in self._shards]

    def provenance_summary(self) -> list[tuple[str, Provenance]]:
        """Per-shard (path, Provenance) — use to audit/filter a mixed-
        generation corpus, e.g. flag any shard whose weight_fingerprint
        doesn't match the network currently being trained, or whose
        provenance is "unknown" (format v1, pre-header)."""
        return [(s.path, s.provenance) for s in self._shards]


def records_to_fens_and_targets(records: np.ndarray, k: float) -> tuple[list[str], np.ndarray]:
    """Convert a batch of raw SAMPLE_DTYPE records into (fens, wl_targets).

    This is the glue train_nnue.py's `.bin`-backed data path uses: encoding.py
    only knows how to encode a `chess.Board`/FEN, not Zchezz's raw mailbox
    board array directly, so records are converted to a minimal FEN string
    here (piece placement + side to move only — castling/ep are NOT
    round-tripped into the FEN because encoding.py's HalfKP features never
    look at them; only piece placement, king squares, and side-to-move
    matter for the feature encoding itself).

    Zchezz mailbox convention: zsq 0=a8 .. 63=h1, piece codes WP=9..BK=22
    (COL_W=8 + type 1..6, COL_B=16 + type 1..6; see CLAUDE.md "Piece
    encoding"). python-chess FEN ranks run 8..1 top to bottom, files a..h
    left to right, which is precisely Zchezz's zsq row-major order — no
    coordinate flip is needed to build the placement string, only the
    piece-code -> FEN-letter mapping below.
    """
    # Zchezz piece code -> (python-chess-style) FEN letter.
    _TYPE_LETTER = {1: "p", 2: "n", 3: "b", 4: "r", 5: "q", 6: "k"}

    fens: list[str] = []
    for rec in records:
        board = rec["board"]
        rows = []
        for rank_start in range(0, 64, 8):
            row = board[rank_start:rank_start + 8]
            fen_row = []
            empty_run = 0
            for code in row:
                code = int(code)
                if code == 0:
                    empty_run += 1
                    continue
                if empty_run:
                    fen_row.append(str(empty_run))
                    empty_run = 0
                color_bit = code & 24     # 8=white, 16=black
                ptype = code & 7
                letter = _TYPE_LETTER.get(ptype, "?")
                fen_row.append(letter.upper() if color_bit == 8 else letter)
            if empty_run:
                fen_row.append(str(empty_run))
            rows.append("".join(fen_row))
        placement = "/".join(rows)   # zsq row-major == FEN rank8..rank1 order
        stm_char = "w" if int(rec["stm"]) == 0 else "b"
        # Castling/ep/halfmove/fullmove are not needed by encoding.py's
        # HalfKP features; emit FEN-legal placeholders.
        fens.append(f"{placement} {stm_char} - - 0 1")

    targets = wl_target(records["eval_cp"], records["game_result"], k)
    return fens, targets


if __name__ == "__main__":
    print(f"SAMPLE_DTYPE itemsize = {SAMPLE_DTYPE.itemsize} bytes")
    print(SAMPLE_DTYPE)

    # Smoke test: fabricate one in-memory record (start position, white to
    # move) and check wl_target()/records_to_fens_and_targets() run.
    board64 = np.zeros(64, dtype=np.uint8)
    # zsq 0=a8 .. 63=h1: rank 8 is row 0, rank 1 is row 7 (last row).
    back_rank = [4, 2, 3, 5, 6, 3, 2, 4]   # R N B Q K B N R (type codes)
    for f in range(8):
        board64[56 + f] = 8 | back_rank[f]      # white back rank (rank 1 = row 7)
        board64[48 + f] = 8 | 1                  # white pawns (rank 2 = row 6)
        board64[8 + f] = 16 | 1                  # black pawns (rank 7 = row 1)
        board64[f] = 16 | back_rank[f]           # black back rank (rank 8 = row 0)

    rec = np.zeros(1, dtype=SAMPLE_DTYPE)
    rec[0]["board"] = board64
    rec[0]["stm"] = 0
    rec[0]["eval_cp"] = 25
    rec[0]["game_result"] = 1

    fens, targets = records_to_fens_and_targets(rec, k=1.0)
    print("fen:", fens[0])
    print("wl_target (k=1.0, should == result_prob == 1.0):", targets[0])
    assert fens[0].startswith("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w")
    assert abs(targets[0] - 1.0) < 1e-6

    fens2, targets2 = records_to_fens_and_targets(rec, k=0.0)
    print("wl_target (k=0.0, should == sigmoid(25/320)):", targets2[0])
    import math
    assert abs(targets2[0] - 1.0 / (1.0 + math.exp(-25 / 320.0))) < 1e-6
    print("OK: dataset smoke test passed")
