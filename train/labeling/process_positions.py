#!/usr/bin/env python3
"""
train/labeling/process_positions.py — the ONE position pipe
============================================================

Read chess positions from ANY supported source, optionally filter them,
optionally (re)evaluate them with Stockfish, and write them out in ANY
combination of supported formats.

    ANY INPUT            [ optional filters ]        ANY OUTPUT(S)
    ─────────            ────────────────────        ─────────────
    .epd                 quiet / endgame /           .parquet
    .pgn   (games)  ──►  score-cap / dedup    ──►    .bin  (SAMPLE_DTYPE)
    .bin   (SAMPLE)      (each one optional,         .epd
    .parquet             ALL of them optional)       .pgn  (FEN stubs)

Nothing here is mandatory except "read something, write something".
With `--filters none` this script is a pure FORMAT CONVERTER:

    python train/labeling/process_positions.py --in games.pgn --out out.bin --filters none
    python train/labeling/process_positions.py --in a.epd     --out a.parquet --filters none
    python train/labeling/process_positions.py --in data/gen7 --out gen7.epd  --filters none

and with filters on it is the dataset builder it always was:

    python train/labeling/process_positions.py \\
        --in data/selfplay_raw_res_data20260401 \\
        --out data/selfplay_cp_zchezz_res_filter_data20260401 \\
        --filters quiet --out-format parquet

Run `--help` for the full flag list, or `--dry-run` to see exactly what a
run WOULD read, filter and write without touching the disk.

────────────────────────────────────────────────────────────────────────
 INPUT FORMATS  (auto-detected from the file extension; --in-format forces)
────────────────────────────────────────────────────────────────────────

  epd      one position per line: `<fen-4-fields> <opcodes>`.
           Recognised opcodes, project convention (identical in
           tests/run_selfplay.py, tests/run_tournament.py and
           engine/c/tools/selfplay.c):
               c0 "<pgn result>"   e.g. "1-0" / "0-1" / "1/2-1/2"
               c1 "<cp>"           WHITE-relative centipawns
               c2 "<mover>"        free text (engine label) — ignored here
               c3 "<ply>"          free text (ply index)    — ignored here
           `ce <cp>` (standard EPD centipawn opcode) is also accepted and
           WINS over c1; `result "<...>"` is accepted as a c0 alias.
           A non-numeric c1 (legacy files wrote c1 "Checkmate") is simply
           treated as "no cp", not as a parse error.

  pgn      real games. Every position of every game is emitted, with
           `result` taken from the [Result] header and `cp` from the move
           comment when one is present. Both comment dialects are read:
               lichess/annotator   { [%eval 1.23] }  /  { [%eval #-3] }
               cutechess-cli       { +0.35/12 0.10s } / { -M3/20 }
           Openings can be skipped with --pgn-skip-plies (default
           PGN_SKIP_PLIES) and games truncated with --pgn-max-plies.

  bin      the packed record shared with C (engine/c/tools/sample.h,
           dtype from train/dataset.py). `eval_cp` and `game_result` are
           STM-relative in the record and are converted to WHITE-relative
           on read (CLAUDE.md rule 10).

  parquet  columns `fen` (required) and any of `cp`, `result`, `wdl`,
           `id`. A stored `wdl` is only used when `cp` is absent — it is
           sigmoid(cp/320), not an outcome.

────────────────────────────────────────────────────────────────────────
 OUTPUT FORMATS  (one or more, simultaneously — never either/or)
────────────────────────────────────────────────────────────────────────

  parquet  columns fen, cp, result (+ id if --emit-id). `wdl` is NOT
           written: it is derivable from cp and a stored copy can rot
           apart from it (CLAUDE.md rule 10).
  bin      SAMPLE_DTYPE records, directly loadable by train/dataset.py
           and by the C tools.
  epd      `<fen> c0 "<result>"; c1 "<cp>";` — the project convention
           above, so anything that reads our EPD reads this too.
  pgn      one single-position "game" per row: [SetUp "1"][FEN ...] with
           the eval as a comment and the result as the game result. This
           is a VIEWING format (paste into a GUI) — a filtered position
           set is not a game, so no moves are written.

 The output destination decides HOW they are written:
   * `--out DIR`   → chunked: DIR/chunk_0000.parquet, chunk_0001.parquet…
                    resumable (progress.json in DIR), one chunk per
                    --chunk-size rows, in every requested format.
   * `--out FILE`  → a single streamed file. The format is inferred from
                    FILE's extension unless --out-format says otherwise.
                    Truncated at start unless --append.

────────────────────────────────────────────────────────────────────────
 HOW MANY POSITIONS: --limit and --sample
────────────────────────────────────────────────────────────────────────

 How MANY records to read, and WHICH ones, are two separate settings:

   --limit 0        read everything (0 = no limit; 'all'/'none' also
                    accepted for backwards compatibility)
   --limit 500000   stop after 500,000 INPUT records (before filtering —
                    see --limit-stage to count OUTPUT rows instead)

   --sample stream     take them in file order — fast, streaming, O(1)
                       memory, resumable. THE DEFAULT.
   --sample reservoir  take a UNIFORM RANDOM sample of --limit records
                       from the whole input. Requires reading every input
                       record first and holding --limit of them in RAM,
                       and cannot be resumed. Use it when the input is
                       ordered (e.g. one game after another) and a prefix
                       would be biased.

────────────────────────────────────────────────────────────────────────
 FILTERS — all optional, individually switchable
────────────────────────────────────────────────────────────────────────

   --filters none            no filtering at all (pure conversion)
   --filters quiet           the tactical-noise filters (default preset)
   --filters endgame         only the endgame selector
   --filters quiet+endgame   both
   --filters config          whatever the FILTER_* constants below say

 Any individual filter can then be forced on or off and WINS over the
 preset:  --filter-in-check / --no-filter-in-check, --no-filter-duplicates,
 --score-cap 2000, --endgame-max-pieces 10, …

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

# Windows consoles default to cp1252 and die with UnicodeEncodeError on the
# box characters used above and in the reports below (same guard as
# tests/bench_nps.py and tests/run_selfplay_native.py).
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

import argparse
import gc
import glob
import json
import math
import os
import pathlib
import random
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Dict, Iterator, List, Optional, Tuple

import chess
import chess.pgn
import numpy as np
import pandas as pd

# train/dataset.py is the AUTHORITATIVE definition of SAMPLE_DTYPE (the
# cross-language .bin record contract shared with engine/c/tools/sample.h).
# Import it rather than hand-rolling a second copy here — a hand-rolled
# duplicate can silently drift from dataset.py's dtype and reinterpret
# every .bin read as garbage (see dataset.py's own docstring warning).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from dataset import SAMPLE_DTYPE  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  (CLAUDE.md rule 8)
#
#  Every option below is settable from the command line, and every CLI
#  flag's `default=` IS the constant here — never a second copy of the
#  literal. Edit these to change what a bare `python process_positions.py`
#  does; use the flags for one-off runs.
# ═══════════════════════════════════════════════════════════════════════════

# ── What to read / where to write ──────────────────────────────────────────
# INPUTS and OUTPUTS are parallel lists: INPUTS[i] is processed into
# OUTPUTS[i]. Each INPUT is a FILE or a DIRECTORY (walked recursively) or a
# glob pattern. Each OUTPUT is a DIRECTORY (chunked, resumable) or a FILE
# (single streamed file, format inferred from its extension).
# On the CLI: repeat --in / --out to give more than one pair.
INPUTS = [
    r'data\selfplay_raw_res_data20260401',
    r'data\selfplay_raw_res_data20260404',
    r'data\selfplay_raw_res_data20260410',
]
OUTPUTS = [
    r'data\selfplay_cp_zchezz_res_filter_data20260401',
    r'data\selfplay_cp_zchezz_res_filter_data20260404',
    r'data\selfplay_cp_zchezz_res_filter_data20260410',
]

IN_FORMAT      = 'auto'        # 'auto' (by extension) | 'epd' | 'pgn' | 'bin' | 'parquet'
OUT_FORMATS    = ['parquet']   # any subset of ['parquet','bin','epd','pgn'], written together.
                               # Ignored when --out is a FILE with a known extension.
APPEND         = False         # single-file output: True = append, False = truncate at start
RESUME         = True          # directory output: continue from progress.json / existing chunks
EMIT_ID        = False         # include the source `id` column in parquet/epd output
DRY_RUN        = False         # read + filter + report, write nothing

# ── How much to read ───────────────────────────────────────────────────────
LIMIT       = 0            # 0 = no limit. >0 = stop after this many records ('all'/'none' also accepted)
LIMIT_STAGE = 'input'      # 'input'  = --limit counts records READ (cheap, predictable runtime)
                           # 'output' = --limit counts rows KEPT after filtering (predictable dataset size)
SAMPLE      = 'stream'     # 'stream' = file order, streaming, resumable
                           # 'reservoir' = uniform random --limit sample (reads all input, holds LIMIT in RAM)
SEED        = 12345        # RNG seed for 'reservoir' sampling and file shuffling

# ── PGN reading ────────────────────────────────────────────────────────────
PGN_SKIP_PLIES = 0         # drop the first N plies of every game (book/opening phase)
PGN_MAX_PLIES  = 0         # 0 = whole game; >0 = stop after N plies
PGN_MIN_PLIES  = 0         # skip games shorter than this many plies (0 = keep all)

# ── Filters ────────────────────────────────────────────────────────────────
# FILTER_PRESET is the wholesale switch; the individual FILTER_* constants
# below are what 'config' means. Resolution order (last wins):
#     FILTER_* constants  →  --filters <preset>  →  explicit --filter-x flag
FILTER_PRESET        = 'config'  # 'none' | 'quiet' | 'endgame' | 'quiet+endgame' | 'config'

FILTER_DUPLICATES    = True   # drop a FEN (first 4 fields) already seen in this run
FILTER_IN_CHECK      = True   # drop positions where the side to move is in check
FILTER_TERMINAL      = True   # drop checkmate / stalemate / no-legal-move positions
FILTER_WIN_CAPTURE   = True   # drop if a capture wins material (victim > attacker + tolerance)
FILTER_EQUAL_CAPTURE = True   # drop if an even capture exists (|victim-attacker| <= tolerance; pxp exempt)
FILTER_SACRIFICE     = False  # also check the OPPONENT's captures (position is tactically hot either way)
FILTER_SCORE_CAP     = True   # drop positions whose |cp| exceeds SCORE_CAP_VALUE
SCORE_CAP_VALUE      = 3000   # centipawns; used only when FILTER_SCORE_CAP is on

FILTER_ENDGAME     = False    # keep ONLY positions matching the endgame criteria below
ENDGAME_MAX_PIECES = 14       # max pieces on the board, kings included
ENDGAME_NO_QUEENS  = True     # reject positions that still have a queen

REQUIRE_CP     = False        # drop rows without a cp value
REQUIRE_RESULT = False        # drop rows without a game result
# A row with NEITHER cp nor result carries no training signal at all and is
# always dropped (counted as `missing_cp_and_result`).

# ── (not a filter) WDL blending ────────────────────────────────────────────
# This script NEVER combines cp and result into one target. It emits the two
# raw columns; the blend
#     target = lambda*result + (1-lambda)*sigmoid(cp/320)
# happens at TRAINING time with a per-dataset lambda (DATASETS block at the
# top of train/train_nnue.py). Baking it here would freeze lambda at
# generation time — exactly the knob you want to anneal between generations.

# ── Stockfish (re)labelling — optional ─────────────────────────────────────
USE_STOCKFISH = False     # True = evaluate every surviving position with SF, overwriting `cp`
SF_PATH       = r"engine\stockfish_fast\stockfish\stockfish-windows-x86-64-avxvnni.exe"
SF_NODES      = 1_000_000 # nodes per position
SF_HASH_MB    = 16        # hash per SF process (one process per worker)
SF_TIMEOUT_S  = 60        # per-position wall-clock limit before the SF process is restarted

# ── Parallelism and buffer sizes ───────────────────────────────────────────
WORKERS          = max(1, (os.cpu_count() or 2) - 1)  # filter/eval worker processes
BATCH_SIZE         = 5_000        # records per task handed to a worker
MAX_ACTIVE_BATCHES = WORKERS * 2  # in-flight tasks (back-pressure on the reader)
CHUNK_SIZE_SAVE    = 10_000       # rows per output chunk in directory mode

# ── Fixed constants (not CLI-settable) ─────────────────────────────────────
PIECE_VALUES = {              # centipawn values used by the capture filters
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   20000,
}
EQUAL_CAPTURE_TOLERANCE = 50   # cp window that counts as an "equal" capture
EPD_SAMPLE_LINES        = 500  # lines sampled to estimate an .epd file's record count
CP_TO_WDL_T             = 320.0  # same constant as nnue.c / train_nnue.py (rule 10)
# ═══════════════════════════════════════════════════════════════════════════

IN_FORMATS  = ('auto', 'epd', 'pgn', 'bin', 'parquet')
OUT_FORMAT_CHOICES = ('parquet', 'bin', 'epd', 'pgn')
EXT_TO_FORMAT = {
    '.epd': 'epd', '.pgn': 'pgn', '.bin': 'bin',
    '.parquet': 'parquet', '.pq': 'parquet',
}

# ═══════════════════════════════════════════════════════════════════════════
#  FORMATO .BIN  (SAMPLE_DTYPE importado de train/dataset.py — ver import
#  no topo do arquivo. NÃO redefinir o dtype aqui.)
# ═══════════════════════════════════════════════════════════════════════════
# Zchezz mailbox: zsq 0=a8..63=h1 → sq_w = zsq ^ 56
# Peças: WP=9, WN=10, WB=11, WR=12, WQ=13, WK=14
#        BP=17, BN=18, BB=19, BR=20, BQ=21, BK=22
#        COL_W=8, COL_B=16, tipo 1..6 (P,N,B,R,Q,K)

_CHESS_TYPE_TO_ZCHEZZ = {
    chess.PAWN:   1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK:   4,
    chess.QUEEN:  5,
    chess.KING:   6,
}

# Castling bitmask, engine/c/zchezz_v400/board.h (CA_WK..CA_BQ) — the same
# byte selfplay.c writes into the record's `castling` field.
CA_WK, CA_WQ, CA_BK, CA_BQ = 1, 2, 4, 8

_CODE_TO_PIECE = {}
for _pt, _code in _CHESS_TYPE_TO_ZCHEZZ.items():
    _CODE_TO_PIECE[8  | _code] = chess.Piece(_pt, chess.WHITE)
    _CODE_TO_PIECE[16 | _code] = chess.Piece(_pt, chess.BLACK)


def board_to_bin_record(board: chess.Board, cp_white: int,
                        result_white: Optional[float]) -> np.ndarray:
    """Converte chess.Board + avaliação white-POV para um registro SAMPLE_DTYPE.

    `result_white` is the WHITE-relative game-outcome PROBABILITY on the
    0.0/0.5/1.0 scale (CLAUDE.md rule 10 — the same convention as the
    `result` column this script emits everywhere else), NOT a PGN string
    ('1-0'/'0-1'/'1/2-1/2'). This is the internal record's canonical
    `result` representation (see `normalize_result_to_prob`), so callers
    must not pass a PGN string here.
    """
    rec = np.zeros(1, dtype=SAMPLE_DTYPE)
    mailbox = rec[0]["board"]

    for sq, piece in board.piece_map().items():
        zsq  = sq ^ 56                      # python-chess sq_w → zsq (a8=0)
        col  = 8 if piece.color == chess.WHITE else 16
        code = col | _CHESS_TYPE_TO_ZCHEZZ[piece.piece_type]
        mailbox[zsq] = code

    rec[0]["stm"] = 0 if board.turn == chess.WHITE else 1
    rec[0]["rule50"] = min(board.halfmove_clock, 255)
    # castling is the ENGINE's bitmask: CA_WK=1, CA_WQ=2, CA_BK=4, CA_BQ=8
    # (board.h, same byte selfplay.c writes).
    rec[0]["castling"] = ((CA_WK if board.has_kingside_castling_rights(chess.WHITE)  else 0) |
                          (CA_WQ if board.has_queenside_castling_rights(chess.WHITE) else 0) |
                          (CA_BK if board.has_kingside_castling_rights(chess.BLACK)  else 0) |
                          (CA_BQ if board.has_queenside_castling_rights(chess.BLACK) else 0))
    rec[0]["ep_file"]  = chess.square_file(board.ep_square) if board.ep_square is not None else 8

    # eval_cp em STM-relative (positivo = bom para quem move)
    stm_factor = 1 if board.turn == chess.WHITE else -1
    cp_stm = int(cp_white) * stm_factor
    rec[0]["eval_cp"] = np.clip(cp_stm, -32000, 32000)

    # game_result STM-relative, derived from the WHITE-relative probability.
    # result_white: 1.0 = White won, 0.0 = Black won, 0.5 = draw, None = unknown.
    if result_white is None:
        gr = 0
    elif result_white >= 0.99:
        gr = 1 if board.turn == chess.WHITE else -1
    elif result_white <= 0.01:
        gr = -1 if board.turn == chess.WHITE else 1
    else:
        gr = 0
    rec[0]["game_result"] = gr
    rec[0]["move_played"] = 0
    rec[0]["_pad"] = 0
    return rec


def bin_record_to_board(rec) -> chess.Board:
    """Converte um registro SAMPLE_DTYPE de volta para um chess.Board."""
    board = chess.Board(fen=None)
    mailbox = rec["board"]
    for zsq in range(64):
        code = int(mailbox[zsq])
        if code == 0:
            continue
        piece = _CODE_TO_PIECE.get(code)
        if piece:
            board.set_piece_at(zsq ^ 56, piece)

    board.turn = chess.WHITE if rec["stm"] == 0 else chess.BLACK

    ca = int(rec["castling"])
    rights = chess.BB_EMPTY
    if ca & CA_WK: rights |= chess.BB_H1
    if ca & CA_WQ: rights |= chess.BB_A1
    if ca & CA_BK: rights |= chess.BB_H8
    if ca & CA_BQ: rights |= chess.BB_A8
    # clean_castling_rights() drops any right whose king/rook is not actually
    # on its home square, so a record with a stale bit cannot produce an
    # illegal FEN here.
    board.castling_rights = rights
    board.castling_rights = board.clean_castling_rights()

    # ep_file 0..7 -> the ep TARGET square: rank 6 when White is to move
    # (Black just double-pushed), rank 3 when Black is to move.
    ep_f = int(rec["ep_file"])
    if 0 <= ep_f <= 7:
        ep_rank = 5 if board.turn == chess.WHITE else 2
        cand = chess.square(ep_f, ep_rank)
        # Only keep it if a real capturable pawn is behind it, otherwise the
        # FEN would advertise an en-passant that cannot exist.
        victim = chess.square(ep_f, 4 if board.turn == chess.WHITE else 3)
        pc = board.piece_at(victim)
        board.ep_square = cand if (pc and pc.piece_type == chess.PAWN
                                   and pc.color != board.turn) else None
    else:
        board.ep_square = None

    board.halfmove_clock = int(rec["rule50"])
    board.fullmove_number = 1
    return board


def bin_record_to_fen(rec) -> str:
    """Converte um registro SAMPLE_DTYPE de volta para FEN."""
    return bin_record_to_board(rec).fen()


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS GERAIS
# ═══════════════════════════════════════════════════════════════════════════

def fen_key(fen: str) -> str:
    return ' '.join(fen.split()[:4])


def format_time(s: float) -> str:
    if s < 60:   return f"{s:.0f}s"
    if s < 3600: return f"{s/60:.1f}m"
    return f"{s/3600:.1f}h"


_RESULT_TO_PROB_OUT = {'1-0': 1.0, '1/2-1/2': 0.5, '0-1': 0.0}
_PROB_OUT_TO_RESULT = {1.0: '1-0', 0.5: '1/2-1/2', 0.0: '0-1'}


def normalize_result_to_prob(raw) -> Optional[float]:
    """Normalize a source's raw `result` field to the canonical WHITE-
    relative probability (0.0/0.5/1.0/None) used everywhere downstream in
    this script.

    Sources disagree on what `result` looks like on the wire:
      - epd / pgn / bin parsers hand back a PGN string ('1-0'/'0-1'/'1/2-1/2').
      - parquet already stores the numeric 0.0/0.5/1.0 WHITE-relative
        probability directly.
    Without this normalization, a numeric parquet `result` silently fails
    to match `_RESULT_TO_PROB_OUT`'s string keys and is coerced to None —
    i.e. every parquet-sourced row quietly loses its game outcome. Doing
    the conversion in exactly one place means every input format, present
    and future, is handled the same way from here on.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        return _RESULT_TO_PROB_OUT.get(raw.strip(), None)
    try:
        f = float(raw)
    except (TypeError, ValueError):
        return None
    if f != f:                    # NaN (an absent parquet value)
        return None
    if f >= 0.99:  return 1.0
    if f <= 0.01:  return 0.0
    if abs(f - 0.5) < 0.01: return 0.5
    return None


def prob_to_result_str(result_white: Optional[float]) -> str:
    """Inverse of _RESULT_TO_PROB_OUT: WHITE-relative 0.0/0.5/1.0 -> PGN string.
    Used only for the text output formats (epd/pgn) that want the human-
    readable PGN result; `result` stays numeric everywhere else."""
    if result_white is None:
        return ''
    return _PROB_OUT_TO_RESULT.get(round(float(result_white), 1), '')


def cp_to_wdl(cp_white: float) -> float:
    """sigmoid(cp/320) — the ONE definition of `wdl` (CLAUDE.md rule 10).
    Not written to any dataset; used only for reporting."""
    return 1.0 / (1.0 + math.exp(-float(cp_white) / CP_TO_WDL_T))


def parse_limit(value) -> int:
    """`--limit` accepts an int, or 'all'/'none'/'' meaning no limit (0).

    The old config used the string 'all' as a magic value that ALSO changed
    the sampling strategy; sampling is now --sample and this returns a plain
    integer, 0 == unlimited, so every code path downstream compares numbers.
    """
    if value is None:
        return 0
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ('', 'all', 'none', 'inf', '0'):
            return 0
        try:
            value = int(v.replace('_', '').replace(',', ''))
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"--limit: expected a non-negative integer or 'all', got {value!r}")
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("--limit must be >= 0 (0 = no limit)")
    return value


def _progress_path(out_dir: str) -> str:
    return os.path.join(out_dir, 'progress.json')


def load_progress(out_dir: str) -> Dict[str, int]:
    p = _progress_path(out_dir)
    if os.path.exists(p):
        try:
            with open(p, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_progress(out_dir: str, consumed: Dict[str, int]):
    p   = _progress_path(out_dir)
    tmp = p + '.tmp'
    try:
        with open(tmp, 'w') as f:
            json.dump(consumed, f, indent=2)
        os.replace(tmp, p)
    except Exception as e:
        print(f"  [Warning] Could not save progress: {e}")


# ═══════════════════════════════════════════════════════════════════════════
#  FILTROS
#
#  The worker process re-sets these module globals from the primitives it is
#  handed (see _worker_pipeline) — a child process gets a fresh import, not
#  the parent's edited state, so the settings must travel as arguments.
# ═══════════════════════════════════════════════════════════════════════════

def is_quiet_advanced(board: chess.Board) -> Tuple[bool, str]:
    """Filtro quiet configurável via variáveis globais."""
    if FILTER_TERMINAL:
        if board.is_checkmate() or board.is_stalemate():
            return False, "terminal"
        if not any(board.legal_moves):
            return False, "no_moves"

    if FILTER_IN_CHECK and board.is_check():
        return False, "in_check"

    if FILTER_WIN_CAPTURE or FILTER_EQUAL_CAPTURE:
        for move in board.legal_moves:
            if not board.is_capture(move):
                continue
            attacker = board.piece_at(move.from_square)
            if attacker is None: continue
            att_val = PIECE_VALUES.get(attacker.piece_type, 0)
            if board.is_en_passant(move):
                vic_type, vic_val = chess.PAWN, 100
            else:
                victim = board.piece_at(move.to_square)
                if victim is None: continue
                vic_type = victim.piece_type
                vic_val  = PIECE_VALUES.get(vic_type, 0)

            # pxp: não filtra como equal capture (movimento posicional normal)
            if attacker.piece_type == chess.PAWN and vic_type == chess.PAWN:
                continue

            diff = vic_val - att_val
            if FILTER_WIN_CAPTURE   and diff >  EQUAL_CAPTURE_TOLERANCE: return False, "winning_capture"
            if FILTER_EQUAL_CAPTURE and abs(diff) <= EQUAL_CAPTURE_TOLERANCE: return False, "equal_capture"

    if FILTER_SACRIFICE:
        # Checar se o adversário tem capturas disponíveis após nossa jogada
        saved_turn = board.turn
        board.turn = not board.turn
        for move in board.generate_pseudo_legal_captures():
            attacker = board.piece_at(move.from_square)
            if attacker is None: continue
            att_val = PIECE_VALUES.get(attacker.piece_type, 0)
            if board.is_en_passant(move):
                vic_val = 100
            else:
                victim = board.piece_at(move.to_square)
                if victim is None: continue
                if victim.piece_type == chess.KING: continue
                vic_val = PIECE_VALUES.get(victim.piece_type, 0)
            if att_val == 100 and vic_val == 100: continue
            diff = vic_val - att_val
            if (FILTER_WIN_CAPTURE   and diff >  EQUAL_CAPTURE_TOLERANCE) or \
               (FILTER_EQUAL_CAPTURE and abs(diff) <= EQUAL_CAPTURE_TOLERANCE):
                board.turn = saved_turn
                return False, "sacrifice_other_side"
        board.turn = saved_turn

    return True, ""


def is_endgame_position(board: chess.Board) -> Tuple[bool, str]:
    if ENDGAME_NO_QUEENS and board.queens:
        return False, "endgame_has_queen"
    if ENDGAME_MAX_PIECES is not None:
        if chess.popcount(board.occupied) > ENDGAME_MAX_PIECES:
            return False, "endgame_too_many_pieces"
    return True, ""


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRADA: descoberta de arquivos + parsers
#
#  Every parser yields the SAME record dict:
#      {'fen': str, 'cp': float|None, 'result': <raw>, 'id': str}
#  `result` is whatever the source had (PGN string or number); it is
#  normalised exactly once, in the worker, by normalize_result_to_prob().
# ═══════════════════════════════════════════════════════════════════════════

def detect_format(path: str) -> Optional[str]:
    """Input format from the file extension, or None if unknown."""
    return EXT_TO_FORMAT.get(os.path.splitext(path)[1].lower())


def collect_input_files(spec: str, in_format: str) -> List[Tuple[str, str]]:
    """Expand one --in entry into a sorted list of (path, format).

    `spec` may be a single FILE, a DIRECTORY (walked recursively) or a glob
    pattern. With --in-format auto every known extension found is taken and
    each file is read with the parser its extension implies, so a directory
    holding both .epd and .pgn is processed in one run. With an explicit
    --in-format only files of that format are taken (a directory is scanned
    for `*.<fmt>`), which is also how you read a file whose extension lies.
    """
    out: List[Tuple[str, str]] = []
    candidates: List[str] = []

    if os.path.isdir(spec):
        exts = EXT_TO_FORMAT.keys() if in_format == 'auto' else \
               [e for e, f in EXT_TO_FORMAT.items() if f == in_format]
        p = pathlib.Path(spec)
        for ext in exts:
            candidates += [str(f) for f in p.rglob('*' + ext)]
    elif any(ch in spec for ch in '*?['):
        candidates = glob.glob(spec, recursive=True)
    elif os.path.isfile(spec):
        candidates = [spec]
    else:
        print(f"  [Warning] Input not found: {spec}")
        return out

    for path in sorted(set(candidates)):
        fmt = in_format if in_format != 'auto' else detect_format(path)
        if fmt is None:
            continue          # unknown extension under --in-format auto
        out.append((path, fmt))
    return out


def _extract_op(ops_str: str, key: str):
    m = re.search(rf'\b{key}\s+"([^"]*)"', ops_str)
    if m: return m.group(1).strip()
    m = re.search(rf'\b{key}\s+([^;]+)', ops_str)
    if m: return m.group(1).strip()
    return None


def _to_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_epd_line(line: str) -> Optional[Dict[str, Any]]:
    """One EPD line -> record dict. See the header's INPUT FORMATS § epd."""
    line = line.strip()
    if not line or line.startswith(('#', '%')):
        return None
    parts = line.split()
    if len(parts) < 4:
        return None
    fen     = ' '.join(parts[:4]) + ' 0 1'
    ops_str = ' '.join(parts[4:])

    res = _extract_op(ops_str, 'c0') or _extract_op(ops_str, 'result')
    # `ce` is the standard EPD centipawn opcode and wins; `c1` is this
    # project's own slot for it. A legacy c1 holding free text (older
    # selfplay EPDs wrote c1 "Checkmate") simply yields None.
    cp = _to_float(_extract_op(ops_str, 'ce'))
    if cp is None:
        cp = _to_float(_extract_op(ops_str, 'c1'))

    return {'fen': fen, 'result': res, 'id': _extract_op(ops_str, 'id') or '', 'cp': cp}


def iter_epd_file(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            rec = parse_epd_line(line)
            if rec:
                yield rec


# Move-comment eval dialects, both WHITE... no: both are STM-relative for
# cutechess and WHITE-relative for lichess. See _comment_cp().
_RE_EVAL_LICHESS   = re.compile(r'\[%eval\s+(#?-?\d+(?:\.\d+)?)\]')
_RE_EVAL_CUTECHESS = re.compile(r'^\s*([+-]?(?:\d+\.\d+|M\d+|-M\d+))/\d+')


def _comment_cp(comment: str, turn_white: bool) -> Optional[float]:
    """WHITE-relative centipawns from a PGN move comment, or None.

    Two dialects, with DIFFERENT frames of reference:
      * lichess / annotators: `[%eval 1.23]`, `[%eval #-3]` — already
        WHITE-relative, used verbatim.
      * cutechess-cli: `{+0.35/12 0.100s}`, `{-M3/20}` — relative to the
        SIDE THAT JUST MOVED, so it is negated when Black moved.
    Mates map to ±30000 cp, the same saturation the rest of the pipeline
    uses for a forced win.
    """
    if not comment:
        return None
    m = _RE_EVAL_LICHESS.search(comment)
    if m:
        tok = m.group(1)
        if tok.startswith('#'):
            return 30000.0 if not tok.startswith('#-') else -30000.0
        return float(tok) * 100.0
    m = _RE_EVAL_CUTECHESS.match(comment)
    if m:
        tok = m.group(1)
        if 'M' in tok:
            cp = 30000.0 if not tok.startswith('-') else -30000.0
        else:
            cp = float(tok) * 100.0
        return cp if turn_white else -cp
    return None


def iter_pgn_file(path: str, skip_plies: int, max_plies: int,
                  min_plies: int) -> Iterator[Dict[str, Any]]:
    """Every position of every game in a PGN file.

    `result` comes from the [Result] header (so every position of a game
    carries the REAL outcome — CLAUDE.md rule 10's ground truth), and `cp`
    from the comment attached to the move played FROM that position, when
    the PGN has one. `id` is "<event>:<round>#<ply>" so a row can be traced
    back to its game.
    """
    with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
        game_no = 0
        while True:
            try:
                game = chess.pgn.read_game(fh)
            except Exception:
                break                       # unrecoverable parse state
            if game is None:
                break
            game_no += 1
            result = game.headers.get('Result', '*')
            if result == '*':
                result = None
            tag = (game.headers.get('Event', '?') or '?').replace(' ', '_')
            node = game
            board = game.board()

            nodes: List[Tuple[chess.Board, str]] = []
            ply = 0
            while node.variations:
                nxt = node.variation(0)
                nodes.append((board.copy(stack=False), nxt.comment or ''))
                board.push(nxt.move)
                node = nxt
                ply += 1
                if max_plies and ply >= max_plies:
                    break

            if min_plies and len(nodes) < min_plies:
                continue

            for idx, (pos, comment) in enumerate(nodes):
                if idx < skip_plies:
                    continue
                yield {
                    'fen':    pos.fen(),
                    'cp':     _comment_cp(comment, pos.turn == chess.WHITE),
                    'result': result,
                    'id':     f"{tag}#{game_no}:{idx}",
                }


def iter_bin_file(path: str) -> Iterator[Dict[str, Any]]:
    """SAMPLE_DTYPE records -> record dicts (STM-relative -> WHITE-relative)."""
    fsize = os.path.getsize(path)
    if fsize % SAMPLE_DTYPE.itemsize != 0:
        print(f"  [Warning] {path}: size is not a multiple of the record size, skipping")
        return
    if fsize == 0:
        return
    arr = np.memmap(path, dtype=SAMPLE_DTYPE, mode='r')
    try:
        for i in range(len(arr)):
            raw = arr[i]
            try:
                fen = bin_record_to_fen(raw)
            except Exception:
                continue

            stm_fac  = 1 if raw["stm"] == 0 else -1
            cp_white = int(raw["eval_cp"]) * stm_fac      # record is STM-relative
            gr = int(raw["game_result"])                  # record is STM-relative
            if   gr > 0: result_str = '1-0' if raw["stm"] == 0 else '0-1'
            elif gr < 0: result_str = '0-1' if raw["stm"] == 0 else '1-0'
            else:        result_str = '1/2-1/2'

            yield {'fen': fen, 'cp': float(cp_white), 'result': result_str, 'id': ''}
    finally:
        del arr


def iter_parquet_file(path: str) -> Iterator[Dict[str, Any]]:
    import pyarrow.parquet as pq
    try:
        pf = pq.ParquetFile(path)
    except Exception as e:
        print(f"  [Warning] Error opening {path}: {e}")
        return
    names = pf.schema_arrow.names
    if 'fen' not in names:
        print(f"  [Warning] {path}: no `fen` column, skipping")
        return
    cols = [c for c in ('fen', 'cp', 'wdl', 'result', 'id') if c in names]
    try:
        for batch in pf.iter_batches(batch_size=100_000, columns=cols):
            col_lists = {c: batch.column(c).to_pylist() for c in cols}
            for i in range(batch.num_rows):
                rec = {c: col_lists[c][i] for c in cols}
                rec.setdefault('cp', None)
                rec.setdefault('result', None)
                rec.setdefault('id', '')
                # A stored `wdl` is sigmoid(cp/320): usable as a cp source ONLY
                # when cp itself is missing (rule 10 — cp always wins).
                if rec.get('cp') is None and rec.get('wdl') is not None:
                    try:
                        w = min(max(float(rec['wdl']), 1e-6), 1 - 1e-6)
                        rec['cp'] = CP_TO_WDL_T * math.log(w / (1 - w))
                    except (TypeError, ValueError):
                        pass
                rec.pop('wdl', None)
                yield rec
    except Exception as e:
        print(f"  [Warning] Error reading {path}: {e}")


def iter_file(path: str, fmt: str, cfg) -> Iterator[Dict[str, Any]]:
    if fmt == 'epd':     return iter_epd_file(path)
    if fmt == 'bin':     return iter_bin_file(path)
    if fmt == 'parquet': return iter_parquet_file(path)
    if fmt == 'pgn':     return iter_pgn_file(path, cfg.pgn_skip_plies,
                                              cfg.pgn_max_plies, cfg.pgn_min_plies)
    raise ValueError(f"unknown input format: {fmt}")


def estimate_source_size(files: List[Tuple[str, str]]) -> Optional[int]:
    """Rough record count for the ETA line. None when it cannot be guessed."""
    total = 0
    known = False
    for path, fmt in files:
        try:
            if fmt == 'parquet':
                import pyarrow.parquet as pq
                total += pq.ParquetFile(path).metadata.num_rows; known = True
            elif fmt == 'bin':
                total += os.path.getsize(path) // SAMPLE_DTYPE.itemsize; known = True
            elif fmt == 'epd':
                size = os.path.getsize(path)
                sampled_bytes = sampled_lines = 0
                with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
                    for line in fh:
                        s = line.strip()
                        if s and not s.startswith(('#', '%')):
                            sampled_bytes += len(line.encode('utf-8', errors='ignore'))
                            sampled_lines += 1
                            if sampled_lines >= EPD_SAMPLE_LINES:
                                break
                if sampled_lines:
                    total += int(size / (sampled_bytes / sampled_lines)); known = True
            elif fmt == 'pgn':
                # ~70 bytes per ply in a commented PGN is a coarse but
                # honest guess; the ETA is advisory only.
                total += os.path.getsize(path) // 70; known = True
        except Exception:
            continue
    return total if known and total else None


def record_stream(files: List[Tuple[str, str]], cfg, skip: int = 0) -> Iterator[Tuple[Dict, int]]:
    """Chain every input file into one stream of (record, consumed_count).

    `consumed_count` counts records READ since the beginning of the input
    (including the ones skipped on resume), which is what progress.json
    stores. Files are visited in sorted order so that count is stable
    across runs — resume depends on it.
    """
    consumed = 0
    for path, fmt in files:
        for rec in iter_file(path, fmt, cfg):
            consumed += 1
            if consumed <= skip:
                continue
            yield rec, consumed


def limited_stream(files: List[Tuple[str, str]], cfg, skip: int) -> Iterator[Tuple[Dict, int]]:
    """Apply --limit / --sample on top of record_stream().

    stream mode     : stop after `limit` records (limit 0 = never stop).
    reservoir mode  : read EVERYTHING, keep a uniform random `limit`-sized
                      sample (Vitter's algorithm R), then emit it shuffled.
                      Resume is not possible here (the sample is drawn from
                      the whole input every time), so `skip` must be 0 —
                      run_pipeline refuses the combination up front.
    """
    # --limit-stage output counts rows that SURVIVE the filters, so the
    # reader must NOT cap itself here — run_pipeline stops it once enough
    # rows have come back from the workers.
    limit = cfg.limit if cfg.limit_stage == 'input' else 0
    if cfg.sample == 'stream' or limit == 0:
        for rec, consumed in record_stream(files, cfg, skip):
            yield rec, consumed
            if limit and consumed - skip >= limit:
                return
        return

    rng = random.Random(cfg.seed)
    reservoir: List[Dict] = []
    seen = 0
    for rec, _ in record_stream(files, cfg, 0):
        seen += 1
        if len(reservoir) < limit:
            reservoir.append(rec)
        else:
            j = rng.randrange(seen)
            if j < limit:
                reservoir[j] = rec
    rng.shuffle(reservoir)
    print(f"  Reservoir: sampled {len(reservoir):,} of {seen:,} records")
    for i, rec in enumerate(reservoir, 1):
        yield rec, i


# ═══════════════════════════════════════════════════════════════════════════
#  STOCKFISH ENGINE (worker — rodando em subprocesso)
# ═══════════════════════════════════════════════════════════════════════════

class _StockfishEngine:
    def __init__(self, sf_path: str, nodes: int, hash_mb: int, timeout: float):
        self._sf_path = sf_path
        self._nodes   = nodes
        self._timeout = timeout
        self._re_cp   = re.compile(r'score cp (-?\d+)')
        self._re_mate = re.compile(r'score mate (-?\d+)')
        self._proc    = None
        self._start(hash_mb)

    def _start(self, hash_mb: int):
        if self._proc:
            try: self._proc.terminate()
            except Exception: pass
        self._proc = subprocess.Popen(
            [self._sf_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self._w("uci")
        self._w(f"setoption name Hash value {hash_mb}")
        self._w("setoption name Threads value 1")
        self._w("isready")
        self._read_until("readyok", 10)

    def _w(self, cmd: str):
        self._proc.stdin.write(cmd + "\n")
        self._proc.stdin.flush()

    def _readline(self, timeout: float) -> str:
        buf = [""]
        ev  = threading.Event()
        def _r():
            try:    buf[0] = self._proc.stdout.readline()
            except Exception: buf[0] = ""
            ev.set()
        threading.Thread(target=_r, daemon=True).start()
        return buf[0] if ev.wait(timeout) else ""

    def _read_until(self, marker: str, timeout: float) -> List[str]:
        lines    = []
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"SF did not emit '{marker}' in {timeout}s")
            line = self._readline(remaining)
            if not line:
                raise TimeoutError(f"SF stdout closed waiting for '{marker}'")
            lines.append(line)
            if marker in line:
                return lines

    def evaluate(self, fen: str) -> int:
        """Retorna cp WHITE-relative."""
        side = fen.split()[1]
        self._w(f"position fen {fen}")
        self._w(f"go nodes {self._nodes}")
        last_cp = 0
        for line in self._read_until("bestmove", self._timeout):
            m = self._re_mate.search(line)
            if m:
                last_cp = 30_000 if int(m.group(1)) > 0 else -30_000
            else:
                m = self._re_cp.search(line)
                if m: last_cp = int(m.group(1))
        return last_cp if side == 'w' else -last_cp

    def close(self):
        try:
            self._w("quit"); self._proc.wait(timeout=3)
        except Exception: pass
        try: self._proc.terminate()
        except Exception: pass


# ═══════════════════════════════════════════════════════════════════════════
#  WORKER
# ═══════════════════════════════════════════════════════════════════════════

def _worker_pipeline(records_batch: List[Dict], use_sf: bool,
                     sf_path: str, sf_nodes: int, sf_hash_mb: int, sf_timeout: float,
                     score_cap: int, filter_score_cap: bool,
                     filter_endgame: bool, endgame_max: int, endgame_no_q: bool,
                     filter_check: bool, filter_terminal: bool,
                     filter_win: bool, filter_equal: bool, filter_sacrifice: bool,
                     filter_quiet: bool, require_cp: bool, require_result: bool,
                     ) -> Tuple[List[Dict], Dict]:
    """Worker: filtra, avalia com SF se necessário, retorna records prontos."""
    # Inject the settings into this CHILD process's globals — is_quiet_advanced
    # and is_endgame_position read them, and a child gets a fresh import of
    # this module, not the parent's edited state.
    global FILTER_IN_CHECK, FILTER_TERMINAL, FILTER_WIN_CAPTURE, FILTER_EQUAL_CAPTURE
    global FILTER_SACRIFICE, FILTER_SCORE_CAP, SCORE_CAP_VALUE
    global FILTER_ENDGAME, ENDGAME_MAX_PIECES, ENDGAME_NO_QUEENS
    FILTER_IN_CHECK      = filter_check
    FILTER_TERMINAL      = filter_terminal
    FILTER_WIN_CAPTURE   = filter_win
    FILTER_EQUAL_CAPTURE = filter_equal
    FILTER_SACRIFICE     = filter_sacrifice
    FILTER_SCORE_CAP     = filter_score_cap
    SCORE_CAP_VALUE      = score_cap
    FILTER_ENDGAME       = filter_endgame
    ENDGAME_MAX_PIECES   = endgame_max
    ENDGAME_NO_QUEENS    = endgame_no_q

    results: List[Dict] = []
    stats:   Dict       = {}

    def bump(key: str):
        stats[key] = stats.get(key, 0) + 1

    engine = None
    if use_sf:
        try:
            engine = _StockfishEngine(sf_path, sf_nodes, sf_hash_mb, sf_timeout)
        except Exception:
            stats['sf_init_error'] = 1
            return results, stats

    try:
        for rec in records_batch:
            fen = rec.get('fen', '')
            if not fen:
                bump('bad_fen'); continue
            try:
                board = chess.Board(fen)
            except Exception:
                bump('bad_fen'); continue

            # ── Filtro quiet ──────────────────────────────────────────────
            if filter_quiet:
                ok, reason = is_quiet_advanced(board)
                if not ok:
                    bump(reason); continue

            # ── Filtro endgame ────────────────────────────────────────────
            if filter_endgame:
                ok, reason = is_endgame_position(board)
                if not ok:
                    bump(reason); continue

            # ── Avaliação ─────────────────────────────────────────────────
            # `result` is ALWAYS the real game outcome from the source, in both
            # branches. Never synthesise it from cp: that would make the
            # trainer's lambda term a second copy of the evaluation term.
            result_white = normalize_result_to_prob(rec.get('result'))

            if use_sf:
                try:
                    cp_white = float(engine.evaluate(fen))
                except TimeoutError:
                    bump('sf_timeout')
                    try: engine.close()
                    except Exception: pass
                    try: engine = _StockfishEngine(sf_path, sf_nodes, sf_hash_mb, sf_timeout)
                    except Exception: engine = None
                    continue
                except Exception:
                    bump('sf_error'); continue
            else:
                cp_white = _to_float(rec.get('cp'))
                if cp_white is not None and cp_white != cp_white:   # NaN
                    cp_white = None

            # A row needs at least ONE of cp / result to carry any signal.
            # NO BLEND HERE: cp and result are emitted as two INDEPENDENT
            # columns and combined only by the trainer (per-dataset LAMBDA in
            # train/train_nnue.py's DATASETS block). The counters below are
            # printed at the end so a source missing one of them is visible
            # instead of silently halving the available signal.
            has_cp  = cp_white is not None
            has_res = result_white is not None
            if not has_cp and not has_res:
                bump('missing_cp_and_result'); continue
            if require_cp and not has_cp:
                bump('require_cp'); continue
            if require_result and not has_res:
                bump('require_result'); continue
            if not has_cp:  bump('result_only')
            if not has_res: bump('cp_only')

            # ── Score cap ─────────────────────────────────────────────────
            if filter_score_cap and has_cp and abs(cp_white) > score_cap:
                bump('score_cap'); continue

            results.append({
                'fen':    fen,
                'cp':     cp_white,      # WHITE-relative centipawns, or None
                # WHITE-RELATIVE, same scale as `wdl`: 1.0 White won,
                # 0.5 draw, 0.0 Black won, None unknown. Numeric (not the
                # PGN string) so the trainer never has to parse text and so
                # the blend lam*result + (1-lam)*wdl stays a convex sum on
                # one scale. See CLAUDE.md rule 10.
                'result': result_white,
                'id':     rec.get('id', '') or '',
            })
    finally:
        if engine:
            try: engine.close()
            except Exception: pass

    return results, stats


# ═══════════════════════════════════════════════════════════════════════════
#  SAÍDA
# ═══════════════════════════════════════════════════════════════════════════

def _epd_line(row: Dict, emit_id: bool) -> str:
    """`<fen4> c0 "<result>"; c1 "<cp>";` — the project's EPD convention
    (tests/run_selfplay.py, tests/run_tournament.py, selfplay.c all agree:
    c0 = result, c1 = cp). An absent value is written as an empty opcode
    rather than a fake 0, so a reader can tell "no eval" from "eval 0"."""
    epd = ' '.join(row['fen'].split()[:4])
    cp_str  = '' if row.get('cp') is None else f'{int(round(row["cp"]))}'
    res_str = prob_to_result_str(row.get('result'))
    out = f'{epd} c0 "{res_str}"; c1 "{cp_str}";'
    if emit_id and row.get('id'):
        out += f' id "{row["id"]}";'
    return out


def _pgn_stub(row: Dict) -> str:
    """A single position as a zero-move PGN game (viewing format)."""
    res = prob_to_result_str(row.get('result')) or '*'
    cp  = row.get('cp')
    ev  = '' if cp is None else f' {{ [%eval {cp/100.0:.2f}] }}'
    return (f'[Event "position"]\n[Site "?"]\n[Date "????.??.??"]\n'
            f'[Round "-"]\n[White "?"]\n[Black "?"]\n[Result "{res}"]\n'
            f'[SetUp "1"]\n[FEN "{row["fen"]}"]\n\n{res}{ev}\n\n')


class OutputSink:
    """Writes rows in every requested format, to a DIRECTORY (chunked and
    resumable) or to a single FILE (streamed).

    Directory mode keeps the historical layout — chunk_NNNN.<ext> per
    CHUNK_SIZE_SAVE rows plus progress.json — so an interrupted run can be
    resumed. File mode writes one continuous file per format, which is what
    you want for a plain conversion (`--in x.epd --out x.bin`).
    """

    def __init__(self, dest: str, formats: List[str], dir_mode: bool,
                 append: bool, chunk_size: int, emit_id: bool, dry_run: bool,
                 start_chunk: int = 0):
        self.dest        = dest
        self.formats     = list(formats)
        self.dir_mode    = dir_mode
        self.append      = append
        self.chunk_size  = chunk_size
        self.emit_id     = emit_id
        self.dry_run     = dry_run
        self.chunk_idx   = start_chunk
        self.rows_written = 0
        self._handles: Dict[str, Any] = {}
        self._pq_writer = None

        if self.dry_run:
            return
        if dir_mode:
            os.makedirs(dest, exist_ok=True)
        else:
            parent = os.path.dirname(os.path.abspath(dest))
            if parent:
                os.makedirs(parent, exist_ok=True)

    # ── file-mode handles (opened lazily so a dry run touches nothing) ──
    def _handle(self, fmt: str):
        if fmt in self._handles:
            return self._handles[fmt]
        path = self._file_path(fmt)
        mode = 'ab' if self.append else 'wb'
        if fmt in ('epd', 'pgn'):
            fh = open(path, 'a' if self.append else 'w', encoding='utf-8')
        else:
            fh = open(path, mode)
        self._handles[fmt] = fh
        return fh

    def _file_path(self, fmt: str) -> str:
        """In file mode with several formats requested, the extra formats get
        the destination's stem plus their own extension."""
        if len(self.formats) == 1:
            return self.dest
        stem, _ = os.path.splitext(self.dest)
        return f'{stem}.{ "parquet" if fmt == "parquet" else fmt }'

    def write(self, rows: List[Dict]):
        if not rows:
            return
        self.rows_written += len(rows)
        if self.dry_run:
            return
        if self.dir_mode:
            self._write_chunk(rows)
        else:
            self._write_stream(rows)

    # ── directory mode ──────────────────────────────────────────────────
    def _write_chunk(self, rows: List[Dict]):
        idx = self.chunk_idx
        for fmt in self.formats:
            path = os.path.join(self.dest, f'chunk_{idx:04d}.{fmt}')
            if fmt == 'parquet':
                self._df(rows).to_parquet(path, index=False)
            elif fmt == 'bin':
                self._write_bin(open(path, 'ab'), rows, close=True)
            elif fmt == 'epd':
                with open(path, 'a', encoding='utf-8') as fh:
                    fh.write('\n'.join(_epd_line(r, self.emit_id) for r in rows) + '\n')
            elif fmt == 'pgn':
                with open(path, 'a', encoding='utf-8') as fh:
                    fh.writelines(_pgn_stub(r) for r in rows)
        self.chunk_idx += 1

    # ── single-file mode ────────────────────────────────────────────────
    def _write_stream(self, rows: List[Dict]):
        for fmt in self.formats:
            if fmt == 'parquet':
                import pyarrow as pa
                import pyarrow.parquet as pq
                table = pa.Table.from_pandas(self._df(rows), preserve_index=False)
                if self._pq_writer is None:
                    # Streamed parquet: one writer, one schema, appended row
                    # groups — so a 50M-row conversion never materialises in
                    # RAM. `--append` cannot extend an existing parquet file
                    # (the format has a footer), so it starts a new one.
                    self._pq_writer = pq.ParquetWriter(self._file_path(fmt), table.schema)
                self._pq_writer.write_table(table)
            elif fmt == 'bin':
                self._write_bin(self._handle(fmt), rows, close=False)
            elif fmt == 'epd':
                fh = self._handle(fmt)
                fh.write('\n'.join(_epd_line(r, self.emit_id) for r in rows) + '\n')
            elif fmt == 'pgn':
                fh = self._handle(fmt)
                fh.writelines(_pgn_stub(r) for r in rows)

    def _df(self, rows: List[Dict]) -> pd.DataFrame:
        cols = ['fen', 'cp', 'result'] + (['id'] if self.emit_id else [])
        df = pd.DataFrame([{c: r.get(c) for c in cols} for r in rows], columns=cols)
        # Fixed dtypes so every chunk/row-group shares one schema (a streamed
        # ParquetWriter rejects a schema change mid-file, and an all-None
        # `cp` chunk would otherwise be inferred as null instead of double).
        df['cp']     = pd.to_numeric(df['cp'],     errors='coerce').astype('float64')
        df['result'] = pd.to_numeric(df['result'], errors='coerce').astype('float64')
        df['fen']    = df['fen'].astype(str)
        if self.emit_id:
            df['id'] = df['id'].fillna('').astype(str)
        return df

    def _write_bin(self, fh, rows: List[Dict], close: bool):
        recs = []
        for r in rows:
            try:
                board = chess.Board(r['fen'])
            except Exception:
                continue
            cp_w = int(round(r['cp'])) if r.get('cp') is not None else 0
            # NOTE: no `r.get('result') or None` fallback — 0.0 (Black won,
            # WHITE-relative) is falsy in Python and would be silently
            # coerced to "unknown".
            recs.append(board_to_bin_record(board, cp_w, r.get('result', None)))
        if recs:
            fh.write(np.concatenate(recs).tobytes())
        if close:
            fh.close()

    def close(self):
        if self._pq_writer is not None:
            self._pq_writer.close()
            self._pq_writer = None
        for fh in self._handles.values():
            try: fh.close()
            except Exception: pass
        self._handles.clear()


# ═══════════════════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def resolve_output(dest: str, out_formats: List[str],
                   explicit_formats: bool) -> Tuple[bool, List[str]]:
    """Decide directory-vs-file mode and the format list for one --out.

    A destination with a known data extension (.parquet/.bin/.epd/.pgn) is a
    FILE; anything else is a DIRECTORY. A file's extension also SETS the
    output format unless --out-format was given explicitly — which is what
    makes `--in a.epd --out a.bin` work with no other flag.
    """
    ext_fmt = detect_format(dest)
    if ext_fmt is None:
        return True, out_formats
    if explicit_formats:
        return False, out_formats
    return False, [ext_fmt]


def run_pipeline(in_spec: str, out_spec: str, cfg) -> int:
    files = collect_input_files(in_spec, cfg.in_format)
    dir_mode, formats = resolve_output(out_spec, cfg.out_formats, cfg.explicit_out_formats)

    filter_quiet = (cfg.filter_in_check or cfg.filter_terminal or
                    cfg.filter_win_capture or cfg.filter_equal_capture or
                    cfg.filter_sacrifice)

    print(f"\n{'='*66}")
    print(f"  {in_spec}  →  {out_spec}")
    print(f"  Input     : {len(files)} file(s)"
          + (f"  [{', '.join(sorted({f for _, f in files}))}]" if files else ""))
    print(f"  Output    : {'directory (chunked' if dir_mode else 'file (streamed'}"
          f", formats={formats})")
    print(f"  Stockfish : {'ON  (' + str(cfg.sf_nodes) + ' nodes)' if cfg.use_stockfish else 'OFF (cp comes from the source)'}")
    print(f"  Limit     : {'none' if cfg.limit == 0 else f'{cfg.limit:,} ({cfg.limit_stage})'}"
          f"  sample={cfg.sample}")
    print(f"  Filters   : quiet={'ON' if filter_quiet else 'OFF'}"
          f"  endgame={'ON' if cfg.filter_endgame else 'OFF'}"
          f"  dedup={'ON' if cfg.filter_duplicates else 'OFF'}"
          f"  score_cap={cfg.score_cap if cfg.filter_score_cap else 'OFF'}")
    if filter_quiet:
        on = [n for n, v in (('in_check', cfg.filter_in_check),
                             ('terminal', cfg.filter_terminal),
                             ('win_capture', cfg.filter_win_capture),
                             ('equal_capture', cfg.filter_equal_capture),
                             ('sacrifice', cfg.filter_sacrifice)) if v]
        print(f"              quiet parts: {', '.join(on)}")
    if cfg.filter_endgame:
        print(f"              max_pieces={cfg.endgame_max_pieces}  no_queens={cfg.endgame_no_queens}")
    if cfg.dry_run:
        print("  DRY RUN   : nothing will be written")
    print(f"{'='*66}")

    if not files:
        print("  Nothing to do.")
        return 0

    est_size = estimate_source_size(files)
    total_estimated = None
    if est_size:
        total_estimated = est_size if cfg.limit == 0 else min(est_size, cfg.limit)
    print(f"  Source: ~{est_size:,} records" if est_size else "  Source: size unknown")

    # ── Resume (directory mode only) ───────────────────────────────────────
    seen_fens: set = set()
    start_chunk     = 0
    resume_consumed = 0
    progress_dict: Dict[str, int] = {}
    pipeline_key = f"{in_spec}|{cfg.in_format}"

    if dir_mode and cfg.resume and not cfg.dry_run:
        existing = []
        for fmt in formats:
            existing += glob.glob(os.path.join(out_spec, f'chunk_*.{fmt}'))
        chunk_nums = [int(m.group(1)) for m in
                      (re.search(r'chunk_(\d+)', os.path.basename(p)) for p in existing) if m]
        start_chunk = (max(chunk_nums) + 1) if chunk_nums else 0
        progress_dict   = load_progress(out_spec)
        resume_consumed = progress_dict.get(pipeline_key, 0)
        if start_chunk > 0:
            print(f"\n  Resuming at chunk {start_chunk} (input record {resume_consumed:,}).")
            if cfg.filter_duplicates and 'parquet' in formats:
                import pyarrow.parquet as pq
                print(f"  Loading already-written FENs from {start_chunk} chunk(s)...")
                for cpath in sorted(glob.glob(os.path.join(out_spec, 'chunk_*.parquet'))):
                    try:
                        pf = pq.ParquetFile(cpath)
                        for batch in pf.iter_batches(200_000, ['fen']):
                            for fval in batch.column('fen').to_pylist():
                                seen_fens.add(fen_key(fval))
                    except Exception as e:
                        print(f"  [Warning] {cpath}: {e}")
                print(f"  {len(seen_fens):,} unique FENs loaded.\n")

    if resume_consumed and cfg.sample == 'reservoir':
        print("  [Error] --sample reservoir cannot resume (the sample is drawn from the\n"
              "          whole input each run). Use --no-resume, or --sample stream.")
        return 1

    sink = OutputSink(out_spec, formats, dir_mode, cfg.append, cfg.chunk_size,
                      cfg.emit_id, cfg.dry_run, start_chunk)

    total_saved    = len(seen_fens) if start_chunk > 0 else 0
    total_rej: Dict[str, int] = {}
    saved_this_run = sent_this_run = 0
    pending_save: List[Dict] = []
    t_start = time.time()
    stop_reading = False

    print(f"  Workers: {cfg.workers}  Batch: {cfg.batch_size:,}  Chunk: {cfg.chunk_size:,}\n")

    worker_kwargs = dict(
        use_sf           = cfg.use_stockfish,
        sf_path          = cfg.sf_path,
        sf_nodes         = cfg.sf_nodes,
        sf_hash_mb       = cfg.sf_hash_mb,
        sf_timeout       = cfg.sf_timeout,
        score_cap        = cfg.score_cap,
        filter_score_cap = cfg.filter_score_cap,
        filter_endgame   = cfg.filter_endgame,
        endgame_max      = cfg.endgame_max_pieces,
        endgame_no_q     = cfg.endgame_no_queens,
        filter_check     = cfg.filter_in_check,
        filter_terminal  = cfg.filter_terminal,
        filter_win       = cfg.filter_win_capture,
        filter_equal     = cfg.filter_equal_capture,
        filter_sacrifice = cfg.filter_sacrifice,
        filter_quiet     = filter_quiet,
        require_cp       = cfg.require_cp,
        require_result   = cfg.require_result,
    )

    def flush_pending(force: bool):
        """Move finished rows from pending_save into the sink, chunk by chunk."""
        nonlocal pending_save, total_saved, saved_this_run
        while pending_save and (force or len(pending_save) >= cfg.chunk_size):
            take = pending_save[:cfg.chunk_size]
            pending_save = pending_save[cfg.chunk_size:]
            sink.write(take)
            total_saved    += len(take)
            saved_this_run += len(take)
            if dir_mode and not cfg.dry_run and cfg.resume:
                progress_dict[pipeline_key] = current_consumed[0]
                save_progress(out_spec, progress_dict)
            elapsed     = time.time() - t_start
            pos_per_sec = saved_this_run / max(elapsed, 1e-9)
            pass_rate   = (saved_this_run / max(sent_this_run, 1)) * 100
            eta_str = ''
            if pos_per_sec > 0 and total_estimated:
                rem   = max(0, total_estimated - current_consumed[0])
                eta_s = (rem * pass_rate / 100) / pos_per_sec
                eta_str = f' | ETA ~{format_time(eta_s)}'
            label = f'Chunk {sink.chunk_idx - 1:04d}' if dir_mode else 'Wrote'
            print(f'  {label} | +{len(take):,} | '
                  f'Total: {total_saved:,} | Pass: {pass_rate:.1f}% | '
                  f'{pos_per_sec:.0f} pos/s{eta_str}')
            gc.collect()

    current_consumed = [resume_consumed]

    with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
        futures: Dict = {}
        producer  = limited_stream(files, cfg, resume_consumed)
        exhausted = False
        buf: List[Dict] = []

        def flush_buf():
            nonlocal buf
            if buf:
                futures[pool.submit(_worker_pipeline, list(buf), **worker_kwargs)] = time.time()
                buf = []

        while not exhausted or futures:
            # 1. feed the pool
            while not exhausted and len(futures) < cfg.max_active_batches:
                if stop_reading:
                    flush_buf(); exhausted = True; break
                try:
                    rec, current_consumed[0] = next(producer)
                except StopIteration:
                    flush_buf(); exhausted = True; break
                if cfg.filter_duplicates:
                    k = fen_key(rec['fen'])
                    if k in seen_fens:
                        total_rej['duplicate'] = total_rej.get('duplicate', 0) + 1
                        continue
                    seen_fens.add(k)
                sent_this_run += 1
                buf.append(rec)
                if len(buf) >= cfg.batch_size:
                    flush_buf()

            # 2. drain finished futures
            if futures:
                done_futs = []
                try:
                    for fut in as_completed(futures, timeout=0.1):
                        done_futs.append(fut)
                except FuturesTimeoutError:
                    pass

                for fut in done_futs:
                    try:
                        res_list, res_stats = fut.result()
                    except Exception as e:
                        print(f"  [Worker error] {e}")
                        del futures[fut]; continue
                    for reason, cnt in res_stats.items():
                        total_rej[reason] = total_rej.get(reason, 0) + cnt
                    pending_save.extend(res_list)
                    del futures[fut]
                    # --limit-stage output: stop reading once enough rows
                    # SURVIVED the filters (the in-flight batches still land).
                    if (cfg.limit and cfg.limit_stage == 'output'
                            and saved_this_run + len(pending_save) >= cfg.limit):
                        stop_reading = True
                    flush_pending(force=False)

        if cfg.limit and cfg.limit_stage == 'output':
            overflow = saved_this_run + len(pending_save) - cfg.limit
            if overflow > 0:
                pending_save = pending_save[:max(0, len(pending_save) - overflow)]
        flush_pending(force=True)

    sink.close()
    elapsed = time.time() - t_start
    print(f'\n  DONE — read {current_consumed[0] - resume_consumed:,} | '
          f'kept {saved_this_run:,} '
          f'({saved_this_run / max(sent_this_run,1) * 100:.1f}%) | {format_time(elapsed)}')
    if total_rej:
        width = max(len(k) for k in total_rej)
        for reason, cnt in sorted(total_rej.items(), key=lambda kv: -kv[1]):
            print(f'    {reason.ljust(width)}  {cnt:,}')
    if cfg.dry_run:
        print('  (dry run — no files were written)')
    return 0


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

_PRESETS = {
    #                    in_check terminal  win   equal  sacri  endgame
    'none':             (False,   False,   False, False, False, False),
    'quiet':            (True,    True,    True,  True,  False, False),
    'endgame':          (False,   False,   False, False, False, True),
    'quiet+endgame':    (True,    True,    True,  True,  False, True),
}


def add_bool(ap, name: str, default: bool, help_on: str):
    """--flag / --no-flag pair whose default is the CONFIGURATION constant.

    `default=None` is deliberately NOT used: argparse needs to report
    "unset" so --filters can override, so both actions store into a dest
    that starts as None and the real default is applied afterwards in
    resolve_config(). The constant is still the single source of truth —
    it is printed in the help text and used as the fallback there.
    """
    dest = name.replace('-', '_')
    g = ap.add_mutually_exclusive_group()
    g.add_argument(f'--{name}', dest=dest, action='store_true', default=None,
                   help=f'{help_on} (config default: {default})')
    g.add_argument(f'--no-{name}', dest=dest, action='store_false',
                   help=argparse.SUPPRESS)


class Cfg:
    """Resolved settings for one run — a plain value object so the worker
    kwargs and the pipeline never read module globals for a decision."""


def resolve_config(args) -> Cfg:
    c = Cfg()
    c.in_format  = args.in_format
    c.out_formats = args.out_format
    c.explicit_out_formats = args.out_format_explicit
    c.append     = args.append
    c.resume     = args.resume
    c.emit_id    = args.emit_id
    c.dry_run    = args.dry_run

    c.limit       = args.limit
    c.limit_stage = args.limit_stage
    c.sample      = args.sample
    c.seed        = args.seed

    c.pgn_skip_plies = args.pgn_skip_plies
    c.pgn_max_plies  = args.pgn_max_plies
    c.pgn_min_plies  = args.pgn_min_plies

    # Filter resolution, last wins:
    #   FILTER_* constants  →  --filters <preset>  →  explicit --filter-x
    base = dict(filter_in_check=FILTER_IN_CHECK, filter_terminal=FILTER_TERMINAL,
                filter_win_capture=FILTER_WIN_CAPTURE,
                filter_equal_capture=FILTER_EQUAL_CAPTURE,
                filter_sacrifice=FILTER_SACRIFICE, filter_endgame=FILTER_ENDGAME)
    if args.filters != 'config':
        keys = ('filter_in_check', 'filter_terminal', 'filter_win_capture',
                'filter_equal_capture', 'filter_sacrifice', 'filter_endgame')
        base = dict(zip(keys, _PRESETS[args.filters]))
    for k, v in base.items():
        explicit = getattr(args, k)
        setattr(c, k, v if explicit is None else explicit)

    c.filter_duplicates = FILTER_DUPLICATES if args.filter_duplicates is None else args.filter_duplicates
    c.filter_score_cap  = FILTER_SCORE_CAP  if args.filter_score_cap  is None else args.filter_score_cap
    # `--filters none` means NO filtering, dedup and score-cap included,
    # unless those two were named explicitly on the command line.
    if args.filters == 'none':
        if args.filter_duplicates is None: c.filter_duplicates = False
        if args.filter_score_cap  is None: c.filter_score_cap  = False
    c.score_cap = args.score_cap

    c.endgame_max_pieces = args.endgame_max_pieces
    c.endgame_no_queens  = ENDGAME_NO_QUEENS if args.endgame_no_queens is None else args.endgame_no_queens

    c.require_cp     = REQUIRE_CP     if args.require_cp     is None else args.require_cp
    c.require_result = REQUIRE_RESULT if args.require_result is None else args.require_result

    c.use_stockfish = USE_STOCKFISH if args.stockfish is None else args.stockfish
    c.sf_path       = args.sf_path
    c.sf_nodes      = args.sf_nodes
    c.sf_hash_mb    = args.sf_hash_mb
    c.sf_timeout    = args.sf_timeout

    c.workers            = args.workers
    c.batch_size         = args.batch_size
    c.chunk_size         = args.chunk_size
    c.max_active_batches = args.max_active_batches
    return c


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog='process_positions.py',
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    io_g = ap.add_argument_group('input / output')
    io_g.add_argument('--in', dest='inputs', action='append', metavar='PATH',
                      help='input FILE, DIRECTORY (recursive) or glob. Repeatable; '
                           f'pairs with --out. Config default: {INPUTS}')
    io_g.add_argument('--out', dest='outputs', action='append', metavar='PATH',
                      help='output DIRECTORY (chunked, resumable) or FILE (streamed; '
                           f'format from its extension). Repeatable. Config default: {OUTPUTS}')
    io_g.add_argument('--in-format', choices=IN_FORMATS, default=IN_FORMAT,
                      help=f'force the input parser instead of using the extension (default: {IN_FORMAT})')
    io_g.add_argument('--out-format', action='append', choices=OUT_FORMAT_CHOICES,
                      help='output format; repeat for several at once. Ignored when --out '
                           f'is a file with a known extension. Config default: {OUT_FORMATS}')
    io_g.add_argument('--emit-id', action='store_true', default=EMIT_ID,
                      help=f'write the source `id` column too (default: {EMIT_ID})')
    io_g.add_argument('--append', action='store_true', default=APPEND,
                      help=f'file output: append instead of truncating (default: {APPEND})')
    io_g.add_argument('--no-resume', dest='resume', action='store_false', default=RESUME,
                      help=f'directory output: ignore progress.json and start over (default resume: {RESUME})')
    io_g.add_argument('--dry-run', action='store_true', default=DRY_RUN,
                      help=f'read and filter, report, write nothing (default: {DRY_RUN})')

    amt = ap.add_argument_group('how much to read')
    amt.add_argument('--limit', type=parse_limit, default=LIMIT, metavar='N',
                     help=f"stop after N records; 0 or 'all' = everything (default: {LIMIT})")
    amt.add_argument('--limit-stage', choices=('input', 'output'), default=LIMIT_STAGE,
                     help=f'count --limit on records READ or rows KEPT (default: {LIMIT_STAGE})')
    amt.add_argument('--sample', choices=('stream', 'reservoir'), default=SAMPLE,
                     help=f'stream = file order (resumable); reservoir = uniform random '
                          f'--limit sample of the whole input (default: {SAMPLE})')
    amt.add_argument('--seed', type=int, default=SEED,
                     help=f'RNG seed for reservoir sampling (default: {SEED})')

    pgn = ap.add_argument_group('pgn input')
    pgn.add_argument('--pgn-skip-plies', type=int, default=PGN_SKIP_PLIES,
                     help=f'drop the first N plies of each game (default: {PGN_SKIP_PLIES})')
    pgn.add_argument('--pgn-max-plies', type=int, default=PGN_MAX_PLIES,
                     help=f'stop each game after N plies; 0 = whole game (default: {PGN_MAX_PLIES})')
    pgn.add_argument('--pgn-min-plies', type=int, default=PGN_MIN_PLIES,
                     help=f'skip games shorter than N plies (default: {PGN_MIN_PLIES})')

    flt = ap.add_argument_group('filters (every one optional)')
    flt.add_argument('--filters', choices=tuple(_PRESETS) + ('config',), default=FILTER_PRESET,
                     help=f"preset: none = pure conversion, quiet, endgame, quiet+endgame, "
                          f"config = the FILTER_* constants (default: {FILTER_PRESET})")
    add_bool(flt, 'filter-duplicates',    FILTER_DUPLICATES,    'drop repeated FENs')
    add_bool(flt, 'filter-in-check',      FILTER_IN_CHECK,      'drop positions in check')
    add_bool(flt, 'filter-terminal',      FILTER_TERMINAL,      'drop mate/stalemate positions')
    add_bool(flt, 'filter-win-capture',   FILTER_WIN_CAPTURE,   'drop positions with a winning capture')
    add_bool(flt, 'filter-equal-capture', FILTER_EQUAL_CAPTURE, 'drop positions with an equal capture')
    add_bool(flt, 'filter-sacrifice',     FILTER_SACRIFICE,     "also check the opponent's captures")
    add_bool(flt, 'filter-score-cap',     FILTER_SCORE_CAP,     'drop positions above --score-cap')
    flt.add_argument('--score-cap', type=int, default=SCORE_CAP_VALUE, metavar='CP',
                     help=f'|cp| above this is dropped (default: {SCORE_CAP_VALUE})')
    add_bool(flt, 'filter-endgame',       FILTER_ENDGAME,       'keep ONLY endgame positions')
    flt.add_argument('--endgame-max-pieces', type=int, default=ENDGAME_MAX_PIECES, metavar='N',
                     help=f'endgame filter: max pieces incl. kings (default: {ENDGAME_MAX_PIECES})')
    add_bool(flt, 'endgame-no-queens',    ENDGAME_NO_QUEENS,    'endgame filter: reject queens')
    add_bool(flt, 'require-cp',           REQUIRE_CP,           'drop rows without cp')
    add_bool(flt, 'require-result',       REQUIRE_RESULT,       'drop rows without a game result')

    sf = ap.add_argument_group('stockfish relabelling (optional)')
    add_bool(sf, 'stockfish', USE_STOCKFISH, 'evaluate every surviving position with Stockfish')
    sf.add_argument('--sf-path', default=SF_PATH, help=f'(default: {SF_PATH})')
    sf.add_argument('--sf-nodes', type=int, default=SF_NODES, help=f'(default: {SF_NODES})')
    sf.add_argument('--sf-hash-mb', type=int, default=SF_HASH_MB, help=f'(default: {SF_HASH_MB})')
    sf.add_argument('--sf-timeout', type=float, default=SF_TIMEOUT_S, help=f'(default: {SF_TIMEOUT_S})')

    ap.add_argument('--show-config', action='store_true',
                    help='print the resolved settings and exit (reads nothing, writes nothing) '
                         '— the same switch every other tool in the repo has')

    perf = ap.add_argument_group('parallelism')
    perf.add_argument('--workers', type=int, default=WORKERS, help=f'(default: {WORKERS})')
    perf.add_argument('--batch-size', type=int, default=BATCH_SIZE, help=f'(default: {BATCH_SIZE})')
    perf.add_argument('--chunk-size', type=int, default=CHUNK_SIZE_SAVE, help=f'(default: {CHUNK_SIZE_SAVE})')
    perf.add_argument('--max-active-batches', type=int, default=MAX_ACTIVE_BATCHES,
                      help=f'(default: {MAX_ACTIVE_BATCHES})')
    return ap


def main(argv=None) -> int:
    ap   = build_parser()
    args = ap.parse_args(argv)

    # --out-format: `action='append'` cannot have a list default without the
    # CLI values being APPENDED to it, so the config default is applied here
    # and a flag also marks the choice as explicit (which is what lets a
    # file extension set the format when it was NOT given).
    args.out_format_explicit = args.out_format is not None
    if args.out_format is None:
        args.out_format = list(OUT_FORMATS)

    inputs  = args.inputs  if args.inputs  else list(INPUTS)
    outputs = args.outputs if args.outputs else list(OUTPUTS)

    if len(inputs) != len(outputs):
        ap.error(f"--in and --out must come in pairs: got {len(inputs)} input(s) "
                 f"and {len(outputs)} output(s)")

    cfg = resolve_config(args)

    if args.show_config:
        print("Resolved configuration:")
        print(f"  INPUTS  = {inputs}")
        print(f"  OUTPUTS = {outputs}")
        width = max(len(k) for k in vars(cfg))
        for k, v in vars(cfg).items():
            print(f"  {k.upper().ljust(width)} = {v!r}")
        return 0

    if cfg.use_stockfish and not os.path.isfile(cfg.sf_path):
        print(f"Error: Stockfish not found at: {cfg.sf_path}")
        return 1
    if cfg.sample == 'reservoir' and cfg.limit == 0:
        ap.error("--sample reservoir needs a --limit (it samples --limit records)")
    if cfg.limit_stage == 'output' and cfg.sample == 'reservoir':
        ap.error("--limit-stage output is meaningless with --sample reservoir "
                 "(the reservoir already samples exactly --limit INPUT records)")

    rc = 0
    for in_spec, out_spec in zip(inputs, outputs):
        rc |= run_pipeline(in_spec, out_spec, cfg)
    return rc


if __name__ == '__main__':
    sys.exit(main())
