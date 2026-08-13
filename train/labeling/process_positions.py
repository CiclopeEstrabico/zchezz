"""
process_positions.py
────────────────────
Script unificado que substitui filter_wdl.py + label_selfplay.py.

Lê posições de múltiplos formatos (epd, parquet, bin), aplica filtros
configuráveis (quiet, endgame) e salva em um ou mais formatos de saída.

Stockfish é OPCIONAL: se USE_STOCKFISH=False, usa cp/wdl já existente na
fonte com WDL blending. Se USE_STOCKFISH=True, avalia cada posição com SF.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SEÇÃO DE CONFIG  ← edite aqui
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

import os, re, glob, sys, time, gc, json, math, random, pathlib, subprocess, threading
import numpy as np
import pandas as pd
import chess
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import List, Dict, Tuple, Any, Optional

# train/dataset.py is the AUTHORITATIVE definition of SAMPLE_DTYPE (the
# cross-language .bin record contract shared with engine/c/tools/sample.h).
# Import it rather than hand-rolling a second copy here — a hand-rolled
# duplicate can silently drift from dataset.py's dtype and reinterpret
# every .bin read as garbage (see dataset.py's own docstring warning).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from dataset import SAMPLE_DTYPE  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════
#  CAMINHOS  (multiplos pares IN→OUT suportados)
# ═══════════════════════════════════════════════════════════════════════════

IN_DIRS = [
    r'data\selfplay_raw_res_data20260401',
    r'data\selfplay_raw_res_data20260404',
    r'data\selfplay_raw_res_data20260410',
]

OUT_DIRS = [
    r'data\selfplay_cp_zchezz_res_filter_data20260401',
    r'data\selfplay_cp_zchezz_res_filter_data20260404',
    r'data\selfplay_cp_zchezz_res_filter_data20260410',
]

# ── Tipo de entrada ────────────────────────────────────────────────────────
# Opções: 'epd'  | 'parquet'  | 'bin'
SRC_TYPE = 'epd'
LIMIT    = 'all'          # int ou 'all'

# ── Formatos de saída ──────────────────────────────────────────────────────
# Lista com um ou mais dos seguintes: 'parquet', 'bin', 'epd'
# Exemplo: ['parquet', 'bin']  →  gera os dois ao mesmo tempo
OUTPUT_FORMATS = ['parquet']

# ── Stockfish (opcional) ───────────────────────────────────────────────────
USE_STOCKFISH = False
SF_PATH       = r"engine\stockfish_fast\stockfish\stockfish-windows-x86-64-avxvnni.exe"
SF_NODES      = 1_000_000
SF_HASH_MB    = 16
SF_TIMEOUT_S  = 60

# ── Filtros quiet (cada um individualmente configurável) ───────────────────
FILTER_DUPLICATES    = True
FILTER_IN_CHECK      = True
FILTER_TERMINAL      = True
FILTER_WIN_CAPTURE   = True   # vitima > atacante + tolerância
FILTER_EQUAL_CAPTURE = True   # |vitima - atacante| <= tolerância (pxp isento)
FILTER_SACRIFICE     = False  # checar capturas do lado adversário também
FILTER_SCORE_CAP     = True
SCORE_CAP_VALUE      = 3000   # centipawns; usado quando FILTER_SCORE_CAP=True

# ── Filtro endgame (off por padrão) ───────────────────────────────────────
FILTER_ENDGAME     = False
ENDGAME_MAX_PIECES = 14       # máximo de peças no tabuleiro (incluindo reis)
ENDGAME_NO_QUEENS  = True     # rejeita posições com rainha(s)

# ── (removido) WDL blending ───────────────────────────────────────────────
# Este script NAO combina cp e result num alvo unico. Ele emite as duas
# colunas cruas; o blend
#     wl_target = lambda*result_prob + (1-lambda)*sigmoid(cp/320)
# e feito no TREINO, com lambda POR DATASET (bloco DATASETS no topo de
# train/train_nnue.py). Assar o alvo aqui congelaria lambda no momento da
# geracao, justamente o parametro que se quer variar entre geracoes.

# ── Paralelismo e tamanho de chunk ────────────────────────────────────────
N_WORKERS          = max(1, os.cpu_count() - 1)
BATCH_SIZE         = 5_000
MAX_ACTIVE_BATCHES = N_WORKERS * 2
CHUNK_SIZE_SAVE    = 10_000

# ─────────────────────────────────────────────────────────────────────────
# Valores de peças para filtros de captura (centipawns)
PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   20000,
}
EQUAL_CAPTURE_TOLERANCE = 50
EPD_SAMPLE_LINES = 500

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

def board_to_bin_record(board: chess.Board, cp_white: int,
                         result_white: Optional[float]) -> np.ndarray:
    """Converte chess.Board + avaliação white-POV para um registro SAMPLE_DTYPE.

    `result_white` is the WHITE-relative game-outcome PROBABILITY on the
    0.0/0.5/1.0 scale (CLAUDE.md rule 10 — the same convention as the
    `result` column this script emits everywhere else), NOT a PGN string
    ('1-0'/'0-1'/'1/2-1/2'). This is the internal record's canonical
    `result` representation (see `_RESULT_TO_PROB_OUT` in the worker), so
    callers must not pass a PGN string here.
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
    rec[0]["castling"] = 0   # sem direitos de roque em posições filtradas
    rec[0]["ep_file"]  = board.ep_square % 8 if board.ep_square is not None else 8

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


def bin_record_to_fen(rec) -> str:
    """Converte um registro SAMPLE_DTYPE de volta para FEN (para filtros chess)."""
    _CODE_TO_PIECE = {}
    for pt, code in _CHESS_TYPE_TO_ZCHEZZ.items():
        _CODE_TO_PIECE[8  | code] = chess.Piece(pt, chess.WHITE)
        _CODE_TO_PIECE[16 | code] = chess.Piece(pt, chess.BLACK)

    board = chess.Board(fen=None)
    mailbox = rec["board"]
    for zsq in range(64):
        code = int(mailbox[zsq])
        if code == 0:
            continue
        piece = _CODE_TO_PIECE.get(code)
        if piece:
            sq_w = zsq ^ 56
            board.set_piece_at(sq_w, piece)

    board.turn = chess.WHITE if rec["stm"] == 0 else chess.BLACK
    board.castling_rights = chess.BB_EMPTY
    ep_f = int(rec["ep_file"])
    board.ep_square = None
    board.halfmove_clock = int(rec["rule50"])
    board.fullmove_number = 1
    return board.fen()


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS GERAIS
# ═══════════════════════════════════════════════════════════════════════════

def fen_key(fen: str) -> str:
    return ' '.join(fen.split()[:4])

def format_time(s: float) -> str:
    if s < 60:   return f"{s:.0f}s"
    if s < 3600: return f"{s/60:.1f}m"
    return f"{s/3600:.1f}h"

def result_to_wdl(res_str: Optional[str]) -> float:
    if not res_str: return 0.5
    if '1-0'  in res_str: return 1.0
    if '0-1'  in res_str: return 0.0
    if '1/2'  in res_str: return 0.5
    return 0.5

_RESULT_TO_PROB_OUT = {'1-0': 1.0, '1/2-1/2': 0.5, '0-1': 0.0}
_PROB_OUT_TO_RESULT = {1.0: '1-0', 0.5: '1/2-1/2', 0.0: '0-1'}

def normalize_result_to_prob(raw) -> Optional[float]:
    """Normalize a source's raw `result` field to the canonical WHITE-
    relative probability (0.0/0.5/1.0/None) used everywhere downstream in
    this script.

    Different SRC_TYPEs disagree on what `result` looks like on the wire:
      - epd / bin parsers hand back a PGN string ('1-0'/'0-1'/'1/2-1/2').
      - parquet already stores the numeric 0.0/0.5/1.0 WHITE-relative
        probability directly (see module docstring's LABEL CONVENTION
        table: "result | ... | parquet: 0.0/0.5/1.0").
    Without this normalization, a numeric parquet `result` silently fails
    to match `_RESULT_TO_PROB_OUT`'s string keys and is coerced to None —
    i.e. every parquet-sourced row quietly loses its game outcome. Doing
    the conversion in exactly one place means every SRC_TYPE, present and
    future, is handled the same way from here on.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        return _RESULT_TO_PROB_OUT.get(raw, None)
    try:
        f = float(raw)
    except (TypeError, ValueError):
        return None
    if f >= 0.99:
        return 1.0
    if f <= 0.01:
        return 0.0
    if abs(f - 0.5) < 0.01:
        return 0.5
    return None

def prob_to_result_str(result_white: Optional[float]) -> str:
    """Inverse of _RESULT_TO_PROB_OUT: WHITE-relative 0.0/0.5/1.0 -> PGN string.
    Used only for output formats (epd) that want the human-readable PGN
    result; the internal `result` field stays numeric everywhere else."""
    if result_white is None:
        return ''
    return _PROB_OUT_TO_RESULT.get(round(float(result_white), 1), '')


def result_str_from_cp(cp_white: float) -> str:
    if cp_white >  50: return '1-0'
    if cp_white < -50: return '0-1'
    return '1/2-1/2'

def _progress_path(out_dir: str) -> str:
    return os.path.join(out_dir, 'progress.json')

def load_progress(out_dir: str) -> Dict[str, int]:
    p = _progress_path(out_dir)
    if os.path.exists(p):
        try:
            with open(p, 'r') as f: return json.load(f)
        except: pass
    return {}

def save_progress(out_dir: str, consumed: Dict[str, int]):
    p   = _progress_path(out_dir)
    tmp = p + '.tmp'
    try:
        with open(tmp, 'w') as f: json.dump(consumed, f, indent=2)
        os.replace(tmp, p)
    except Exception as e:
        print(f"  [Warning] Could not save progress: {e}")


# ═══════════════════════════════════════════════════════════════════════════
#  FILTROS
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
#  PARSERS DE ENTRADA
# ═══════════════════════════════════════════════════════════════════════════

def _extract_op(ops_str: str, key: str):
    m = re.search(rf'\b{key}\s+"([^"]*)"', ops_str)
    if m: return m.group(1).strip()
    m = re.search(rf'\b{key}\s+([^;]+)', ops_str)
    if m: return m.group(1).strip()
    return None

def parse_epd_line(line: str):
    line = line.strip()
    if not line or line.startswith(('#', '%')): return None
    parts = line.split()
    if len(parts) < 4: return None
    fen     = ' '.join(parts[:4]) + ' 0 1'
    ops_str = ' '.join(parts[4:])

    res = _extract_op(ops_str, 'c0') or _extract_op(ops_str, 'result')
    cp_val = _extract_op(ops_str, 'ce') or _extract_op(ops_str, 'c1')
    cp = None
    if cp_val is not None:
        try: cp = float(cp_val)
        except: pass

    return {'fen': fen, 'result': res, 'id': _extract_op(ops_str, 'id'), 'cp': cp}


def estimate_source_size(in_dir: str, src_type: str) -> Optional[int]:
    if not os.path.exists(in_dir): return None
    try:
        p = pathlib.Path(in_dir)
        if src_type == 'parquet':
            import pyarrow.parquet as pq
            files = [str(f) for f in p.rglob('*.parquet')]
            return sum(pq.ParquetFile(f).metadata.num_rows for f in files) or None
        elif src_type == 'epd':
            files = [str(f) for f in p.rglob('*.epd')]
            if not files: return None
            total_bytes  = sum(os.path.getsize(f) for f in files)
            sampled_bytes = sampled_lines = 0
            for fpath in sorted(files):
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as fh:
                    for line in fh:
                        s = line.strip()
                        if s and not s.startswith(('#', '%')):
                            sampled_bytes += len(line.encode('utf-8', errors='ignore'))
                            sampled_lines += 1
                            if sampled_lines >= EPD_SAMPLE_LINES: break
                if sampled_lines >= EPD_SAMPLE_LINES: break
            if sampled_lines == 0: return None
            return int(total_bytes / (sampled_bytes / sampled_lines))
        elif src_type == 'bin':
            files = [str(f) for f in p.rglob('*.bin')]
            total = sum(os.path.getsize(f) for f in files)
            return total // SAMPLE_DTYPE.itemsize if total else None
    except: return None


def get_positions_generator(in_dir: str, src_type: str, limit, resume_consumed: int):
    """Gerador unificado: yield (rec_dict, consumed_count)."""
    import pyarrow.parquet as pq

    if not os.path.exists(in_dir):
        print(f"  [Warning] Input not found: {in_dir}")
        return

    skip = resume_consumed
    p    = pathlib.Path(in_dir)

    # ── EPD ─────────────────────────────────────────────────────────────────
    if src_type == 'epd':
        files = sorted(str(f) for f in p.rglob('*.epd'))
        if not files: return

        if limit == 'all':
            consumed = 0
            for fpath in files:
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as fh:
                    for line in fh:
                        rec = parse_epd_line(line)
                        if not rec: continue
                        if consumed < skip:
                            consumed += 1; continue
                        consumed += 1
                        yield rec, consumed
        else:
            # Reservoir sampling para limite
            random.shuffle(files)
            reservoir: List[Dict] = []
            raw_count = 0
            for fpath in files:
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as fh:
                    for line in fh:
                        rec = parse_epd_line(line)
                        if not rec: continue
                        raw_count += 1
                        if len(reservoir) < limit:
                            reservoir.append(rec)
                        else:
                            j = random.randrange(raw_count)
                            if j < limit: reservoir[j] = rec
            random.shuffle(reservoir)
            consumed = 0
            for rec in reservoir:
                if consumed < skip: consumed += 1; continue
                consumed += 1
                yield rec, consumed

    # ── Parquet ─────────────────────────────────────────────────────────────
    elif src_type == 'parquet':
        files = [str(f) for f in p.rglob('*.parquet')]
        if not files: return
        random.shuffle(files)
        per_file_limit = math.ceil(limit / len(files)) if limit != 'all' else None
        consumed = count = 0

        for fpath in files:
            if limit != 'all' and count >= limit: break
            try:
                pf   = pq.ParquetFile(fpath)
                cols = [c for c in ['fen','cp','wdl','result','id'] if c in pf.schema_arrow.names]
                file_count = 0
                for batch in pf.iter_batches(batch_size=100_000, columns=cols):
                    col_lists = {col: batch.column(col).to_pylist() for col in cols}
                    for i in range(batch.num_rows):
                        if limit != 'all' and count >= limit: break
                        if consumed < skip: consumed += 1; continue
                        consumed += 1; count += 1; file_count += 1
                        yield {col: col_lists[col][i] for col in cols}, consumed
                    if limit != 'all' and count >= limit: break
                    if per_file_limit and file_count >= per_file_limit: break
            except Exception as e:
                print(f"  [Warning] Error reading {fpath}: {e}")

    # ── BIN ──────────────────────────────────────────────────────────────────
    elif src_type == 'bin':
        files = sorted(str(f) for f in p.rglob('*.bin'))
        if not files: return

        _CODE_TO_PIECE = {}
        for pt, code in _CHESS_TYPE_TO_ZCHEZZ.items():
            _CODE_TO_PIECE[8  | code] = chess.Piece(pt, chess.WHITE)
            _CODE_TO_PIECE[16 | code] = chess.Piece(pt, chess.BLACK)

        consumed = count = 0
        for fpath in files:
            fsize = os.path.getsize(fpath)
            if fsize % SAMPLE_DTYPE.itemsize != 0:
                print(f"  [Warning] {fpath}: tamanho inválido para .bin, pulando")
                continue
            arr = np.memmap(fpath, dtype=SAMPLE_DTYPE, mode='r')
            for i in range(len(arr)):
                if limit != 'all' and count >= limit: break
                if consumed < skip: consumed += 1; continue

                raw = arr[i]
                try:
                    # Reconstruir board para checar validade
                    board = chess.Board(fen=None)
                    for zsq in range(64):
                        code = int(raw["board"][zsq])
                        if code == 0: continue
                        piece = _CODE_TO_PIECE.get(code)
                        if piece: board.set_piece_at(zsq ^ 56, piece)
                    board.turn = chess.WHITE if raw["stm"] == 0 else chess.BLACK
                    board.castling_rights = chess.BB_EMPTY
                    board.halfmove_clock  = int(raw["rule50"])
                    board.fullmove_number = 1
                    fen = board.fen()
                except Exception:
                    consumed += 1; continue

                # eval_cp no .bin é STM-relative → converter para White-POV
                cp_stm   = int(raw["eval_cp"])
                stm_fac  = 1 if raw["stm"] == 0 else -1
                cp_white = cp_stm * stm_fac

                gr = int(raw["game_result"])
                if   gr > 0: result_str = '1-0' if raw["stm"] == 0 else '0-1'
                elif gr < 0: result_str = '0-1' if raw["stm"] == 0 else '1-0'
                else:        result_str = '1/2-1/2'

                rec = {
                    'fen': fen, 'cp': float(cp_white),
                    'result': result_str, 'id': '',
                    'wdl': None,
                }
                consumed += 1; count += 1
                yield rec, consumed

            del arr
            if limit != 'all' and count >= limit: break


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
            except: pass
        self._proc = subprocess.Popen(
            [self._sf_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self._w(f"uci")
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
            except: buf[0] = ""
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
            if marker in line: return lines

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
        try: self._w("quit"); self._proc.wait(timeout=3)
        except: pass
        try: self._proc.terminate()
        except: pass


# ═══════════════════════════════════════════════════════════════════════════
#  WORKER
# ═══════════════════════════════════════════════════════════════════════════

def _worker_pipeline(records_batch: List[Dict], use_sf: bool,
                     sf_path: str, sf_nodes: int, sf_hash_mb: int, sf_timeout: float,
                     score_cap: int, filter_score_cap: bool,
                     filter_endgame: bool, endgame_max: int, endgame_no_q: bool,
                     filter_dup: bool, filter_check: bool, filter_terminal: bool,
                     filter_win: bool, filter_equal: bool, filter_sacrifice: bool,
                     filter_quiet: bool) -> Tuple[List[Dict], Dict]:
    """Worker: filtra, avalia com SF se necessário, retorna records prontos."""
    # Injetar globals para is_quiet_advanced / is_endgame_position
    # (workers rodam em processo filho; as globals precisam ser re-setadas)
    import chess
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

    engine = None
    if use_sf:
        try:
            engine = _StockfishEngine(sf_path, sf_nodes, sf_hash_mb, sf_timeout)
        except Exception as e:
            stats['sf_init_error'] = 1
            return results, stats

    try:
        for rec in records_batch:
            fen = rec.get('fen', '')
            if not fen:
                stats['bad_fen'] = stats.get('bad_fen', 0) + 1
                continue

            try:
                board = chess.Board(fen)
            except Exception:
                stats['bad_fen'] = stats.get('bad_fen', 0) + 1
                continue

            # ── Filtro quiet ──────────────────────────────────────────────
            if filter_quiet:
                ok, reason = is_quiet_advanced(board)
                if not ok:
                    stats[reason] = stats.get(reason, 0) + 1
                    continue

            # ── Filtro endgame ────────────────────────────────────────────
            if filter_endgame:
                ok, reason = is_endgame_position(board)
                if not ok:
                    stats[reason] = stats.get(reason, 0) + 1
                    continue

            # ── Avaliação ─────────────────────────────────────────────────
            if use_sf:
                try:
                    cp_white = engine.evaluate(fen)
                except TimeoutError:
                    stats['sf_timeout'] = stats.get('sf_timeout', 0) + 1
                    try: engine.close()
                    except: pass
                    try: engine = _StockfishEngine(sf_path, sf_nodes, sf_hash_mb, sf_timeout)
                    except: engine = None
                    continue
                except Exception:
                    stats['sf_error'] = stats.get('sf_error', 0) + 1
                    continue
                # `result` must stay the REAL game outcome from the source.
                # It used to be synthesised here as result_str_from_cp(cp) —
                # i.e. the game "result" was invented from Stockfish's own
                # evaluation. That is not ground truth, and it would make the
                # trainer's lambda term a second copy of the eval term:
                # lambda*f(cp) + (1-lambda)*f(cp), with lambda doing nothing.
                # If the source has no result (a bare EPD of positions, with
                # no game attached), leave it empty and let the trainer fall
                # back to the cp term and report it.
                result_white = normalize_result_to_prob(rec.get('result'))
            else:
                # Usa cp já existente
                cp_white = None
                if 'cp' in rec and rec['cp'] is not None:
                    try: cp_white = float(rec['cp'])
                    except: pass
                result_white = normalize_result_to_prob(rec.get('result'))

                # NO BLEND HERE. This script emits `cp` and `result` as two
                # INDEPENDENT columns; combining them into a single training
                # target is the TRAINER's job (train/train_nnue.py, per-dataset
                # LAMBDA in the DATASETS block).
                #
                # Why this matters, concretely: the historical datasets already
                # have a `wdl` column that is NOT the game outcome — it is
                # exactly sigmoid(cp/320), a deterministic transform of `cp`
                # (verified numerically: cp=298 -> wdl=0.7173 -> T=320.0). The
                # old blend here read that column as if it were ground truth,
                # so it computed
                #     lambda*sigmoid(cp) + (1-lambda)*sigmoid(cp) = sigmoid(cp)
                # and LAMBDA SILENTLY DID NOTHING. Baking a target into the
                # dataset also freezes lambda at generation time, when it is
                # precisely the knob you want to anneal across generations.
                #
                # A position is kept if it has AT LEAST ONE of cp / result;
                # only positions with neither are dropped. The counters below
                # are reported at the end so a source missing one of them is
                # visible instead of silently halving the available signal.
                has_cp  = cp_white is not None
                has_res = result_white is not None
                if not has_cp and not has_res:
                    stats['missing_cp_and_result'] = stats.get('missing_cp_and_result', 0) + 1
                    continue
                if not has_cp:
                    stats['result_only'] = stats.get('result_only', 0) + 1
                if not has_res:
                    stats['cp_only'] = stats.get('cp_only', 0) + 1

            # ── Score cap ─────────────────────────────────────────────────
            if filter_score_cap and cp_white is not None and abs(cp_white) > score_cap:
                stats['score_cap'] = stats.get('score_cap', 0) + 1
                continue

            results.append({
                'fen':    fen,
                'cp':     cp_white,      # white-relative centipawns, or None
                # WHITE-RELATIVE, same scale as `wdl`: 1.0 White won,
                # 0.5 draw, 0.0 Black won, None unknown. Numeric (not the
                # PGN string) so the trainer never has to parse text and so
                # the blend lam*result + (1-lam)*wdl stays a convex sum on
                # one scale. See CLAUDE.md rule 10.
                'result': result_white,
                'id':     rec.get('id', ''),
                'board_obj': board,   # passado internamente para bin writer
            })

    finally:
        if engine:
            try: engine.close()
            except: pass

    return results, stats


# ═══════════════════════════════════════════════════════════════════════════
#  GRAVADORES DE SAÍDA
# ═══════════════════════════════════════════════════════════════════════════

def _write_chunk(rows: List[Dict], out_dir: str, chunk_idx: int,
                 output_formats: List[str]):
    """Salva chunk em um ou mais formatos."""
    clean = [{k: v for k, v in r.items() if k != 'board_obj'} for r in rows]

    if 'parquet' in output_formats:
        path = os.path.join(out_dir, f'chunk_{chunk_idx:04d}.parquet')
        pd.DataFrame(clean).to_parquet(path, index=False)

    if 'epd' in output_formats:
        path = os.path.join(out_dir, f'chunk_{chunk_idx:04d}.epd')
        with open(path, 'a', encoding='utf-8') as fh:
            for r in clean:
                fen_parts = r['fen'].split()
                epd = ' '.join(fen_parts[:4])
                cp_str  = f'{int(r["cp"])}' if r['cp'] is not None else '0'
                res_str = prob_to_result_str(r.get('result'))
                fh.write(f'{epd} ce {cp_str}; c0 "{res_str}";\n')

    if 'bin' in output_formats:
        path = os.path.join(out_dir, f'chunk_{chunk_idx:04d}.bin')
        bin_recs = []
        for r in rows:
            fen = r['fen']
            board = r.get('board_obj')
            if board is None:
                try: board = chess.Board(fen)
                except: continue
            cp_w   = int(r['cp']) if r['cp'] is not None else 0
            # NOTE: must NOT use `r.get('result') or None`-style fallback —
            # 0.0 (Black won, WHITE-relative) is falsy in Python and would
            # be silently coerced to "unknown" by an `or` chain.
            res_white = r.get('result', None)
            bin_recs.append(board_to_bin_record(board, cp_w, res_white))
        if bin_recs:
            arr = np.concatenate(bin_recs)
            with open(path, 'ab') as fh:
                fh.write(arr.tobytes())


# ═══════════════════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(in_dir: str, out_dir: str):
    filter_quiet = (FILTER_IN_CHECK or FILTER_TERMINAL or
                    FILTER_WIN_CAPTURE or FILTER_EQUAL_CAPTURE or FILTER_SACRIFICE)

    print(f"\n{'='*62}")
    print(f"  Pipeline: {os.path.basename(in_dir)} → {os.path.basename(out_dir)}")
    print(f"  Stockfish : {'ON  (' + str(SF_NODES) + ' nodes)' if USE_STOCKFISH else 'OFF (usando cp/wdl existente)'}")
    print(f"  Formatos  : {OUTPUT_FORMATS}")
    print(f"  Filtros   : quiet={'ON' if filter_quiet else 'OFF'}  endgame={'ON' if FILTER_ENDGAME else 'OFF'}")
    if FILTER_ENDGAME:
        print(f"              max_pieces={ENDGAME_MAX_PIECES}  no_queens={ENDGAME_NO_QUEENS}")
    print(f"{'='*62}")

    os.makedirs(out_dir, exist_ok=True)

    # ── Estimar tamanho ────────────────────────────────────────────────────
    est_size = estimate_source_size(in_dir, SRC_TYPE)
    lim_str  = "all" if LIMIT == 'all' else f"{LIMIT:,}"
    est_str  = f"~{est_size:,}" if est_size else "?"
    total_estimated = None
    if est_size:
        total_estimated = est_size if LIMIT == 'all' else min(est_size, LIMIT)
    print(f"  Fonte: {est_str} registros (limit={lim_str})")

    # ── Resume ─────────────────────────────────────────────────────────────
    seen_fens: set = set()
    existing_chunks = sorted(glob.glob(os.path.join(out_dir, 'chunk_*.parquet')))
    # Também checar bins
    existing_chunks += sorted(glob.glob(os.path.join(out_dir, 'chunk_*.bin')))
    existing_chunks = sorted(set(existing_chunks))
    # Número de chunks = máximo idx + 1
    chunk_nums = []
    for cp in existing_chunks:
        m = re.search(r'chunk_(\d+)', os.path.basename(cp))
        if m: chunk_nums.append(int(m.group(1)))
    start_chunk = (max(chunk_nums) + 1) if chunk_nums else 0

    progress_dict   = load_progress(out_dir)
    pipeline_key    = f"{in_dir}|{SRC_TYPE}"
    resume_consumed = progress_dict.get(pipeline_key, 0)

    if start_chunk > 0:
        print(f"\n  Resumindo do chunk {start_chunk}.")
        if FILTER_DUPLICATES and 'parquet' in OUTPUT_FORMATS:
            import pyarrow.parquet as pq
            print(f"  Carregando FENs vistos ({start_chunk} chunks)...")
            for cp in sorted(glob.glob(os.path.join(out_dir, 'chunk_*.parquet'))):
                try:
                    pf = pq.ParquetFile(cp)
                    for batch in pf.iter_batches(200_000, ['fen']):
                        for fval in batch.column('fen').to_pylist():
                            seen_fens.add(fen_key(fval))
                except Exception as e:
                    print(f"  [Warning] {cp}: {e}")
            print(f"  {len(seen_fens):,} FENs únicos carregados.\n")

    total_saved    = len(seen_fens) if start_chunk > 0 else 0
    total_rej      = {}
    saved_this_run = 0
    sent_this_run  = 0
    pending_save:  List[Dict] = []
    chunk_idx      = start_chunk
    t_start        = time.time()

    print(f"  Workers: {N_WORKERS}  Batch: {BATCH_SIZE:,}  Chunk: {CHUNK_SIZE_SAVE:,}\n")

    # ── Argumentos para worker (passados como primitivos — pickle-safe) ────
    worker_kwargs = dict(
        use_sf          = USE_STOCKFISH,
        sf_path         = SF_PATH,
        sf_nodes        = SF_NODES,
        sf_hash_mb      = SF_HASH_MB,
        sf_timeout      = SF_TIMEOUT_S,
        score_cap       = SCORE_CAP_VALUE,
        filter_score_cap= FILTER_SCORE_CAP,
        filter_endgame  = FILTER_ENDGAME,
        endgame_max     = ENDGAME_MAX_PIECES,
        endgame_no_q    = ENDGAME_NO_QUEENS,
        filter_dup      = FILTER_DUPLICATES,
        filter_check    = FILTER_IN_CHECK,
        filter_terminal = FILTER_TERMINAL,
        filter_win      = FILTER_WIN_CAPTURE,
        filter_equal    = FILTER_EQUAL_CAPTURE,
        filter_sacrifice= FILTER_SACRIFICE,
        filter_quiet    = filter_quiet,
    )

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures: Dict = {}
        producer      = get_positions_generator(in_dir, SRC_TYPE, LIMIT, resume_consumed)
        exhausted     = False
        _buf_recs: List[Dict] = []
        _current_consumed = resume_consumed

        def _flush_buf():
            nonlocal _buf_recs
            if _buf_recs:
                fut = pool.submit(_worker_pipeline, list(_buf_recs), **worker_kwargs)
                futures[fut] = time.time()
                _buf_recs = []

        while not exhausted or futures:

            # 1. Alimentar pool
            while not exhausted and len(futures) < MAX_ACTIVE_BATCHES:
                try:
                    rec, _current_consumed = next(producer)
                    if FILTER_DUPLICATES:
                        k = fen_key(rec['fen'])
                        if k in seen_fens:
                            total_rej['duplicate'] = total_rej.get('duplicate', 0) + 1
                            continue
                        seen_fens.add(k)
                    sent_this_run += 1
                    _buf_recs.append(rec)
                    if len(_buf_recs) >= BATCH_SIZE: _flush_buf()
                except StopIteration:
                    _flush_buf(); exhausted = True; break

            # 2. Drenar futures completos
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

                    while len(pending_save) >= CHUNK_SIZE_SAVE:
                        save_slice   = pending_save[:CHUNK_SIZE_SAVE]
                        pending_save = pending_save[CHUNK_SIZE_SAVE:]
                        _write_chunk(save_slice, out_dir, chunk_idx, OUTPUT_FORMATS)
                        total_saved    += CHUNK_SIZE_SAVE
                        saved_this_run += CHUNK_SIZE_SAVE
                        elapsed         = time.time() - t_start
                        progress_dict[pipeline_key] = _current_consumed
                        save_progress(out_dir, progress_dict)
                        pos_per_sec = saved_this_run / max(elapsed, 1e-9)
                        pass_rate   = (saved_this_run / max(sent_this_run, 1)) * 100
                        eta_str = ''
                        if pos_per_sec > 0 and total_estimated:
                            rem   = max(0, total_estimated - _current_consumed)
                            eta_s = (rem * pass_rate / 100) / pos_per_sec
                            eta_str = f'ETA ~{format_time(eta_s)}'
                        print(
                            f'  Chunk {chunk_idx:04d} | '
                            f'+{CHUNK_SIZE_SAVE:,} | '
                            f'Total: {total_saved:,} | '
                            f'Pass: {pass_rate:.1f}% | '
                            f'{pos_per_sec:.0f} pos/s'
                            + (f' | {eta_str}' if eta_str else '')
                        )
                        chunk_idx += 1
                        gc.collect()

        # 3. Flush restante
        if pending_save:
            _write_chunk(pending_save, out_dir, chunk_idx, OUTPUT_FORMATS)
            total_saved    += len(pending_save)
            saved_this_run += len(pending_save)
            progress_dict[pipeline_key] = _current_consumed
            save_progress(out_dir, progress_dict)
            print(f'  Final chunk {chunk_idx:04d} | +{len(pending_save):,} | Total: {total_saved:,}')

    elapsed = time.time() - t_start
    print(f'\n  DONE  — {saved_this_run:,} salvos  |  {format_time(elapsed)}')
    print(f'  Rejeições: {total_rej}')


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    if len(IN_DIRS) != len(OUT_DIRS):
        print("Erro: IN_DIRS e OUT_DIRS devem ter o mesmo comprimento.")
        return

    if USE_STOCKFISH and not os.path.isfile(SF_PATH):
        print(f"Erro: Stockfish não encontrado em: {SF_PATH}")
        return

    for in_dir, out_dir in zip(IN_DIRS, OUT_DIRS):
        run_pipeline(in_dir, out_dir)


if __name__ == '__main__':
    main()
