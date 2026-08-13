"""
generate_endgames.py  (v4)
──────────────────────────
Gera posições sintéticas de endgame, avalia com Stockfish e salva em
um ou mais formatos (parquet, bin, epd). Filtros quiet e score-cap
são individualmente configuráveis via variáveis no topo do arquivo.

BUG FIX 1 — is_check() → was_into_check() (v3)
BUG FIX 2 — mirror() herda castling rights (v3)
NOVO (v4) — OUTPUT_FORMATS: ['parquet'], ['bin'], ['parquet','bin'], etc.
            Filtros totalmente configuráveis por variáveis de topo.

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

import os, re, sys, time, gc, math, random, threading, json, glob
import chess
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
import subprocess
from typing import List, Dict, Tuple, Optional

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG  ← edite aqui
# ═══════════════════════════════════════════════════════════════════════════

SF_PATH  = r"engine\stockfish_fast\stockfish\stockfish-windows-x86-64-avxvnni.exe"
OUT_DIR  = r"data\synthetic_endgame_cp_sf_filter_data20260414"

# ── Formatos de saída ──────────────────────────────────────────────────────
# Qualquer subconjunto de: 'parquet', 'bin', 'epd'
OUTPUT_FORMATS = ['parquet', 'bin']

# ── Stockfish ─────────────────────────────────────────────────────────────
SF_NODES     = 1_000_000
SF_HASH_MB   = 16
SF_TIMEOUT_S = 60

# ── Paralelismo ────────────────────────────────────────────────────────────
WORKERS          = max(1, os.cpu_count() - 1)
BATCH_SIZE         = 2_000
MAX_ACTIVE_BATCHES = WORKERS * 2
CHUNK_SIZE_SAVE    = 20_000

POSITIONS_PER_GROUP = 30_000

# State file — persists shuffle seed and progress for safe resume
# STATE_FILE is derived from OUT_DIR at USE time (see state_file()), because
# --out can change OUT_DIR after this block runs.
def state_file() -> str:
    """Resume state, always next to the CURRENT OUT_DIR."""
    return os.path.join(OUT_DIR, "progress.json")

# ── WDL curve ──────────────────────────────────────────────────────────────
CP_MAX_MATE = 3000
K_DECAY     = 0.07
# cp -> wdl temperature. MUST stay 320: the same number appears in nnue.c's
# output scale and in train_nnue.py. Changing it here alone rescales every
# label this script writes relative to the rest of the corpus.
CP_TO_WDL_T = 320

# ── Filtros quiet (cada um individualmente configurável) ───────────────────
FILTER_IN_CHECK      = True   # rejeita lado-a-mover em xeque
FILTER_WIN_CAPTURE   = True   # rejeita captura ganhante disponível
FILTER_EQUAL_CAPTURE = True   # rejeita captura igual disponível (pxp isento)
FILTER_SACRIFICE     = False  # checar capturas do adversário também
FILTER_SCORE_CAP     = False  # rejeitar posições com |cp| > SCORE_CAP_VALUE
SCORE_CAP_VALUE      = 3000

# ── Constantes de peças ────────────────────────────────────────────────────
PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   20000,
}
EQUAL_CAPTURE_TOLERANCE = 50

# ── Restrição de grupos (opcional) ─────────────────────────────────────────
ONLY_GROUPS = []      # [] = todos os ENDGAME_GROUPS; senão, lista de labels
                      # ("KQK", "KRK", ...) a gerar nesta execução
DRY_RUN     = False   # True = mostra o plano (grupos, contagens, saídas) e sai

# ═══════════════════════════════ COMMAND LINE ════════════════════════════════
# O bloco CONFIG acima É a interface: `python train/labeling/generate_endgames.py`
# roda exatamente o que ele diz. Cada constante também é uma flag que a
# sobrescreve, e `--show-config` imprime a configuração resolvida e sai.
# Ver utils/cliconf.py e CLAUDE.md regra 8.
# ═════════════════════════════════════════════════════════════════════════════
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "utils"))
from cliconf import force_utf8_stdio, override_from_cli  # noqa: E402

force_utf8_stdio()

CLI = [
    ("OUT_DIR",             "--out",            str,  "output folder"),
    ("OUTPUT_FORMATS",      "--out-format",     list, "output format: parquet | bin | epd (repeatable)"),
    ("POSITIONS_PER_GROUP", "--per-group",      int,  "positions generated per material group"),
    ("ONLY_GROUPS",         "--group",          list, "generate ONLY this material label (repeatable)"),
    ("DRY_RUN",             "--dry-run",        bool, "print the plan and exit, generating nothing"),
    ("SF_PATH",             "--sf-path",        str,  "Stockfish binary used for labelling"),
    ("SF_NODES",            "--sf-nodes",       int,  "nodes per position"),
    ("SF_HASH_MB",          "--sf-hash-mb",     int,  "hash per Stockfish process"),
    ("SF_TIMEOUT_S",        "--sf-timeout",     float,"per-position timeout, seconds"),
    ("WORKERS",           "--workers",        int,  "worker processes"),
    ("BATCH_SIZE",          "--batch-size",     int,  "positions per worker task"),
    ("MAX_ACTIVE_BATCHES",  "--max-active-batches", int, "in-flight worker tasks"),
    ("CHUNK_SIZE_SAVE",     "--chunk-size",     int,  "rows per output chunk"),
    ("CP_MAX_MATE",         "--cp-max-mate",    int,  "|cp| assigned to a forced mate"),
    ("K_DECAY",             "--k-decay",        float,"distance-to-mate decay in the cp curve"),
    ("FILTER_IN_CHECK",     "--filter-in-check",     bool, "reject positions in check"),
    ("FILTER_WIN_CAPTURE",  "--filter-win-capture",  bool, "reject positions with a winning capture"),
    ("FILTER_EQUAL_CAPTURE","--filter-equal-capture",bool, "reject positions with an equal capture"),
    ("FILTER_SACRIFICE",    "--filter-sacrifice",    bool, "also check the opponent's captures"),
    ("FILTER_SCORE_CAP",    "--filter-score-cap",    bool, "reject |cp| above --score-cap"),
    ("SCORE_CAP_VALUE",     "--score-cap",      int,  "|cp| ceiling for --filter-score-cap"),
]

# ═══════════════════════════════════════════════════════════════════════════
#  MATERIAL COMBINATIONS
# ═══════════════════════════════════════════════════════════════════════════
# Each entry: (white_extras, black_extras, label, material_floor_cp)
# Kings are implicit. Mirror (colors swapped) is generated automatically.
# material_floor_cp: minimum |cp| assigned to any winning position here.

P = chess.PAWN
N = chess.KNIGHT
B = chess.BISHOP
R = chess.ROOK
Q = chess.QUEEN

ENDGAME_GROUPS: List[Tuple[List, List, str, int]] = [

    # ══════════════════════════════════════════════════════════════════════
    # 4-MAN  (K + up to 2 non-king pieces, total pieces on board = 4)
    # ══════════════════════════════════════════════════════════════════════

    ([Q],    [],   "KQK",    900),   # queen vs bare king
    ([R],    [],   "KRK",    500),   # rook vs bare king
    ([P],    [],   "KPK",    100),   # pawn vs bare king — often drawn
    ([B, B], [],   "KBBK",   660),   # two bishops vs bare king — theoretical win
    ([B, N], [],   "KBNK",   650),   # bishop+knight vs bare king — longest win
    # KNK and KBK are theoretical draws — omitted intentionally
    # KNNk is a draw in almost all positions — omitted
    ([B, P], [],   "KBPKx",  430),   # bishop + pawn vs bare king
    ([N, P], [],   "KNPKx",  420),   # knight + pawn vs bare king

    # ══════════════════════════════════════════════════════════════════════
    # 5-MAN
    # ══════════════════════════════════════════════════════════════════════

    # Two pieces vs bare king
    ([Q, R],  [],  "KQRK",  1400),
    ([R, R],  [],  "KRRK",  1000),
    ([Q, B],  [],  "KQBK",  1230),
    ([Q, N],  [],  "KQNK",  1220),
    ([R, B],  [],  "KRBK",   830),
    ([R, N],  [],  "KRNK",   820),

    # Piece + pawn vs bare king
    ([Q, P],  [],  "KQPKx",  1000),
    ([R, P],  [],  "KRPKx",   600),
    ([B, P, P], [], "KBPPKx",  530),  # bishop + two pawns vs bare king

    # Two pawns vs bare king
    ([P, P],  [],  "KPPKx",   200),

    # Imbalanced: one strong piece vs one weaker
    ([Q],  [R],   "KQKr",    400),
    ([Q],  [B],   "KQKb",    570),
    ([Q],  [N],   "KQKn",    580),
    ([R],  [B],   "KRKb",    170),   # often drawn
    ([R],  [N],   "KRKn",    180),   # often drawn

    # Piece vs pawn
    ([Q],  [P],   "KQKp",    800),
    ([R],  [P],   "KRKp",    400),
    ([B],  [P],   "KBKp",     70),   # often drawn
    ([N],  [P],   "KNKp",     50),   # often drawn

    # Pawn endings
    ([P],  [P],   "KPKp",      0),   # often drawn — SF decides
    ([P, P], [P], "KPPKp",   100),   # two pawns vs one

    # Two bishops vs pawn
    ([B, B], [P], "KBBKp",   560),

    # ══════════════════════════════════════════════════════════════════════
    # 6-MAN
    # ══════════════════════════════════════════════════════════════════════

    # Two pieces + pawn vs bare king
    ([Q, R, P],    [],  "KQRPKx",  1500),
    ([R, R, P],    [],  "KRRPKx",  1100),
    ([R, B, P],    [],  "KRBPKx",   930),
    ([R, N, P],    [],  "KRNPKx",   920),
    ([Q, P, P],    [],  "KQPPKx",  1100),
    ([R, P, P],    [],  "KRPPKx",   700),
    ([B, P, P],    [],  "KBPPKx",   430),
    ([N, P, P],    [],  "KNPPKx",   420),

    # Three pawns vs bare king
    ([P, P, P],    [],  "KPPPKx",   300),

    # Piece + pawns vs piece
    ([Q, P],  [R],    "KQPKr",    500),
    ([R, P],  [B],    "KRPKb",    330),
    ([R, P],  [N],    "KRPKn",    320),
    ([R, P],  [P],    "KRPKp",    500),
    ([B, P],  [N],    "KBPKn",     50),   # often drawn
    ([N, P],  [B],    "KNPKb",     50),   # often drawn

    # Rook + minor vs rook
    ([R, B],  [R],    "KRBKr",    330),
    ([R, N],  [R],    "KRNKr",    320),

    # Rook vs bishop/knight + pawn
    ([R],     [B, P], "KRKbp",    170),
    ([R],     [N, P], "KRKnp",    150),

    # Queen vs rook + minor / pawn
    ([Q],     [R, P], "KQKrp",    200),
    ([Q],     [B, B], "KQKbb",    240),
    ([Q],     [N, B], "KQKnb",    230),
    ([Q],     [R, N], "KQKrn",    180),

    # Pawn endings (6-man)
    ([P, P],  [P],    "KPPKp6",   100),   # two vs one (mirror of KPPKp but 6-man context)
    ([P, P],  [P, P], "KPPKpp",     0),   # two vs two — SF decides
    ([P, P, P], [P],  "KPPPKp",   200),   # three vs one

    # ══════════════════════════════════════════════════════════════════════
    # 8-MAN  — King + 3 Pawns vs King + 3 Pawns
    # ══════════════════════════════════════════════════════════════════════

    ([P, P, P], [P, P, P], "KPPPKppp",  0),   # SF decides; highly imbalanced structures
]

# ═══════════════════════════════════════════════════════════════════════════
#  WDL HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def mate_distance_to_cp(mate_dist: int, material_floor: int) -> int:
    dist = max(1, mate_dist)
    cp   = int(CP_MAX_MATE * math.exp(-K_DECAY * (dist - 1)))
    return max(material_floor, cp)


def cp_to_wdl(cp: float) -> float:
    return 1.0 / (1.0 + math.exp(-cp / CP_TO_WDL_T))


# ═══════════════════════════════════════════════════════════════════════════
#  FORMATO .BIN  (contrato cross-language com dataset.py / train_nnue.py)
# ═══════════════════════════════════════════════════════════════════════════
# Zchezz mailbox: zsq 0=a8..63=h1 (sq_w = zsq ^ 56)
# Códigos: WP=9..WK=14, BP=17..BK=22  (COL_W=8, COL_B=16, tipo 1..6)

SAMPLE_DTYPE = np.dtype([
    ("board",       "u1", (64,)),
    ("stm",         "u1"),
    ("rule50",      "u1"),
    ("castling",    "u1"),
    ("ep_file",     "u1"),
    ("eval_cp",     "<i2"),
    ("game_result", "i1"),
    ("move_played", "<u2"),
    ("_pad",        "<u2"),
], align=False)
assert SAMPLE_DTYPE.itemsize == 75

_CHESS_TYPE_TO_ZCHEZZ = {
    chess.PAWN:   1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK:   4,
    chess.QUEEN:  5,
    chess.KING:   6,
}

def board_to_bin_record(board: chess.Board, cp_white: int, result_str: str) -> np.ndarray:
    """Converte board + avaliação white-POV para um registro SAMPLE_DTYPE."""
    rec = np.zeros(1, dtype=SAMPLE_DTYPE)
    mailbox = rec[0]["board"]
    for sq, piece in board.piece_map().items():
        zsq  = sq ^ 56
        col  = 8 if piece.color == chess.WHITE else 16
        mailbox[zsq] = col | _CHESS_TYPE_TO_ZCHEZZ[piece.piece_type]
    rec[0]["stm"]     = 0 if board.turn == chess.WHITE else 1
    rec[0]["rule50"]  = min(board.halfmove_clock, 255)
    rec[0]["castling"] = 0
    rec[0]["ep_file"]  = board.ep_square % 8 if board.ep_square is not None else 8
    stm_factor        = 1 if board.turn == chess.WHITE else -1
    rec[0]["eval_cp"] = int(np.clip(cp_white * stm_factor, -32000, 32000))
    if '1-0' in (result_str or ''):
        gr = 1 if board.turn == chess.WHITE else -1
    elif '0-1' in (result_str or ''):
        gr = -1 if board.turn == chess.WHITE else 1
    else:
        gr = 0
    rec[0]["game_result"] = gr
    rec[0]["move_played"] = 0
    rec[0]["_pad"]        = 0
    return rec


def _write_chunk_endgame(rows: List[Dict], chunk_idx: int):
    """Salva um chunk em todos os OUTPUT_FORMATS configurados."""
    if 'parquet' in OUTPUT_FORMATS:
        path = os.path.join(OUT_DIR, f"chunk_{chunk_idx:04d}.parquet")
        clean = [{k: v for k, v in r.items() if k != 'board_obj'} for r in rows]
        pd.DataFrame(clean).to_parquet(path, index=False)

    if 'epd' in OUTPUT_FORMATS:
        path = os.path.join(OUT_DIR, f"chunk_{chunk_idx:04d}.epd")
        with open(path, 'a', encoding='utf-8') as fh:
            for r in rows:
                fen_parts = r['fen'].split()
                epd = ' '.join(fen_parts[:4])
                cp_s  = str(int(r['cp'])) if r.get('cp') is not None else '0'
                res_s = r.get('result', '') or ''
                id_s = r.get('id', '')
                fh.write(f'{epd} ce {cp_s}; c0 "{res_s}"; id "{id_s}";\n')

    if 'bin' in OUTPUT_FORMATS:
        path = os.path.join(OUT_DIR, f"chunk_{chunk_idx:04d}.bin")
        bin_recs = []
        for r in rows:
            board = r.get('board_obj')
            if board is None:
                try: board = chess.Board(r['fen'])
                except: continue
            cp_w = int(r['cp']) if r.get('cp') is not None else 0
            bin_recs.append(board_to_bin_record(board, cp_w, r.get('result', '')))
        if bin_recs:
            arr = np.concatenate(bin_recs)
            with open(path, 'ab') as fh:
                fh.write(arr.tobytes())


# ═══════════════════════════════════════════════════════════════════════════
#  QUIET FILTER
# ═══════════════════════════════════════════════════════════════════════════

def is_quiet(board: chess.Board) -> Tuple[bool, str]:
    """
    Retorna (True, "") se a posição é quiet; (False, reason) caso contrário.
    Controlado pelas variáveis FILTER_IN_CHECK, FILTER_WIN_CAPTURE,
    FILTER_EQUAL_CAPTURE e FILTER_SACRIFICE definidas no CONFIG.
    """
    if FILTER_IN_CHECK and board.is_check():
        return False, "in_check"

    if FILTER_WIN_CAPTURE or FILTER_EQUAL_CAPTURE:
        for move in board.generate_pseudo_legal_captures():
            attacker = board.piece_at(move.from_square)
            if attacker is None:
                continue
            att_val = PIECE_VALUES.get(attacker.piece_type, 0)
            if board.is_en_passant(move):
                vic_type, vic_val = chess.PAWN, 100
            else:
                victim = board.piece_at(move.to_square)
                if victim is None:
                    continue
                vic_type = victim.piece_type
                vic_val  = PIECE_VALUES.get(vic_type, 0)
            # pxp: não filtra como equal capture
            if attacker.piece_type == chess.PAWN and vic_type == chess.PAWN:
                continue
            diff = vic_val - att_val
            if FILTER_WIN_CAPTURE   and diff >  EQUAL_CAPTURE_TOLERANCE:
                return False, "winning_capture"
            if FILTER_EQUAL_CAPTURE and abs(diff) <= EQUAL_CAPTURE_TOLERANCE:
                return False, "equal_capture"

    if FILTER_SACRIFICE:
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


# ═══════════════════════════════════════════════════════════════════════════
#  POSITION GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

def _place_pieces(
    white_pieces: List[int],
    black_pieces: List[int],
    rng: random.Random,
    max_attempts: int = 400,
) -> Optional[chess.Board]:
    n_pieces = 2 + len(white_pieces) + len(black_pieces)
    if n_pieces > 64:
        return None

    for _ in range(max_attempts):
        squares = rng.sample(range(64), n_pieces)
        wk_sq, bk_sq = squares[0], squares[1]

        if chess.square_distance(wk_sq, bk_sq) < 2:
            continue

        board = chess.Board(fen=None)
        board.set_piece_at(wk_sq, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(bk_sq, chess.Piece(chess.KING, chess.BLACK))

        sq_iter = iter(squares[2:])
        valid   = True

        for pt in white_pieces:
            sq = next(sq_iter)
            if pt == P and chess.square_rank(sq) in (0, 7):
                valid = False; break
            board.set_piece_at(sq, chess.Piece(pt, chess.WHITE))
        if not valid:
            continue

        for pt in black_pieces:
            sq = next(sq_iter)
            if pt == P and chess.square_rank(sq) in (0, 7):
                valid = False; break
            board.set_piece_at(sq, chess.Piece(pt, chess.BLACK))
        if not valid:
            continue

        board.turn            = rng.choice([chess.WHITE, chess.BLACK])
        board.castling_rights = chess.BB_EMPTY
        board.ep_square       = None

        if not board.is_valid():        continue
        if board.is_game_over():        continue
        if board.was_into_check():      continue   # BUG FIX: reject illegal previous move, not stm-in-check

        return board

    return None


def generate_all_records(shuffle_seed: int) -> List[Dict]:
    """
    Build the complete list of positions for all groups deterministically.
    Same shuffle_seed always produces the same list in the same order.
    """
    all_records: List[Dict] = []

    for g_idx, (wp, bp, label, _floor) in enumerate(ENDGAME_GROUPS):
        rng      = random.Random(g_idx * 99991 + shuffle_seed)
        records  = []
        mirrors  = []
        attempts = 0

        while len(records) < POSITIONS_PER_GROUP and attempts < POSITIONS_PER_GROUP * 60:
            attempts += 1
            board = _place_pieces(wp, bp, rng)
            if board:
                records.append({"fen": board.fen(), "group": label})

        for rec in records:
            try:
                m = chess.Board(rec["fen"]).mirror()
                # mirror() re-parses the FEN so castling_rights survive even
                # though we cleared them on the source board.  Any non-empty
                # rights on a synthetic endgame position cause is_castling()
                # (called inside generate_legal_moves → is_game_over) to crash.
                m.castling_rights = chess.BB_EMPTY
                m.ep_square       = None
                if m.is_valid() and not m.is_game_over() and not m.was_into_check():
                    mirrors.append({"fen": m.fen(), "group": label + "_m"})
            except Exception:
                pass

        all_records.extend(records)
        all_records.extend(mirrors)

    # Round-robin interleave across groups before the global shuffle so that
    # a partial run has balanced coverage rather than completing early groups first.
    # Re-build from per-group buckets.
    group_buckets = []
    for wp, bp, label, _floor in ENDGAME_GROUPS:
        recs_in_group = [r for r in all_records
                         if r["group"] in (label, label + "_m")]
        group_buckets.append(recs_in_group)

    interleaved: List[Dict] = []
    iters = [iter(b) for b in group_buckets]
    active = list(range(len(iters)))
    while active:
        next_active = []
        for i in active:
            rec = next(iters[i], None)
            if rec is not None:
                interleaved.append(rec)
                next_active.append(i)
        active = next_active

    all_records = interleaved

    # Deterministic shuffle — same seed = same order on every restart
    random.Random(shuffle_seed).shuffle(all_records)
    return all_records


# ═══════════════════════════════════════════════════════════════════════════
#  STOCKFISH ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class StockfishEngine:
    def __init__(self):
        self.proc     = None
        self._re_cp   = re.compile(r'score cp (-?\d+)')
        self._re_mate = re.compile(r'score mate (-?\d+)')
        self._start()

    def _start(self):
        if self.proc is not None:
            try: self.proc.terminate()
            except: pass
        self.proc = subprocess.Popen(
            [SF_PATH],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self._w("uci")
        self._w(f"setoption name Hash value {SF_HASH_MB}")
        self._w("setoption name Threads value 1")
        self._w("isready")
        self._read_until("readyok", timeout=10)

    def _w(self, cmd: str):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _readline(self, timeout: float) -> str:
        buf  = [""]
        done = threading.Event()
        def _r():
            try:   buf[0] = self.proc.stdout.readline()
            except: buf[0] = ""
            done.set()
        threading.Thread(target=_r, daemon=True).start()
        return buf[0] if done.wait(timeout) else ""

    def _read_until(self, marker: str, timeout: float = SF_TIMEOUT_S) -> List[str]:
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

    def evaluate_full(self, fen: str, nodes: int) -> Tuple[int, Optional[int]]:
        """
        Returns (cp_stm, mate_dist).
          cp_stm    : centipawns from side-to-move POV; ±30000 sentinel for mate
          mate_dist : None = no forced mate found
                      positive N = stm mates in N moves
                      negative N = stm is mated in N moves
        """
        self._w(f"position fen {fen}")
        self._w(f"go nodes {nodes}")

        cp_stm    = 0
        mate_dist = None

        for line in self._read_until("bestmove"):
            if "info" not in line or "score" not in line:
                continue
            m = self._re_mate.search(line)
            if m:
                md        = int(m.group(1))
                mate_dist = md
                cp_stm    = 30_000 if md > 0 else -30_000
            else:
                m = self._re_cp.search(line)
                if m:
                    cp_stm = int(m.group(1))

        return cp_stm, mate_dist

    def close(self):
        try: self._w("quit"); self.proc.wait(timeout=3)
        except: pass
        try: self.proc.terminate()
        except: pass


# ═══════════════════════════════════════════════════════════════════════════
#  GROUP FLOOR LOOKUP  (module-level so workers can access it after fork)
# ═══════════════════════════════════════════════════════════════════════════

_GROUP_FLOOR: Dict[str, int] = {}
for _wp, _bp, _lbl, _fl in ENDGAME_GROUPS:
    _GROUP_FLOOR[_lbl]        = _fl
    _GROUP_FLOOR[_lbl + "_m"] = _fl


# ═══════════════════════════════════════════════════════════════════════════
#  WORKER
# ═══════════════════════════════════════════════════════════════════════════

def _worker_evaluate(records: List[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
    engine  = StockfishEngine()
    results: List[Dict] = []
    stats:   Dict[str, int] = {}

    try:
        for rec in records:
            fen   = rec["fen"]
            group = rec["group"]
            floor = _GROUP_FLOOR.get(group, 0)

            # Quiet filter — applied before SF to avoid wasting engine time
            try:
                board = chess.Board(fen)
                ok, reason = is_quiet(board)
            except Exception:
                stats["bad_fen"] = stats.get("bad_fen", 0) + 1
                continue
            if not ok:
                stats[reason] = stats.get(reason, 0) + 1
                continue

            try:
                cp_stm, mate_dist = engine.evaluate_full(fen, SF_NODES)
            except TimeoutError:
                stats["engine_timeout"] = stats.get("engine_timeout", 0) + 1
                try: engine.close()
                except: pass
                engine = StockfishEngine()
                continue
            except Exception:
                stats["engine_error"] = stats.get("engine_error", 0) + 1
                continue

            # White-POV cp
            side         = fen.split()[1]
            cp_white_raw = cp_stm if side == "w" else -cp_stm

            # Apply mate-distance curve for forced-mate positions
            if mate_dist is not None:
                cp_mate  = mate_distance_to_cp(abs(mate_dist), floor)
                cp_white = cp_mate if cp_white_raw > 0 else -cp_mate
            else:
                cp_white = cp_white_raw

            # Score cap filter (opcional)
            if FILTER_SCORE_CAP and abs(cp_white) > SCORE_CAP_VALUE:
                stats["score_cap"] = stats.get("score_cap", 0) + 1
                continue

            # `result` is left EMPTY on purpose. These positions are
            # GENERATED and labelled by an engine — no game was ever played
            # from them, so there is no game outcome to record.
            #
            # This used to synthesise one from the evaluation:
            #     result = "1-0" if cp_white > 50 else "0-1" if cp_white < -50 else "1/2-1/2"
            # which is not ground truth, it is the labelling engine's own
            # opinion wearing a result's clothes. The damage is specific and
            # measurable: the training target is
            #     lam*result + (1-lam)*sigmoid(cp/320),
            # so a result derived from cp makes both terms functions of cp
            # and lambda becomes a no-op — while looking like it works. The
            # same bug existed in process_positions.py's Stockfish branch and
            # was removed there; see CLAUDE.md rule 10.
            #
            # Consequence for the datasets already produced with the old
            # code (synthetic_endgame_cp_sf_filter_data20260413/2): their `result` column is NOT an
            # outcome. Keep lam = 0.00 for them — see data/Data.md.
            result = ""

            results.append({
                "fen":      fen,
                "cp":       cp_white,
                "wdl":      cp_to_wdl(cp_white),
                "result":   result,
                "id":       group,
                "board_obj": board,  # passado para _write_chunk_endgame
            })

    finally:
        engine.close()

    return results, stats


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def format_time(s: float) -> str:
    if s < 60:   return f"{s:.0f}s"
    if s < 3600: return f"{s/60:.1f}m"
    h = int(s // 3600); m = int((s % 3600) // 60)
    return f"{h}h{m:02d}m"


def _load_state() -> dict:
    if os.path.exists(state_file()):
        try:
            with open(state_file()) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_state(state: dict):
    with open(state_file(), "w") as f:
        json.dump(state, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # --group restricts ENDGAME_GROUPS for this run. It is applied to the
    # module global because generate_all_records() and the reporting below
    # both read that list; an unknown label is a typo and aborts rather than
    # silently generating the full set.
    global ENDGAME_GROUPS
    if ONLY_GROUPS:
        wanted = set(ONLY_GROUPS)
        unknown = wanted - {lbl for _, _, lbl, _ in ENDGAME_GROUPS}
        if unknown:
            raise SystemExit(f"ABORT: --group named unknown material label(s): "
                             f"{sorted(unknown)}")
        ENDGAME_GROUPS = [g for g in ENDGAME_GROUPS if g[2] in wanted]

    if not DRY_RUN:                 # a dry run must not even create the folder
        os.makedirs(OUT_DIR, exist_ok=True)

    # ── WDL curve preview ─────────────────────────────────────────────────
    print("\n  WDL curve (mate-distance encoding):")
    for d, tgt in [(1,"~0.9999"),(5,"~0.9997"),(10,"~0.9988"),
                   (20,"~0.9916"),(30,"~0.9670"),(50,"~0.844")]:
        cp = mate_distance_to_cp(d, 0)   # floor=0: show the raw curve, not group-specific clamp
        print(f"    Mate {d:2d}  cp={cp:5d}  wdl={cp_to_wdl(cp):.4f}  (target {tgt})")
    print()

    # ── Group summary ─────────────────────────────────────────────────────
    by_men: Dict[int, int] = {}
    for wp, bp, lbl, fl in ENDGAME_GROUPS:
        n = 2 + len(wp) + len(bp)
        by_men[n] = by_men.get(n, 0) + 1
    print(f"  Material groups: {len(ENDGAME_GROUPS)} total  "
          + "  ".join(f"({k}-man: {v})" for k, v in sorted(by_men.items())))
    print(f"  Positions/group: {POSITIONS_PER_GROUP:,} + auto-mirrors\n")

    # ── Resume state ──────────────────────────────────────────────────────
    state        = _load_state()
    shuffle_seed = state.get("shuffle_seed", random.randint(0, 2**31))
    already_done = state.get("records_done", 0)
    chunk_idx    = state.get("chunk_idx", 0)

    if already_done > 0:
        print(f"  Resuming: {already_done:,} records already processed, "
              f"chunk_idx={chunk_idx}.")
    else:
        state["shuffle_seed"] = shuffle_seed
        if not DRY_RUN:
            _save_state(state)
        print(f"  Fresh start. Shuffle seed: {shuffle_seed}")

    # ── Generate all positions (fully deterministic) ──────────────────────
    print("\n  Building position list...")
    t0          = time.time()
    all_records = generate_all_records(shuffle_seed)
    gen_time    = time.time() - t0

    total_positions  = len(all_records)
    total_to_process = total_positions - already_done

    print(f"  {total_positions:,} positions generated in {gen_time:.1f}s\n")
    print(f"  {'Group':<14} {'count':>8}")
    print(f"  {'-'*24}")
    for wp, bp, lbl, fl in ENDGAME_GROUPS:
        cnt = sum(1 for r in all_records if r["group"] in (lbl, lbl+"_m"))
        print(f"  {lbl:<14} {cnt:>8,}")
    print()

    if DRY_RUN:
        print("  DRY RUN — the plan above is all that happens; "
              "no Stockfish is started and nothing is written.")
        return

    if total_to_process <= 0:
        print("  All positions already evaluated. Nothing to do.")
        return

    remaining_records = all_records[already_done:]

    print(f"  To evaluate this run : {total_to_process:,}")
    print(f"  Workers              : {WORKERS}")
    print(f"  Batch size           : {BATCH_SIZE:,}")
    print(f"  Chunk size           : {CHUNK_SIZE_SAVE:,}")
    print(f"  Output               : {OUT_DIR}\n")

    batches = [
        remaining_records[i : i + BATCH_SIZE]
        for i in range(0, len(remaining_records), BATCH_SIZE)
    ]

    t_start        = time.time()
    pending_save:  List[Dict] = []
    saved_this_run = 0
    records_done   = already_done
    total_stats:   Dict[str, int] = {}
    group_counts:  Dict[str, int] = dict(state.get("group_counts", {}))

    print(f"  Starting evaluation ({len(batches)} batches)...\n")

    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures:   dict = {}
        batch_iter      = iter(batches)
        exhausted       = False

        while not exhausted or futures:

            # Fill pool
            while not exhausted and len(futures) < MAX_ACTIVE_BATCHES:
                try:
                    b   = next(batch_iter)
                    fut = pool.submit(_worker_evaluate, b)
                    futures[fut] = len(b)
                except StopIteration:
                    exhausted = True

            if not futures:
                break

            # Drain completed futures
            done_futs = []
            try:
                for f in as_completed(futures, timeout=0.1):
                    done_futs.append(f)
            except FuturesTimeoutError:
                pass

            if not done_futs:
                time.sleep(0.05)
                continue

            for fut in done_futs:
                batch_n = futures.pop(fut)
                try:
                    res_list, res_stats = fut.result()
                except Exception as e:
                    print(f"  [Worker error] {e}")
                    records_done += batch_n
                    continue

                for k, v in res_stats.items():
                    total_stats[k] = total_stats.get(k, 0) + v

                for r in res_list:
                    grp = r["id"]
                    group_counts[grp] = group_counts.get(grp, 0) + 1

                pending_save.extend(res_list)
                records_done += batch_n

                # Flush complete chunks
                while len(pending_save) >= CHUNK_SIZE_SAVE:
                    save_slice   = pending_save[:CHUNK_SIZE_SAVE]
                    pending_save = pending_save[CHUNK_SIZE_SAVE:]

                    _write_chunk_endgame(save_slice, chunk_idx)

                    saved_this_run += CHUNK_SIZE_SAVE
                    elapsed         = time.time() - t_start
                    pos_per_sec     = saved_this_run / max(elapsed, 1e-9)

                    done_in_run  = records_done - already_done
                    remaining    = max(0, total_to_process - done_in_run)
                    eta_s        = remaining / max(pos_per_sec, 1e-9)

                    state["records_done"] = records_done
                    state["chunk_idx"]    = chunk_idx + 1
                    state["group_counts"] = group_counts
                    _save_state(state)

                    gc.collect()
                    print(
                        f"  Chunk {chunk_idx:04d} | "
                        f"+{CHUNK_SIZE_SAVE:,} | "
                        f"{pos_per_sec:.0f} pos/s | "
                        f"Done {done_in_run:,}/{total_to_process:,} | "
                        f"In-flight: {len(futures)} | "
                        f"ETA: {format_time(eta_s)}"
                    )
                    chunk_idx += 1

        # Final partial chunk
        if pending_save:
            _write_chunk_endgame(pending_save, chunk_idx)
            saved_this_run += len(pending_save)
            state["records_done"] = total_positions
            state["chunk_idx"]    = chunk_idx + 1
            state["group_counts"] = group_counts
            _save_state(state)
            print(f"  Final chunk {chunk_idx:04d} | "
                  f"+{len(pending_save):,} | "
                  f"Total saved this run: {saved_this_run:,}")

    elapsed = time.time() - t_start
    print(f"\n  DONE in {format_time(elapsed)}")
    print(f"  Positions saved this run : {saved_this_run:,}")
    print(f"  Speed                    : {saved_this_run/max(elapsed,1e-9):.0f} pos/s")
    print(f"  Engine stats             : {total_stats}")
    print(f"  Output                   : {OUT_DIR}")
    print(f"  Resume state saved to    : {state_file()}")
    print(f"\n  To add more data, increase POSITIONS_PER_GROUP and re-run.")
    print(f"  Delete {state_file()} to start completely fresh.")


if __name__ == "__main__":
    override_from_cli(globals(), CLI, description=__doc__, prog="generate_endgames.py")
    main()
