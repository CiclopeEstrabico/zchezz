# ═══════════════════════════════════════════════════════════
#  NNUE Training — v188H  (Quantization-Aware Training - Full QAT from Epoch 0)
#
#  Changes from mixtrain3.py:
#    - Architecture unchanged: 799 → 256 → 64 → 1
#    - Starts immediately with full int16/int8 QAT (assuming loaded weights are already clamped)
#    - No warmup phases
#    - Verbose format from mixtrain2
# ═══════════════════════════════════════════════════════════

import os, chess, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json, gc, time, math

BASE_DIR = 'C:/nnue_checkpoints'
CKPT_DIR = f'{BASE_DIR}/checkpoints'
os.makedirs(CKPT_DIR, exist_ok=True)

DATASET_NAME = 'qat_nnu3_v4'   # Nome diferente para não confundir com checkpoints antigos

# ════════════════════════════════════════════════════════════
#  QUANTIZATION CONSTANTS
# ════════════════════════════════════════════════════════════
QA         = 255    # L1 weight/accumulator scale
QB         = 64     # L2/L3 weight scale
RELU_CLIP  = 1.0    # ClippedReLU upper bound (maps to QA in int space)

# ════════════════════════════════════════════════════════════
#  OPÇÕES GLOBAIS DE AMOSTRAGEM
# ════════════════════════════════════════════════════════════

RESAMPLE_EACH_EPOCH = True

# ════════════════════════════════════════════════════════════
#  PERCENTUAIS DE CADA DATASET
#  (0.0 = ignorado | 1.0 = 100%)
# ════════════════════════════════════════════════════════════

TRAIN_PCT_SET1  = 0.00  # Lichess raw WDL                (~113M posições, 17 shards)
TRAIN_PCT_SET2  = 0.00  # Lichess quiet WDL              (~25M posições, 1460 chunks)
TRAIN_PCT_SET3  = 0.35  # Lichess new quiet N 400k           
TRAIN_PCT_SET4  = 0.03  # Lichess new quiet no SF       
TRAIN_PCT_SET5  = 0.02  # extra quiet 5000 nodes            (~5.7M posições, 10 chunks) 
TRAIN_PCT_SET6  = 0.70  # extra-quiet-n60k_sf
TRAIN_PCT_SET7  = 0.35  # extra-quiet-endgames-d12_sf
TRAIN_PCT_SET8  = 0.55  # extra-quiet-endgames-d14_sf
TRAIN_PCT_SET9  = 1.00  # selfplay_sf_nodes1M_wdl
TRAIN_PCT_SET10 = 0.68  # selfplay-endgames-d12_lichess-n500k_sf
TRAIN_PCT_SET11 = 0.90  # endgame_synthetic1 
TRAIN_PCT_SET12 = 0.78  # selfplay_04-04_n50k_sf
TRAIN_PCT_SET13 = 1.00  # endgame_synthetic2
TRAIN_PCT_SET14 = 1.00  # selfplay_10-04_n50k_sf
TRAIN_PCT_SET15 = 1.00  # selfplay_endgame_14-04_n100k_sf

# ════════════════════════════════════════════════════════════
#  MODO DE AMOSTRAGEM POR DATASET
# ════════════════════════════════════════════════════════════

MODE_SET1  = 'lines'
MODE_SET2  = 'shards'
MODE_SET3  = 'shards'
MODE_SET4  = 'shards'
MODE_SET5  = 'shards'
MODE_SET6  = 'shards'
MODE_SET7  = 'shards'
MODE_SET8  = 'shards'
MODE_SET9  = 'lines'
MODE_SET10 = 'lines'
MODE_SET11 = 'lines'
MODE_SET12 = 'shards'
MODE_SET13 = 'shards'
MODE_SET14 = 'shards'
MODE_SET15 = 'shards'

# ════════════════════════════════════════════════════════════
#  Configuração de cada dataset
# ════════════════════════════════════════════════════════════
DATASETS = [
    {
        'name':         'liches-raw_cp_wdl',
        'input_dir':    f'{BASE_DIR}/data/lichess-raw_cp_wdl',
        'encoded_dir':  None,
        'input_suffix': '_filtered.parquet',
        'target_col':   'wdl',
        'train_pct':    TRAIN_PCT_SET1,
        'pct_mode':     MODE_SET1,
    },
    {
        'name':         'lichess-quiet_cp_wdl',
        'input_dir':    f'{BASE_DIR}/data/lichess-quiet_cp_wdl',
        'encoded_dir':  None,
        'input_suffix': '_quiet.parquet',
        'target_col':   'wdl',
        'train_pct':    TRAIN_PCT_SET2,
        'pct_mode':     MODE_SET2,
    },
    {
        'name':         'lichess-newquiet-n400k_sf',
        'input_dir':    f'{BASE_DIR}/data/lichess-newquiet-n400k_sf',
        'encoded_dir':  None,
        'input_suffix': '.parquet',
        'target_col':   'wdl',
        'train_pct':    TRAIN_PCT_SET3,
        'pct_mode':     MODE_SET3,
    },
    {
        'name':         'lichess-newquiet-nosf',
        'input_dir':    f'{BASE_DIR}/data/lichess-newquiet-nosf',
        'encoded_dir':  None,
        'input_suffix': '.parquet',
        'target_col':   'wdl',
        'train_pct':    TRAIN_PCT_SET4,
        'pct_mode':     MODE_SET4,
    },
    {
        'name':         'extra-quiet-n5k_sf',
        'input_dir':    f'{BASE_DIR}/data/extra-quiet-n5k_sf',
        'encoded_dir':  None,
        'input_suffix': '.parquet',
        'target_col':   'wdl',
        'train_pct':    TRAIN_PCT_SET5,
        'pct_mode':     MODE_SET5,
    },
    {
        'name':         'extra-quiet-n60k_sf',
        'input_dir':    f'{BASE_DIR}/data/extra-quiet-n60k_sf',
        'encoded_dir':  None,
        'input_suffix': '.parquet',
        'target_col':   'wdl',
        'train_pct':    TRAIN_PCT_SET6,
        'pct_mode':     MODE_SET6,
    },
    {
        'name':         'extra-quiet-endgames-d12_sf',
        'input_dir':    f'{BASE_DIR}/data/extra-quiet-endgames-d12_sf',
        'encoded_dir':  None,
        'input_suffix': '.parquet',
        'target_col':   'wdl',
        'train_pct':    TRAIN_PCT_SET7,
        'pct_mode':     MODE_SET7,
    },
    {
        'name':         'extra-quiet-endgames-d14_sf',
        'input_dir':    f'{BASE_DIR}/data/extra-quiet-endgames-d14_sf',
        'encoded_dir':  None,
        'input_suffix': '.parquet',
        'target_col':   'wdl',
        'train_pct':    TRAIN_PCT_SET8,
        'pct_mode':     MODE_SET8,
    },
    {
        'name':         'miscelaneous-n1M_sf',
        'input_dir':    f'{BASE_DIR}/data/miscelaneous-n1M_sf',
        'encoded_dir':  None,
        'input_suffix': '.parquet',
        'target_col':   'wdl',
        'train_pct':    TRAIN_PCT_SET9,
        'pct_mode':     MODE_SET9,
    },
    {
        'name':         'selfplay-endgames-d12_lichess-n500k_sf',
        'input_dir':    f'{BASE_DIR}/data/selfplay-endgames-d12_lichess-n500k_sf',
        'encoded_dir':  None,
        'input_suffix': '.parquet',
        'target_col':   'wdl',
        'train_pct':    TRAIN_PCT_SET10,
        'pct_mode':     MODE_SET10,
    },
    {
        'name':         'endgame_synthetic1',
        'input_dir':    f'{BASE_DIR}/data/endgame_synthetic1',
        'encoded_dir':  None,
        'input_suffix': '.parquet',
        'target_col':   'wdl',
        'train_pct':    TRAIN_PCT_SET11,
        'pct_mode':     MODE_SET11,
    },
    {
        'name':         'selfplay_04-04_n50k_sf',
        'input_dir':    f'{BASE_DIR}/data/selfplay_04-04_n50k_sf',
        'encoded_dir':  None,
        'input_suffix': '.parquet',
        'target_col':   'wdl',
        'train_pct':    TRAIN_PCT_SET12,
        'pct_mode':     MODE_SET12,
    },
    {
        'name':         'endgame_synthetic2',
        'input_dir':    f'{BASE_DIR}/data/endgame_synthetic2',
        'encoded_dir':  None,
        'input_suffix': '.parquet',
        'target_col':   'wdl',
        'train_pct':    TRAIN_PCT_SET13,
        'pct_mode':     MODE_SET13,
    },
    {
        'name':         'selfplay_10-04_n50k_sf',
        'input_dir':    f'{BASE_DIR}/data/selfplay_10-04_n50k_sf',
        'encoded_dir':  None,
        'input_suffix': '.parquet',
        'target_col':   'wdl',
        'train_pct':    TRAIN_PCT_SET14,
        'pct_mode':     MODE_SET14,
    },
    {
        'name':         'selfplay_endgame_14-04_n100k_sf',
        'input_dir':    f'{BASE_DIR}/data/selfplay_endgame_14-04_n100k_sf',
        'encoded_dir':  None,
        'input_suffix': '.parquet',
        'target_col':   'wdl',
        'train_pct':    TRAIN_PCT_SET15,
        'pct_mode':     MODE_SET15,
    },

]

# ════════════════════════════════════════════════════════════
#  Resolve quais sets estão ativos + mede linhas reais
# ════════════════════════════════════════════════════════════
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor as _TPE

def _rows_of(path):
    """Lê apenas o metadata do parquet — sem carregar dados, < 1ms por arquivo."""
    try:
        return pq.ParquetFile(path).metadata.num_rows
    except Exception:
        return 0

def _measure_dataset(ds):
    """Mede todas as linhas de todos os shards de um dataset em paralelo."""
    if not os.path.exists(ds['input_dir']):
        return [], []
    shard_files = sorted([f for f in os.listdir(ds['input_dir'])
                          if f.endswith(ds['input_suffix'])])
    if not shard_files:
        return shard_files, []
    paths = [os.path.join(ds['input_dir'], f) for f in shard_files]
    with _TPE(max_workers=min(16, len(paths))) as ex:
        counts = list(ex.map(_rows_of, paths))
    return shard_files, counts

active_datasets = []
print("\n  Medindo tamanhos reais dos datasets...")
for ds in DATASETS:
    if ds['train_pct'] <= 0.0:
        print(f"  ⏭  ignorado (0%): {ds['name']}")
        continue

    shard_files, counts = _measure_dataset(ds)

    if not shard_files:
        print(f"  ⚠️  Nenhum arquivo em {ds['input_dir']} — pulando.")
        continue

    # Remove shards com 0 linhas (corruptos / vazios)
    valid = [(f, c) for f, c in zip(shard_files, counts) if c > 0]
    if not valid:
        print(f"  ⚠️  Todos os shards de {ds['name']} retornaram 0 linhas — pulando.")
        continue
    shard_files, counts = zip(*valid)
    shard_files = list(shard_files)

    total_rows     = sum(counts)
    avg_rows_shard = total_rows / len(counts)

    mode   = ds['pct_mode']
    n_take = len(shard_files) if mode == 'lines' else max(1, round(len(shard_files) * ds['train_pct']))

    if mode == 'lines':
        # sorteia linhas dentro de cada shard → usa train_pct como fração de linhas
        approx = round(total_rows * ds['train_pct'])
    else:
        # sorteia shards inteiros → n_take shards completos
        approx = round(n_take * avg_rows_shard)

    active_datasets.append({
        **ds,
        'all_shards':     shard_files,
        'shard_counts':   list(counts),   # linhas reais por shard
        'total_rows':     total_rows,
        'avg_rows_shard': avg_rows_shard,
        'n_take':         n_take,
    })
    mode_label = '(% de linhas/shard)' if mode == 'lines' else '(shards completos)'
    print(f"  ✓  {ds['name']:44s}  {ds['train_pct']*100:5.1f}%  [{mode:6s}]"
          f"  {len(shard_files):4d} shards  total={total_rows:>12,}  avg={avg_rows_shard:>9,.0f}"
          f"  → ~{approx:>12,} pos/época  {mode_label}")

assert active_datasets, "❌ Nenhum dataset ativo!"
print(f"  ℹ️  Reamostragem: {'nova a cada época' if RESAMPLE_EACH_EPOCH else 'fixa'}")

# ════════════════════════════════════════════════════════════
#  Hiperparâmetros fixos
# ════════════════════════════════════════════════════════════
EPOCHS     = 200
BATCH_SIZE = 16384
N_WORKERS  = os.cpu_count()

# ════════════════════════════════════════════════════════════
#  VRAM/RAM Safety & Device Selection
# ════════════════════════════════════════════════════════════
FORCE_DEVICE = 'auto'

MAX_POSITIONS_VRAM = 1_100_000
MAX_POSITIONS_RAM  = 1_500_000

if FORCE_DEVICE == 'cuda' and torch.cuda.is_available():
    device = torch.device('cuda')
elif FORCE_DEVICE == 'cpu':
    device = torch.device('cpu')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_POSITIONS = MAX_POSITIONS_VRAM if device.type == 'cuda' else MAX_POSITIONS_RAM

print(f"\nDevice: {device}  |  Workers: {N_WORKERS}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
else:
    print("Rodando em CPU")

# ════════════════════════════════════════════════════════════
#  DIMENSÕES — altere aqui se quiser experimentar
# ════════════════════════════════════════════════════════════
INPUT_MAIN    = 768   # HM encoding (não muda)
INPUT_EXTRA   = 31    # features manuais de final (ver encode_extra)
INPUT_TOTAL   = INPUT_MAIN + INPUT_EXTRA   # 799
HIDDEN1       = 256   # L1 (não muda)
HIDDEN2       = 64    # L2: era 32, agora 64  ← PHASE 1 CHANGE

# ════════════════════════════════════════════════════════════
#  QAT HELPERS
# ════════════════════════════════════════════════════════════

def fake_quant_int16(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    limit = 32767.0 / scale
    x_clamp = tensor.clamp(-limit, limit)
    x_q = (x_clamp * scale).round() / scale
    return tensor + (x_q - tensor).detach()

def fake_quant_int8(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    limit = 127.0 / scale
    x_clamp = tensor.clamp(-limit, limit)
    x_q = (x_clamp * scale).round() / scale
    return tensor + (x_q - tensor).detach()

def fake_quant_bias_int32(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    x_q = (tensor * scale).round() / scale
    return tensor + (x_q - tensor).detach()

class ClippedReLU(nn.Module):
    def __init__(self, clip: float = 1.0):
        super().__init__()
        self.clip = clip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(0.0, self.clip)

    def extra_repr(self) -> str:
        return f'clip={self.clip}'

# ════════════════════════════════════════════════════════════
#  QAT MODEL
# ════════════════════════════════════════════════════════════

class NNUE(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1   = nn.Linear(INPUT_TOTAL, HIDDEN1)
        self.act1 = ClippedReLU(RELU_CLIP)
        self.l2   = nn.Linear(HIDDEN1, HIDDEN2)
        self.act2 = ClippedReLU(RELU_CLIP)
        self.l3   = nn.Linear(HIDDEN2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── L1 ──────────────────────────────────────────────────────────
        w1 = fake_quant_int16(self.l1.weight, QA)
        b1 = fake_quant_bias_int32(self.l1.bias, float(QA))

        h1 = self.act1(F.linear(x, w1, b1))

        # ── Fake-quant L1 activation (simulates uint8 output) ───────────
        h1_q = (h1 * QA).round().clamp(0, QA) / QA
        h1 = h1 + (h1_q - h1).detach()   # STE

        # ── L2 ──────────────────────────────────────────────────────────
        w2 = fake_quant_int8(self.l2.weight, QB)
        b2 = fake_quant_bias_int32(self.l2.bias, float(QA * QB))

        h2 = self.act2(F.linear(h1, w2, b2))

        # Fake-quant L2 activation (simulates uint8 output)
        h2_q = (h2 * QB).round().clamp(0, QB) / QB
        h2 = h2 + (h2_q - h2).detach()   # STE

        # ── L3 ──────────────────────────────────────────────────────────
        w3 = fake_quant_int8(self.l3.weight, QB)

        out = torch.sigmoid(F.linear(h2, w3, self.l3.bias))
        return out.squeeze(1)

def clamp_weights_(model: NNUE) -> None:
    with torch.no_grad():
        lim1 = 32767.0 / QA
        model.l1.weight.clamp_(-lim1, lim1)
        model.l1.bias.clamp_(  -lim1, lim1)

        lim2 = 127.0 / QB
        model.l2.weight.clamp_(-lim2, lim2)
        model.l3.weight.clamp_(-lim2, lim2)

# ════════════════════════════════════════════════════════════
#  Carregamento de checkpoint
# ════════════════════════════════════════════════════════════
def find_checkpoints(ckpt_dir):
    all_jsons = [
        f for f in os.listdir(ckpt_dir)
        if f.startswith('nnue_') and f.endswith('.json')
    ]
    if not all_jsons:
        return None

    def load_meta(fname):
        with open(os.path.join(ckpt_dir, fname)) as f:
            meta = json.load(f)
        if 'weights' not in meta:
            keys = [k for k in meta if k not in ('epoch', 'avg_loss', 'lr', 'timestamp', 'dataset', 'arch', 'qa', 'qb', 'relu_clip', 'qat', 'qat_level')]
            meta['weights'] = {k: meta.pop(k) for k in keys}
        return meta

    latest_fname = max(all_jsons, key=lambda f: os.path.getmtime(os.path.join(ckpt_dir, f)))
    return latest_fname, load_meta(latest_fname)

def remap_weights_v187_to_v188(state: dict) -> dict:
    mapping = {
        'net.0.weight': 'l1.weight', 'net.0.bias': 'l1.bias',
        'net.2.weight': 'l2.weight', 'net.2.bias': 'l2.bias',
        'net.4.weight': 'l3.weight', 'net.4.bias': 'l3.bias',
    }
    return {mapping.get(k, k): torch.tensor(v) for k, v in state.items()}

print(f"\n── Carregando modelo ({DATASET_NAME}) ──")
latest_ckpt = find_checkpoints(CKPT_DIR)
model = NNUE().to(device)

if latest_ckpt:
    fname, meta = latest_ckpt
    try:
        state = remap_weights_v187_to_v188(meta['weights'])
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  ⚠️  Pesos faltando: {missing}")

        if meta.get('dataset') == DATASET_NAME:
            start_epoch = meta['epoch']
            resume_lr   = meta['lr']
            print(f"  ✓ Retomando (MESMO DATASET): {fname}")
        else:
            start_epoch = 0
            resume_lr   = 8e-6
            print(f"  ✓ Transferência de pesos (NOVO DATASET): {fname}")
            print(f"    Dataset anterior: {meta.get('dataset', 'desconhecido')}")
            print(f"    Iniciando época do zero com LR=8.00e-06")

        print(f"  Epoch {start_epoch} | loss: {meta['avg_loss']:.5f} | lr: {resume_lr:.2e}")
    except Exception as e:
        start_epoch = 0
        resume_lr   = 1e-5
        print(f"  ⚠️  Incompatível: {fname}. Iniciando do zero. Erro: {e}")
else:
    start_epoch = 0
    resume_lr   = 1e-5
    print("  ⚠️  NENHUM CHECKPOINT ENCONTRADO — pesos aleatórios.")

# ════════════════════════════════════════════════════════════
#  Encoding
# ════════════════════════════════════════════════════════════

def to_wdl(values, col):
    if col == 'wdl': return values.astype(np.float16)
    return (1.0 / (1.0 + np.exp(-values / 320.0))).astype(np.float16)

PIECE_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}

def encode_chunk(fens):
    """
    Retorna dois arrays: (N, 768) bits em uint8 bit-packed para (N, 96), e (N, 31) float16
    """
    import chess, numpy as np
    N = len(fens)
    out_bits = np.zeros((N, 768), dtype=np.uint8)
    out_extra = np.zeros((N, 31), dtype=np.float16)
    MAXCNT = np.array([8, 2, 2, 2, 1, 1], dtype=np.float16)

    for i, fen in enumerate(fens):
        board = chess.Board(fen)

        if board.turn == chess.BLACK:
            board = board.mirror()

        piece_dict = board.piece_map()

        for sq, piece in piece_dict.items():
            color_offset = 0 if piece.color == chess.WHITE else 6
            out_bits[i, (PIECE_MAP[piece.piece_type] + color_offset) * 64 + sq] = 1

        cnt_w = np.zeros(6, dtype=np.float16)
        cnt_b = np.zeros(6, dtype=np.float16)
        for sq, piece in piece_dict.items():
            idx = PIECE_MAP[piece.piece_type]
            if piece.color == chess.WHITE:
                cnt_w[idx] += 1
            else:
                cnt_b[idx] += 1

        out_extra[i, 0:6] = cnt_w / MAXCNT
        out_extra[i, 6:12] = cnt_b / MAXCNT

        mat_vals = np.array([1, 3, 3, 5, 9, 0], dtype=np.float16)
        total_mat = float(np.dot(cnt_w + cnt_b, mat_vals))
        out_extra[i, 12] = total_mat / 78.0

        out_extra[i, 13] = 1.0  # sempre brancas após mirror

        # peões passados — white
        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces(chess.PAWN, chess.BLACK)
        for sq in white_pawns:
            file_ = chess.square_file(sq)
            rank_ = chess.square_rank(sq)
            blocking = False
            for f2 in range(max(0, file_-1), min(8, file_+2)):
                for r2 in range(rank_+1, 8):
                    if chess.square(f2, r2) in black_pawns:
                        blocking = True
                        break
                if blocking:
                    break
            if not blocking:
                out_extra[i, 14 + file_] = 1.0

        # peões passados — black
        for sq in black_pawns:
            file_ = chess.square_file(sq)
            rank_ = chess.square_rank(sq)
            blocking = False
            for f2 in range(max(0, file_-1), min(8, file_+2)):
                for r2 in range(0, rank_):
                    if chess.square(f2, r2) in white_pawns:
                        blocking = True
                        break
                if blocking:
                    break
            if not blocking:
                out_extra[i, 22 + file_] = 1.0

        # distância Chebyshev entre reis
        wk_sq = board.king(chess.WHITE)
        bk_sq = board.king(chess.BLACK)
        if wk_sq is not None and bk_sq is not None:
            wk_f, wk_r = chess.square_file(wk_sq), chess.square_rank(wk_sq)
            bk_f, bk_r = chess.square_file(bk_sq), chess.square_rank(bk_sq)
            cheb = max(abs(wk_f - bk_f), abs(wk_r - bk_r))
            out_extra[i, 30] = cheb / 7.0

    return np.packbits(out_bits, axis=1), out_extra

def build_tensors_parallel(df, target_col):
    fens   = df['fen'].tolist()
    y_base = to_wdl(df[target_col].values, target_col)

    # Inverte perspectiva do alvo para pretas (mirror feito no encode)
    is_black = np.array([fen.split(' ')[1] == 'b' for fen in fens])
    y_base[is_black] = 1.0 - y_base[is_black]

    size   = max(1, len(fens) // max(1, N_WORKERS))
    splits = [fens[i:i+size] for i in range(0, len(fens), size)]

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        results = list(ex.map(encode_chunk, splits))

    Xs_bits = [r[0] for r in results]
    Xs_extra = [r[1] for r in results]

    return torch.from_numpy(np.concatenate(Xs_bits)), torch.from_numpy(np.concatenate(Xs_extra)), torch.from_numpy(y_base)

def pt_name(fname, input_suffix):
    return fname.replace(input_suffix, '_encoded.pt')

def sample_epoch_shards(active_datasets):
    all_shards = []
    for ds in active_datasets:
        if ds['pct_mode'] == 'lines':
            chosen = ds['all_shards']
        else:
            chosen = random.sample(ds['all_shards'], ds['n_take'])
        for fname in chosen:
            all_shards.append((fname, ds))
    random.shuffle(all_shards)
    return all_shards

def yield_safe_chunks_mixed(epoch_shards, max_pos):
    acc_X_bits, acc_X_extra, acc_y = [], [], []
    current_size = 0

    for fname, ds in epoch_shards:
        input_dir    = ds['input_dir']
        input_suffix = ds['input_suffix']
        target_col   = ds['target_col']
        train_pct    = ds['train_pct']
        pct_mode     = ds['pct_mode']
        row_frac     = train_pct if pct_mode == 'lines' else 1.0

        encoded_dir  = ds['encoded_dir'] or ds['input_dir']
        pt_path = os.path.join(encoded_dir, pt_name(fname, input_suffix))

        if not os.path.exists(pt_path):
            # Ler parquet e gerar cache inteiro
            import pyarrow.parquet as pq
            try:
                parquet_file = pq.ParquetFile(os.path.join(input_dir, fname))
            except Exception as e:
                print(f"  ⚠️  Erro abrindo parquet {fname}: {e}")
                continue
                
            all_X_bits, all_X_extra, all_y = [], [], []
            print(f"\n  ⏳ Gerando cache para {fname}...")
            for batch in parquet_file.iter_batches(batch_size=100_000):
                df_chunk = batch.to_pandas()
                X_bits_part, X_extra_part, y_part = build_tensors_parallel(df_chunk, target_col)
                all_X_bits.append(X_bits_part)
                all_X_extra.append(X_extra_part)
                all_y.append(y_part)

            if all_X_bits:
                X_bits_full = torch.cat(all_X_bits)
                X_extra_full = torch.cat(all_X_extra)
                y_full = torch.cat(all_y)
                print(f"  💾 Salvando cache compacto .pt: {pt_path} ({len(X_bits_full)} pos)")
                torch.save({'X_bits': X_bits_full, 'X_extra': X_extra_full, 'y': y_full}, pt_path)
                del X_bits_full, X_extra_full, y_full, all_X_bits, all_X_extra, all_y
            else:
                continue

        # Ler do cache .pt
        try:
            data = torch.load(pt_path, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"  ⚠️  Erro carregando {pt_path}: {e}")
            continue
            
        if 'X_bits' not in data:
            print(f"\n  ⚠️  Cache antigo incompatível detectado em: {fname}. Delete os arquivos _encoded.pt antigos e reinicie.")
            continue

        X_bits_shard = data['X_bits']
        X_extra_shard = data['X_extra']
        y_shard = data['y']
        del data

        if row_frac < 1.0:
            n_rows = len(X_bits_shard)
            n_keep = max(1, round(n_rows * row_frac))
            idx    = torch.randperm(n_rows)[:n_keep]
            X_bits_shard = X_bits_shard[idx]
            X_extra_shard = X_extra_shard[idx]
            y_shard = y_shard[idx]
            del idx

        offset = 0
        while offset < len(X_bits_shard):
            take   = max_pos - current_size
            X_bits_part = X_bits_shard[offset: offset + take]
            X_extra_part = X_extra_shard[offset: offset + take]
            y_part = y_shard[offset: offset + take]
            
            acc_X_bits.append(X_bits_part)
            acc_X_extra.append(X_extra_part)
            acc_y.append(y_part)
            current_size += len(X_bits_part)
            offset += len(X_bits_part)

            if current_size >= max_pos:
                cat_bits = torch.cat(acc_X_bits).numpy() # (N, 96) uint8
                cat_extra = torch.cat(acc_X_extra).to(dtype=torch.float32)
                unpacked_bits = np.unpackbits(cat_bits, axis=1).astype(np.float32)
                
                X_final = torch.cat([torch.from_numpy(unpacked_bits), cat_extra], dim=1).to(device)
                y_final = torch.cat(acc_y).to(device=device, dtype=torch.float32)
                yield X_final, y_final
                
                acc_X_bits, acc_X_extra, acc_y = [], [], []
                current_size = 0
                gc.collect()

        del X_bits_shard, X_extra_shard, y_shard
        gc.collect()

    if acc_X_bits:
        cat_bits = torch.cat(acc_X_bits).numpy()
        cat_extra = torch.cat(acc_X_extra).to(dtype=torch.float32)
        unpacked_bits = np.unpackbits(cat_bits, axis=1).astype(np.float32)
        
        X_final = torch.cat([torch.from_numpy(unpacked_bits), cat_extra], dim=1).to(device)
        y_final = torch.cat(acc_y).to(device=device, dtype=torch.float32)
        yield X_final, y_final
        del acc_X_bits, acc_X_extra, acc_y
        gc.collect()

# ════════════════════════════════════════════════════════════
#  Setup
# ════════════════════════════════════════════════════════════
approx_rows = int(sum(
    ds['total_rows'] * ds['train_pct'] if ds['pct_mode'] == 'lines'
    else ds['n_take'] * ds['avg_rows_shard']
    for ds in active_datasets
))

total_steps  = EPOCHS * (approx_rows // max(1, BATCH_SIZE))
total_chunks = max(1, math.ceil(approx_rows / MAX_POSITIONS))
PRINT_EVERY_N_CHUNKS = max(1, total_chunks // 20)   # ~20 linhas por época

print(f"\nTotal estimado de posições : ~{approx_rows:,}")
print(f"Chunks estimados por epoch : ~{total_chunks}  (MAX_POSITIONS={MAX_POSITIONS:,})")
print(f"Print permanente a cada    :  {PRINT_EVERY_N_CHUNKS} chunks")
print(f"Arquitetura                : {INPUT_TOTAL}→{HIDDEN1}→{HIDDEN2}→1  [QAT v188H]")
print(f"QAT schedule               : FULL QAT from Epoch 0 (int16+int8)")
print(f"QA={QA}  QB={QB}  RELU_CLIP={RELU_CLIP}")

optimizer = torch.optim.Adam(model.parameters(), lr=resume_lr, weight_decay=1e-4)

if start_epoch > 0:
    for group in optimizer.param_groups:
        group.setdefault('initial_lr', resume_lr)

scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max(1, total_steps), eta_min=1e-6,
    last_epoch=start_epoch * (approx_rows // BATCH_SIZE) - 1 if start_epoch > 0 else -1
)
loss_fn = nn.BCELoss()
model.train()

# ════════════════════════════════════════════════════════════
#  Loop de Treino
# ════════════════════════════════════════════════════════════
for epoch in range(start_epoch, EPOCHS):
    qat_label = 'QAT-ALL (int16+int8)'

    print(f"\n{'═'*85}")
    print(f"EPOCH {epoch+1}/{EPOCHS}   lr={optimizer.param_groups[0]['lr']:.2e}   [{qat_label}]")
    print(f"{'═'*85}")

    if epoch == start_epoch or RESAMPLE_EACH_EPOCH:
        epoch_shards = sample_epoch_shards(active_datasets)

    epoch_loss       = 0.0
    epoch_mae        = 0.0
    epoch_std        = 0.0
    epoch_max_err    = 0.0
    epoch_batches    = 0
    chunk_idx        = 0
    epoch_start_time = time.time()
    positions_done   = 0

    for X_t, y_t in yield_safe_chunks_mixed(epoch_shards, MAX_POSITIONS):
        chunk_idx += 1
        n_samples   = len(X_t)
        positions_done += n_samples

        dataset = TensorDataset(X_t, y_t)
        loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False)
        del X_t, y_t
        gc.collect()

        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            preds = model(xb)
            loss  = loss_fn(preds, yb)
            loss.backward()

            with torch.no_grad():
                abs_err = torch.abs(preds - yb)
                epoch_mae += abs_err.mean().item()
                epoch_std += abs_err.std().item()
                max_e = abs_err.max().item()
                if max_e > epoch_max_err:
                    epoch_max_err = max_e

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler_cosine.step()

            # Hard-clamp weights to quantization range after every step
            clamp_weights_(model)

            epoch_loss    += loss.item()
            epoch_batches += 1

        del dataset, loader
        gc.collect()

        elapsed  = time.time() - epoch_start_time
        progress = min(1.0, positions_done / max(1, approx_rows))

        if chunk_idx > 0 and positions_done > 0:
            eta_total = elapsed / progress
            eta_rem   = max(0.0, eta_total - elapsed)
            m, s = divmod(int(eta_rem), 60)
            h, m = divmod(m, 60)
            eta_str = f"{h:02d}h{m:02d}m" if h > 0 else f"{m:02d}m{s:02d}s"
        else:
            eta_str = "--m--s"

        avg_loss = epoch_loss / max(1, epoch_batches)
        avg_mae  = epoch_mae  / max(1, epoch_batches)
        avg_std  = epoch_std  / max(1, epoch_batches)

        status_line = (
            f"  Ep {epoch+1}/{EPOCHS}  Chunk {chunk_idx:>4} (~{total_chunks}) | "
            f"{progress*100:5.1f}% | {positions_done/1e6:.2f}M pos | "
            f"L: {avg_loss:.4f} | MAE: {avg_mae:.3f}(±{avg_std:.3f}) | ETA: {eta_str}"
        )

        # Sempre atualiza a linha viva com \r
        print(status_line + "      ", end='\r', flush=True)

        # A cada N chunks imprime uma linha PERMANENTE para ter histórico visível
        if chunk_idx % PRINT_EVERY_N_CHUNKS == 0:
            print(status_line)

    # Linha final da época — permanente, mostra o total real de chunks
    avg_epoch = epoch_loss / max(epoch_batches, 1)
    avg_mae_e = epoch_mae  / max(epoch_batches, 1)
    print(f"\n  ✓ Epoch {epoch+1}/{EPOCHS} concluída | "
          f"{chunk_idx} chunks reais (~{total_chunks} estimado) | "
          f"{positions_done/1e6:.2f}M pos | "
          f"Loss: {avg_epoch:.5f} | MAE: {avg_mae_e:.4f} | [{qat_label}]")

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    ckpt_path = os.path.join(CKPT_DIR, f'nnue_{DATASET_NAME}_epoch{epoch+1:02d}_{timestamp}.json')
    with open(ckpt_path, 'w') as f:
        json.dump({
            'epoch':        epoch + 1,
            'dataset':      DATASET_NAME,
            'avg_loss':     avg_epoch,
            'lr':           optimizer.param_groups[0]['lr'],
            'timestamp':    timestamp,
            'arch':         {'input': INPUT_TOTAL, 'h1': HIDDEN1, 'h2': HIDDEN2},
            'qat':          True,
            'qat_level':    2,
            'qa':           QA,
            'qb':           QB,
            'relu_clip':    RELU_CLIP,
            'weights':      {k: v.cpu().tolist() for k, v in model.state_dict().items()},
        }, f)

    print(f"  Checkpoint salvo: {ckpt_path}")
    if device.type == 'cuda':
        print(f"  Pico VRAM GPU: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

print("\n✓ Treino completo!")
