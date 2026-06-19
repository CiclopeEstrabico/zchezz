import os, re, glob, time, gc, chess, random, threading, json, math, pathlib
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
import subprocess
from typing import List, Dict, Tuple, Any, Optional

# ===========================================================================
#  CONFIG
# ===========================================================================

SF_PATH  = r"engine\stockfish_fast\stockfish\stockfish-windows-x86-64-avxvnni.exe"
OUT_DIR  = r'data\selfplay_endgame_14-04_n100k_sf'

SF_HASH_MB = 16
N_WORKERS    = max(1, os.cpu_count() - 1)

BATCH_SIZE         = 5_000          # each worker batch; smaller = more frequent progress
MAX_ACTIVE_BATCHES = N_WORKERS * 2  # enough to keep all workers busy
CHUNK_SIZE_SAVE    = 10_000         # save & print progress every 10k saved positions

# Lines to sample when estimating EPD file row count from file size
EPD_SAMPLE_LINES = 500

# -- Global filter defaults --------------------------------------------------
DEFAULT_FILTERS = dict(
    filter_duplicates    = True,
    filter_in_check      = True,
    filter_terminal      = True,
    filter_win_capture   = True,
    filter_equal_capture = True,
    filter_sacrifice     = False,
    filter_score_cap     = True,
    score_cap_value      = 1000,
    limit                = 80_000_000,
    # Endgame filter (disabled by default)
    filter_endgame       = False,
    endgame_max_pieces   = 14,   # total pieces incl. kings
                                 #  7 = tablebase territory
                                 # 10 = typical rook/minor endings
                                 # 14 = soft "few pieces" endgame
    endgame_no_queens    = True,
    # Engine search
    sf_search_mode       = 'nodes',  # 'nodes', 'depth', or 'none' (passthrough: keep existing cp/wdl)
    sf_nodes             = 100_000,
    sf_depth             = 12,
)

# -- Input sources -----------------------------------------------------------
# Mandatory keys: path, src_type ('epd'|'parquet'), limit (int or 'all')
# Any DEFAULT_FILTERS key can be overridden per source.
# Plain 3-tuples (path, src_type, limit) still accepted.

INPUT_SOURCES = [
    # dict(
    #     path     = r'data\selfplay_raw_wdl',
    #     src_type = 'epd',
    #     limit    = 'all',
    # ),
    # dict(
    #     path            = r'data\lichess_quiet_wdl',
    #     src_type        = 'parquet',
    #     limit           = 200_000_000,
    #     score_cap_value = 900,
    #     sf_search_mode  = 'none'
    # ),
    # dict(
    #     path              = r'data\extra_quiet_sf_nodes5000_wdl',
    #     src_type          = 'parquet',
    #     limit             = 7_000_000,
    #     score_cap_value   = 350,    
    #     sf_search_mode    = 'depth',
    #     sf_nodes          = 60_000,
    # ),
    # dict(
    #     path              = r'data\selfplay_raw_10-04_wdl',
    #     src_type          = 'epd',
    #     limit             = 'all',      # fast sequential read, no 14M RAM reservoir
    #     score_cap_value   = 1000,    
    #     sf_search_mode    = 'nodes',
    #     sf_nodes          = 50_000,
    # )
    dict(
        path              = r'data\selfplay_endgame_raw_14-04_wdl',
        src_type          = 'epd',
        limit             = 'all',      # fast sequential read, no 14M RAM reservoir
        score_cap_value   = 3000,    
    )
]
  
# -- Misc --------------------------------------------------------------------
PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   20000,
}
EQUAL_CAPTURE_TOLERANCE = 50
SF_POSITION_TIMEOUT_S   = 60

# ===========================================================================
#  SOURCE HELPERS
# ===========================================================================

def _normalise_source(src) -> Dict[str, Any]:
    if isinstance(src, (tuple, list)):
        path, src_type, limit = src
        src = dict(path=path, src_type=src_type, limit=limit)
    cfg = dict(DEFAULT_FILTERS)
    cfg.update(src)
    return cfg

def _source_key(cfg: Dict) -> str:
    """Stable string key used for progress tracking and buffer change detection."""
    return f"{cfg['path']}|{cfg['src_type']}"

# ===========================================================================
#  SOURCE SIZE ESTIMATION  (for ETA on 'all' sources too)
# ===========================================================================

def estimate_source_size(cfg: Dict) -> Optional[int]:
    """
    Returns estimated total record count, or None on failure.

    Parquet: reads only row-group metadata (zero data I/O, very fast).
    EPD:     sums all file sizes (one stat() call each), then samples
             EPD_SAMPLE_LINES lines to compute mean bytes/line, then
             estimates total = total_bytes / mean_bytes_per_line.
             The sample is taken from the start of the first file only,
             so it adds negligible time even for huge datasets.
    """
    path     = cfg['path']
    src_type = cfg['src_type']

    if not os.path.exists(path):
        return None

    try:
        p = pathlib.Path(path)
        if src_type == 'parquet':
            import pyarrow.parquet as pq
            # rglob treats ** robustly on both Windows and POSIX
            files = [str(f) for f in p.rglob('*.parquet')]
            total = 0
            for fpath in files:
                pf     = pq.ParquetFile(fpath)
                total += pf.metadata.num_rows
            return total if total > 0 else None

        elif src_type == 'epd':
            files = [str(f) for f in p.rglob('*.epd')]
            if not files:
                return None
            total_bytes = sum(os.path.getsize(f) for f in files)
            if total_bytes == 0:
                return None

            # Sample bytes-per-line from the beginning of the first sorted file
            sampled_bytes = 0
            sampled_lines = 0
            for fpath in sorted(files):
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        stripped = line.strip()
                        if stripped and not stripped.startswith(('#', '%')):
                            sampled_bytes += len(line.encode('utf-8', errors='ignore'))
                            sampled_lines += 1
                            if sampled_lines >= EPD_SAMPLE_LINES:
                                break
                if sampled_lines >= EPD_SAMPLE_LINES:
                    break

            if sampled_lines == 0:
                return None
            bpl = sampled_bytes / sampled_lines
            return int(total_bytes / bpl)

    except Exception:
        return None

# ===========================================================================
#  PROGRESS FILE
# ===========================================================================

def _progress_path() -> str:
    return os.path.join(OUT_DIR, 'progress.json')

def load_progress() -> Dict[str, int]:
    """Returns {source_key: records_consumed}. Empty dict if no file exists."""
    p = _progress_path()
    if os.path.exists(p):
        try:
            with open(p, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"  [Warning] Could not read progress file: {e}")
    return {}

def save_progress(consumed: Dict[str, int]):
    """Atomically overwrite the progress file."""
    p   = _progress_path()
    tmp = p + '.tmp'
    try:
        with open(tmp, 'w') as f:
            json.dump(consumed, f, indent=2)
        os.replace(tmp, p)
    except Exception as e:
        print(f"  [Warning] Could not save progress: {e}")

# ===========================================================================
#  PARSING
# ===========================================================================

def parse_epd_line(line: str):
    line = line.strip()
    if not line or line.startswith(('#', '%')):
        return None
    parts = line.split()
    if len(parts) < 4:
        return None
    fen     = ' '.join(parts[:4]) + ' 0 1'
    ops_str = ' '.join(parts[4:])
    return {'fen': fen, 'result': _extract_op(ops_str, 'c0'), 'id': _extract_op(ops_str, 'id')}

def _extract_op(ops_str: str, key: str):
    m = re.search(rf'\b{key}\s+"([^"]*)"', ops_str)
    if m: return m.group(1).strip()
    m = re.search(rf'\b{key}\s+([^;]+)', ops_str)
    if m: return m.group(1).strip()
    return None

# ===========================================================================
#  FILTERS
# ===========================================================================

def is_quiet_advanced(board: chess.Board, cfg: Dict) -> Tuple[bool, str]:
    if cfg['filter_terminal']:
        if board.is_checkmate() or board.is_stalemate():
            return False, "terminal"
        if not any(board.generate_legal_moves()):
            return False, "no_moves"

    if cfg['filter_in_check'] and board.is_check():
        return False, "in_check"

    if cfg['filter_win_capture'] or cfg['filter_equal_capture']:
        for move in board.generate_pseudo_legal_captures():
            attacker = board.piece_at(move.from_square)
            if attacker is None:
                continue
            att_val = PIECE_VALUES.get(attacker.piece_type, 0)
            if board.is_en_passant(move):
                vic_val = 100
            else:
                victim = board.piece_at(move.to_square)
                if victim is None:
                    continue
                vic_val = PIECE_VALUES.get(victim.piece_type, 0)
            if att_val == 100 and vic_val == 100:
                continue
            diff = vic_val - att_val
            if cfg['filter_win_capture']   and diff > EQUAL_CAPTURE_TOLERANCE:
                return False, "winning_capture"
            if cfg['filter_equal_capture'] and abs(diff) <= EQUAL_CAPTURE_TOLERANCE:
                return False, "equal_capture"

    # Opponent captures: flip turn directly (no null-move push, no pop needed)
    if cfg['filter_sacrifice']:
        board.turn = not board.turn
        for move in board.generate_pseudo_legal_captures():
            attacker = board.piece_at(move.from_square)
            if attacker is None:
                continue
            att_val = PIECE_VALUES.get(attacker.piece_type, 0)
            if board.is_en_passant(move):
                vic_val = 100
            else:
                victim = board.piece_at(move.to_square)
                if victim is None:
                    continue
                if victim.piece_type == chess.KING:
                    continue
                vic_val = PIECE_VALUES.get(victim.piece_type, 0)
            if att_val == 100 and vic_val == 100:
                continue
            diff = vic_val - att_val
            if (cfg['filter_win_capture']   and diff > EQUAL_CAPTURE_TOLERANCE) or \
               (cfg['filter_equal_capture'] and abs(diff) <= EQUAL_CAPTURE_TOLERANCE):
                board.turn = not board.turn
                return False, "sacrifice_other_side"
        board.turn = not board.turn

    return True, ""


def is_endgame(board: chess.Board, cfg: Dict) -> Tuple[bool, str]:
    """
    Returns (passes, reason). Called only when cfg['filter_endgame'] is True.

    endgame_no_queens:   reject if any queen on board.
                         Uses board.queens bitboard directly (faster than
                         board.pieces() which allocates a SquareSet object).
    endgame_max_pieces:  reject if total piece count > threshold.
      7  = tablebase territory (K + up to 5 pieces)
      10 = typical rook / minor-piece endings
      14 = soft endgame, few pieces remaining
    """
    if cfg['endgame_no_queens'] and board.queens:
        return False, "endgame_has_queen"

    if cfg['endgame_max_pieces'] is not None:
        if chess.popcount(board.occupied) > cfg['endgame_max_pieces']:
            return False, "endgame_too_many_pieces"

    return True, ""

# ===========================================================================
#  STOCKFISH ENGINE WRAPPER
# ===========================================================================

class StockfishEngine:
    def __init__(self):
        self.proc     = None
        self._q       = None
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
        self._start_reader_thread()
        self._w('uci')
        self._w(f'setoption name Hash value {SF_HASH_MB}')
        self._w('setoption name Threads value 1')
        self._w('isready')
        self._read_until('readyok', timeout=10)

    def _w(self, cmd: str):
        self.proc.stdin.write(cmd + '\n')
        self.proc.stdin.flush()

    # ------------------------------------------------------------------
    # Single persistent reader thread + SimpleQueue.
    # Eliminates the per-line Thread creation that was happening before
    # (one thread per SF output line = hundreds of threads per position).
    # ------------------------------------------------------------------
    def _start_reader_thread(self):
        import queue
        self._q = queue.SimpleQueue()
        def _reader():
            try:
                for line in self.proc.stdout:
                    self._q.put(line)
            except Exception:
                pass
            self._q.put(None)   # sentinel: process died / stdout closed
        threading.Thread(target=_reader, daemon=True).start()

    def _readline(self, timeout: float) -> str:
        try:
            line = self._q.get(timeout=timeout)
            return line if line is not None else ''
        except Exception:
            return ''   # timeout or queue error

    def _read_until(self, marker: str, timeout: float = SF_POSITION_TIMEOUT_S):
        lines    = []
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"SF did not emit '{marker}' in {timeout}s")
            line = self._readline(remaining)
            if not line:
                raise TimeoutError(f"SF stdout closed/hung waiting for '{marker}'")
            lines.append(line)
            if marker in line:
                return lines

    def evaluate(self, fen: str, cfg: Dict) -> int:
        """Centipawns from side-to-move POV. Raises TimeoutError on hang."""
        self._w(f'position fen {fen}')
        if cfg.get('sf_search_mode') == 'depth':
            self._w(f"go depth {cfg['sf_depth']}")
        else:
            self._w(f"go nodes {cfg['sf_nodes']}")
        last_cp  = 0
        _re_cp   = self._re_cp
        _re_mate = self._re_mate
        for line in self._read_until('bestmove'):
            if 'score' in line:   # skip lines without score fast
                m = _re_mate.search(line)
                if m:
                    last_cp = 30_000 if int(m.group(1)) > 0 else -30_000
                else:
                    m = _re_cp.search(line)
                    if m:
                        last_cp = int(m.group(1))
        return last_cp

    def close(self):
        try:
            self._w('quit')
            self.proc.wait(timeout=3)
        except Exception:
            pass
        try: self.proc.terminate()
        except: pass

# ===========================================================================
#  WORKER PROCESS
# ===========================================================================

def _worker_pipeline(records_batch: List[Dict], cfg: Dict):
    """Runs in a worker process. Returns (results_list, stats_dict).

    Two-phase design:
      Phase 1 (cheap): run quiet + endgame filter on all positions.
      Phase 2: Stockfish evaluates only the positions that passed.
    This avoids paying SF startup / UCI overhead on positions that
    would be filtered anyway.
    """
    passthrough = cfg.get('sf_search_mode') == 'none'
    results: List[Dict] = []
    stats:   Dict       = {}

    # ── Phase 1: quiet filter ────────────────────────────────────────────────
    quiet_recs: List[Dict] = []
    for rec in records_batch:
        fen = rec.get('fen', '')
        if not fen:
            stats['bad_fen'] = stats.get('bad_fen', 0) + 1
            continue
        try:
            board  = chess.Board(fen)
            ok, reason = is_quiet_advanced(board, cfg)
        except Exception:
            stats['bad_fen'] = stats.get('bad_fen', 0) + 1
            continue
        if not ok:
            stats[reason] = stats.get(reason, 0) + 1
            continue
        if cfg['filter_endgame']:
            ok, reason = is_endgame(board, cfg)
            if not ok:
                stats[reason] = stats.get(reason, 0) + 1
                continue
        quiet_recs.append(rec)

    if not quiet_recs:
        return results, stats

    # ── Phase 2a: passthrough — no SF needed ─────────────────────────────────
    if passthrough:
        for rec in quiet_recs:
            if 'cp' in rec and rec['cp'] is not None:
                cp_white = float(rec['cp'])
                wdl      = float(rec['wdl']) if 'wdl' in rec and rec['wdl'] is not None \
                           else 1.0 / (1.0 + np.exp(-cp_white / 320.0))
            elif 'wdl' in rec and rec['wdl'] is not None:
                wdl_val  = max(1e-7, min(1.0 - 1e-7, float(rec['wdl'])))
                cp_white = -320.0 * np.log(1.0 / wdl_val - 1.0)
                wdl      = wdl_val
            else:
                stats['missing_cp'] = stats.get('missing_cp', 0) + 1
                continue
            if cfg['filter_score_cap'] and abs(cp_white) > cfg['score_cap_value']:
                stats['score_cap'] = stats.get('score_cap', 0) + 1
                continue
            results.append({'fen': rec['fen'], 'cp': cp_white, 'wdl': wdl,
                            'result': rec.get('result', ''), 'id': rec.get('id', '')})
        return results, stats

    # ── Phase 2b: Stockfish evaluation (one engine, serial) ──────────────────
    engine = StockfishEngine()
    try:
        for rec in quiet_recs:
            fen = rec['fen']
            try:
                side     = fen.split()[1]
                last_cp  = engine.evaluate(fen, cfg)
                cp_white = last_cp if side == 'w' else -last_cp
                wdl      = 1.0 / (1.0 + np.exp(-cp_white / 320.0))
            except TimeoutError:
                stats['engine_timeout'] = stats.get('engine_timeout', 0) + 1
                try: engine.close()
                except: pass
                engine = StockfishEngine()
                continue
            except Exception:
                stats['engine_error'] = stats.get('engine_error', 0) + 1
                continue
            if cfg['filter_score_cap'] and abs(cp_white) > cfg['score_cap_value']:
                stats['score_cap'] = stats.get('score_cap', 0) + 1
                continue
            results.append({'fen': rec['fen'], 'cp': cp_white, 'wdl': wdl,
                            'result': rec.get('result', ''), 'id': rec.get('id', '')})
    finally:
        try: engine.close()
        except: pass

    return results, stats


# ===========================================================================
#  DATA LOADER
# ===========================================================================

def get_positions_generator(normalised_sources: List[Dict],
                             resume_consumed:   Dict[str, int]):
    """
    Yields (record_dict, source_cfg) pairs.

    resume_consumed is a live dict shared with main().
    The generator increments resume_consumed[key] as it yields each record,
    so main() always has an up-to-date count to write to the progress file.

    Records already consumed in a previous run are skipped with minimal work:
    EPD 'all'  -> skip by line counter (one increment + compare per line)
    EPD limit  -> reservoir is built over all lines; skip applies after
    Parquet    -> skip by record counter inside iter_batches loop
    """
    import pyarrow.parquet as pq

    for cfg in normalised_sources:
        path     = cfg['path']
        src_type = cfg['src_type']
        limit    = cfg['limit']
        key      = _source_key(cfg)

        if not os.path.exists(path):
            print(f"  [Warning] Source not found: {path}")
            continue

        skip = resume_consumed.get(key, 0)

        sf_mode  = cfg['sf_search_mode']
        mode_str = ('passthrough (no SF)' if sf_mode == 'none'
                    else f"depth {cfg['sf_depth']}" if sf_mode == 'depth'
                    else f"{cfg['sf_nodes']:,} nodes")
        eg_str   = (f"  endgame<={cfg['endgame_max_pieces']}pc"
                    + (" no-Q" if cfg['endgame_no_queens'] else "")
                    if cfg['filter_endgame'] else "")
        skip_str = f"  (resuming: skipping first {skip:,})" if skip else ""
        print(f"  Source: {os.path.basename(path)} ({src_type}) "
              f"cap={cfg['score_cap_value']} sf={mode_str}{eg_str}{skip_str}")

        # ── EPD ─────────────────────────────────────────────────────────────
        if src_type == 'epd':
            p = pathlib.Path(path)
            files = [str(f) for f in p.rglob('*.epd')]
            if not files:
                continue

            if limit == 'all':
                # Sequential - skip consumed records cheaply
                consumed = 0
                for fpath in sorted(files):
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            rec = parse_epd_line(line)
                            if not rec:
                                continue
                            if consumed < skip:
                                consumed += 1
                                continue
                            consumed += 1
                            resume_consumed[key] = consumed
                            yield rec, cfg

            else:
                # Reservoir sample (Algorithm R) - uniform random across all files
                # Build the full reservoir first, then yield, skipping as needed.
                random.shuffle(files)
                reservoir: List[Dict] = []
                raw_count = 0

                for fpath in files:
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            rec = parse_epd_line(line)
                            if not rec:
                                continue
                            raw_count += 1
                            if len(reservoir) < limit:
                                reservoir.append(rec)
                            else:
                                j = random.randrange(raw_count)
                                if j < limit:
                                    reservoir[j] = rec

                random.shuffle(reservoir)  # remove any insertion-order bias
                consumed = 0
                for rec in reservoir:
                    if consumed < skip:
                        consumed += 1
                        continue
                    consumed += 1
                    resume_consumed[key] = consumed
                    yield rec, cfg

        # ── Parquet ─────────────────────────────────────────────────────────
        elif src_type == 'parquet':
            p = pathlib.Path(path)
            files = [str(f) for f in p.rglob('*.parquet')]
            if not files:
                continue
            random.shuffle(files)

            # ceil per-file limit avoids systematic under-sampling from floor division
            per_file_limit = math.ceil(limit / len(files)) if limit != 'all' else None

            consumed    = 0   # raw records seen from this source (incl. skipped)
            count       = 0   # records actually yielded this run

            for fpath in files:
                if limit != 'all' and count >= limit:
                    break
                try:
                    pf   = pq.ParquetFile(fpath)
                    cols = [c for c in ['fen', 'cp', 'wdl', 'result', 'id']
                            if c in pf.schema_arrow.names]
                    file_count = 0
                    for batch in pf.iter_batches(batch_size=100_000, columns=cols):
                        # Fast path: read columns as python lists once per batch,
                        # then zip into dicts — avoids pandas DataFrame allocation.
                        col_lists = {col: batch.column(col).to_pylist()
                                     for col in cols}
                        n_rows = batch.num_rows
                        for i in range(n_rows):
                            if limit != 'all' and count >= limit:
                                break
                            if consumed < skip:
                                consumed += 1
                                continue
                            consumed   += 1
                            count      += 1
                            file_count += 1
                            resume_consumed[key] = consumed
                            yield {col: col_lists[col][i] for col in cols}, cfg
                        if limit != 'all' and count >= limit:
                            break
                        if per_file_limit and file_count >= per_file_limit:
                            break
                except Exception as e:
                    print(f"  [Warning] Error reading {fpath}: {e}")

# ===========================================================================
#  HELPERS
# ===========================================================================

def fen_key(fen: str) -> str:
    return ' '.join(fen.split()[:4])

def format_time(s: float) -> str:
    if s < 60:   return f"{s:.0f}s"
    if s < 3600: return f"{s/60:.1f}m"
    return f"{s/3600:.1f}h"

# ===========================================================================
#  MAIN
# ===========================================================================

def main():
    normalised_sources = [_normalise_source(s) for s in INPUT_SOURCES]

    print(f'  Output dir : {OUT_DIR}')
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Estimate source sizes for ETA ────────────────────────────────────────
    print("  Estimating source sizes...")
    source_sizes: Dict[str, Optional[int]] = {}
    total_estimated_input: Optional[int]   = 0

    for cfg in normalised_sources:
        key   = _source_key(cfg)
        limit = cfg['limit']
        est   = estimate_source_size(cfg)
        source_sizes[key] = est

        # What we'll actually consume from this source
        if est is not None:
            effective = est if limit == 'all' else min(est, limit)
        else:
            effective = None if limit == 'all' else limit

        # Print summary line
        est_str = f"~{est:,} records" if est is not None else "size unknown"
        lim_str = "all" if limit == 'all' else f"{limit:,}"
        print(f"    {os.path.basename(cfg['path'])}: {est_str}  (limit={lim_str})")

        # Accumulate total estimated input
        if total_estimated_input is not None:
            if effective is not None:
                total_estimated_input += effective
            else:
                total_estimated_input = None  # unknown source -> can't sum

    if total_estimated_input is not None:
        print(f"  Total estimated input : ~{total_estimated_input:,} records")
    else:
        print(f"  Total estimated input : unknown (some sources have no size estimate)")

    # ── Resume ───────────────────────────────────────────────────────────────
    seen_fens: set           = set()
    existing                 = sorted(glob.glob(os.path.join(OUT_DIR, 'chunk_*.parquet')))
    start_chunk              = len(existing)
    resume_consumed: Dict[str, int] = {}

    if start_chunk > 0:
        print(f'\n  Resuming from chunk {start_chunk}.')
        resume_consumed = load_progress()
        if resume_consumed:
            for cfg in normalised_sources:
                n = resume_consumed.get(_source_key(cfg), 0)
                if n:
                    print(f"    {os.path.basename(cfg['path'])}: "
                          f"skipping {n:,} already-consumed records")

        print(f'  Loading seen FENs from {start_chunk} output chunk(s)...')
        import pyarrow.parquet as pq
        for p in existing:
            try:
                pf = pq.ParquetFile(p)
                for batch in pf.iter_batches(batch_size=200_000, columns=['fen']):
                    for fval in batch.column('fen').to_pylist():
                        seen_fens.add(fen_key(fval))
            except Exception as e:
                print(f"  [Warning] Could not read {p}: {e}")
        print(f'  Loaded {len(seen_fens):,} unique seen FENs.\n')

    # ── Counters ─────────────────────────────────────────────────────────────
    total_saved    = len(seen_fens) if start_chunk > 0 else 0
    total_rej      = {}
    saved_this_run = 0   # positions saved since this invocation
    sent_this_run  = 0   # positions sent to workers (post-dedup) this invocation

    pending_save = []
    chunk_idx    = start_chunk
    t_start      = time.time()

    print(f"  Workers    : {N_WORKERS}  |  "
          f"Batch : {BATCH_SIZE:,}  |  "
          f"Max in-flight : {MAX_ACTIVE_BATCHES}")
    print("  Starting pipeline...\n")

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures: Dict = {}   # future -> submit_time
        producer      = get_positions_generator(normalised_sources, resume_consumed)
        exhausted     = False

        _buf_recs: List[Dict] = []
        _buf_key:  str        = ''
        _cur_cfg:  Dict       = {}

        def _flush_buf():
            nonlocal _buf_recs
            if _buf_recs:
                fut = pool.submit(_worker_pipeline, list(_buf_recs), _cur_cfg)
                futures[fut] = time.time()
                _buf_recs = []

        while not exhausted or futures:

            # ── 1. Feed pool ─────────────────────────────────────────────────
            while not exhausted and len(futures) < MAX_ACTIVE_BATCHES:
                try:
                    rec, cfg = next(producer)
                    skey     = _source_key(cfg)

                    if cfg.get('filter_duplicates', True):
                        k = fen_key(rec['fen'])
                        if k in seen_fens:
                            total_rej['duplicate'] = total_rej.get('duplicate', 0) + 1
                            continue
                        seen_fens.add(k)

                    sent_this_run += 1

                    if _buf_recs and skey != _buf_key:
                        _flush_buf()

                    _buf_key = skey
                    _cur_cfg = cfg
                    _buf_recs.append(rec)

                    if len(_buf_recs) >= BATCH_SIZE:
                        _flush_buf()

                except StopIteration:
                    _flush_buf()
                    exhausted = True
                    break

            # ── 2. Drain completed futures ────────────────────────────────────
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
                        del futures[fut]
                        continue

                    for reason, cnt in res_stats.items():
                        total_rej[reason] = total_rej.get(reason, 0) + cnt

                    pending_save.extend(res_list)
                    del futures[fut]

                    while len(pending_save) >= CHUNK_SIZE_SAVE:
                        save_slice   = pending_save[:CHUNK_SIZE_SAVE]
                        pending_save = pending_save[CHUNK_SIZE_SAVE:]

                        save_path = os.path.join(OUT_DIR, f'chunk_{chunk_idx:04d}.parquet')
                        pd.DataFrame(save_slice).to_parquet(save_path, index=False)

                        total_saved    += CHUNK_SIZE_SAVE
                        saved_this_run += CHUNK_SIZE_SAVE
                        elapsed         = time.time() - t_start

                        # Write progress atomically after every chunk
                        save_progress(dict(resume_consumed))

                        # -- Stats --------------------------------------------
                        pos_per_sec = saved_this_run / max(elapsed, 1e-9)
                        # Pass rate = fraction of dedup-passed records that
                        # survived all filters and were saved
                        pass_rate   = (saved_this_run / max(sent_this_run, 1)) * 100

                        # ETA: how many input records remain, scaled by pass rate
                        eta_str = ''
                        if pos_per_sec > 0:
                            total_consumed = sum(resume_consumed.values())
                            if total_estimated_input is not None and pass_rate > 0:
                                remaining_input = max(0, total_estimated_input - total_consumed)
                                eta_s   = (remaining_input * pass_rate / 100) / pos_per_sec
                                eta_str = f'ETA ~{format_time(eta_s)}'
                            else:
                                eta_str = 'ETA ?'

                        print(
                            f'  Chunk {chunk_idx:04d} | '
                            f'+{CHUNK_SIZE_SAVE:,} | '
                            f'Total: {total_saved:,} | '
                            f'Sent: {sent_this_run:,} | '
                            f'Pass: {pass_rate:.1f}% | '
                            f'{pos_per_sec:.0f} pos/s | '
                            f'In-flight: {len(futures)} | '
                            f'Elapsed: {format_time(elapsed)}'
                            + (f' | {eta_str}' if eta_str else '')
                        )
                        chunk_idx += 1
                        gc.collect()

        # ── 3. Flush remaining ────────────────────────────────────────────────
        if pending_save:
            save_path = os.path.join(OUT_DIR, f'chunk_{chunk_idx:04d}.parquet')
            pd.DataFrame(pending_save).to_parquet(save_path, index=False)
            total_saved    += len(pending_save)
            saved_this_run += len(pending_save)
            save_progress(dict(resume_consumed))
            print(f'  Final chunk {chunk_idx:04d} | '
                  f'+{len(pending_save):,} | Total: {total_saved:,}')

    elapsed = time.time() - t_start
    print(f'\n  DONE')
    print(f'    Total saved  : {total_saved:,}')
    print(f'    This run     : {saved_this_run:,} saved / {sent_this_run:,} sent to workers')
    print(f'    Rejections   : {total_rej}')
    print(f'    Elapsed      : {format_time(elapsed)}')
    print(f'    Throughput   : {saved_this_run / max(elapsed, 1e-9):.1f} pos/s')


if __name__ == '__main__':
    main()
