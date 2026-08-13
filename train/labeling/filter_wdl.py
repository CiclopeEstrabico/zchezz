import os, re, glob, time, gc, chess, random, math, pathlib
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import List, Dict, Tuple, Any, Optional

# ===========================================================================
#  CONFIG
# ===========================================================================

LAMBDA_WDL = 0.4
N_WORKERS  = max(1, os.cpu_count() - 1)

BATCH_SIZE         = 5_000          # each worker batch; smaller = more frequent progress
MAX_ACTIVE_BATCHES = N_WORKERS * 2  # enough to keep all workers busy
CHUNK_SIZE_SAVE    = 10_000         # save & print progress every 10k saved positions
EPD_SAMPLE_LINES   = 500

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
    limit                = 'all',
    
    # Endgame filter (disabled by default)
    filter_endgame       = False,
    endgame_max_pieces   = 14,   
    endgame_no_queens    = True,
)

# -- Multipath Configuration ---------------------------------------------------
# You can define multiple input directories and their corresponding output directories.
# The script will process them in parallel or sequentially. If multiple inputs map
# to the same output directory, it will correctly append chunks and track progress per input.
IN_DIRS = [
    r'data\selfplay_raw_01-04_wdl',
    r'data\selfplay_raw_04-04_wdl',
    r'data\selfplay_raw_10-04_wdl'
]

OUT_DIRS = [
    r'data\selfplay_01-04_wdl',
    r'data\selfplay_04-04_wdl',
    r'data\selfplay_10-04_wdl'
]

# Source configuration (applies to all paths)
SRC_TYPE = 'epd'
LIMIT = 'all'
SCORE_CAP_VALUE = 3000

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

# ===========================================================================
#  HELPERS
# ===========================================================================

def _normalise_pipeline(pipe) -> Dict[str, Any]:
    cfg = dict(DEFAULT_FILTERS)
    cfg.update(pipe)
    return cfg

def _progress_path(out_dir: str) -> str:
    return os.path.join(out_dir, 'progress.json')

def load_progress(out_dir: str) -> Dict[str, int]:
    p = _progress_path(out_dir)
    if os.path.exists(p):
        try:
            with open(p, 'r') as f:
                import json
                return json.load(f)
        except Exception as e:
            print(f"  [Warning] Could not read progress file: {e}")
    return {}

def save_progress(out_dir: str, consumed: Dict[str, int]):
    p   = _progress_path(out_dir)
    tmp = p + '.tmp'
    try:
        import json
        with open(tmp, 'w') as f:
            json.dump(consumed, f, indent=2)
        os.replace(tmp, p)
    except Exception as e:
        print(f"  [Warning] Could not save progress: {e}")

def fen_key(fen: str) -> str:
    return ' '.join(fen.split()[:4])

def format_time(s: float) -> str:
    if s < 60:   return f"{s:.0f}s"
    if s < 3600: return f"{s/60:.1f}m"
    return f"{s/3600:.1f}h"

def result_to_wdl(res_str: Optional[str]) -> float:
    if not res_str:
        return 0.5
    if '1-0' in res_str:
        return 1.0
    elif '0-1' in res_str:
        return 0.0
    elif '1/2' in res_str:
        return 0.5
    return 0.5

# ===========================================================================
#  PARSING & SIZE ESTIMATION
# ===========================================================================

def _extract_op(ops_str: str, key: str):
    m = re.search(rf'\b{key}\s+"([^"]*)"', ops_str)
    if m: return m.group(1).strip()
    m = re.search(rf'\b{key}\s+([^;]+)', ops_str)
    if m: return m.group(1).strip()
    return None

def parse_epd_line(line: str):
    line = line.strip()
    if not line or line.startswith(('#', '%')):
        return None
    parts = line.split()
    if len(parts) < 4:
        return None
    fen     = ' '.join(parts[:4]) + ' 0 1'
    ops_str = ' '.join(parts[4:])
    
    res = _extract_op(ops_str, 'c0')
    if not res:
        res = _extract_op(ops_str, 'result')
        
    cp_val = _extract_op(ops_str, 'ce')
    if cp_val is None:
        cp_val = _extract_op(ops_str, 'c1')
    cp = None
    if cp_val is not None:
        try:
            cp = float(cp_val)
        except:
            pass

    return {'fen': fen, 'result': res, 'id': _extract_op(ops_str, 'id'), 'cp': cp}

def estimate_source_size(cfg: Dict) -> Optional[int]:
    path     = cfg['in_dir']
    src_type = cfg['src_type']

    if not os.path.exists(path):
        return None

    try:
        p = pathlib.Path(path)
        if src_type == 'parquet':
            import pyarrow.parquet as pq
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
    if cfg['endgame_no_queens'] and board.queens:
        return False, "endgame_has_queen"

    if cfg['endgame_max_pieces'] is not None:
        if chess.popcount(board.occupied) > cfg['endgame_max_pieces']:
            return False, "endgame_too_many_pieces"

    return True, ""

# ===========================================================================
#  WORKER PROCESS
# ===========================================================================

def _worker_pipeline(records_batch: List[Dict], cfg: Dict):
    results: List[Dict] = []
    stats:   Dict       = {}

    for rec in records_batch:
        fen = rec.get('fen', '')
        if not fen:
            stats['bad_fen'] = stats.get('bad_fen', 0) + 1
            continue
            
        # 1. Chess Logic Filters
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

        # 2. Extract CP
        cp_white = None
        if 'cp' in rec and rec['cp'] is not None:
            try:
                cp_white = float(rec['cp'])
            except:
                pass
        
        if cp_white is None:
            stats['missing_cp'] = stats.get('missing_cp', 0) + 1
            continue
            
        # 3. Apply Score Cap Filter
        if cfg['filter_score_cap'] and abs(cp_white) > cfg['score_cap_value']:
            stats['score_cap'] = stats.get('score_cap', 0) + 1
            continue

        # 4. WDL Blending Logic
        # Calculate WDL_original from string result (or fallback to existing wdl key if parsed from parquet)
        if 'wdl' in rec and rec['wdl'] is not None and not isinstance(rec['wdl'], str):
            wdl_original = float(rec['wdl'])
        else:
            res_str = rec.get('result', '')
            wdl_original = result_to_wdl(res_str)

        # Calculate WDL_fromcp
        # Formula: 1.0 / (1.0 + exp(-cp/320.0))
        # Protect against overflow
        try:
            wdl_fromcp = 1.0 / (1.0 + np.exp(-cp_white / 320.0))
        except OverflowError:
            wdl_fromcp = 1.0 if cp_white > 0 else 0.0
            
        # Calculate Blended Score
        wdl_score = LAMBDA_WDL * wdl_original + (1.0 - LAMBDA_WDL) * wdl_fromcp

        results.append({
            'fen': rec['fen'], 
            'cp': cp_white, 
            'wdl': wdl_score,
            'result': rec.get('result', ''), 
            'id': rec.get('id', '')
        })

    return results, stats

# ===========================================================================
#  DATA LOADER
# ===========================================================================

def get_positions_generator(cfg: Dict, resume_consumed: int):
    import pyarrow.parquet as pq

    path     = cfg['in_dir']
    src_type = cfg['src_type']
    limit    = cfg['limit']

    if not os.path.exists(path):
        print(f"  [Warning] Input Source not found: {path}")
        return

    skip = resume_consumed

    eg_str   = (f"  endgame<={cfg['endgame_max_pieces']}pc"
                + (" no-Q" if cfg['endgame_no_queens'] else "")
                if cfg['filter_endgame'] else "")
    skip_str = f"  (resuming: skipping first {skip:,})" if skip else ""
    print(f"  Source: {os.path.basename(path)} ({src_type}) "
          f"cap={cfg['score_cap_value']}{eg_str}{skip_str}")

    # ── EPD ─────────────────────────────────────────────────────────────
    if src_type == 'epd':
        p = pathlib.Path(path)
        files = [str(f) for f in p.rglob('*.epd')]
        if not files:
            return

        if limit == 'all':
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
                        yield rec, consumed
        else:
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

            random.shuffle(reservoir) 
            consumed = 0
            for rec in reservoir:
                if consumed < skip:
                    consumed += 1
                    continue
                consumed += 1
                yield rec, consumed

    # ── Parquet ─────────────────────────────────────────────────────────
    elif src_type == 'parquet':
        p = pathlib.Path(path)
        files = [str(f) for f in p.rglob('*.parquet')]
        if not files:
            return
        random.shuffle(files)

        per_file_limit = math.ceil(limit / len(files)) if limit != 'all' else None

        consumed    = 0   
        count       = 0   

        for fpath in files:
            if limit != 'all' and count >= limit:
                break
            try:
                pf   = pq.ParquetFile(fpath)
                cols = [c for c in ['fen', 'cp', 'wdl', 'result', 'id']
                        if c in pf.schema_arrow.names]
                file_count = 0
                for batch in pf.iter_batches(batch_size=100_000, columns=cols):
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
                        yield {col: col_lists[col][i] for col in cols}, consumed
                    if limit != 'all' and count >= limit:
                        break
                    if per_file_limit and file_count >= per_file_limit:
                        break
            except Exception as e:
                print(f"  [Warning] Error reading {fpath}: {e}")

# ===========================================================================
#  PIPELINE EXECUTION
# ===========================================================================

def run_pipeline(cfg: Dict):
    in_dir  = cfg['in_dir']
    out_dir = cfg['out_dir']
    limit   = cfg['limit']
    
    print(f"\n{'='*60}")
    print(f"  Starting Pipeline")
    print(f"  IN  : {in_dir}")
    print(f"  OUT : {out_dir}")
    print(f"{'='*60}")
    
    os.makedirs(out_dir, exist_ok=True)

    print("  Estimating source size...")
    est_size = estimate_source_size(cfg)
    total_estimated_input = None
    if est_size is not None:
        total_estimated_input = est_size if limit == 'all' else min(est_size, limit)
        est_str = f"~{est_size:,} records"
    else:
        est_str = "size unknown"
    
    lim_str = "all" if limit == 'all' else f"{limit:,}"
    print(f"    Input size: {est_str} (limit={lim_str})")

    # ── Resume ───────────────────────────────────────────────────────────────
    seen_fens: set = set()
    existing       = sorted(glob.glob(os.path.join(out_dir, 'chunk_*.parquet')))
    start_chunk    = len(existing)
    
    progress_dict = load_progress(out_dir)
    pipeline_key = f"{in_dir}|{cfg.get('src_type', 'epd')}"
    resume_consumed = progress_dict.get(pipeline_key, 0)

    if start_chunk > 0:
        print(f'\n  Resuming from chunk {start_chunk}.')
        if resume_consumed:
            print(f"    Skipping {resume_consumed:,} already-consumed records")

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
    saved_this_run = 0   
    sent_this_run  = 0   

    pending_save = []
    chunk_idx    = start_chunk
    t_start      = time.time()

    print(f"  Workers    : {N_WORKERS}  |  "
          f"Batch : {BATCH_SIZE:,}  |  "
          f"Max in-flight : {MAX_ACTIVE_BATCHES}")
    print("  Starting worker pool...\n")

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures: Dict = {}   
        producer      = get_positions_generator(cfg, resume_consumed)
        exhausted     = False

        _buf_recs: List[Dict] = []
        _current_consumed_count = resume_consumed

        def _flush_buf():
            nonlocal _buf_recs
            if _buf_recs:
                fut = pool.submit(_worker_pipeline, list(_buf_recs), cfg)
                futures[fut] = time.time()
                _buf_recs = []

        while not exhausted or futures:

            # ── 1. Feed pool ─────────────────────────────────────────────────
            while not exhausted and len(futures) < MAX_ACTIVE_BATCHES:
                try:
                    rec, _current_consumed_count = next(producer)

                    if cfg.get('filter_duplicates', True):
                        k = fen_key(rec['fen'])
                        if k in seen_fens:
                            total_rej['duplicate'] = total_rej.get('duplicate', 0) + 1
                            continue
                        seen_fens.add(k)

                    sent_this_run += 1
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

                        save_path = os.path.join(out_dir, f'chunk_{chunk_idx:04d}.parquet')
                        pd.DataFrame(save_slice).to_parquet(save_path, index=False)

                        total_saved    += CHUNK_SIZE_SAVE
                        saved_this_run += CHUNK_SIZE_SAVE
                        elapsed         = time.time() - t_start

                        progress_dict[pipeline_key] = _current_consumed_count
                        save_progress(out_dir, progress_dict)

                        pos_per_sec = saved_this_run / max(elapsed, 1e-9)
                        pass_rate   = (saved_this_run / max(sent_this_run, 1)) * 100

                        eta_str = ''
                        if pos_per_sec > 0:
                            if total_estimated_input is not None and pass_rate > 0:
                                remaining_input = max(0, total_estimated_input - _current_consumed_count)
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
            save_path = os.path.join(out_dir, f'chunk_{chunk_idx:04d}.parquet')
            pd.DataFrame(pending_save).to_parquet(save_path, index=False)
            total_saved    += len(pending_save)
            saved_this_run += len(pending_save)
            progress_dict[pipeline_key] = _current_consumed_count
            save_progress(out_dir, progress_dict)
            print(f'  Final chunk {chunk_idx:04d} | '
                  f'+{len(pending_save):,} | Total: {total_saved:,}')

    elapsed = time.time() - t_start
    print(f'\n  DONE with Pipeline')
    print(f'    Total saved  : {total_saved:,}')
    print(f'    This run     : {saved_this_run:,} saved / {sent_this_run:,} sent to workers')
    print(f'    Rejections   : {total_rej}')
    print(f'    Elapsed      : {format_time(elapsed)}')
    print(f'    Throughput   : {saved_this_run / max(elapsed, 1e-9):.1f} pos/s')


def main():
    if len(IN_DIRS) != len(OUT_DIRS):
        print("Error: IN_DIRS and OUT_DIRS must have the same length.")
        return

    base_cfg = _normalise_pipeline({
        'src_type': SRC_TYPE,
        'limit': LIMIT,
        'score_cap_value': SCORE_CAP_VALUE
    })

    for in_dir, out_dir in zip(IN_DIRS, OUT_DIRS):
        cfg = dict(base_cfg)
        cfg['in_dir'] = in_dir
        cfg['out_dir'] = out_dir
        run_pipeline(cfg)


if __name__ == '__main__':
    main()
