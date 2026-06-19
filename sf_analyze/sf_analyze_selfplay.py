"""
sf_analyze_epd.py
─────────────────
Reads all .epd files in INPUT_DIR, filters positions, evaluates each one
with Stockfish (side-to-move aware), and saves annotated chunks to OUT_DIR.

Output format  : Parquet files (chunk_NNNN.parquet), each ≤ CHUNK_SIZE rows.
Output columns : fen | cp | wdl | result | id

Filters (all individually toggleable via FILTER_* flags):
  • Duplicate FENs          (clock-field-stripped)
  • Positions in check
  • Checkmate / stalemate positions (terminal)
  • Positions with a winning capture available (non-quiet)
      — "winning capture": attacker value < victim value
  • Positions with an equal capture available (non-quiet)
      — "equal capture":   attacker value ≈ victim value (±100 cp tolerance)
  • Positions with eval outside [-FILTER_SCORE_CAP, +FILTER_SCORE_CAP] cp
      (removes wildly won/lost positions of limited training value)

Smart resume: already-written chunks are counted; the pipeline skips the
corresponding input positions automatically.

Usage
─────
  python sf_analyze_epd.py            # uses CONFIG below
  python sf_analyze_epd.py --help     # prints this docstring

All paths and knobs live in the CONFIG section — no CLI args needed.
"""

import os, sys, re, glob, time, gc, chess, argparse
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG  ← edit here
# ═══════════════════════════════════════════════════════════════════════════

SF_PATH    = r"engine\stockfish_fast\stockfish\stockfish-windows-x86-64-avxvnni.exe"
INPUT_DIR  = r'data\selfplay_raw_wdl'
OUT_DIR    = r'data\selfplay_sf_nodes1M_wdl'

SF_NODES   = 1_000_000      # Stockfish nodes per position (higher = stronger/slower)
SF_HASH_MB = 16         # Stockfish hash table size in MB per worker process
N_WORKERS  = max(1, (os.cpu_count()) - 1)  # parallel SF processes
CHUNK_SIZE = 50_000     # positions per output parquet file

# ── Filters ────────────────────────────────────────────────────────────────
FILTER_DUPLICATES    = True   # drop repeated FENs (4-field key)
FILTER_IN_CHECK      = True   # drop positions where side-to-move is in check
FILTER_TERMINAL      = True   # drop checkmate / stalemate positions
FILTER_WIN_CAPTURE   = True   # drop positions with a winning capture (SEE > 0)
FILTER_EQUAL_CAPTURE = True   # drop positions with an equal capture (SEE ≈ 0)
FILTER_SCORE_CAP     = True   # drop positions with |eval| > SCORE_CAP_VALUE
SCORE_CAP_VALUE      = 3000   # centipawns — only used when FILTER_SCORE_CAP=True

# Piece values for capture filtering (centipawns)
PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   20000,
}
EQUAL_CAPTURE_TOLERANCE = 50  # cp — captures within this margin count as "equal"

# ═══════════════════════════════════════════════════════════════════════════

# ── EPD parser ─────────────────────────────────────────────────────────────

def parse_epd_line(line: str):
    """
    Parse one EPD line.  Returns a dict with at minimum:
        fen    — full FEN string (4-field prefix + ' 0 1')
        result — "1-0" | "0-1" | "1/2-1/2" | None
        id     — opaque string or None
        bm     — best-move UCI string or None
    Returns None if the line is malformed or a comment.
    """
    line = line.strip()
    if not line or line.startswith('#') or line.startswith('%'):
        return None

    # The first 4 space-separated tokens are the FEN fields
    parts = line.split()
    if len(parts) < 4:
        return None

    epd_prefix = ' '.join(parts[:4])          # "board side castling ep"
    fen = epd_prefix + ' 0 1'                 # make it a full FEN

    # Parse operations (key "value"; or key token;)
    ops_str = ' '.join(parts[4:])
    result = _op(ops_str, 'c0')
    id_val = _op(ops_str, 'id')
    bm_val = _op(ops_str, 'bm')

    return {'fen': fen, 'result': result, 'id': id_val, 'bm': bm_val}


_OP_RE = re.compile(r'(\w+)\s+"([^"]*)"')    # key "value"
_OP_TOKEN_RE = re.compile(r'(\w+)\s+([^;]+)') # key token(s)

def _op(ops_str: str, key: str):
    """Extract the value of an EPD operation by key, or None."""
    m = re.search(rf'\b{key}\s+"([^"]*)"', ops_str)
    if m:
        return m.group(1).strip()
    m = re.search(rf'\b{key}\s+([^;]+)', ops_str)
    if m:
        return m.group(1).strip()
    return None


# ── Position filters ───────────────────────────────────────────────────────

def _has_winning_or_equal_capture(board: chess.Board) -> tuple[bool, bool]:
    """
    Returns (has_winning_capture, has_equal_capture).
    A capture is 'winning'  if victim_value > attacker_value + TOLERANCE.
    A capture is 'equal'    if |victim_value - attacker_value| <= TOLERANCE.
    Uses a simple static exchange approximation (no full SEE tree).

    Pawn-captures-pawn is intentionally excluded from the equal-capture
    check: pawn trades are a normal part of positional play and should not
    cause a position to be filtered out as non-quiet.  Pawn captures of
    any other piece type (knight, bishop, rook, queen) are still checked
    as winning captures since the pawn is gaining material.
    """
    has_win = False
    has_eq  = False
    for move in board.legal_moves:
        if not board.is_capture(move):
            continue

        attacker_piece = board.piece_at(move.from_square)
        if attacker_piece is None:
            continue
        attacker_type = attacker_piece.piece_type
        attacker_val  = PIECE_VALUES.get(attacker_type, 0)

        # En-passant: captured piece is always a pawn, not on to_square
        if board.is_en_passant(move):
            victim_type = chess.PAWN
            victim_val  = PIECE_VALUES[chess.PAWN]
        else:
            victim_piece = board.piece_at(move.to_square)
            if victim_piece is None:
                continue
            victim_type = victim_piece.piece_type
            victim_val  = PIECE_VALUES.get(victim_type, 0)

        # ── Pawn x Pawn: skip entirely (normal positional exchange) ──────
        if attacker_type == chess.PAWN and victim_type == chess.PAWN:
            continue

        diff = victim_val - attacker_val
        if diff > EQUAL_CAPTURE_TOLERANCE:
            has_win = True
        elif abs(diff) <= EQUAL_CAPTURE_TOLERANCE:
            has_eq = True

        if has_win and has_eq:
            break   # no need to continue further

    return has_win, has_eq


def is_quiet(board: chess.Board) -> tuple[bool, str]:
    """
    Returns (passes_all_filters, reason_for_rejection).
    An empty reason string means the position is accepted.
    """
    if FILTER_TERMINAL:
        if board.is_checkmate() or board.is_stalemate():
            return False, "terminal"
        if not any(True for _ in board.legal_moves):   # no legal moves (shouldn't happen after above)
            return False, "no_moves"

    if FILTER_IN_CHECK:
        if board.is_check():
            return False, "in_check"

    if FILTER_WIN_CAPTURE or FILTER_EQUAL_CAPTURE:
        has_win, has_eq = _has_winning_or_equal_capture(board)
        if FILTER_WIN_CAPTURE and has_win:
            return False, "winning_capture"
        if FILTER_EQUAL_CAPTURE and has_eq:
            return False, "equal_capture"

    return True, ""


def fen_key(fen: str) -> str:
    """4-field dedup key (strips halfmove + fullmove counters)."""
    return ' '.join(fen.split()[:4])


# ── Stockfish worker ───────────────────────────────────────────────────────

def _sf_eval_batch(args):
    """
    Worker function: receives a list of FEN strings, evaluates each with
    Stockfish, and returns a list of {'fen', 'cp', 'wdl'} dicts.

    Critically: Stockfish always reports score from the *perspective of the
    side to move*, so we negate the cp when it's Black to move so that the
    stored cp is always White-relative (positive = good for White).
    """
    fens, sf_path, nodes, hash_mb = args

    proc = subprocess.Popen(
        [sf_path],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, bufsize=1
    )

    def write(cmd):
        proc.stdin.write(cmd + '\n')
        proc.stdin.flush()

    write('uci')
    write(f'setoption name Hash value {hash_mb}')
    write('setoption name Threads value 1')
    write('isready')
    for line in proc.stdout:
        if 'readyok' in line:
            break

    re_cp   = re.compile(r'score cp (-?\d+)')
    re_mate = re.compile(r'score mate (-?\d+)')

    results = []

    for fen in fens:
        # Determine side to move from FEN
        side = fen.split()[1]   # 'w' or 'b'

        write(f'position fen {fen}')
        write(f'go nodes {nodes}')

        last_cp = 0
        for line in proc.stdout:
            if 'info' in line and 'score' in line:
                m = re_mate.search(line)
                if m:
                    last_cp = 30_000 if int(m.group(1)) > 0 else -30_000
                else:
                    m = re_cp.search(line)
                    if m:
                        last_cp = int(m.group(1))
            if 'bestmove' in line:
                break

        # Stockfish returns score relative to side-to-move.
        # Normalise to White-relative so that all training labels are consistent.
        cp_white = last_cp if side == 'w' else -last_cp

        # WDL sigmoid (White wins = 1.0)
        wdl = 1.0 / (1.0 + np.exp(-cp_white / 320.0))
        results.append({'fen': fen, 'cp': cp_white, 'wdl': wdl})

    write('quit')
    try:
        proc.terminate()
    except Exception:
        pass

    return results


# ── Input loading ──────────────────────────────────────────────────────────

def iter_epd_files(input_dir: str):
    """Yields file paths for every .epd file found, sorted, without duplicates.

    Previously this used two globs:
        glob('**/*.epd', recursive=True)   ← matches root + all subdirs
        glob('*.epd')                      ← matches root only  ← DUPLICATE
    Files sitting directly in input_dir appeared in both results and were
    processed twice.  The fix is a single recursive glob, which already
    covers root-level files.
    """
    return sorted(glob.glob(os.path.join(input_dir, '**', '*.epd'), recursive=True))


def load_and_filter_epds(epd_files: list[str], skip: int, seen_fens: set) -> tuple[list[dict], int]:
    """
    Reads EPD files in order, skips the first `skip` positions (resume),
    applies all enabled filters, and returns (accepted_records, total_read).

    Each accepted record: {'fen': str, 'result': str|None, 'id': str|None}
    We read exactly one "pass" — callers should call in a loop to fill chunks.
    """
    accepted = []
    total_read = 0
    rejected_counts = {}

    for epd_path in epd_files:
        with open(epd_path, 'r', encoding='utf-8', errors='ignore') as fh:
            for raw_line in fh:
                rec = parse_epd_line(raw_line)
                if rec is None:
                    continue

                total_read += 1

                # Resume: skip positions already processed
                if total_read <= skip:
                    if FILTER_DUPLICATES:
                        seen_fens.add(fen_key(rec['fen']))
                    continue

                fen = rec['fen']

                # Duplicate filter
                if FILTER_DUPLICATES:
                    key = fen_key(fen)
                    if key in seen_fens:
                        rejected_counts['duplicate'] = rejected_counts.get('duplicate', 0) + 1
                        continue
                    seen_fens.add(key)

                # Board-level filters
                try:
                    board = chess.Board(fen)
                except ValueError:
                    rejected_counts['bad_fen'] = rejected_counts.get('bad_fen', 0) + 1
                    continue

                ok, reason = is_quiet(board)
                if not ok:
                    rejected_counts[reason] = rejected_counts.get(reason, 0) + 1
                    continue

                accepted.append(rec)

    return accepted, total_read, rejected_counts


# ── Main pipeline ──────────────────────────────────────────────────────────

def main():
    if '--help' in sys.argv or '-h' in sys.argv:
        print(__doc__)
        return

    print()
    print('═' * 60)
    print('  sf_analyze_epd.py  — EPD filter + Stockfish annotator')
    print('═' * 60)
    print(f'  Input dir  : {INPUT_DIR}')
    print(f'  Output dir : {OUT_DIR}')
    print(f'  SF path    : {SF_PATH}')
    print(f'  SF nodes   : {SF_NODES:,}')
    print(f'  Workers    : {N_WORKERS}')
    print(f'  Chunk size : {CHUNK_SIZE:,}')
    print()
    print('  Filters enabled:')
    print(f'    Duplicates     : {FILTER_DUPLICATES}')
    print(f'    In check       : {FILTER_IN_CHECK}')
    print(f'    Terminal       : {FILTER_TERMINAL}')
    print(f'    Win capture    : {FILTER_WIN_CAPTURE}')
    print(f'    Equal capture  : {FILTER_EQUAL_CAPTURE}  (tol. ±{EQUAL_CAPTURE_TOLERANCE} cp)')
    print(f'    Score cap      : {FILTER_SCORE_CAP}  (|eval| > {SCORE_CAP_VALUE} cp)')
    print()

    if not os.path.isfile(SF_PATH):
        print(f'ERROR: Stockfish not found at: {SF_PATH}')
        sys.exit(1)

    epd_files = iter_epd_files(INPUT_DIR)
    if not epd_files:
        print(f'ERROR: No .epd files found in {INPUT_DIR}')
        sys.exit(1)
    print(f'  Found {len(epd_files)} EPD file(s): {[os.path.basename(f) for f in epd_files]}')

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Smart resume ──────────────────────────────────────────────────────
    existing = sorted(glob.glob(os.path.join(OUT_DIR, 'chunk_*.parquet')))
    start_chunk = len(existing)
    # Count how many accepted (post-filter) positions are already saved
    already_saved = start_chunk * CHUNK_SIZE
    print(f'\n  Resume from chunk {start_chunk:04d} ({already_saved:,} positions already saved).')

    # We need to know how many raw EPD lines to skip so that the already-saved
    # accepted positions are not re-processed.  We can't know that exactly without
    # re-scanning; instead we track `seen_fens` from existing chunks' FEN column.
    seen_fens: set[str] = set()
    if start_chunk > 0 and FILTER_DUPLICATES:
        print('  Rebuilding seen-FEN set from existing chunks...')
        for p in existing:
            df = pd.read_parquet(p, columns=['fen'])
            for f in df['fen']:
                seen_fens.add(fen_key(f))
        print(f'  {len(seen_fens):,} FENs loaded.')

    # ── Read + filter all input into a flat list (streaming by chunk) ─────
    # We iterate EPD files once, building a buffer until it reaches CHUNK_SIZE,
    # then evaluate + save, then continue.

    chunk_idx   = start_chunk
    pending     = []          # accepted records not yet evaluated
    total_in    = 0           # total raw EPD lines read
    total_out   = 0           # total positions written
    total_rej   = {}          # rejection reason → count
    t_start     = time.time()

    def process_chunk(records: list[dict], cidx: int) -> int:
        """Evaluate a list of records with Stockfish, apply score cap, save."""
        fens = [r['fen'] for r in records]

        # Distribute across workers
        sub_size  = max(1, len(fens) // N_WORKERS)
        sub_lists = [fens[i:i + sub_size] for i in range(0, len(fens), sub_size)]
        args_list = [(sl, SF_PATH, SF_NODES, SF_HASH_MB) for sl in sub_lists if sl]

        sf_results = []
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = [pool.submit(_sf_eval_batch, a) for a in args_list]
            for fut in as_completed(futures):
                try:
                    sf_results.extend(fut.result())
                except Exception as e:
                    print(f'\n  [Worker error] {e}')

        if not sf_results:
            return 0

        # Build lookup: fen → (cp, wdl)
        sf_map = {r['fen']: (r['cp'], r['wdl']) for r in sf_results}

        rows = []
        for rec in records:
            pair = sf_map.get(rec['fen'])
            if pair is None:
                continue
            cp, wdl = pair

            # Score cap filter (applied here because we need the SF eval)
            if FILTER_SCORE_CAP and abs(cp) > SCORE_CAP_VALUE:
                total_rej['score_cap'] = total_rej.get('score_cap', 0) + 1
                continue

            rows.append({
                'fen':    rec['fen'],
                'cp':     cp,
                'wdl':    wdl,
                'result': rec.get('result', ''),
                'id':     rec.get('id', ''),
            })

        if not rows:
            return 0

        df = pd.DataFrame(rows)
        out_path = os.path.join(OUT_DIR, f'chunk_{cidx:04d}.parquet')
        df.to_parquet(out_path, index=False)
        del df; gc.collect()
        return len(rows)

    # ── Stream through EPD files ──────────────────────────────────────────
    print()
    skip_input = already_saved  # approximate: skip this many accepted positions
    # We handle resume by tracking how many accepted positions we've seen so far
    # across the scan, and only adding to `pending` after we've skipped enough.
    accepted_so_far = 0

    for epd_path in epd_files:
        fname = os.path.basename(epd_path)
        print(f'  Reading: {fname}')

        with open(epd_path, 'r', encoding='utf-8', errors='ignore') as fh:
            for raw_line in fh:
                rec = parse_epd_line(raw_line)
                if rec is None:
                    continue

                total_in += 1
                fen = rec['fen']

                # ── Duplicate filter ─────────────────────────────────────
                if FILTER_DUPLICATES:
                    key = fen_key(fen)
                    if key in seen_fens:
                        total_rej['duplicate'] = total_rej.get('duplicate', 0) + 1
                        continue
                    seen_fens.add(key)

                # ── Board-level filters ──────────────────────────────────
                try:
                    board = chess.Board(fen)
                except ValueError:
                    total_rej['bad_fen'] = total_rej.get('bad_fen', 0) + 1
                    continue

                ok, reason = is_quiet(board)
                if not ok:
                    total_rej[reason] = total_rej.get(reason, 0) + 1
                    continue

                # ── Resume skip ──────────────────────────────────────────
                accepted_so_far += 1
                if accepted_so_far <= already_saved:
                    # Position already in an existing chunk — skip evaluation
                    continue

                pending.append(rec)

                # ── Flush chunk ──────────────────────────────────────────
                if len(pending) >= CHUNK_SIZE:
                    t0 = time.time()
                    print(f'\n  ── Chunk {chunk_idx:04d} ({len(pending):,} positions) ──')
                    written = process_chunk(pending, chunk_idx)
                    elapsed = time.time() - t0
                    total_out += written
                    chunk_idx += 1
                    pending.clear()
                    gc.collect()
                    print(f'     Saved {written:,} rows | {written/elapsed:.1f} pos/s | '
                          f'Total out: {total_out:,}')

                    # Progress summary
                    elapsed_total = time.time() - t_start
                    rej_str = ', '.join(f'{k}: {v:,}' for k, v in total_rej.items())
                    print(f'     Raw in: {total_in:,} | Rejected: [{rej_str}]')

    # ── Final partial chunk ───────────────────────────────────────────────
    if pending:
        t0 = time.time()
        print(f'\n  ── Final chunk {chunk_idx:04d} ({len(pending):,} positions) ──')
        written = process_chunk(pending, chunk_idx)
        elapsed = time.time() - t0
        total_out += written
        print(f'     Saved {written:,} rows | {written/elapsed:.1f} pos/s')

    # ── Final report ──────────────────────────────────────────────────────
    elapsed_total = time.time() - t_start
    print()
    print('═' * 60)
    print('  DONE')
    print('═' * 60)
    print(f'  Raw EPD lines read : {total_in:,}')
    print(f'  Positions written  : {total_out + already_saved:,}  '
          f'({already_saved:,} pre-existing + {total_out:,} new)')
    print(f'  Chunks on disk     : {chunk_idx + (1 if pending else 0)}')
    print(f'  Total time         : {elapsed_total:.1f}s')
    print(f'  Average speed      : {total_out / max(1, elapsed_total):.1f} pos/s')
    print()
    print('  Rejection breakdown:')
    for reason, count in sorted(total_rej.items(), key=lambda x: -x[1]):
        print(f'    {reason:<20s}: {count:>10,}')
    print()
    print(f'  Output dir: {OUT_DIR}')
    print()


if __name__ == '__main__':
    main()
