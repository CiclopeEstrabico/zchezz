#!/usr/bin/env python3
"""validate_book.py — Validate all opening book entries for legality and quality.

1. Parse every FEN key and verify it's a legal chess position
2. Verify every move is legal from that position
3. (Optional) Use Stockfish to evaluate each position+move and flag bad moves (>-100cp)
4. Report and optionally generate a cleaned book
"""
import re, sys, io, os, json, subprocess, time, glob

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import chess

# Auto-detect the latest engine version
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE_C_DIR = os.path.join(BASE_DIR, "engine", "c")
_engine_dirs = sorted(glob.glob(os.path.join(ENGINE_C_DIR, "zchezz_v*")))
LATEST_ENGINE_DIR = _engine_dirs[-1] if _engine_dirs else os.path.join(ENGINE_C_DIR, "zchezz_v305")

HTML_PATH = os.path.join(LATEST_ENGINE_DIR, "zchezz_wasm.html")
# Try to find Stockfish for evaluation
SF_PATHS = [
    os.path.join(BASE_DIR, "engine", "stockfish", "stockfish.exe"),
]
SF_PATH = None
for p in SF_PATHS:
    if os.path.exists(p):
        SF_PATH = p
        break

# Also try the Zchezz engine itself for evaluation
ZCHEZZ_PATH = os.path.join(LATEST_ENGINE_DIR, "zchezz.exe")


def extract_book(html_path):
    """Extract OPENING_BOOK entries from HTML."""
    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the OPENING_BOOK block
    start = content.find('const OPENING_BOOK = {')
    end = content.find('};', start)
    book_text = content[start:end+2]
    
    entries = re.findall(r'"([^"]+)":\s*\[([^\]]+)\]', book_text)
    book = {}
    for fen_key, moves_str in entries:
        moves = [m.strip().strip('"') for m in moves_str.split(',')]
        book[fen_key] = moves
    return book


def fen_key_to_full_fen(fen_key):
    """Convert 4-field FEN key to full 6-field FEN."""
    parts = fen_key.split()
    if len(parts) == 4:
        return f"{parts[0]} {parts[1]} {parts[2]} {parts[3]} 0 1"
    return fen_key


def validate_legality(book):
    """Check every FEN is valid and every move is legal."""
    errors = []
    warnings = []
    
    for fen_key, moves in book.items():
        full_fen = fen_key_to_full_fen(fen_key)
        
        # 1. Check FEN is valid
        try:
            board = chess.Board(full_fen)
        except ValueError as e:
            errors.append(f"INVALID FEN: {fen_key} — {e}")
            continue
        
        if not board.is_valid():
            errors.append(f"INVALID POSITION: {fen_key}")
            continue
        
        # 2. Check each move is legal
        bad_moves = []
        for uci_move in moves:
            try:
                move = chess.Move.from_uci(uci_move)
            except ValueError:
                bad_moves.append(uci_move)
                errors.append(f"INVALID UCI: {fen_key} → {uci_move}")
                continue
            
            if move not in board.legal_moves:
                bad_moves.append(uci_move)
                errors.append(f"ILLEGAL MOVE: {fen_key} → {uci_move}")
    
    return errors, warnings


def evaluate_with_engine(book, engine_path, depth=10, max_positions=None):
    """Evaluate each position+move with an engine to find bad moves.
    Returns dict: {(fen_key, move): score_cp}
    """
    proc = subprocess.Popen(
        [engine_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, bufsize=1,
        cwd=os.path.dirname(engine_path)
    )
    
    def send(cmd):
        proc.stdin.write(cmd + "\n")
        proc.stdin.flush()
    
    def read_until(target, timeout=10):
        deadline = time.time() + timeout
        lines = []
        while time.time() < deadline:
            line = proc.stdout.readline().strip()
            if not line: continue
            lines.append(line)
            if target in line: return lines
        return lines
    
    send("uci")
    read_until("uciok")
    send("setoption name Threads value 1")
    send("isready")
    read_until("readyok")
    
    results = {}
    positions = list(book.items())
    if max_positions:
        positions = positions[:max_positions]
    
    total = len(positions)
    for i, (fen_key, moves) in enumerate(positions):
        full_fen = fen_key_to_full_fen(fen_key)
        
        try:
            board = chess.Board(full_fen)
        except:
            continue
        
        for uci_move in moves:
            try:
                move = chess.Move.from_uci(uci_move)
                if move not in board.legal_moves:
                    continue
            except:
                continue
            
            # Make the move and evaluate the resulting position
            board_copy = board.copy()
            board_copy.push(move)
            result_fen = board_copy.fen()
            
            send("ucinewgame")
            send(f"position fen {result_fen}")
            send(f"go depth {depth}")
            
            lines = read_until("bestmove", timeout=5)
            
            # Parse the last "info" line for score
            score_cp = None
            for line in reversed(lines):
                if "score cp" in line:
                    m = re.search(r'score cp (-?\d+)', line)
                    if m:
                        score_cp = int(m.group(1))
                        # Negate because we evaluated AFTER the move (opponent's perspective)
                        score_cp = -score_cp
                        break
                elif "score mate" in line:
                    m = re.search(r'score mate (-?\d+)', line)
                    if m:
                        mate = int(m.group(1))
                        score_cp = -30000 if mate < 0 else 30000
                        score_cp = -score_cp
                        break
            
            if score_cp is not None:
                results[(fen_key, uci_move)] = score_cp
        
        if (i + 1) % 100 == 0:
            print(f"    Evaluated {i+1}/{total} positions...", flush=True)
    
    send("quit")
    try:
        proc.wait(timeout=3)
    except:
        proc.kill()
    
    return results


def main():
    print("=" * 65)
    print("  Opening Book Validation")
    print("=" * 65)
    
    # Extract book
    print("\n[1] Extracting book from HTML...")
    book = extract_book(HTML_PATH)
    print(f"    {len(book)} positions, {sum(len(v) for v in book.values())} total moves")
    
    # Legality check
    print("\n[2] Checking legality...")
    errors, warnings = validate_legality(book)
    
    if errors:
        print(f"    ❌ {len(errors)} errors found!")
        for e in errors[:20]:
            print(f"      {e}")
        if len(errors) > 20:
            print(f"      ... and {len(errors)-20} more")
    else:
        print(f"    ✅ All positions and moves are legal!")
    
    # Engine evaluation
    engine = ZCHEZZ_PATH if os.path.exists(ZCHEZZ_PATH) else SF_PATH
    if engine:
        print(f"\n[3] Evaluating move quality with engine ({os.path.basename(engine)})...")
        print(f"    Using depth 8 for speed...")
        scores = evaluate_with_engine(book, engine, depth=8)
        
        # Find bad moves (losing more than 100cp = 1 pawn)
        bad_moves = [(k, v) for k, v in scores.items() if v < -100]
        dubious_moves = [(k, v) for k, v in scores.items() if -100 <= v < -50]
        
        print(f"\n    Total evaluated: {len(scores)}")
        print(f"    Bad moves (>1 pawn loss): {len(bad_moves)}")
        print(f"    Dubious moves (0.5-1 pawn loss): {len(dubious_moves)}")
        
        if bad_moves:
            print(f"\n    ❌ Bad moves to remove:")
            bad_sorted = sorted(bad_moves, key=lambda x: x[1])
            for (fen, move), score in bad_sorted[:30]:
                print(f"      {score:+5d}cp  {fen[:40]}... → {move}")
            
            # Generate cleaned book
            print(f"\n[4] Generating cleaned book...")
            bad_set = set(k for k, v in bad_moves)
            cleaned = {}
            removed = 0
            for fen_key, moves in book.items():
                good_moves = [m for m in moves if (fen_key, m) not in bad_set]
                if good_moves:
                    cleaned[fen_key] = good_moves
                else:
                    removed += 1
            
            print(f"    Removed {len(bad_set)} bad moves, {removed} empty positions")
            print(f"    Cleaned book: {len(cleaned)} positions")
            
            # Write cleaned book
            lines = []
            for fk in sorted(cleaned.keys()):
                ms = ', '.join(f'"{m}"' for m in cleaned[fk])
                lines.append(f'      "{fk}": [{ms}]')
            
            out_path = r"c:\Zchezz\tests\cleaned_book.js"
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write("    const OPENING_BOOK = {\n")
                f.write(",\n".join(lines))
                f.write("\n    };\n")
            
            print(f"    Written to {out_path}")
        else:
            print(f"\n    ✅ No bad moves found! Book is clean.")
    else:
        print("\n[3] No engine found for evaluation — skipping quality check")
    
    print(f"\n{'=' * 65}")
    print("  Done!")
    print("=" * 65)


if __name__ == "__main__":
    main()
