#!/usr/bin/env python3
"""test_html_features.py — Validate key HTML features in zchezz_wasm.html
Checks: opening book, SVG placeholders, piece selector, clock management.
Does NOT require a browser — just parses the HTML.
"""
import re, sys, io, os, glob

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# Auto-detect the latest engine version
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE_C_DIR = os.path.join(BASE_DIR, "engine", "c")
_engine_dirs = sorted(glob.glob(os.path.join(ENGINE_C_DIR, "zchezz_v*")))
LATEST_ENGINE_DIR = _engine_dirs[-1] if _engine_dirs else os.path.join(ENGINE_C_DIR, "zchezz_v305")

HTML_PATH = os.path.join(LATEST_ENGINE_DIR, "zchezz_wasm.html")

with open(HTML_PATH, 'r', encoding='utf-8') as f:
    html = f.read()

print("=" * 60)
print("  HTML Feature Validation")
print("=" * 60)
errors = 0

# 1. Opening Book
print("\n[1] Opening Book...")
book_match = re.findall(r'"([^"]+)":\s*\[([^\]]+)\]', html[html.find('OPENING_BOOK'):html.find('OPENING_BOOK') + 300000])
print(f"    Entries: {len(book_match)}")
if len(book_match) < 500:
    print("    ⚠️  Book seems too small!")
    errors += 1
else:
    print("    ✅ Book has good coverage")

# Check key openings
start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"
found_start = any(fen == start_fen for fen, _ in book_match)
if found_start:
    for fen, moves in book_match:
        if fen == start_fen:
            move_list = [m.strip().strip('"') for m in moves.split(',')]
            print(f"    Start position has {len(move_list)} moves: {move_list[:6]}")
            if len(move_list) >= 4:
                print("    ✅ Good variety at start")
            else:
                print("    ⚠️  Low variety at start")
                errors += 1
            break
else:
    print("    ⚠️  Start position not in book!")
    errors += 1

# Check e4 response
e4_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -"
for fen, moves in book_match:
    if fen == e4_fen:
        move_list = [m.strip().strip('"') for m in moves.split(',')]
        print(f"    After e4: {len(move_list)} responses: {move_list[:6]}")
        if len(move_list) >= 3:
            print("    ✅ Good variety after e4")
        break

# 2. SVG Placeholders
print("\n[2] SVG Piece Sets...")
has_merida_placeholder = "PIECE_SVG_DATA" in html
has_cburnett_placeholder = "PIECE_SVG_CBURNETT" in html
has_staunty_placeholder = "PIECE_SVG_STAUNTY" in html
print(f"    Merida placeholder: {'✅' if has_merida_placeholder else '❌'}")
print(f"    CBurnett placeholder: {'✅' if has_cburnett_placeholder else '❌'}")
print(f"    Staunty placeholder: {'✅' if has_staunty_placeholder else '❌'}")
if not has_cburnett_placeholder:
    errors += 1

# 3. Piece Style Buttons
print("\n[3] Piece Style Buttons...")
buttons = re.findall(r'setPieceStyle\((\d+)\)', html)
unique_buttons = sorted(set(buttons))
print(f"    Button indices: {unique_buttons}")
expected = ['0', '1', '3', '4', '5']
if set(expected).issubset(set(unique_buttons)):
    print("    ✅ All piece style buttons present")
else:
    missing = set(expected) - set(unique_buttons)
    print(f"    ⚠️  Missing buttons: {missing}")
    errors += 1

# 4. setPieceStyle function
print("\n[4] setPieceStyle function...")
if "switchSvgSet('cburnett')" in html:
    print("    ✅ CBurnett SVG switching implemented")
else:
    print("    ⚠️  CBurnett switching not found")
    errors += 1
if "switchSvgSet('staunty')" in html:
    print("    ✅ Staunty SVG switching implemented")
else:
    print("    ⚠️  Staunty switching not found")
    errors += 1

# 5. Clock Management
print("\n[5] Clock Time Management...")
if "fullmove" in html and "movesLeft" in html:
    print("    ✅ Fullmove-based time management")
else:
    print("    ⚠️  Clock management may be outdated")
    
# Check new time constants
if "0.75" in html and "0.25" in html:
    print("    ✅ Conservative increment (0.75) and hardcap (0.25)")
else:
    print("    ⚠️  Time constants may be old")

# 6. loadSvgPieces
print("\n[6] SVG Loading System...")
if "loadOneSvgSet" in html:
    print("    ✅ Multi-set SVG loading implemented")
else:
    print("    ⚠️  Old single-set loading")
    errors += 1

if "switchSvgSet" in html:
    print("    ✅ SVG set switching function found")
else:
    print("    ⚠️  switchSvgSet missing")
    errors += 1

# Summary
print(f"\n{'=' * 60}")
if errors == 0:
    print("  ✅ All checks passed!")
else:
    print(f"  ⚠️  {errors} issue(s) found")
print("=" * 60)
