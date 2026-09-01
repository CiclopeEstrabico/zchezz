#!/usr/bin/env python3
"""Validate an opening-book block embedded in a Zchezz HTML artifact."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "utils"))
from repo_paths import latest_version, wasm_bundle  # noqa: E402

# ═══════════════ CONFIGURATION ═══════════════
HTML_PATH = ""
EVALUATE = False
WRITE_CLEANED = False
# ═════════════════════════════════════════════

def parse_bool(text: str) -> bool:
    value = text.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {text!r}")

def extract_book(html: str) -> dict[str, list[str]]:
    marker = html.find("OPENING_BOOK")
    if marker < 0:
        return {}
    window = html[marker: marker + 500_000]
    entries = re.findall(r'"([^"]+)":\s*\[([^\]]+)\]', window)
    book: dict[str, list[str]] = {}
    for fen_key, moves_text in entries:
        moves = [m.strip().strip('"') for m in moves_text.split(",") if m.strip()]
        if moves:
            book[fen_key] = moves
    return book

def full_fen(key: str) -> str:
    parts = key.split()
    return f"{key} 0 1" if len(parts) == 4 else key

def validate(book: dict[str, list[str]]) -> list[str]:
    errors: list[str] = []
    for fen, moves in book.items():
        try:
            board = chess.Board(full_fen(fen))
        except ValueError as exc:
            errors.append(f"invalid FEN {fen!r}: {exc}")
            continue
        if not board.is_valid():
            errors.append(f"invalid chess position: {fen}")
            continue
        for text in moves:
            try:
                move = chess.Move.from_uci(text)
            except ValueError:
                errors.append(f"invalid UCI move {text!r} in {fen}")
                continue
            if move not in board.legal_moves:
                errors.append(f"illegal book move {text} in {fen}")
    return errors

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", default=HTML_PATH)
    parser.add_argument("--evaluate", type=parse_bool, default=EVALUATE)
    parser.add_argument("--write-cleaned", type=parse_bool, default=WRITE_CLEANED)
    args = parser.parse_args()

    path = Path(args.html) if args.html else wasm_bundle(latest_version())
    if not path.is_file():
        print(f"SKIP: HTML artifact not found: {path}")
        return 0

    html = path.read_text(encoding="utf-8", errors="replace")
    book = extract_book(html)
    if not book:
        print("FAIL: OPENING_BOOK block not found or empty")
        return 1

    errors = validate(book)
    print(f"positions: {len(book)}")
    print(f"moves: {sum(len(v) for v in book.values())}")

    if args.evaluate:
        print("INFO: engine quality evaluation is intentionally not part of this deterministic contract.")
    if args.write_cleaned:
        print("INFO: deterministic contract does not rewrite the book.")

    if errors:
        for error in errors[:50]:
            print(f"FAIL: {error}")
        if len(errors) > 50:
            print(f"FAIL: ... {len(errors) - 50} more")
        return 1

    print("PASS: all embedded opening-book positions and moves are legal")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
