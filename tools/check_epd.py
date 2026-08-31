#!/usr/bin/env python3
"""Basic structural validation for EPD/FEN collections."""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def valid_fen_prefix(text: str) -> tuple[bool, str]:
    fields = text.strip().split()
    if len(fields) < 4:
        return False, "fewer than four FEN fields"
    board, turn, castling, ep = fields[:4]
    ranks = board.split("/")
    if len(ranks) != 8:
        return False, "board does not contain eight ranks"
    valid_pieces = set("prnbqkPRNBQK")
    for rank in ranks:
        count = 0
        for char in rank:
            if char.isdigit() and "1" <= char <= "8":
                count += int(char)
            elif char in valid_pieces:
                count += 1
            else:
                return False, f"invalid board character {char!r}"
        if count != 8:
            return False, f"rank expands to {count} squares"
    if turn not in ("w", "b"):
        return False, "side to move is not w/b"
    if castling != "-" and any(char not in "KQkq" for char in castling):
        return False, "invalid castling field"
    if ep != "-" and (len(ep) != 2 or ep[0] not in "abcdefgh" or ep[1] not in "36"):
        return False, "invalid en-passant square"
    return True, ""


def scan(path: Path) -> tuple[int, list[str]]:
    failures: list[str] = []
    seen: set[str] = set()
    count = 0
    for line_no, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        count += 1
        fen = " ".join(line.split()[:4])
        ok, reason = valid_fen_prefix(line)
        if not ok:
            failures.append(f"{path}:{line_no}: {reason}")
        if fen in seen:
            failures.append(f"{path}:{line_no}: duplicate position")
        seen.add(fen)
    return count, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="EPD files; default scans repository *.epd outside artifacts")
    args = parser.parse_args()
    if args.paths:
        files = [Path(value) for value in args.paths]
    else:
        files = [path for path in ROOT.rglob("*.epd") if "artifacts" not in path.parts and ".git" not in path.parts]
    failures: list[str] = []
    total = 0
    for path in files:
        count, bad = scan(path)
        total += count
        failures.extend(bad)
        print(f"{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}: {count} position(s)")
    for failure in failures:
        print("FAIL:", failure)
    print(f"checked {len(files)} file(s), {total} position(s), {len(failures)} issue(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

