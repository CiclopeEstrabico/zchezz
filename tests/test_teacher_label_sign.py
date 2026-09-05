#!/usr/bin/env python3
"""Regression test for the v3.14 -> v5 teacher score-frame contract."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train" / "labeling"))
import label_with_teacher as lwt


class FakeTeacher(lwt.UciTeacher):
    def __init__(self, lines):
        self.lines = lines
        self.nodes = 1
        self.timeout = 1.0
        self.re_cp = re.compile(r"score cp (-?\d+)")
        self.re_mate = re.compile(r"score mate (-?\d+)")

    def write(self, command):
        pass

    def read_until(self, marker, timeout):
        return self.lines


WHITE = "7k/8/8/8/8/8/4Q3/7K w - - 0 1"
BLACK = "7k/8/8/8/8/8/4Q3/7K b - - 0 1"

for fen in (WHITE, BLACK):
    assert FakeTeacher(["info depth 1 score cp 137", "bestmove h8g8"])._evaluate_once(fen) == 137
    assert FakeTeacher(["info depth 1 score cp -91", "bestmove h8g8"])._evaluate_once(fen) == -91
    assert FakeTeacher(["info depth 1 score mate 3", "bestmove h8g8"])._evaluate_once(fen) == lwt.CP_CLIP
    assert FakeTeacher(["info depth 1 score mate -2", "bestmove h8g8"])._evaluate_once(fen) == -lwt.CP_CLIP

print("teacher UCI white-relative score contract: OK")
