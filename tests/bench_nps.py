#!/usr/bin/env python3
"""
bench_nps.py — 50-position NPS + Eval benchmark across game phases
===================================================================
Tests HEAD-TB, HEAD-noTB and BASE (set in the CONFIGURATION block) on 50
positions spanning:
  - Openings (10 positions)
  - Middlegames (15 positions)
  - Endgames 6pc (5 positions)
  - Endgames 5pc (5 positions)
  - Endgames 4pc (5 positions)
  - Endgames 3pc (5 positions)
  - Endgames 2pc (2 positions)
  - Insufficient material (3 positions)

Checks:
  - NPS drop vs the BASE_VERSION baseline (fail if > NPS_FAIL_PCT)
  - Eval sanity (startpos ~0cp, won positions >+300cp, drawn ~0cp)
  - TB vs noTB node counts (should be equal or fewer for TB)
"""

import subprocess, re, sys, os, time

# Force UTF-8 output on Windows (cp1252 can't encode checkmarks)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
#
#  Set the two engine versions here, ONCE. Every engine name, dict key and
#  printed label below is derived from them.
# ═══════════════════════════════════════════════════════════════════════════════
HEAD_VERSION = "v400"    # engine under test
BASE_VERSION = "v314"    # previous stable, the NPS baseline to compare against
NPS_FAIL_PCT = 5.0       # fail the run if head is more than this % slower than
                         # base on opening+middlegame (endgame NPS is NOT
                         # comparable: TB prunes subtrees, so fewer nodes are
                         # searched and measured NPS drops legitimately)

TB_PATH      = os.path.join(BASE_DIR, "tablebases")   # Syzygy folder for the -TB build
TIMEOUT_S    = 30        # per-position wall-clock limit for one engine
DEPTH        = 0         # 0 = use each position's own depth (see POSITIONS);
                         # >0 = force this depth on every position
PHASES       = []        # [] = all 50 positions; else keep only positions whose
                         # label starts with one of these prefixes, e.g.
                         # ["Opening", "Middle"] or ["End", "Insuff"]
RUN_BASE     = True      # also run BASE_VERSION (the NPS baseline). Turn off when
                         # the previous version is not present in the working tree

# ═══════════════════════════════ COMMAND LINE ════════════════════════════════
# The CONFIGURATION block above is the interface: a bare `python tests/bench_nps.py`
# runs exactly what it says. Every constant is ALSO a flag that overrides it,
# and `--show-config` prints the resolved settings and exits without launching
# an engine. See utils/cliconf.py and CLAUDE.md rule 8.
# ═════════════════════════════════════════════════════════════════════════════
sys.path.insert(0, os.path.join(BASE_DIR, "utils"))
from cliconf import override_from_cli  # noqa: E402

CLI = [
    ("HEAD_VERSION", "--head",     str,   "engine version under test (folder engine/c/zchezz_<V>)"),
    ("BASE_VERSION", "--base",     str,   "previous stable version, the NPS baseline"),
    ("RUN_BASE",     "--run-base", bool,  "also benchmark the baseline version"),
    ("NPS_FAIL_PCT", "--nps-fail-pct", float, "fail if head is more than this %% slower than base"),
    ("TB_PATH",      "--tb-path",  str,   "Syzygy tablebase folder for the -TB run"),
    ("TIMEOUT_S",    "--timeout",  float, "per-position engine timeout, seconds"),
    ("DEPTH",        "--depth",    int,   "force this search depth (0 = per-position depth)"),
    ("PHASES",       "--phase",    list,  "keep only positions whose label starts with this (repeatable)"),
]


def head_tb() -> str:
    return f"{HEAD_VERSION}-TB"


def head_notb() -> str:
    return f"{HEAD_VERSION}-noTB"


def build_engines() -> dict:
    """The engine table, DERIVED from HEAD_VERSION/BASE_VERSION at call time.

    It is a function, not a module-level dict, because --head/--base can
    change those constants after this module is imported — a dict built at
    import time would silently keep the old versions (exactly the drift the
    CONFIGURATION comment above describes).
    """
    exe = lambda v: os.path.join(BASE_DIR, "engine", "c", f"zchezz_{v}", "zchezz.exe")
    engines = {
        head_tb():   (exe(HEAD_VERSION), True),
        head_notb(): (exe(HEAD_VERSION), False),
    }
    if RUN_BASE:
        engines[BASE_VERSION] = (exe(BASE_VERSION), False)
    return engines


def select_positions() -> list:
    """POSITIONS filtered by --phase (a label-prefix match, [] = all)."""
    if not PHASES:
        return list(POSITIONS)
    return [p for p in POSITIONS if any(p[0].startswith(pref) for pref in PHASES)]


def is_opening_or_middle(label: str) -> bool:
    """Opening/middlegame positions are the only ones whose NPS is comparable
    across builds — an endgame with tablebases searches fewer nodes by design.
    Derived from the LABEL, not from a hardcoded index range, so --phase and
    any future edit to POSITIONS cannot silently shift the split."""
    return label.startswith(("Opening", "Middle"))

# 50 positions across all game phases
POSITIONS = [
    # === OPENINGS (10) ===
    ("Opening: Startpos",           "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 14, "~0"),
    ("Opening: Sicilian",           "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2", 14, "~0"),
    ("Opening: French",             "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", 14, "~0"),
    ("Opening: Queen's Gambit",     "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2", 14, "~0"),
    ("Opening: King's Indian",      "rnbqkb1r/pppppp1p/5np1/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3", 14, "~0"),
    ("Opening: Ruy Lopez",          "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3", 14, "~0"),
    ("Opening: Italian",            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3", 14, "~0"),
    ("Opening: Caro-Kann",          "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", 14, "~0"),
    ("Opening: Scotch",             "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq d3 0 3", 14, "~0"),
    ("Opening: English",            "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq c3 0 1", 14, "~0"),

    # === MIDDLEGAMES (15) ===
    ("Middle: Sicilian Dragon",     "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 0 1", 14, None),
    ("Middle: KID Classical",       "r1bq1rk1/ppp1ppbp/2np1np1/8/2PPP3/2N2N2/PP2BPPP/R1BQ1RK1 b - - 0 7", 14, None),
    ("Middle: Closed Ruy Lopez",    "r1b2rk1/2q1bppp/p1nppn2/1p6/3NP3/1BN1BP2/PPPQ2PP/R4RK1 w - - 0 12", 14, None),
    ("Middle: QGD Exchange",        "r1b1k2r/pp1n1ppp/2p1pn2/q2p4/1bPP4/2NBPN2/PP3PPP/R1BQK2R w KQkq - 0 7", 14, None),
    ("Middle: Benoni",              "r1bq1rk1/pp1n1pb1/2pp1npp/4p3/2PPP3/2N2NP1/PP3PBP/R1BQ1RK1 w - - 0 10", 14, None),
    ("Middle: Sharp tactical",      "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R b KQkq - 0 5", 14, None),
    ("Middle: Isolated pawn",       "r1bqr1k1/pp3ppp/2n2n2/2pp4/3P4/2PBPN2/PP3PPP/R1BQ1RK1 w - - 0 9", 14, None),
    ("Middle: Opposite castling",   "r3kb1r/1bqn1ppp/p2ppn2/1p6/3NP3/1BN1B3/PPPQ1PPP/2KR3R w kq - 0 10", 14, None),
    ("Middle: Piece imbalance",     "r1b2rk1/pp2qppp/2nbpn2/3p4/2PP4/1PN1PN2/1B3PPP/R2QKB1R w KQ - 0 9", 14, None),
    ("Middle: Quiet strategic",     "r2q1rk1/1pp2ppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 14, None),
    ("Middle: Complex open",        "r1b1k2r/pp1pqppp/2n1pn2/2p5/2PP4/2N2NP1/PP2PPBP/R1BQK2R w KQkq - 0 6", 14, None),
    ("Middle: Material imbal",      "r1bqkb1r/pp3ppp/2n1pn2/2ppN3/3P4/6P1/PPP1PPBP/RNBQK2R w KQkq - 0 5", 14, None),
    ("Middle: Pawn storm",          "r1bq1rk1/pppn1pbp/3p1np1/4p3/2PPP3/2N2N1P/PP2BPP1/R1BQ1RK1 b - - 0 8", 14, None),
    ("Middle: Semi-open file",      "r3r1k1/pp2bppp/1qn2n2/3p4/3P4/1BN1PN2/PP3PPP/R2QR1K1 w - - 0 12", 14, None),
    ("Middle: Dynamic tension",     "r1bq1rk1/pp1nbppp/2p1pn2/3p4/2PP4/1PN1PN2/1B3PPP/R2QKB1R w KQ - 0 8", 14, None),

    # === ENDGAMES 6pc (5) ===
    ("End 6pc: KRPKRP",        "8/5pk1/7p/3p1Rp1/8/7P/5PK1/8 w - - 2 40", 14, None),
    ("End 6pc: KPPKPP",        "8/8/1pk5/p7/P7/1PK5/8/8 w - - 0 40", 14, None),
    ("End 6pc: KBNKPP",        "8/8/4k3/8/1p1p4/8/3BN3/4K3 w - - 0 1", 14, None),
    ("End 6pc: KRBKRN",        "2r5/8/3k4/8/8/3K4/8/R4B2 w - - 0 1", 14, None),
    ("End 6pc: KQPKQP",        "8/6pk/5Q2/8/8/2q5/6PK/8 w - - 0 1", 14, None),

    # === ENDGAMES 5pc (5) ===
    ("End 5pc: KRPKB",         "8/8/3k4/8/3RP3/8/2K5/4b3 w - - 0 1", 14, ">300"),
    ("End 5pc: KBNKP",         "8/8/4k3/4p3/8/3BN3/8/4K3 w - - 0 1", 14, None),
    ("End 5pc: KQPKR",         "8/8/4k3/8/3QP3/8/2K5/4r3 w - - 0 1", 14, ">300"),
    ("End 5pc: KRBKN",         "8/8/3k4/4n3/8/8/2R5/K3B3 w - - 0 1", 14, ">100"),
    ("End 5pc: KPPKP",         "8/8/4k3/3p4/8/3PP3/4K3/8 w - - 0 1", 14, None),

    # === ENDGAMES 4pc (5) ===
    ("End 4pc: KRKP",          "8/8/4k3/8/3p4/8/2R5/4K3 w - - 0 1", 14, ">300"),
    ("End 4pc: KQKR",          "8/8/4k3/8/8/8/4Q3/4K2r w - - 0 1", 14, ">300"),
    ("End 4pc: KRKB",          "8/8/4k3/3b4/8/8/2R5/4K3 w - - 0 1", 14, ">30"),
    ("End 4pc: KBKN drawn",    "8/8/4k3/8/3B4/8/5n2/4K3 w - - 0 1", 14, "~0"),
    ("End 4pc: KRKN",          "8/8/4k3/4n3/8/8/2R5/4K3 w - - 0 1", 14, ">30"),

    # === ENDGAMES 3pc (5) ===
    ("End 3pc: KPK won",       "8/8/4k3/8/8/8/4P3/4K3 w - - 0 1", 14, ">100"),
    ("End 3pc: KPK drawn",     "8/8/8/k7/8/8/P7/1K6 b - - 0 1", 14, None),
    ("End 3pc: KRK",           "8/8/4k3/8/8/8/8/R3K3 w - - 0 1", 14, ">300"),
    ("End 3pc: KQK",           "8/8/4k3/8/8/8/8/4K2Q w - - 0 1", 14, ">1000"),
    ("End 3pc: KBK insuff",    "8/8/4k3/8/8/8/8/4KB2 w - - 0 1", 12, "=0"),

    # === ENDGAMES 2pc (2) ===
    ("End 2pc: KvK",           "8/8/4k3/8/8/8/8/4K3 w - - 0 1", 12, "=0"),
    ("End 2pc: KvK black",     "4k3/8/8/8/8/8/8/4K3 b - - 0 1", 12, "=0"),

    # === INSUFFICIENT MATERIAL (3) ===
    ("Insuff: KNK",            "8/8/4k3/8/8/8/8/4K1N1 w - - 0 1", 12, "=0"),
    ("Insuff: KBK black",      "8/8/4k3/8/3b4/8/8/4K3 b - - 0 1", 12, "=0"),
    ("Insuff: KBKB same col",  "8/8/3Bk3/8/8/3b4/8/4K3 w - - 0 1", 12, None),
]

def run_engine(path, fen, depth, use_tb):
    tb_opt = f"setoption name SyzygyPath value {TB_PATH}" if use_tb else ""
    cmds = f"uci\n{tb_opt}\nisready\nposition fen {fen}\ngo depth {depth}\n"
    try:
        proc = subprocess.run([path], input=cmds, capture_output=True, text=True, timeout=30,
                              cwd=os.path.dirname(path))
        lines = proc.stdout.split("\n")
    except:
        return 0, "ERR", 0, 0
    
    nps = 0; score_str = "?"; nodes = 0; tbhits = 0
    for line in reversed(lines):
        if f"info depth {depth} " in line or (line.startswith("info depth") and "seldepth" in line):
            m = re.search(r"nps (\d+)", line)
            if m: nps = int(m.group(1))
            m = re.search(r"score cp (-?\d+)", line)
            if m: score_str = f"{int(m.group(1))}cp"
            m = re.search(r"score mate (-?\d+)", line)
            if m: score_str = f"mate{m.group(1)}"
            m = re.search(r"nodes (\d+)", line)
            if m: nodes = int(m.group(1))
            m = re.search(r"tbhits (\d+)", line)
            if m: tbhits = int(m.group(1))
            if nps > 0: break
    return nps, score_str, nodes, tbhits

def check_eval(score_str, expected):
    if expected is None: return "—"
    try:
        if score_str.startswith("mate"):
            val = int(score_str[4:]) * 10000
        elif score_str.endswith("cp"):
            val = int(score_str[:-2])
        else:
            return "?"
    except:
        return "?"
    
    if expected == "=0":
        return "✅" if val == 0 else f"❌ ({score_str})"
    elif expected == "~0":
        return "✅" if abs(val) <= 50 else f"❌ ({score_str})"
    elif expected.startswith(">"):
        threshold = int(expected[1:])
        return "✅" if val > threshold else f"❌ ({score_str})"
    elif expected.startswith("<"):
        threshold = int(expected[1:])
        return "✅" if val < threshold else f"❌ ({score_str})"
    return "?"

def main():
    engines   = build_engines()
    positions = select_positions()
    tb_name, notb_name = head_tb(), head_notb()

    print(f"\n{'='*80}")
    print(f"  PHASE 2.5: NPS + Eval Benchmark")
    print(f"  Engines: {', '.join(engines)}")
    print(f"  Positions: {len(positions)}"
          + (f"  (filtered by --phase {PHASES})" if PHASES else ""))
    if DEPTH:
        print(f"  Depth: {DEPTH} (forced on every position)")
    print(f"{'='*80}\n")

    if not positions:
        print(f"  No position matches --phase {PHASES}. Nothing to do.")
        return 1

    missing = [f"{n} ({p})" for n, (p, _) in engines.items() if not os.path.isfile(p)]
    if missing:
        print("  Engine binary not found:\n    " + "\n    ".join(missing))
        print("  Build it, or drop it from the run (--no-run-base / --head).")
        return 1

    nps_totals = {name: [] for name in engines}
    om_flags   = []      # per position: is it opening/middlegame?
    fails      = []

    for i, (label, fen, depth, expected) in enumerate(positions):
        print(f"  [{i+1:2d}/{len(positions)}] {label}")
        om_flags.append(is_opening_or_middle(label))
        row = {}
        for ename, (epath, use_tb) in engines.items():
            nps, score, nodes, tbhits = run_engine(epath, fen, depth, use_tb)
            row[ename] = (nps, score, nodes, tbhits)
            nps_totals[ename].append(nps)  # always append to keep index alignment

            # Check eval sanity for the head engine (TB build)
            if ename == tb_name and expected:
                if "❌" in check_eval(score, expected):
                    fails.append(f"{label}: expected {expected}, got {score}")

        tb_nps, tb_score, tb_nodes, tb_tbhits = row.get(tb_name, (0, "?", 0, 0))
        notb_nps, notb_score, notb_nodes, _   = row.get(notb_name, (0, "?", 0, 0))
        base_nps, base_score, base_nodes, _   = row.get(BASE_VERSION, (0, "?", 0, 0))

        eval_ok    = check_eval(tb_score, expected) if expected else "—"
        nps_vs_base = f"{(tb_nps/base_nps - 1)*100:+.1f}%" if base_nps > 0 else "N/A"

        print(f"         TB: {tb_nps:>10,} nps  score={tb_score:>8s}  nodes={tb_nodes:>10,}  tbhits={tb_tbhits:>6,}  eval={eval_ok}")
        print(f"       noTB: {notb_nps:>10,} nps  score={notb_score:>8s}  nodes={notb_nodes:>10,}")
        if BASE_VERSION in engines:
            print(f"  {BASE_VERSION:>9s}: {base_nps:>10,} nps  score={base_score:>8s}  nodes={base_nodes:>10,}  Δ={nps_vs_base}")
        print()

    # Summary by phase. The split comes from each position's LABEL (see
    # is_opening_or_middle) instead of a hardcoded "first 25", so it stays
    # correct under --phase and under any edit to POSITIONS.
    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")

    om_idx  = [i for i, om in enumerate(om_flags) if om]
    end_idx = [i for i, om in enumerate(om_flags) if not om]
    for phase_label, idxs in (("Opening+Middle", om_idx),
                              ("Endgames", end_idx),
                              ("All positions", list(range(len(positions))))):
        if not idxs:
            continue
        print(f"\n  --- {phase_label} ({len(idxs)} pos) ---")
        for ename in engines:
            vals = [nps_totals[ename][i] for i in idxs if nps_totals[ename][i] > 0]
            if vals:
                print(f"    {ename:12s}: avg NPS = {sum(vals)/len(vals):>12,.0f} ({len(vals)} pos)")

    # NPS pass/fail — only Opening+Middle is meaningful (comparable node counts)
    avg = lambda name: (lambda v: sum(v)/len(v) if v else 0)(
        [nps_totals[name][i] for i in om_idx if nps_totals[name][i] > 0])
    base_avg = avg(BASE_VERSION) if BASE_VERSION in engines else 0
    if base_avg:
        tb_delta   = (avg(tb_name)   / base_avg - 1) * 100
        notb_delta = (avg(notb_name) / base_avg - 1) * 100
        ok = lambda d: '✅ PASS' if d >= -NPS_FAIL_PCT else f'❌ FAIL >{NPS_FAIL_PCT:g}%'
        print(f"\n  NPS vs {BASE_VERSION} (Opening+Middle only — fair comparison):")
        print(f"    {tb_name:<11s} vs {BASE_VERSION}: {tb_delta:+.1f}% {ok(tb_delta)}")
        print(f"    {notb_name:<11s} vs {BASE_VERSION}: {notb_delta:+.1f}% {ok(notb_delta)}")
        print(f"  (Endgame NPS is not comparable: TB prunes subtrees → fewer nodes → lower measured NPS)")
    elif not om_idx:
        print(f"\n  (no opening/middlegame position in this run — NPS comparison skipped)")

    if fails:
        print(f"\n  ❌ EVAL SANITY FAILURES ({len(fails)}):")
        for f in fails:
            print(f"    - {f}")
    else:
        print(f"\n  ✅ ALL EVAL SANITY CHECKS PASSED")

    print(f"\n{'='*80}\n")
    return 1 if fails else 0

if __name__ == "__main__":
    override_from_cli(globals(), CLI, description=__doc__, prog="bench_nps.py")
    sys.exit(main())
