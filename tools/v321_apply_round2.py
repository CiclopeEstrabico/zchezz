#!/usr/bin/env python3
"""Apply one disposable v3.21 round-2 candidate for A/B Elo testing."""
from pathlib import Path
import argparse

VARIANTS = [
    "qs_failhigh",
    "qs_improve",
    "rfp_ga",
    "nmp_ga",
    "probcut_ga",
    "futility_base_ga",
    "futility_adj_ga",
    "lmr_aggressive",
    "lmr_conservative",
]

p = argparse.ArgumentParser()
p.add_argument("variant", choices=VARIANTS)
p.add_argument("--root", default="engine/c/zchezz_v321cand")
a = p.parse_args()
root = Path(a.root)
path = root / "search.c"


def replace_once(old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{label}: expected exactly 1 match, found {n}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


if a.variant == "qs_failhigh":
    old = """        if (sc > qs_best) { qs_best = sc; best_move_qs = moves[i]; }
        if (sc >= beta) {
            return beta;
        }
"""
    new = """        if (sc > qs_best) { qs_best = sc; best_move_qs = moves[i]; }
        if (sc >= beta) {
            /* v3.21 candidate: cache only qsearch fail-highs. */
            if (!ss->time_up)
                tt_store(b->hash, qs_best, 0, TT_LOWER, &best_move_qs, ply, stand);
            return beta;
        }
"""
    replace_once(old, new, a.variant)

elif a.variant == "qs_improve":
    old = """    /* Don't store non-cutoff qsearch results — they pollute the TT
     * with depth-0 entries that displace more valuable deeper entries */

    return alpha;
"""
    new = """    /* v3.21 candidate: cache only searched-capture improvements.
     * Fail-highs remain unstored so this isolates the non-cutoff effect. */
    if (qs_best > stand && !ss->time_up) {
        int from_move = best_move_qs.from || best_move_qs.to;
        tt_store(b->hash, qs_best, 0,
                 (from_move && qs_best > qs_orig_alpha) ? TT_EXACT : TT_UPPER,
                 from_move ? &best_move_qs : NULL,
                 ply, stand);
    }

    return alpha;
"""
    replace_once(old, new, a.variant)

elif a.variant == "rfp_ga":
    replace_once(
        "int rfp_margin = depth*90 - (improving ? 50 : 0);",
        "int rfp_margin = depth*105 - (improving ? 24 : 0);",
        a.variant,
    )

elif a.variant == "nmp_ga":
    replace_once(
        "if (static_eval - beta > 200) R += 1;",
        "if (static_eval - beta > 134) R += 1;",
        a.variant,
    )

elif a.variant == "probcut_ga":
    replace_once(
        "int pc_beta  = beta + 200;",
        "int pc_beta  = beta + 215;",
        a.variant,
    )

elif a.variant == "futility_base_ga":
    replace_once(
        "static const int fut_base[9] = {0,150,300,450,600,750,900,1050,1200};",
        "static const int fut_base[9] = {0,91,182,273,364,455,546,637,728};",
        a.variant,
    )

elif a.variant == "futility_adj_ga":
    replace_once(
        "int fut_adj = improving ? 0 : 50;",
        "int fut_adj = improving ? 0 : 76;",
        a.variant,
    )

elif a.variant == "lmr_aggressive":
    replace_once(
        "double v = log((double)d) * log((double)m) / 1.5;",
        "double v = log((double)d) * log((double)m) / 1.35;",
        a.variant,
    )

elif a.variant == "lmr_conservative":
    replace_once(
        "double v = log((double)d) * log((double)m) / 1.5;",
        "double v = log((double)d) * log((double)m) / 1.65;",
        a.variant,
    )

print(f"applied {a.variant} to {root}")
