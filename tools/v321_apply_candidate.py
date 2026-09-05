#!/usr/bin/env python3
"""Apply one disposable v3.21 search candidate for A/B testing."""
from pathlib import Path
import argparse

p = argparse.ArgumentParser()
p.add_argument("variant", choices=["qsearch_tt", "ga_margins"])
p.add_argument("--root", default="engine/c/zchezz_v321cand")
a = p.parse_args()
root = Path(a.root)


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{label}: expected exactly 1 match, found {n}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


if a.variant == "qsearch_tt":
    path = root / "search.c"
    old = """        if (sc > qs_best) { qs_best = sc; best_move_qs = moves[i]; }
        if (sc >= beta) {
            return beta;
        }
        if (sc > alpha) alpha = sc;
    }

    /* Don't store non-cutoff qsearch results — they pollute the TT
     * with depth-0 entries that displace more valuable deeper entries */

    return alpha;
"""
    new = """        if (sc > qs_best) { qs_best = sc; best_move_qs = moves[i]; }
        if (sc >= beta) {
            if (!ss->time_up)
                tt_store(b->hash, qs_best, 0, TT_LOWER, &best_move_qs, ply, stand);
            return beta;
        }
        if (sc > alpha) alpha = sc;
    }

    /* v3.21 candidate: persist qsearch results only when a searched
     * capture improves over stand-pat. Pure stand-pat nodes stay unstored. */
    if (qs_best > stand && !ss->time_up) {
        int from_move = best_move_qs.from || best_move_qs.to;
        tt_store(b->hash, qs_best, 0,
                 (from_move && qs_best > qs_orig_alpha) ? TT_EXACT : TT_UPPER,
                 from_move ? &best_move_qs : NULL,
                 ply, stand);
    }

    return alpha;
"""
    replace_once(path, old, new, a.variant)

elif a.variant == "ga_margins":
    path = root / "search.c"
    text = path.read_text(encoding="utf-8")
    reps = [
        ("int rfp_margin = depth*90 - (improving ? 50 : 0);",
         "int rfp_margin = depth*105 - (improving ? 24 : 0);"),
        ("if (static_eval - beta > 200) R += 1;",
         "if (static_eval - beta > 134) R += 1;"),
        ("int pc_beta  = beta + 200;",
         "int pc_beta  = beta + 215;"),
        ("int fut_adj = improving ? 0 : 50;",
         "int fut_adj = improving ? 0 : 76;"),
        ("static const int fut_base[9] = {0,150,300,450,600,750,900,1050,1200};",
         "static const int fut_base[9] = {0,91,182,273,364,455,546,637,728};"),
    ]
    for old, new in reps:
        if text.count(old) != 1:
            raise SystemExit(f"ga_margins missing/duplicate: {old}")
        text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")

print(f"applied {a.variant} to {root}")
