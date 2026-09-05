#!/usr/bin/env python3
"""Convert one disposable v3.21 source copy to a compact 24-byte TT AoS.

Only the physical TT layout changes. Replacement/generation/search policy stays
identical to the v3.21 baseline so Elo/NPS effects can be measured in isolation.
"""
from pathlib import Path
import argparse

p = argparse.ArgumentParser()
p.add_argument("--root", default="engine/c/zchezz_v321cand")
a = p.parse_args()
root = Path(a.root)
sh = root / "search.h"
sc = root / "search.c"
mc = root / "main.c"


def once(path: Path, old: str, new: str, label: str) -> None:
    t = path.read_text(encoding="utf-8")
    n = t.count(old)
    if n != 1:
        raise SystemExit(f"{label}: expected 1 match, found {n}")
    path.write_text(t.replace(old, new, 1), encoding="utf-8")

old = '''/* ── Transposition table entry (SoA layout for cache efficiency) ── */
extern uint64_t TT_H[TT_SIZE];   /* hash (64-bit) */
extern int32_t  TT_S[TT_SIZE];   /* score  */
extern int32_t  TT_D[TT_SIZE];   /* (depth<<8)|flag */
extern uint16_t TT_G[TT_SIZE];   /* generation */
extern int32_t  TT_M[TT_SIZE];   /* packed move */
extern int32_t  TT_E[TT_SIZE];   /* cached static eval */
extern uint16_t TT_GEN;
'''
new = '''/* ── v3.21 candidate: compact AoS transposition entry ───────────
 * 24 bytes/entry. Two entries occupy 48 contiguous bytes, normally one
 * cache-line working set after the first load. Field widths preserve the
 * v3.20/v3.21 semantics exactly. */
typedef struct __attribute__((packed, aligned(8))) {
    uint64_t hash;
    int32_t  score;
    int32_t  move;
    int32_t  eval;
    uint16_t gen;
    uint16_t dflag;
} TTEntry;
_Static_assert(sizeof(TTEntry) == 24, "TTEntry must remain 24 bytes");
extern TTEntry TT[TT_SIZE];
extern uint16_t TT_GEN;
'''
once(sh, old, new, "search.h extern layout")

start = sc.read_text(encoding="utf-8")
old = '''/* ── TT (global SoA arrays) ──────────────────────────────────────
 * Transposition table with Structure-of-Arrays layout for cache locality.
 * Each logical entry contains: hash, score, depth+flags, generation,
 * packed move, and static eval.  Two entries per slot (2-bucket scheme).
 *
 * TT_H[i]  — 64-bit Zobrist hash (full, not truncated)
 * TT_S[i]  — stored score (mate scores adjusted for ply distance)
 * TT_D[i]  — packed: bits [15:8]=depth, bits [1:0]=flag (EXACT/LOWER/UPPER)
 * TT_G[i]  — generation counter (16-bit, wraps at 65536)
 * TT_M[i]  — packed move (from|to<<6|prom<<12|epc<<15|castle<<16)
 * TT_E[i]  — static eval at the time of storage (for pruning decisions)
 * TT_GEN   — current generation; incremented once per ID iteration
 *            by the main thread only (helpers read but don't write).
 *
 * Total memory: TT_SIZE * (8+4+4+2+4+4) = TT_SIZE * 26 bytes
 *   Native (4M entries): ~104 MB
 *   WASM   (512K entries): ~13 MB
 */
uint64_t TT_H[TT_SIZE];
int32_t  TT_S[TT_SIZE];
int32_t  TT_D[TT_SIZE];
uint16_t TT_G[TT_SIZE];
int32_t  TT_M[TT_SIZE];
int32_t  TT_E[TT_SIZE];
uint16_t TT_GEN = 0;
'''
new = '''/* ── v3.21 candidate: compact AoS TT ────────────────────────────
 * Same two-bucket replacement and generation policy as the baseline;
 * only the physical layout changes. Two entries = 48 contiguous bytes. */
TTEntry TT[TT_SIZE];
uint16_t TT_GEN = 0;
'''
once(sc, old, new, "search.c TT definitions")

old = '''    /* Bucket 0: depth-preferred (replace only if deeper or stale generation) */
    int exist_depth0 = (TT_D[base] >> 8) & 0xFF;
    if (!TT_H[base] || TT_G[base] != TT_GEN || depth >= exist_depth0) {
        /* Cascade displaced entry to bucket 1 (preserve it for move ordering) */
        if (TT_H[base] && TT_G[base] == TT_GEN && depth >= exist_depth0) {
            TT_H[base+1] = TT_H[base]; TT_S[base+1] = TT_S[base];
            TT_D[base+1] = TT_D[base]; TT_G[base+1] = TT_G[base];
            TT_M[base+1] = TT_M[base]; TT_E[base+1] = TT_E[base];
        }
        TT_H[base] = hash; TT_S[base] = stored_score;
        TT_D[base] = packed_df; TT_G[base] = TT_GEN;
        TT_M[base] = packed_mv; TT_E[base] = se;
        return;
    }

    /* Bucket 1: always-replace (catches shallow/recent entries) */
    TT_H[base+1] = hash; TT_S[base+1] = stored_score;
    TT_D[base+1] = packed_df; TT_G[base+1] = TT_GEN;
    TT_M[base+1] = packed_mv; TT_E[base+1] = se;
'''
new = '''    int exist_depth0 = (TT[base].dflag >> 8) & 0xFF;
    if (!TT[base].hash || TT[base].gen != TT_GEN || depth >= exist_depth0) {
        if (TT[base].hash && TT[base].gen == TT_GEN && depth >= exist_depth0)
            TT[base+1] = TT[base];
        TT[base].hash  = hash;
        TT[base].score = stored_score;
        TT[base].dflag = (uint16_t)packed_df;
        TT[base].gen   = TT_GEN;
        TT[base].move  = packed_mv;
        TT[base].eval  = se;
        return;
    }
    TT[base+1].hash  = hash;
    TT[base+1].score = stored_score;
    TT[base+1].dflag = (uint16_t)packed_df;
    TT[base+1].gen   = TT_GEN;
    TT[base+1].move  = packed_mv;
    TT[base+1].eval  = se;
'''
once(sc, old, new, "tt_store")

old = '''        if (TT_H[idx] != hash) continue;
        if (TT_G[idx] != TT_GEN) {
            /* Stale generation: reuse the stored move for ordering, but not the score */
            out->score  = TT_EVAL_NONE;
            out->depth  = 0;
            out->flag   = TT_UPPER;
            out->static_eval = TT_EVAL_NONE;
            unpack_move(TT_M[idx], &out->move);
            return 2;   /* 2 = stale hit (move only) */
        }
        int d = TT_D[idx];
        out->score  = tt_score_read(TT_S[idx], ply);
        out->depth  = (d >> 8) & 0xFF;
        out->flag   = d & 3;
        out->static_eval = TT_E[idx];
        unpack_move(TT_M[idx], &out->move);
'''
new = '''        const TTEntry *e = &TT[idx];
        if (e->hash != hash) continue;
        if (e->gen != TT_GEN) {
            out->score  = TT_EVAL_NONE;
            out->depth  = 0;
            out->flag   = TT_UPPER;
            out->static_eval = TT_EVAL_NONE;
            unpack_move(e->move, &out->move);
            return 2;
        }
        int d = e->dflag;
        out->score  = tt_score_read(e->score, ply);
        out->depth  = (d >> 8) & 0xFF;
        out->flag   = d & 3;
        out->static_eval = e->eval;
        unpack_move(e->move, &out->move);
'''
once(sc, old, new, "tt_probe")

t = sc.read_text(encoding="utf-8")
old_pref = "__builtin_prefetch(&TT_H[((b->hash ^ ZR_side) & TT_MASK) * TT_BUCKETS], 0, 1);"
count = t.count(old_pref)
if count != 4:
    raise SystemExit(f"prefetch: expected 4 matches, found {count}")
t = t.replace(old_pref, "__builtin_prefetch(&TT[((b->hash ^ ZR_side) & TT_MASK) * TT_BUCKETS], 0, 1);")
sc.write_text(t, encoding="utf-8")

old = '''    /* Clear TT hash entries and set all static evals to NONE */
    memset(TT_H, 0, sizeof(TT_H));
    for (int i = 0; i < TT_SIZE; i++) TT_E[i] = TT_EVAL_NONE;
'''
new = '''    memset(TT, 0, sizeof(TT));
    for (int i = 0; i < TT_SIZE; i++) TT[i].eval = TT_EVAL_NONE;
'''
once(sc, old, new, "search_init TT clear")

old = '''    for (int i = 0; i < sample; i++) {
        if (TT_H[i] != 0 && TT_G[i] == TT_GEN) used++;
    }
'''
new = '''    for (int i = 0; i < sample; i++) {
        if (TT[i].hash != 0 && TT[i].gen == TT_GEN) used++;
    }
'''
once(mc, old, new, "main hashfull")

print("applied compact TT AoS candidate")
