#!/usr/bin/env python3
"""Materialize the v3.20 source upgrades from the v3.14 baseline.

Accepted v3.20 changes are deliberately narrow and independently testable:
- UCI version 3.20;
- optional AVX-VNNI L2 NNUE kernel (AVX2 fallback remains canonical/portable);
- TT generation remains stable between moves and is advanced by ucinewgame only;
- root never takes a TT score cutoff, preserving a real PV/bestmove;
- aborted searches never write their unwound score into TT.

No v4 pruning constants or NNUE weights/features are imported here.
The script is idempotent and fails if the expected source shape changes.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NNUE = ROOT / "engine/c/zchezz_v320/nnue.c"
MAIN = ROOT / "engine/c/zchezz_v320/main.c"
SEARCH = ROOT / "engine/c/zchezz_v320/search.c"

vnni_block = r'''#if defined(__AVXVNNI__)
    /* v3.20: optional AVX-VNNI path ported from v4.02. VPDPBUSD performs
     * uint8 activations x int8 weights directly into int32 accumulators,
     * avoiding the maddubs+madd+add sequence and its intermediate int16
     * saturation. The AVX2 path below remains the portable fallback. */
    {
        for (int o = 0; o < NN_L2_OUT; o += 4) {
            const int8_t *row0 = _nnL2W + (o+0) * NN_L2_IN;
            const int8_t *row1 = _nnL2W + (o+1) * NN_L2_IN;
            const int8_t *row2 = _nnL2W + (o+2) * NN_L2_IN;
            const int8_t *row3 = _nnL2W + (o+3) * NN_L2_IN;
            __m256i sum0 = _mm256_setzero_si256();
            __m256i sum1 = _mm256_setzero_si256();
            __m256i sum2 = _mm256_setzero_si256();
            __m256i sum3 = _mm256_setzero_si256();
            for (int i = 0; i < NN_L2_IN; i += 32) {
                __m256i a = _mm256_load_si256((const __m256i*)(relu1 + i));
                sum0 = _mm256_dpbusd_epi32(sum0, a, _mm256_load_si256((const __m256i*)(row0 + i)));
                sum1 = _mm256_dpbusd_epi32(sum1, a, _mm256_load_si256((const __m256i*)(row1 + i)));
                sum2 = _mm256_dpbusd_epi32(sum2, a, _mm256_load_si256((const __m256i*)(row2 + i)));
                sum3 = _mm256_dpbusd_epi32(sum3, a, _mm256_load_si256((const __m256i*)(row3 + i)));
            }
            acc2[o+0] = _nnL2B[o+0] + _hsum_epi32(sum0);
            acc2[o+1] = _nnL2B[o+1] + _hsum_epi32(sum1);
            acc2[o+2] = _nnL2B[o+2] + _hsum_epi32(sum2);
            acc2[o+3] = _nnL2B[o+3] + _hsum_epi32(sum3);
        }
    }
#elif defined(__AVX2__)
'''

# --- NNUE: optional VNNI path, twice (nnue_eval + nnue_eval_bb) ---
needle = "#ifdef __AVX2__\n    {\n        __m256i ones = _mm256_set1_epi16(1);"
replacement = vnni_block + "    {\n        __m256i ones = _mm256_set1_epi16(1);"
text = NNUE.read_text(encoding="utf-8")
if "v3.20: optional AVX-VNNI path ported from v4.02" not in text and \
   "v3.20: AVX-VNNI path ported from v4.02" not in text:
    count = text.count(needle)
    if count != 2:
        raise SystemExit(f"expected exactly two v3.14 L2 AVX2 kernels, found {count}")
    text = text.replace(needle, replacement, 2)
    NNUE.write_text(text, encoding="utf-8")

# --- UCI version ---
main = MAIN.read_text(encoding="utf-8")
old_version = '#define ENGINE_VERSION "3.14"'
new_version = '#define ENGINE_VERSION "3.20"'
if new_version not in main:
    if old_version not in main:
        raise SystemExit("could not find the v3.14 ENGINE_VERSION macro")
    main = main.replace(old_version, new_version, 1)
    main = main.replace("main.c — Zchezz v3.14 UCI engine", "main.c — Zchezz v3.20 UCI engine", 1)
    MAIN.write_text(main, encoding="utf-8")

# --- Search: stable TT generation + root guard + abort-safe stores ---
s = SEARCH.read_text(encoding="utf-8")

stable_marker = "v3.20: TT generation stays stable for the whole game"
if stable_marker not in s:
    old = "    /* Only main thread increments TT generation — helpers share it */\n    if (p->start_depth <= 1) TT_GEN = (TT_GEN+1) & 0xFFFF;\n"
    new = (
        "    /* v3.20: TT generation stays stable for the whole game.\n"
        "     * cmd_ucinewgame() is the single generation boundary. This lets\n"
        "     * positions reached again on later moves reuse both TT scores and\n"
        "     * moves instead of degrading every old hit to move-ordering only. */\n"
    )
    if old not in s:
        raise SystemExit("could not find v3.14 per-search TT generation bump")
    s = s.replace(old, new, 1)

root_marker = "v3.20: never take a TT score cutoff at ply 0"
if root_marker not in s:
    old = "        if (tte_hit == 1 && tte.depth >= depth && !(ply == 0 && ss->excluded_root_n > 0)) {"
    new = (
        "        /* v3.20: never take a TT score cutoff at ply 0. With TT\n"
        "         * generations now stable across moves, a previous root entry\n"
        "         * can be deep enough to cut immediately and leave no fresh PV. */\n"
        "        if (tte_hit == 1 && tte.depth >= depth && ply > 0 && ss->excluded_root_n == 0) {"
    )
    if old not in s:
        raise SystemExit("could not find v3.14 root TT cutoff condition")
    s = s.replace(old, new, 1)

abort_marker = "v3.20: do not poison a persistent TT with aborted-search bounds"
if abort_marker not in s:
    old = (
        "    /* Store best result in TT for future visits */\n"
        "    if (best_move.from||best_move.to)\n"
        "        tt_store(b->hash, best, depth, flag, &best_move, ply, raw_eval);\n"
    )
    new = (
        "    /* v3.20: do not poison a persistent TT with aborted-search bounds.\n"
        "     * Scores propagated while time/stop is unwinding are not valid\n"
        "     * bounds and can otherwise survive into subsequent moves. */\n"
        "    if ((best_move.from||best_move.to) && !ss->time_up)\n"
        "        tt_store(b->hash, best, depth, flag, &best_move, ply, raw_eval);\n"
    )
    if old not in s:
        raise SystemExit("could not find v3.14 final TT store")
    s = s.replace(old, new, 1)

SEARCH.write_text(s, encoding="utf-8")
print("v3.20 source upgrades materialized")
