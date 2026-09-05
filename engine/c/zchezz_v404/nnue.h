/* v4.04 compatibility shim
 *
 * The candidate deliberately reuses the proven v3.14 NNU3 evaluator while
 * keeping the v4.02 search/UCI layer.  v4.02's helper-thread setup performs
 * one architecture-specific assignment:
 *
 *     my_nnue->net = g_nnue_net;
 *
 * NNU3 has no per-accumulator NnueNet pointer: its loaded weights are global,
 * immutable, and shared safely by all threads.  The assignment is therefore
 * semantically a no-op for this backend.  Map that one field write onto a
 * cache-key slot which has just been zeroed by memset; assigning zero changes
 * no NNU3 state.  Keeping the shim here lets the experiment reuse the exact
 * v3.14 evaluator implementation and weights without forking 45 KB of UCI
 * code merely to delete one line.
 *
 * If this hybrid is promoted, replace this experiment shim with an explicit
 * backend-neutral helper in main.c/nnue.h rather than retaining the macro.
 */
#pragma once
#include "../zchezz_v314/nnue.h"

#define net cache_key[0]
#define g_nnue_net 0
