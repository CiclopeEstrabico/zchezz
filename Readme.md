# Zchezz ♟️

**A chess engine, entirely vibe coded with Claude.**

Written in C with a custom-trained NNUE evaluation, playable directly in the browser via WebAssembly. Compiles to a single self-contained HTML file that works fully offline — no server, no install, no dependencies. Just double-click and play.

▶ **[Play against Zchezz in your browser](https://gitzambrano.github.io/zchezz/)**

---

### Highlights

| | |
|---|---|
| **Version** | v4.00 — `engine/c/zchezz_v400/` |
| **Search** | Alpha-beta with staged move generation, aspiration windows, PVS, Lazy SMP |
| **Evaluation** | Custom NNUE, HalfKP-4Bucket — int16/int8 quantized, AVX2 SIMD + WASM SIMD |
| **Endgames** | Syzygy tablebase support (3-4-5 piece WDL + DTZ probing) |
| **Opening book** | Polyglot .bin format, with built-in ECO opening name recognition |
| **Analysis** | Multi-PV (up to 5 lines), eval bar, blunder detection, eval graph |
| **Platforms** | Windows, Linux, Android (Termux), WebAssembly (any modern browser) |
| **Offline** | Single HTML bundle — works from `file://`, no server needed |
| **UCI** | Full UCI protocol compliance (15 commands, 12 configurable options) |

### Features

- 🧠 **NNUE evaluation** — HalfKP-4Bucket, 2560 features per perspective, dual-perspective accumulators, incremental updates, Quantization-Aware Training
- ⚡ **Lazy SMP** — shared transposition table, staggered helper depths, lock-free design
- 🔄 **Staged move generation** — TT move → captures → quiets → losing captures, avoiding full move list allocation
- 🎯 **Full search suite** — null move pruning, LMR, futility, razoring, ProbCut, singular extensions, IIR, SEE pruning
- 🌐 **Browser play** — complete chess UI with game clock, move animation, sound effects, opening name display, 5 piece styles, 6 board themes
- 📊 **Analysis mode** — engine analysis with depth control, Multi-PV, eval bar, eval graph, blunder check, FEN/PGN import/export
- 📱 **Mobile responsive** — full touch support with adaptive layout

---

## Author

**Gustavo José Zambrano**

---

## Training data

The network was trained on filtered data for quiet positions using PyTorch.

**First generation**

- Lichess dataset
- Ethereal dataset
- Quiet dataset
- Self-play from old versions with traditional evaluation function

**New generation**

- 4 generations of iterative self-play
- Generated endgame positions

Datasets store two columns: `cp`, the evaluation in centipawns, and `result`,
the real game outcome (0.0 / 0.5 / 1.0). Both are White-relative. `wdl` is
`sigmoid(cp / 320)` — a transform of `cp`, never an outcome, and never stored,
because a stored copy can rot apart from the `cp` it came from. The training
target is the blend `lam * result + (1 - lam) * wdl`, computed at training time
with a per-dataset `lam`. See `data/Data.md` for the full corpus map.

### Training the network

Everything is set in the `CONFIGURATION` and `DATASETS` blocks at the top of
`train/train_nnue.py`. A bare run does exactly what those blocks say; the CLI
flags only override them.

| Setting | Meaning |
|---|---|
| `LR` | learning rate. ~1e-3 from random weights; ~1e-5 to refine a trained net |
| `EPOCHS` | also sets the schedule: `CosineAnnealingLR(T_max=EPOCHS)`, so a small value anneals the LR to `eta_min` quickly |
| `DATASETS` | one entry per dataset: `pct` (fraction used per epoch, 0.0 disables), `mode` (`lines` samples rows, `shards` samples whole files), `lam` (how much game result is blended into the target) |

**Reading the loss.** The target is continuous (0..1), not 0/1, so the BCE floor
is *not* 0.693 — it is the mean entropy of the target distribution, measured at
**0.620**. Compare `val_loss` against that number, and read `val_mae` as the
direct error in wdl units. Judging a run against 0.693 makes a nearly converged
network look like a failure.

```bash
python train/train_nnue.py                 # trains what the config block says
python train/train_nnue.py --show-config   # prints the resolved settings, runs nothing
python train/export_nnu4.py                # checkpoint -> NNU4 weight file
```

---

## NNUE

### Architecture — HalfKP-4Bucket

Implemented in `engine/c/zchezz_v400/nnue.c`.

```
Features : HalfKP-4Bucket, 2560 per perspective (no hand-built extra features —
           they emerge from local features for free)
L1       : 2560 → 512   int16 weights, 100% incrementally updated accumulator
Activate : SCReLU   c = clamp(x, 0, QA);  out = (c*c) >> 8
Concat   : [L1(stm) 512 | L1(opp) 512] = 1024 uint8 — order is ALWAYS [stm, opp];
           swapping it makes the engine evaluate winning positions as lost
L2       : 1024 → 32    int8 weights (maddubs kernel), ClippedReLU [0, QB]
L3       : 32 → 1       int8 → float32, output centipawns relative to STM, clamp ±2000
Weights  : NNU4 format, 2,656,464 bytes
```

`QA = 255`, `QB = 64` are the L1/L2 scale constants. `sizeof(NnueAccum)` is ~257 KB —
per-thread, not per-move, and cheap relative to search tree size. Inference is int16/int8 quantized with AVX2 SIMD on native
and WASM SIMD in the browser. Both perspectives share the same L1 weight matrix.

The accumulator is maintained as a stack of `NN_ACC_STACK` (128) frames. Each frame stores
not just `acc_w`/`acc_b` but also each perspective's **king bucket and a "dirty" flag** —
see § Lazy bucket refresh below for why.

#### Coordinate invariant

Critical — read this before touching feature indexing in `nnue.c`:

```
Zchezz mailbox square       : zsq, 0 = a8 ... 63 = h1
White-POV square             : sq_w = zsq ^ 56   (== python-chess square, a1 = 0)
Black-POV square              : sq_b = zsq        (== sq_w mirrored vertically)

king_bucket(s), s already in POV coordinates:
    bit 0 = (s % 8) >= 4      king-side half
    bit 1 = (s / 8) >= 4      far half (ranks 5-8 in that POV)

    bucket_w = king_bucket(wk ^ 56)
    bucket_b = king_bucket(bk)

Relative piece index (the king is NEVER a feature):
    friendly P,N,B,R,Q -> 0..4
    enemy    P,N,B,R,Q -> 5..9

feature = bucket * 640 + rel * 64 + pov_sq       (0 .. 2559)
```

Both perspectives use the **same** L1 weight matrix. Concat order is always `[stm, opp]`.

#### Lazy bucket refresh

A king move that changes its own perspective's bucket invalidates **every** feature of that
perspective (the bucket is baked into every feature index). `nnue_push_na` cannot rebuild on
the spot — `board_make` calls it with the board still in its *pre-move* state. So a
king-bucket-crossing push only raises a dirty flag; the eval call rebuilds that perspective
from the current board on demand ("lazy refresh", Stockfish-NNUE-5 style).

The dirty flag and the king bucket live on a **per-frame stack**, not flat fields:

- a `pop` after a king move would restore the wrong bucket if the bucket were a flat field;
- a pending dirty flag must survive descending another ply — the child frame is built on top
  of a stale accumulator and is equally invalid, so it inherits the flag.

If a node never calls eval (e.g. a tablebase cutoff), the flag is simply discarded on pop.


#### NNU4 weight file format

```
"NNU4"                    4 B    magic
epoch                     uint32
dims[5]                   uint32 × 5  = 2560, 512, 1024, 32, 32
scales[4]                 float32 × 4 = QA=255, QB=64, SHIFT=8, OUT_SCALE=0.078125
L1W  [2560][512]  int16   feature-major (one feature = one contiguous 1 KB row)
L1B  [512]        int32   scale QA
L2W  [32][1024]   int8    output-major, NOT transposed
L2B  [32]         int32   scale QA_EFF × QB
L3W  [32]         int8
L3B               float32 unquantized
```

Total file size: 2,656,464 bytes. The loader validates `dims[]` against the compiled
architecture and refuses mismatched files with an explicit message. `OUT_SCALE = 320 / QB²`; the `320`
is the same cp↔WDL temperature used by `to_wdl()` in training — changing one without the
other silently rescales the whole eval.

### Input encoding

**Half-Mirror (HM) features.** The 768 HM features are a dual-perspective encoding of the piece map. For each piece on the board, two feature indices are activated simultaneously — one in the White accumulator and one in the Black accumulator — using mirrored square coordinates so that each side always sees its own pieces as "friendly" without needing to rotate the network. The index formula is `color_offset × 64 + piece_type × 64 + square`, where the square is flipped vertically for the opponent's perspective.

**Dual accumulator.** Two independent 256-element `int16_t` accumulators are maintained in parallel at all times: `_acc_buf_w` (White's perspective) and `_acc_buf_b` (Black's perspective). Both are kept live and updated in lockstep on every make/unmake. At eval time the side-to-move accumulator is selected, the extra-feature delta is added, and the result is passed through ClippedReLU — there is no perspective swap or extra copy.

**Accumulator stack.** The accumulators are managed as a stack of depth 512 (`NN_ACC_DEPTH`), mirroring the board's undo stack. `nnue_push` copies the top-of-stack accumulators and applies an incremental update for the move being made; `nnue_pop` simply decrements the stack pointer. This means unmake is O(1) — no recomputation needed.

**Endgame features.** The 31 appended features give the network explicit positional context that bitboard features cannot easily encode. Piece counts are normalized by their theoretical maximum (8 pawns, 2 knights/bishops/rooks, 1 queen). Total material sums all piece values and normalizes by 78 (the maximum possible). Passed pawn features are binary per-file flags computed with a forward-ray check: a pawn is passed if no opposing pawn blocks or guards any square on its promotion path on the same or adjacent files. King–king Chebyshev distance is the max of the rank and file differences, normalized to [0, 1]. One constant-1 bias feature is also included.

---

### int16 / int8 Quantization

All weights and activations are converted from float32 to fixed-point integers using two scale factors, QA and QB, which must match exactly between the training script, the converter, and the C engine.

```
QA     = 255     (L1 weight scale — maps float [−128.6, +128.6] → int16 [−32767, +32767])
QB     = 64      (L2/L3 weight scale — maps float [−1.984, +1.984] → int8 [−127, +127])
SHIFT  = 8       (right-shift applied after L1 accumulation, ≈ log2(QA))
OUT_SCALE = 320 / (QB × QB) = 320 / 4096 ≈ 0.078
```

**L1 (int16 weights).** Each weight in L1 is multiplied by QA and rounded to the nearest integer, then stored as `int16_t`. The L1 bias is similarly scaled by QA and stored as `int32_t`. Because each input feature is binary (0 or 1) and there are at most 32 active pieces, the maximum accumulator value before ClippedReLU is `32 × 32767 = 1,048,544`, which fits safely in `int32_t` during accumulation. After ClippedReLU, the output is clamped to `[0, QA] = [0, 255]`, which fits in a `uint8_t`.

**L2 (int8 weights).** Each L2 weight is multiplied by QB and rounded to `int8_t`. The L2 bias is pre-scaled by `QA × QB = 16320` and stored as `int32_t`. The L2 dot product accumulates `uint8_t` activations × `int8_t` weights into `int32_t`, then right-shifts by SHIFT (8) to bring the scale back to QB before ClippedReLU. The output is clamped to `[0, QB] = [0, 64]`.

**L3 (float32).** L3 weights are quantized to `int8_t` (scale QB) in the NNU4 file but the final dot product `L3W · relu2` uses the integer weights scaled back to float for the final scalar. The bias is kept as `float32` — this 64→1 dot product is not a bottleneck and avoiding one more integer scale conversion keeps the output path simple and exact.

**Scale chain summary:**

```
Input features:  binary {0, 1}
After L1 matmul: int32 scale QA  →  ClippedReLU  →  uint8 [0, 255]
After L2 matmul: int32 scale QA×QB  →  >>SHIFT  →  ClippedReLU  →  uint8 [0, QB]
After L3 matmul: float32  ×  OUT_SCALE  →  ×320 cp
```

This gives a total throughput gain of ~2× over float32 inference on AVX2 hardware, since 16 `int16_t` elements fit in a 256-bit register versus 8 `float32` elements.

---

### Forward pass (integer)

```
1.  acc1  = L1B + L1W_T × features                 (int32, scale QA)
2.  relu1 = clamp(acc1, 0, 255)                     (uint8, scale QA)
3.  acc2  = L2B + L2W_T × relu1                     (int32, scale QA×QB)
4.  acc2  = acc2 >> SHIFT                            (int32, scale QB)
5.  relu2 = clamp(acc2, 0, QB)                       (uint8, scale QB)
6.  raw   = L3B + (L3W · relu2) × OUT_SCALE         (float32, WDL logit)
7.  cp    = clamp(raw × 320, −2000, +2000)           (centipawns, STM-relative)
```

The sigmoid is baked into the training target (WDL labels in [0, 1]). The network output is a raw logit, scaled directly to centipawns at inference time — no transcendental function in the eval hot path.

---

### Quantization-Aware Training (QAT)

Standard post-training quantization clips and rounds weights after training has converged, which degrades accuracy because the float32 network was free to use values outside the integer representable range. QAT instead simulates quantization *during* training so the network learns to work within the integer constraints before weights are ever committed to integers.

#### ClippedReLU

The standard `ReLU` after L1 and L2 is replaced with `ClippedReLU(clip=1.0)`:

```
ClippedReLU(x) = clamp(x, 0.0, 1.0)
```

During training the output stays in `[0, 1.0]` (float). In the engine, `1.0` maps exactly to `QA = 255`, giving a clean bijection between the training float range and the `uint8` inference range with no rounding artifacts at the boundary.

#### Fake-quantization (Straight-Through Estimator)

In every forward pass the weights are temporarily snapped to their nearest representable integer value and then divided back to float, simulating the rounding that will happen at deployment:

```python
# int16 simulation (L1 weights, scale QA=255)
limit = 32767 / QA                         # ≈ 128.5
x_clamped = clamp(w, −limit, +limit)
x_quantized = round(x_clamped × QA) / QA  # snapped to integer grid
```

For the backward pass, gradients flow through the rounding as if it were an identity (the Straight-Through Estimator). This lets the optimizer push weights toward better integer-grid values without losing the gradient signal. The same STE trick is applied to L1, L2, and L3 weights, their biases, and the post-ReLU activations.

#### Hard weight clamping

After every optimizer step, all weights are hard-clamped to their representable float range:

```python
lim1 = 32767 / QA  # ≈ 128.5  — L1 weight max
lim2 = 127   / QB  # ≈ 1.984  — L2/L3 weight max
```

Without this, the Adam optimizer's momentum can carry weights outside the clamp on the *next* step, wasting gradient budget on values that will always be clipped. Hard clamping after every step keeps the weights on the integer grid and prevents drift.

#### Training schedule (mixtrain3)

Since the network starts from float32 weights trained by `mixtrain2`, a warm-up schedule is used to ease the transition to integer constraints:

| Epoch range | QAT state         | What's quantized                            |
| ----------- | ----------------- | ------------------------------------------- |
| 0 – 14     | float32 (warm-up) | Nothing — lets ClippedReLU stabilize       |
| 15 – 39    | QAT-L1            | L1 weights/bias + L1 activation             |
| 40 +        | QAT-ALL           | L1 (int16) + L2/L3 (int8) + all activations |

---

### Performance optimisations

**AVX2 int16 accumulation (L1).** The L1 update uses `_mm256_add_epi16` to process 16 `int16_t` elements per register — twice the throughput of float32 `_mm256_add_ps`. The maximum safe accumulator value is `32 pieces × 32767 = 1,048,544`, well within `int32_t`, so there is no risk of overflow during the dot product.

**Cached extra-feature delta.** The 31 endgame feature rows are folded into a per-stack-slot precomputed delta (`_ext_buf`). At eval time no board scan or 31-row multiply is needed — the precomputed delta is simply added to the HM accumulator before ClippedReLU.

**Transposed L2 weights.** The L2 weight matrix is stored in `[in × out]` order (`_nnL2W_T[i × 64 + o]`) so the fused outer-product loop keeps `acc2[64]` hot in registers while streaming `relu1[256]` sequentially. GCC `-O3` auto-vectorises this with AVX/SSE on native and Emscripten with `-msimd128` on WASM.

**Sparsity skip.** The L2 forward pass short-circuits any input where `relu1[i] == 0`, exploiting the natural post-ReLU sparsity of the first hidden layer.

**In-memory WASM loader.** Weights are loaded directly from an `ArrayBuffer` into the WASM heap via `nnue_load_from_mem`, avoiding any filesystem dependency in the browser. The L2 transpose is applied during loading, so the runtime weight layout is always optimal regardless of how the NNU4 file was produced.

---

### Weight file format — NNU4

```
Offset  Size    Content
──────────────────────────────────────────────────────────────────────────────────
0        4 B    Magic: "NNU4"
4        4 B    Epoch (uint32, little-endian)
8       20 B    5 × uint32 dims: [L1_IN, L1_OUT, L2_IN, L2_OUT, L3_IN]
                              = [799,    256,     256,    64,     64  ]
28      16 B    4 × float32 scale params: [QA=255, QB=64, SHIFT=8, OUT_SCALE≈0.078]
44       …      L1W_T  int16[799 × 256]   weights × QA,  transposed (row = feature)
                L1B    int32[256]          bias × QA
                L2W_T  int8[256 × 64]     weights × QB,  transposed (row = feature)
                L2B    int32[64]           bias × QA × QB
                L3W    int8[64]            weights × QB
                L3B    float32[1]          bias (kept float for simplicity)
```

Total file size: **~426 KB** (vs ~860 KB for the float32 NNU2 format — 50% reduction).

---

## Per-instance objects — `TTable` and `NnueNet`

The transposition table and the NNUE weight set are structs with `create`/`destroy`
constructors (`search.h`: `TTable`; `nnue.h`: `NnueNet`), not bare global arrays. The UCI
binary still allocates one process-wide default of each (`g_tt`, `g_nnue_net`) in
`search_init()` / `nnue_init()`, so normal single-engine usage is unaffected — this exists
so two independent searches, or two different networks, can run in one process:

- **`engine/c/tools/selfplay.c`** runs N games in parallel (one thread per game) and needs a
  TT per worker. It SHARES one `TTable` between the two colors of a game (same engine, same
  weights, no information leak, half the memory per worker) and clears it between games with
  a physical `tt_clear()` rather than `tt_new_generation()` — repetition-draw scores are
  history-dependent, so a generation bump alone wouldn't stop a stale "draw by repetition"
  entry from game N being misread as real in game N+1.
- **`engine/c/tools/arena.c`** (the A/B strength gate) runs two adversarial engine instances
  at once, possibly with different weight files, and ISOLATES a `TTable` per player since
  sharing would leak information from one side's search into the other's.

Lazy SMP helper threads are unaffected: they still receive the **same** `TTable` pointer as
the main thread — sharing the TT between helpers of one search is intentional (see § Search
→ Symmetric Multiprocessing).

---

## Board representation

The board layer is the foundation everything else is built on. It is designed for both correctness and speed. Static global search state, magic-bitboard move generation, and no heap allocation in the hot path keep it both correct and fast; `MAX_MOVES` (256) bounds the per-position move list.

### Dual representation

Every position is maintained simultaneously as a flat mailbox (`uint8_t b[64]`) for fast piece lookup by square, and as a set of twelve bitboards (`uint64_t bb[12]`, one per piece type) for efficient bulk operations. Both are kept perfectly in sync through every make/unmake.

### Magic bitboards

Slider attack generation (rook, bishop, queen) uses **magic bitboards** with fully precomputed attack tables. Given the current occupancy, a slider's full attack set is computed in a single masked multiply and table lookup — O(1) regardless of board density. Knight and king attacks use precomputed 64-entry leaper tables. This makes the SEE loop and qsearch capture generation driven entirely by efficient bitboard hardware.

### Zobrist hashing

Every position carries a running **Zobrist hash** updated incrementally on make/unmake. The hash encodes piece placement, side to move, castling rights, and en-passant file. It is used to probe and store in the transposition table and to detect repetitions throughout the search tree.

### Make / Unmake

The engine uses an explicit undo stack (`UndoFrame g_undo[512]`) rather than copy-make. Every `board_make` pushes a full snapshot of castling rights, en-passant, halfmove clock, king squares, and the pre-move hash. `board_unmake` pops it and restores the board array and bitboards in-place — no allocation, no heap traffic in the search hot path.

### Special moves

Full support for en-passant, all four castling rights with legal-position verification, and queen/rook/bishop/knight promotions. Every special case is handled correctly in both the move generator and the NNUE accumulator update path.

---

## Search

### Iterative deepening

The engine always starts at depth 1 and searches progressively deeper, retaining the best move from each completed iteration. The previous iteration's score seeds the aspiration window for the next. If the search is interrupted by the time or node budget, the last completed iteration's best move is returned. `MAX_PLY` (64) bounds recursion depth; all per-search state is `_Thread_local` to support Lazy SMP without locking.

### Aspiration windows

From depth 3 onward, each iteration opens with a symmetric window of ±20 cp around the previous score. On a fail-low or fail-high the window expands exponentially (doubling each miss, capped at 500 cp) and the search is re-run. After up to 6 re-searches it falls back to a full-width window.

### Transposition table

The TT is a **Structure-of-Arrays** layout with separate flat arrays for hash, score, depth+flag, generation, best move, and static eval. This keeps each field's access pattern cache-friendly.

- **Native:** 4 M entries (~112 MB)
- **WASM:** 512 K entries (~14 MB)
- **Generation aging:** each call to `search_best` increments `TT_GEN`, and only the main thread does so — helper threads never bump it, avoiding a race. Stale entries still yield the stored move for ordering but their score is not used for cutoffs.
- **Mate distance correction:** scores above ±9000 cp are stored and read back with a ply adjustment so mate-in-N scores are correctly compared across different depths.
- **Probe order:** the TT is probed *before* the tablebase (Stockfish's ordering) — a TT hit can short-circuit a TB disk probe entirely.

### Pruning and reductions

| Technique                            | Condition                                                                                                                                             |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Null Move Pruning**          | Non-PV, not in check, has major pieces,`static_eval >= beta`. R = 3 + depth/3, capped at 6. +1 if eval margin > 200.                               |
| **Reverse Futility Pruning**   | Non-PV, non-check, depth 2–9. Margin =`depth × 90`, reduced by 50 cp if improving.                                                               |
| **Futility Pruning**           | Non-PV, quiet, not giving check, depth 1–4. Margin = 150–600 cp per depth level.                                                                    |
| **Razoring**                   | Non-PV, non-check, depth 1. Drops into qsearch with a tight window.                                                                                   |
| **ProbCut**                    | Non-PV, non-check, depth ≥ 5, beta < 18000. Shallow search (depth − 4) with β + 200 to prune captures.                                             |
| **Late Move Reductions (LMR)** | Applied from the 4th legal quiet move at depth ≥ 3. Reduction =`log(depth) × log(moveIdx) / 1.5`, minimum 1. PV nodes reduce by one less.         |
| **Late Move Pruning (LMP)**    | Non-check, depth ≤ 7. Quiet moves beyond a depth-scaled limit are skipped outright.                                                                  |
| **Singular Extensions**        | Depth ≥ 7, TT hit at depth − 4. If a re-search with `score − depth×2` as beta fails low, the TT move is singular and gets a +1 depth extension. |
| **Check Extension**            | Checks at depth 1 extend by 1 ply to avoid the horizon effect.                                                                                        |
| **IIR**                        | At depth ≥ 4 with no TT move and not in check, depth is reduced by 1 before move generation.                                                         |

### Symmetric Multiprocessing (SMP)

The engine supports **Lazy SMP** with up to 8 threads. Each worker thread runs an independent iterative-deepening search on the same position, sharing a single global transposition table. Threads use slightly different search parameters (node limits, reduction tweaks) to diversify the search tree exploration. The main thread collects the best result from all workers.

- **UCI option**: `setoption name Threads value <N>` (1–128, default 1)
- **WASM**: single-threaded only (browser SharedArrayBuffer requires COOP/COEP headers not available on file:// protocol)
- **Thread safety**: TT is lock-free (struct-of-arrays layout); per-board undo stack and repetition history ensure thread isolation
- **TT move validation**: corrupt TT entries from hash collisions are detected and skipped (piece/color/castle bounds checking) before board_make, preventing rare crashes under heavy TT contention
- **Helper thread management**: asynchronous cancellation with timed-join fallback and 8 MB stacks for reliable cleanup
- **Staggered start depths**: the main thread runs full iterative deepening from depth 1 (with aspiration windows and `info` output); helper 0 starts at depth 2, helper 1 at depth 3, and helper *i* ≥ 2 at depth `2i+1` — this spreads the helpers across different parts of the tree instead of duplicating the main thread's early, cheap iterations
- **MultiPV time budget**: each PV line gets its own deadline reset, so line 2 doesn't starve because line 1 used the whole move-time budget

### Quiescence search

Quiescence is entered at depth 0 and searches captures and promotions only (plus all moves when in check). It applies:

- **Stand-pat** cutoff against beta
- **Delta pruning** globally (queen value + 50 cp, or 2× queen + 50 if a passer is on the 7th rank)
- **Per-move delta pruning** — skips captures where `stand_pat + capture_gain + 50 < alpha`
- **SEE pruning** — losing captures (`SEE < 0`) are skipped entirely
- **Pick-best** (selection sort) rather than full sort — avoids scoring moves that will be pruned before being searched

### Static Exchange Evaluation (SEE)

SEE uses the magic bitboard attack tables directly. The attacker board is rebuilt via a bitboard occupancy mask and updated incrementally as pieces are removed, so discovered attackers (X-ray attacks through vacated squares) are revealed automatically. This makes SEE both fast and exact — the full minimax retrograde scoring is computed in a temporary scratch board.

### Staged move generation

The engine uses **staged move generation** in the alpha-beta search. Instead of generating all moves upfront and sorting them, moves are generated in phases:

| Stage | What's generated | Why |
| ----- | ---------------- | --- |
| **TT move** | Hash move only (no generation needed) | Best move from a previous search — often causes a cutoff immediately |
| **Captures** | `board_gen_captures()` — captures + promotions only | MVV-LVA sorted, SEE-pruned; most cutoffs happen here |
| **Quiets** | `board_gen_quiets()` — non-captures only | Scored by killer/counter/history; LMR/LMP applied |
| **Losing captures** | Deferred bad captures (SEE < 0) | Rarely searched — only if all quiets fail to produce a cutoff |

This avoids generating quiet moves at all in positions where a capture or the TT move produces a beta cutoff, saving significant move generation overhead. Measured improvement: **+21 ELO** vs v3.09.

### Move ordering

| Priority         | Score                                                     | Move type |
| ---------------- | --------------------------------------------------------- | --------- |
| 2 000 000        | TT / PV move                                              |           |
| 1 700 000+       | Promotions                                                |           |
| 1 600 000+       | Winning or equal captures (SEE ≥ 0), sorted by MVV-LVA   |           |
| 900 000          | Killer move slot 0                                        |           |
| 800 000          | Killer move slot 1                                        |           |
| 780 000          | Counter move                                              |           |
| 700 000+         | Rook to 7th rank                                          |           |
| 600 000+         | Moves attacking the opponent's king zone (Chebyshev ≤ 2) |           |
| 200 000+         | Quiet moves (base + history + CMH)                        |           |
| 100 000–200 000 | Losing captures (SEE < 0)                                 |           |

### History heuristics

Three complementary tables are updated on every beta cutoff by a quiet move:

- **Main history** (`int32_t mv_history[64×64]`): keyed on from–to pair
- **Continuation history** — two CMH slots (`int16_t cont_hist[2][64][64×64]`):
  - Slot 0: conditioned on where the move at ply−1 landed (1-ply CMH)
  - Slot 1: conditioned on where the move at ply−2 landed (2-ply CMH)
- **Counter move** (`int32_t counter_move[64×64]`): the single best refutation of the previous move

All three tables contribute additively to quiet move scores. Moves that caused a cutoff receive a bonus of `depth²`; moves searched before the cutoff receive the same value as a penalty. At the start of each new search the history tables are aged (divided by 4) to prevent saturation while preserving relative ordering.

### Improving flag

Before any pruning decision at a non-check node, the engine computes an **improving** flag: the current static eval is compared to the static eval two plies ago (stored in `prev_static_eval[ply-2]`). If the current position is improving — meaning the score is higher now than it was two plies ago — pruning margins are tightened (RFP margin reduced by 50 cp, LMP limits more generous). If not improving, the engine is more aggressive with pruning. This single flag meaningfully increases search efficiency in positional middlegames where the engine is losing ground slowly.

### Node types and PV search

The engine implements full **PVS (Principal Variation Search)**. The first legal move at each node is searched with the full `[alpha, beta]` window. All subsequent moves are first searched with a null window `[alpha, alpha+1]` at reduced depth (LMR applied). If the null-window search exceeds alpha — indicating the move might actually be better — a costly full re-search with the complete window is triggered to get the exact score. PV nodes (where `beta - alpha > 1`) always receive one fewer reduction in LMR, ensuring the principal variation is searched accurately.

### Contempt

The engine uses a **contempt factor of 15 cp**. When a position is first repeated (not yet a legal draw), the engine returns `+CONTEMPT` from the repeating side's perspective rather than 0. This penalises the side that chose to repeat, preventing the engine from looping in winning positions while still accepting a draw when genuinely behind. True 3-fold repetition and the 50-move rule always return an exact 0.

- **50-move rule** — tracked via the halfmove clock
- **Threefold repetition** — Zobrist hashes maintained in a global history array. First repetition returns `+CONTEMPT` (15 cp) to discourage repeating in a winning position. True 3-fold returns 0.
- **Stalemate / checkmate** — detected when `legal_count == 0` at the leaf

---

## Browser interface

The HTML file (`zchezz_wasm.html`) is a complete, self-contained chess application. Aside from the WebAssembly engine it needs no server and no installation. The bundled version (`zchezz_bundle.html`) embeds the WASM binary and weights directly and works by double-clicking the file.

### Game panel

Play against Zchezz at full engine strength.

- **Side selection** — choose to play as White or Black at any time
- **Flip board** — rotate the board 180°
- **Search mode** — choose between fixed Depth (d5–d15), Time per move (0.5s–30s), or Game clock
- **Game clock** — timed play with configurable time controls (1+0, 3+0, 3+2, 5+0, 5+3, 10+0, 15+10) with increment support. Live countdown clocks with active/low-time visual indicators
- **New game** — resets board and history instantly
- **Move log** — full game in SAN notation; click any move to jump to that position
- **Piece animation** — smooth slide animation on engine moves
- **Status line** — shows engine thinking state, book moves, game result
- **Opening name** — ECO code and opening name displayed live as moves are played (Ruy Lopez, Sicilian Dragon, QGD, King's Indian, London System, and many more)
- **Sound effects** — distinct tones for quiet moves, captures, checks, and game end (toggleable)

### Opening book

- **Built-in ECO table** — recognises dozens of major openings directly from position FEN
- **Polyglot .bin book** — load any standard Polyglot binary opening book from disk. The engine computes the full Polyglot Zobrist hash client-side and binary-searches the file for book moves
- **Toggle** — short-click the book button to open a file picker; long-press (500 ms) to toggle the book on or off without losing the loaded file. Mobile haptic feedback included

### Analysis panel

A dedicated board and engine for post-game and mid-game analysis.

- **Engine toggle** — start or stop continuous engine analysis at any depth (d5–d15 or ∞)
- **Infinite analysis** — uses a `startDepth` optimization: once a depth has been searched, the TT already holds its result, so each further depth call only searches the *new* depth instead of restarting from depth 1
- **Eval bar** — animated white/black percentage bar with centipawn score
- **Multi-PV analysis** — display up to `MAX_MULTI_PV` (6) engine lines simultaneously (currently exposed as 1–3 in the UI toolbar's +/− buttons; the browser UI supports up to 5)
- **Move navigation** — step forward and backward through the game one move at a time
- **Eval graph** — a plotted evaluation curve over the full game; click any point to jump to that move
- **Blunder check** — one-click full-game annotation. The engine analyses every position and marks blunders, mistakes, and inaccuracies automatically
- **FEN** — load any position by pasting a FEN string, or copy the current position FEN to clipboard
- **PGN** — load a full game from PGN notation, or copy the current game as PGN

### Settings

Accessible via the ⚙ button in-game.

- **Piece style** — Outline, Solid (filled), Letters (monospace), or Merida (SVG)
- **Board color** — Classic (brown), Green, Blue, Coral, Walnut, or Night (dark gold)
- **Highlight last move** — toggleable yellow highlight on from/to squares
- **Sound effects** — enable or disable all audio
- **Debug log** — collapsible UCI traffic log for development

---

## Engine build

The build system is **shared** across engine versions and lives in `engine/build/`
(`Makefile`, `build_native.bat`, `build_wasm.bat`, `build_termux.sh`, `bundle.py`,
`pieces/`) — it is not duplicated per `zchezz_vXXX/` folder. Every target takes
`ENGINE=vXXX` (default `v400`) to select which `engine/c/zchezz_vXXX/` folder to build
from. This machine has `mingw32-make`, not `make`.

### Native — Windows / Linux

```bash
cd engine/build
build_native.bat v400              # Windows one-click compile (MinGW); ENGINE arg optional
mingw32-make ENGINE=v400 native    # Windows, via make
make ENGINE=v400 native            # Linux
# binary lands in engine/c/zchezz_v400/zchezz.exe
```

### Native tools (selfplay / arena)

```bash
cd engine/build
mingw32-make ENGINE=v400 arena     # -> engine/build/arena.exe
mingw32-make ENGINE=v400 selfplay  # -> engine/build/selfplay.exe
```

### Android / Termux

```bash
cd engine/build
./build_termux.sh v400
```

### WebAssembly (requires [Emscripten](https://emscripten.org))

```bash
cd engine/build
build_wasm.bat v400
# Compiles WASM, bundles HTML, and copies the result to the repo-root index.html
```

The bundler (`bundle.py`) reads the compiled WASM binary, the NNUE weight file, and the JS worker, base64-encodes them, and splices everything directly into the HTML as inline constants. It also embeds the CBurnett, Merida, and Staunty piece SVGs from `engine/build/pieces/` so every piece style is available without any network request. The version string is parsed automatically from the engine folder name (e.g. `zchezz_v400` → `4.00`). The resulting file is fully self-contained — no server, no network, no dependencies.

WASM builds compile with `-DNO_TABLEBASES -DNO_BOOK` (no file I/O is available in the
browser sandbox) and `-msimd128` for 128-bit WASM SIMD. The build is single-threaded (no
`SharedArrayBuffer` without COOP/COEP headers, which `file://` can't provide) and all search
state is `static` rather than `_Thread_local`, since there's only ever one thread.

### Native tools — self-play (Python driver)

`tests/run_selfplay_native.py` drives the compiled `selfplay.exe` (see § Native tools above)
as a Python wrapper — the same role `run_arena.py` plays for `arena.exe`. Build the binary
first with `mingw32-make ENGINE=v400 selfplay` or the one-click
`engine/build/build_selfplay.bat`.

### Tests

```bash
python tests/test_perft.py                 # 37 positions, engine from the config block
python tests/test_uci_extended.py          # full UCI suite, 13 groups
python tests/bench_nps.py                  # 50-position NPS + eval sanity
```

Each accepts flags that override its config block — useful while iterating:

```bash
python tests/test_perft.py --only Kiwipete --max-depth 3   # one position, shallow
python tests/test_uci_extended.py --only T3 --only T7      # two test groups
python tests/bench_nps.py --phase Opening --no-run-base    # opening positions, head only
python tests/test_perft.py v400                            # legacy positional form still works
```

See `CLAUDE.md` for the full Phase 1–9 testing workflow.

---

## UCI protocol

```
uci
isready
ucinewgame
setoption name NNUE value <path>
setoption name Threads value <1-8>
setoption name Hash value <MB>
setoption name Contempt value <-100 to 100>
setoption name MoveOverhead value <0-5000>
setoption name MultiPV value <1-4>
setoption name Ponder value <true|false>
setoption name SyzygyPath value <path>
setoption name SyzygyProbeDepth value <1-100>
setoption name SyzygyProbeLimit value <0-7>
setoption name Syzygy50MoveRule value <true|false>
position startpos [moves ...]
position fen <fen> [moves ...]
go depth <n>
go movetime <ms>
go wtime <ms> btime <ms> [winc <ms>] [binc <ms>]
go infinite
go ponder
go nodes <n>
go mate <n>
go searchmoves <move1> [<move2> ...]
stop
quit
d                           (debug: print board + FEN + eval + hash)
eval                        (show NNUE static eval)
debug on|off                (enable/disable debug output)
ponderhit
register
```

### Info output fields

```
info depth <d> seldepth <d> score cp <cp> nodes <n> nps <n>
     time <ms> hashfull <permille> tbhits <n> pv <moves>
```

---

## Browser usage

The engine is distributed as a single self-contained HTML file (`zchezz_bundle.html`) produced by `bundle.py`. It embeds the WASM binary, NNUE weights, and all UI assets as base64 — no web server or external files needed. Just double-click it in any modern browser.

To (re-)build the bundle after recompiling:

```bash
python bundle.py zchezz_wasm.html zchezz_wasm.js zchezz_wasm.wasm nnue_weights.bin
```

The bundled engine exposes `window.zchezzSearch(params, cb)` for programmatic use:

```js
// params
{ fen, moves, depth, timeLimitMs, id }

// result (success)
{ id, uci, score, nodes, pv }

// result (error)
{ id, error }
```

### `SearchResult` struct (WASM interop)

The lower-level `SearchResult` struct is 1920 bytes, mapped in JS via `DataView`:

| Offset | Field | Type |
|---|---|---|
| 0 | `best` | `Move` (20 bytes) |
| 20 | `score` | `int32` |
| 24 | `depth` | `int32` |
| 28 | *(padding)* | 4 bytes |
| 32 | `nodes` | `int64` |
| 40 | `tb_hits` | `int64` |
| 48 | `pv` | 256-byte string |
| 304 | `num_pvs` | `int32` |
| 308 | `scores[]` | 6 × `int32` (24 bytes) |
| 332 | `pvs[]` | 6 × 256 bytes (1536 bytes) |
| 1868 | `bests[]` | 6 × `Move` |

---

## Syzygy tablebases

v3.00+ supports **Syzygy endgame tablebases** (3-4-5 piece) via the [Fathom](https://github.com/jdart1/Fathom) library.

- **WDL probing** during search — returns exact Win/Draw/Loss scores for positions with ≤5 pieces
- **DTZ probing** at the root — selects the fastest winning move
- **Draw cutoff** — Stockfish-style WDL draw handling: draws return 0cp immediately (no further search), while wins/losses are stored in the TT at depth+6; blessed/cursed (50-move-rule-affected) results use the current depth instead
- **Insufficient material** — KvK, KBvK, KNvK detected as dead draws via a fast bitboard check (a handful of OR operations + popcount), returned immediately with zero NNUE evaluation overhead
- **Square/bitboard mapping** — Zchezz's a8=0 maps to Fathom's a1=0 via `sq ^ 56`; bitboards are vertically flipped with `__builtin_bswap64()`
- **Key guards** — `rule50 == 0` is checked before every probe, and a cardinality filter restricts deep-node-only probing once piece count equals the configured limit
- **Thread-safe** — WDL probes can be called from search threads
- **No tablebases required** — the engine works identically without them
- **WASM builds** automatically exclude tablebase code (compiled with `-DNO_TABLEBASES`)
- **Download** — tables live in `tablebases/` (gitignored) and come from [tablebase.sesse.net](http://tablebase.sesse.net/syzygy/3-4-5/)

To use tablebases, download the Syzygy 3-4-5 piece tables (~938 MB) and set the path:

```
setoption name SyzygyPath value /path/to/tablebases
```

Table files are available from [tablebase.sesse.net](http://tablebase.sesse.net/syzygy/3-4-5/).

---

## File structure

See `docs/folder_structure.md` for the full, regularly-regenerated tree. Summary:

```
zchezz/
├── engine/
│   ├── build/                      SHARED build system (Makefile, build_*.bat/.sh, bundle.py, pieces/)
│   ├── c/tools/                    SHARED native tool sources (selfplay.c, arena.c/.h) — current-API-only, see § engine/c/tools/ below
│   ├── c/zchezz_v314/              Frozen, deployed engine (~2900 Elo) — playable in the browser today
│   └── c/zchezz_v400/              Engine source code (current — NNUE not yet trained)
│       ├── board.c / board.h          Board state, bitboards, magic attacks, make/unmake
│       ├── search.c / search.h        Alpha-beta, staged move gen, TT, LMR, NMP, Lazy SMP
│       ├── nnue.c / nnue.h            NNUE inference, incremental accumulator, NNU4 loader
│       ├── main.c                     UCI protocol, entry point, SMP threads, perft
│       ├── syzygy.c / syzygy.h + tbprobe.*/tbchess.c/tbconfig.h   Fathom tablebase integration
│       ├── book.c / book.h            Polyglot opening book support
│       ├── nnue_weights.bin           Trained weights (NNU4 format, ~2.6 MB) — not present until trained
│       ├── zchezz_wasm.html           Browser UI source
│       └── README.md                  v4.00 architecture status (Portuguese)
│
├── tests/                         Test & match scripts (flat, version-less — see naming convention below)
├── train/                         NNUE training code (PyTorch, flat, version-less)
│   └── labeling/                  Stockfish-based dataset labeling scripts (was sf_analyze/)
├── utils/                         kill_ghosts.py only (Termux files moved to engine/build/)
├── docs/                          folder_structure.md, v400_implementation_plan.md
├── openings/                      Opening books (gitignored) — lines/ (PGN), positions/ (EPD), book.bin
├── endgames/                      Endgame EPD test positions (gitignored)
├── tablebases/                    Syzygy 3-4-5 piece tables (gitignored, ~938 MB)
├── data/ · checkpoints/           Training data and checkpoints (gitignored)
└── index.html                     GitHub Pages deployment (auto-updated by build_wasm.bat)
```

### How every tool is configured

Each Python tool opens with a `CONFIGURATION` block: one documented constant per
setting, with units. **That block is the interface** — running the tool with no
arguments does exactly what it says, and editing a constant is the normal way to
change a run. Every constant is also a command-line flag that overrides it for
scripted or one-off use, and the flag's default IS the constant, so the help text
can never disagree with the file:

```bash
python tests/run_tournament.py                        # runs the config block
python tests/run_tournament.py --games 200 --movetime 100
python tests/run_tournament.py --show-config          # print settings, run nothing
python tests/run_tournament.py --help                 # every flag + its real default
```

`--show-config` exists everywhere and is the cheap way to check a long job before
starting it. Booleans always come in pairs (`--pgn` / `--no-pgn`), so a constant
that defaults to on can still be turned off from the command line. List settings
are repeatable flags, and repeating one replaces the configured list rather than
appending to it.

The plumbing lives in `utils/cliconf.py`, which also holds the **shared
configuration vocabulary**: one name per concept across all tools (`GAMES`,
`CONCURRENCY`, `MOVETIME_MS`, `MAX_PLIES`, `SEED`, `WORKERS`, `RESULTS_DIR`,
`SAVE_PGN`/`SAVE_EPD`/`SAVE_BIN`, `DRY_RUN`, `ONLY`, …). A `DEFAULT_` prefix marks
a knob specific to one tool. The vocabulary is defined in that one file and not
copied into the individual scripts.

### Naming convention — `tests/`, `train/` and `utils/`

`tests/` and `train/` are flat and version-less — one toolset tracking the current
engine, unlike `engine/c/zchezz_vXXX/`, which is version-suffixed and duplicated per
release. Neither ever gets a `vNNN` subfolder. `utils/` holds helpers used by both
(`cliconf.py`) plus standalone maintenance scripts (`kill_ghosts.py`).

**`tests/`:**

| Prefix | Meaning |
|---|---|
| `test_*` | Pass/fail correctness check |
| `bench_*` | Performance measurement |
| `run_*` | Harness that plays games or runs a long job |
| `debug_*` | One-off scratch script (gitignored, not part of the suite) |
| bare noun (e.g. `elo_calc.py`) | Shared library, or a fixture generator (e.g. `make_random_nnu4.py`) |

**`train/`:**

| Form | Meaning |
|---|---|
| bare noun (`encoding`, `model`, `dataset`) | Library module, imported by other scripts |
| `verb_noun` (`train_nnue`, `export_nnu4`, `check_parity`) | Executable script |

### Complete file inventory

Tables below cover every tracked (and gitignore-documented) file in the repo, organized
by directory, with what changed name in the reorg noted in the third column.

#### Root

| File | What it does | Renamed from |
|---|---|---|
| `README.md` | This file — full project documentation | `Readme.md` |
| `CLAUDE.md` | Development guide and project rules (versioning, testing phases, build layout) | — |
| `index.html` | GitHub Pages deployment — the WASM bundle served at the public play URL | — |
| `.gitignore` | Ignore rules for build artifacts, generated data, and large downloaded assets | — |
| `docs/folder_structure.md` | Regenerated repo-tree reference | `FolderStructure.md` |
| `docs/v400_implementation_plan.md` | v4.00 HalfKP-4Bucket architecture design doc (Portuguese) | `zchezz_v400_implementation_plan.md` |

#### `engine/build/` — shared build system (new directory)

| File | What it does | Renamed from |
|---|---|---|
| `Makefile` | `ENGINE=vXXX`-selectable targets: `native`, `wasm`, `bundle`, `selfplay`, `arena` | new (was one `Makefile` per version folder) |
| `build_native.bat` | Windows native compile script, `ENGINE` selectable via arg (default v400) | per-version `compile_zchezz.bat` |
| `build_wasm.bat` | Windows WASM compile + HTML bundle script | per-version `build_wasm.bat` |
| `build_termux.sh` | Automated Termux build/test suite | `utils/RunZchezzTermux.sh` |
| `termux.md` | Termux quick-reference command cheat sheet | `utils/RunZchezzTermux.md` |
| `bundle.py` | Merges compiled WASM + JS + NNUE weights + SVG pieces into one self-contained HTML file | per-version `bundle.py` |
| `pieces/cburnett/*.svg` | CBurnett piece set (default) used by `bundle.py` | `/pieces/cburnett/*.svg` (repo root) |
| `pieces/merida/*.svg` | Merida piece set used by `bundle.py` | `/pieces/merida/*.svg` (repo root) |
| `pieces/staunty/*.svg` | Staunty piece set used by `bundle.py` | `/pieces/staunty/*.svg` (repo root) |
| `build_selfplay.bat` | Windows one-click compile of `selfplay.exe`, `ENGINE` selectable via arg (default v400) | new |
| `arena.exe`, `selfplay.exe` | Compiled shared tool binaries (gitignored, built here rather than per-version) | — |

#### `engine/c/tools/` — shared native tool sources (new directory)

Tracks the **current** engine API only (`TTable`, `NnueNet`, `SearchParams` fields that
only exist in the newest `zchezz_vXXX/`) — will not compile against `zchezz_v314`. That's
by design, not a defect to fix: `arena.c` compares engine **versions** for the SPRT
promotion gate (CLAUDE.md rule 9) by driving an already-built `zchezz.exe` as an external
UCI subprocess (its `uci:` player kind) — it never links an old version's `.c` files
directly; only same-version, in-process comparisons (two weight files, or two search-constant
sets, against the *current* engine) use the fast in-process `net:` player kind. `selfplay.c`
always generates data from the current engine — there is no cross-version self-play
requirement. So when a new engine version is cut, these tool sources are **not** copied
into it — they keep compiling as-is against whichever folder `ENGINE=` points at. This
directory has no separate README; the rationale lives here. It has been
merged here and the file deleted.

| File | What it does | Renamed from |
|---|---|---|
| `selfplay.c` | N-games-in-parallel self-play data generator; emits packed `.bin` and PGN | `zchezz_v400/tools/selfplay_native.c` |
| `arena.c` / `arena.h` | A/B match harness — the SPRT strength gate for the bootstrap loop; embeddable as a library for a future SPSA tuner | new |
| `opening_pool.c` / `opening_pool.h` | Shared SAN↔Move conversion and byte-offset opening-book indexing, used by both `arena.c` and `selfplay.c` | factored out of `arena.c` |
| `sample.h` | Packed `.bin` training-sample record format (`eval_cp`, `game_result`, STM-relative — see § Training-data naming convention in `CLAUDE.md`). 75-byte record: `board[64]` (mailbox, Zchezz encoding, 0=empty, WP=9..BK=22, sq 0=a8), `stm` (uint8), `rule50` (uint8), `castling` (uint8), `ep_file` (uint8, 0..7, 8=none), `eval_cp` (int16, STM-relative, from the search that chose the move), `game_result` (int8, +1/0/-1 from the mover's POV, filled in on a second pass once the game result is known), `move_played` (uint16), `_pad` (uint16) | new |
| `test_sprt_synthetic.c` | Throwaway synthetic SPRT/Elo math check (no games played, hand-picked W/D/L); links `arena.c` compiled with `-DARENA_NO_MAIN` | new |

#### `engine/c/zchezz_v314/` — frozen, deployed engine

Unmodified since release, per the "never modify a released version" rule — no renames.
Holds its own `Makefile`, `compile_zchezz.bat`, `build_wasm.bat`, `bundle.py`, and
`zchezz_bundle.html` (the shared build system in `engine/build/` only targets v400+).

#### `engine/c/zchezz_v400/` — current engine (core only)

Engine core sources plus the browser UI source. Build files, tool sources, and piece
SVGs live in the shared directories above. There is one README for the whole project,
not one per version folder.

| File | What it does |
|---|---|
| `board.c` / `board.h` | Board state, magic-bitboard attacks, Zobrist hashing, make/unmake |
| `search.c` / `search.h` | Alpha-beta, staged move generation, TT, LMR/NMP/pruning, Lazy SMP |
| `nnue.c` / `nnue.h` | HalfKP-4Bucket NNUE inference, incremental accumulator, NNU4 loader |
| `main.c` | UCI protocol handling, entry point, SMP thread spawn, perft |
| `syzygy.c` / `syzygy.h` | Zchezz ↔ Fathom tablebase integration layer |
| `tbprobe.c` / `tbprobe.h`, `tbchess.c`, `tbconfig.h` | Vendored Fathom tablebase library |
| `book.c` / `book.h` | Polyglot opening book support |
| `poly_keys.h`, `stdendian.h` | Polyglot Zobrist key table; endian-portable helpers |
| `nnue_weights.bin` | Trained weights, NNU4 format (~2.6 MB) — **not present**, network untrained |
| `zchezz_wasm.html` | Browser UI source (bundled into `index.html` by `build_wasm.bat`) |
| `zchezz.exe` | Native Windows binary (gitignored, built per-version) |

#### `tests/` — flat, version-less test & match scripts

| File | What it does | Renamed from |
|---|---|---|
| `test_perft.py` | Perft correctness across 37 positions (make/unmake edge cases) | `perft_test.py` |
| `test_uci.py` | Minimal UCI handshake/search smoke test | new |
| `test_uci_extended.py` | Comprehensive UCI suite — handshake, TB, book, MultiPV, threads, options | `uci_test.py` |
| `test_browser.py` | Automated browser tests for the HTML/WASM UI (clocks, MultiPV, analysis) | `browser_test.py` |
| `test_browser_html.py` | Static HTML feature validation (no browser needed) | `test_html_features.py` |
| `test_book.py` | Validates opening book entries for legality and quality | `validate_book.py` |
| `test_move_parsing.py` | Verifies the engine applies long move sequences correctly | new |
| `test_nnue_accumulator.py` | Drives a verify-instrumented build to prove the incremental accumulator matches a from-scratch rebuild | new |
| `test_selfplay_bin.py` | Sanity-checks `.bin` shards produced by `engine/c/tools/selfplay.c` | new |
| `test_two_nets.c` | Proves `NnueNet` is truly per-instance — two nets loaded and evaluated independently in one process | new |
| `run_tournament.py` | Universal tournament runner — H2H, anchor-ELO estimation, EPD suites, all via one config block | `tournament.py` |
| `run_tournament_quick.py` | Quick 200-game H2H regression test between two engine versions | `tournament_quick.py` |
| `run_selfplay.py` | Python/UCI-subprocess self-play data generator (PGN + Stockfish relabel pipeline) | `selfplay.py` |
| `run_suite.py` | EPD test suite runner (WAC, STS, etc.) | `suite_runner.py` |
| `run_arena.py` | Driver/wrapper for the native A/B arena (`engine/c/tools/arena.c`) | new |
| `run_selfplay_native.py` | Driver/wrapper for the native selfplay generator (`engine/c/tools/selfplay.c`), analogous to `run_arena.py` | new |
| `bench_nps.py` | 50-position NPS + eval sanity benchmark across game phases | — |
| `elo_calc.py` | Shared ELO difference + confidence-interval library (trinomial and pentanomial models) | — |
| `compare_suites.py` | Compares EPD suite results between engine versions | `suite_compare.py` |
| `make_random_nnu4.py` | Emits a valid random-weight NNU4 binary fixture for structural testing before any training data exists | new |
| `debug_game.py` | One-off scratch script — finds which move desyncs the engine (gitignored) | `test_debug_game.py` |
| `debug_engine.py` | One-off scratch script — diagnoses the native UCI engine layer by layer (gitignored) | new |
| `suites/` | EPD test suite data (gitignored) | root `test_suites/` |

#### `train/` — flat, version-less NNUE training code

| File | What it does | Renamed from |
|---|---|---|
| `encoding.py` | Single source of truth (Python side) for HalfKP-4Bucket feature encoding | new |
| `model.py` | HalfKP-4Bucket model definition, SCReLU/ClippedReLU, QAT fake-quantization | new |
| `dataset.py` | Packed `.bin` self-play dataset reader (numpy structured dtype, memmap, multi-shard) | new |
| `train_nnue.py` | Training script (QAT, NNU4) for HalfKP-4Bucket | `mixtrain.py` |
| `export_nnu4.py` | Converts a training checkpoint into the NNU4 binary weight format | `convert_nnue.py` |
| `check_parity.py` | Cross-checks the Python feature encoder against the compiled `nnue.c` (17,720 cases) | new |

L1 is a sparse `nn.EmbeddingBag`, not `nn.Linear` — with only ~30 active features out of
2560, a dense `(N, 2560)` uint8 tensor would waste over 98% of the matmul and cost 2.5 KB
per position, which doesn't scale to millions of positions. Direct consequence: `L1W` is
stored `[2560, 512]` (feature-major) in the NNU4 file — the `EmbeddingBag` weight is
already in that layout, so `export_nnu4.py` does **not** transpose it, unlike an earlier
plan that assumed a `[512, 2560]` dense layer.

#### `train/labeling/` — Stockfish-based dataset labeling (was `sf_analyze/`)

| File | What it does | Renamed from |
|---|---|---|
| `process_positions.py` | **The one position pipe.** Reads `.epd`, `.pgn`, `.bin` or `.parquet`; optionally filters (quiet, endgame, score cap, dedup) and re-evaluates with Stockfish; writes any combination of `parquet`/`bin`/`epd`/`pgn`. With `--filters none` it is a plain format converter | replaces `label_selfplay.py`, `label_quiet_*.py`, `label_all_quiet.py`, `filter_wdl.py` |
| `generate_endgames.py` | Generates synthetic endgame positions by material group and labels them with Stockfish | `endgame_generator.py` |
| `normalize_columns.py` | Rewrites a dataset to the canonical `fen`/`cp`/`result` column shape | new |
| `merge_datasets.py` | Joins two extractions of the SAME positions (one with `cp`, one with `result`) into one dataset, hash-joined on FEN with FEN re-verification | new |
| `fix_column_names.py` | Renames or drops a mislabelled column after re-verifying, per file, that the column really holds what the name claims | new |

**Format conversion is not a separate tool.** Anything that reads positions and
writes positions goes through `process_positions.py`:

```bash
# .pgn games -> packed .bin training samples, no filtering
python train/labeling/process_positions.py --in games.pgn --out gen7.bin --filters none

# a raw self-play EPD folder -> filtered parquet dataset (the default pipeline)
python train/labeling/process_positions.py --in data/selfplay_raw --out data/selfplay_filtered

# see exactly what a bare run would do, without touching the disk
python train/labeling/process_positions.py --show-config
python train/labeling/process_positions.py --dry-run
```

#### `utils/`

| File | What it does | Note |
|---|---|---|
| `cliconf.py` | Config-block + CLI plumbing shared by every Python tool, and the single copy of the shared configuration vocabulary (one name per concept, project-wide) | new |
| `kill_ghosts.py` | Force-kills stray engine/helper processes left over from crashed runs | Termux docs moved to `engine/build/`; `OpeningBook.bin` moved to `openings/book.bin` |

#### `openings/` (gitignored — large downloaded data)

| Path | What it does | Renamed from |
|---|---|---|
| `lines/*.pgn` | Flat PGN opening-move-sequence files used to vary game starts | — |
| `positions/*.epd` | Opening position EPDs (+ `.idx` byte-offset indexes) | `openings_positions/` |
| `book.bin` | Polyglot opening book | `utils/OpeningBook.bin` |

#### `endgames/` (gitignored)

EPD endgame test positions (`endgames.epd`, `endgames_cdb95105.epd` + `.idx` indexes) used by Phase 4 of the regression suite.

#### Other gitignored directories

`tablebases/` — Syzygy 3-4-5 piece tables (~938 MB). `data/`, `checkpoints/` — training
data and model checkpoints. `test_suites/` (the old root-level EPD suite folder) was
removed; its contents live at `tests/suites/` now.

---

## Requirements

**Engine (build)**

- GCC ≥ 9 or Clang ≥ 10 (C11)
- Emscripten ≥ 3.1 (WASM target only)

**Bundler (`bundle.py`)**

- Python ≥ 3.10

**Training (optional)**

- PyTorch, python-chess, pandas, pyarrow, numpy

---
