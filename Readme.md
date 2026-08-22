# Zchezz ♟️

**A 2900 ELO chess engine, entirely vibe coded with Claude.**

Written in C with a custom-trained NNUE evaluation, playable directly in the browser via WebAssembly. Compiles to a single self-contained HTML file that works fully offline — no server, no install, no dependencies. Just double-click and play.

▶ **[Play against Zchezz in your browser](https://gitzambrano.github.io/zchezz/)**

---

### Highlights

|                        |                                                                            |
| ---------------------- | -------------------------------------------------------------------------- |
| **Version**      | v4.02 —`engine/c/zchezz_v402/`                                          |
| **Search**       | Alpha-beta with staged move generation, aspiration windows, PVS, Lazy SMP  |
| **Evaluation**   | Custom NNUE, HalfKP-4Bucket — int16/int8 quantized, AVX2 SIMD + WASM SIMD |
| **Endgames**     | Syzygy tablebase support (3-4-5 piece WDL + DTZ probing)                   |
| **Opening book** | Polyglot .bin format, with built-in ECO opening name recognition           |
| **Analysis**     | Multi-PV (up to 5 lines), eval bar, blunder detection, eval graph          |
| **Platforms**    | Windows, Linux, Android (Termux), WebAssembly (any modern browser)         |
| **Offline**      | Single HTML bundle — works from`file://`, no server needed              |
| **UCI**          | Full UCI protocol compliance (15 commands, 12 configurable options)        |

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

The network is trained in PyTorch on quiet-position data: human games (lichess and
assorted collections) evaluated by Stockfish, several generations of iterative
self-play, and generated endgame positions. **173.5 M positions.**

Datasets store two columns: `cp`, the evaluation in centipawns, and `result`, the
real game outcome (0.0 / 0.5 / 1.0). Both are White-relative. `wdl` is`sigmoid(cp / 320)` — a transform of `cp`, never an outcome, and never stored, since a stored copy can rot apart from the `cp` it came from. The training target is the blend `lam * result + (1 - lam) * wdl`, computed at training time with a per-dataset `lam`.

### Training the network

Everything is set in the `CONFIGURATION` and `DATASETS` blocks at the top of
`train/train_nnue.py`. A bare run does exactly what those blocks say; CLI flags
only override them.

| Setting      | Meaning                                                                                                                                                                                   |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `LR`       | learning rate. ~1e-3 from random weights; ~1e-5 to refine a trained net                                                                                                                   |
| `EPOCHS`   | also sets the schedule:`CosineAnnealingLR(T_max=EPOCHS)`, so a small value anneals the LR quickly                                                                                       |
| `DATASETS` | per dataset:`pct` (fraction used per epoch, 0.0 disables), `mode` (`lines` samples rows, `shards` samples whole files), `lam` (how much game result is blended into the target) |

The target is continuous (0..1), not 0/1, so the BCE floor is **not** 0.693 — it
is the entropy of the target distribution, measured at **0.620**. Compare
`val_loss` against that; `val_mae` is the direct error in wdl units.

```bash
python train/train_nnue.py                 # trains what the config block says
python train/train_nnue.py --show-config   # prints resolved settings, runs nothing
python train/export_nnu4.py                # checkpoint -> NNU4 weight file
```

---

## NNUE

### Architecture — HalfKP-4Bucket

Implemented in `engine/c/zchezz_v402/nnue.c`.

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

**L3 (float32).** L3 weights are quantized to `int8_t` (scale QB) in the NNU4 file but the final dot product `L3W · relu2` uses the integer weights scaled back to float for the final scalar. The bias is kept as `float32` — this 32→1 dot product is not a bottleneck and avoiding one more integer scale conversion keeps the output path simple and exact.

**Scale chain summary:**

```
Input features:  binary {0, 1}
After L1 matmul: int32 scale QA  →  SCReLU  →  uint8 [0, 255]
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

Post-training quantization degrades accuracy because a float32 network is free to
use values outside the integer range. QAT simulates quantization *during* training,
so the network learns to work within the integer constraints.

- **ClippedReLU** replaces ReLU after L1 and L2: `clamp(x, 0.0, 1.0)`. Training
  output stays in `[0, 1]`, and `1.0` maps exactly to `QA = 255` — a clean
  bijection with the `uint8` inference range, no boundary rounding artifacts.
- **Fake quantization.** Each forward pass snaps weights to the nearest
  representable integer and divides back to float; the backward pass treats the
  rounding as identity (Straight-Through Estimator), so the optimizer can push
  weights toward better integer-grid values without losing gradient signal.
  Applied to L1, L2 and L3 weights, biases, and post-ReLU activations.
- **Hard clamping** after every optimizer step, to `32767/QA ≈ 128.5` (L1) and
  `127/QB ≈ 1.984` (L2/L3). Without it, Adam's momentum carries weights past the
  clamp on the next step, spending gradient budget on values that will be clipped.

### Performance optimisations

**AVX2 int16 accumulation (L1).** The L1 update uses `_mm256_add_epi16` to process 16 `int16_t` elements per register — twice the throughput of float32 `_mm256_add_ps`. The maximum safe accumulator value is `32 pieces × 32767 = 1,048,544`, well within `int32_t`, so there is no risk of overflow during the dot product.

**Cached extra-feature delta.** The 31 endgame feature rows are folded into a per-stack-slot precomputed delta (`_ext_buf`). At eval time no board scan or 31-row multiply is needed — the precomputed delta is simply added to the HM accumulator before ClippedReLU.

**Output-major L2 weights.** The L2 weight matrix is stored as `[NN_L2_OUT][NN_L2_IN]` = `[32][1024]`, row-major per output, so each output neuron's 1024 weights are contiguous and the `maddubs` kernel streams `relu1[1024]` once per output. GCC `-O3` auto-vectorises with AVX2 on native and Emscripten `-msimd128` on WASM.

**Sparsity skip.** The L2 forward pass short-circuits any input where `relu1[i] == 0`, exploiting the natural post-ReLU sparsity of the first hidden layer.

**In-memory WASM loader.** Weights are loaded directly from an `ArrayBuffer` into the WASM heap via `nnue_load_from_mem`, avoiding any filesystem dependency in the browser. The L2 transpose is applied during loading, so the runtime weight layout is always optimal regardless of how the NNU4 file was produced.

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
## Search (v4.02 changes, branch `v402-search-strength`)

v4.02 keeps the v4.01 architecture and network and reworks search policy
and memory layout. Every change below was gated with 800-game blind
arenas at 100 ms/move against the previous binary before being kept;
changes that regressed were reverted (see "Rejected" list).

### Kept (validated)

- **TT entries are packed 24-byte structs** (`TTEntry`, AoS) instead of six
  parallel arrays: a probe touches one cache line instead of up to six.
- **TT generation is stable within a game.** The generation counter is bumped
  once per `ucinewgame` (and physically cleared by self-play/arena per game),
  never per move. Scores from previous moves are reusable again.
- **The root never takes TT score cutoffs** (only TT-move ordering), so a deep
  entry written by an earlier search of the same position cannot end a new
  search with no PV.
- **Aborted searches store nothing.** When `time_up` fires, `tt_store` calls in
  `alpha_beta`/`qsearch` are skipped: bounds unwinding out of an aborted
  subtree are garbage, and with a stable generation they would poison later moves.
- **Futility pruning runs before `board_make`**, guarded by a cheap magic-based
  direct-check test (`quiet_direct_check`), saving make/unmake + NNUE push/pop
  for pruned quiets.
- **qsearch persists fail-highs and improved results** (depth-0 entries; EXACT
  only when a real searched move produced the score).
- **AVX-VNNI `dpbusd` kernel for NNUE L2** (~+7% NPS, bit-exact result).
- **`board_is_attacked` early-exit reordering; NNUE dirty-side skip on
  `nnue_push_na`; weight-row prefetching.**
- **GA-tuned search constants** (`SearchTunables` defaults): rfp_mult 105,
  rfp_improving_bonus 24, nmp_eval_bonus_threshold 134, probcut_margin 215,
  fut_mult 91, fut_improving_adj 76.
- **SEE pruning of quiet moves** at depth <= 3 (`see_board` extended to empty
  targets); killers/counters/direct checks exempted (-24% nodes).

### Measured strength (800 games, 100 ms/move, single-threaded)

| match | Elo |
|---|---|
| v402 vs v401 | **+157 +/- 20** |
| v402 vs v314 | **-146 +/- 22** (was -268 +/- 30) |

Phase checks: perft 37/37, UCI extended 119/120 (known non-blocking T3.2c),
bench node counts deterministic.

### Rejected after negative/tie SPRTs

capture history + 3rd CMH slot (+/-0 but more nodes), removing history aging,
per-search TT generation bump removal without abort guards (-88),
stand-pat fail-high stores, counter-move persistence, exact per-piece direct
checks + UNKNOWN check hints (-35 at fast TC), evasion CMH ordering in qsearch,
a second GA harvest round (noise).

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

| Technique                            | Condition                                                                                                                                            |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Null Move Pruning**          | Non-PV, not in check, has major pieces,`static_eval >= beta`. R = 3 + depth/3, capped at 6. +1 if eval margin > 200.                               |
| **Reverse Futility Pruning**   | Non-PV, non-check, depth 2–9. Margin =`depth × 90`, reduced by 50 cp if improving.                                                               |
| **Futility Pruning**           | Non-PV, quiet, not giving check, depth 1–4. Margin = 150–600 cp per depth level.                                                                   |
| **Razoring**                   | Non-PV, non-check, depth 1. Drops into qsearch with a tight window.                                                                                  |
| **ProbCut**                    | Non-PV, non-check, depth ≥ 5, beta < 18000. Shallow search (depth − 4) with β + 200 to prune captures.                                            |
| **Late Move Reductions (LMR)** | Applied from the 4th legal quiet move at depth ≥ 3. Reduction =`log(depth) × log(moveIdx) / 1.5`, minimum 1. PV nodes reduce by one less.        |
| **Late Move Pruning (LMP)**    | Non-check, depth ≤ 7. Quiet moves beyond a depth-scaled limit are skipped outright.                                                                 |
| **Singular Extensions**        | Depth ≥ 7, TT hit at depth − 4. If a re-search with`score − depth×2` as beta fails low, the TT move is singular and gets a +1 depth extension. |
| **Check Extension**            | Checks at depth 1 extend by 1 ply to avoid the horizon effect.                                                                                       |
| **IIR**                        | At depth ≥ 4 with no TT move and not in check, depth is reduced by 1 before move generation.                                                        |

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

| Stage                     | What's generated                                       | Why                                                                   |
| ------------------------- | ------------------------------------------------------ | --------------------------------------------------------------------- |
| **TT move**         | Hash move only (no generation needed)                  | Best move from a previous search — often causes a cutoff immediately |
| **Captures**        | `board_gen_captures()` — captures + promotions only | MVV-LVA sorted, SEE-pruned; most cutoffs happen here                  |
| **Quiets**          | `board_gen_quiets()` — non-captures only            | Scored by killer/counter/history; LMR/LMP applied                     |
| **Losing captures** | Deferred bad captures (SEE < 0)                        | Rarely searched — only if all quiets fail to produce a cutoff        |

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
# binary lands in engine/c/zchezz_v402/zchezz.exe
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

The bundler (`bundle.py`) reads the compiled WASM binary, the NNUE weight file, and the JS worker, base64-encodes them, and splices everything directly into the HTML as inline constants. It also embeds the CBurnett, Merida, and Staunty piece SVGs from `engine/build/pieces/` so every piece style is available without any network request. The version string is parsed automatically from the engine folder name (e.g. `zchezz_v402` → `4.02`). The resulting file is fully self-contained — no server, no network, no dependencies.

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

| Offset | Field         | Type                        |
| ------ | ------------- | --------------------------- |
| 0      | `best`      | `Move` (20 bytes)         |
| 20     | `score`     | `int32`                   |
| 24     | `depth`     | `int32`                   |
| 28     | *(padding)* | 4 bytes                     |
| 32     | `nodes`     | `int64`                   |
| 40     | `tb_hits`   | `int64`                   |
| 48     | `pv`        | 256-byte string             |
| 304    | `num_pvs`   | `int32`                   |
| 308    | `scores[]`  | 6 ×`int32` (24 bytes)    |
| 332    | `pvs[]`     | 6 × 256 bytes (1536 bytes) |
| 1868   | `bests[]`   | 6 ×`Move`                |

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
│   ├── c/tools/                    SHARED native tool sources (selfplay.c, arena.c/.h) — current-API-only
│   ├── c/zchezz_v314/              Previous released engine (~2900 Elo, measured) — the build index.html still serves
│   └── c/zchezz_v402/              Engine source code (v4.02)
│       ├── board.c / board.h          Board state, bitboards, magic attacks, make/unmake
│       ├── search.c / search.h        Alpha-beta, staged move gen, TT, LMR, NMP, Lazy SMP
│       ├── nnue.c / nnue.h            NNUE inference, incremental accumulator, NNU4 loader
│       ├── main.c                     UCI protocol, entry point, SMP threads, perft
│       ├── syzygy.c / syzygy.h + tbprobe.*/tbchess.c/tbconfig.h   Fathom tablebase integration
│       ├── book.c / book.h            Polyglot opening book support
│       ├── nnue_weights.bin           Trained weights (NNU4 format, ~2.6 MB)
│       └── zchezz_wasm.html           Browser UI source
│
├── tests/                         Test & match scripts (flat, version-less)
├── train/                         NNUE training code (PyTorch, flat, version-less)
│   └── labeling/                  Stockfish-based dataset labeling scripts
├── utils/                         Shared by tests/ and train/ — cliconf.py (config-block + CLI plumbing), kill_ghosts.py
├── docs/                          folder_structure.md, v400_implementation_plan.md
├── openings/                      Opening books (gitignored) — lines/ (PGN), positions/ (EPD), book.bin
├── endgames/                      Endgame EPD test positions (gitignored)
├── tablebases/                    Syzygy 3-4-5 piece tables (gitignored, ~938 MB)
├── data/ · checkpoints/           Training data and checkpoints (gitignored)
└── index.html                     GitHub Pages deployment (auto-updated by build_wasm.bat)
```

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
