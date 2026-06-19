# Zchezz

A chess engine written in C with a custom-trained NNUE evaluation, playable directly in the browser via WebAssembly. Runs natively on Windows, Linux, and Android (Termux), and compiles to a single self-contained HTML file that works fully offline.

Entirely vibe coded with **Claude Sonnet 4.6**.

▶ **[Play against Zchezz in your browser](https://ciclopeestrabico.github.io/zchezz/zchezz_bundle.html)**

**Measured Elo: ~2781 ±4** (against Stockfish anchors at movetime 200ms, without opening book or tablebases).

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

---

## NNUE

### Architecture

The network is a fully-connected feed-forward network (NNUE — Efficiently Updatable Neural Network) with a 799-feature input, two hidden layers, and a scalar WDL output. Starting from version NNU3, all layers use integer fixed-point arithmetic at inference time.

```
Input: 799 features  (float32 during training, cast to int at inference)
  ├── 768  Half-Mirror (HM) bitboard encoding
  │         12 piece types × 64 squares, side-to-move perspective
  └──  31  Endgame features (appended, computed per position)
            stm piece counts   ×6   (normalized)
            opp piece counts   ×6   (normalized)
            total material     ×1   (normalized by 78)
            constant bias      ×1
            stm passed pawns   ×8   (per file, binary)
            opp passed pawns   ×8   (per file, binary)
            king–king distance ×1   (Chebyshev, normalized by 7)

L1: int16[799 → 256]  bias int32[256]  + ClippedReLU(0, 255)  ← incremental accumulator (dual POV)
L2: int8[256 → 64]    bias int32[64]   + ClippedReLU(0, 64)
L3: float32[64 → 1]   bias float32[1]  + Sigmoid → ×320 cp, clamped ±2000
```

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

**L3 (float32).** L3 weights are quantized to `int8_t` (scale QB) in the NNU3 file but the final dot product `L3W · relu2` uses the integer weights scaled back to float for the final scalar. The bias is kept as `float32` — this 64→1 dot product is not a bottleneck and avoiding one more integer scale conversion keeps the output path simple and exact.

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

**In-memory WASM loader.** Weights are loaded directly from an `ArrayBuffer` into the WASM heap via `nnue_load_from_mem`, avoiding any filesystem dependency in the browser. The L2 transpose is applied during loading, so the runtime weight layout is always optimal regardless of how the NNU3 file was produced.

---

### Weight file format — NNU3

```
Offset  Size    Content
──────────────────────────────────────────────────────────────────────────────────
0        4 B    Magic: "NNU3"
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


## Board representation

The board layer is the foundation everything else is built on. It is designed for both correctness and speed.

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

The engine always starts at depth 1 and searches progressively deeper, retaining the best move from each completed iteration. The previous iteration's score seeds the aspiration window for the next. If the search is interrupted by the time or node budget, the last completed iteration's best move is returned.

### Aspiration windows

From depth 3 onward, each iteration opens with a symmetric window of ±20 cp around the previous score. On a fail-low or fail-high the window expands exponentially (doubling each miss, capped at 500 cp) and the search is re-run. After up to 6 re-searches it falls back to a full-width window.

### Transposition table

The TT is a **Structure-of-Arrays** layout with separate flat arrays for hash, score, depth+flag, generation, best move, and static eval. This keeps each field's access pattern cache-friendly.

- **Native:** 4 M entries (~112 MB)
- **WASM:** 512 K entries (~14 MB)
- **Generation aging:** each call to `search_best` increments `TT_GEN`. Stale entries still yield the stored move for ordering but their score is not used for cutoffs.
- **Mate distance correction:** scores above ±9000 cp are stored and read back with a ply adjustment so mate-in-N scores are correctly compared across different depths.

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
- **Thread safety**: TT is lock-free (struct-of-arrays layout); per-thread state uses static arrays indexed by thread ID

### Quiescence search

Quiescence is entered at depth 0 and searches captures and promotions only (plus all moves when in check). It applies:

- **Stand-pat** cutoff against beta
- **Delta pruning** globally (queen value + 50 cp, or 2× queen + 50 if a passer is on the 7th rank)
- **Per-move delta pruning** — skips captures where `stand_pat + capture_gain + 50 < alpha`
- **SEE pruning** — losing captures (`SEE < 0`) are skipped entirely
- **Pick-best** (selection sort) rather than full sort — avoids scoring moves that will be pruned before being searched

### Static Exchange Evaluation (SEE)

SEE uses the magic bitboard attack tables directly. The attacker board is rebuilt via a bitboard occupancy mask and updated incrementally as pieces are removed, so discovered attackers (X-ray attacks through vacated squares) are revealed automatically. This makes SEE both fast and exact — the full minimax retrograde scoring is computed in a temporary scratch board.

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
- **Eval bar** — animated white/black percentage bar with centipawn score
- **Multi-PV analysis** — display 1 to 3 best engine lines simultaneously. Use the +/− buttons in the analysis toolbar to add or remove lines
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

### Native — Linux / Windows

```bash
make
./zchezz --nnue nnue_weights.bin
```

### Android / Termux

```bash
make termux
./zchezz --nnue nnue_weights.bin
```

### WebAssembly (requires [Emscripten](https://emscripten.org))

```bash
make wasm
# Produces: zchezz_wasm.js + zchezz_wasm.wasm
# Serve alongside nnue_weights.bin and zchezz_wasm.html
```

### Self-contained offline HTML

```bash
make wasm-bundle
# Produces: zchezz_bundle.html — double-click to play, no server needed
```

The bundler (`bundle.py`) reads the compiled WASM binary, the NNUE weight file, and the JS worker, base64-encodes them, and splices everything directly into the HTML as inline constants. It also extracts Merida piece SVGs from `python-chess` and injects them so the Merida piece style is available without any network request. The version string is parsed automatically from the current folder name (e.g. `zchezz_v163B` → `1.63b`). The resulting file is fully self-contained — no server, no network, no dependencies.

### Tests

```bash
make test
```

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

---

## Syzygy tablebases

v3.00 supports **Syzygy endgame tablebases** (3-4-5 piece) via the [Fathom](https://github.com/jdart1/Fathom) library.

- **WDL probing** during search — returns exact Win/Draw/Loss scores for positions with ≤5 pieces
- **DTZ probing** at the root — selects the fastest winning move
- **Thread-safe** — WDL probes can be called from search threads
- **No tablebases required** — the engine works identically without them
- **WASM builds** automatically exclude tablebase code (compiled with `-DNO_TABLEBASES`)

To use tablebases, download the Syzygy 3-4-5 piece tables (~938 MB) and set the path:

```
setoption name SyzygyPath value /path/to/tablebases
```

Table files are available from [tablebase.sesse.net](http://tablebase.sesse.net/syzygy/3-4-5/).

---

## File structure

```
engine/c/zchezz_v305/
├── board.c / board.h          Board state, bitboards, magic attacks, make/unmake
├── search.c / search.h        Alpha-beta, TT, all pruning and heuristics, Lazy SMP
├── nnue.c / nnue.h            NNUE forward pass, incremental accumulator, NNU3 loader
├── main.c                     UCI protocol, entry point, SMP thread management
├── syzygy.c / syzygy.h        Zchezz ↔ Fathom integration layer
├── book.c / book.h            Polyglot opening book support
├── tbprobe.c / tbprobe.h      Fathom library (tablebase probing)
├── tbchess.c                  Fathom internal move generator
├── tbconfig.h                 Fathom configuration for Zchezz
├── stdendian.h                Endianness compatibility shim
├── poly_keys.h                Polyglot Zobrist key constants
├── Makefile                   Native / WASM build targets
│
├── zchezz_wasm.html           Browser UI source (game + analysis)
├── zchezz_bundle.html         Fully self-contained offline bundle (output of bundle.py)
├── bundle.py                  Offline HTML bundler — embeds WASM + weights + UI into one file
│
└── nnue_weights.bin           Trained weights (NNU3 format, ~426 KB)

pieces/                        SVG piece sets (cburnett, merida, staunty)
```

---

## Requirements

**Engine (build)**

- GCC ≥ 9 or Clang ≥ 10 (C11)
- Emscripten ≥ 3.1 (WASM target only)

**Bundler (`bundle.py`)**

- Python ≥ 3.10
- `python-chess` (optional — required only for Merida SVG piece embedding)
- PyTorch, python-chess, pandas, pyarrow, numpy

---

## Next steps

- Staged move generation — generate captures first, avoid generating quiet moves when a beta cutoff is found early
- WASM multi-threading — Web Workers for browser SMP support
- NNUE retraining — new generation of self-play data with v305 for improved evaluation
- 6-piece Syzygy support — extend tablebase probing to 6-piece endgames
