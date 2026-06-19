# Zchezz — Folder Structure

## Root Files

| File | Description |
|------|-------------|
| `Readme.md` | Full project documentation — architecture, training pipeline, ELO history |
| `CLAUDE.md` | AI assistant instructions — build commands, conventions, key decisions |
| `FolderStructure.md` | This file — explains every folder and file |
| `.gitignore` | Excludes binaries, data, external engines, tablebases |

---

## `engine/` — All Engines

### `engine/c/zchezz_v305/` — v3.05: Current Engine

The current and only tracked engine version. Features Lazy SMP, opening book, Syzygy tablebases, MultiPV, and full browser UI.

| File | Description |
|------|-------------|
| `main.c` | UCI protocol handler, entry point, Lazy SMP thread management |
| `board.c` / `board.h` | Board state, bitboards, magic attacks, make/unmake move |
| `search.c` / `search.h` | Alpha-beta, TT, LMR, NMP, move ordering, Lazy SMP (staggered depths) |
| `nnue.c` / `nnue.h` | NNUE inference, incremental accumulator updates, NNU3 loader |
| `syzygy.c` / `syzygy.h` | Zchezz ↔ Fathom integration layer (bitboard mapping) |
| `tbconfig.h` | Fathom library configuration for Zchezz |
| `stdendian.h` | Endianness compatibility shim |
| `book.c` / `book.h` | Polyglot opening book support |
| `poly_keys.h` | Polyglot Zobrist key constants |
| `tbprobe.c` / `tbprobe.h` | Fathom library (gitignored — fetch from upstream) |
| `tbchess.c` | Fathom library (gitignored — fetch from upstream) |
| `nnue_weights.bin` | Trained NNUE weights (NNU3 format, ~426 KB, 799→256→64→1) |
| `Makefile` | Build targets: `native`, `wasm`, `bundle` |
| `build_wasm.bat` | Windows WASM build script (calls emcc + bundle.py) |
| `bundle.py` | HTML bundler (WASM + weights + JS + SVG pieces → single offline HTML file) |
| `zchezz_wasm.html` | Browser UI source (chessboard, analysis panel, settings) |
| `zchezz_wasm.js` | Emscripten-generated JS glue (auto-generated, gitignored) |
| `zchezz_wasm.wasm` | WebAssembly binary (~65 KB, compiled with `-O3 -msimd128`, gitignored) |
| `zchezz_bundle.html` | Standalone HTML (~1.1 MB, works offline, double-click to play) |

### `engine/stockfish/` — Stockfish Anchor *(gitignored)*

| File | Description |
|------|-------------|
| `stockfish.exe` | Full Stockfish with `UCI_LimitStrength` for ELO testing |

### `engine/old/` — Archived Engine Versions *(gitignored)*

All previous Zchezz versions (v152–v304).

---

## `pieces/` — SVG Piece Sets

SVG piece images used by `bundle.py` to embed piece graphics in the standalone HTML bundle.

| Directory | Description |
|-----------|-------------|
| `cburnett/` | CBurnett piece set (default style) |
| `merida/` | Merida piece set |
| `staunty/` | Staunty piece set |

Each directory contains 12 SVG files: `{w,b}{P,N,B,R,Q,K}.svg`

---

## `openings/` — Opening Books (PGN)

20 PGN files used for engine testing. Key files:

| File | Lines | Description |
|------|-------|-------------|
| `Blitz_Testing_4moves.pgn` | 13,000+ | 4-move openings, main testing book |
| `8moves_v3.pgn` | 30,000+ | 8-move deep openings |
| `2moves_LT_1000.pgn` | 1,000 | 2-move short openings |
| `Noomen_Testsuite_2012.pgn` | 300 | Standard test suite |
| `UHO_MEGA_2022_+110_+149.pgn` | 50,000+ | Balanced UHO openings |

---

## `tests/` — Test & Match Scripts

All scripts auto-detect the latest engine version — no hardcoded paths.

| File | Description |
|------|-------------|
| `uci_test.py` | UCI protocol compliance tests |
| `browser_test.py` | Browser/WASM interaction tests (Playwright) |
| `test_html_features.py` | HTML feature validation (parses HTML, no browser needed) |
| `test_tournament.py` | Round-robin tournament runner |
| `validate_book.py` | Opening book legality and quality validation |
| `concurrent_match.py` | Multi-worker A-vs-B match |
| `quick_match.py` | Simple A-vs-B match with paired colors |
| `parallel_match.py` | Parallel match runner |
| `tournament_complete.py` | Full ELO tournament vs multiple Stockfish anchors |
| `tournament_elo.py` | ELO estimation vs configured anchor engines |
| `selfplay.py` | Self-play data generation for NNUE training |
| `suite_runner.py` | EPD test suite runner (WAC, STS, etc.) |
| `suite_compare.py` | Compare suite results between engine versions |
| `debug_engine.py` | Engine debugging utilities (gitignored) |

---

## `test_suites/` — EPD Test Suites

18 EPD files for tactical/strategic testing:

| Suite | Focus |
|-------|-------|
| `wacnew.epd` | Win At Chess — 300 tactical puzzles |
| `sts1-sts15_v6.epd` | Strategic Test Suite (15 categories) |
| `kaufman.epd` | Kaufman positions (25 puzzles) |
| `bratko-kopec.epd` | Bratko-Kopec endgame/middlegame test |
| `nolot.epd` | Nolot difficult positions |
| `lct2.epd` | Louguet Chess Test v2 |
| `eigenmann_endgame_test.epd` | Endgame testing |
| `zugzwang.epd` | Zugzwang positions |
| `fortresses.epd` | Fortress/drawn positions |

---

## `train/` — NNUE Training Code (PyTorch)

| File | Description |
|------|-------------|
| `mixtrain.py` | Main training script — QAT (Quantization-Aware Training), NNU3 format, 799→256→64→1 architecture |
| `convert_nnue.py` | Converts PyTorch checkpoint → `nnue_weights.bin` (NNU3 binary format) |

### `train/test/` — Training Verification Scripts

| File | Description |
|------|-------------|
| `check_ranges.py` | Verify weight ranges after quantization |
| `check_sparsity.py` | Analyze L1 bias sparsity patterns |
| `test_infer.py` | Test NNUE inference against PyTorch reference |
| `test_infer_float.py` | Float precision inference test |
| `test_infer_quant.py` | Quantized inference verification |
| `test_sums.py` | Accumulator checksum validation |
| `test_features.py/c` | Feature extraction correctness |
| `test_avx.c` | AVX instruction test |
| `test_search_bench.c` | Search benchmark with NNUE |

---

## `sf_analyze/` — Stockfish Analysis Scripts

Scripts for generating analyzed training data using Stockfish:

| File | Description |
|------|-------------|
| `sf_analyze_selfplay.py` | Analyze self-play games with SF (nodes 1M) |
| `sf_analyze_quiet_d8.py` | Filter quiet positions, analyze at depth 8 |
| `sf_analyze_quiet_n5000.py` | Filter quiet positions, analyze at nodes 5000 |
| `sf_analyze_all_quietfilter.py` | Combined quiet filter + SF analysis |
| `endgame_generator.py` | Generate synthetic endgame positions |
| `wdl_filter.py` | Filter and merge WDL-labeled datasets |

---

## `utils/` — Utility Files

| File | Description |
|------|-------------|
| `kill_ghosts.py` | Kill orphaned engine processes |
| `OpeningBook.bin` | Binary opening book (Polyglot format) |

---

## Gitignored Data Directories

These directories exist locally but are excluded from git:

| Directory | Description |
|-----------|-------------|
| `data/` | Training data (PGN, EPD, WDL-labeled positions) |
| `checkpoints/` | PyTorch training checkpoints |
| `tablebases/` | Syzygy tablebases (3-4-5 piece, ~938 MB) |
| `engine/stockfish/` | Stockfish binary |
| `engine/old/` | Archived Zchezz versions |
| `endgames/` | Endgame training EPDs |
| `openings_positions/` | Opening position EPDs |

---

## Build Commands

### Windows (GCC/MinGW)
```bash
cd engine/c/zchezz_v305
mingw32-make native    # Windows
make native            # Linux

# Or manually:
gcc -O3 -ffast-math -D_GNU_SOURCE -std=c11 -mavxvnni -mavx2 -I. \
    -o zchezz.exe main.c board.c search.c nnue.c syzygy.c tbprobe.c book.c \
    -static -lm -pthread
```

### WASM (Emscripten)
```bash
build_wasm.bat         # Windows (emcc + bundle.py)
# Or:
make wasm              # Produces zchezz_wasm.js + zchezz_wasm.wasm
make bundle            # Produces zchezz_bundle.html
```

### HTML Bundle
```bash
python bundle.py zchezz_wasm.html zchezz_wasm.js zchezz_wasm.wasm nnue_weights.bin
```
