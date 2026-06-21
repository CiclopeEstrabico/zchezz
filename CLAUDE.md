# Zchezz — Project Guide

## Overview

Zchezz is a chess engine written in C with NNUE evaluation, targeting both native platforms (Windows/Linux/Android) and WebAssembly for browser play. The engine communicates via the UCI protocol.

## CRITICAL DEVELOPMENT RULES

### 1. NO REGRESSIONS — EVER
- Every code change MUST be validated with regression tests before merging.
- **Minimum regression test:** 500 games at 100ms + 200 games at 200ms, 14 concurrency, vs the previous stable version (e.g., v313 vs v312). Calculate combined ELO from all 700 games.
- If ANY test shows regression, **STOP**, diagnose, fix, and restart ALL tests from scratch.

### 2. Versioning
- Each new version gets a new folder: `engine/c/zchezz_vXXX/`
- Version number increments: v312 → v313 → v314, etc.
- NEVER modify a released version's code — create a new version folder instead.

### 3. Testing Workflow (Iterate Until Goal Is Met)

Every new version MUST pass ALL phases in order before deployment.
If ANY phase fails → fix → restart from Phase 1.

#### Phase 1: Pre-Flight Sanity Checks (before ANY games)
These checks catch regressions in seconds. Run after every recompile.

- **Perft:** `python tests/perft_test.py vXXX` — must pass 37/37
- **UCI test:** `python tests/uci_test.py vXXX` — verify handshake, NNUE, search, TB, MultiPV, threads
- **50-Position NPS + Eval Benchmark:** `python tests/bench_nps.py`
  - Tests opening (10), middlegame (15), endgame 6/5/4/3/2-piece (20), insufficient material (5)
  - NPS: compare vXXX-TB vs vXXX-noTB vs previous stable. Fail if >5% drop in opening/middle
  - Eval sanity: startpos ≈ 0cp (±50), KQK > +1000cp, KvK/KBvK/KNvK = 0cp, won endgames > threshold
  - Note: endgame NPS is NOT comparable (TB prunes subtrees → fewer nodes → lower measured NPS)

#### Phase 2: Quick Validation (50 games)
- Fast check that a fix doesn't make things worse
- 50 games at 300ms, 4 concurrency, TB vs noTB or vXXX vs previous

#### Phase 3: TB Effectiveness (200 games from openings)
- 200 games: vXXX-TB vs vXXX-noTB, 300ms movetime, **4 concurrency** (TB disk I/O)
- Target: TB ≥ noTB (ELO difference includes zero or positive in CI)
- If TB is hurting → fix TB integration → restart from Phase 1

#### Phase 4: Endgame TB Test (200 games from /endgames/)
- 200 games from `/endgames/` EPD positions (exchange colors)
- vXXX-TB vs vXXX-noTB, 300ms movetime, 4 concurrency
- TB should show strongest advantage here (fewer pieces = more TB hits)

#### Phase 5: Regression Test (700 games vs previous stable)
- **Stage A:** 500 games at 100ms + 200 games at 200ms, 14 concurrency
- vXXX vs v(XXX-1), no TB for both (pure engine comparison)
- Target: no regression (ELO ≥ -10)
- **Stage B:** Multi-thread verification
  - 100 games: 1 concurrent, Threads=1 vs Threads=4, 100ms movetime
  - Verify no crashes, hangs, or data races
  - Both thread counts should play at similar strength (±20 ELO)

#### Phase 6: WASM Bundle Verification
- Compile: `build_wasm.bat` from engine version directory
- Open `zchezz_bundle.html` in browser and verify:
  - Engine doesn't freeze after opening book moves
  - Analysis MultiPV (4-5 lines) updates ALL lines at depth 14+
  - Infinite search mode progresses smoothly through depths
  - Single PV analysis works correctly

#### Phase 7: Documentation & Cleanup
- Update `README.md` with version changes
- Ensure all source files have comprehensive comments
- Clean up unused generated files

#### Phase 8: Git Commit & Push
- Final git commit with version number and key changes
- Push to GitHub — only final tested version goes to engine folder
- Verify `index.html` (GitHub Pages) is updated via `build_wasm.bat`

#### Phase 9 (EXTRA): ELO Calibration vs Stockfish
- Uses `tournament.py` with Stockfish as anchor (known ELO ~2800)
- 600 games at 100ms + 200 games at 200ms, 14 concurrency
- Calculates absolute ELO rating for the engine
- This is informational — does not block deployment

### 4. Use Idle Time Productively — CODE REVIEW IS FUNDAMENTAL
- While tests are running, **review the engine code line by line**.
- Understand how EVERY function works step by step.
- Look for bugs in TB integration, NNUE accumulator management, thread safety, TT handling, browser integration and so on.
- Check for off-by-one errors, missing guards, race conditions.
- This is NOT optional — it's the most valuable use of idle time and catches bugs before they cost 200-game test runs.

### 5. Git & GitHub Deployment
- Only the FINAL tested version goes to the engine folder on GitHub.
- Commit message must include: version number, key changes.
- `index.html` in repo root is the GitHub Pages deployment (auto-updated by `build_wasm.bat`).
- **Before pushing:** verify `zchezz_bundle.html` works in browser (game play + analysis + MultiPV).

### 6. WASM Bundle Compilation
- Run `build_wasm.bat` from the engine version directory.
- This calls: `emcc` (compile C → WASM) → `bundle.py` (merge HTML + JS + WASM + weights) → copy to `index.html`.
- WASM builds use `-DNO_TABLEBASES -DNO_BOOK` (no file I/O in browser).
- After building, **test in browser**: game play, opening book, analysis mode, MultiPV (4-5 lines at depth 14+).
- The browser version should be mobile first but also work in desktop.

### 7. Documentation & Reporting
- Document all changes in the `README.md` file with clarity and details.
- Document all changes in the engine with clarity and details. We want no regressions in future work.


## Project Structure

```
Zchezz/
├── engine/
│   └── c/zchezz_vXXX/        # Engine source (one folder per version)
│       ├── *.c / *.h          # Engine source files
│       ├── nnue_weights.bin   # NNUE weights (~426 KB, NNU3 format)
│       ├── Makefile           # Build targets: native, wasm, bundle
│       ├── build_wasm.bat     # Windows WASM build script
│       ├── bundle.py          # HTML bundler (WASM + weights + JS → single HTML)
│       ├── zchezz_wasm.html   # Browser UI source
│       └── zchezz_bundle.html # Standalone HTML (~1.1 MB, works offline)
│
├── pieces/                    # SVG piece sets (required for bundle.py)
│   ├── cburnett/              # CBurnett SVGs (default)
│   ├── merida/                # Merida SVGs
│   └── staunty/               # Staunty SVGs
│
├── openings/                  # Opening books (PGN/EPD format)
│   ├── Blitz_Testing_4moves.pgn  # 13K+ openings for testing
│   └── 8moves_v3.pgn            # 30K+ deep openings
│
├── endgames/                  # Endgame EPD positions for testing
│
├── tests/                     # Test & match scripts
│   ├── tournament.py          # Universal tournament runner (H2H + anchors + ELO)
│   ├── tournament_quick.py    # Quick H2H regression test
│   ├── elo_calc.py            # Shared ELO calculation (trinomial, 95% CI)
│   ├── perft_test.py          # Perft correctness (37 positions)
│   ├── bench_nps.py           # 50-position NPS + Eval benchmark (3-way engine comparison)
│   ├── test_uci.py            # UCI protocol compliance tests
│   ├── uci_test.py            # Extended UCI tests
│   ├── selfplay.py            # Self-play data generation for NNUE training
│   ├── suite_runner.py        # EPD test suite runner (WAC, STS, etc.)
│   ├── suite_compare.py       # Compare suite results between versions
│   └── validate_book.py       # Opening book validation
│
├── train/                     # NNUE training code (PyTorch)
│   ├── mixtrain.py            # Main training script (QAT, NNU3)
│   └── convert_nnue.py        # PyTorch → NNU3 binary converter
│
├── tablebases/                # Syzygy tables (gitignored, 3-4-5 piece, ~938 MB)
├── test_suites/               # EPD test suites (WAC, STS, etc.)
├── data/                      # Training data (gitignored)
├── checkpoints/               # Training checkpoints (gitignored)
│
├── Readme.md                  # Full project documentation
├── CLAUDE.md                  # This file — development guide
├── FolderStructure.md         # Detailed folder structure
└── .gitignore
```

## Build Instructions

### Native (Windows/Linux)

```bash
cd engine/c/zchezz_vXXX
# Final binary must be named zchezz.exe:
gcc -O3 -ffast-math -D_GNU_SOURCE -std=c11 -mavxvnni -mavx2 \
    -I. -Wno-unused-variable -Wno-unused-but-set-variable \
    -Wno-maybe-uninitialized -Wno-misleading-indentation \
    -Wno-sign-compare -Wno-unused-function -Wno-parentheses \
    -o zchezz.exe main.c board.c search.c nnue.c syzygy.c tbprobe.c book.c \
    -static -lm -pthread
```

### WebAssembly (Emscripten)

```bash
cd engine/c/zchezz_vXXX
build_wasm.bat   # Compiles WASM + bundles HTML + copies to index.html
```

### Android/Termux

```bash
cd engine/c/zchezz_vXXX
make termux
```

## Architecture Notes

### Board (`board.c`)
- Dual representation: mailbox `uint8_t b[64]` + 12 bitboards `uint64_t bb[12]`
- Magic bitboards for slider attacks
- Zobrist hashing for TT and repetition detection
- Undo stack (no copy-make)

### Search (`search.c`)
- Static global state (no heap allocation in search hot path)
- Per-search state is `_Thread_local` for Lazy SMP support
- TT is Structure-of-Arrays layout (cache-friendly)
- TT probe BEFORE TB probe (Stockfish ordering)
- TB results: wins/losses stored at depth+6, draws cut off immediately (return 0, Stockfish-style). Blessed/cursed use current depth.
- Insufficient material detection: KvK, KNvK, KBvK return draw immediately (zero NNUE overhead)
- Lazy SMP: threads share TT, helpers start at staggered depths
  - Main thread: full ID from depth 1 with aspiration + info output
  - Helper 0: starts at depth 2
  - Helper 1: starts at depth 3
  - Helper 2+: starts at depth 2*i+1
  - TT_GEN only incremented by main thread (no race condition)
- MultiPV: per-PV time budget reset (each PV line gets its own deadline)

### NNUE (`nnue.c`)
- 799→256→64→1 architecture (HalfKP + 31 extra features)
- int16/int8 quantized inference with AVX2 SIMD + WASM SIMD
- Per-thread NnueAccum struct (acc_w, acc_b, ext cache, ~4.3 KB)
  - Main thread: static global `g_nnue_accum` (board.c)
  - SMP helpers: heap-allocated with zmalloc32 (32-byte aligned)
  - Weight arrays remain global read-only (safe for MT)
- Accumulator stack depth 512

### Syzygy Tablebases (`syzygy.c` / `tbprobe.c`)
- Tables: 3-4-5 piece (290 files, ~938 MB)
- Location: `tablebases/` (gitignored)
- Download: `http://tablebase.sesse.net/syzygy/3-4-5/`
- Integration: Fathom library (jdart1 fork)
- Square mapping: Zchezz a8=0 ↔ Fathom a1=0, convert via `sq ^ 56`
- Bitboard mapping: vertical flip via `__builtin_bswap64()`
- Compile without TB: `-DNO_TABLEBASES` (stubs everything out)
- **Key guards:** rule50==0 check before probing, cardinality filter (pieces < limit → all depths, pieces == limit → deep nodes only)

### Opening Book (`book.c`)
- Polyglot .bin format support
- Full Zobrist hash computation client-side
- Binary search for book moves

### WASM / Browser Bundle
- **No multithreading**: WASM build is single-threaded
- **Static state**: all search state is static in WASM (no TLS)
- **TT size**: reduced to 512K entries (~14 MB) in WASM vs 4M entries (~112 MB) native
- **SIMD**: uses `-msimd128` for 128-bit WASM SIMD
- **Bundle**: `bundle.py` injects WASM, NNUE weights, and JS into a single self-contained HTML file
- **SVG pieces**: CBurnett, Merida, and Staunty piece sets embedded from `pieces/` folder
- **MultiPV**: Supports up to 5 analysis lines in the browser UI
- **Analysis UI**: Depth selector (8-20 + ∞), start/stop, PV +/- controls
- **Infinite analysis**: uses `startDepth` optimization — each depth call only searches the NEW depth (TT has previous results)

### SearchResult struct (WASM interop)
The `SearchResult` struct is 1920 bytes, mapped in JS via `DataView`:
- Offset 0: `best` (Move, 20 bytes)
- Offset 20: `score` (int32)
- Offset 24: `depth` (int32)
- Offset 28: padding (4 bytes)
- Offset 32: `nodes` (int64)
- Offset 40: `tb_hits` (int64)
- Offset 48: `pv` (256-byte string)
- Offset 304: `num_pvs` (int32)
- Offset 308: `scores[]` (6 × int32 = 24 bytes)
- Offset 332: `pvs[]` (6 × 256 = 1536 bytes)
- Offset 1868: `bests[]` (6 × Move)

## Coding Conventions

- C11 standard, no C++ features
- All search state is `_Thread_local` (for Lazy SMP)
- Mutable NNUE state is per-thread via `NnueAccum *` in Board struct
- WASM build uses static (no TLS, single-threaded)
- Comments in English, user-facing strings may be in Portuguese
- Piece encoding: COL_W=8, COL_B=16, type 1-6 (P,N,B,R,Q,K)
- Square encoding: a8=0, h1=63 (rank 0 = rank 8, file 0 = file a)
- Test scripts: engine path configured in tournament.py config block

## Key Constants

| Constant | Value | Location |
|----------|-------|----------|
| TT_SIZE (native) | 4,194,304 | search.h |
| TT_SIZE (WASM) | 524,288 | search.h |
| MAX_PLY | 64 | search.h |
| MAX_MOVES | 256 | board.h |
| MAX_MULTI_PV | 6 | search.h |
| QA (NNUE L1 scale) | 255 | nnue.c |
| QB (NNUE L2/L3 scale) | 64 | nnue.c |
| Aspiration delta | 20 cp | search.c |
| RFP margin | depth×90 | search.c |
| NMP R | 3+depth/3, cap 6 | search.c |
| LMR divisor | 1.5 | search.c |
