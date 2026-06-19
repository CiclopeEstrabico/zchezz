# Zchezz — Project Guide

## Overview

Zchezz is a chess engine written in C with NNUE evaluation, targeting both native platforms (Windows/Linux/Android) and WebAssembly for browser play. The engine communicates via the UCI protocol.

## Project Structure

```
Zchezz/
├── engine/
│   └── c/zchezz_v305/        # Current engine (v3.05 — Lazy SMP, opening book, TB)
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
├── openings/                  # Opening books (PGN format)
│   └── Blitz_Testing_4moves.pgn  # 13K+ openings for testing
│
├── tests/                     # Test & match scripts (version-agnostic)
│   ├── uci_test.py            # UCI protocol compliance tests
│   ├── browser_test.py        # Browser/WASM interaction tests
│   ├── quick_match.py         # Generic A-vs-B match (PGN openings, paired colors)
│   ├── concurrent_match.py    # Parallel multi-worker match runner
│   ├── tournament_elo.py      # ELO estimation vs Stockfish anchors
│   ├── tournament_complete.py # Full tournament with multiple anchors
│   ├── validate_book.py       # Opening book legality and quality validation
│   ├── test_html_features.py  # HTML feature validation (no browser needed)
│   ├── suite_runner.py        # Test suite runner (WAC, STS, etc.)
│   ├── suite_compare.py       # Compare suite results between versions
│   ├── selfplay.py            # Self-play data generation for NNUE training
│   └── test_tournament.py     # Round-robin tournament runner
│
├── train/                     # NNUE training code (PyTorch)
│   ├── mixtrain.py            # Main training script (QAT, NNU3)
│   └── convert_nnue.py        # PyTorch → NNU3 binary converter
│
├── sf_analyze/                # Stockfish analysis scripts (data generation)
├── utils/                     # Utility files
│   ├── kill_ghosts.py         # Kill orphaned engine processes
│   └── OpeningBook.bin        # Binary opening book (Polyglot format)
│
├── test_suites/               # EPD test suites (WAC, STS, etc.)
├── data/                      # Training data (gitignored)
├── checkpoints/               # Training checkpoints (gitignored)
├── tablebases/                # Syzygy tables (gitignored)
│
├── Readme.md                  # Full project documentation
├── CLAUDE.md                  # This file
├── FolderStructure.md         # Detailed folder structure
└── .gitignore
```

## Build Instructions

### Native (Windows/Linux)

```bash
cd engine/c/zchezz_v305
mingw32-make native    # Windows
make native            # Linux
./zchezz --nnue nnue_weights.bin
```

Or manually:
```bash
gcc -O3 -ffast-math -D_GNU_SOURCE -std=c11 -mavxvnni -mavx2 \
    -I. -Wno-unused-variable -Wno-unused-but-set-variable \
    -Wno-maybe-uninitialized -Wno-misleading-indentation \
    -Wno-sign-compare -Wno-unused-function -Wno-parentheses \
    -o zchezz.exe main.c board.c search.c nnue.c syzygy.c tbprobe.c book.c \
    -static -lm -pthread
```

Note: v305 includes `syzygy.c`, `tbprobe.c`, and `book.c` in the build. For WASM, add `-DNO_TABLEBASES -DNO_BOOK` and omit those files.

### WebAssembly (Emscripten)

```bash
cd engine/c/zchezz_v305
build_wasm.bat         # Windows (calls emcc + bundle.py)
# Or:
make wasm              # Produces zchezz_wasm.js + zchezz_wasm.wasm
make bundle            # Produces zchezz_bundle.html (self-contained)
```

### Android/Termux

```bash
cd engine/c/zchezz_v305
make termux
```

## Testing

All test scripts auto-detect the latest engine version. No hardcoded paths.

### Quick Match (version comparison)

```bash
python tests/quick_match.py ENGINE_A.exe ENGINE_B.exe NameA NameB 200 openings/Blitz_Testing_4moves.pgn
```

### ELO Estimation (vs Stockfish anchors)

```bash
python tests/tournament_elo.py
python tests/tournament_complete.py
```

### HTML Feature Validation

```bash
python tests/test_html_features.py
```

### Opening Book Validation

```bash
python tests/validate_book.py
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
- Lazy SMP: threads share TT, helpers start at staggered depths
  - Main thread: full ID from depth 1 with aspiration + info output
  - Helper 0: starts at depth 2
  - Helper 1: starts at depth 3
  - Helper 2+: starts at depth 2*i+1
  - TT_GEN only incremented by main thread (no race condition)

### NNUE (`nnue.c`)
- 799→256→64→1 architecture (HalfKP + 31 extra features)
- int16/int8 quantized inference with AVX2 SIMD + WASM SIMD
- Incremental accumulator updated in board_make/board_unmake via nnue_push/nnue_pop
- Accumulator stack depth 512

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
- **Mobile**: Responsive layout with toolbar above PV lines

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
- WASM build uses static (no TLS, single-threaded)
- Comments in English, user-facing strings may be in Portuguese
- Piece encoding: COL_W=16, COL_B=24, type 1-6 (P,N,B,R,Q,K)
- Square encoding: a8=0, h1=63 (rank 0 = rank 8, file 0 = file a)
- Test scripts auto-detect the latest engine version (no hardcoded paths)

## Version History

| Version | Changes |
|---------|---------|
| v3.05 | **Current** — Lazy SMP fix, opening book, TB root probing, SVG piece sets |

## Syzygy Tablebases

- Tables: 3-4-5 piece (290 files, ~938 MB)
- Location: `tablebases/` (gitignored)
- Download: `http://tablebase.sesse.net/syzygy/3-4-5/`
- Integration: Fathom library (jdart1 fork)
- Square mapping: Zchezz a8=0 ↔ Fathom a1=0, convert via `sq ^ 56`
- Bitboard mapping: vertical flip via `__builtin_bswap64()`
- Compile without TB: `-DNO_TABLEBASES` (stubs everything out)

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
