# Architecture

Zchezz is a C11 UCI chess engine with NNUE evaluation, native Lazy SMP search, Syzygy integration, Polyglot opening-book support, and a WebAssembly/browser target.

## Engine core

Versioned source currently lives under `engine/c/zchezz_vXXX/`. This layout is intentionally retained during the professionalization work. History migration to a single active source tree plus Git tags is a separate decision after the safety net is stable.

The main subsystems are:

- `board.c/.h` — board representation, move generation, make/unmake, Zobrist state, draw detection;
- `search.c/.h` — search, transposition table, pruning/reductions, Lazy SMP;
- `nnue.c/.h` — HalfKP-4Bucket NNU4 evaluation and incremental accumulators;
- `syzygy.c` and `tbprobe.c` — tablebase integration;
- `book.c` — Polyglot book support;
- `main.c` — UCI process and native entry point.

## Board representation

The board uses a mailbox plus twelve 64-bit bitboards. Cached white, black, and total occupancies must equal the bitboard unions. The Zobrist key is maintained incrementally and must equal a full recomputation after every move.

## NNUE ownership

Mutable NNUE accumulator state is per search thread through `Board::nnue`. Weight matrices are read-only after loading. The concat order is `[stm, opp]`. A king move that crosses a perspective bucket invalidates that perspective and triggers lazy rebuild at evaluation time.

## Search ownership

Lazy SMP helpers share the main transposition table intentionally. Per-search mutable state is thread-local/per-instance as documented in `AGENTS.md` and `search.h`.

## Browser target

The browser target is WebAssembly, single-threaded, with no filesystem tablebase/book dependency. `engine/build/bundle.py` produces the standalone HTML artifact.

