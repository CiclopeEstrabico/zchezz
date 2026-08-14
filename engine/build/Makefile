# engine/build/Makefile — shared build system for all Zchezz engine versions.
#
# Build files and tool sources are shared across engine versions; only the
# per-version engine core (engine/c/zchezz_vXXX/*.c *.h) is duplicated per
# release, per CLAUDE.md rule 2 (Versioning).
#
# IMPORTANT — engine/c/tools/ tracks the CURRENT engine API only.
# selfplay.c / arena.c / arena.h use TTable, NnueNet, SearchParams fields
# etc. that only exist in the newest engine folder. They will NOT compile
# against an older version (e.g. zchezz_v314) — that is by design, not a
# bug. arena.exe never links an old version's .c files; it compares older
# versions through its `uci:` external-process player, driving an
# already-built zchezz.exe for that version instead. See
# README.md ("engine/c/tools/" section) in the repo root.
#
# Usage (ENGINE selects which engine/c/zchezz_vXXX/ folder to build from;
# defaults to the newest release below):
#   mingw32-make ENGINE=v400 native      # -> ../c/zchezz_v400/zchezz.exe
#   mingw32-make ENGINE=v400 arena       # -> ./arena.exe    (this dir)
#   mingw32-make ENGINE=v400 selfplay    # -> ./selfplay.exe (this dir)
#   mingw32-make ENGINE=v400 wasm        # -> ../c/zchezz_v400/zchezz_wasm.{js,wasm}
#   mingw32-make ENGINE=v400 bundle      # -> ../c/zchezz_v400/zchezz_bundle.html
#
# `make` (Linux) or `mingw32-make` (this project's Windows dev machines,
# which ship mingw32-make rather than a `make` alias) both work — the
# Makefile itself is unchanged either way.

# ═══════════════════════════════════════════════ CONFIGURATION ═══
ENGINE  ?= v401        # engine/c/zchezz_$(ENGINE)/ — default = newest version
# ════════════════════════════════════════════════════════════════

EDIR    = ../c/zchezz_$(ENGINE)
TOOLDIR = ../c/tools

CC      = gcc
CFLAGS  = -O3 -ffast-math -D_GNU_SOURCE -std=c11 -mavxvnni -mavx2 -I$(EDIR)
WARNS   = -Wno-unused-variable -Wno-unused-but-set-variable -Wno-maybe-uninitialized \
          -Wno-misleading-indentation -Wno-sign-compare -Wno-unused-function -Wno-parentheses
LDFLAGS = -static -lm -pthread

SRCS    = $(EDIR)/main.c $(EDIR)/board.c $(EDIR)/search.c $(EDIR)/nnue.c $(EDIR)/syzygy.c $(EDIR)/tbprobe.c $(EDIR)/book.c
TARGET  = $(EDIR)/zchezz.exe

# WASM build (no tablebases, no book file I/O)
EMCC     = emcc
EMFLAGS  = -O3 -std=c11 -msimd128 -DNO_TABLEBASES -DNO_BOOK \
           -s MODULARIZE=1 -s EXPORT_NAME=ZchezzEngine \
           -s 'EXPORTED_FUNCTIONS=["_board_apply_moves","_board_init","_search_init","_board_load_fen","_search_best_sret","_nnue_load_from_mem","_nnue_reset","_nnue_ready","_malloc","_free"]' \
           -s 'EXPORTED_RUNTIME_METHODS=["ccall","cwrap","HEAPU8"]' \
           -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=268435456 \
           -s MAXIMUM_MEMORY=536870912 -s STACK_SIZE=4194304 \
           -s 'ENVIRONMENT=web,worker' -s NO_EXIT_RUNTIME=1
WASM_SRCS = $(EDIR)/main.c $(EDIR)/board.c $(EDIR)/search.c $(EDIR)/nnue.c $(EDIR)/syzygy.c $(EDIR)/book.c

ifeq ($(OS),Windows_NT)
    PYTHON = python
else
    PYTHON = python3
endif

.PHONY: native clean wasm bundle selfplay arena ga_tune

native: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(WARNS) -o $@ $^ $(LDFLAGS)

# Native in-process self-play data generator (Appendix F.2/F.3 of
# docs/v400_implementation_plan.md). Links board.c/search.c/nnue.c but
# NOT main.c (which owns main()) and NOT syzygy.c/tbprobe.c/book.c —
# -DNO_TABLEBASES/-DNO_BOOK stub those calls out entirely (see
# ../c/tools/selfplay.c's header comment). Output: selfplay.exe, built
# in THIS directory (engine/build/), not the version folder — it is a
# shared tool binary, not a per-version artifact.
SELFPLAY_SRCS = $(TOOLDIR)/selfplay.c $(TOOLDIR)/opening_pool.c $(EDIR)/board.c $(EDIR)/search.c $(EDIR)/nnue.c
selfplay: $(SELFPLAY_SRCS)
	$(CC) $(CFLAGS) $(WARNS) -DNO_TABLEBASES -DNO_BOOK -o selfplay.exe $(SELFPLAY_SRCS) $(LDFLAGS)

# Native in-process A/B arena (implementation plan Appendix F.4), the
# SPRT gate for Appendix F.5's bootstrap loop. Links board.c/search.c/
# nnue.c but NOT main.c/syzygy.c/tbprobe.c/book.c — same
# -DNO_TABLEBASES/-DNO_BOOK rationale as `selfplay` above (see
# ../c/tools/arena.c's header comment). Output: arena.exe, built in THIS
# directory (engine/build/) — shared tool binary, not per-version.
ARENA_SRCS = $(TOOLDIR)/arena.c $(TOOLDIR)/opening_pool.c $(EDIR)/board.c $(EDIR)/search.c $(EDIR)/nnue.c
arena: $(ARENA_SRCS)
	$(CC) $(CFLAGS) $(WARNS) -DNO_TABLEBASES -DNO_BOOK -o arena.exe $(ARENA_SRCS) $(LDFLAGS)

# Native genetic-algorithm tuner for search.c's pruning/reduction constants
# (implementation plan Appendix F.4's "SPSA tuner", built as a GA instead —
# see ../c/tools/ga_tune.c's header comment). Links arena.c compiled with
# -DARENA_NO_MAIN (reused as a library ONLY for board_init/search_init,
# OpeningPool/load_openings, and elo_diff_ci() — see that header for what
# is and is not reused from arena.c) plus board.c/search.c/nnue.c, same
# -DNO_TABLEBASES/-DNO_BOOK rationale as `arena`/`selfplay` above. Output:
# ga_tune.exe, built in THIS directory — shared tool binary, not per-version.
GA_TUNE_SRCS = $(TOOLDIR)/ga_tune.c $(TOOLDIR)/arena.c $(TOOLDIR)/opening_pool.c $(EDIR)/board.c $(EDIR)/search.c $(EDIR)/nnue.c
ga_tune: $(GA_TUNE_SRCS)
	$(CC) $(CFLAGS) $(WARNS) -DNO_TABLEBASES -DNO_BOOK -DARENA_NO_MAIN -o ga_tune.exe $(GA_TUNE_SRCS) $(LDFLAGS)

wasm: $(WASM_SRCS)
	$(EMCC) $(EMFLAGS) $(WARNS) -I$(EDIR) -o $(EDIR)/zchezz_wasm.js $(WASM_SRCS)

bundle: wasm
	$(PYTHON) bundle.py $(EDIR)/zchezz_wasm.html $(EDIR)/zchezz_wasm.js $(EDIR)/zchezz_wasm.wasm $(EDIR)/nnue_weights.bin

clean:
	rm -f $(TARGET) arena.exe selfplay.exe ga_tune.exe *.o $(EDIR)/zchezz_wasm.js $(EDIR)/zchezz_wasm.wasm $(EDIR)/zchezz_bundle.html
