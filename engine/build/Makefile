# Shared build system for all Zchezz engine versions.

ENGINE     ?= v403
CC         ?= gcc
ARCH_FLAGS ?= -mavxvnni -mavx2

EDIR    = ../c/zchezz_$(ENGINE)
TOOLDIR = ../c/tools

CFLAGS  = -O3 -ffast-math -D_GNU_SOURCE -std=c11 $(ARCH_FLAGS) -I$(EDIR)
WARNS   = -Wno-unused-variable -Wno-unused-but-set-variable -Wno-maybe-uninitialized \
          -Wno-misleading-indentation -Wno-sign-compare -Wno-unused-function -Wno-parentheses
LDFLAGS = -static -lm -pthread

CORE_SRCS = $(EDIR)/main.c $(EDIR)/board.c $(EDIR)/search.c $(EDIR)/nnue.c $(EDIR)/syzygy.c $(EDIR)/book.c
TB_C      = $(EDIR)/tbprobe.c
TB_H      = $(EDIR)/tbprobe.h

ifneq ($(and $(wildcard $(TB_C)),$(wildcard $(TB_H))),)
    HAVE_FATHOM = 1
    NATIVE_SRCS = $(CORE_SRCS) $(TB_C)
    TB_DEFS =
else
    HAVE_FATHOM = 0
    NATIVE_SRCS = $(CORE_SRCS)
    TB_DEFS = -DNO_TABLEBASES
endif

TARGET = $(EDIR)/zchezz.exe

EMCC     = emcc
EMFLAGS  = -O3 -std=c11 -msimd128 -DNO_TABLEBASES -DNO_BOOK \
           -s MODULARIZE=1 -s EXPORT_NAME=ZchezzEngine \
           -s 'EXPORTED_FUNCTIONS=["_board_apply_moves","_board_init","_search_init","_board_load_fen","_search_best_sret","_nnue_load_from_mem","_nnue_reset","_nnue_ready","_malloc","_free"]' \
           -s 'EXPORTED_RUNTIME_METHODS=["ccall","cwrap","HEAPU8"]' \
           -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=268435456 \
           -s MAXIMUM_MEMORY=536870912 -s STACK_SIZE=4194304 \
           -s 'ENVIRONMENT=web,worker' -s NO_EXIT_RUNTIME=1
WASM_SRCS = $(CORE_SRCS)

ifeq ($(OS),Windows_NT)
    PYTHON = python
else
    PYTHON = python3
endif

.PHONY: native debug sanitize test-c clean wasm bundle selfplay arena ga_tune build-info require-tablebases

build-info:
	@echo "ENGINE=$(ENGINE)"
	@echo "CC=$(CC)"
	@echo "HAVE_FATHOM=$(HAVE_FATHOM)"
	@if [ "$(HAVE_FATHOM)" = "0" ]; then echo "Tablebases: disabled"; else echo "Tablebases: enabled"; fi

require-tablebases:
	@if [ "$(HAVE_FATHOM)" != "1" ]; then \
		echo "ERROR: Fathom sources unavailable. Required: $(TB_C) and $(TB_H)"; \
		exit 1; \
	fi

native: $(TARGET)

$(TARGET): $(NATIVE_SRCS)
	@if [ "$(HAVE_FATHOM)" = "0" ]; then echo "NOTE: building with NO_TABLEBASES"; fi
	$(CC) $(CFLAGS) $(TB_DEFS) $(WARNS) -o $@ $^ $(LDFLAGS)

debug: $(NATIVE_SRCS)
	$(CC) -O0 -g3 -D_GNU_SOURCE -std=c11 -I$(EDIR) $(TB_DEFS) -Wall -Wextra -Wpedantic -Wshadow \
		-o $(EDIR)/zchezz_debug.exe $(NATIVE_SRCS) -lm -pthread

sanitize: $(NATIVE_SRCS)
	$(CC) -O1 -g3 -fno-omit-frame-pointer -fsanitize=address,undefined -D_GNU_SOURCE -std=c11 -I$(EDIR) $(TB_DEFS) \
		-o $(EDIR)/zchezz_sanitize.exe $(NATIVE_SRCS) -lm -pthread -fsanitize=address,undefined

test-c: ../c/tests/test_engine_invariants.c $(EDIR)/board.c $(EDIR)/nnue.c
	$(CC) -O2 -g -D_GNU_SOURCE -std=c11 -I$(EDIR) -o test_engine_invariants.exe \
		../c/tests/test_engine_invariants.c $(EDIR)/board.c $(EDIR)/nnue.c -lm
	./test_engine_invariants.exe $(EDIR)/nnue_weights.bin

SELFPLAY_SRCS = $(TOOLDIR)/selfplay.c $(TOOLDIR)/opening_pool.c $(EDIR)/board.c $(EDIR)/search.c $(EDIR)/nnue.c
selfplay: $(SELFPLAY_SRCS)
	$(CC) $(CFLAGS) $(WARNS) -DNO_TABLEBASES -DNO_BOOK -o selfplay.exe $(SELFPLAY_SRCS) $(LDFLAGS)

ARENA_SRCS = $(TOOLDIR)/arena.c $(TOOLDIR)/opening_pool.c $(EDIR)/board.c $(EDIR)/search.c $(EDIR)/nnue.c
arena: $(ARENA_SRCS)
	$(CC) $(CFLAGS) $(WARNS) -DNO_TABLEBASES -DNO_BOOK -o arena.exe $(ARENA_SRCS) $(LDFLAGS)

GA_TUNE_SRCS = $(TOOLDIR)/ga_tune.c $(TOOLDIR)/arena.c $(TOOLDIR)/opening_pool.c $(EDIR)/board.c $(EDIR)/search.c $(EDIR)/nnue.c
ga_tune: $(GA_TUNE_SRCS)
	$(CC) $(CFLAGS) $(WARNS) -DNO_TABLEBASES -DNO_BOOK -DARENA_NO_MAIN -o ga_tune.exe $(GA_TUNE_SRCS) $(LDFLAGS)

wasm: $(WASM_SRCS)
	$(EMCC) $(EMFLAGS) $(WARNS) -I$(EDIR) -o $(EDIR)/zchezz_wasm.js $(WASM_SRCS)

bundle: wasm
	$(PYTHON) bundle.py $(EDIR)/zchezz_wasm.html $(EDIR)/zchezz_wasm.js $(EDIR)/zchezz_wasm.wasm $(EDIR)/nnue_weights.bin

clean:
	rm -f $(TARGET) $(EDIR)/zchezz_debug.exe $(EDIR)/zchezz_sanitize.exe \
		test_engine_invariants.exe arena.exe selfplay.exe ga_tune.exe *.o \
		$(EDIR)/zchezz_wasm.js $(EDIR)/zchezz_wasm.wasm $(EDIR)/zchezz_bundle.html
