# engine/build/Makefile — shared build system for Zchezz engine versions.

ENGINE     ?= v403
ARCH_FLAGS ?= -mavxvnni -mavx2
STATIC_FLAG ?= -static

# GNU Make defines CC=cc as a built-in default. Override only that built-in
# value. Environment and command-line selections such as CC=clang win.
ifeq ($(origin CC),default)
CC = gcc
endif

EDIR    = ../c/zchezz_$(ENGINE)
TOOLDIR = ../c/tools

CFLAGS  = -O3 -ffast-math -D_GNU_SOURCE -std=c11 $(ARCH_FLAGS) -I$(EDIR)
WARNS   = -Wno-unused-variable -Wno-unused-but-set-variable -Wno-maybe-uninitialized \
          -Wno-misleading-indentation -Wno-sign-compare -Wno-unused-function -Wno-parentheses
EMWARNS = -Wno-unused-variable -Wno-unused-but-set-variable -Wno-uninitialized \
          -Wno-misleading-indentation -Wno-sign-compare -Wno-unused-function -Wno-parentheses
LDFLAGS = $(STATIC_FLAG) -lm -pthread

CORE_SRCS = $(EDIR)/main.c $(EDIR)/board.c $(EDIR)/search.c $(EDIR)/nnue.c $(EDIR)/syzygy.c $(EDIR)/book.c
TB_C      = $(EDIR)/tbprobe.c
TB_H      = $(EDIR)/tbprobe.h

ifneq ($(and $(wildcard $(TB_C)),$(wildcard $(TB_H))),)
HAVE_FATHOM = 1
NATIVE_SRCS = $(CORE_SRCS) $(TB_C)
TB_DEFS =
TB_STATUS = enabled
else
HAVE_FATHOM = 0
NATIVE_SRCS = $(CORE_SRCS)
TB_DEFS = -DNO_TABLEBASES
TB_STATUS = disabled
endif

TARGET = $(EDIR)/zchezz.exe

EMCC = emcc
EMFLAGS = -O3 -std=c11 -msimd128 -DNO_TABLEBASES -DNO_BOOK \
          -s MODULARIZE=1 -s EXPORT_NAME=ZchezzEngine \
          -s 'EXPORTED_FUNCTIONS=["_board_apply_moves","_board_init","_search_init","_board_load_fen","_search_best_sret","_nnue_load_from_mem","_nnue_reset_global","_nnue_ready","_malloc","_free"]' \
          -s 'EXPORTED_RUNTIME_METHODS=["ccall","cwrap","HEAPU8"]' \
          -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=268435456 \
          -s MAXIMUM_MEMORY=536870912 -s STACK_SIZE=4194304 \
          -s 'ENVIRONMENT=web,worker' -s NO_EXIT_RUNTIME=1
WASM_SRCS = $(CORE_SRCS)
WASM_TEMPLATE = zchezz_wasm.html

ifeq ($(OS),Windows_NT)
PYTHON = python
RUN_LOCAL = .\\
else
PYTHON = python3
RUN_LOCAL = ./
endif

.PHONY: native debug sanitize test-c clean wasm bundle selfplay arena ga_tune \
        build-info require-tablebases

build-info:
	@echo ENGINE=$(ENGINE)
	@echo CC=$(CC)
	@echo HAVE_FATHOM=$(HAVE_FATHOM)
	@echo Tablebases: $(TB_STATUS)

ifeq ($(HAVE_FATHOM),1)
require-tablebases:
	@echo Fathom sources available for $(ENGINE).
	@echo $(TB_C)
	@echo $(TB_H)
else
require-tablebases:
	$(error Fathom sources unavailable for $(ENGINE). Required: $(TB_C) and $(TB_H))
endif

native: $(TARGET)

$(TARGET): $(NATIVE_SRCS)
	@echo Building $(ENGINE); tablebases $(TB_STATUS).
	$(CC) $(CFLAGS) $(TB_DEFS) $(WARNS) -o $@ $^ $(LDFLAGS)

debug: $(NATIVE_SRCS)
	$(CC) -O0 -g3 -D_GNU_SOURCE -std=c11 -I$(EDIR) $(TB_DEFS) \
		-Wall -Wextra -Wpedantic -Wshadow \
		-o $(EDIR)/zchezz_debug.exe $(NATIVE_SRCS) -lm -pthread

sanitize: $(NATIVE_SRCS)
	$(CC) -O1 -g3 -fno-omit-frame-pointer -fsanitize=address,undefined \
		-D_GNU_SOURCE -std=c11 -I$(EDIR) $(TB_DEFS) \
		-o $(EDIR)/zchezz_sanitize.exe $(NATIVE_SRCS) \
		-lm -pthread -fsanitize=address,undefined

test-c: ../c/tests/test_engine_invariants.c $(EDIR)/board.c $(EDIR)/nnue.c
	$(CC) -O2 -g -D_GNU_SOURCE -std=c11 -I$(EDIR) \
		-o test_engine_invariants.exe \
		../c/tests/test_engine_invariants.c $(EDIR)/board.c $(EDIR)/nnue.c -lm
	$(RUN_LOCAL)test_engine_invariants.exe $(EDIR)/nnue_weights.bin

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
	$(EMCC) $(EMFLAGS) $(EMWARNS) -I$(EDIR) -o $(EDIR)/zchezz_wasm.js $(WASM_SRCS)

bundle: wasm $(WASM_TEMPLATE)
	$(PYTHON) bundle_shared.py --template $(WASM_TEMPLATE) --engine-dir $(EDIR)

clean:
	$(PYTHON) clean_generated.py --engine-dir $(EDIR)
