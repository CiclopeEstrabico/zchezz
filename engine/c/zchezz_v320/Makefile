# Makefile for Zchezz v304
# Build: make native
# WASM:  make wasm    (requires Emscripten)
# Bundle: make bundle  (WASM + single HTML)

CC      = gcc
CFLAGS  = -O3 -ffast-math -D_GNU_SOURCE -std=c11 -mavxvnni -mavx2 -I.
WARNS   = -Wno-unused-variable -Wno-unused-but-set-variable -Wno-maybe-uninitialized \
          -Wno-misleading-indentation -Wno-sign-compare -Wno-unused-function -Wno-parentheses
LDFLAGS = -static -lm -pthread

SRCS    = main.c board.c search.c nnue.c syzygy.c tbprobe.c book.c
TARGET  = zchezz.exe

# WASM build (no tablebases, no book file I/O)
EMCC     = emcc
EMFLAGS  = -O3 -std=c11 -msimd128 -DNO_TABLEBASES -DNO_BOOK \
           -s MODULARIZE=1 -s EXPORT_NAME=ZchezzEngine \
           -s 'EXPORTED_FUNCTIONS=["_board_apply_moves","_board_init","_search_init","_board_load_fen","_search_best_sret","_nnue_load_from_mem","_nnue_reset","_nnue_ready","_malloc","_free"]' \
           -s 'EXPORTED_RUNTIME_METHODS=["ccall","cwrap","HEAPU8"]' \
           -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=268435456 \
           -s MAXIMUM_MEMORY=536870912 -s STACK_SIZE=4194304 \
           -s 'ENVIRONMENT=web,worker' -s NO_EXIT_RUNTIME=1
WASM_SRCS = main.c board.c search.c nnue.c syzygy.c book.c

ifeq ($(OS),Windows_NT)
    PYTHON = python
else
    PYTHON = python3
endif

.PHONY: native clean wasm bundle

native: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(WARNS) -o $@ $^ $(LDFLAGS)

wasm: $(WASM_SRCS)
	$(EMCC) $(EMFLAGS) $(WARNS) -o zchezz_wasm.js $(WASM_SRCS)

bundle: wasm
	$(PYTHON) bundle.py zchezz_wasm.html zchezz_wasm.js zchezz_wasm.wasm nnue_weights.bin

clean:
	rm -f $(TARGET) *.o zchezz_wasm.js zchezz_wasm.wasm zchezz_bundle.html
