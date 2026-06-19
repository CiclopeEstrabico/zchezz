@echo off
call C:\emsdk\emsdk_env.bat >nul 2>&1
emcc -O3 -std=c11 -msimd128 -DNO_TABLEBASES -DNO_BOOK ^
  -s MODULARIZE=1 -s EXPORT_NAME=ZchezzEngine ^
  -s "EXPORTED_FUNCTIONS=[\"_board_apply_moves\",\"_board_init\",\"_search_init\",\"_board_load_fen\",\"_search_best_sret\",\"_nnue_load_from_mem\",\"_nnue_reset\",\"_nnue_ready\",\"_malloc\",\"_free\"]" ^
  -s "EXPORTED_RUNTIME_METHODS=[\"ccall\",\"cwrap\",\"HEAPU8\"]" ^
  -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=268435456 ^
  -s MAXIMUM_MEMORY=536870912 -s STACK_SIZE=4194304 ^
  -s "ENVIRONMENT=web,worker" -s NO_EXIT_RUNTIME=1 ^
  -Wno-unused-variable -Wno-unused-but-set-variable -Wno-uninitialized ^
  -Wno-misleading-indentation -Wno-sign-compare -Wno-unused-function -Wno-parentheses ^
  -o zchezz_wasm.js main.c board.c search.c nnue.c syzygy.c book.c
if errorlevel 1 (
  echo WASM build FAILED
  exit /b 1
)
echo WASM build OK
echo Rebuilding bundle...
set PYTHONIOENCODING=utf-8
python bundle.py zchezz_wasm.html zchezz_wasm.js zchezz_wasm.wasm nnue_weights.bin
if errorlevel 1 (
  echo Bundle build FAILED
  exit /b 1
)
echo Bundle build OK
echo Deploying to GitHub Pages...
copy /Y zchezz_bundle.html ..\..\..\docs\index.html >nul
echo docs/index.html updated
