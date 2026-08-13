@echo off
REM ═══════════════════════ CONFIGURATION ═══════════════════════
REM ENGINE selects engine/c/zchezz_%ENGINE%/ — pass as first arg to override.
set ENGINE=v400
REM ═══════════════════════════════════════════════════════════
if not "%~1"=="" set ENGINE=%~1
cd /d "%~dp0"
set EDIR=..\c\zchezz_%ENGINE%

call C:\emsdk\emsdk_env.bat >nul 2>&1
emcc -O3 -std=c11 -msimd128 -DNO_TABLEBASES -DNO_BOOK ^
  -I%EDIR% ^
  -s MODULARIZE=1 -s EXPORT_NAME=ZchezzEngine ^
  -s "EXPORTED_FUNCTIONS=[\"_board_apply_moves\",\"_board_init\",\"_search_init\",\"_board_load_fen\",\"_search_best_sret\",\"_nnue_load_from_mem\",\"_nnue_reset_global\",\"_nnue_ready\",\"_malloc\",\"_free\"]" ^
  -s "EXPORTED_RUNTIME_METHODS=[\"ccall\",\"cwrap\",\"HEAPU8\"]" ^
  -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=268435456 ^
  -s MAXIMUM_MEMORY=536870912 -s STACK_SIZE=4194304 ^
  -s "ENVIRONMENT=web,worker" -s NO_EXIT_RUNTIME=1 ^
  -Wno-unused-variable -Wno-unused-but-set-variable -Wno-uninitialized ^
  -Wno-misleading-indentation -Wno-sign-compare -Wno-unused-function -Wno-parentheses ^
  -o %EDIR%\zchezz_wasm.js %EDIR%\main.c %EDIR%\board.c %EDIR%\search.c %EDIR%\nnue.c %EDIR%\syzygy.c %EDIR%\book.c
if errorlevel 1 (
  echo WASM build FAILED
  exit /b 1
)
echo WASM build OK
echo Rebuilding bundle...
set PYTHONIOENCODING=utf-8
python bundle.py %EDIR%\zchezz_wasm.html %EDIR%\zchezz_wasm.js %EDIR%\zchezz_wasm.wasm %EDIR%\nnue_weights.bin
if errorlevel 1 (
  echo Bundle build FAILED
  exit /b 1
)
echo Bundle build OK
echo Deploying to GitHub Pages...
copy /Y %EDIR%\zchezz_bundle.html ..\..\index.html >nul
echo index.html updated
