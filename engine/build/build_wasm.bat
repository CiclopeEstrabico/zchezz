@echo off
setlocal EnableExtensions

set "ENGINE=%~1"
if not defined ENGINE set /p ENGINE=<"%~dp0..\ACTIVE_ENGINE"
if not defined ENGINE (
    echo ERROR: could not resolve active engine from engine\ACTIVE_ENGINE.
    exit /b 1
)

cd /d "%~dp0\..\.."

for /f "delims=" %%B in ('python -c "import sys; sys.path.insert(0, 'utils'); from repo_paths import previous_version; print(previous_version('%ENGINE%'))"') do set "BASELINE=%%B"
if not defined BASELINE (
    echo ERROR: could not resolve baseline for %ENGINE%.
    exit /b 1
)

echo [1/5] Ensuring shared WASM template...
python tools\promote_wasm_template.py
if errorlevel 1 exit /b 1

echo [2/5] Activating Emscripten if needed...
where emcc >nul 2>nul
if errorlevel 1 if defined ZCHEZZ_EMSDK if exist "%ZCHEZZ_EMSDK%\emsdk_env.bat" call "%ZCHEZZ_EMSDK%\emsdk_env.bat" >nul
where emcc >nul 2>nul
if errorlevel 1 if exist "%USERPROFILE%\emsdk\emsdk_env.bat" call "%USERPROFILE%\emsdk\emsdk_env.bat" >nul
where emcc >nul 2>nul
if errorlevel 1 if exist "C:\emsdk\emsdk_env.bat" call "C:\emsdk\emsdk_env.bat" >nul
where emcc >nul 2>nul
if errorlevel 1 (
    echo ERROR: emcc not found.
    echo Run engine\build\setup_web_windows.bat once, then retry.
    exit /b 1
)

echo [3/5] Checking Python web dependencies...
python -c "import playwright, chess" >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python web dependencies are missing.
    echo Run: python -m pip install -e ".[all]"
    exit /b 1
)

echo [4/5] Running canonical web profile for %ENGINE% against %BASELINE%...
python tests\run_tests.py web --version %ENGINE% --baseline %BASELINE% --keep-going
if errorlevel 1 exit /b 1

echo [5/5] Promoting tested bundle to index.html...
set "BUNDLE=engine\c\zchezz_%ENGINE%\zchezz_bundle.html"
if not exist "%BUNDLE%" (
    echo ERROR: expected bundle not found: %BUNDLE%
    exit /b 1
)
copy /Y "%BUNDLE%" index.html >nul
python -c "from pathlib import Path; import sys; sys.path.insert(0, 'engine/build'); from bundle import parse_version; expected=parse_version('zchezz_%ENGINE%'); text=Path('index.html').read_text(encoding='utf-8'); assert f'Zchezz NNUE {expected}' in text, f'index.html version mismatch: expected {expected}'"
if errorlevel 1 exit /b 1

echo PASS: index.html now publishes %ENGINE%.
exit /b 0
