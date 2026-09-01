@echo off
setlocal EnableExtensions

set "ENGINE=v403"
if not "%~1"=="" set "ENGINE=%~1"

cd /d "%~dp0\..\.."

echo [1/4] Ensuring shared WASM template...
python tools\promote_wasm_template.py
if errorlevel 1 exit /b 1

echo [2/4] Activating Emscripten if needed...
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

echo [3/4] Checking Python web dependencies...
python -c "import playwright, chess" >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python web dependencies are missing.
    echo Run: python -m pip install -e ".[all]"
    exit /b 1
)

echo [4/4] Running canonical web profile...
python tests\run_tests.py web --version %ENGINE% --baseline v402 --keep-going
exit /b %errorlevel%
