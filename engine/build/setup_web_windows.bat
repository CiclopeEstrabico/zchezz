@echo off
setlocal EnableExtensions
cd /d "%~dp0\..\.."

echo === Zchezz Web/WASM one-time setup ===

echo [1/6] Promoting shared template
python tools\promote_wasm_template.py
if errorlevel 1 exit /b 1

echo [2/6] Emscripten SDK
where emcc >nul 2>nul
if errorlevel 1 (
    if not exist "%USERPROFILE%\emsdk\emsdk.bat" (
        where git >nul 2>nul
        if errorlevel 1 (
            echo ERROR: Git is required to install the official emsdk.
            exit /b 1
        )
        echo Cloning official emsdk into %USERPROFILE%\emsdk ...
        git clone https://github.com/emscripten-core/emsdk.git "%USERPROFILE%\emsdk"
        if errorlevel 1 exit /b 1
    ) else (
        echo Existing emsdk checkout found at %USERPROFILE%\emsdk
    )

    pushd "%USERPROFILE%\emsdk"

    REM For a Git-cloned emsdk, update with git pull rather than `emsdk update`.
    git pull --ff-only
    if errorlevel 1 (
        echo WARN: git pull failed. Continuing with the existing emsdk checkout.
    )

    call emsdk.bat install latest
    if errorlevel 1 (popd & exit /b 1)

    call emsdk.bat activate latest
    if errorlevel 1 (popd & exit /b 1)

    call emsdk_env.bat
    if errorlevel 1 (popd & exit /b 1)

    popd
)

where emcc
if errorlevel 1 (
    echo ERROR: emcc is still unavailable after emsdk activation.
    exit /b 1
)

emcc --version
if errorlevel 1 exit /b 1

echo [3/6] Python web dependencies
python -m pip install -e ".[all]"
if errorlevel 1 exit /b 1

echo [4/6] Playwright Chromium
python -m playwright install chromium
if errorlevel 1 exit /b 1

echo [5/6] Repository/web wiring checks
python tools\promote_wasm_template.py --check
if errorlevel 1 exit /b 1

python tools\check_repo.py
if errorlevel 1 exit /b 1

echo [6/6] Canonical web profile
python tests\run_tests.py web --version v403 --baseline v402 --keep-going
if errorlevel 1 exit /b 1

echo.
echo PASS: local Web/WASM profile is green.
echo Commit engine\build\zchezz_wasm.html with the patch.
exit /b 0
