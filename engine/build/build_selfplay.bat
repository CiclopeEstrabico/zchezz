@echo off
REM ═══════════════════════ CONFIGURATION ═══════════════════════
REM ENGINE selects engine/c/zchezz_%ENGINE%/ — pass as first arg to override.
set ENGINE=v322
REM ═══════════════════════════════════════════════════════════
if not "%~1"=="" set ENGINE=%~1
set "PATH=C:\mingw64\bin;%PATH%"
cd /d "%~dp0"
echo Compiling Zchezz selfplay tool (ENGINE=%ENGINE%)...
mingw32-make.exe ENGINE=%ENGINE% selfplay
if %ERRORLEVEL% equ 0 (
    echo.
    echo SUCCESS: Compilation complete! -^> selfplay.exe
    echo.
) else (
    echo.
    echo ERROR: Compilation failed.
    echo.
)
pause
