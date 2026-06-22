@echo off
set "PATH=C:\mingw64\bin;%PATH%"
cd /d "%~dp0"
echo Compiling Zchezz...
mingw32-make.exe
if %ERRORLEVEL% equ 0 (
    echo.
    echo SUCCESS: Compilation complete!
    echo.
) else (
    echo.
    echo ERROR: Compilation failed.
    echo.
)
pause
