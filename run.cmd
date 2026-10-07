@echo off
rem ===========================================================================
rem  StartLux-Decision  -  one-click launcher.  Double-click this file.
rem
rem  Brings up the whole local stack:
rem      llama-server (llama.cpp Vulkan, GPU) -> 127.0.0.1:8081
rem      decision service /v1/systemone       -> 127.0.0.1:8090
rem
rem  Command line (optional):
rem      run.cmd --follow                 keep a live health line; Ctrl+C stops everything
rem      run.cmd --mario                  also play level 1-1 with jev-mario
rem      run.cmd --mario --parallel 6     ... with the 11 options in parallel
rem      run.cmd --model 2b               use the 2B Q4_K_M engine
rem      run.cmd --selftest               exercise all four question types
rem      run.cmd --status                 show what is up, change nothing
rem
rem  Every message the user sees is printed by run_local.py, so this batch file
rem  stays pure ASCII and cannot be mangled by the console code page.
rem ===========================================================================
chcp 65001 >nul
setlocal
set "HERE=%~dp0"
set "PY=%STARTLUX_PY%"
if not defined PY set "PY=C:\Users\Libai\.workbuddy\binaries\python\versions\3.13.12\python.exe"

if not exist "%PY%" (
    echo [XX] Python interpreter not found:
    echo      %PY%
    echo      Set STARTLUX_PY to the full path of python.exe and run this again.
    echo.
    pause
    exit /b 1
)

"%PY%" -X utf8 "%HERE%run_local.py" %*

if errorlevel 1 (
    echo.
    echo [XX] run_local.py exited with code %ERRORLEVEL%
    pause
)
endlocal
