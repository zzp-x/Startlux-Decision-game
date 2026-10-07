@echo off
rem ===========================================================================
rem  StartLux-Decision  -  stop every service the launcher started.
rem      llama-server  (port 8081 or 8082)
rem      decision service (port 8090)
rem ===========================================================================
chcp 65001 >nul
setlocal
set "HERE=%~dp0"
set "PY=%STARTLUX_PY%"
if not defined PY set "PY=C:\Users\Libai\.workbuddy\binaries\python\versions\3.13.12\python.exe"

if not exist "%PY%" (
    echo [XX] Python interpreter not found:
    echo      %PY%
    echo.
    pause
    exit /b 1
)

"%PY%" -X utf8 "%HERE%run_local.py" --stop

if errorlevel 1 (
    echo.
    echo [XX] something is still listening; see the messages above.
    pause
)
endlocal
