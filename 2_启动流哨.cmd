@echo off
rem ASCII-only batch: Chinese messages come from delivery\msg\*.txt via TYPE.
setlocal DisableDelayedExpansion
cd /d "%~dp0"
if errorlevel 1 exit /b 1
title Liushao
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
for /f "tokens=2 delims=:" %%G in ('chcp') do set "PREVCP=%%G"
chcp 65001 >nul

"runtime\python\python.exe" -I -X utf8 "delivery\preflight.py"
if errorlevel 1 goto notinstalled
if not exist "demo-state" mkdir "demo-state"
set "LIUSHAO_BROWSER_ACTIVATION_ID=%RANDOM%-%RANDOM%-%RANDOM%-%RANDOM%"
start "" /b "%COMSPEC%" /d /c delivery\browser-activate.cmd %LIUSHAO_BROWSER_ACTIVATION_ID%
".venv\Scripts\python.exe" -X utf8 "delivery\run_demo.py" %*
set "RESULT=%ERRORLEVEL%"
type nul >"demo-state\browser-open-%LIUSHAO_BROWSER_ACTIVATION_ID%.cancel"
goto finish

:notinstalled
echo This folder is not installed yet.
call :say not-installed
set "RESULT=1"
goto finish

:finish
if defined PREVCP chcp %PREVCP% >nul
if not "%RESULT%"=="0" if not defined LIUSHAO_NONINTERACTIVE pause
exit /b %RESULT%

:say
if exist "delivery\msg\%~1.txt" type "delivery\msg\%~1.txt"
exit /b 0
