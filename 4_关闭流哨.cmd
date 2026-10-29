@echo off
rem Stop only the verified service owned by this folder.
setlocal DisableDelayedExpansion
cd /d "%~dp0"
if errorlevel 1 exit /b 1
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
for /f "tokens=2 delims=:" %%G in ('chcp') do set "PREVCP=%%G"
chcp 65001 >nul
"runtime\python\python.exe" -I -X utf8 "delivery\preflight.py"
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -X utf8 "delivery\run_demo.py" --stop
set "RESULT=%ERRORLEVEL%"
if defined PREVCP chcp %PREVCP% >nul
if not "%RESULT%"=="0" if not defined LIUSHAO_NONINTERACTIVE pause
exit /b %RESULT%
