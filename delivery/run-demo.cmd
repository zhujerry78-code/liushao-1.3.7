@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"
set "PYTHONUTF8=1"
if not exist ".venv\Scripts\python.exe" (
  echo Run install.cmd first with 64-bit Python 3.12.
  if not defined LIUSHAO_NONINTERACTIVE pause
  exit /b 1
)
".venv\Scripts\python.exe" delivery\run_demo.py %*
set "LIUSHAO_RESULT=%ERRORLEVEL%"
if not defined LIUSHAO_NONINTERACTIVE pause
exit /b %LIUSHAO_RESULT%
