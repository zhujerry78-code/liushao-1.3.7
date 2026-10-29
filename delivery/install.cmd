@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"
set "PYTHONUTF8=1"
if defined LIUSHAO_PYTHON (
  "%LIUSHAO_PYTHON%" delivery\setup_demo.py %*
) else (
  py -3.12 delivery\setup_demo.py %*
)
set "LIUSHAO_RESULT=%ERRORLEVEL%"
if not defined LIUSHAO_NONINTERACTIVE pause
exit /b %LIUSHAO_RESULT%
