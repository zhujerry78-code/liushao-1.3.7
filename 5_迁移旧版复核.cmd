@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"
if errorlevel 1 exit /b 1
set "PYTHONUTF8=1"
"runtime\python\python.exe" -X utf8 "delivery\migrate_reviews.py" --interactive
set "RESULT=%ERRORLEVEL%"
pause
exit /b %RESULT%
