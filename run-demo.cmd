@echo off
rem ASCII-only batch: Chinese messages come from delivery\msg\*.txt via TYPE.
setlocal DisableDelayedExpansion
cd /d "%~dp0"
if errorlevel 1 exit /b 1
set "PYTHONUTF8=1"
for /f "tokens=2 delims=:" %%G in ('chcp') do set "PREVCP=%%G"
if not exist ".venv\Scripts\python.exe" goto notinstalled
".venv\Scripts\python.exe" -X utf8 delivery\run_demo.py %*
set "LIUSHAO_RESULT=%ERRORLEVEL%"
goto finish

:notinstalled
chcp 65001 >nul
echo Run the one-click installer first.
call :say run-not-installed
set "LIUSHAO_RESULT=1"
goto finish

:finish
if defined PREVCP chcp %PREVCP% >nul
if not "%LIUSHAO_RESULT%"=="0" if not defined LIUSHAO_NONINTERACTIVE pause
exit /b %LIUSHAO_RESULT%

:say
if exist "delivery\msg\%~1.txt" type "delivery\msg\%~1.txt"
exit /b 0
