@echo off
rem ASCII-only batch: every Chinese message lives in delivery\msg\*.txt and is
rem printed with TYPE, so cmd never has to parse multi-byte text.
rem The package ships its own CPython 3.12 under runtime\python: no install,
rem no admin rights, no network. Missing it means the tree was damaged.
setlocal DisableDelayedExpansion
cd /d "%~dp0"
if errorlevel 1 exit /b 1
title Liushao setup
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
for /f "tokens=2 delims=:" %%G in ('chcp') do set "PREVCP=%%G"
chcp 65001 >nul

echo.
call :say banner
echo.

call "delivery\find-python.cmd"
if not defined PYEXE goto nopython

call :say found-python
echo       "%PYEXE%"
"%PYEXE%" -X utf8 -c "import sys;print('      '+sys.version.replace(chr(10),' '))" 2>nul
echo.
call :say installing-deps
echo.
if not exist "demo-state" mkdir "demo-state"
set "LIUSHAO_BROWSER_ACTIVATION_ID=%RANDOM%-%RANDOM%-%RANDOM%-%RANDOM%"
start "" /b "%COMSPEC%" /d /c delivery\browser-activate.cmd %LIUSHAO_BROWSER_ACTIVATION_ID%
"%PYEXE%" -X utf8 "delivery\oneclick.py" %*
set "RESULT=%ERRORLEVEL%"
type nul >"demo-state\browser-open-%LIUSHAO_BROWSER_ACTIVATION_ID%.cancel"
echo.
if not "%RESULT%"=="0" goto failed
goto finish

:failed
echo Exit code: %RESULT%
call :say failed
goto finish

:nopython
echo No usable 64-bit Python 3.12 found (expected runtime\python\python.exe).
call :say no-python
set "RESULT=1"
goto finish

:finish
if defined PREVCP chcp %PREVCP% >nul
echo.
if not "%RESULT%"=="0" if not defined LIUSHAO_NONINTERACTIVE pause
exit /b %RESULT%

:say
rem call :say <name>  prints delivery\msg\<name>.txt when present
if exist "delivery\msg\%~1.txt" type "delivery\msg\%~1.txt"
exit /b 0
