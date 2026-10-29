@echo off
rem ASCII-only batch: Chinese messages come from delivery\msg\*.txt via TYPE.
setlocal DisableDelayedExpansion
cd /d "%~dp0"
if errorlevel 1 exit /b 1
title Liushao self-check
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
for /f "tokens=2 delims=:" %%G in ('chcp') do set "PREVCP=%%G"
chcp 65001 >nul

echo.
call :say selfcheck-banner
echo.

set "SCPY="
call "delivery\find-python.cmd"
if defined PYEXE set "SCPY=%PYEXE%"
if not defined SCPY goto nopython

echo Interpreter: "%SCPY%"
echo.
"%SCPY%" -X utf8 "delivery\selfcheck.py"
set "RESULT=%ERRORLEVEL%"
goto finish

:nopython
call :say selfcheck-nopython
set "REPORT=%USERPROFILE%\Desktop\Liushao-basic-diagnostic-%RANDOM%-%RANDOM%.txt"
if not exist "%USERPROFILE%\Desktop" set "REPORT=%CD%\Liushao-basic-diagnostic-%RANDOM%-%RANDOM%.txt"
echo Liushao basic diagnostic > "%REPORT%"
echo generated: %DATE% %TIME% >> "%REPORT%"
echo install dir: "%CD%" >> "%REPORT%"
echo OS: >> "%REPORT%"
ver >> "%REPORT%" 2>&1
echo Python launchers: >> "%REPORT%"
py -0p >> "%REPORT%" 2>&1
echo python on PATH: >> "%REPORT%"
where python >> "%REPORT%" 2>&1
echo.
echo Report written to: "%REPORT%"
set "RESULT=1"
goto finish

:finish
if defined PREVCP chcp %PREVCP% >nul
echo.
if not defined LIUSHAO_NONINTERACTIVE pause
exit /b %RESULT%

:say
if exist "delivery\msg\%~1.txt" type "delivery\msg\%~1.txt"
exit /b 0
