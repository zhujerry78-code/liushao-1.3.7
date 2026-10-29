@echo off
rem --------------------------------------------------------------------
rem  Technical entry: install only, no launch. Students should use 1_.
rem  ASCII-only batch: Chinese messages come from delivery\msg\*.txt.
rem  Offline installation through the same serialized one-click entry.
rem --------------------------------------------------------------------
setlocal DisableDelayedExpansion
cd /d "%~dp0"
if errorlevel 1 exit /b 1
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "LIUSHAO_LOG_DIR=%~dp0install-logs"
if not exist "%LIUSHAO_LOG_DIR%\" mkdir "%LIUSHAO_LOG_DIR%" 2>nul
if not exist "%LIUSHAO_LOG_DIR%\" goto nologdir
:choose_log
set "LIUSHAO_BOOTSTRAP_LOG=%LIUSHAO_LOG_DIR%\bootstrap-%RANDOM%-%RANDOM%.log"
if exist "%LIUSHAO_BOOTSTRAP_LOG%" goto choose_log
for /f "tokens=2 delims=:" %%G in ('chcp') do set "LIUSHAO_PREVIOUS_CP=%%G"
chcp 65001 >nul

call "delivery\find-python.cmd"
>"%LIUSHAO_BOOTSTRAP_LOG%" echo Liushao Python startup check
if not defined PYEXE goto nopython
>>"%LIUSHAO_BOOTSTRAP_LOG%" echo Selected interpreter: "%PYEXE%"
echo Startup log: "%LIUSHAO_BOOTSTRAP_LOG%"
call :say install-interpreter
echo   "%PYEXE%"
"%PYEXE%" -X utf8 -c "import sys, runpy; print(sys.executable); print(sys.version); runpy.run_path('delivery/setup_demo.py', run_name='liushao_install_preflight')" >>"%LIUSHAO_BOOTSTRAP_LOG%" 2>&1
set "LIUSHAO_RESULT=%ERRORLEVEL%"
>>"%LIUSHAO_BOOTSTRAP_LOG%" echo Python startup exit code: %LIUSHAO_RESULT%
if not "%LIUSHAO_RESULT%"=="0" goto pythonfailed

if not "%~1"=="" goto passthrough
"%PYEXE%" -X utf8 delivery\oneclick.py --no-launch
goto installed
:passthrough
"%PYEXE%" -X utf8 delivery\oneclick.py --no-launch %*
:installed
set "LIUSHAO_RESULT=%ERRORLEVEL%"
>>"%LIUSHAO_BOOTSTRAP_LOG%" echo Installer exit code: %LIUSHAO_RESULT%
goto finish

:nologdir
echo Cannot create install-logs. Extract the ZIP into a writable fresh folder.
call :say install-cannot-log
set "LIUSHAO_RESULT=1"
goto finish

:pythonfailed
type "%LIUSHAO_BOOTSTRAP_LOG%"
echo Python could not start. Keep install-logs and use a complete 64-bit Python 3.12.
call :say install-python-failed
goto finish

:nopython
type "%LIUSHAO_BOOTSTRAP_LOG%"
echo No 64-bit Python 3.12 found.
call :say install-no-python
set "LIUSHAO_RESULT=1"
goto finish

:finish
if defined LIUSHAO_PREVIOUS_CP chcp %LIUSHAO_PREVIOUS_CP% >nul
if not defined LIUSHAO_NONINTERACTIVE pause
exit /b %LIUSHAO_RESULT%

:say
if exist "delivery\msg\%~1.txt" type "delivery\msg\%~1.txt"
exit /b 0
