@echo off
rem ====================================================================
rem  Subroutine only: call "delivery\find-python.cmd" then read %PYEXE%.
rem  PYEXE is set only for a 64-bit Python 3.12; nothing is modified.
rem  ASCII-only on purpose: Chinese messages live in delivery\msg\*.txt.
rem  Order: explicit override -> bundled runtime -> launcher -> dirs -> registry -> PATH.
rem ====================================================================
:findpython
set "PYEXE="
set "CANDIDATE=%LIUSHAO_PYTHON%"
call :probe
if defined PYEXE goto :eof

rem Bundled, tested interpreter: no install, no admin rights, fully offline.
rem pushd/popd normalises the path so logs show ...\runtime\python\python.exe.
pushd "%~dp0..\runtime\python" 2>nul
if errorlevel 1 goto afterbundled
set "CANDIDATE=%CD%\python.exe"
call :probe
popd
if defined PYEXE goto :eof
:afterbundled

for /f "usebackq delims=" %%I in (`py -3.12 -c "import sys;print(sys.executable)" 2^>nul`) do (set "CANDIDATE=%%I"
call :probe
)
if defined PYEXE goto :eof

set "CANDIDATE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
call :probe
if defined PYEXE goto :eof
if defined PYEXE goto :eof
if defined PYEXE goto :eof

for /f "tokens=2,*" %%A in ('reg query "HKCU\Software\Python\PythonCore\3.12\InstallPath" /v ExecutablePath 2^>nul ^| findstr /i "ExecutablePath"') do (set "CANDIDATE=%%B"
call :probe
)
if defined PYEXE goto :eof
for /f "tokens=2,*" %%A in ('reg query "HKCU\Software\Python\PythonCore\3.12\InstallPath" /ve 2^>nul ^| findstr /i "REG_SZ"') do (set "CANDIDATE=%%Bpython.exe"
call :probe
)
if defined PYEXE goto :eof
for /f "tokens=2,*" %%A in ('reg query "HKLM\Software\Python\PythonCore\3.12\InstallPath" /reg:64 /v ExecutablePath 2^>nul ^| findstr /i "ExecutablePath"') do (set "CANDIDATE=%%B"
call :probe
)
if defined PYEXE goto :eof
for /f "tokens=2,*" %%A in ('reg query "HKLM\Software\Python\PythonCore\3.12\InstallPath" /reg:64 /ve 2^>nul ^| findstr /i "REG_SZ"') do (set "CANDIDATE=%%Bpython.exe"
call :probe
)
if defined PYEXE goto :eof

for %%I in (python.exe) do (set "CANDIDATE=%%~$PATH:I"
call :probe
)
goto :eof

:probe
rem Accept 64-bit 3.12 only; every other interpreter is skipped on purpose.
if defined PYEXE goto :eof
if "%CANDIDATE%"=="" goto :eof
if not exist "%CANDIDATE%" goto :eof
"%CANDIDATE%" -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,12) and sys.maxsize>2**32 else 9)" >nul 2>&1
if errorlevel 1 goto :eof
set "PYEXE=%CANDIDATE%"
goto :eof
