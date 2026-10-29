@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"
if errorlevel 1 exit /b 1
if not exist ".venv\Scripts\python.exe" (
  echo Please follow README.md to create the Python environment first.
  pause
  exit /b 1
)
if not defined LIUSHAO_ARTIFACTS_DIR set "LIUSHAO_ARTIFACTS_DIR=%CD%\artifacts"
if not exist "%LIUSHAO_ARTIFACTS_DIR%\model.joblib" (
  echo Missing model.joblib. Check LIUSHAO_ARTIFACTS_DIR and the trusted model handoff described in README.md.
  pause
  exit /b 1
)
if not exist "%LIUSHAO_ARTIFACTS_DIR%\model.json" (
  echo Missing model.json. Check LIUSHAO_ARTIFACTS_DIR and the matching model handoff described in README.md.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py --server.headless false %*
set "LIUSHAO_EXIT_CODE=%ERRORLEVEL%"
pause
exit /b %LIUSHAO_EXIT_CODE%
