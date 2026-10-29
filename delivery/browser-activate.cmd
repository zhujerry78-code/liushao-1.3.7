@echo off
rem This runs under the original desktop CMD, before the packaged Python starts.
rem Read only a constrained loopback URL. Never execute file contents as commands.
setlocal DisableDelayedExpansion
cd /d "%~dp0.."
if errorlevel 1 exit /b 1
set "ACTIVATION_ID=%~1"
set "REQUEST=demo-state\browser-open-%ACTIVATION_ID%.url.txt"
set "PROFILE_REQUEST=demo-state\browser-open-%ACTIVATION_ID%.profile-id.txt"
set "CANCEL=demo-state\browser-open-%ACTIVATION_ID%.cancel"
set "OPENED="
set "BROWSER="
if exist "%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe" set "BROWSER=%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"
if exist "%PROGRAMFILES%\Google\Chrome\Application\chrome.exe" set "BROWSER=%PROGRAMFILES%\Google\Chrome\Application\chrome.exe"
if exist "%LOCALAPPDATA%\Microsoft\Edge\Application\msedge.exe" set "BROWSER=%LOCALAPPDATA%\Microsoft\Edge\Application\msedge.exe"
if exist "%PROGRAMFILES%\Microsoft\Edge\Application\msedge.exe" set "BROWSER=%PROGRAMFILES%\Microsoft\Edge\Application\msedge.exe"
if exist "%PROGRAMFILES(X86)%\Microsoft\Edge\Application\msedge.exe" set "BROWSER=%PROGRAMFILES(X86)%\Microsoft\Edge\Application\msedge.exe"
set /a REMAINING=900

:poll
if exist "%CANCEL%" goto finish
if defined OPENED goto wait
if not exist "%REQUEST%" goto wait
if not exist "%PROFILE_REQUEST%" goto wait
set "PROFILE_ID="
for /f "delims=" %%P in ('findstr /r /x /c:"[0-9a-f][0-9a-f]*" "%PROFILE_REQUEST%"') do set "PROFILE_ID=%%P"
if not defined PROFILE_ID goto wait
set "BROWSER_PROFILE=%LOCALAPPDATA%\Liushao\BrowserProfiles\%PROFILE_ID%"
for /f "delims=" %%U in ('findstr /r /x /c:"http://127[.]0[.]0[.]1:[0-9][0-9]*/?_liushao_launch=[0-9a-f][0-9a-f]*" "%REQUEST%"') do (
  call :open "%%U"
  set "OPENED=1"
)

:wait
set /a REMAINING-=1
if %REMAINING% leq 0 goto finish
ping.exe -n 2 -w 1000 127.0.0.1 >nul 2>nul
goto poll

:finish
if exist "%CANCEL%" del /q "%CANCEL%" >nul 2>nul
exit /b 0

:open
if not defined BROWSER goto default_browser
start "" "%BROWSER%" "--app=%~1" "--user-data-dir=%BROWSER_PROFILE%" --no-first-run
exit /b 0
:default_browser
start "" "%~1"
exit /b 0
