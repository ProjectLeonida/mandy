@echo off
title Leonida - Mod Manager
rem Double-click to open the mod manager window. It manages <port>\mods and never touches the game package.
setlocal
cd /d "%~dp0"
set "PY_EXE="
set "PY_ARG="
for %%V in (3.14 3.13 3.12 3) do (
  if not defined PY_EXE (
    py -%%V -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>nul && (set "PY_EXE=py" & set "PY_ARG=-%%V")
  )
)
if not defined PY_EXE (
  where python >nul 2>nul && (
    python -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>nul && set "PY_EXE=python"
  )
)
if not defined PY_EXE call :find_user_python
if not defined PY_EXE (
  echo Python 3.12 or newer is needed: https://www.python.org/downloads/
  echo Tick "Add python.exe to PATH" during setup.
  if not defined LEONIDA_NO_PAUSE pause
  exit /b 1
)
"%PY_EXE%" %PY_ARG% -I "%~dp0tools\leonida_mods.py" gui %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  if not defined LEONIDA_NO_PAUSE pause
)
exit /b %RC%

:find_user_python
for /f "delims=" %%D in ('dir /b /ad /o-n "%LOCALAPPDATA%\Programs\Python\Python3*" 2^>nul') do call :try_python "%LOCALAPPDATA%\Programs\Python\%%D\python.exe"
exit /b 0

:try_python
if defined PY_EXE exit /b 0
if not exist "%~1" exit /b 0
"%~1" -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>nul && set "PY_EXE=%~1"
exit /b 0
