@echo off
REM ======================================================================
REM  run_presentation.bat - set up Python (first time only) and start the
REM  interactive Readings 6 presentation in fullscreen.
REM  Double-click it, or run it from a terminal in this folder.
REM  Extra arguments are passed through, e.g.:  run_presentation.bat --start 7
REM
REM  PyTorch has Windows builds for Python 3.10-3.14 but NOT 3.15 yet, so
REM  this script picks a compatible Python with the "py" launcher
REM  (3.13 first) instead of whatever "python" happens to be.
REM ======================================================================
setlocal
cd /d "%~dp0"

REM --- reuse .venv only if it was made with a compatible Python ----------
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,14) else 1)"
    if errorlevel 1 (
        echo The existing .venv uses an unsupported Python version - rebuilding it ...
        rmdir /s /q ".venv"
    )
)
if exist ".venv\Scripts\python.exe" goto have_venv

REM --- find a compatible Python -------------------------------------------
set "PYVER="
for %%V in (3.13 3.12 3.14 3.11 3.10) do (
    if not defined PYVER (
        py -%%V -c "import sys" >nul 2>&1 && set "PYVER=%%V"
    )
)
if not defined PYVER goto no_python

echo [1/3] Creating virtual environment with Python %PYVER% ...
py -%PYVER% -m venv .venv
if errorlevel 1 goto no_python

:have_venv
call ".venv\Scripts\activate.bat"
python -c "import torch, pygame, gymnasium, numpy" >nul 2>&1
if not errorlevel 1 goto run

echo [2/3] Installing packages - first time only, takes a few minutes ...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Pinned versions failed - trying the latest versions instead ...
    python -m pip install torch gymnasium numpy pygame-ce matplotlib imageio
)
python -c "import torch, pygame, gymnasium, numpy" >nul 2>&1
if errorlevel 1 goto install_failed

:run
echo [3/3] Starting the presentation - press Esc to quit ...
python presentation\interactive_presentation.py --fullscreen %*
if errorlevel 1 pause
exit /b 0

:no_python
echo.
echo  No compatible Python found.
echo  PyTorch needs Python 3.10 to 3.14 - it does not support Python 3.15 yet.
echo  Install Python 3.13 from https://www.python.org/downloads/
echo  - you can keep 3.15 installed too - then run this file again.
echo.
pause
exit /b 1

:install_failed
echo.
echo  The packages did not install. Scroll up for the first error message.
echo  Deleting the .venv folder and running this file again often fixes it.
echo.
pause
exit /b 1
