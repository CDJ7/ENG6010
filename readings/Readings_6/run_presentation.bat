@echo off
REM ======================================================================
REM  run_presentation.bat - set up Python (first time only) and start the
REM  interactive Readings 6 presentation in fullscreen.
REM  Double-click it, or run it from a terminal in this folder.
REM  Extra arguments are passed through, e.g.:  run_presentation.bat --start 7
REM ======================================================================
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating virtual environment in .venv ...
    py -3 -m venv .venv
    if errorlevel 1 (
        echo Could not create the venv. Install Python 3.10 or newer from python.org
        echo and tick "Add python.exe to PATH" during setup.
        pause
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"

python -c "import torch, pygame, gymnasium" >nul 2>&1
if errorlevel 1 (
    echo [2/3] Installing packages - first time only, takes a few minutes ...
    python -m pip install --upgrade pip
    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Pinned versions not available for this Python, installing latest instead ...
        python -m pip install gymnasium numpy pygame matplotlib imageio
    )
)

echo [3/3] Starting the presentation - press Esc to quit ...
python presentation\interactive_presentation.py --fullscreen %*
if errorlevel 1 pause
