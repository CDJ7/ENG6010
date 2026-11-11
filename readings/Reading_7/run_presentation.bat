@echo off
REM ======================================================================
REM  run_presentation.bat - set up Python (first time only) and start the
REM  interactive Reading 7 slideshow in fullscreen.
REM  Double-click it, or run it from a terminal in this folder.
REM  Extra arguments are passed through, e.g.:  run_presentation.bat --start 7
REM ======================================================================
setlocal
cd /d "%~dp0"

REM --- reuse .venv only if it was made with a supported Python (3.12-3.15)
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import sys; sys.exit(0 if (3,12) <= sys.version_info[:2] <= (3,15) else 1)"
    if errorlevel 1 (
        echo The existing .venv uses an unsupported Python version - rebuilding it ...
        rmdir /s /q ".venv"
    )
)
if exist ".venv\Scripts\python.exe" goto have_venv

REM --- find a Python with the "py" launcher (3.13 first, like Readings 6)
set "PYVER="
for %%V in (3.13 3.12 3.14 3.15) do (
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
python -c "import numpy, pygame" >nul 2>&1
if not errorlevel 1 goto run

echo [2/3] Installing numpy and pygame-ce - first time only ...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Pinned versions failed - trying the latest versions instead ...
    python -m pip install numpy pygame-ce
)
python -c "import numpy, pygame" >nul 2>&1
if errorlevel 1 goto install_failed

:run
echo [3/3] Starting the slideshow - press Esc to quit ...
python presentation\interactive_presentation.py --fullscreen %*
if errorlevel 1 pause
exit /b 0

:no_python
echo.
echo  No supported Python found. Install Python 3.13 from https://www.python.org/downloads/
echo  and tick "Add python.exe to PATH", then run this file again.
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
