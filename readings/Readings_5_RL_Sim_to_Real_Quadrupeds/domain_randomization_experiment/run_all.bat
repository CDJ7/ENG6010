@echo off
REM ======================================================================
REM  run_all.bat - set up the virtual environment and run the full pipeline
REM  Double-click it, or run it from a Command Prompt in this folder.
REM  Total time on a laptop CPU: roughly 1.5-2.5 hours (7 training runs).
REM ======================================================================
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [1/6] Creating virtual environment with Python 3.11 ...
    py -3.11 -m venv .venv
    if errorlevel 1 (
        echo Could not create the venv. Is Python 3.11 installed from python.org?
        pause
        exit /b 1
    )
)

echo [2/6] Installing requirements (first time takes a few minutes) ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo [3/6] Quick smoke test ...
".venv\Scripts\python.exe" quick_test.py
if errorlevel 1 goto :error

echo [4/6] Training both controllers, 3 seeds each, then the phase-2 controller ...
".venv\Scripts\python.exe" train.py --all --timesteps 1500000
if errorlevel 1 goto :error
".venv\Scripts\python.exe" train.py --mode randomized_plus --seed 0 --timesteps 1500000
if errorlevel 1 goto :error

echo [5/6] Evaluating on all test conditions ...
".venv\Scripts\python.exe" evaluate.py
if errorlevel 1 goto :error

echo [6/6] Making plots and the summary table ...
".venv\Scripts\python.exe" plot_results.py
if errorlevel 1 goto :error

echo.
echo Done! Plots and CSVs are in the results folder.
echo Watch a controller with:  .venv\Scripts\python.exe watch.py --model fixed --condition low_friction
pause
exit /b 0

:error
echo.
echo Something failed - see the message above and the Troubleshooting section of README.md.
pause
exit /b 1
