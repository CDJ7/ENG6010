"""
evaluate_and_plot.py - convenience wrapper: run evaluate.py, then plot_results.py.
(Used by the "Evaluate + Plots" VS Code run configuration.)
"""
import subprocess
import sys

import config as C

if __name__ == "__main__":
    for script in ["evaluate.py", "plot_results.py"]:
        print(f"\n>>> python {script}")
        subprocess.run([sys.executable, str(C.ROOT_DIR / script)], check=True, cwd=C.ROOT_DIR)
