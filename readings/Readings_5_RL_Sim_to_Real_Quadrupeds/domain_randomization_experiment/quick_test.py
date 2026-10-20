"""
quick_test.py - a few-minute end-to-end smoke test of the whole pipeline.

1. gymnasium's check_env on the custom environment (fixed and randomized)
2. train both conditions for a tiny budget      -> runs_quick/
3. evaluate with a handful of episodes            -> results_quick/
4. make every plot                                 -> results_quick/
5. export a policy to numpy and run the deployment loop (phase-2 tools)

The real models in runs/ and results in results/ are never touched.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    t0 = time.time()
    env = dict(os.environ, DR_QUICK="1")

    print(">>> gymnasium check_env")
    from gymnasium.utils.env_checker import check_env
    from envs.walker_env import WalkerEnv
    check_env(WalkerEnv(), skip_render_check=True)
    check_env(WalkerEnv(randomize=True), skip_render_check=True)
    check_env(WalkerEnv(randomize=True, extras=True), skip_render_check=True)
    print("check_env passed for fixed, randomized and randomized_plus environments")

    steps = [
        ["train.py", "--quick"],
        ["evaluate.py", "--episodes", "5", "--sweep-episodes", "3"],
        ["plot_results.py"],
        ["export_policy.py"],
        ["deploy_sim.py", "--episodes", "2"],
    ]
    for cmd in steps:
        print(f"\n>>> python {' '.join(cmd)}")
        subprocess.run([sys.executable, str(ROOT / cmd[0]), *cmd[1:]], check=True, cwd=ROOT, env=env)

    print(f"\nQuick test PASSED in {(time.time() - t0) / 60:.1f} min. "
          f"Outputs are in {ROOT / 'runs_quick'} and {ROOT / 'results_quick'}")
