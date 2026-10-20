"""
export_and_deploy.py - phase-2 demo in one go: export every policy to numpy, then run the
deployment loop with the randomized and fixed policies on the pseudo-real robot.
(Used by the "Phase 2: export policy + deployment-loop demo" VS Code run configuration.)
"""
import subprocess
import sys

import config as C

if __name__ == "__main__":
    steps = [["export_policy.py"],
             ["deploy_sim.py", "--run", "randomized_seed0", "--condition", "pseudo_real", "--episodes", "5"],
             ["deploy_sim.py", "--run", "fixed_seed0", "--condition", "pseudo_real", "--episodes", "5"]]
    for cmd in steps:
        print(f"\n>>> python {' '.join(cmd)}")
        subprocess.run([sys.executable, str(C.ROOT_DIR / cmd[0]), *cmd[1:]], check=True, cwd=C.ROOT_DIR)
