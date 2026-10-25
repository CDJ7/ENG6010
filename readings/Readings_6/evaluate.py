"""
evaluate.py - compare controllers on the real simulator.

Every controller is run on the same set of evaluation episodes (same
initial states, chosen by the env seed) and we record

* the episode return (sum of Pendulum rewards over 200 steps; 0 is perfect,
  random play gives roughly -1200), and
* the wall-clock time per decision, i.e. how long act(obs) takes.

The second number is the point of the whole project: Mode-2 planning is
accurate but slow, Mode-1 should be nearly as accurate and far faster.
"""

import numpy as np

from collect_data import rollout


def eval_seeds_for(seed, n):
    """Evaluation initial states never used for training or DAgger."""
    return [1_000_000 + seed * 1000 + i for i in range(n)]


def evaluate_controller(env, controller, seeds):
    """Run one episode per env seed; return the list of returns and times."""
    returns, times = [], []
    for s in seeds:
        ep = rollout(env, controller, seed=s)
        returns.append(ep["ret"])
        times.extend(ep["decision_times"].tolist())
    return {"returns": returns, "decision_times_s": times}


def summarize(results_by_controller):
    """Pool results over seeds into mean +/- std of return and decision time.

    Args:
        results_by_controller: {name: {"returns": [...], "decision_times_s": [...],
                                       "per_seed_mean_return": [...]}}
    """
    out = {}
    for name, r in results_by_controller.items():
        rets = np.asarray(r["returns"])
        t_ms = 1000.0 * np.asarray(r["decision_times_s"])
        out[name] = {
            "n_episodes": int(len(rets)),
            "return_mean": float(rets.mean()),
            "return_std": float(rets.std()),
            "per_seed_mean_return": [float(x) for x in r["per_seed_mean_return"]],
            "decision_time_ms_mean": float(t_ms.mean()),
            "decision_time_ms_std": float(t_ms.std()),
            "returns": [float(x) for x in rets],
        }
    return out
