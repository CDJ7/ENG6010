"""
main.py - run the whole "Mode-2 -> Mode-1" pipeline with one command.

    python main.py              # full experiment (3 seeds), writes results/
    python main.py --quick      # tiny smoke test, writes results_quick/
    python main.py --symlog     # train the world model on symlog targets

For every seed the pipeline does:

  1. collect random-action data in Pendulum-v1 (saved to data/*.npz)
  2. train the world model  f(s, a) -> s' - s
  3. measure its open-loop prediction error for horizons 1..30
  4. Mode-2: CEM MPC inside the learned model (optionally a model-based
     RL loop: collect with CEM, add to the data, retrain; --mbrl-rounds)
  5. Mode-1: distil the CEM planner into a small policy with BC + DAgger
  6. evaluate Random / Random-shooting MPC / CEM MPC / Distilled policy
     on the same unseen start states (return + decision time)

and finally:

  7. make all plots, GIFs and results/summary.json
"""

import argparse
import json
import os
import time

import numpy as np
import torch

import plots
from collect_data import (RandomController, collect_random_dataset, episodes_to_dataset,
                          merge_datasets, rollout)
from config import get_config
from evaluate import eval_seeds_for, evaluate_controller, summarize
from planner import CEMPlanner, RandomShootingPlanner
from policy import PolicyController, distill_with_dagger
from utils import make_env, set_seed
from world_model import multi_step_error, train_world_model

CONTROLLERS = ["Random", "Random-shooting MPC", "CEM MPC", "Distilled policy"]


def mbrl_loop(cfg, seed, env, model, data, heldout, rec):
    """Optional model-based RL loop (step 4b).

    Each round: plan with CEM in the current model on the same start
    states, add the visited transitions to the dataset, retrain.
    """
    print("[4b] model-based RL loop: collect with CEM, retrain")
    loop_seeds = [seed * 1000 + 700 + j for j in range(cfg.mbrl_episodes_per_round)]
    rounds = {"round": [], "return_mean": [], "return_std": [], "n_transitions": []}
    for k in range(cfg.mbrl_rounds + 1):
        cem = CEMPlanner(model, cfg, seed=seed)
        eps = [rollout(env, cem, seed=s) for s in loop_seeds]
        rets = [e["ret"] for e in eps]
        rounds["round"].append(k)
        rounds["return_mean"].append(float(np.mean(rets)))
        rounds["return_std"].append(float(np.std(rets)))
        rounds["n_transitions"].append(int(len(data["obs"])))
        print(f"    model {k}: CEM return {np.mean(rets):8.1f} +/- {np.std(rets):6.1f} "
              f"(trained on {len(data['obs'])} transitions)")
        if k == cfg.mbrl_rounds:
            break
        data = merge_datasets(data, episodes_to_dataset(eps))
        model, h = train_world_model(data, cfg, seed)
        rec["wm_histories"].append((f"model {k + 1} (+ CEM data)", h))
    rec["mbrl_rounds"] = rounds
    rec["pred_err"].append((f"model {cfg.mbrl_rounds} (final, + CEM data)",
                            multi_step_error(model, heldout, cfg.max_eval_horizon)))
    return model, data


def run_seed(cfg, seed, out):
    """Steps 1-6 for one seed. Returns everything needed for plots/summary."""
    print(f"\n=== seed {seed} ===")
    set_seed(seed)
    env = make_env(cfg.env_id)
    rec = {}

    # 1. Collect random data ------------------------------------------------
    data, _ = collect_random_dataset(cfg, seed, cfg.n_random_episodes,
                                     path=os.path.join(cfg.data_dir, f"random_seed{seed}.npz"))
    # Held-out random trajectories (different env seeds) for the horizon test.
    _, heldout = collect_random_dataset(cfg, seed + 777, cfg.n_heldout_episodes)
    print(f"[1] collected {len(data['obs'])} random transitions")

    # 2. Train the world model on random data --------------------------------
    print("[2] training world model")
    model, h0 = train_world_model(data, cfg, seed)
    rec["wm_histories"] = [("model 0 (random data)", h0)]

    # 3. Compounding error of the random-data model --------------------------
    rec["pred_err"] = [("model 0 (random data)", multi_step_error(model, heldout, cfg.max_eval_horizon))]

    # 4. Mode-2: the CEM / random-shooting planners (planner.py) use `model`
    # directly; they are evaluated in step 6 and act as the teacher in step 5.
    # 4b. Optional model-based RL loop (off by default, see config.py).
    if cfg.mbrl_rounds > 0:
        model, data = mbrl_loop(cfg, seed, env, model, data, heldout, rec)

    # Optional ablation: same dataset, the other target transform (DreamerV3 symlog).
    if cfg.compare_symlog and seed == cfg.seeds[0]:
        alt, h_alt = train_world_model(data, cfg, seed, use_symlog=not cfg.use_symlog)
        tag = "symlog" if not cfg.use_symlog else "no symlog"
        rec["pred_err"].append((f"model {cfg.mbrl_rounds} ({tag} targets)",
                                multi_step_error(alt, heldout, cfg.max_eval_horizon)))
        rec["symlog_ablation_val_loss"] = h_alt["val_loss"][-1]

    # 5. Mode-1: distil the planner with BC + DAgger --------------------------
    print("[5] distilling CEM planner into a policy (BC + DAgger)")
    teacher = CEMPlanner(model, cfg, seed=seed + 1)
    policy, dagger = distill_with_dagger(teacher, env, cfg, seed)
    rec["dagger"] = dagger

    # 6. Evaluate all four controllers on the same unseen start states -------
    print("[6] evaluating controllers")
    controllers = {
        "Random": RandomController(seed),
        "Random-shooting MPC": RandomShootingPlanner(model, cfg, seed=seed + 2),
        "CEM MPC": CEMPlanner(model, cfg, seed=seed + 3),
        "Distilled policy": PolicyController(policy),
    }
    seeds = eval_seeds_for(seed, cfg.n_eval_episodes)
    rec["eval"] = {}
    for name, ctrl in controllers.items():
        r = evaluate_controller(env, ctrl, seeds)
        rec["eval"][name] = r
        print(f"    {name:20s} return {np.mean(r['returns']):8.1f} +/- {np.std(r['returns']):6.1f}"
              f" | {1000 * np.mean(r['decision_times_s']):8.3f} ms/decision")
    env.close()
    rec["controllers"] = controllers
    rec["model"], rec["policy"] = model, policy
    return rec


def make_media(cfg, controllers, out):
    """Render one swing-up (pole starts hanging down) per controller."""
    import imageio.v2 as imageio
    env = make_env(cfg.env_id, render=True)
    gif_dir = os.path.join(out, "gifs")
    os.makedirs(gif_dir, exist_ok=True)
    frames_by, returns = {}, {}
    for name, ctrl in controllers.items():
        ep = rollout(env, ctrl, seed=424242, start_hanging_down=True, render=True)
        frames = [f[::2, ::2] for f in ep["frames"]]          # 500px -> 250px
        frames_by[name], returns[name] = frames, ep["ret"]
        fname = name.lower().replace(" ", "_").replace("-", "_") + ".gif"
        imageio.mimsave(os.path.join(gif_dir, fname), frames[::cfg.gif_frame_skip],
                        duration=1000 * cfg.gif_frame_skip / 30, loop=0)
    env.close()
    plots.plot_filmstrip(frames_by, out)
    return returns


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="tiny smoke-test run")
    ap.add_argument("--symlog", action="store_true", help="DreamerV3 symlog on world-model targets")
    ap.add_argument("--seeds", type=int, nargs="+", help="override the list of seeds")
    ap.add_argument("--mbrl-rounds", type=int, help="rounds of collect-with-CEM + retrain (default 0)")
    ap.add_argument("--no-gif", action="store_true", help="skip rendering")
    args = ap.parse_args()

    cfg = get_config(args.quick)
    if args.symlog:
        cfg.use_symlog = True
    if args.seeds:
        cfg.seeds = args.seeds
    if args.mbrl_rounds is not None:
        cfg.mbrl_rounds = args.mbrl_rounds
    if args.no_gif:
        cfg.make_gifs = False
    torch.set_num_threads(cfg.torch_threads)
    out = cfg.results_dir
    os.makedirs(out, exist_ok=True)
    t_start = time.time()

    records = {seed: run_seed(cfg, seed, out) for seed in cfg.seeds}
    first = records[cfg.seeds[0]]

    # Pool evaluation results over seeds.
    pooled = {}
    for name in CONTROLLERS:
        pooled[name] = {"returns": [], "decision_times_s": [], "per_seed_mean_return": []}
        for rec in records.values():
            r = rec["eval"][name]
            pooled[name]["returns"] += r["returns"]
            pooled[name]["decision_times_s"] += r["decision_times_s"]
            pooled[name]["per_seed_mean_return"].append(float(np.mean(r["returns"])))
    summary = summarize(pooled)

    # 7. Plots ------------------------------------------------------------------
    print("\n[7] plots")
    plots.plot_wm_loss(first["wm_histories"], out)
    plots.plot_prediction_error(first["pred_err"], out)
    plots.plot_return_bars(summary, out)
    plots.plot_decision_time(summary, out)
    plots.plot_dagger({s: r["dagger"] for s, r in records.items()},
                      summary["CEM MPC"]["return_mean"], summary["Random"]["return_mean"], out)
    if cfg.mbrl_rounds > 0:
        plots.plot_mbrl_rounds({s: r["mbrl_rounds"] for s, r in records.items()}, out)
    gif_returns = make_media(cfg, first["controllers"], out) if cfg.make_gifs else None

    # Save the seed-0 networks (small) so they can be inspected later.
    os.makedirs(os.path.join(out, "models"), exist_ok=True)
    torch.save(first["model"].state_dict(), os.path.join(out, "models", "world_model_seed0.pt"))
    torch.save(first["policy"].state_dict(), os.path.join(out, "models", "policy_seed0.pt"))

    # summary.json ---------------------------------------------------------------
    cem, pol, rnd = (summary[k] for k in ("CEM MPC", "Distilled policy", "Random"))
    headline = {
        "speedup_policy_vs_cem": cem["decision_time_ms_mean"] / pol["decision_time_ms_mean"],
        "speedup_policy_vs_random_shooting":
            summary["Random-shooting MPC"]["decision_time_ms_mean"] / pol["decision_time_ms_mean"],
        # 1.0 = as good as CEM, 0.0 = as bad as random
        "policy_normalized_score": (pol["return_mean"] - rnd["return_mean"])
                                   / (cem["return_mean"] - rnd["return_mean"]),
        "policy_return_gap_vs_cem": pol["return_mean"] - cem["return_mean"],
    }
    pred = {label: e for label, e in first["pred_err"]}
    result = {
        "config": {k: v for k, v in vars(cfg).items()},
        "controllers": summary,
        "headline": headline,
        "world_model": {
            "seed0_final_losses": {label: {"train": h["train_loss"][-1], "val": h["val_loss"][-1]}
                                   for label, h in first["wm_histories"]},
            "seed0_prediction_error": pred,
        },
        "mbrl_rounds": {str(s): r.get("mbrl_rounds") for s, r in records.items()},
        "dagger": {str(s): r["dagger"] for s, r in records.items()},
        "gif_episode_returns_seed0": gif_returns,
        "torch_version": torch.__version__,
        "runtime_seconds": time.time() - t_start,
    }
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== final results (pooled over seeds) ===")
    for name in CONTROLLERS:
        s = summary[name]
        print(f"{name:20s} return {s['return_mean']:8.1f} +/- {s['return_std']:6.1f} "
              f"| {s['decision_time_ms_mean']:8.3f} ms/decision")
    print(f"policy is {headline['speedup_policy_vs_cem']:.0f}x faster than CEM, "
          f"normalised score {headline['policy_normalized_score']:.3f}")
    print(f"total runtime {result['runtime_seconds'] / 60:.1f} min -> {out}/")


if __name__ == "__main__":
    main()
