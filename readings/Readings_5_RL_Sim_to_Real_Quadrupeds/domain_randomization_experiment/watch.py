"""
watch.py - watch a trained controller walk on any test condition.

    python watch.py --model fixed --condition low_friction
    python watch.py --model randomized --condition low_friction
    python watch.py --model fixed --condition pseudo_real --record     # save an MP4 instead

The live view uses mujoco.viewer.launch_passive (works on Windows). If the viewer cannot
open (OpenGL problems, remote desktop, ...), the script falls back to recording an MP4 with
mujoco.Renderer + imageio into results/videos/.
"""
import argparse
import time

import numpy as np
import torch

import config as C
from envs.walker_env import WalkerEnv
from evaluate import find_runs, load_policy


def pick_run(mode: str, seed):
    runs = [r for r in find_runs() if r[0] == mode and (seed is None or r[1] == seed)]
    if not runs:
        raise SystemExit(f"No trained '{mode}' model found in runs/. Train one first.")
    return runs[0]


def policy_action(model, vecnorm, obs):
    action, _ = model.predict(vecnorm.normalize_obs(obs[None]), deterministic=True)
    return action[0]


def watch_live(model, vecnorm, env, episodes: int, ep_seed: int):
    import mujoco.viewer
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = env.torso_id
        viewer.cam.distance = 2.5
        viewer.cam.elevation = -10
        viewer.cam.azimuth = 90
        for ep in range(episodes):
            obs, _ = env.reset(seed=ep_seed + ep)
            done, info = False, {}
            while viewer.is_running() and not done:
                t0 = time.time()
                obs, _, term, trunc, info = env.step(policy_action(model, vecnorm, obs))
                done = term or trunc
                viewer.sync()
                time.sleep(max(0.0, C.CONTROL_DT - (time.time() - t0)))   # real time
            if not viewer.is_running():
                break
            print(f"episode {ep}: distance {info['distance']:.2f} m, "
                  f"{'FELL' if info['fell'] else 'stayed up'}, success={info['success']}")
            time.sleep(1.0)


def record_mp4(model, vecnorm, env, out_path, ep_seed: int):
    import imageio
    frames = []
    obs, _ = env.reset(seed=ep_seed)
    done, info = False, {}
    while not done:
        obs, _, term, trunc, info = env.step(policy_action(model, vecnorm, obs))
        done = term or trunc
        frames.append(env.render())
    for _ in range(25):                    # hold the last frame for half a second
        frames.append(frames[-1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=int(round(1 / C.CONTROL_DT)))
    print(f"Saved {out_path}  (distance {info['distance']:.2f} m, "
          f"{'FELL' if info['fell'] else 'stayed up'}, success={info['success']})")


if __name__ == "__main__":
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description="Watch a trained controller")
    parser.add_argument("--model", choices=C.ALL_MODES, default="randomized")
    all_conditions = {**C.TEST_CONDITIONS, **C.PROBE_CONDITIONS}
    parser.add_argument("--condition", choices=list(all_conditions), default="nominal")
    parser.add_argument("--seed", type=int, default=None, help="training seed of the model (default: first found)")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--episode-seed", type=int, default=C.EVAL_SEED)
    parser.add_argument("--record", action="store_true", help="record an MP4 instead of opening the viewer")
    args = parser.parse_args()

    mode, seed, run_dir = pick_run(args.model, args.seed)
    model, vecnorm = load_policy(run_dir)
    physics = all_conditions[args.condition]
    env = WalkerEnv(physics=physics)
    print(f"Model: {run_dir.name} | condition: {args.condition} {physics}")
    out = C.VIDEOS_DIR / f"{mode}_seed{seed}_{args.condition}.mp4"

    if args.record:
        record_mp4(model, vecnorm, env, out, args.episode_seed)
    else:
        try:
            watch_live(model, vecnorm, env, args.episodes, args.episode_seed)
        except Exception as e:   # viewer failed (OpenGL / display problem) -> MP4 fallback
            print(f"Viewer could not open ({e!r}). Falling back to MP4 recording...")
            record_mp4(model, vecnorm, WalkerEnv(physics=physics), out, args.episode_seed)
    env.close()
