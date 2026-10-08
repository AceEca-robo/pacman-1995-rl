#!/usr/bin/env python3
"""Time the parts of one train.py iteration (stage A of the 2026-10-08 night).

    python scripts/profile_train.py --config configs/dqn_v03.yaml --iters 2000

Builds the same envs / learner / buffer as train.py, fills the buffer with
random play, then times per iteration: the vector env step (game + obs),
obs packing, the batched greedy inference, n-step bookkeeping + buffer add,
PER sampling and the learner update (per update, and per env step at the
config's train_every). Prints ms per iteration and the implied steps/s.
"""

import argparse
import os
import sys
import time

import gymnasium as gym
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from agents.dqn import DQNLearner, NStep, make_buffer, pack_grids  # noqa: E402
from env.pacman_env import PacmanEnv, load_env_config, obs_dims  # noqa: E402
from train import load_config  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "dqn_v03.yaml"))
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--num-envs", type=int, default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    n = args.num_envs or cfg["num_envs"]
    env_cfg = load_env_config(cfg["env_config"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_channels, vec_dim = obs_dims(env_cfg)
    learner = DQNLearner(cfg, device, env_cfg["actions"], n_channels, vec_dim)
    buffer = make_buffer({**cfg, "buffer_size": 100000}, n_channels, vec_dim)
    envs = gym.vector.SyncVectorEnv([lambda: PacmanEnv(config=env_cfg)] * n,
                                    autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    rng = np.random.default_rng(0)
    obs, _ = envs.reset(seed=0)
    nsteps = [NStep(cfg["n_step"], cfg["gamma"]) for _ in range(n)]
    cur = pack_grids(obs["grid"])
    t = dict.fromkeys(["env_step", "pack", "infer", "nstep_add", "sample", "update"], 0.0)
    timer = time.perf_counter
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)

    for it in range(args.iters + 200):
        if it == 200:  # warm-up done (compile, caches)
            t = dict.fromkeys(t, 0.0)
        t0 = timer()
        q = learner.q_values(cur, obs["vec"])
        actions = np.where(rng.random(n) < 0.1, rng.integers(5, size=n), q.argmax(1))
        t1 = timer()
        nobs, rew, term, trunc, info = envs.step(actions)
        t2 = timer()
        nxt = pack_grids(nobs["grid"])
        t3 = timer()
        for i in range(n):
            for tr in nsteps[i].push(cur[i], obs["vec"][i], actions[i], rew[i], nxt[i],
                                     nobs["vec"][i], term[i], term[i] or trunc[i]):
                buffer.add(*tr)
        t4 = timer()
        cur, obs = nxt, nobs
        t["infer"] += t1 - t0
        t["env_step"] += t2 - t1
        t["pack"] += t3 - t2
        t["nstep_add"] += t4 - t3
    n_upd = 300
    for k in range(n_upd + 20):
        if k == 20:
            sync()
            t["sample"] = t["update"] = 0.0
        t0 = timer()
        batch, idx, w = buffer.sample(rng, cfg["batch_size"], 0.4)
        t1 = timer()
        _, _, td = learner.update(batch, w)
        buffer.update_priorities(idx, td.cpu().numpy())  # syncs
        t2 = timer()
        t["sample"] += t1 - t0
        t["update"] += t2 - t1
    envs.close()
    per_it = {k: v / args.iters * 1e3 for k, v in t.items() if k not in ("sample", "update")}
    upd_per_it = n / cfg["train_every"]
    per_it["sample"] = t["sample"] / n_upd * 1e3 * upd_per_it
    per_it["update"] = t["update"] / n_upd * 1e3 * upd_per_it
    total = sum(per_it.values())
    print(f"num_envs {n}, train_every {cfg['train_every']}, batch {cfg['batch_size']}, "
          f"network {cfg['network']['conv_channels']} / {cfg['network']['hidden']}")
    for k, v in per_it.items():
        print(f"  {k:<10} {v:7.3f} ms/iter  {v / total:5.1%}")
    print(f"  per update: sample {t['sample'] / n_upd * 1e3:.2f} ms, "
          f"update {t['update'] / n_upd * 1e3:.2f} ms")
    print(f"  total {total:.3f} ms/iter -> {n / total * 1e3:,.0f} steps/s")


if __name__ == "__main__":
    main()
