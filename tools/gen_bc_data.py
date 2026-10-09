#!/usr/bin/env python3
"""Behaviour cloning data: the heuristic agent plays games started on a random
maze of 1..16 (env seed and start level drawn per game), executing a random
action with probability --epsilon; every step stores the observation and the
heuristic's action (the label, also when a random action was executed).

    python tools/gen_bc_data.py --steps 3000000 --seed-start 100000 --out runs/bc_data/train
    python tools/gen_bc_data.py --steps 100000 --seed-start 900000 --out runs/bc_data/val
    python tools/gen_bc_data.py --checkpoint runs/bc0/bc_best.pt --steps 500000 \
        --seed-start 300000 --out runs/bc_data/dagger1     # DAgger: BC plays, heuristic labels

Grids are bit-packed as in the replay buffer (agents/dqn.py pack_grids).
Each worker writes <out>_wNN.npz with obs uint8 (N, planes, PACKED),
vec float32 (N, 3), act uint8 (N,) = heuristic labels, level uint8 (N,) =
the maze being played (1..16; later levels repeat boards randomly).
Worker w plays seeds seed_start + w * 100000 + k, k = 0, 1, ...
"""

import argparse
import multiprocessing
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agents.dqn import n_planes, pack_grid, PACKED  # noqa: E402
from agents.heuristic_agent import HeuristicAgent  # noqa: E402
from env.pacman_env import CHANNELS, LEVELS, PacmanEnv  # noqa: E402

SEED_STRIDE = 100000


def worker(args):
    w, steps, a = args
    import torch
    torch.set_num_threads(1)
    teacher = HeuristicAgent(n_actions=5)
    player = None
    if a.checkpoint:  # DAgger: the BC policy acts
        from agents.bc import BCAgent
        player = BCAgent(a.checkpoint, device="cpu")
    env = PacmanEnv(config={"actions": 5, "start_level": 1,
                            "max_episode_steps": a.max_episode_steps})
    rng = np.random.default_rng([a.seed_start, w])
    obs_buf = np.zeros((steps, n_planes(len(CHANNELS)), PACKED), np.uint8)
    vec_buf = np.zeros((steps, 3), np.float32)
    act_buf = np.zeros(steps, np.uint8)
    lvl_buf = np.zeros(steps, np.uint8)
    i, games, t0 = 0, 0, time.perf_counter()
    try:
        while i < steps:
            seed = a.seed_start + w * SEED_STRIDE + games
            level = int(rng.integers(1, LEVELS + 1))
            obs, info = env.reset(seed=seed, options={"start_level": level})
            if player:
                player.reset(seed=seed)
            games += 1
            while i < steps:
                label = teacher.act(obs)
                obs_buf[i] = pack_grid(obs["grid"])
                vec_buf[i] = obs["vec"]
                act_buf[i] = label
                lvl_buf[i] = (info["level"] - 1) % LEVELS + 1
                i += 1
                if rng.random() < a.epsilon:
                    action = int(rng.integers(5))
                else:
                    action = player.act(obs) if player else label
                obs, _, term, trunc, info = env.step(action)
                if term or trunc:
                    break
    finally:
        env.close()
    path = f"{a.out}_w{w:02d}.npz"
    np.savez_compressed(path, obs=obs_buf, vec=vec_buf, act=act_buf, level=lvl_buf)
    return w, games, time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--steps", type=int, required=True, help="transitions in total")
    p.add_argument("--seed-start", type=int, required=True)
    p.add_argument("--out", required=True, help="path prefix of the shards")
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--checkpoint", help="BC checkpoint that acts instead of the heuristic (DAgger)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--max-episode-steps", type=int, default=10000)
    a = p.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    per = [a.steps // a.workers + (w < a.steps % a.workers) for w in range(a.workers)]
    t0 = time.perf_counter()
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(a.workers) as pool:
        for w, games, dt in pool.imap_unordered(worker, [(w, per[w], a) for w in range(a.workers)]):
            print(f"worker {w}: {per[w]} steps, {games} games, {dt:.0f} s", flush=True)
    print(f"{a.steps} steps in {time.perf_counter() - t0:.0f} s -> {a.out}_w*.npz")


if __name__ == "__main__":
    main()
