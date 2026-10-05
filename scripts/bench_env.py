#!/usr/bin/env python3
"""Environment throughput: steps/s for one PacmanEnv and for gymnasium.vector
(sync and async) with 4 and 8 copies, random actions.

    python scripts/bench_env.py [--seconds 5] [--sizes 4 8]
"""

import argparse
import os
import sys
import time

import gymnasium as gym
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.pacman_env import PacmanEnv  # noqa: E402


def bench_single(seconds):
    env = PacmanEnv(seed=0)
    try:
        env.reset(seed=0)
        rng = np.random.default_rng(0)
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            _, _, term, trunc, _ = env.step(int(rng.integers(5)))
            n += 1
            if term or trunc:
                env.reset()
        return n / (time.perf_counter() - t0)
    finally:
        env.close()


def bench_vector(cls, num, seconds):
    vec = cls([lambda i=i: PacmanEnv(seed=i) for i in range(num)])
    try:
        vec.reset(seed=0)
        vec.action_space.seed(0)
        for _ in range(10):  # warm up (async workers)
            vec.step(vec.action_space.sample())
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            vec.step(vec.action_space.sample())
            n += num
        return n / (time.perf_counter() - t0)
    finally:
        vec.close()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--sizes", type=int, nargs="+", default=[4, 8])
    args = p.parse_args()

    print(f"cpus: {os.cpu_count()}, {args.seconds:g} s per run, random actions")
    print(f"{'setup':<16}{'env steps/s':>12}")
    print(f"{'single':<16}{bench_single(args.seconds):>12,.0f}")
    for name, cls in (("sync", gym.vector.SyncVectorEnv), ("async", gym.vector.AsyncVectorEnv)):
        for num in args.sizes:
            print(f"{f'{name} x{num}':<16}{bench_vector(cls, num, args.seconds):>12,.0f}")


if __name__ == "__main__":
    main()
