#!/usr/bin/env python3
"""Greedy agreement of a DQN checkpoint with the demo actions of a DQfD demo
file (train.py dqfd.demo_file), and the margin by which Q(s, a_demo) leads.

    python scripts/demo_agreement.py runs/dqfd0/checkpoint.pt runs/dqfd_demo/demo.npz [--n 100000]
"""

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from train_eval import Scorer  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("checkpoint")
    p.add_argument("demo")
    p.add_argument("--n", type=int, default=100000)
    args = p.parse_args()
    with np.load(args.demo) as z:
        valid = np.flatnonzero(z["valid"])
        idx = np.sort(np.random.default_rng(0).choice(valid, min(args.n, len(valid)), replace=False))
        obs, vec, act = z["obs"][idx], z["vec"][idx], z["act"][idx]
    sc = Scorer(args.checkpoint, "cuda" if torch.cuda.is_available() else "cpu")
    q = np.concatenate([sc.q_values(obs[i:i + 4096], vec[i:i + 4096])
                        for i in range(0, len(idx), 4096)])
    qa = q[np.arange(len(act)), act]
    other = q.copy()
    other[np.arange(len(act)), act] = -np.inf
    lead = qa - other.max(1)
    print(f"{args.checkpoint}: agreement {np.mean(q.argmax(1) == act):.4f} on {len(idx)} demo "
          f"states; Q(s, a_demo) - best other: median {np.median(lead):.2f}, "
          f"share >= 0.8 {np.mean(lead >= 0.8):.3f}; mean Q {q.max(1).mean():.1f}")


if __name__ == "__main__":
    main()
