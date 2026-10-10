#!/usr/bin/env python3
"""Compare short runs of scripts/train.py (part A smoke tests, 2026-10-10):
training episode reward per bin of env steps, the evals, and speed.

    python scripts/compare_smoke.py smoke2m_thread smoke2m_pool64 smoke2m_pool128 \
        [--bin 250000] [--plot docs/smoke_learner_process.png]

Speed: mean of the perf/steps_per_sec log points from --speed-from steps on.
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from plot_runs import load_scalars  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs", nargs="+")
    p.add_argument("--bin", type=int, default=250000)
    p.add_argument("--speed-from", type=int, default=300000)
    p.add_argument("--plot", default=None)
    args = p.parse_args()
    data = {r: load_scalars(os.path.join(ROOT, "runs", r)) for r in args.runs}
    top = max(int(d["episode/reward"][0].max()) for d in data.values())
    edges = list(range(0, top + args.bin, args.bin))
    print("episode reward, mean per bin of env steps (episodes ending in the bin)")
    print("| run | " + " | ".join(f"{a / 1e6:.2f}-{b / 1e6:.2f}M" for a, b in zip(edges, edges[1:]))
          + " | steps/s |")
    print("|---" * (len(edges) + 1) + "|")
    for r, d in data.items():
        s, v = d["episode/reward"]
        cells = [f"{v[(s >= a) & (s < b)].mean():.1f}" if ((s >= a) & (s < b)).any() else "-"
                 for a, b in zip(edges, edges[1:])]
        ss, sv = d["perf/steps_per_sec"]
        speed = sv[ss >= args.speed_from].mean()
        print(f"| {r} | " + " | ".join(cells) + f" | {speed:,.0f} |")
    print("\nevals: steps, food from level 1, mean per-maze clear share, levels from level 1")
    for r in args.runs:
        with open(os.path.join(ROOT, "runs", r, "evals.jsonl")) as f:
            rows = [json.loads(line) for line in f if '"phase"' not in line]
        print(r + ": " + "; ".join(
            f"{e['steps'] / 1e6:.2f}M {e['mean_food']:.0f} / "
            f"{np.mean([e[k] for k in e if k.startswith('maze') and k.endswith('_cleared')]):.3f}"
            f" / {e['mean_levels_cleared']:.2f}" for e in rows))
    if args.plot:
        fig, ax = plt.subplots(figsize=(7, 4))
        for r, d in data.items():
            s, v = d["episode/reward"]
            mids = [(a + b) / 2 for a, b in zip(edges, edges[1:]) if ((s >= a) & (s < b)).any()]
            means = [v[(s >= a) & (s < b)].mean() for a, b in zip(edges, edges[1:])
                     if ((s >= a) & (s < b)).any()]
            ax.plot(np.array(mids) / 1e6, means, marker="o", label=r)
        ax.set_xlabel("env steps (M)")
        ax.set_ylabel(f"episode reward (mean per {args.bin / 1e3:.0f}k steps)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(ROOT, args.plot), dpi=120)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
