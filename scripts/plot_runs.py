#!/usr/bin/env python3
"""Plots for the README from TensorBoard logs of scripts/train.py runs.

    python scripts/plot_runs.py dqn0 [more runs...] [--out-dir docs] [--smooth 200]

Writes <out-dir>/training.png (episode reward and score over env steps,
rolling median over --smooth episodes) and <out-dir>/eval.png (median eval
reward, median eval score and share of eval episodes clearing level 1 at
each checkpoint). Baselines from docs/results.md are drawn as dashed lines.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_COLUMNS = {"reward": "median reward", "score": "median score"}


def load_scalars(run_dir):
    acc = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    acc.Reload()
    out = {}
    for tag in acc.Tags()["scalars"]:
        events = acc.Scalars(tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])
        order = np.argsort(steps, kind="stable")  # resumed runs may log out of order
        out[tag] = (steps[order], values[order])
    return out


def rolling_median(values, window):
    if len(values) < window:
        window = max(1, len(values))
    return np.array([np.median(values[max(0, i - window + 1):i + 1]) for i in range(len(values))])


def load_baselines(path):
    """{"heuristic": {"median reward": 401.2, ...}, ...} from docs/results.md."""
    if not os.path.exists(path):
        return {}
    header, rows = None, {}
    with open(path) as f:
        for line in f:
            if not line.startswith("| "):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cells[0] == "agent":
                header = cells
            elif header and cells[0] in ("random", "heuristic"):
                rows[cells[0]] = dict(zip(header, cells))
    return {name: {k: float(v) for k, v in row.items() if k in BASELINE_COLUMNS.values()}
            for name, row in rows.items()}


def draw_baselines(ax, baselines, column):
    for name, style in (("heuristic", "--"), ("random", ":")):
        if name in baselines and column in baselines[name]:
            ax.axhline(baselines[name][column], color="grey", linestyle=style, linewidth=1,
                       label=f"{name} (median, 100 episodes)")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs", nargs="+", help="run names under runs/")
    p.add_argument("--out-dir", default=os.path.join(ROOT, "docs"))
    p.add_argument("--smooth", type=int, default=200, help="episodes in the rolling median")
    args = p.parse_args()

    data = {r: load_scalars(os.path.join(ROOT, "runs", r)) for r in args.runs}
    baselines = load_baselines(os.path.join(ROOT, "docs", "results.md"))
    os.makedirs(args.out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, (tag, key, title) in zip(axes, (("episode/reward", "reward", "Episode reward"),
                                            ("episode/score", "score", "Game score"))):
        for run, sc in data.items():
            if tag in sc:
                steps, values = sc[tag]
                ax.plot(steps, values, alpha=0.12, linewidth=0.5)
                ax.plot(steps, rolling_median(values, args.smooth), linewidth=1.6,
                        color=ax.lines[-1].get_color(), label=f"{run} (rolling median)")
        draw_baselines(ax, baselines, BASELINE_COLUMNS[key])
        ax.set_title(f"{title}, training episodes")
        ax.set_xlabel("env steps")
        ax.xaxis.set_major_formatter(matplotlib.ticker.EngFormatter())
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    if baselines.get("heuristic"):
        axes[1].set_yscale("symlog", linthresh=100)  # ghost chains give a heavy tail
        axes[1].set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "training.png"), dpi=120)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    panels = (("eval/median_reward", "reward", "Median eval reward"),
              ("eval/median_score", "score", "Median eval score"),
              ("eval/level1_cleared", None, "Eval episodes clearing level 1"))
    for ax, (tag, key, title) in zip(axes, panels):
        for run, sc in data.items():
            if tag in sc:
                ax.plot(*sc[tag], marker="o", markersize=3, label=run)
        if key:
            draw_baselines(ax, baselines, BASELINE_COLUMNS[key])
        else:
            ax.set_ylim(-0.05, 1.05)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(title)
        ax.set_xlabel("env steps")
        ax.xaxis.set_major_formatter(matplotlib.ticker.EngFormatter())
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "eval.png"), dpi=120)
    print("wrote", os.path.join(args.out_dir, "training.png"), os.path.join(args.out_dir, "eval.png"))


if __name__ == "__main__":
    main()
