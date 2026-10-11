#!/usr/bin/env python3
"""Table and curve of a long DQfD run against BC and the heuristic on the
run's eval protocol (20 games per maze + 20 from level 1, seeds 1000..1019).

    python scripts/dqfd_report.py dqfd0 --plot docs/dqfd0_curve.png

Rows: the supervisor's evaluations (runs/<run>/sv_evals.jsonl, every 10M);
references: BC bc0_dagger1 (runs/dqfd_demo/protocol_evals.jsonl) and the
heuristic (runs/ref_seed1000/heuristic.json, evaluate.py --levels on the
same seeds). The plot adds the training's own evals (evals.jsonl, every 2M).
"""

import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")


def jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def references():
    bc = jsonl(os.path.join(RUNS, "dqfd_demo", "protocol_evals.jsonl"))[0]
    with open(os.path.join(RUNS, "ref_seed1000", "heuristic.json")) as f:
        h = json.load(f)
    shares = [v["cleared"] for v in h["per_level"].values()]
    heur = {"mean_maze_cleared": float(np.mean(shares)), "min_maze_cleared": min(shares),
            "mean_levels_cleared": h["from_level1"]["mean_levels_cleared"],
            "mean_food": float(np.mean([g["food_eaten"] for g in h["games"]
                                        if g["start_level"] == 1]))}
    return {"BC bc0_dagger1": bc, "heuristic": heur}


def table(run):
    rows = jsonl(os.path.join(RUNS, run, "sv_evals.jsonl"))
    out = ["| | per-maze mean | per-maze min | levels from level 1 | food from level 1 |",
           "|---|---|---|---|---|"]

    def line(name, r):
        food = f"{r['mean_food']:.0f}" if r.get("mean_food") is not None else "-"
        return (f"| {name} | {r['mean_maze_cleared']:.1%} | {r['min_maze_cleared']:.0%} "
                f"| {r['mean_levels_cleared']:.2f} | {food} |")
    for r in sorted(rows, key=lambda r: r["steps"]):
        out.append(line(f"{run} @ {round(r['steps'] / 1e6)}M", r))
    for name, r in references().items():
        out.append(line(name, r))
    return "\n".join(out)


def plot(run, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    train = [r for r in jsonl(os.path.join(RUNS, run, "evals.jsonl")) if "phase" not in r]
    sv = sorted(jsonl(os.path.join(RUNS, run, "sv_evals.jsonl")), key=lambda r: r["steps"])
    refs = references()
    blue, orange, aqua, ink, muted = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#8a8984"
    panels = [("mean_maze_cleared", "per-maze clear share, mean of 16", True),
              ("min_maze_cleared", "per-maze clear share, minimum", True),
              ("mean_levels_cleared", "levels cleared from level 1 (mean)", False)]
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 9.5), sharex=True)
    for ax, (key, title, pct) in zip(axes, panels):
        x = [r["steps"] / 1e6 for r in train]
        ax.plot(x, [r[key] for r in train], color=blue, lw=1, alpha=0.45,
                label="training eval (every 2M)")
        ax.plot([r["steps"] / 1e6 for r in sv], [r[key] for r in sv], color=blue, lw=2,
                marker="o", ms=6, label="supervisor eval (every 10M)")
        for (name, r), c in zip(refs.items(), (orange, aqua)):
            ax.axhline(r[key], color=c, lw=1.5, ls="--", label=name)
            ax.annotate(name.split()[0], (1.0, r[key]), xycoords=("axes fraction", "data"),
                        xytext=(4, 0), textcoords="offset points", va="center", color=ink,
                        fontsize=8)
        ax.set_title(title, loc="left", fontsize=10, color=ink)
        if pct:
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.grid(axis="y", color="#e4e3df", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(muted)
        ax.tick_params(colors=muted, labelsize=8)
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    axes[-1].set_xlabel("env steps of phase 1 (M)", fontsize=9, color=ink)
    fig.suptitle(f"{run}: DQfD phase 1, eval protocol seeds 1000..1019 (20 games each)",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run")
    p.add_argument("--plot", help="PNG path")
    args = p.parse_args()
    print(table(args.run))
    if args.plot:
        plot(args.run, args.plot)


if __name__ == "__main__":
    main()
