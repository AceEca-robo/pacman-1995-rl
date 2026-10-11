#!/usr/bin/env python3
"""Final-protocol table with 95% bootstrap intervals.

    python scripts/final_table.py runs/v03_final/heuristic.json runs/v03_final/dqfd0@80M.json ...

Reads per-game results of `train_eval.py --games-out` (games with "maze" for
the per-maze games) or `evaluate.py --levels` (games with "start_level"; the
level-1 games are shared by both sets there, the per-maze game on maze 1 is
the same game cut at its clear). Per agent: mean and minimum over the 16
mazes of the share that clears the start maze, levels cleared in a row from
level 1 (mean / median / max). Intervals: 2000 bootstrap resamples of the
games within each maze (and of the level-1 games), percentile 2.5..97.5.
"""

import argparse
import json
import os

import numpy as np


def load(path):
    with open(path) as f:
        d = json.load(f)
    label = d.get("label") or d.get("agent") or os.path.basename(path)
    maze, from1 = {}, []
    if any("start_level" in g for g in d["games"]):  # evaluate.py --levels
        n = d["per_level"]["1"]["episodes"] if "1" in d["per_level"] else None
        by = {}
        for g in d["games"]:
            by.setdefault(g["start_level"], []).append(g)
        for lv, gs in by.items():
            gs = sorted(gs, key=lambda g: g["seed"])
            maze[lv] = np.array([g["levels_cleared"] >= 1 for g in gs[:n]], float)
        from1 = [g["levels_cleared"] for g in sorted(by[1], key=lambda g: g["seed"])
                 [:d["from_level1"]["episodes"]]]
    else:  # train_eval.py --games-out
        for g in d["games"]:
            if "maze" in g:
                maze.setdefault(g["maze"], []).append(g["levels_cleared"] >= 1)
            else:
                from1.append(g["levels_cleared"])
        maze = {lv: np.array(v, float) for lv, v in maze.items()}
    return label, maze, np.array(from1, float)


def ci(samples):
    return np.percentile(samples, [2.5, 97.5])


def stats(maze, from1, rng, n_boot=2000):
    mazes = sorted(maze)
    shares = np.array([maze[lv].mean() for lv in mazes])
    boot = np.array([[rng.choice(maze[lv], len(maze[lv])).mean() for lv in mazes]
                     for _ in range(n_boot)])
    lv_boot = np.array([rng.choice(from1, len(from1)) for _ in range(n_boot)])
    return {
        "mean": shares.mean(), "mean_ci": ci(boot.mean(1)),
        "min": shares.min(), "min_ci": ci(boot.min(1)), "argmin": mazes[int(shares.argmin())],
        "levels_mean": from1.mean(), "levels_mean_ci": ci(lv_boot.mean(1)),
        "levels_median": float(np.median(from1)), "levels_max": int(from1.max()),
        "n_maze": int(np.mean([len(maze[lv]) for lv in mazes])), "n_from1": len(from1),
        "shares": dict(zip(mazes, shares)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("files", nargs="+")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()
    rng = np.random.default_rng(0)
    rows = []
    for path in args.files:
        label, maze, from1 = load(path)
        rows.append((label, stats(maze, from1, rng)))
    print("| agent | games | per-maze mean [95% CI] | min [95% CI] (maze) "
          "| levels in a row: mean [95% CI] / median / max |")
    print("|---|---|---|---|---|")
    for label, s in rows:
        print(f"| {label} | {s['n_maze']} per maze, {s['n_from1']} from level 1 "
              f"| {s['mean']:.1%} [{s['mean_ci'][0]:.1%}, {s['mean_ci'][1]:.1%}] "
              f"| {s['min']:.0%} [{s['min_ci'][0]:.0%}, {s['min_ci'][1]:.0%}] ({s['argmin']}) "
              f"| {s['levels_mean']:.2f} [{s['levels_mean_ci'][0]:.2f}, "
              f"{s['levels_mean_ci'][1]:.2f}] / {s['levels_median']:g} / {s['levels_max']} |")
    mazes = sorted(rows[0][1]["shares"])
    print()
    print("| agent | " + " | ".join(str(m) for m in mazes) + " |")
    print("|---|" + "---|" * len(mazes))
    for label, s in rows:
        print(f"| {label} | " + " | ".join(f"{s['shares'][m]:.0%}" for m in mazes) + " |")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({label: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                               for k, v in s.items() if k != "shares"}
                       | {"shares": {str(k): float(v) for k, v in s["shares"].items()}}
                       for label, s in rows}, f, indent=1, default=float)


if __name__ == "__main__":
    main()
