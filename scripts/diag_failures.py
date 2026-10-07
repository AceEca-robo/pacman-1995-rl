#!/usr/bin/env python3
"""Why level 1 is lost: deaths vs getting stuck, for a checkpoint.

    python scripts/diag_failures.py --checkpoint runs/dqn8/best_food.pt --name dqn8

Plays greedy games on seeds 0..games-1 (default env, no prefixes) until level
1 ends: cleared, game over, or the step limit. For every life lost on level 1
it records where pacman was, where the ghosts were (and their states), the
food left; for games that end on level 1 it records the outcome (died: game
over; stuck: step limit), the food cells left and how long pacman went
without eating at the end. Writes runs/diag/failures_<name>.json and
docs/<name>_failures.png (food left in the failed games; where lives were
lost on level 1, all games).
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from env.pacman_env import HEIGHT, WIDTH, PacmanEnv, load_env_config, maze_layouts  # noqa: E402


def food_cells(state):
    return [(x, y) for y, row in enumerate(state["grid"]) for x, ch in enumerate(row) if ch in ".o"]


def play(agent, seeds):
    env = PacmanEnv(config={**load_env_config(None), **agent.env_overrides})
    games = []
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            agent.reset(seed=seed)
            deaths, steps, last_food, still = [], 0, 0, 0
            prev_pos = None
            while True:
                obs, _, term, trunc, info = env.step(agent.act(obs))
                steps += 1
                st, ev = env._state, info["events"]
                if ev["eaten_dot"] or ev["eaten_energizer"]:
                    last_food = steps
                pos = (st["pacman"]["x"], st["pacman"]["y"])
                still = still + 1 if pos == prev_pos else 0
                prev_pos = pos
                if info["level"] > 1:
                    outcome = "cleared"
                    break
                if ev["death"]:
                    ghosts = [{"x": g["x"], "y": g["y"], "state": g["state"]} for g in st["ghosts"]]
                    near = [abs(g["x"] - pos[0]) + abs(g["y"] - pos[1]) for g in ghosts
                            if g["state"] == "normal"]
                    deaths.append({"step": steps, "cell": pos, "ghosts": ghosts,
                                   "nearest_normal_ghost": min(near) if near else None,
                                   "food_left": len(food_cells(st)), "lives_left": st["lives"]})
                if term or trunc:
                    outcome = "died" if term else "stuck"
                    break
            left = [] if outcome == "cleared" else food_cells(env._state)
            games.append({"seed": seed, "outcome": outcome, "steps": steps, "deaths": deaths,
                          "food_left": len(left), "left_cells": left,
                          "steps_since_food": steps - last_food, "standing_at_end": still})
    finally:
        env.close()
    return games


def plot(games, name, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    walls = np.array([[c == "#" for c in row] for row in maze_layouts()[0]])
    wall_rgba = np.zeros((HEIGHT, WIDTH, 4))
    wall_rgba[walls] = (0.55, 0.55, 0.6, 1.0)
    failed = [g for g in games if g["outcome"] != "cleared"]
    left = np.zeros((HEIGHT, WIDTH))
    for g in failed:
        for x, y in g["left_cells"]:
            left[y, x] += 1
    died = np.zeros((HEIGHT, WIDTH))
    for g in games:
        for d in g["deaths"]:
            died[d["cell"][1], d["cell"][0]] += 1
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, data, cmap, title in (
            (axes[0], left, "Reds", f"food left at the end, {len(failed)} failed games (count)"),
            (axes[1], died, "Purples", f"lives lost on level 1, all {len(games)} games (count)")):
        im = ax.imshow(np.ma.masked_where(data == 0, data), cmap=cmap, vmin=0,
                       vmax=max(1, data.max()))
        ax.imshow(wall_rgba)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(0, WIDTH, 4))
        ax.set_yticks(range(0, HEIGHT, 4))
        fig.colorbar(im, ax=ax, fraction=0.03)
    n = Counter(g["outcome"] for g in games)
    fig.suptitle(f"{name}: {len(games)} greedy games, cleared {n['cleared']}, "
                 f"died {n['died']}, stuck {n['stuck']}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--seed", type=int, default=0, help="first seed")
    args = p.parse_args()
    from agents.ppo import load_agent
    agent = load_agent(args.checkpoint)
    games = play(agent, range(args.seed, args.seed + args.games))
    plot(games, args.name, os.path.join(ROOT, "docs", f"{args.name}_failures.png"))
    os.makedirs(os.path.join(ROOT, "runs", "diag"), exist_ok=True)
    with open(os.path.join(ROOT, "runs", "diag", f"failures_{args.name}.json"), "w") as f:
        json.dump(games, f)

    failed = [g for g in games if g["outcome"] != "cleared"]
    n = Counter(g["outcome"] for g in games)
    print(f"{args.name}: cleared {n['cleared']}, died {n['died']}, stuck {n['stuck']}")
    for g in failed:
        d = "; ".join(f"step {x['step']} at {tuple(x['cell'])}, nearest normal ghost "
                      f"{x['nearest_normal_ghost']}, food left {x['food_left']}" for x in g["deaths"])
        print(f"  seed {g['seed']:2d} {g['outcome']:5s} food left {g['food_left']:3d}, "
              f"{g['steps_since_food']} steps since food, standing {g['standing_at_end']}; "
              f"deaths: {d or '-'}")
    cells = Counter(tuple(c) for g in failed for c in g["left_cells"])
    common = [(c, k) for c, k in cells.most_common() if k >= 5]
    print("cells with food left in >= 5 failed games:", common or "none")
    all_deaths = [d for g in games for d in g["deaths"]]
    print(f"lives lost on level 1: {len(all_deaths)} in all games, "
          f"{sum(len(g['deaths']) for g in failed)} in the failed ones; most common cells:",
          Counter(tuple(d["cell"]) for d in all_deaths).most_common(6))


if __name__ == "__main__":
    main()
