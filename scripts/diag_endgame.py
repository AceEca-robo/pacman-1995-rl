#!/usr/bin/env python3
"""Where level 1 is left unfinished: leftover food and pacman's visits.

    python scripts/diag_endgame.py --agent heuristic [--games 100]
    python scripts/diag_endgame.py --agent dqn --checkpoint runs/dqn1s1/best_food.pt \\
        --epsilon 0.05 --name dqn1s1_eps005

Plays games on seeds 0..games-1 (no shaping, hunger or prefixes) only until
level 1 ends: level cleared, game over or the step limit. Per cell it counts
in how many games food was still there at that point, and how many steps
pacman spent there on level 1. Writes runs/diag/endgame_<name>.json and
docs/leftover_<name>.png (left: share of games with food left in the cell;
right: pacman's visits, log scale).
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agents.heuristic_agent import HeuristicAgent  # noqa: E402
from env.pacman_env import HEIGHT, WIDTH, PacmanEnv, load_env_config  # noqa: E402

POCKET = [(25, 19), (27, 19)]  # the two dots dqn1s1 never ate (x, y)


def play(agent, games, env_overrides):
    env = PacmanEnv(config={**load_env_config(None), **env_overrides})
    left = np.zeros((HEIGHT, WIDTH), int)
    visits = np.zeros((HEIGHT, WIDTH), int)
    per_game = []
    try:
        for seed in range(games):
            obs, _ = env.reset(seed=seed)
            agent.reset(seed=seed)
            start_food = None
            steps = 0
            while True:
                st = env._state
                if start_food is None:
                    start_food = sum(r.count(".") + r.count("o") for r in st["grid"])
                p = st["pacman"]
                visits[p["y"], p["x"]] += 1
                last_level1 = st
                obs, _, term, trunc, info = env.step(agent.act(obs))
                steps += 1
                if info["level"] > 1 or term or trunc:
                    break
            cleared = info["level"] > 1
            grid = None if cleared else env._state["grid"]
            if grid is None:
                remaining = 0
            else:
                remaining = 0
                for y, row in enumerate(grid):
                    for x, ch in enumerate(row):
                        if ch in ".o":
                            left[y, x] += 1
                            remaining += 1
            per_game.append({"seed": seed, "cleared": cleared, "food_left": remaining,
                             "steps": steps, "lives_left": int(last_level1["lives"]),
                             "pocket_left": [bool(grid and grid[y][x] in ".o") for x, y in POCKET]})
    finally:
        env.close()
    return left, visits, per_game


def plot(left, visits, walls, games, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    wall_rgba = np.zeros((HEIGHT, WIDTH, 4))
    wall_rgba[walls] = (0.55, 0.55, 0.6, 1.0)
    share = np.ma.masked_where(left == 0, left / games)
    im = axes[0].imshow(share, cmap="Reds", vmin=0, vmax=1)
    axes[0].imshow(wall_rgba)
    axes[0].set_title("share of games with food left here at the end of level 1")
    fig.colorbar(im, ax=axes[0], fraction=0.03)
    vis = np.ma.masked_where(visits == 0, visits)
    im = axes[1].imshow(vis, cmap="viridis", norm=LogNorm(vmin=1, vmax=max(2, visits.max())))
    axes[1].imshow(wall_rgba)
    axes[1].set_title("pacman's steps per cell on level 1 (log)")
    fig.colorbar(im, ax=axes[1], fraction=0.03)
    for ax in axes:
        for x, y in POCKET:
            ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, ec="black", lw=1.2))
        ax.set_xticks(range(0, WIDTH, 4))
        ax.set_yticks(range(0, HEIGHT, 4))
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--agent", choices=("heuristic", "dqn"), required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--epsilon", type=float, default=0.0)
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--name", help="output name (default: the agent)")
    args = p.parse_args()

    if args.agent == "heuristic":
        agent, overrides = HeuristicAgent(), {}
    else:
        from agents.dqn import DQNAgent
        agent = DQNAgent(args.checkpoint, epsilon=args.epsilon)
        overrides = agent.env_overrides
    name = args.name or args.agent
    left, visits, per_game = play(agent, args.games, overrides)

    from env.pacman_env import maze_layouts
    walls = np.array([[c == "#" for c in row] for row in maze_layouts()[0]])
    cleared = np.mean([g["cleared"] for g in per_game])
    pocket = [int(left[y, x]) for x, y in POCKET]
    pocket_visits = [int(visits[y, x]) for x, y in POCKET]
    title = (f"{name}: {args.games} games, level 1 cleared {cleared:.0%}, "
             f"mean food left {np.mean([g['food_left'] for g in per_game]):.1f}")
    plot(left, visits, walls, args.games, title, os.path.join(ROOT, "docs", f"leftover_{name}.png"))
    out = {"name": name, "agent": args.agent, "checkpoint": args.checkpoint,
           "epsilon": args.epsilon, "games": args.games, "level1_cleared": float(cleared),
           "pocket_cells": POCKET, "pocket_left_games": pocket, "pocket_visit_steps": pocket_visits,
           "top_left_cells": sorted(((int(left[y, x]), (x, y)) for y in range(HEIGHT)
                                     for x in range(WIDTH) if left[y, x]), reverse=True)[:10],
           "left": left.tolist(), "visits": visits.tolist(), "games_detail": per_game}
    os.makedirs(os.path.join(ROOT, "runs", "diag"), exist_ok=True)
    with open(os.path.join(ROOT, "runs", "diag", f"endgame_{name}.json"), "w") as f:
        json.dump(out, f)
    print(f"{title}; pocket {POCKET}: food left in {pocket} games, pacman steps there "
          f"{pocket_visits}; most often left: {out['top_left_cells'][:5]}")


if __name__ == "__main__":
    main()
