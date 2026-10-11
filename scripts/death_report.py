#!/usr/bin/env python3
"""Where and how a checkpoint loses: per-maze games with every death analysed.

    python scripts/death_report.py --checkpoint runs/best_v03/best.pt \
        --episodes 50 --seed 5000 --out runs/best_v03/deaths.json --plots docs/v03_deaths

Plays --episodes games started on each of the 16 mazes (greedy, maze lv on
seeds seed + (lv - 1) * episodes ..), each to the maze's clear, game over or
--max-steps ticks (the eval protocol: train.py eval_level_max_steps). A game
that does not clear is a "death" loss (game over) or a "stuck" loss (cut at
--max-steps). For every life lost, from the state just before it:
  - food left on the board, supertime > 0 or not, the tick (since the game's
    start and since the life's start);
  - the kind: "squeezed" if every way out of pacman's cell (the corridor's
    two directions, or all exits at a junction) has a normal ghost within
    --radius steps; the cells within --radius are split between the exits
    by their shortest path from pacman (exit_regions), so one ghost blocks
    one direction only, however short the maze's loops;
    otherwise "head-on" if the nearest normal ghost is in the direction
    pacman moves, "from behind" if it is in another one, "standing" if
    pacman is not moving, "supertime ran out" if no normal ghost is within
    --radius but a ghost is still hunted (supertime ended this tick), "no
    normal ghost" otherwise.
Writes the per-death records and summaries as JSON, prints the tables, and
with --plots draws <prefix>_mazes.png (death cells on each maze) and
<prefix>_summary.png.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from agents.dqn import pack_grids  # noqa: E402
from env.pacman_env import (HEIGHT, LEVELS, WIDTH, PacmanEnv, _passable_neighbours,  # noqa: E402
                            load_env_config, maze_layouts)
from train_eval import Scorer  # noqa: E402

DIRS = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}


def exit_regions(cells, pac, radius):
    """The cells within radius steps of pac (flat index), each assigned to
    the exit (pacman-passable neighbour of pac) its shortest path leaves
    through (BFS from all exits at once, never back through pac; ties go to
    the first exit). -> ({cell: (exit, distance)}, exits)."""
    nbrs = _passable_neighbours(cells)
    exits = list(nbrs[pac])
    region = {e: (e, 1) for e in exits}
    frontier, d = exits, 1
    while frontier and d < radius:
        d += 1
        nxt = []
        for c in frontier:
            for nb in nbrs[c]:
                if nb != pac and nb not in region:
                    region[nb] = (region[c][0], d)
                    nxt.append(nb)
        frontier = nxt
    return region, exits


def classify(prev, radius):
    """Kind of a death from the state before it -> (kind, exits, distance
    of the nearest normal ghost within radius or None)."""
    cells = "".join(prev["grid"])
    p = prev["pacman"]
    pac = p["y"] * WIDTH + p["x"]
    ghosts = [g["y"] * WIDTH + g["x"] for g in prev["ghosts"] if g["state"] == "normal"]
    region, exits = exit_regions(cells, pac, radius)
    near = {}  # exit -> nearest normal ghost distance in its region
    for gc in ghosts:
        if gc in region:
            e, d = region[gc]
            near[e] = min(near.get(e, d), d)
    near_d = 0 if pac in ghosts else (min(near.values()) if near else None)
    if len(exits) >= 2 and len(near) == len(exits):
        return "squeezed", len(exits), near_d
    if near_d is None:
        # in practice: supertime ran out this tick next to a still hunted ghost
        hunted = any(g["state"] == "hunted" for g in prev["ghosts"])
        return ("supertime ran out" if hunted else "no normal ghost"), len(exits), None
    if p["dir"] not in DIRS:
        return "standing", len(exits), near_d
    if near_d == 0:
        return "head-on", len(exits), 0
    dx, dy = DIRS[p["dir"]]
    ahead = (p["y"] + dy) * WIDTH + (p["x"] + dx)
    nearest = min(near, key=near.get)
    return ("head-on" if nearest == ahead else "from behind"), len(exits), near_d


def play(scorer, envs, tasks, max_steps, radius):
    """tasks: (maze, seed). -> (games, deaths)."""
    games, deaths, live = [], [], []
    tasks = list(tasks)

    def start(env, task):
        obs, info = env.reset(seed=task[1], options={"start_level": task[0]})
        return {"env": env, "obs": obs, "task": task, "t": 0, "life_t": 0,
                "start": info["level"], "deaths": 0}

    for env in envs:
        if tasks:
            live.append(start(env, tasks.pop(0)))
    while live:
        q = scorer.q_values(pack_grids(np.stack([g["obs"]["grid"] for g in live]),
                                       live[0]["env"].grid_extras),
                            np.stack([g["obs"]["vec"] for g in live]))
        for g, a in zip(list(live), q.argmax(1)):
            env = g["env"]
            prev = env._state
            obs, r, term, trunc, info = env.step(int(a))
            g["obs"], g["t"], g["life_t"] = obs, g["t"] + 1, g["life_t"] + 1
            ev = info["events"]
            if ev["death"]:
                kind, n_exits, near_d = classify(prev, radius)
                food = sum(row.count(".") + row.count("o") for row in prev["grid"])
                deaths.append({"maze": g["task"][0], "seed": g["task"][1], "tick": g["t"],
                               "life_tick": g["life_t"], "food_left": food,
                               "super": prev["supertime_left"] > 0, "kind": kind,
                               "x": prev["pacman"]["x"], "y": prev["pacman"]["y"],
                               "exits": n_exits, "ghost_dist": near_d,
                               "ghost_states": [gh["state"] for gh in prev["ghosts"]],
                               "final": info["lives"] == 0})
                g["deaths"] += 1
                g["life_t"] = 0
            cleared = info["level"] > g["start"]
            cut = g["t"] >= max_steps
            if term or trunc or cut or cleared:
                games.append({"maze": g["task"][0], "seed": g["task"][1], "length": g["t"],
                              "cleared": bool(cleared), "deaths": g["deaths"],
                              "outcome": "cleared" if cleared else
                              ("death" if term else "stuck")})
                if tasks:
                    g.update(start(env, tasks.pop(0)))
                else:
                    live.remove(g)
    return games, deaths


def summarize(games, deaths):
    n_lost = sum(not g["cleared"] for g in games)
    out = Counter(g["outcome"] for g in games)
    s = {"games": len(games), "cleared": out["cleared"] / len(games),
         "lost": n_lost, "lost_death": out["death"], "lost_stuck": out["stuck"],
         "deaths": len(deaths)}
    if deaths:
        nd = len(deaths)
        s.update({
            "share_endgame_lt20": sum(d["food_left"] < 20 for d in deaths) / nd,
            "share_super": sum(d["super"] for d in deaths) / nd,
            "share_first200": sum(d["tick"] <= 200 for d in deaths) / nd,
            "share_life_first200": sum(d["life_tick"] <= 200 for d in deaths) / nd,
            "kinds": {k: v / nd for k, v in Counter(d["kind"] for d in deaths).most_common()},
            "food_left_median": float(np.median([d["food_left"] for d in deaths])),
            "food_left_quartiles": [float(x) for x in
                                    np.percentile([d["food_left"] for d in deaths], [25, 75])],
        })
    per_maze = {}
    for lv in sorted({g["maze"] for g in games}):
        gs = [g for g in games if g["maze"] == lv]
        ds = [d for d in deaths if d["maze"] == lv]
        per_maze[lv] = {"cleared": float(np.mean([g["cleared"] for g in gs])),
                        "deaths": len(ds), "deaths_per_game": len(ds) / len(gs),
                        "lost_death": sum(g["outcome"] == "death" for g in gs),
                        "lost_stuck": sum(g["outcome"] == "stuck" for g in gs),
                        "squeezed": sum(d["kind"] == "squeezed" for d in ds)}
    s["per_maze"] = per_maze
    return s


def plot(deaths, s, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    layouts = maze_layouts()
    fig, axes = plt.subplots(4, 4, figsize=(16, 12.5))
    vmax = 1
    heat = {}
    for lv in range(1, LEVELS + 1):
        h = np.zeros((HEIGHT, WIDTH))
        for d in deaths:
            if d["maze"] == lv:
                h[d["y"], d["x"]] += 1
        heat[lv] = h
        vmax = max(vmax, h.max())
    for lv, ax in zip(range(1, LEVELS + 1), axes.ravel()):
        walls = np.array([[ch == "#" for ch in row] for row in layouts[lv - 1]], float)
        ax.imshow(walls, cmap=ListedColormap(["#ffffff", "#b8bcc4"]), interpolation="nearest")
        h = np.ma.masked_where(heat[lv] == 0, heat[lv])
        im = ax.imshow(h, cmap="inferno_r", vmin=0, vmax=vmax, interpolation="nearest")
        m = s["per_maze"][lv]
        ax.set_title(f"maze {lv}: {m['deaths']} deaths, cleared {m['cleared']:.0%}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="deaths in the cell")
    fig.suptitle("Death cells per maze (state before the death)", fontsize=13)
    fig.savefig(prefix + "_mazes.png", dpi=80, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    mazes = sorted(s["per_maze"])
    ax = axes[0]
    ax.bar([str(m) for m in mazes], [s["per_maze"][m]["deaths_per_game"] for m in mazes],
           color="#4a6fa5")
    ax.set_xlabel("start maze")
    ax.set_ylabel("deaths per game")
    ax.set_title("Deaths per game by maze")
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    food = [d["food_left"] for d in deaths]
    ax.hist(food, bins=np.arange(0, max(food + [1]) + 10, 10), color="#4a6fa5")
    ax.set_xlabel("food left at the death")
    ax.set_ylabel("deaths")
    ax.set_title(f"Food left (< 20: {s.get('share_endgame_lt20', 0):.0%})")
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[2]
    kinds = s.get("kinds", {})
    ax.barh(list(kinds)[::-1], [v * 100 for v in list(kinds.values())[::-1]], color="#4a6fa5")
    ax.set_xlabel("% of deaths")
    ax.set_title(f"Kind (supertime > 0: {s.get('share_super', 0):.0%})")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(prefix + "_summary.png", dpi=80, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=5000)
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--radius", type=int, default=5)
    p.add_argument("--envs", type=int, default=20)
    p.add_argument("--out", required=True)
    p.add_argument("--plots", default=None, help="prefix of the two PNG files")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    env_cfg = load_env_config(ck["config"]["env_config"])
    eval_cfg = {**env_cfg, "shaping": {**env_cfg["shaping"], "enabled": False},
                "hunger_limit": 0, "prefix_prob": 0.0, "endgame_dot": {"enabled": False},
                "start_level": 1, "max_episode_steps": None}
    scorer = Scorer(args.checkpoint, device)
    tasks = [(lv, args.seed + (lv - 1) * args.episodes + i)
             for lv in range(1, LEVELS + 1) for i in range(args.episodes)]
    envs = [PacmanEnv(config=eval_cfg) for _ in range(args.envs)]
    t0 = time.perf_counter()
    try:
        games, deaths = play(scorer, envs, tasks, args.max_steps, args.radius)
    finally:
        for e in envs:
            e.close()
    s = summarize(games, deaths)
    s.update(checkpoint=args.checkpoint, seed=args.seed, episodes=args.episodes,
             radius=args.radius, max_steps=args.max_steps,
             time_s=round(time.perf_counter() - t0, 1))
    with open(args.out, "w") as f:
        json.dump({"summary": s, "games": games, "deaths": deaths}, f)
    print(json.dumps({k: v for k, v in s.items() if k != "per_maze"}, indent=1))
    print("| maze | cleared | deaths/game | lost: death | lost: stuck | squeezed |")
    print("|---|---|---|---|---|---|")
    for lv, m in sorted(s["per_maze"].items(), key=lambda kv: -kv[1]["deaths_per_game"]):
        print(f"| {lv} | {m['cleared']:.0%} | {m['deaths_per_game']:.2f} | {m['lost_death']} "
              f"| {m['lost_stuck']} | {m['squeezed']} |")
    if args.plots:
        plot(deaths, s, args.plots)


if __name__ == "__main__":
    main()
