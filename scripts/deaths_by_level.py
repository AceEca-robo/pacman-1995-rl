#!/usr/bin/env python3
"""Deaths by level in games started on level 1 (to game over).

    python scripts/deaths_by_level.py --checkpoint runs/best_v04/best.pt \
        --episodes 200 --seed 4000 --out runs/best_v04/deaths_by_level.json \
        --plot docs/v04_deaths_by_level.png

Plays --episodes greedy games from level 1 (seeds seed..), each to game over
or --max-steps ticks (the eval protocol's eval_max_episode_steps), on the
death report's eval env (no shaping, hunger limit or prefixes). For every
life lost: the level, ticks since the level's start, food left, and the kind
(scripts/death_report.py's classify: squeezed / supertime ran out / ...). Per
level: games that reached it, cleared it, lives on arrival, deaths, deaths
per 1000 ticks played on it (games stuck there until --max-steps left out),
games stuck there, and P(clear | arrived with 3 lives) — the same
start as a final-protocol game on that maze. Level N is maze N in the game
(gamedata.cc: setboardlevel); past 16 the maze is random, found here by its
walls.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from agents.dqn import pack_grids  # noqa: E402
from death_report import classify  # noqa: E402
from env.pacman_env import PacmanEnv, load_env_config, maze_layouts  # noqa: E402
from train_eval import Scorer  # noqa: E402

WALLS = [tuple(row.replace("_", "O") for row in m) for m in maze_layouts()]


def maze_of(grid):
    """Maze number (1-16) of a state's grid, matched by its walls."""
    w = tuple("".join("O" if c in "O_" else "-" for c in row) for row in grid)
    for i, m in enumerate(WALLS):
        if w == tuple("".join("O" if c == "O" else "-" for c in row) for row in m):
            return i + 1
    return None


def play(scorer, envs, seeds, max_steps, radius):
    games, deaths, live = [], [], []
    seeds = list(seeds)

    def start(env, seed):
        obs, info = env.reset(seed=seed, options={"start_level": 1})
        return {"env": env, "obs": obs, "seed": seed, "t": 0, "level": info["level"],
                "level_t": 0, "lives_in": info["lives"], "visits": []}

    def close_visit(g, cleared, cut=False):
        g["visits"].append({"seed": g["seed"], "level": g["level"], "lives_in": g["lives_in"],
                            "ticks": g["level_t"], "cleared": cleared, "cut": cut})

    for env in envs:
        if seeds:
            live.append(start(env, seeds.pop(0)))
    while live:
        q = scorer.q_values(pack_grids(np.stack([g["obs"]["grid"] for g in live]),
                                       live[0]["env"].grid_extras),
                            np.stack([g["obs"]["vec"] for g in live]))
        for g, a in zip(list(live), q.argmax(1)):
            env = g["env"]
            prev = env._state
            obs, r, term, trunc, info = env.step(int(a))
            g["obs"], g["t"], g["level_t"] = obs, g["t"] + 1, g["level_t"] + 1
            if info["events"]["death"]:
                kind, n_exits, near_d = classify(prev, radius)
                deaths.append({"seed": g["seed"], "level": g["level"],
                               "maze": g["level"] if g["level"] <= 16 else maze_of(prev["grid"]),
                               "level_tick": g["level_t"], "tick": g["t"],
                               "food_left": sum(row.count(".") + row.count("o")
                                                for row in prev["grid"]),
                               "super": prev["supertime_left"] > 0, "kind": kind,
                               "exits": n_exits, "ghost_dist": near_d,
                               "final": info["lives"] == 0})
            if info["level"] > g["level"]:
                close_visit(g, True)
                g.update(level=info["level"], level_t=0, lives_in=info["lives"])
            if term or trunc or g["t"] >= max_steps:
                close_visit(g, False, cut=not term)
                games.append({"seed": g["seed"], "length": g["t"], "score": info["score"],
                              "levels_cleared": g["level"] - 1,
                              "outcome": "game over" if term else "cut",
                              "visits": g["visits"]})
                if seeds:
                    g.update(start(env, seeds.pop(0)))
                else:
                    live.remove(g)
    return games, deaths


def summarize(games, deaths):
    n = len(games)
    visits = [v for g in games for v in g["visits"]]
    per = {}
    for lv in sorted({v["level"] for v in visits}):
        vs = [v for v in visits if v["level"] == lv]
        ds = [d for d in deaths if d["level"] == lv]
        v3 = [v for v in vs if v["lives_in"] >= 3]
        # a game cut at max_steps is stuck on its last level: its ticks and
        # deaths there are left out of the rate
        cut = [v for v in vs if v["cut"]]
        ticks = sum(v["ticks"] for v in vs if not v["cut"])
        n_cut_deaths = sum(sum(d["seed"] == c["seed"] for d in ds) for c in cut)
        kinds = Counter(d["kind"] for d in ds)
        per[lv] = {"reached": len(vs), "reached_share": len(vs) / n,
                   "cleared": sum(v["cleared"] for v in vs),
                   "lives_in_mean": float(np.mean([v["lives_in"] for v in vs])),
                   "deaths": len(ds), "deaths_per_visit": len(ds) / len(vs),
                   "deaths_per_1000_ticks": 1000 * (len(ds) - n_cut_deaths) / max(ticks, 1),
                   "deaths_per_1000_ticks_ci": [  # normal approximation to Poisson
                       1000 * max(0.0, (len(ds) - n_cut_deaths)
                                  - 1.96 * np.sqrt(len(ds) - n_cut_deaths)) / max(ticks, 1),
                       1000 * ((len(ds) - n_cut_deaths)
                               + 1.96 * np.sqrt(len(ds) - n_cut_deaths)) / max(ticks, 1)],
                   "ticks": ticks, "stuck": len(cut),
                   "clear_given_3_lives": (sum(v["cleared"] for v in v3) / len(v3)) if v3 else None,
                   "n_3_lives": len(v3),
                   "kinds": dict(kinds),
                   "squeezed_share": kinds["squeezed"] / len(ds) if ds else None,
                   "level_tick_median": float(np.median([d["level_tick"] for d in ds])) if ds else None,
                   "food_left_median": float(np.median([d["food_left"] for d in ds])) if ds else None,
                   "lt20_food_share": (sum(d["food_left"] < 20 for d in ds) / len(ds)) if ds else None}
    nd = len(deaths)
    return {"games": n, "deaths": nd,
            "levels_cleared_mean": float(np.mean([g["levels_cleared"] for g in games])),
            "levels_cleared_max": max(g["levels_cleared"] for g in games),
            "cut": sum(g["outcome"] == "cut" for g in games),
            "kinds": {k: v / nd for k, v in Counter(d["kind"] for d in deaths).most_common()},
            "per_level": per}


def plot(s, deaths, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    per = s["per_level"]
    lvs = [lv for lv in per if per[lv]["reached"] >= 5]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    a = ax[0, 0]
    a.bar([str(lv) for lv in per], [per[lv]["reached_share"] for lv in per], color="#4C78A8")
    a.set_title(f"games reaching each level (of {s['games']}, from level 1)")
    a.set_xlabel("level"), a.set_ylabel("share of games")
    a2 = a.twinx()
    a2.plot([str(lv) for lv in per], [per[lv]["lives_in_mean"] for lv in per], "o-",
            color="#F58518")
    a2.set_ylabel("mean lives on arrival", color="#F58518"), a2.set_ylim(0, 3.2)
    a = ax[0, 1]
    r = np.array([per[lv]["deaths_per_1000_ticks"] for lv in lvs])
    ci = np.array([per[lv]["deaths_per_1000_ticks_ci"] for lv in lvs]).T
    a.errorbar([str(lv) for lv in lvs], r, yerr=[r - ci[0], ci[1] - r], fmt="o-",
               color="#E45756", capsize=3)
    a.set_ylim(0, None)
    a2 = a.twinx()
    a2.plot([str(lv) for lv in lvs],
            [per[lv]["clear_given_3_lives"] if per[lv]["n_3_lives"] >= 5 else np.nan
             for lv in lvs], "s--", color="#4C78A8")
    a.set_title("deaths per 1000 ticks (95% CI) and P(clear | 3 lives, n >= 5)")
    a.set_xlabel("level"), a.set_ylabel("deaths / 1000 ticks", color="#E45756")
    a2.set_ylabel("P(clear | 3 lives)", color="#4C78A8"), a2.set_ylim(0, 1)
    a = ax[1, 0]
    kinds = ["squeezed", "supertime ran out", "from behind", "head-on", "standing",
             "no normal ghost"]
    colors = ["#E45756", "#F58518", "#72B7B2", "#54A24B", "#B279A2", "#9D755D"]
    bottom = np.zeros(len(lvs))
    for k, c in zip(kinds, colors):
        v = np.array([per[lv]["kinds"].get(k, 0) / max(per[lv]["deaths"], 1) for lv in lvs])
        if v.any():
            a.bar([str(lv) for lv in lvs], v, bottom=bottom, color=c, label=k)
            bottom += v
    a.set_title("kind of death by level (share of that level's deaths)")
    a.set_xlabel("level"), a.legend(fontsize=8, loc="lower right")
    a = ax[1, 1]
    for lv, c in zip(lvs[:6], ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B", "#B279A2"]):
        t = [d["level_tick"] for d in deaths if d["level"] == lv]
        if t:
            a.hist(t, bins=np.arange(0, 2001, 100), histtype="step", color=c, label=f"level {lv}")
    a.set_title("ticks since the level's start at each death")
    a.set_xlabel("ticks on the level"), a.set_ylabel("deaths"), a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=4000)
    p.add_argument("--max-steps", type=int, default=30000)
    p.add_argument("--radius", type=int, default=5)
    p.add_argument("--envs", type=int, default=20)
    p.add_argument("--out", required=True)
    p.add_argument("--plot", default=None)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    env_cfg = load_env_config(ck["config"]["env_config"])
    eval_cfg = {**env_cfg, "shaping": {**env_cfg["shaping"], "enabled": False},
                "hunger_limit": 0, "prefix_prob": 0.0, "endgame_dot": {"enabled": False},
                "start_level": 1, "max_episode_steps": None}
    scorer = Scorer(args.checkpoint, device)
    envs = [PacmanEnv(config=eval_cfg) for _ in range(args.envs)]
    t0 = time.perf_counter()
    try:
        games, deaths = play(scorer, envs, range(args.seed, args.seed + args.episodes),
                             args.max_steps, args.radius)
    finally:
        for e in envs:
            e.close()
    s = summarize(games, deaths)
    s.update(checkpoint=args.checkpoint, seed=args.seed, episodes=args.episodes,
             max_steps=args.max_steps, time_s=round(time.perf_counter() - t0, 1))
    with open(args.out, "w") as f:
        json.dump({"summary": s, "games": games, "deaths": deaths}, f)
    print(json.dumps({k: v for k, v in s.items() if k != "per_level"}, indent=1))
    print("| level | reached | cleared | lives in | deaths/visit | deaths/1000 ticks "
          "| P(clear, 3 lives) | stuck | squeezed | med. level tick | med. food left |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for lv, m in s["per_level"].items():
        c3 = "—" if m["clear_given_3_lives"] is None else \
            f"{m['clear_given_3_lives']:.0%} ({m['n_3_lives']})"
        sq = "—" if m["squeezed_share"] is None else f"{m['squeezed_share']:.0%}"
        print(f"| {lv} | {m['reached']} ({m['reached_share']:.0%}) | {m['cleared']} "
              f"| {m['lives_in_mean']:.2f} | {m['deaths_per_visit']:.2f} "
              f"| {m['deaths_per_1000_ticks']:.2f} | {c3} | {m['stuck']} | {sq} "
              f"| {m['level_tick_median'] or '—'} | {m['food_left_median'] or '—'} |")
    if args.plot:
        plot(s, deaths, args.plot)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
