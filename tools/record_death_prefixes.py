#!/usr/bin/env python3
"""Record death prefixes: an agent's own greedy games, and for every life it
loses the actions from the game's start up to --offsets ticks before the
death (default 40 and 20), so PacmanEnv (prefix_prob / prefix_mix) can start
training episodes shortly before the agent's own deaths.

    python tools/record_death_prefixes.py --checkpoint runs/best_v04/best.pt \
        --games 400 --seed 5000 --out data/death_prefixes_v04.json

Game i is played on env seed --seed + i from a start level drawn by the env
from --start-level (default random:1-16, PacmanEnv.reset(seed=...)), to game
over or --max-steps ticks, on the death report's env (no shaping, hunger
limit, prefixes or episode limit; the checkpoint's observation). A death
whose level started fewer than --min-level-tick ticks before it is dropped
(its prefixes would start in the previous life or level). The game is
deterministic for a seed and an action sequence, so a prefix replays to the
same board, pacman and ghosts.

Format as tools/record_prefixes.py (endgame prefixes), each prefix with
"seed", "level" (the level the prefix ends on), "start_level" (the game's),
"actions", "steps", "food_left", "score", and also "lives", "pacman" [x, y]
and "ghosts" [[x, y], ..] at its end (the env's replay checks them),
"before_death" (the offset), "maze" (1-16; past level 16 found by the
walls), "kind" (scripts/death_report.py classify of the state before the
death) and "level_tick" (ticks since the level's start at the death).
Without --checkpoint the heuristic plays (for tests).
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, deque

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from agents.dqn import pack_grids  # noqa: E402
from env.pacman_env import PacmanEnv, load_env_config  # noqa: E402


def snapshot(st):
    return {"level": st["level"], "lives": st["lives"], "score": st["score"],
            "food_left": sum(r.count(".") + r.count("o") for r in st["grid"]),
            "pacman": [st["pacman"]["x"], st["pacman"]["y"]],
            "ghosts": [[g["x"], g["y"]] for g in st["ghosts"]]}


def env_config(obs_cfg, start_level):
    cfg = load_env_config({"obs": obs_cfg or {}})
    return {**cfg, "shaping": {**cfg["shaping"], "enabled": False}, "hunger_limit": 0,
            "prefix_prob": 0.0, "endgame_dot": {"enabled": False},
            "max_episode_steps": None, "start_level": start_level}


def record(policy, envs, seeds, offsets=(40, 20), min_level_tick=60, max_steps=10000,
           radius=6):
    """policy(list of obs) -> actions. Plays one game per seed on envs (in
    parallel), returns (prefixes, games)."""
    from death_report import classify
    from deaths_by_level import maze_of
    keep = max(offsets) + 1
    seeds, live, prefixes, games = list(seeds), [], [], []

    def start(env, seed):
        obs, info = env.reset(seed=seed)
        return {"env": env, "obs": obs, "seed": seed, "start": info["level"], "actions": [],
                "level_t": 0, "hist": deque([(0, snapshot(env._state))], maxlen=keep),
                "deaths": 0, "kept": 0}

    for env in envs:
        if seeds:
            live.append(start(env, seeds.pop(0)))
    while live:
        acts = policy([g["obs"] for g in live])
        for g, a in zip(list(live), acts):
            env = g["env"]
            prev = env._state
            obs, _, term, _, info = env.step(int(a))
            g["obs"] = obs
            g["actions"].append(int(a))
            t = len(g["actions"])
            g["level_t"] += 1
            st = env._state
            if info["events"]["death"]:
                g["deaths"] += 1
                if g["level_t"] >= min_level_tick:
                    kind = classify(prev, radius)[0]
                    maze = prev["level"] if prev["level"] <= 16 else maze_of(prev["grid"])
                    hist = dict(g["hist"])
                    for off in offsets:
                        snap = hist[t - off]
                        assert snap["level"] == prev["level"]
                        prefixes.append({"seed": g["seed"], "level": snap["level"],
                                         "start_level": g["start"],
                                         "actions": g["actions"][:t - off], "steps": t - off,
                                         **{k: snap[k] for k in ("food_left", "score", "lives",
                                                                 "pacman", "ghosts")},
                                         "before_death": off, "maze": maze, "kind": kind,
                                         "level_tick": g["level_t"]})
                    g["kept"] += 1
            if st["level"] != prev["level"]:
                g["level_t"] = 0
            g["hist"].append((t, snapshot(st)))
            if term or t >= max_steps:
                games.append({"seed": g["seed"], "start_level": g["start"], "length": t,
                              "level_end": st["level"], "deaths": g["deaths"],
                              "kept_deaths": g["kept"], "outcome": "game over" if term else "cut"})
                if seeds:
                    g.update(start(env, seeds.pop(0)))
                else:
                    live.remove(g)
    return prefixes, games


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", help="DQN checkpoint (default: the heuristic)")
    p.add_argument("--games", type=int, default=400)
    p.add_argument("--seed", type=int, default=5000)
    p.add_argument("--start-level", default="random:1-16")
    p.add_argument("--offsets", type=int, nargs="+", default=[40, 20])
    p.add_argument("--min-level-tick", type=int, default=60)
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--envs", type=int, default=20)
    p.add_argument("--out", default=os.path.join(ROOT, "data", "death_prefixes.json"))
    args = p.parse_args()
    if args.checkpoint:
        import torch
        from train_eval import Scorer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        env_cfg = ck.get("env_config") or {}
        if env_cfg.get("actions", 5) != 5:
            sys.exit("prefixes are replayed with 5 actions; this agent differs")
        scorer = Scorer(args.checkpoint, device)
        name = os.path.relpath(args.checkpoint, ROOT)
        obs_cfg = env_cfg.get("obs") or {}
        envs = [PacmanEnv(config=env_config(obs_cfg, args.start_level))
                for _ in range(args.envs)]
        extras = envs[0].grid_extras

        def policy(obs):
            q = scorer.q_values(pack_grids(np.stack([o["grid"] for o in obs]), extras),
                                np.stack([o["vec"] for o in obs]))
            return q.argmax(1)
    else:
        from agents.heuristic_agent import HeuristicAgent
        agent, name = HeuristicAgent(), "heuristic"
        envs = [PacmanEnv(config=env_config({}, args.start_level)) for _ in range(args.envs)]

        def policy(obs):
            return [agent.act(o) for o in obs]
    t0 = time.time()
    try:
        prefixes, games = record(policy, envs, range(args.seed, args.seed + args.games),
                                 tuple(args.offsets), args.min_level_tick, args.max_steps)
    finally:
        for e in envs:
            e.close()
    deaths = sum(g["deaths"] for g in games)
    kept = sum(g["kept_deaths"] for g in games)
    first = [x for x in prefixes if x["before_death"] == args.offsets[0]]
    summary = {"games": len(games), "deaths": deaths, "kept_deaths": kept,
               "dropped_early": deaths - kept,
               "cut_games": sum(g["outcome"] == "cut" for g in games),
               "by_maze": dict(sorted(Counter(x["maze"] for x in first).items(),
                                      key=lambda kv: (kv[0] is None, kv[0] or 0))),
               "by_kind": dict(Counter(x["kind"] for x in first).most_common()),
               "by_lives": dict(sorted(Counter(x["lives"] for x in first).items())),
               "start_levels": dict(sorted(Counter(g["start_level"] for g in games).items())),
               "steps_mean": float(np.mean([x["steps"] for x in prefixes])) if prefixes else 0,
               "steps_max": max((x["steps"] for x in prefixes), default=0),
               "seconds": round(time.time() - t0)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"agent": name, "n_actions": 5,
                   "seeds": [args.seed, args.seed + args.games - 1],
                   "start_level": args.start_level, "offsets": args.offsets,
                   "min_level_tick": args.min_level_tick, "max_steps": args.max_steps,
                   "summary": summary, "games": games, "prefixes": prefixes}, f)
    print(json.dumps(summary, indent=1))
    print(f"{len(prefixes)} prefixes ({kept} deaths x {len(args.offsets)}) -> {args.out}")


if __name__ == "__main__":
    main()
