#!/usr/bin/env python3
"""Record endgame prefixes: an agent's actions from the start of a game until
at most --max-left food items (dots + energizers) are left on level 1, for
env seeds --seed-start .. --seed-start + --seeds - 1. Only games where it got
there without losing a life are kept. PacmanEnv (prefix_prob / prefix_file)
replays them on reset, so training episodes can start near the end of level 1.

    python tools/record_prefixes.py [--seeds 100] [--max-left 15] [--out data/endgame_prefixes.json]
    python tools/record_prefixes.py --checkpoint runs/dqn8/best_food.pt --seed-start 2000 \
        --seeds 600 --out data/dqn8_prefixes.json      # the agent's own greedy games
    python tools/record_prefixes.py --level 1-16 --keep 25 --seed-start 3000 \
        --seeds 400 --out data/endgame_prefixes_all.json   # all 16 mazes

--level N (or a-b) records on that level / each of those levels (game
--level), each prefix stores its "level"; with --keep K a level stops after
K kept prefixes (seeds tried in order, at most --seeds per level). Levels
are recorded in parallel processes.

The game is deterministic for a seed and an action sequence, so replaying
the actions reproduces the board, ghosts and all.
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agents.heuristic_agent import HeuristicAgent  # noqa: E402
from env.pacman_env import PacmanEnv  # noqa: E402


def food_left(state):
    return sum(row.count(".") + row.count("o") for row in state["grid"])


def record(seeds, max_left, agent=None, n_actions=5, level=1, keep=None):
    agent = agent or HeuristicAgent(n_actions=n_actions)
    env = PacmanEnv(config={**getattr(agent, "env_overrides", {}), "actions": n_actions,
                            "max_episode_steps": None, "start_level": level})
    kept, skipped = [], {"lost_a_life": 0, "game_over": 0}
    try:
        for seed in seeds:
            if keep is not None and len(kept) >= keep:
                break
            obs, _ = env.reset(seed=seed)
            agent.reset(seed=seed)
            actions = []
            while True:
                a = agent.act(obs)
                obs, _, term, _, info = env.step(a)
                actions.append(int(a))
                st = env._state
                if info["lives"] < 3 or term:
                    skipped["game_over" if term else "lost_a_life"] += 1
                    break
                if st["level"] == level and food_left(st) <= max_left:
                    kept.append({"seed": seed, "level": level, "actions": actions,
                                 "steps": len(actions), "food_left": food_left(st),
                                 "score": st["score"]})
                    break
    finally:
        env.close()
    return kept, skipped


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", type=int, default=100, help="number of env seeds")
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--checkpoint", help="record a DQN/PPO checkpoint's greedy games "
                                        "instead of the heuristic's")
    p.add_argument("--max-left", type=int, default=15)
    p.add_argument("--out", default=os.path.join(ROOT, "data", "endgame_prefixes.json"))
    p.add_argument("--level", default="1", help="level N or levels a-b to record on")
    p.add_argument("--keep", type=int, help="stop a level after this many kept prefixes")
    args = p.parse_args()
    lo, _, hi = args.level.partition("-")
    levels = list(range(int(lo), int(hi or lo) + 1))
    agent, name = None, "heuristic"
    if args.checkpoint:
        from agents.ppo import load_agent
        agent, name = load_agent(args.checkpoint), os.path.relpath(args.checkpoint, ROOT)
        if agent.n_actions != 5 or any((agent.env_overrides.get("obs") or {}).values()):
            sys.exit("prefixes are replayed in the default 5-action env; this agent differs")
    seeds = range(args.seed_start, args.seed_start + args.seeds)
    if len(levels) == 1:
        results = [record(seeds, args.max_left, agent, level=levels[0], keep=args.keep)]
    else:
        if agent is not None:
            sys.exit("several levels: heuristic only")
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(min(len(levels), os.cpu_count() or 1)) as ex:
            results = list(ex.map(_record_level, [(seeds, args.max_left, lv, args.keep)
                                                  for lv in levels]))
    kept = [k for ks, _ in results for k in ks]
    skipped = {lv: sk for lv, (_, sk) in zip(levels, results)} if len(levels) > 1 \
        else results[0][1]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"agent": name, "n_actions": 5, "seeds": [seeds.start, seeds.stop - 1],
                   "levels": levels, "keep": args.keep,
                   "max_food_left": args.max_left, "skipped": skipped, "prefixes": kept}, f)
    for lv, (ks, sk) in zip(levels, results):
        tried = len(ks) + sum(sk.values())
        print(f"{name} level {lv}: kept {len(ks)} of {tried} seeds ({sk}); prefix length "
              f"{min((k['steps'] for k in ks), default=0)}.."
              f"{max((k['steps'] for k in ks), default=0)} steps")
    print(f"{len(kept)} prefixes -> {args.out}")


def _record_level(job):
    seeds, max_left, level, keep = job
    return record(seeds, max_left, level=level, keep=keep)


if __name__ == "__main__":
    main()
