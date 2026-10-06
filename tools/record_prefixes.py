#!/usr/bin/env python3
"""Record endgame prefixes: the heuristic agent's actions from the start of a
game until at most --max-left food items (dots + energizers) are left on
level 1, for env seeds 0..--seeds-1. Only games where it got there without
losing a life are kept. PacmanEnv (prefix_prob / prefix_file) replays them
on reset, so training episodes can start near the end of level 1.

    python tools/record_prefixes.py [--seeds 100] [--max-left 15] [--out data/endgame_prefixes.json]

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


def record(seeds, max_left, n_actions=5):
    agent = HeuristicAgent(n_actions=n_actions)
    env = PacmanEnv(config={"actions": n_actions, "max_episode_steps": None})
    kept, skipped = [], {"lost_a_life": 0, "game_over": 0}
    try:
        for seed in range(seeds):
            obs, _ = env.reset(seed=seed)
            actions = []
            while True:
                a = agent.act(obs)
                obs, _, term, _, info = env.step(a)
                actions.append(int(a))
                st = env._state
                if info["lives"] < 3 or term:
                    skipped["game_over" if term else "lost_a_life"] += 1
                    break
                if st["level"] == 1 and food_left(st) <= max_left:
                    kept.append({"seed": seed, "actions": actions, "steps": len(actions),
                                 "food_left": food_left(st), "score": st["score"]})
                    break
    finally:
        env.close()
    return kept, skipped


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", type=int, default=100)
    p.add_argument("--max-left", type=int, default=15)
    p.add_argument("--out", default=os.path.join(ROOT, "data", "endgame_prefixes.json"))
    args = p.parse_args()
    kept, skipped = record(args.seeds, args.max_left)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"agent": "heuristic", "n_actions": 5, "seeds": args.seeds,
                   "max_food_left": args.max_left, "skipped": skipped, "prefixes": kept}, f)
    print(f"kept {len(kept)} of {args.seeds} seeds ({skipped}); prefix length "
          f"{min(k['steps'] for k in kept)}..{max(k['steps'] for k in kept)} steps -> {args.out}")


if __name__ == "__main__":
    main()
