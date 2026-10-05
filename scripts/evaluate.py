#!/usr/bin/env python3
"""Evaluate a baseline agent on Pacman1995-v0.

    python scripts/evaluate.py --agent heuristic --episodes 100 --seed 0

Episode i is played with env seed (and agent seed) seed + i. Prints a
markdown table and updates the agent's row in docs/results.md.
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agents.heuristic_agent import HeuristicAgent  # noqa: E402
from agents.random_agent import RandomAgent  # noqa: E402
from env.pacman_env import PacmanEnv  # noqa: E402

AGENTS = {"random": RandomAgent, "heuristic": HeuristicAgent}
HEADER = ("| agent | episodes | seeds | mean reward | median reward | mean score "
          "| median score | max score | level 1 cleared | level 1 without death "
          "| mean length | truncated | time, s | commit |")
RULE = "|" + "---|" * HEADER.count(" | ") + "---|"
RESULTS_INTRO = """# Baseline results

`scripts/evaluate.py`, environment `configs/env_default.yaml`
(episode = one game of 3 lives, truncated at `max_episode_steps`).
Reward is the env's event reward (see docs/observation.md), score the game's.
"Level 1 cleared" = share of episodes that reached level 2; "without death"
= reached it before losing any life. Length is in env steps (game ticks).

"""


def run(agent_name, episodes, seed, env_config, agent_config):
    agent = AGENTS[agent_name](agent_config) if agent_name == "heuristic" else AGENTS[agent_name](seed)
    env = PacmanEnv(config=env_config)
    scores, rewards, lengths, max_levels, clean, truncs = [], [], [], [], [], []
    t0 = time.perf_counter()
    try:
        for i in range(episodes):
            obs, info = env.reset(seed=seed + i)
            agent.reset(seed=seed + i)
            n, level, total, deaths, clean_clear = 0, info["level"], 0.0, 0, False
            while True:
                obs, r, term, trunc, info = env.step(agent.act(obs))
                n += 1
                total += r
                ev = info["events"]
                if ev["level_up"] and level == 1 and deaths == 0:
                    clean_clear = True
                deaths += ev["death"]
                level = max(level, info["level"])
                if term or trunc:
                    break
            scores.append(info["score"])
            rewards.append(total)
            clean.append(clean_clear)
            lengths.append(n)
            max_levels.append(level)
            truncs.append(trunc)
    finally:
        env.close()
    elapsed = time.perf_counter() - t0
    scores = np.array(scores)
    return {
        "agent": agent_name,
        "episodes": episodes,
        "seeds": f"{seed}..{seed + episodes - 1}",
        "mean reward": f"{np.mean(rewards):.1f}",
        "median reward": f"{np.median(rewards):.1f}",
        "mean score": f"{scores.mean():.0f}",
        "median score": f"{np.median(scores):.0f}",
        "max score": f"{scores.max()}",
        "level 1 cleared": f"{np.mean(np.array(max_levels) >= 2):.0%}",
        "level 1 without death": f"{np.mean(clean):.0%}",
        "mean length": f"{np.mean(lengths):.0f}",
        "truncated": f"{np.mean(truncs):.0%}",
        "time, s": f"{elapsed:.1f}",
        "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                 capture_output=True, text=True).stdout.strip(),
    }


def row(res):
    return "| " + " | ".join(str(v) for v in res.values()) + " |"


def update_results(path, res):
    """Rewrite the table in path with res's row replaced; "## " sections after
    the table are kept."""
    rows, tail = {}, ""
    if os.path.exists(path):
        with open(path) as f:
            text = f.read()
        if "\n## " in text:
            tail = text[text.index("\n## "):]
            text = text[:text.index("\n## ")]
        for line in text.splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if line.startswith("| ") and line.strip() != HEADER and cells[0] in AGENTS:
                rows[cells[0]] = line
    rows[res["agent"]] = row(res)
    with open(path, "w") as f:
        f.write(RESULTS_INTRO + HEADER + "\n" + RULE + "\n")
        for name in AGENTS:
            if name in rows:
                f.write(rows[name] + "\n")
        f.write(tail)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--agent", choices=sorted(AGENTS), required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env-config", default=None, help="default configs/env_default.yaml")
    p.add_argument("--agent-config", default=None, help="heuristic only; default configs/heuristic.yaml")
    p.add_argument("--results", default=os.path.join(ROOT, "docs", "results.md"),
                   help="markdown file to update; empty string to skip")
    args = p.parse_args()

    res = run(args.agent, args.episodes, args.seed, args.env_config, args.agent_config)
    print(HEADER)
    print(RULE)
    print(row(res))
    if args.results:
        update_results(args.results, res)


if __name__ == "__main__":
    main()
