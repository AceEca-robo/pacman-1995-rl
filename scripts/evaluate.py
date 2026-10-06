#!/usr/bin/env python3
"""Evaluate a baseline agent on Pacman1995-v0.

    python scripts/evaluate.py --agent heuristic --episodes 100 --seed 0
    python scripts/evaluate.py --agent dqn --checkpoint runs/dqn0/best.pt

Episode i is played with env seed (and agent seed) seed + i. Prints a
markdown table and updates the agent's row in docs/results.md.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agents.heuristic_agent import HeuristicAgent  # noqa: E402
from agents.dqn import DQNAgent  # noqa: E402
from agents.ppo import PPOAgent  # noqa: E402
from agents.random_agent import RandomAgent  # noqa: E402
from env.pacman_env import PacmanEnv, load_env_config  # noqa: E402

AGENTS = ("random", "heuristic", "dqn", "ppo")
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


def make_agent(args):
    """Returns (agent, label for the results table)."""
    if args.agent == "random":
        label = "random" if args.actions == 5 else "random (4 actions)"
        return RandomAgent(args.seed, args.actions), label
    if args.agent == "heuristic":
        label = "heuristic" if args.actions == 5 else "heuristic (4 actions)"
        return HeuristicAgent(args.agent_config, args.actions), label
    if not args.checkpoint:
        sys.exit(f"--agent {args.agent} needs --checkpoint")
    path = os.path.abspath(args.checkpoint)
    label = os.path.relpath(path, os.path.join(ROOT, "runs")) if path.startswith(ROOT) else path
    if args.agent == "ppo":
        return PPOAgent(path, seed=args.seed), f"ppo {label}"
    agent = DQNAgent(path, epsilon=args.epsilon, seed=args.seed)
    return agent, f"dqn {label}"


LONG_GAP = 30  # steps without food that count as "long"


def food_gaps(food_steps, length):
    """Steps without eating: gaps between consecutive food events (dots or
    energizers), the longest gap, the steps after the last food, and the
    share of the episode spent in stretches of >= LONG_GAP steps without
    food (the tail after the last food included)."""
    gaps = np.diff(food_steps) if len(food_steps) > 1 else np.array([])
    tail = length - food_steps[-1] if food_steps else length
    stretches = np.append(gaps, tail)
    return {"food_eaten": len(food_steps),
            "mean_gap": float(gaps.mean()) if len(gaps) else None,
            "max_gap": int(gaps.max()) if len(gaps) else None,
            "steps_after_last_food": int(tail),
            "long_gap_share": float(stretches[stretches >= LONG_GAP].sum() / length)}


def run(agent, label, episodes, seed, env_config, episodes_out=None):
    # the env must offer the agent's actions (4 or 5); shaping stays as in
    # env_config (off by default), so rewards compare across agents
    env = PacmanEnv(config={**load_env_config(env_config), **agent.env_overrides})
    scores, rewards, lengths, max_levels, clean, truncs = [], [], [], [], [], []
    per_episode = []
    t0 = time.perf_counter()
    try:
        for i in range(episodes):
            obs, info = env.reset(seed=seed + i)
            agent.reset(seed=seed + i)
            n, level, total, deaths, clean_clear = 0, info["level"], 0.0, 0, False
            food_steps = []
            while True:
                obs, r, term, trunc, info = env.step(agent.act(obs))
                n += 1
                total += r
                ev = info["events"]
                if ev["eaten_dot"] or ev["eaten_energizer"]:
                    food_steps.append(n)
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
            per_episode.append({"seed": seed + i, "reward": total, "score": int(info["score"]),
                                "length": n, "level": int(level), "deaths": deaths,
                                **food_gaps(food_steps, n)})
    finally:
        env.close()
    if episodes_out:
        with open(episodes_out, "w") as f:
            for e in per_episode:
                f.write(json.dumps(e) + "\n")
    elapsed = time.perf_counter() - t0
    scores = np.array(scores)
    return {
        "agent": label,
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
            if line.startswith("| ") and line.strip() != HEADER and cells[0] and cells[0] != "---" \
                    and not set(cells[0]) <= set("-"):
                rows[cells[0]] = line
    rows[res["agent"]] = row(res)
    def order(label):
        kind = label.split()[0]
        return (AGENTS.index(kind) if kind in AGENTS else len(AGENTS), label)
    # build everything first, then replace the file in one step: a failure
    # here must not leave a truncated results file
    text = (RESULTS_INTRO + HEADER + "\n" + RULE + "\n"
            + "".join(rows[label] + "\n" for label in sorted(rows, key=order)) + tail)
    with open(path + ".tmp", "w") as f:
        f.write(text)
    os.replace(path + ".tmp", path)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--agent", choices=AGENTS, required=True)
    p.add_argument("--checkpoint", help="dqn/ppo: best.pt or checkpoint.pt from scripts/train.py "
                                        "or scripts/train_ppo.py")
    p.add_argument("--epsilon", type=float, default=0.0, help="dqn only: random action probability")
    p.add_argument("--label", help="row label in the results table (default from the agent)")
    p.add_argument("--episodes-out", help="write per-episode stats (incl. steps without food) "
                                          "as JSON lines here")
    p.add_argument("--actions", type=int, choices=(4, 5), default=5,
                   help="random/heuristic: action set; dqn uses its checkpoint's")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env-config", default=None, help="default configs/env_default.yaml")
    p.add_argument("--agent-config", default=None, help="heuristic only; default configs/heuristic.yaml")
    p.add_argument("--results", default=os.path.join(ROOT, "docs", "results.md"),
                   help="markdown file to update; empty string to skip")
    args = p.parse_args()

    agent, label = make_agent(args)
    res = run(agent, args.label or label, args.episodes, args.seed, args.env_config,
              args.episodes_out)
    print(HEADER)
    print(RULE)
    print(row(res))
    if args.results:
        update_results(args.results, res)


if __name__ == "__main__":
    main()
