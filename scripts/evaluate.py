#!/usr/bin/env python3
"""Evaluate a baseline agent on Pacman1995-v0.

    python scripts/evaluate.py --agent heuristic --episodes 100 --seed 0
    python scripts/evaluate.py --agent dqn --checkpoint runs/dqn0/best.pt
    python scripts/evaluate.py --agent heuristic --levels levels.json \
        --episodes 100 --level-episodes 20 --workers 8

Episode i is played with env seed (and agent seed) seed + i. Prints a
markdown table and updates the agent's row in docs/results.md.

--levels runs the multi-maze suite instead (no results.md row): --episodes
games started on level 1 (levels cleared per game: median, mean; the
selection metric since v0.3 is the mean) and, for each of the 16 levels,
--level-episodes games started on it (share that clears the start level).
Games run to game over or --max-steps ticks.
"""

import argparse
import json
import multiprocessing
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
from env.pacman_env import LEVELS, PacmanEnv, load_env_config  # noqa: E402

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
    agent = DQNAgent(path, epsilon=args.epsilon, seed=args.seed,
                     device=getattr(args, "device", None))
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


def make_env(agent, env_config, **overrides):
    # the env must offer the agent's actions (4 or 5); shaping stays as in
    # env_config (off by default), so rewards compare across agents
    return PacmanEnv(config={**load_env_config(env_config), **overrides, **agent.env_overrides})


def play(env, agent, seed, options=None):
    """One episode with env/agent seed `seed`; returns its stats (level =
    highest level reached, start_level = the level it started on)."""
    obs, info = env.reset(seed=seed, options=options)
    agent.reset(seed=seed)
    start = info["level"]
    n, level, total, deaths, clean_clear = 0, start, 0.0, 0, False
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
    return {"seed": seed, "reward": total, "score": int(info["score"]), "length": n,
            "level": int(level), "deaths": deaths, "start_level": int(start),
            "levels_cleared": int(level - start), "truncated": bool(trunc),
            "clean_clear": clean_clear, **food_gaps(food_steps, n)}


def run(agent, label, episodes, seed, env_config, episodes_out=None):
    env = make_env(agent, env_config)
    per_episode = []
    t0 = time.perf_counter()
    try:
        for i in range(episodes):
            per_episode.append(play(env, agent, seed + i))
    finally:
        env.close()
    scores = [e["score"] for e in per_episode]
    rewards = [e["reward"] for e in per_episode]
    lengths = [e["length"] for e in per_episode]
    max_levels = [e["level"] for e in per_episode]
    truncs = [e["truncated"] for e in per_episode]
    clean = [e.pop("clean_clear") for e in per_episode]
    for e in per_episode:  # the per-episode file keeps its old fields
        for k in ("start_level", "levels_cleared", "truncated"):
            del e[k]
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


# --- multi-maze suite (--levels) -------------------------------------------

_worker = {}


def _init_worker(args):
    import torch
    torch.set_num_threads(1)
    if args.agent in ("dqn", "ppo"):
        args.device = "cpu"  # many processes: no CUDA context each
    agent, _ = make_agent(args)
    _worker["agent"] = agent
    _worker["env"] = make_env(agent, args.env_config, max_episode_steps=args.max_steps)


def _play_task(task):
    level, seed = task
    e = play(_worker["env"], _worker["agent"], seed, {"start_level": level})
    e.pop("clean_clear")
    return e


def levels_suite(args, label):
    """Games from level 1 (args.episodes) and from each level 1..16
    (args.level_episodes), seeds args.seed..; returns the summary dict."""
    tasks = [(1, args.seed + i) for i in range(max(args.episodes, args.level_episodes))]
    tasks += [(lv, args.seed + i) for lv in range(2, LEVELS + 1) for i in range(args.level_episodes)]
    # long games first, so the pool does not end on one straggler
    tasks.sort(key=lambda t: t[0] != 1)
    t0 = time.perf_counter()
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(args,)) as pool:
        games = pool.map(_play_task, tasks, chunksize=1)
    by = {}
    for g in games:
        by.setdefault(g["start_level"], []).append(g)
    from1 = sorted(by[1], key=lambda g: g["seed"])[:args.episodes]
    cleared = np.array([g["levels_cleared"] for g in from1])
    per_level = {}
    for lv in range(1, LEVELS + 1):
        gs = sorted(by[lv], key=lambda g: g["seed"])[:args.level_episodes]
        per_level[lv] = {
            "episodes": len(gs),
            "cleared": float(np.mean([g["levels_cleared"] >= 1 for g in gs])),
            "mean_food": float(np.mean([g["food_eaten"] for g in gs])),
            "mean_length": float(np.mean([g["length"] for g in gs])),
            "truncated": float(np.mean([g["truncated"] for g in gs])),
        }
    return {
        "agent": label,
        "seeds": f"{args.seed}..{args.seed + args.episodes - 1}",
        "max_steps": args.max_steps,
        "from_level1": {
            "episodes": len(from1),
            "mean_levels_cleared": float(cleared.mean()),
            "median_levels_cleared": float(np.median(cleared)),
            "max_levels_cleared": int(cleared.max()),
            "levels_cleared": cleared.tolist(),
            "truncated": float(np.mean([g["truncated"] for g in from1])),
            "mean_score": float(np.mean([g["score"] for g in from1])),
        },
        "per_level": per_level,
        "min_level_cleared": min(v["cleared"] for v in per_level.values()),
        "time_s": round(time.perf_counter() - t0, 1),
        "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                 capture_output=True, text=True).stdout.strip(),
        "games": games,
    }


def levels_markdown(res):
    f1 = res["from_level1"]
    lines = [f"{res['agent']}: from level 1, {f1['episodes']} games ({res['seeds']}): "
             f"levels cleared mean {f1['mean_levels_cleared']:.2f}, "
             f"median {f1['median_levels_cleared']:g}, max {f1['max_levels_cleared']}",
             "", "| start level | " + " | ".join(str(lv) for lv in range(1, LEVELS + 1)) + " |",
             "|---|" + "---|" * LEVELS,
             "| cleared | " + " | ".join(f"{res['per_level'][lv]['cleared']:.0%}"
                                         for lv in range(1, LEVELS + 1)) + " |"]
    return "\n".join(lines)


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
    p.add_argument("--levels", metavar="JSON",
                   help="run the multi-maze suite (see above) and write its results here")
    p.add_argument("--level-episodes", type=int, default=20,
                   help="--levels: games started on each of the 16 levels")
    p.add_argument("--max-steps", type=int, default=30000,
                   help="--levels: ticks before a game is cut (env max_episode_steps)")
    p.add_argument("--workers", type=int, default=8, help="--levels: processes (CPU)")
    args = p.parse_args()

    if args.levels:
        label = args.label or make_agent(args)[1]
        res = levels_suite(args, label)
        with open(args.levels, "w") as f:
            json.dump(res, f, indent=1)
        print(levels_markdown(res))
        return
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
