#!/usr/bin/env python3
"""Summary table of all runs' final 100-game evaluations (seeds 0..99, eps 0,
no shaping/hunger) as the section "## Summary of all runs" of docs/results.md
(replaced on each run; a "## " section after the table, which evaluate.py keeps).

    python scripts/summary_table.py

Rows come from per-game files (evaluate.py --episodes-out): runs/night_final/
for the earlier runs, runs/<run>/final_eps0.jsonl (+ final_eps005.jsonl) from
scripts/final_eval.py for dqn5 and later. Seed groups (<name>, <name>s1,
<name>s2) get a mean +- std line.
"""

import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
RESULTS = os.path.join(ROOT, "docs", "results.md")
HEADING = "## Summary of all runs"

EARLIER = [  # (run, checkpoint, what changed, per-game file)
    ("dqn0", "best.pt (4.75M)", "baseline DQN, 5M", "night_final/final_dqn0_100.jsonl"),
    ("dqn2", "best.pt (4.5M)", "+ heuristic warm start", "night_final/final_dqn2_100.jsonl"),
    ("dqn1", "best.pt (11.0M, by mean)", "20M steps, eps over 3M", "night_final/final_dqn1_mean_100.jsonl"),
    ("dqn3b", "best.pt (4.75M)", "+ distance shaping, 4 actions", "night_final/final_dqn3b_100.jsonl"),
    ("dqn4", "best.pt (5.0M)", "+ hunger_limit 200, 4 actions", "night_final/final_dqn4_100.jsonl"),
    ("ppo0", "best.pt (2.0M)", "PPO, hunger_limit 200, 4 actions", "night_final/final_ppo0_100.jsonl"),
]
LATER = {  # run -> what changed (checkpoint: best_food.pt)
    "dqn1": "20M steps, eps over 3M; seed 0, picked by food",
    "dqn5": "dqn1 + food-distance channel + steps since food, 10M",
    "dqn6": "dqn5 + n_step 5, gamma 0.995",
}


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def nums(games):
    return {"food": np.mean([g["food_eaten"] for g in games]),
            "level1": np.mean([g["level"] >= 2 for g in games]),
            "reward": np.mean([g["reward"] for g in games]),
            "score": np.median([g["score"] for g in games]),
            "max_score": max(g["score"] for g in games),
            "stuck": np.mean([g["length"] >= 10000 for g in games]),
            "after": np.median([g["steps_after_last_food"] for g in games])}


def row(run, ck, what, n, n005):
    eps = f"{n005['food']:.1f} / {n005['reward']:.1f}" if n005 else "-"
    return (f"| {run} | {ck} | {what} | {n['food']:.1f} | {n['level1']:.0%} | {n['reward']:.1f} | "
            f"{n['score']:.0f} | {n['max_score']} | {n['stuck']:.0%} | {n['after']:.0f} | {eps} |")


def best_food_steps(run):
    path = os.path.join(RUNS, run, "best_food.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)["steps"]
    return None


def main():
    lines = [HEADING, "",
             "Final evaluation of each run's chosen checkpoint: 100 games, seeds 0..99, "
             "eps 0, no shaping or hunger_limit (dqn5 on: best_food.pt, the checkpoint with "
             "the most food in the supervisor's 20-game evaluations). Food = dots + "
             "energizers eaten per game (172 on level 1). Stuck = games that hit the "
             "10000-step limit. Last column: the same checkpoint with eps 0.05.", "",
             "| run | checkpoint | change | mean food | level 1 cleared | mean reward | "
             "median score | max score | stuck | steps after last food | eps 0.05: food / reward |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    groups = {}
    for run, ck, what, f in EARLIER:
        lines.append(row(run, ck, what, nums(load(os.path.join(RUNS, f))), None))
    later = sorted(r for r in os.listdir(RUNS)
                   if os.path.exists(os.path.join(RUNS, r, "final_eps0.jsonl")))
    for run in later:
        base = re.sub(r"s\d+$", "", run)
        what = LATER.get(base, "")
        if base != run:
            what = f"{base} setup, seed {run[len(base) + 1:]}"
        steps = best_food_steps(run)
        ck = f"best_food.pt ({steps / 1e6:.1f}M)" if steps else "best_food.pt"
        n = nums(load(os.path.join(RUNS, run, "final_eps0.jsonl")))
        p005 = os.path.join(RUNS, run, "final_eps005.jsonl")
        n005 = nums(load(p005)) if os.path.exists(p005) else None
        lines.append(row(run, ck, what, n, n005))
        groups.setdefault(base, []).append(n)
    for base, ns in groups.items():
        if len(ns) > 1:
            lines += ["", f"{base} over {len(ns)} seeds (mean +- std): "
                      + ", ".join(f"{k} {np.mean([n[k] for n in ns]):.1f} +- "
                                  f"{np.std([n[k] for n in ns]):.1f}"
                                  for k in ("food", "reward", "score"))
                      + f", level 1 cleared {np.mean([n['level1'] for n in ns]):.0%}"]
    block = "\n".join(lines) + "\n"
    # a "## " section after the evaluate.py table: evaluate.py keeps those
    with open(RESULTS) as f:
        text = f.read()
    if HEADING in text:
        start = text.index(HEADING)
        nxt = text.find("\n## ", start + 1)
        text = text[:start] + block + (text[nxt + 1:] if nxt >= 0 else "")
    else:
        text = text.rstrip("\n") + "\n\n" + block
    with open(RESULTS, "w") as f:
        f.write(text)
    print(block)


if __name__ == "__main__":
    main()
