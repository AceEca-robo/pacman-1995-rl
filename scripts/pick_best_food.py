#!/usr/bin/env python3
"""Pick runs/<run>/best_food.pt: the supervisor-evaluated checkpoint
(ck_<steps>.pt, 20 games on seeds 0..19) with the highest mean food eaten
(dots + energizers per game). Ties go to the earlier checkpoint.

    python scripts/pick_best_food.py dqn5
"""

import glob
import json
import os
import re
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")


def food_by_checkpoint(run):
    """{steps: mean food eaten} for evaluated checkpoints that still exist."""
    out = {}
    for path in glob.glob(os.path.join(RUNS, run, "sv_eval_*.jsonl")):
        steps = int(re.search(r"sv_eval_(\d+)", path).group(1))
        if not os.path.exists(os.path.join(RUNS, run, f"ck_{steps}.pt")):
            continue
        with open(path) as f:
            games = [json.loads(line) for line in f if line.strip()]
        if games:
            out[steps] = float(np.mean([g["food_eaten"] for g in games]))
    return out


def pick(run):
    """Copies the best checkpoint to best_food.pt; returns (steps, mean food) or None."""
    foods = food_by_checkpoint(run)
    if not foods:
        return None
    steps = max(sorted(foods), key=lambda s: foods[s])
    dst = os.path.join(RUNS, run, "best_food.pt")
    shutil.copy(os.path.join(RUNS, run, f"ck_{steps}.pt"), dst + ".tmp")
    os.replace(dst + ".tmp", dst)
    with open(os.path.join(RUNS, run, "best_food.json"), "w") as f:
        json.dump({"steps": steps, "mean_food": foods[steps], "all": foods}, f, indent=1)
    return steps, foods[steps]


if __name__ == "__main__":
    for run in sys.argv[1:]:
        print(run, pick(run))
