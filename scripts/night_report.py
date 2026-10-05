#!/usr/bin/env python3
"""Tables for docs/night_report.md from the run logs.

    python scripts/night_report.py dqn1 dqn3b [--ref runs/night_ref] > part.md

Per run: the training script's own evals (every 250k steps, 20 greedy games
on eval seeds 1000000..) and the supervisor's evals (every 500k steps, 20
games on seeds 0..19, with steps without food), speed and supervisor events.
--ref: per-episode files of reference agents (evaluate.py --episodes-out)
for the steps-without-food comparison.
"""

import argparse
import glob
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def fmt_m(steps):
    return f"{steps / 1e6:.2f}M"


def food_row(episodes):
    def med(key):
        vals = [e[key] for e in episodes if e[key] is not None]
        return np.median(vals) if vals else float("nan")
    return (f"{np.median([e['reward'] for e in episodes]):.1f} | "
            f"{np.median([e['score'] for e in episodes]):.0f} | "
            f"{max(e['score'] for e in episodes)} | "
            f"{np.mean([e['level'] >= 2 for e in episodes]):.0%} | "
            f"{np.median([e['length'] for e in episodes]):.0f} | "
            f"{med('food_eaten'):.0f} | {med('mean_gap'):.1f} | {med('max_gap'):.0f} | "
            f"{med('steps_after_last_food'):.0f} | "
            f"{np.mean([e['long_gap_share'] for e in episodes]):.0%}")


FOOD_HEADER = ("| | median reward | median score | max score | level 1 cleared | median length "
               "| food eaten | mean gap | longest gap | steps after last food | in 30+ gaps |\n"
               "|---|---|---|---|---|---|---|---|---|---|---|")


def run_section(run):
    out = [f"### {run}", ""]
    evals = read_jsonl(os.path.join(RUNS, run, "evals.jsonl"))
    out += ["Training-script evals (every 250k steps, 20 greedy games, seeds 1000000..1000019):", "",
            "| steps | median reward | median score | max score | level 1 cleared | mean length |",
            "|---|---|---|---|---|---|"]
    for e in evals:
        eps = e["episodes"]
        out.append(f"| {fmt_m(e['steps'])} | {e['median_reward']:.1f} | {e['median_score']:.0f} | "
                   f"{max(x['score'] for x in eps)} | {e['level1_cleared']:.0%} | "
                   f"{e['mean_length']:.0f} |")
    sv = sorted(glob.glob(os.path.join(RUNS, run, "sv_eval_*.jsonl")),
                key=lambda p: int(re.search(r"sv_eval_(\d+)", p).group(1)))
    if sv:
        out += ["", "Supervisor evals (every 500k steps, 20 greedy games, seeds 0..19, no shaping); "
                "gaps are steps between eaten food, medians over games:", "",
                FOOD_HEADER.replace("| |", "| steps |", 1)]
        for path in sv:
            steps = int(re.search(r"sv_eval_(\d+)", path).group(1))
            out.append(f"| {fmt_m(steps)} | {food_row(read_jsonl(path))} |")
    speeds = [int(s.replace(",", "")) for s in
              re.findall(r"([\d,]+) steps/s", open(os.path.join(RUNS, f"{run}.log")).read())]
    if speeds:
        tail = speeds[-50:]
        out += ["", f"Speed: median {np.median(speeds):,.0f} steps/s over the run, "
                f"min {min(speeds):,}, last 50 log lines median {np.median(tail):,.0f}."]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs", nargs="+")
    p.add_argument("--ref", default=os.path.join(RUNS, "night_ref"))
    args = p.parse_args()

    out = []
    refs = sorted(glob.glob(os.path.join(args.ref, "*.jsonl")))
    if refs:
        out += ["### Reference agents (20 games, seeds 0..19, no shaping)", "", FOOD_HEADER]
        for path in refs:
            name = os.path.basename(path)[:-6].replace("runs_", "").replace("_best.pt", "/best.pt")
            out.append(f"| {name} | {food_row(read_jsonl(path))} |")
        out.append("")
    for run in args.runs:
        out += run_section(run) + [""]
    log = os.path.join(RUNS, "supervisor.log")
    if os.path.exists(log):
        events = [line.rstrip() for line in open(log)
                  if re.search(r"CRASH|slow|error|failed|finished|skipped", line)]
        out += ["### Supervisor events", "", "```", *(events or ["(none)"]), "```"]
    print("\n".join(out))


if __name__ == "__main__":
    main()
