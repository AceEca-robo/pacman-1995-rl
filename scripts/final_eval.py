#!/usr/bin/env python3
"""Final evaluation of a run's checkpoint for the autonomous session.

    python scripts/final_eval.py dqn5 [--checkpoint best_food.pt]

Runs scripts/evaluate.py: 100 games with eps 0 and with eps 0.05 (seeds
0..99, no shaping/hunger), and the ghost-free check (3 games eps 0, the game
is deterministic then; 20 games eps 0.05). Each adds its row to
docs/results.md; per-game stats go to runs/<run>/final_*.jsonl. Prints and
saves (runs/<run>/final_summary.json) the criteria numbers:

  SUCCESS: level 1 cleared >= 30% or mean food >= 160
  PARTIAL: mean food >= 155 or mean reward > 75
  FAILURE: otherwise
(on the 100 eps-0 games; the same numbers on seeds 20..99 only, which were
not used to pick best_food.pt, are reported alongside).
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def run_eval(run, ck, label, out, episodes, epsilon, env_config=None):
    import torch
    algo = torch.load(ck, map_location="cpu", weights_only=False).get("algo", "dqn")
    cmd = [PY, os.path.join(ROOT, "scripts", "evaluate.py"), "--agent", algo, "--checkpoint", ck,
           "--episodes", str(episodes), "--seed", "0", "--label", label, "--episodes-out", out]
    if epsilon:
        cmd += ["--epsilon", str(epsilon)]
    if env_config:
        cmd += ["--env-config", env_config]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if res.returncode != 0:
        sys.exit(res.stderr)
    print(res.stdout.strip().splitlines()[-1], flush=True)
    with open(out) as f:
        return [json.loads(line) for line in f]


def numbers(games):
    return {"games": len(games),
            "mean_food": float(np.mean([g["food_eaten"] for g in games])),
            "level1_cleared": float(np.mean([g["level"] >= 2 for g in games])),
            "mean_reward": float(np.mean([g["reward"] for g in games])),
            "median_score": float(np.median([g["score"] for g in games])),
            "stuck": float(np.mean([g["length"] >= 10000 for g in games])),
            "after_last_food": float(np.median([g["steps_after_last_food"] for g in games])),
            "long_gap_share": float(np.mean([g["long_gap_share"] for g in games]))}


def verdict(n):
    if n["level1_cleared"] >= 0.30 or n["mean_food"] >= 160:
        return "SUCCESS"
    if n["mean_food"] >= 155 or n["mean_reward"] > 75:
        return "PARTIAL"
    return "FAILURE"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run")
    p.add_argument("--checkpoint", default="best_food.pt")
    args = p.parse_args()
    d = os.path.join(ROOT, "runs", args.run)
    ck = os.path.join(d, args.checkpoint)
    name = f"{args.run}/{args.checkpoint}"
    summary = {"checkpoint": name}
    for eps, tag in ((0.0, "eps0"), (0.05, "eps005")):
        games = run_eval(args.run, ck, f"{name}, eps {eps:g}", os.path.join(d, f"final_{tag}.jsonl"),
                         100, eps)
        summary[tag] = numbers(games)
        summary[tag + "_seeds20_99"] = numbers([g for g in games if g["seed"] >= 20])
    for eps, n, tag in ((0.0, 3, "noghosts_eps0"), (0.05, 20, "noghosts_eps005")):
        games = run_eval(args.run, ck, f"{name}, no ghosts, eps {eps:g}",
                         os.path.join(d, f"final_{tag}.jsonl"), n, eps,
                         os.path.join(ROOT, "configs", "env_noghosts.yaml"))
        summary[tag] = numbers(games)
    summary["verdict"] = verdict(summary["eps0"])
    summary["verdict_seeds20_99"] = verdict(summary["eps0_seeds20_99"])
    with open(os.path.join(d, "final_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    for k in ("eps0", "eps0_seeds20_99", "eps005", "noghosts_eps0", "noghosts_eps005"):
        n = summary[k]
        print(f"{k:18s} food {n['mean_food']:6.1f}  level1 {n['level1_cleared']:4.0%}  "
              f"reward {n['mean_reward']:7.1f}  score {n['median_score']:6.0f}  "
              f"stuck {n['stuck']:4.0%}  after-last-food {n['after_last_food']:6.0f}")
    print("verdict:", summary["verdict"], "| on seeds 20..99:", summary["verdict_seeds20_99"])


if __name__ == "__main__":
    main()
