#!/usr/bin/env python3
"""Verdict of the night-2 criteria from runs/<run>/final_summary.json
(scripts/final_eval.py; 100 games, no prefixes):

  SUCCESS: level 1 cleared >= 30% with eps 0 or eps 0.05
  PARTIAL: mean food >= 170 or level 1 cleared > 0% (either eps)
  FAILURE: otherwise

    python scripts/night2_verdict.py dqn7 dqn8
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def verdict(summary):
    evals = [summary["eps0"], summary["eps005"]]
    if any(e["level1_cleared"] >= 0.30 for e in evals):
        return "SUCCESS"
    if any(e["mean_food"] >= 170 or e["level1_cleared"] > 0 for e in evals):
        return "PARTIAL"
    return "FAILURE"


if __name__ == "__main__":
    for run in sys.argv[1:]:
        with open(os.path.join(ROOT, "runs", run, "final_summary.json")) as f:
            s = json.load(f)
        e0, e5 = s["eps0"], s["eps005"]
        print(f"{run}: {verdict(s)} | eps 0: food {e0['mean_food']:.1f}, level 1 "
              f"{e0['level1_cleared']:.0%}, reward {e0['mean_reward']:.1f}, stuck {e0['stuck']:.0%} | "
              f"eps 0.05: food {e5['mean_food']:.1f}, level 1 {e5['level1_cleared']:.0%}, "
              f"reward {e5['mean_reward']:.1f}")
