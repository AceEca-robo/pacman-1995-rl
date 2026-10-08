#!/usr/bin/env python3
"""v0.3 ("four mazes") verdict and table from evaluate.py --levels results
(final protocol: --seed 0 --episodes 100 --level-episodes 25
--level-range 1-4 --stop-at-clear):

  SUCCESS: >= 50% of the games started on each of mazes 1..4 clear it
  PARTIAL: >= 25% on each
  FAILURE: otherwise

    python scripts/v03_verdict.py runs/levels/v03s0.json --reference runs/levels/heuristic_v03.json

Prints a markdown table (agent vs reference: per-maze clears, levels cleared
in a row from level 1 and the share of games from level 1 that clear at
least 1..4 levels) and the verdict line.
"""

import argparse
import json

import numpy as np

MAZES = (1, 2, 3, 4)


def verdict(res):
    shares = [res["per_level"][str(m)]["cleared"] for m in MAZES]
    if min(shares) >= 0.50:
        return "SUCCESS"
    if min(shares) >= 0.25:
        return "PARTIAL"
    return "FAILURE"


def row(res):
    f1 = res["from_level1"]
    cleared = np.array(f1["levels_cleared"])
    cells = [f"{res['per_level'][str(m)]['cleared']:.0%}" for m in MAZES]
    runs = [f"{np.mean(cleared >= k):.0%}" for k in range(1, 5)]
    n = res["per_level"][str(MAZES[0])]["episodes"]
    return (f"| {res['agent']} | " + " | ".join(cells)
            + f" | {min(res['per_level'][str(m)]['cleared'] for m in MAZES):.0%}"
            + f" | {f1['mean_levels_cleared']:.2f} / {f1['median_levels_cleared']:g} / "
              f"{f1['max_levels_cleared']} | " + " | ".join(runs) + f" | {n} / {f1['episodes']} |")


def table(results):
    head = ("| agent | maze 1 | maze 2 | maze 3 | maze 4 | min | from level 1: levels cleared "
            "mean / median / max | >= 1 | >= 2 | >= 3 | >= 4 | games per maze / from level 1 |")
    return "\n".join([head, "|---" * 13 + "|"] + [row(r) for r in results])


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("result")
    p.add_argument("--reference", help="e.g. the heuristic's result file")
    args = p.parse_args()
    with open(args.result) as f:
        res = json.load(f)
    results = [res]
    if args.reference:
        with open(args.reference) as f:
            results.append(json.load(f))
    print(table(results))
    shares = ", ".join(f"{res['per_level'][str(m)]['cleared']:.0%}" for m in MAZES)
    print(f"\n{res['agent']}: {verdict(res)} (mazes 1-4: {shares}; "
          f"SUCCESS >= 50% on each, PARTIAL >= 25% on each)")


if __name__ == "__main__":
    main()
