#!/usr/bin/env python3
"""Compare two death reports (scripts/death_report.py JSON) side by side.

    python scripts/compare_deaths.py runs/v03_deaths/dqfd1@110M.json \
        runs/dqfd2_deaths/dqfd2@134M.json --labels best_v03 dqfd2@134M \
        --out docs/dqfd2_deaths_compare.png

Left: deaths per game by kind; right: deaths per game on each maze.
Prints the same numbers as a markdown table.
"""

import argparse
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = ["#2a78d6", "#eb6834"]  # categorical slots 1, 2
KINDS = ["squeezed", "supertime ran out", "from behind", "head-on", "standing"]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("reports", nargs=2)
    p.add_argument("--labels", nargs=2, required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    reps = [json.load(open(r)) for r in args.reports]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [1, 2.2]})
    w = 0.38
    print("| | " + " | ".join(args.labels) + " |\n|---|---|---|")
    rows = {}
    for i, (rep, lab) in enumerate(zip(reps, args.labels)):
        n = rep["summary"]["games"]
        kinds = Counter(d["kind"] for d in rep["deaths"])
        y = [kinds[k] / n for k in KINDS]
        a1.bar(np.arange(len(KINDS)) + (i - 0.5) * w, y, w - 0.02, color=COLORS[i], label=lab)
        mz = Counter(d["maze"] for d in rep["deaths"])
        a2.bar(np.arange(1, 17) + (i - 0.5) * w,
               [mz[m] / sum(g["maze"] == m for g in rep["games"]) for m in range(1, 17)],
               w - 0.02, color=COLORS[i], label=lab)
        s = rep["summary"]
        rows.setdefault("cleared", []).append(f"{s['cleared']:.1%}")
        rows.setdefault("deaths per game", []).append(f"{s['deaths'] / n:.2f}")
        for k in KINDS:
            rows.setdefault(f"{k} per game", []).append(f"{kinds[k] / n:.3f} ({kinds[k] / s['deaths']:.1%})")
        rows.setdefault("< 20 food left", []).append(f"{s['share_endgame_lt20']:.1%}")
        rows.setdefault("in the life's first 200 ticks", []).append(f"{s['share_life_first200']:.1%}")
    for k, v in rows.items():
        print(f"| {k} | " + " | ".join(v) + " |")
    a1.set_xticks(range(len(KINDS)), KINDS, rotation=20, ha="right")
    a1.set_ylabel("deaths per game")
    a1.set_title("Deaths by kind (16 mazes x 50 games)")
    a2.set_xticks(range(1, 17))
    a2.set_xlabel("maze")
    a2.set_title("Deaths per game on each maze")
    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", color="#dddddd", lw=0.6)
        a.set_axisbelow(True)
        a.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)


if __name__ == "__main__":
    main()
