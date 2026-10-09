#!/usr/bin/env python3
"""v0.3 on all 16 mazes (BC -> RL night): verdict, table and plot from
evaluate.py --levels results on the final protocol (--seed 0 --episodes 100
--level-episodes 25 --stop-at-clear):

  SUCCESS: >= 50% of the games started on each of the 16 mazes clear it
  PARTIAL: >= 30% on each
  FAILURE: otherwise

    python scripts/v03_16_report.py runs/levels/heuristic_v03_16.json \\
        runs/levels/bc_final.json runs/levels/rl_final.json --plot docs/levels_v03.png

Prints a markdown table per agent (16 per-maze clear shares, their minimum,
levels cleared in a row from level 1: mean / median / max) and each agent's
verdict. --plot draws per-maze clears of all agents (grouped bars) and the
distribution of levels cleared in a row from level 1.
"""

import argparse
import json

import numpy as np

MAZES = range(1, 17)
COLORS = ("#8a8a85", "#2a78d6", "#d9822b", "#3a9e5c")
INK, MUTED = "#1f1f1e", "#6b6b66"


def verdict(res):
    low = min(res["per_level"][str(m)]["cleared"] for m in MAZES)
    return "SUCCESS" if low >= 0.50 else "PARTIAL" if low >= 0.30 else "FAILURE"


def table(results):
    head = ("| agent | " + " | ".join(f"M{m}" for m in MAZES)
            + " | min | levels in a row from level 1: mean / median / max | verdict |")
    lines = [head, "|---" * (len(MAZES) + 4) + "|"]
    for r in results:
        f1 = r["from_level1"]
        cells = [f"{r['per_level'][str(m)]['cleared']:.0%}" for m in MAZES]
        low = min(r["per_level"][str(m)]["cleared"] for m in MAZES)
        lines.append(f"| {r['agent']} | " + " | ".join(cells) + f" | {low:.0%} | "
                     f"{f1['mean_levels_cleared']:.2f} / {f1['median_levels_cleared']:g} / "
                     f"{f1['max_levels_cleared']} | {verdict(r)} |")
    n = results[0]["per_level"]["1"]["episodes"]
    lines.append(f"\n{n} games per maze (seeds 0..{n - 1}), "
                 f"{results[0]['from_level1']['episodes']} games from level 1 "
                 f"(seeds {results[0]['seeds']}).")
    return "\n".join(lines)


def plot(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw={"width_ratios": [2.6, 1]})
    k = len(results)
    width = 0.8 / k
    x = np.array(list(MAZES))
    for i, r in enumerate(results):
        share = [r["per_level"][str(m)]["cleared"] * 100 for m in MAZES]
        a.bar(x + (i - (k - 1) / 2) * width, share, width=width, color=COLORS[i % len(COLORS)],
              label=r["agent"], zorder=2)
    a.axhline(50, color=INK, lw=0.8, ls="--", zorder=3)
    a.axhline(30, color=MUTED, lw=0.8, ls=":", zorder=3)
    n = results[0]["per_level"]["1"]["episodes"]
    a.set(xticks=x, ylim=(0, 110), xlabel="maze (start level)",
          ylabel=f"games clearing the maze, % ({n} per maze)")
    a.set_title("clears per maze (dashed: 50%, dotted: 30%)", loc="left", color=INK)
    a.legend(frameon=False, loc="upper center", ncol=k)
    top = max(max(r["from_level1"]["levels_cleared"]) for r in results)
    bins = np.arange(top + 1)
    for i, r in enumerate(results):
        counts = np.bincount(r["from_level1"]["levels_cleared"], minlength=top + 1)
        b.bar(bins + (i - (k - 1) / 2) * width, counts, width=width,
              color=COLORS[i % len(COLORS)], zorder=2,
              label=f"{r['agent']}: mean {r['from_level1']['mean_levels_cleared']:.2f}")
    b.set(xlabel="levels cleared in a row from level 1", ylabel="games", xticks=bins)
    b.set_title(f"{results[0]['from_level1']['episodes']} games from level 1", loc="left",
                color=INK)
    b.legend(frameon=False)
    for ax in (a, b):
        ax.grid(axis="y", color="#e4e4e0", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=110)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("results", nargs="+")
    p.add_argument("--plot", help="PNG with all agents")
    args = p.parse_args()
    results = []
    for path in args.results:
        with open(path) as f:
            results.append(json.load(f))
    print(table(results))
    for r in results:
        print(f"{r['agent']}: {verdict(r)}")
    if args.plot:
        plot(results, args.plot)


if __name__ == "__main__":
    main()
