#!/usr/bin/env python3
"""Tables and plots from evaluate.py --levels results.

    python scripts/levels_report.py runs/levels/heuristic.json runs/levels/mm1.json ...
    python scripts/levels_report.py runs/levels/mm1.json --plot docs/levels_mm1.png \\
        --reference runs/levels/heuristic.json

Prints a markdown table: one row per result file, the share of games that
clear each start level, and levels cleared per game from level 1.
--plot draws the first file's per-level clears (bars) next to the reference's
(markers) and its levels-cleared distribution from level 1.
"""

import argparse
import json

import numpy as np

LEVELS = 16
BLUE, GRAY, INK, MUTED = "#2a78d6", "#8a8a85", "#1f1f1e", "#6b6b66"


def load(path):
    with open(path) as f:
        return json.load(f)


def levels_of(res):
    """The start levels of a result file's per-level games (1..16 for old files)."""
    return res.get("levels") or list(range(1, LEVELS + 1))


def table(results):
    levels = levels_of(results[0])
    head = ("| agent | from level 1: mean / median / max levels cleared | "
            + " | ".join(f"L{lv}" for lv in levels) + " | min |")
    lines = [head, "|---|---|" + "---|" * (len(levels) + 1)]
    for r in results:
        f1 = r["from_level1"]
        cells = [f"{r['per_level'][str(lv)]['cleared']:.0%}" for lv in levels]
        lines.append(f"| {r['agent']} | {f1['mean_levels_cleared']:.2f} / "
                     f"{f1['median_levels_cleared']:g} / {f1['max_levels_cleared']} | "
                     + " | ".join(cells)
                     + f" | {min(r['per_level'][str(lv)]['cleared'] for lv in levels):.0%} |")
    return "\n".join(lines)


def plot(res, ref, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    levels = np.array(levels_of(res))
    share = [res["per_level"][str(lv)]["cleared"] * 100 for lv in levels]
    wide = len(levels) > 6
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4) if wide else (10, 4),
                               gridspec_kw={"width_ratios": [2.2, 1] if wide else [1, 1]})
    a.bar(levels, share, width=0.6, color=BLUE, label=res["agent"], zorder=2)
    if ref:
        rshare = [ref["per_level"][str(lv)]["cleared"] * 100 for lv in levels]
        a.plot(levels, rshare, "D", color=GRAY, ms=6, label=ref["agent"], zorder=3)
    n = res["per_level"][str(levels[0])]["episodes"]
    a.set(xticks=levels, ylim=(0, 118), xlabel="start level (maze)",
          ylabel=f"games clearing the start level, % ({n} per level)")
    a.set_title(f"{res['agent']}: clears per maze", loc="left", color=INK)
    a.legend(frameon=False, loc="upper center", ncol=2)
    f1 = res["from_level1"]
    counts = np.bincount(f1["levels_cleared"])
    b.bar(np.arange(len(counts)), counts, width=0.6, color=BLUE, zorder=2)
    b.set(xlabel="levels cleared in one game from level 1", ylabel="games",
          xticks=np.arange(len(counts)))
    b.set_title(f"mean {f1['mean_levels_cleared']:.2f}, median {f1['median_levels_cleared']:g} "
                f"({f1['episodes']} games)", loc="left", color=INK)
    for ax in (a, b):
        ax.grid(axis="y", color="#e4e4e0", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=110)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("results", nargs="+")
    p.add_argument("--plot", help="PNG for the first result file")
    p.add_argument("--reference", help="result file drawn as markers on the plot")
    args = p.parse_args()
    results = [load(x) for x in args.results]
    print(table(results))
    if args.plot:
        plot(results[0], load(args.reference) if args.reference else None, args.plot)


if __name__ == "__main__":
    main()
