#!/usr/bin/env python3
"""Training curve of a run against its start checkpoint's level.

    python scripts/plot_curve.py dqfd2 --ref-label best_v03 --ref-json runs/best_v03/ref_seed1000.json \
        --out docs/dqfd2_curve.png

Per-maze clear share (mean over the 16 mazes) and levels cleared from level
1 of the run's training evals (runs/<run>/evals.jsonl, every eval_every
steps; light) and of its supervisor evals (sv_evals.jsonl, every 10M;
markers), all on the eval protocol (20 games per maze, seeds 1000..1019).
The reference (a train_eval.py --out row of the start checkpoint on the same
protocol) is a dashed line. Supervisor rows with "episodes" 40 (seeds
1000..1039) are drawn apart from the 20-game ones (sv_evals_20games.jsonl
is read too). --final adds the final protocol (train_eval.py --out rows
labelled <run>@<N>M, 100 games per maze, seeds 2000..) as orange points,
--final-ref its row of the start checkpoint (dotted).
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLUE, GRAY, INK, MUTED = "#2a78d6", "#8a8984", "#0b0b0b", "#52514e"
ORANGE = "#eb6834"


def rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run")
    p.add_argument("--ref-json", required=True)
    p.add_argument("--ref-label", default="start")
    p.add_argument("--out", required=True)
    p.add_argument("--final", nargs="*", default=[], help="summary.jsonl files of the final protocol")
    p.add_argument("--final-ref", default=None, help="label of the start's final-protocol row")
    args = p.parse_args()
    d = os.path.join(ROOT, "runs", args.run)
    ev = [r for r in rows(os.path.join(d, "evals.jsonl")) if "phase" not in r]
    sv_all = sorted(rows(os.path.join(d, "sv_evals.jsonl")) +
                    rows(os.path.join(d, "sv_evals_20games.jsonl")), key=lambda r: r["steps"])
    sv = [r for r in sv_all if r.get("episodes", 20) == 20]
    sv40 = [r for r in sv_all if r.get("episodes") == 40]
    fin, fin_ref = {}, None
    for path in args.final:
        for r in rows(path):
            lab = r.get("label", "")
            if lab.startswith(args.run + "@") and lab.endswith("M"):
                fin[int(lab[len(args.run) + 1:-1]) * 1_000_000] = r
            elif lab == args.final_ref:
                fin_ref = r
    fin = sorted(fin.items())
    ref = rows(args.ref_json)[-1]
    s0 = ref.get("steps")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, key, name, fmt in ((axes[0], "mean_maze_cleared", "per-maze clear share (mean of 16)",
                                "{:.0%}"),
                               (axes[1], "mean_levels_cleared", "levels cleared from level 1",
                                "{:.2f}")):
        x = [(r["steps"] - s0) / 1e6 for r in ev]
        ax.plot(x, [r[key] for r in ev], color=BLUE, lw=1, alpha=0.45,
                label=f"{args.run}, every eval")
        xs = [(r["steps"] - s0) / 1e6 for r in sv if r["steps"] > s0]
        ys = [r[key] for r in sv if r["steps"] > s0]
        ax.plot(xs, ys, color=BLUE, lw=2, marker="o", ms=6, label=f"{args.run}, every 10M")
        if sv40:
            ax.plot([(r["steps"] - s0) / 1e6 for r in sv40], [r[key] for r in sv40],
                    color=BLUE, lw=2, marker="s", ms=6, mfc="white",
                    label=f"{args.run}, every 10M, 40 games")
        if fin:
            ax.plot([(st - s0) / 1e6 for st, _ in fin], [r[key] for _, r in fin], color=ORANGE,
                    lw=0, marker="D", ms=7, label="final protocol (100 games per maze)")
        if fin_ref:
            ax.axhline(fin_ref[key], color=ORANGE, lw=1.2, ls=":")
            ax.annotate(f"{args.ref_label} final {fmt.format(fin_ref[key])}", (0, fin_ref[key]),
                        xytext=(4, -12), textcoords="offset points", color=MUTED, fontsize=9)
        ax.axhline(ref[key], color=GRAY, lw=1.5, ls="--")
        ax.annotate(f"{args.ref_label} {fmt.format(ref[key])}", (0, ref[key]),
                    xytext=(4, 4), textcoords="offset points", color=MUTED, fontsize=9)
        if sv40:
            xs, ys = [(r["steps"] - s0) / 1e6 for r in sv40], [r[key] for r in sv40]
        if xs:
            ax.annotate(fmt.format(ys[-1]), (xs[-1], ys[-1]), xytext=(6, -3),
                        textcoords="offset points", color=INK, fontsize=9)
        ax.set_xlabel(f"M steps after {args.ref_label} ({s0 / 1e6:.0f}M)")
        ax.set_title(name, fontsize=11, color=INK, loc="left")
        ax.grid(axis="y", color="#e4e3df", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        if key == "mean_maze_cleared":
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle(f"{args.run} against {args.ref_label}: eval protocol, 20 games per maze, "
                 "seeds 1000..1019" + ("\n(squares: 40, seeds 1000..1039)" if sv40 else "")
                 + ("; final protocol 100 per maze, seeds 2000.." if fin else ""),
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(args.out, dpi=90, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
