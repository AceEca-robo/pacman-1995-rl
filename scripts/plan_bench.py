#!/usr/bin/env python3
"""Decision latency of the planner on ONE game (the play.py --agent planner
case), per config: mean / p95 / max seconds per decision and decisions/s.

    python scripts/plan_bench.py configs/plan_honest_lb.yaml configs/plan_grid_h60k16.yaml

Each config plays the same games (--levels, seeds from --seed) in the sim
for --decisions decisions each; the GPU should be otherwise idle. The game
in real time gives a decision 0.25 s (0.125 s while pacman is super: every
state line follows a timing(super) call that sleeps up to 0.25 / 2 s,
game/timing.cc; measured with play.py --viz, see docs/results.md).
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.plan_eval import Planner, QNet, Stall, load_cfg  # noqa: E402
from sim.csim import CSim  # noqa: E402


def bench(cfg, net, levels, seed, n):
    planner = Planner(net, cfg)
    lat, second = [], []
    for level in levels:
        sim, stall = CSim(seed, level), Stall(cfg)
        prev = sim.info()
        for t in range(n):
            if prev[0] == 0:
                break
            t0 = time.perf_counter()
            o = sim.obs(net.cell_safety)
            q = net(o["grid"][None], o["vec"][None])
            a, st = planner.choose(sim.buf[None], prev[None], q, [[seed, level, t]],
                                   [stall.visits()])
            lat.append(time.perf_counter() - t0)
            second.append(bool(st["second_round"][0] or st["breaker"][0]))
            sim.advance(int(a[0]))
            info = sim.info()
            stall.update(int(info[2]), info[1] != prev[1] or info[4] < prev[4])
            prev = info
    lat, second = np.array(lat), np.array(second)
    return {"decisions": len(lat), "mean_s": float(lat.mean()),
            "p95_s": float(np.percentile(lat, 95)), "max_s": float(lat.max()),
            "decisions_per_s": float(1 / lat.mean()), "second_round": float(second.mean()),
            "mean_s_second_round": float(lat[second].mean()) if second.any() else None}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("configs", nargs="+")
    p.add_argument("--levels", default="9,14", help="comma list of start levels")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--decisions", type=int, default=300, help="per game")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", help="JSON with the results")
    args = p.parse_args()
    levels = [int(x) for x in args.levels.split(",")]
    net = None
    res = {}
    for path in args.configs:
        cfg = load_cfg(path)
        if net is None:
            net = QNet(os.path.join(ROOT, cfg["checkpoint"]), args.device)
            bench({**cfg, "plan": {**cfg["plan"], "horizon": 5}}, net, levels[:1], args.seed, 5)
        r = bench(cfg, net, levels, args.seed, args.decisions)
        res[os.path.basename(path)] = r
        print(f"{os.path.basename(path)}: {r['decisions_per_s']:.1f} decisions/s, mean "
              f"{r['mean_s'] * 1000:.0f} ms, p95 {r['p95_s'] * 1000:.0f} ms, max "
              f"{r['max_s'] * 1000:.0f} ms (second round {r['second_round']:.0%})", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
