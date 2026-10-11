#!/usr/bin/env python3
"""Keep the checkpoints whose training eval looks good enough for the final
protocol: train.py overwrites runs/<run>/checkpoint.pt at every eval (every
eval_every steps), the supervisor copies only every --every steps; this
copies checkpoint.pt to good_<steps>.pt whenever the eval row written with
it (evals.jsonl, same steps) has a per-maze mean >= --mean and min >= --min.

    python scripts/keep_good.py dqfd1 --mean 0.80 --min 0.55
"""

import argparse
import json
import os
import shutil
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run")
    p.add_argument("--mean", type=float, default=0.80)
    p.add_argument("--min", type=float, default=0.55)
    p.add_argument("--poll", type=float, default=10.0)
    args = p.parse_args()
    d = os.path.join(ROOT, "runs", args.run)
    ck, seen = os.path.join(d, "checkpoint.pt"), None
    while True:
        try:
            mtime = os.path.getmtime(ck)
            if mtime != seen:
                tmp = os.path.join(d, "good.tmp")
                shutil.copy(ck, tmp)
                steps = torch.load(tmp, map_location="cpu", weights_only=False)["steps"]
                rows = {}
                with open(os.path.join(d, "evals.jsonl")) as f:
                    for line in f:
                        r = json.loads(line)
                        if "phase" not in r:
                            rows[r["steps"]] = r
                r = rows.get(steps)
                if r is not None:
                    seen = mtime
                    keep = r["mean_maze_cleared"] >= args.mean and r["min_maze_cleared"] >= args.min
                    print(f"{time.strftime('%H:%M:%S')} {steps}: {r['mean_maze_cleared']:.3f} / "
                          f"{r['min_maze_cleared']:.2f}{'  kept' if keep else ''}", flush=True)
                    if keep:
                        os.replace(tmp, os.path.join(d, f"good_{steps}.pt"))
                if os.path.exists(tmp):
                    os.remove(tmp)
        except Exception as e:  # a checkpoint being replaced: next poll
            print(f"{time.strftime('%H:%M:%S')} {e!r}", flush=True)
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
