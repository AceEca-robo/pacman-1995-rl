#!/usr/bin/env python3
"""Keep the checkpoints whose training eval looks good enough for the final
protocol: train.py overwrites runs/<run>/checkpoint.pt at every eval (every
eval_every steps), the supervisor copies only every --every steps; this
copies checkpoint.pt to good_<steps>.pt whenever the eval row written with
it (evals.jsonl, same steps) has a per-maze mean >= --mean and min >= --min.

    python scripts/keep_good.py dqfd1 --mean 0.80 --min 0.55

--extra-episodes N scores each such checkpoint on N more games (from level 1
and per maze, seeds eval_seed + eval_episodes ..; scripts/train_eval.py)
and judges the union with the training eval's games, e.g. the training
eval's 20 games (seeds 1000..1019) + 20 more = seeds 1000..1039; the merged
rows go to runs/<run>/evals_merged.jsonl.
"""

import argparse
import json
import os
import shutil
import subprocess
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv", "bin", "python")


def merged(d, ck, r, extra):
    """The training eval row r (episodes games from seed eval_seed) joined
    with extra more games on ck: per-maze shares and levels as the mean over
    all games."""
    import yaml
    with open(os.path.join(d, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    n, seed = cfg["eval_episodes"], cfg["eval_seed"]
    tmp = os.path.join(d, "good_extra.tmp.jsonl")
    if os.path.exists(tmp):
        os.remove(tmp)
    subprocess.run([PY, os.path.join(ROOT, "scripts", "train_eval.py"),
                    "--config", os.path.join(d, "config.yaml"), "--checkpoint", ck,
                    "--seed", str(seed + n), "--episodes", str(extra), "--out", tmp],
                   check=True, capture_output=True, cwd=ROOT)
    with open(tmp) as f:
        x = json.loads(f.read().strip().splitlines()[-1])
    os.remove(tmp)
    w = n / (n + extra)
    mazes = sorted(int(k[4:-8]) for k in r if k.startswith("maze") and k.endswith("_cleared"))
    out = {"steps": r["steps"], "episodes": n + extra, "seed": seed}
    for k in ["mean_levels_cleared", "level1_cleared", "mean_food",
              *(f"maze{lv}_cleared" for lv in mazes)]:
        out[k] = w * r[k] + (1 - w) * x[k]
    shares = [out[f"maze{lv}_cleared"] for lv in mazes]
    out["mean_maze_cleared"] = sum(shares) / len(shares)
    out["min_maze_cleared"] = min(shares)
    out["max_levels_cleared"] = max(r.get("max_levels_cleared", 0), x["max_levels_cleared"])
    with open(os.path.join(d, "evals_merged.jsonl"), "a") as f:
        f.write(json.dumps(out) + "\n")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run")
    p.add_argument("--mean", type=float, default=0.80)
    p.add_argument("--min", type=float, default=0.55)
    p.add_argument("--poll", type=float, default=10.0)
    p.add_argument("--extra-episodes", type=int, default=0,
                   help="judge on the training eval's games plus this many more (see above)")
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
                    if args.extra_episodes:
                        r = merged(d, tmp, r, args.extra_episodes)
                    # shares are counts / games: 0.8 can come out as 0.7999999999999998
                    keep = r["mean_maze_cleared"] >= args.mean - 1e-9 and \
                        r["min_maze_cleared"] >= args.min - 1e-9
                    print(f"{time.strftime('%H:%M:%S')} {steps}: {r['mean_maze_cleared']:.3f} / "
                          f"{r['min_maze_cleared']:.2f} / levels {r['mean_levels_cleared']:.2f}{'  kept' if keep else ''}", flush=True)
                    if keep:
                        os.replace(tmp, os.path.join(d, f"good_{steps}.pt"))
                if os.path.exists(tmp):
                    os.remove(tmp)
        except Exception as e:  # a checkpoint being replaced: next poll
            print(f"{time.strftime('%H:%M:%S')} {e!r}", flush=True)
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
