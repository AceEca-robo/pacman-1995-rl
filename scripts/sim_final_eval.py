#!/usr/bin/env python3
"""The final protocol (train_eval.py --episodes 100 --seed 2000 --seed-blocks)
played in the sim (sim/csim.py) instead of the real game, resumable.

    python scripts/sim_final_eval.py --checkpoint runs/dist0/best.pt --label dist0 \
        --out runs/distill_eval/dist0.json
    python scripts/sim_final_eval.py ... --resume

The same games as train.py's eval_tasks with the config's protocol: 100
games from level 1 (seeds 2000..2099, cut at eval_max_episode_steps) and
100 games on each of eval_levels started there (maze lv on seeds 2000 +
(lv - 1) * 100 .., stopped at the clear, cut at eval_level_max_steps),
greedy. The sim repeats the game state for state (scripts/verify_sim.py;
the C observation equals PacmanEnv's), so only float differences between
devices can change a game. Every finished game is appended to
<out minus .json>.jsonl; --resume skips the games already there. The end
result <out> has train_eval.py --games-out's format (scripts/final_table.py
reads it).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from plan_eval import QNet, batch_advance, batch_info, batch_obs  # noqa: E402
from train import eval_tasks, load_config  # noqa: E402


def play(net, tasks, cfg, batch, sink, log_every=60):
    from sim.csim import SIZE, CSim
    max1 = int(cfg["eval_max_episode_steps"])
    max_lv = int(cfg["eval_level_max_steps"])
    queue, games, rows = list(tasks), [], np.zeros((0, SIZE), np.uint8)
    t0, last, done = time.time(), time.time(), 0
    while queue or games:
        while len(games) < batch and queue:
            seed, level, stop = queue.pop(0)
            sim = CSim()
            sim.reset(seed, level)
            info = sim.info()
            games.append({"seed": seed, "start": level, "stop": stop, "length": 0, "level": level,
                          "eats": int(info[4]), "food": 0, "score": int(info[3])})
            rows = np.concatenate([rows, sim.buf[None]])
        grid, vec = batch_obs(net, rows)
        act = net(grid, vec).argmax(1).astype(np.int32)
        batch_advance(rows, act)
        info = batch_info(rows)
        keep = []
        for i, g in enumerate(games):
            g["length"] += 1
            lives, level, eats = int(info[i, 0]), int(info[i, 1]), int(info[i, 4])
            if level == g["level"] and eats < g["eats"]:
                g["food"] += g["eats"] - eats
            g["eats"], g["level"], g["score"] = eats, max(g["level"], level), int(info[i, 3])
            cut = g["length"] >= (max_lv if g["stop"] else max1)
            if lives == 0 or cut or (g["stop"] and g["level"] > g["start"]):
                res = {"seed": g["seed"], "score": g["score"], "length": g["length"],
                       "level": g["level"], "food": g["food"],
                       "levels_cleared": g["level"] - g["start"], "truncated": bool(cut and lives)}
                if g["stop"]:
                    res["maze"] = g["start"]
                sink(res)
                done += 1
            else:
                keep.append(i)
        games, rows = [games[i] for i in keep], rows[keep]
        if time.time() - last >= log_every:
            last = time.time()
            print(f"{time.strftime('%H:%M:%S')} {done} of {len(tasks)} games done, "
                  f"{len(games)} running, {time.time() - t0:.0f} s", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", default=os.path.join(ROOT, "runs", "dqfd2", "config.yaml"),
                   help="training config with the eval protocol (best_v04's run)")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=2000)
    p.add_argument("--batch", type=int, default=600, help="games in lock-step")
    p.add_argument("--levels", default=None, help="comma list: only these mazes (and no level-1 "
                   "games unless 0 is listed) — for spot checks")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    cfg.update(eval_episodes=args.episodes, eval_level_episodes=args.episodes,
               eval_seed=args.seed, eval_seed_blocks=True)
    tasks = eval_tasks(cfg)
    if args.levels:
        keep = {int(x) for x in args.levels.split(",")}
        tasks = [t for t in tasks if (t[1] if t[2] else 0) in keep]
    jsonl = os.path.splitext(args.out)[0] + ".jsonl"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    done = []
    if os.path.exists(jsonl):
        if not args.resume:
            sys.exit(f"{jsonl} exists; use --resume to continue it")
        with open(jsonl) as f:
            done = [json.loads(line) for line in f if line.strip()]
    key = {(g["seed"], g.get("maze", 0)) for g in done}
    todo = [t for t in tasks if (t[0], t[1] if t[2] else 0) not in key]
    print(f"{len(tasks)} games, {len(done)} done already, {len(todo)} to play", flush=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = QNet(os.path.join(ROOT, args.checkpoint) if not os.path.isabs(args.checkpoint)
               else args.checkpoint, device)
    if device == "cpu":
        torch.set_num_threads(os.cpu_count())
    # level-1 games are the longest: start them first
    todo.sort(key=lambda t: t[2])
    with open(jsonl, "a") as f:
        def sink(g):
            f.write(json.dumps(g) + "\n")
            f.flush()
        play(net, todo, cfg, args.batch, sink)
    with open(jsonl) as f:
        games = [json.loads(line) for line in f if line.strip()]
    games.sort(key=lambda g: (g.get("maze", 0), g["seed"]))
    lv1 = [g for g in games if "maze" not in g]
    mazes = sorted({g["maze"] for g in games if "maze" in g})
    shares = {lv: float(np.mean([g["levels_cleared"] >= 1 for g in games if g.get("maze") == lv]))
              for lv in mazes}
    out = {"label": args.label, "checkpoint": args.checkpoint, "protocol": "sim", "device": device,
           "episodes": args.episodes, "seed": args.seed, "seed_blocks": True,
           "mean_maze_cleared": float(np.mean(list(shares.values()))) if shares else None,
           "min_maze_cleared": min(shares.values()) if shares else None,
           **{f"maze{lv}_cleared": v for lv, v in shares.items()}, "games": games}
    if lv1:
        cl = np.array([g["levels_cleared"] for g in lv1])
        out.update(mean_levels_cleared=float(cl.mean()), median_levels_cleared=float(np.median(cl)),
                   max_levels_cleared=int(cl.max()),
                   level1_cleared=float(np.mean([g["level"] >= 2 for g in lv1])))
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(" ".join(f"M{lv} {v:.0%}" for lv, v in shares.items()))
    print(f"{args.label}: per-maze mean {out['mean_maze_cleared'] or 0:.1%}, "
          f"min {out['min_maze_cleared'] or 0:.0%}; from level 1: "
          f"levels {out.get('mean_levels_cleared', float('nan')):.2f} "
          f"(median {out.get('median_levels_cleared', float('nan')):.0f}, "
          f"max {out.get('max_levels_cleared', 0)})", flush=True)


if __name__ == "__main__":
    main()
