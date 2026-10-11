#!/usr/bin/env python3
"""Score a BC or DQN checkpoint on a training config's eval protocol (the
games train.py's evaluate() plays: eval_episodes from level 1, then
eval_level_episodes started on each of eval_levels, stopped at the clear and
cut at eval_level_max_steps; seeds eval_seed..), greedy.

    python scripts/train_eval.py --config configs/dqfd.yaml --checkpoint runs/bc0_dagger1/bc_best.pt

Prints the per-maze clear shares, their mean and minimum, and levels cleared
from level 1; --out appends the summary as a JSON line.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from agents.bc import load_bc  # noqa: E402
from agents.dqn import GridDecoder, QNetwork  # noqa: E402
from env.pacman_env import PacmanEnv, grid_extras, load_env_config, obs_dims  # noqa: E402
from train import evaluate, load_config  # noqa: E402


class Scorer:
    """q_values(packed, vecs) of a BC (logits) or DQN (Q values) checkpoint."""

    def __init__(self, path, device):
        ck = torch.load(path, map_location=device, weights_only=False)
        self.device = device
        if "net" in ck:  # BC
            self.net, _ = load_bc(path, device)
        else:
            n_channels, vec_dim = obs_dims(ck.get("env_config") or {})
            self.net = QNetwork(ck["config"]["network"], ck.get("n_actions", 5), vec_dim,
                                n_channels).to(device).eval()
            self.net.load_state_dict(ck["learner"]["online"])
        env_cfg = ck.get("env_config") or {}
        self.decode = GridDecoder(device, obs_dims(env_cfg)[0], grid_extras(env_cfg))

    @torch.no_grad()
    def q_values(self, packed, vecs):
        g = self.decode(torch.as_tensor(packed, device=self.device))
        return self.net(g, torch.as_tensor(vecs, device=self.device)).cpu().numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--label", default=None)
    p.add_argument("--out", default=None, help="append the summary here (JSON lines)")
    p.add_argument("--episodes", type=int, default=None,
                   help="games from level 1 and per maze (default the config's), seeds eval_seed..")
    p.add_argument("--seed", type=int, default=None, help="eval_seed (default the config's)")
    p.add_argument("--seed-blocks", action="store_true",
                   help="maze lv on seeds seed + (lv - 1) * episodes .. (evaluate.py --seed-blocks)")
    p.add_argument("--games-out", default=None, help="write every game's result here (JSON)")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.episodes:
        cfg["eval_episodes"] = cfg["eval_level_episodes"] = args.episodes
    if args.seed is not None:
        cfg["eval_seed"] = args.seed
    cfg["eval_seed_blocks"] = args.seed_blocks
    env_cfg = load_env_config(cfg["env_config"])
    eval_env_cfg = {**env_cfg, "shaping": {**env_cfg["shaping"], "enabled": False},
                    "hunger_limit": 0, "prefix_prob": 0.0, "endgame_dot": {"enabled": False},
                    "start_level": 1}
    if cfg.get("eval_max_episode_steps"):
        eval_env_cfg["max_episode_steps"] = cfg["eval_max_episode_steps"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    envs = [PacmanEnv(config=eval_env_cfg) for _ in range(cfg.get("eval_envs") or 20)]
    try:
        res = evaluate(Scorer(args.checkpoint, device), envs, cfg)
    finally:
        for e in envs:
            e.close()
    lv1 = [r for r in res if "maze" not in r]
    shares = {lv: float(np.mean([r["levels_cleared"] >= 1 for r in res if r.get("maze") == lv]))
              for lv in cfg.get("eval_levels") or []}
    out = {"label": args.label or args.checkpoint, "config": args.config,
           "episodes": cfg["eval_level_episodes"],
           "seed": cfg["eval_seed"], "seed_blocks": args.seed_blocks,
           "mean_levels_cleared": float(np.mean([r["levels_cleared"] for r in lv1])),
           "median_levels_cleared": float(np.median([r["levels_cleared"] for r in lv1])),
           "max_levels_cleared": int(max(r["levels_cleared"] for r in lv1)),
           "level1_cleared": float(np.mean([r["level"] >= 2 for r in lv1])),
           "mean_food": float(np.mean([r["food"] for r in lv1])),
           "mean_maze_cleared": float(np.mean(list(shares.values()))) if shares else None,
           "min_maze_cleared": min(shares.values()) if shares else None,
           **{f"maze{lv}_cleared": v for lv, v in shares.items()}}
    print(" ".join(f"M{lv} {v:.0%}" for lv, v in shares.items()))
    print(f"{out['label']}: per-maze mean {out['mean_maze_cleared']:.1%}, "
          f"min {out['min_maze_cleared']:.0%}; from level 1: levels {out['mean_levels_cleared']:.2f}, "
          f"level 1 cleared {out['level1_cleared']:.0%}, food {out['mean_food']:.0f}")
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(out) + "\n")
    if args.games_out:
        with open(args.games_out, "w") as f:
            json.dump({**out, "games": res}, f, default=int)


if __name__ == "__main__":
    main()
