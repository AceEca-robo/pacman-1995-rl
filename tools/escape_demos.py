#!/usr/bin/env python3
"""Escape demos: for every recorded death, perturbed continuations of the
checkpoint from k ticks before it; the best one that survives becomes a
demo episode.

    python tools/escape_demos.py --config configs/dqfd4.yaml \
        --out data/escape_demos.npz --windows-out data/death_windows_v04.json

1. Windows: every T-20 death prefix of --prefixes is replayed once to find
   how long the death's life (and level) had lasted at the death
   ("life_tick"); the prefixes with life_tick >= --min-life are written to
   --windows-out (the env's prefix_mix "before_death: [lo, hi]" cuts them).
2. Rollouts (tools/avoidability.py run): from k ticks before each death
   (--horizons, default 80 100 120, inside its life), --rollouts
   continuations of --checkpoint with eps --eps (random actions; seeded by
   death, horizon and rollout). A rollout survives if it loses no life in
   --window ticks. The best survivor of a state: all survivors last the
   whole window, so the one that ate the most food (the first on a tie).
3. Demo transitions: each best survivor is replayed in an env with the
   training config's env (--config: its observation and rewards, no
   hunger limit or prefixes) and written as one episode (truncated after
   the window, so its last transitions bootstrap) with the config's n_step
   and gamma, in the format of SharedReplay.save_demo (one stream), plus
   per-episode metadata arrays ep_*. The replay checks that the episode
   loses no life and eats the same food as the rollout.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from avoidability import AGENT, prefix_at, run, select  # noqa: E402


def life_tick(prefix):
    """Ticks the life (and level) of the death had lasted at the death, for
    a T-20 death prefix: replays it and finds the last change of lives or
    level."""
    from env.pacman_env import ACTIONS, PacmanEnv
    env = _env_cache.get("env")
    if env is None:
        env = _env_cache["env"] = PacmanEnv(config={"start_level": 1})
    env._restart_game(prefix["seed"], int(prefix.get("start_level", prefix["level"])))
    st = env._recv()
    last, change = (st["lives"], st["level"]), 0
    for t, a in enumerate(prefix["actions"], 1):
        env._res["conn"].sendall(ACTIONS[a])
        st = env._recv()
        if (st["lives"], st["level"]) != last:
            last, change = (st["lives"], st["level"]), t
    assert last == (prefix["lives"], prefix["level"])
    return prefix["steps"] + prefix["before_death"] - change


_env_cache = {}


def _life_ticks(prefixes):
    return [life_tick(p) for p in prefixes]


def windows(prefixes, workers):
    t20 = [p for p in prefixes if p["before_death"] == 20]
    chunks = [t20[i::workers] for i in range(workers)]
    with mp.get_context("spawn").Pool(workers) as pool:
        out = pool.map(_life_ticks, chunks)
    ticks = [None] * len(t20)
    for i, part in enumerate(out):
        ticks[i::workers] = part
    return [{**p, "life_tick": int(t)} for p, t in zip(t20, ticks)]


def build_demos(env_cfg, n_step, gamma, deaths, best, window):
    """best: [(death index, k, rollout dict)] -> (npz arrays, per-episode
    metadata)."""
    from agents.dqn import pack_grids
    from agents.shared_replay import SharedReplay, StreamWriter
    from env.pacman_env import PacmanEnv, obs_dims, grid_extras
    from agents.dqn import n_planes
    n_ch, vec_dim = obs_dims(env_cfg)
    planes = n_planes(n_ch, grid_extras(env_cfg))
    size = sum(len(b["actions"]) + 1 for _, _, b in best) + 2
    replay = SharedReplay.create([], [size], planes, vec_dim)
    env = PacmanEnv(config=env_cfg)
    extras = env.grid_extras
    meta = []
    try:
        stream = StreamWriter(replay, [0], n_step, gamma)

        def start(i):
            di, k, b = best[i]
            env.replay_prefix(prefix_at(deaths[di], k))
            obs, _ = env._after_reset(int(deaths[di]["seed"]))
            return obs

        def packed(obs):
            return pack_grids(obs["grid"][None], extras), obs["vec"][None]

        obs = start(0)
        stream.begin(*packed(obs))
        for i, (di, k, b) in enumerate(best):
            acts, food, deaths_seen = b["actions"], 0, 0
            for t, a in enumerate(acts):
                obs, rew, term, trunc, info = env.step(a)
                ev = info["events"]
                food += ev["eaten_dot"] + ev["eaten_energizer"]
                deaths_seen += ev["death"]
                assert not term
                g, v = packed(obs)
                if t < len(acts) - 1:
                    stream.step(np.array([a]), np.array([rew], np.float32), g, v,
                                [False], [False], g[:0], v[:0])
                else:  # cut after the window: truncated, the next episode follows
                    if i + 1 < len(best):
                        ng, nv = packed(start(i + 1))
                    else:
                        ng, nv = np.zeros_like(g), np.zeros_like(v)
                    stream.step(np.array([a]), np.array([rew], np.float32), ng, nv,
                                [False], [True], g, v)
            if deaths_seen or food != b["food"] or len(acts) != window:
                raise RuntimeError(f"escape replay of death {di} at T-{k} diverged")
            d = deaths[di]
            meta.append({"death": d["id"], "horizon": k, "food": food, "maze": d["maze"],
                         "seed": d["seed"], "kind": d["kind"]})
        arrays = {k: getattr(replay, k)[:replay.demo_frames].copy()
                  for k in SharedReplay.DEMO_ARRAYS}
        arrays.update(sizes=replay.sizes[:1].copy(), written=replay.written[:1].copy(),
                      final=replay.final[:1].copy())
    finally:
        env.close()
        replay.close()
    return arrays, meta


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--prefixes", default=os.path.join(ROOT, "data", "death_prefixes_v04.json"))
    p.add_argument("--checkpoint", default=os.path.join(ROOT, "runs", "best_v04", "best.pt"))
    p.add_argument("--config", required=True, help="training config: env, n_step, gamma")
    p.add_argument("--horizons", type=int, nargs="+", default=[80, 100, 120])
    p.add_argument("--rollouts", type=int, default=40)
    p.add_argument("--window", type=int, default=200)
    p.add_argument("--eps", type=float, default=0.3)
    p.add_argument("--min-life", type=int, default=80)
    p.add_argument("--deaths", type=int, default=None, help="a random subset (default all)")
    p.add_argument("--workers", type=int, default=30)
    p.add_argument("--out", required=True)
    p.add_argument("--windows-out", required=True)
    args = p.parse_args()
    import torch
    from env.pacman_env import load_env_config
    from train import load_config
    from train_eval import Scorer
    t0 = time.time()
    with open(args.prefixes) as f:
        data = json.load(f)
    win = windows(data["prefixes"], args.workers)
    kept = [w for w in win if w["life_tick"] >= args.min_life]
    with open(args.windows_out, "w") as f:
        json.dump({"agent": data["agent"], "n_actions": data["n_actions"],
                   "source": os.path.relpath(args.prefixes, ROOT), "min_life": args.min_life,
                   "prefixes": kept}, f)
    print(f"windows: {len(kept)} of {len(win)} deaths with a life >= {args.min_life} ticks "
          f"-> {args.windows_out} ({time.time() - t0:.0f} s)", flush=True)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    obs_cfg = (ck.get("env_config") or {}).get("obs") or {}
    scorer = Scorer(args.checkpoint, "cuda" if torch.cuda.is_available() else "cpu")
    deaths = select(data["prefixes"], args.deaths or len(win), None, 0)
    ticks = {(w["seed"], w["steps"]): w["life_tick"] for w in win}
    for d in deaths:  # jobs_for skips horizons past "level_tick": use the life's length
        d["level_tick"] = ticks[(d["seed"], d["death_step"] - 20)]
    res = run(scorer, obs_cfg, deaths, args.horizons, args.rollouts, args.window,
              (args.eps, 0.0), args.workers, agents=(AGENT,), greedy=False)
    best = sorted([(di, k, r["best"]) for (di, k), r in res.items()
                   if r["eligible"] and r["best"] is not None])
    states = Counter(k for (di, k), r in res.items() if r["eligible"])
    print(f"rollouts done ({time.time() - t0:.0f} s): {len(best)} of {sum(states.values())} "
          f"states have a survivor", flush=True)

    cfg = load_config(os.path.join(ROOT, args.config))
    env_cfg = load_env_config(cfg["env_config"])
    env_cfg = {**env_cfg, "hunger_limit": 0, "prefix_prob": 0.0, "max_episode_steps": None}
    arrays, meta = build_demos(env_cfg, cfg["n_step"], cfg["gamma"], deaths, best,
                               args.window)
    for key in ("death", "horizon", "food", "maze", "seed"):
        arrays[f"ep_{key}"] = np.array([m[key] for m in meta], np.int64)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, args.out)
    with_exit = sorted({di for di, _, _ in best})
    summary = {
        "deaths": len(deaths),
        "deaths_eligible": len({di for (di, k), r in res.items() if r["eligible"]}),
        "deaths_with_escape": len(with_exit),
        "states": dict(sorted(states.items())),
        # launched states that failed the replay's level / lives check
        # (should be 0: life_tick and the check agree)
        "states_not_eligible": sum(not r["eligible"] for r in res.values()),
        "states_with_escape": dict(sorted(Counter(k for _, k, _ in best).items())),
        "episodes": len(best), "transitions": int(arrays["valid"].sum()),
        "frames": int(arrays["sizes"].sum()),
        "escape_by_maze": dict(sorted(Counter(deaths[di]["maze"] for di in with_exit).items())),
        "deaths_by_maze": dict(sorted(Counter(d["maze"] for d in deaths).items())),
        "escape_by_kind": dict(Counter(deaths[di]["kind"] for di in with_exit).most_common()),
        "deaths_by_kind": dict(Counter(d["kind"] for d in deaths).most_common()),
        "seconds": round(time.time() - t0)}
    with open(os.path.splitext(args.out)[0] + "_summary.json", "w") as f:
        json.dump({"args": vars(args), **summary}, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
