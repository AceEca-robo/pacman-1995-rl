#!/usr/bin/env python3
"""Verify sim/ against the real game, state line by state line.

    python scripts/verify_sim.py --policy random --games 50 --ticks 3000
    python scripts/verify_sim.py --policy sticky --games 50 --ticks 3000
    python scripts/verify_sim.py --policy agent --checkpoint runs/best_v04/best.pt

Game i: env seed --seed + i, start level drawn from --levels (default
random:1-16) with numpy seed --seed + i. random: uniform over U/D/L/R/N;
sticky: keeps an action, a new random one with probability 0.1 per step
(reaches energizers, eats ghosts); agent: a DQN checkpoint (greedy) plays
through PacmanEnv and the sim follows its actions. Each game is --ticks
steps (state lines) in one game process, through game overs. The game's
raw JSON line and sim's state_json() must be byte-equal (with --sim c and
--policy agent also the observation the agent got from PacmanEnv and the
C sim's obs()); the first
difference of a game is reported and the game stops there.
"""

import argparse
import json
import multiprocessing
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from env.pacman_env import parse_start_level  # noqa: E402
from sim.verify import compare, diff_lines  # noqa: E402

COVERAGE = ("level_cleared", "level_over_16", "game_over", "ghost_eaten", "bonus_spawned",
            "bonus_point_eaten", "bonus_life_eaten", "super_extra_move", "ghost_newdir_no_dir",
            "ghost_otherdir_no_dir")


def counting_sim():
    """PacmanSim that counts how often the rare branches ran (py only)."""
    from sim import pacman_sim as ps

    class CountingSim(ps.PacmanSim):
        def reset(self, *a, **k):
            self.cov = dict.fromkeys(COVERAGE, 0)
            return super().reset(*a, **k)

        def _levelup(self):
            self.cov["level_cleared"] += 1
            self.cov["level_over_16"] += self.level >= ps.LEVELS
            super()._levelup()

        def _gd_start(self):
            if hasattr(self, "level"):
                self.cov["game_over"] += 1
            super()._gd_start()

        def _ghost_die(self, g):
            self.cov["ghost_eaten"] += 1
            super()._ghost_die(g)

        def _bonus_tick(self):
            had = self.bon
            super()._bonus_tick()
            self.cov["bonus_spawned"] += had == ps.NO_BONUS and self.bon != ps.NO_BONUS

        def _check(self):
            bon, lives, score = self.bon, self.lives, self.score
            super()._check()
            if bon != ps.NO_BONUS and self.bon == ps.NO_BONUS:
                self.cov["bonus_point_eaten" if bon == ps.BONUS_POINT else "bonus_life_eaten"] += 1

        def _advance(self, d):
            self.cov["super_extra_move"] += self.phase == 1 and self.pst == ps.SUPER
            super()._advance(d)

        def _newdir(self, g):
            self.cov["ghost_newdir_no_dir"] += g.d > ps.RIGHT
            super()._newdir(g)

        def _otherdir(self, g):
            self.cov["ghost_otherdir_no_dir"] += g.d > ps.RIGHT
            return super()._otherdir(g)
    return CountingSim


def sim_class(name):
    if name == "py":
        return counting_sim()
    from sim.csim import CSim
    return CSim


def game_plan(seed, i, levels):
    lo, hi = parse_start_level(levels)
    rng = np.random.default_rng(seed + i)
    return seed + i, int(rng.integers(lo, hi + 1)), rng


def policy_actions(policy, rng, n):
    if policy == "random":
        return rng.integers(0, 5, n).tolist()
    acts, a = [], int(rng.integers(0, 5))
    for _ in range(n):
        if rng.random() < 0.1:
            a = int(rng.integers(0, 5))
        acts.append(a)
    return acts


def run_random(task):
    policy, seed, i, levels, ticks, sim = task
    s, level, rng = game_plan(seed, i, levels)
    n, d, sim = compare(s, level, policy_actions(policy, rng, ticks), sim_class(sim))
    return {"coverage": getattr(sim, "cov", None), "seed": s, "level": level, "steps": n - 1 if d is None else n,
            "diverged": d is not None, "at": n if d else None,
            "diff": diff_lines(*d) if d else None,
            "game_line": d[0].decode() if d else None}


def run_agent(args):
    """The agent plays the real game through PacmanEnv; the sim follows."""
    from agents.dqn import DQNAgent
    from env.pacman_env import PacmanEnv, load_env_config
    from sim.verify import diff_lines
    agent = DQNAgent(os.path.abspath(args.checkpoint), epsilon=0.0, seed=args.seed)
    env = PacmanEnv(config={**load_env_config(None), **agent.env_overrides})
    raw = []

    def recv():
        line = env._res["rfile"].readline()
        if not line:
            raise RuntimeError("game closed the connection")
        env._fresh = False
        raw.append(line)
        return json.loads(line)
    env._recv = recv
    Sim = sim_class(args.sim)
    out = []
    for i in range(args.games):
        s, level, _ = game_plan(args.seed, i, args.levels)
        raw.clear()
        obs, _ = env.reset(seed=s, options={"start_level": level})
        agent.reset(seed=s)
        sim = Sim()
        sim.reset(s, level)
        res = {"seed": s, "level": level, "steps": 0, "diverged": False, "at": None,
               "diff": None, "game_line": None}
        cell_safety = bool((agent.env_overrides.get("obs") or {}).get("cell_safety"))
        for t in range(args.ticks + 1):
            line = sim.state_json()
            if raw[-1] != line:
                res.update(diverged=True, at=t, diff=diff_lines(raw[-1], line),
                           game_line=raw[-1].decode())
                break
            if args.sim == "c":     # the observation the agent saw, rebuilt in C
                o = sim.obs(cell_safety)
                if not (np.array_equal(o["grid"], obs["grid"]) and np.array_equal(o["vec"], obs["vec"])):
                    res.update(diverged=True, at=t, diff=["observation differs"])
                    break
            res["steps"] = t
            if t == args.ticks:
                break
            a = agent.act(obs)
            obs, *_ = env.step(a)
            sim.advance(int(a))
        res["coverage"] = getattr(sim, "cov", None)
        out.append(res)
        print(f"game {i}: seed {s} level {level} steps {res['steps']}"
              + (f" DIVERGED at {res['at']}: {res['diff']}" if res["diverged"] else ""),
              flush=True)
    env.close()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--policy", choices=("random", "sticky", "agent"), default="random")
    p.add_argument("--checkpoint", default=os.path.join(ROOT, "runs", "best_v04", "best.pt"))
    p.add_argument("--games", type=int, default=50)
    p.add_argument("--ticks", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--levels", default="random:1-16")
    p.add_argument("--sim", choices=("py", "c"), default="py")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", help="write the per-game results as JSON")
    args = p.parse_args()
    t0 = time.time()
    if args.policy == "agent":
        res = run_agent(args)
    else:
        tasks = [(args.policy, args.seed, i, args.levels, args.ticks, args.sim)
                 for i in range(args.games)]
        with multiprocessing.Pool(args.workers) as pool:
            res = pool.map(run_random, tasks)
        for i, r in enumerate(res):
            if r["diverged"]:
                print(f"game {i}: seed {r['seed']} level {r['level']} DIVERGED at {r['at']}: {r['diff']}")
    steps = sum(r["steps"] for r in res)
    if res[0]["coverage"] is not None:
        cov = {k: sum(r["coverage"][k] for r in res) for k in COVERAGE}
        print("coverage:", ", ".join(f"{k} {v}" for k, v in cov.items()))
    bad = sum(r["diverged"] for r in res)
    print(f"{args.policy} ({args.sim}): {len(res)} games, {steps} steps compared, "
          f"{bad} diverged, {time.time() - t0:.1f} s")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "steps": steps, "coverage": cov if res[0]["coverage"] else None, "diverged": bad, "games": res}, f, indent=1)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
