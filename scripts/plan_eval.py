#!/usr/bin/env python3
"""Evaluate "best_v04 + rollout": lookahead planning in the sim (sim/).

    python scripts/plan_eval.py                    # planner, configs/plan.yaml
    python scripts/plan_eval.py --mode greedy      # best_v04 alone, same games
    python scripts/plan_eval.py --levels 7,13 --level-episodes 5   # a pilot

At every step (state line) of a game the planner tries the 4 directions:
for each, K rollouts in a clone of the sim: the direction, then H-1 steps
of best_v04 greedy with probability-epsilon uniform noise (a different
noise draw per rollout). Directions where more than death_threshold of the
rollouts lose a life are dropped; of the rest the one with the highest Q
of the network is played; if all are dropped, the one with the fewest
deaths (then the highest Q). "N" (keep going) moves pacman exactly like
its current direction, so it counts as that direction with Q = max(Q[d],
Q[N]) and is played as N when Q[N] is the larger (pacman standing still:
N is not a candidate). The candidates are rolled out lazily in Q order
(first the best, the other three only if it is dropped): the same choice as
rolling out all four, as the noise of each (step, direction, rollout) is
drawn from its own seed.

The games are played in the sim itself, which repeats the real game state
for state (scripts/verify_sim.py); --replay-check replays some finished
games' actions in the real game and compares the final state line.
Protocol as scripts/evaluate.py --levels --stop-at-clear (see the config).
"""

import argparse
import json
import multiprocessing
import os
import sys
import time
from collections import Counter, deque

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

LEVELS = 16
N_ACTION = 4    # env action "N"


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


class QNet:
    def __init__(self, checkpoint, device):
        import torch
        from agents.dqn import DQNAgent
        torch.set_num_threads(1)
        agent = DQNAgent(checkpoint, epsilon=0.0, device=device)
        self.torch, self.net, self.device = torch, agent.net, agent.device
        self.cell_safety = bool((agent.env_overrides.get("obs") or {}).get("cell_safety"))
        self.env_overrides = agent.env_overrides
        assert agent.n_actions == 5
        self.forwards = 0

    def __call__(self, grid, vec):
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(grid), 1024):
                g = torch.from_numpy(grid[i:i + 1024]).to(self.device)
                v = torch.from_numpy(vec[i:i + 1024]).to(self.device)
                out.append(self.net(g, v).float().cpu().numpy())
        self.forwards += 1
        return np.concatenate(out)


def batch_obs(net, buf):
    from sim.csim import SimBatch
    b = SimBatch(0)
    b.buf = buf
    return b.obs(cell_safety=net.cell_safety)


def batch_advance(buf, actions):
    from sim.csim import SimBatch
    b = SimBatch(0)
    b.buf = buf
    b.advance(actions)


def batch_reseed(buf, seeds):
    from sim.csim import SimBatch
    b = SimBatch(0)
    b.buf = buf
    b.reseed(seeds)


def batch_info(buf):
    from sim.csim import SimBatch
    b = SimBatch(0)
    b.buf = buf
    return b.info()


class Planner:
    def __init__(self, net, cfg):
        p = cfg["plan"]
        self.net, self.H, self.K = net, int(p["horizon"]), int(p["rollouts"])
        self.eps, self.thr = float(p["epsilon"]), float(p["death_threshold"])
        self.ghost_rng = p.get("ghost_rng", "oracle")
        assert self.ghost_rng in ("oracle", "reseed"), self.ghost_rng
        lb = p.get("loop_breaker") or {}
        self.penalty = float(lb.get("penalty", 0.0))
        self.sim_steps = 0

    def rollout(self, rows, first, u, ra, reseeds=None):
        """rows: (R, SIZE) sims; first: (R,) first action; u, ra: (R, H)
        noise; reseeds: (R,) random() seeds of the clones ("honest"
        planner) or None (the clones keep the game's own random() state).
        Returns bool (R,): lost a life within H steps."""
        buf = rows.copy()
        if reseeds is not None:
            batch_reseed(buf, reseeds)
        lives = batch_info(buf)[:, 0]
        batch_advance(buf, first)
        self.sim_steps += len(buf)
        now = batch_info(buf)[:, 0]
        dead = now < lives
        lives = now
        idx = np.flatnonzero(~dead)
        for t in range(1, self.H):
            if not len(idx):
                break
            sub = buf[idx]
            grid, vec = batch_obs(self.net, sub)
            a = self.net(grid, vec).argmax(1).astype(np.int32)
            noisy = u[idx, t] < self.eps
            a[noisy] = ra[idx, t][noisy]
            batch_advance(sub, a)
            self.sim_steps += len(sub)
            now = batch_info(sub)[:, 0]
            died = now < lives[idx]
            lives[idx] = now
            buf[idx] = sub
            dead[idx[died]] = True
            idx = idx[~died]
        return dead

    def choose(self, buf, info, q, noise_seeds, visits=None):
        """buf (G, SIZE) game states, info (G, 6), q (G, 5); noise_seeds: a
        seed sequence per game for this step; visits: per game None or a
        Counter of recently visited cells (loop breaker active: all four
        directions are rolled out and the kept one with the highest
        Q - penalty * visits[cell after the move] is played). Returns
        (actions, stats)."""
        G, K, H = len(buf), self.K, self.H
        pd = info[:, 5]
        qd = q[:, :4].copy()
        act = np.tile(np.arange(4, dtype=np.int32), (G, 1))
        for g in range(G):
            if pd[g] <= 3 and q[g, N_ACTION] > q[g, pd[g]]:
                qd[g, pd[g]] = q[g, N_ACTION]
                act[g, pd[g]] = N_ACTION
        order = np.argsort(-qd, axis=1, kind="stable")
        u = np.empty((G, 4, K, H))
        ra = np.empty((G, 4, K, H), np.int32)
        for g, s in enumerate(noise_seeds):
            rng = np.random.default_rng(s)
            u[g] = rng.random((4, K, H))
            ra[g] = rng.integers(0, 5, (4, K, H))
        # the k-th clone of every direction gets the same ghost future
        # (common random numbers: the directions are compared on equal terms)
        rs = None
        if self.ghost_rng == "reseed":
            rs = np.empty((G, K), np.uint32)
            for g, s in enumerate(noise_seeds):
                rs[g] = np.random.default_rng([*s, 1]).integers(1, 2**31 - 1, K)
        visits = visits if visits is not None else [None] * G
        breaker = np.array([v is not None for v in visits])
        frac = np.full((G, 4), np.nan)   # by rank in `order`
        c0 = order[:, 0]
        gi = np.repeat(np.arange(G), K)
        ki = np.tile(np.arange(K), G)
        dead = self.rollout(buf[gi], act[gi, c0[gi]], u[gi, c0[gi], ki], ra[gi, c0[gi], ki],
                            None if rs is None else rs[gi, ki])
        frac[:, 0] = dead.reshape(G, K).mean(1)
        bad = np.flatnonzero((frac[:, 0] > self.thr) | breaker)
        if len(bad):
            gi = np.repeat(bad, 3 * K)
            ri = np.tile(np.repeat(np.arange(1, 4), K), len(bad))
            ki = np.tile(np.arange(K), 3 * len(bad))
            ci = order[gi, ri]
            dead = self.rollout(buf[gi], act[gi, ci], u[gi, ci, ki], ra[gi, ci, ki],
                                None if rs is None else rs[gi, ki])
            frac[bad, 1:] = dead.reshape(len(bad), 3, K).mean(2)
        rank = np.zeros(G, np.int64)
        all_bad = np.zeros(G, bool)
        for g in bad:
            ok = np.flatnonzero(frac[g] <= self.thr)
            if len(ok) and visits[g] is not None:
                one = np.repeat(buf[g:g + 1], 4, 0)
                batch_advance(one, act[g, order[g]])
                cells = batch_info(one)[:, 2]
                score = [qd[g, order[g, r]] - self.penalty * visits[g][cells[r]] for r in ok]
                rank[g] = ok[int(np.argmax(score))]
            elif len(ok):
                rank[g] = ok[0]
            else:
                rank[g] = int(np.argmin(frac[g]))   # first minimum: highest Q
                all_bad[g] = True
        choice = order[np.arange(G), rank]
        actions = act[np.arange(G), choice]
        return actions, {"second_round": np.isin(np.arange(G), bad) & ~breaker,
                         "all_dropped": all_bad, "breaker": breaker, "frac_best": frac[:, 0]}


class Stall:
    """Loop breaker state of one game (plan.loop_breaker): after
    stall_steps steps without food, the next active_steps decisions get the
    visited-cell penalty (cells of the last stall_steps steps)."""

    def __init__(self, cfg):
        lb = cfg["plan"].get("loop_breaker") or {}
        self.on = bool(lb)
        self.stall, self.active = int(lb.get("stall_steps", 0)), int(lb.get("active_steps", 0))
        self.cells = deque()
        self.count = Counter()
        self.last_food, self.until, self.t = 0, -1, 0

    def update(self, cell, ate):
        """After a step: pacman's cell, whether food was eaten (or a level
        cleared)."""
        self.t += 1
        if ate:
            self.last_food = self.t
        self.cells.append(cell)
        self.count[cell] += 1
        if len(self.cells) > self.stall:
            self.count[self.cells.popleft()] -= 1
        if self.on and self.t >= self.until and self.t - self.last_food >= self.stall:
            self.until = self.t + self.active

    def visits(self):
        """Counter for Planner.choose, or None when inactive."""
        return self.count if self.t < self.until else None


def play_games(tasks, cfg, mode, device, worker=0):
    """Play the (level, seed, stop_at_clear) tasks in lock-step batches."""
    from sim.csim import SIZE, CSim
    net = QNet(os.path.join(ROOT, cfg["checkpoint"]), device)
    planner = Planner(net, cfg)
    G = int(cfg["run"]["games_per_worker"])
    max_steps = cfg["eval"]["max_steps"]
    max_steps = float("inf") if max_steps is None else int(max_steps)
    queue = list(tasks)
    games, rows = [], np.zeros((0, SIZE), np.uint8)
    results = []
    decisions, t0 = 0, time.perf_counter()
    while queue or games:
        while len(games) < G and queue:
            level, seed, stop = queue.pop(0)
            sim = CSim()
            sim.reset(seed, level)
            info = sim.info()
            games.append({"seed": seed, "start_level": level, "stop": stop, "length": 0,
                          "deaths": 0, "level": level, "lives": int(info[0]), "eats": int(info[4]),
                          "food_eaten": 0, "actions": [], "overrides": 0, "second_round": 0,
                          "all_dropped": 0, "breaker": 0, "stall": Stall(cfg)})
            rows = np.concatenate([rows, sim.buf[None]])
        info = batch_info(rows)
        grid, vec = batch_obs(net, rows)
        q = net(grid, vec)
        greedy = q.argmax(1).astype(np.int32)
        if mode == "plan":
            seeds = [[g["seed"], g["start_level"], g["length"]] for g in games]
            actions, st = planner.choose(rows, info, q, seeds,
                                         [g["stall"].visits() for g in games])
            # the planner's choice moves pacman like the greedy action when
            # both are the same direction (N counts as pacman's direction)
            same = (actions == greedy)
            for i, g in enumerate(games):
                g["overrides"] += int(not same[i])
                g["second_round"] += int(st["second_round"][i])
                g["all_dropped"] += int(st["all_dropped"][i])
                g["breaker"] += int(st["breaker"][i])
        else:
            actions = greedy
        batch_advance(rows, actions)
        decisions += len(games)
        info = batch_info(rows)
        keep = []
        for i, g in enumerate(games):
            g["length"] += 1
            g["actions"].append(int(actions[i]))
            lives, level, eats = int(info[i, 0]), int(info[i, 1]), int(info[i, 4])
            if lives < g["lives"]:
                g["deaths"] += g["lives"] - lives
            ate = level != g["level"] or eats < g["eats"]
            if level == g["level"] and eats < g["eats"]:
                g["food_eaten"] += g["eats"] - eats
            g["stall"].update(int(info[i, 2]), ate)
            g["lives"], g["eats"] = lives, eats
            g["level"] = max(g["level"], level)
            g["score"] = int(info[i, 3])
            end = lives == 0 or g["length"] >= max_steps or (g["stop"] and g["level"] > g["start_level"])
            if end:
                g["truncated"] = bool(lives > 0 and g["length"] >= max_steps
                                      and not (g["stop"] and g["level"] > g["start_level"]))
                g["levels_cleared"] = g["level"] - g["start_level"]
                g["actions"] = "".join("UDLRN"[a] for a in g["actions"])
                del g["stall"]
                results.append(g)
            else:
                keep.append(i)
        games = [games[i] for i in keep]
        rows = rows[keep]
    dt = time.perf_counter() - t0
    return {"worker": worker, "games": results, "decisions": decisions, "time_s": dt,
            "rollout_sim_steps": planner.sim_steps, "forwards": net.forwards}


def _worker(args):
    tasks, cfg, mode, device, worker = args
    return play_games(tasks, cfg, mode, device, worker)


def make_tasks(cfg, levels):
    e = cfg["eval"]
    seed, n1, nl = int(e["seed"]), int(e["episodes"]), int(e["level_episodes"])
    tasks = []
    if 1 in levels:
        tasks += [(1, seed + i, False) for i in range(max(n1, nl))]
    tasks += [(lv, seed + i, True) for lv in levels if lv != 1 for i in range(nl)]
    return tasks


def summarize(games, cfg, levels):
    e = cfg["eval"]
    by = {}
    for g in games:
        by.setdefault(g["start_level"], []).append(g)
    per_level = {}
    for lv in levels:
        gs = sorted(by[lv], key=lambda g: g["seed"])[:int(e["level_episodes"])]
        per_level[lv] = {"episodes": len(gs),
                         "cleared": float(np.mean([g["levels_cleared"] >= 1 for g in gs])),
                         "mean_length": float(np.mean([g["length"] for g in gs])),
                         "deaths": float(np.mean([g["deaths"] for g in gs])),
                         "truncated": float(np.mean([g["truncated"] for g in gs]))}
    out = {"per_level": per_level,
           "per_maze_mean": float(np.mean([v["cleared"] for v in per_level.values()])),
           "min_level_cleared": min(v["cleared"] for v in per_level.values())}
    if 1 in by:
        from1 = sorted(by[1], key=lambda g: g["seed"])[:int(e["episodes"])]
        cl = np.array([g["levels_cleared"] for g in from1])
        out["from_level1"] = {"episodes": len(from1), "mean_levels_cleared": float(cl.mean()),
                              "median_levels_cleared": float(np.median(cl)),
                              "max_levels_cleared": int(cl.max()),
                              "mean_score": float(np.mean([g["score"] for g in from1])),
                              "truncated": float(np.mean([g["truncated"] for g in from1]))}
    n = sum(g["length"] for g in games)
    out["decisions"] = n
    out["overrides"] = sum(g["overrides"] for g in games) / n
    out["second_round"] = sum(g["second_round"] for g in games) / n
    out["all_dropped"] = sum(g["all_dropped"] for g in games) / n
    out["breaker"] = sum(g["breaker"] for g in games) / n
    out["deaths_per_game"] = float(np.mean([g["deaths"] for g in games]))
    return out


def replay_check(games, n):
    """Replay n games' actions in the real game; the last state lines must
    be equal to the sim's."""
    from sim.csim import CSim
    from sim.verify import GameProc
    bad = 0
    for g in games[:n]:
        acts = ["UDLRN".index(c) for c in g["actions"]]
        sim = CSim()
        sim.reset(g["seed"], g["start_level"])
        proc = GameProc(g["seed"], g["start_level"])
        try:
            line = proc.recv()
            for a in acts:
                proc.send(a)
                line = proc.recv()
                sim.advance(a)
        finally:
            proc.close()
        bad += line != sim.state_json()
    return bad


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "plan.yaml"))
    p.add_argument("--mode", choices=("plan", "greedy"), default="plan")
    p.add_argument("--levels", default=None, help="comma list of start levels (default 1..16)")
    p.add_argument("--episodes", type=int, help="override eval.episodes")
    p.add_argument("--level-episodes", type=int, help="override eval.level_episodes")
    p.add_argument("--threshold", type=float, help="override plan.death_threshold")
    p.add_argument("--workers", type=int, help="override run.workers")
    p.add_argument("--replay-check", type=int, default=0, metavar="N",
                   help="replay N finished games in the real game")
    p.add_argument("--out", help="JSON with the summary and every game (default runs/plan_eval/)")
    args = p.parse_args()
    cfg = load_cfg(args.config)
    if args.episodes is not None:
        cfg["eval"]["episodes"] = args.episodes
    if args.level_episodes is not None:
        cfg["eval"]["level_episodes"] = args.level_episodes
    if args.threshold is not None:
        cfg["plan"]["death_threshold"] = args.threshold
    if args.workers is not None:
        cfg["run"]["workers"] = args.workers
    levels = [int(x) for x in args.levels.split(",")] if args.levels else list(range(1, LEVELS + 1))
    tasks = make_tasks(cfg, levels)
    tasks.sort(key=lambda t: t[0] != 1)          # the long games first
    nw = int(cfg["run"]["workers"])
    split = [tasks[i::nw] for i in range(nw)]
    t0 = time.perf_counter()
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(nw) as pool:
        parts = pool.map(_worker, [(s, cfg, args.mode, cfg["run"]["device"], i)
                                   for i, s in enumerate(split) if s])
    wall = time.perf_counter() - t0
    games = [g for part in parts for g in part["games"]]
    res = summarize(games, cfg, levels)
    res.update(mode=args.mode, config=cfg, levels=levels, wall_s=round(wall, 1),
               decisions_per_s=round(res["decisions"] / wall, 1),
               rollout_sim_steps=sum(pt["rollout_sim_steps"] for pt in parts),
               commit=os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip())
    res["rollout_sim_steps_per_s"] = round(res["rollout_sim_steps"] / wall, 1)
    if args.replay_check:
        res["replay_check"] = {"games": min(args.replay_check, len(games)),
                               "diverged": replay_check(games, args.replay_check)}
    out = args.out or os.path.join(ROOT, "runs", "plan_eval",
                                   f"{args.mode}_{time.strftime('%m%d_%H%M')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({**res, "games": games}, f)
    print(json.dumps({k: v for k, v in res.items() if k not in ("config", "per_level")}, indent=1))
    print("per level cleared:", " ".join(f"{lv}:{v['cleared']:.0%}" for lv, v in res["per_level"].items()))
    print("written", out)


if __name__ == "__main__":
    main()
