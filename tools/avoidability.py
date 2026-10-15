#!/usr/bin/env python3
"""Death avoidability: from the state k ticks before a recorded death, can
a perturbed continuation survive the next --window ticks?

    python tools/avoidability.py --deaths 300 --out runs/avoid/avoid.json \
        --plot docs/v04_avoidability.png

Deaths come from a death-prefix file (tools/record_death_prefixes.py, its
T-20 prefixes); --deaths of them are drawn at random (--sample-seed). For
each death and each horizon k (--horizons, default 20 40 80 120 160) the
game is replayed to k ticks before the death (the T-20 actions cut by k-20;
the game is deterministic) and continued --rollouts times by the checkpoint
(greedy with probability 1 - --eps-agent, else a uniform random action) and
--rollouts times by the heuristic (--eps-heuristic). The rollouts differ
only in their action RNG (seeded by death, horizon, agent and rollout). A
rollout survives if it loses no life in --window ticks (a level cleared in
the window counts as survival).

A horizon counts for a death only if its state is in the death's level and
life (same level, same lives as at T-20, more than k ticks into the level);
otherwise it is skipped and reported as not eligible. Per horizon a greedy
rollout of the checkpoint is also played: it must die exactly k ticks later
(a check that the replay reproduces the recorded game). A horizon beyond
the window (default 400) is a control: the greedy game does not die in the
window, so its rollouts measure how often the noise alone kills.

Rollouts run in --workers processes, each with one env; the checkpoint's Q
values are computed in the main process in batches over the workers.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from multiprocessing.connection import wait

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

AGENT, HEUR = 0, 1
NAMES = ("agent", "heuristic")


def _worker(conn, obs_cfg):
    from agents.dqn import pack_grids
    from agents.heuristic_agent import HeuristicAgent
    from env.pacman_env import PacmanEnv
    from record_death_prefixes import env_config
    env = PacmanEnv(config=env_config(obs_cfg, 1))
    extras = env.grid_extras
    heur = HeuristicAgent()
    obs = None

    def reply(obs, **kw):
        conn.send({"grid": pack_grids(obs["grid"][None], extras)[0], "vec": obs["vec"], **kw})

    try:
        while True:
            cmd, arg = conn.recv()
            if cmd == "start":
                try:  # checks the level and the lives
                    env.replay_prefix(arg)
                except RuntimeError:
                    conn.send({"level": -1, "lives": -1})
                    continue
                st = env._state
                obs, _ = env._after_reset(int(arg["seed"]))
                reply(obs, level=st["level"], lives=st["lives"])
            elif cmd == "step":  # arg: an action, or None for the heuristic's
                a = heur.act(obs) if arg is None else arg
                obs, _, term, _, info = env.step(int(a))
                ev = info["events"]
                reply(obs, death=bool(ev["death"]), done=bool(term), action=int(a),
                      food=int(ev["eaten_dot"] + ev["eaten_energizer"]))
            elif cmd == "close":
                break
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        env.close()


def select(prefixes, n, horizons, seed):
    """n random deaths (their T-20 prefixes) -> list of death dicts."""
    t20 = [p for p in prefixes if p["before_death"] == 20]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(t20), size=min(n, len(t20)), replace=False)
    deaths = []
    for i in sorted(idx):
        p = t20[i]
        deaths.append({"id": int(i), "seed": p["seed"], "start_level": p["start_level"],
                       "level": p["level"], "lives": p["lives"], "maze": p["maze"],
                       "kind": p["kind"], "level_tick": p["level_tick"],
                       "death_step": p["steps"] + 20, "actions": p["actions"]})
    return deaths


def prefix_at(d, k):
    """The replay prefix k ticks before death d (no position checks: only
    the T-20 state was recorded)."""
    n = d["death_step"] - k
    return {"seed": d["seed"], "start_level": d["start_level"], "level": d["level"],
            "lives": d["lives"], "actions": d["actions"][:n]}


def jobs_for(deaths, horizons, rollouts, agents=(AGENT, HEUR), greedy=True):
    for di, d in enumerate(deaths):
        for k in horizons:
            if k > d["level_tick"] - 1:
                continue
            if greedy:
                yield (di, k, AGENT, -1)  # greedy check
            for agent in agents:
                for j in range(rollouts):
                    yield (di, k, agent, j)


def run(scorer, obs_cfg, deaths, horizons, rollouts, window, eps, workers, log_every=60,
        agents=(AGENT, HEUR), greedy=True):
    """-> {(di, k): {"eligible", "greedy_death_tick", "agent": [..], "heuristic": [..],
    "best"}}, per rollout the tick of its death (1 = the first step) or None
    if it survived the window; "best": the surviving rollout that ate the
    most food (first one on a tie) as {"agent", "rollout", "food",
    "actions", "random" (per action: drawn by eps)}, or None."""
    ctx = mp.get_context("spawn")
    pipes, procs = [], []
    for _ in range(workers):
        a, b = ctx.Pipe()
        pr = ctx.Process(target=_worker, args=(b, obs_cfg), daemon=True)
        pr.start()
        pipes.append(a)
        procs.append(pr)
    jobs = jobs_for(deaths, horizons, rollouts, agents, greedy)
    res = {}
    state = {}  # worker -> rollout state
    t0 = last = time.time()
    done_jobs = 0

    def launch(w):
        job = next(jobs, None)
        if job is None:
            return False
        di, k, agent, j = job
        key = (di, k)
        if key in res and not res[key]["eligible"]:
            return launch(w)
        rng = np.random.default_rng([deaths[di]["id"], k, agent, j + 1])
        state[w] = {"job": job, "rng": rng, "t": 0, "phase": "start", "acts": [], "food": 0,
                    "rand": []}
        pipes[w].send(("start", prefix_at(deaths[di], k)))
        return True

    def finish(w, tick):
        nonlocal done_jobs
        s = state.pop(w)
        di, k, agent, j = s["job"]
        r = res[(di, k)]
        if j < 0:
            r["greedy_death_tick"] = tick
        else:
            r[NAMES[agent]].append(tick)
            if tick is None and (r["best"] is None or s["food"] > r["best"]["food"]):
                r["best"] = {"agent": NAMES[agent], "rollout": j, "food": s["food"],
                             "actions": s["acts"], "random": s["rand"]}
        done_jobs += 1

    try:
        for w in range(workers):
            if not launch(w):
                break
        while state:
            ready = wait([pipes[w] for w in state])
            msgs = {}
            for w in list(state):
                if pipes[w] in ready:
                    msgs[w] = pipes[w].recv()
            need_q = []
            for w, m in msgs.items():
                s = state[w]
                di, k, agent, j = s["job"]
                if s["phase"] == "start":
                    d = deaths[di]
                    key = (di, k)
                    ok = m["level"] == d["level"] and m["lives"] == d["lives"]
                    if key not in res:
                        res[key] = {"eligible": ok, "greedy_death_tick": None,
                                    "agent": [], "heuristic": [], "best": None}
                    if not ok:
                        state.pop(w)
                        launch(w)
                        continue
                    s["phase"] = "run"
                else:
                    s["t"] += 1
                    s["acts"].append(m["action"])
                    s["food"] += m["food"]
                    if m["death"] or m["done"]:
                        finish(w, s["t"] if m["death"] else None)
                        launch(w)
                        continue
                    if s["t"] >= window:
                        finish(w, None)
                        launch(w)
                        continue
                s["obs"] = m
                e = 0.0 if j < 0 else eps[agent]
                if e and s["rng"].random() < e:
                    s["rand"].append(True)
                    pipes[w].send(("step", int(s["rng"].integers(5))))
                else:
                    s["rand"].append(False)
                    if agent == HEUR:
                        pipes[w].send(("step", None))
                    else:
                        need_q.append(w)
            if need_q:
                q = scorer.q_values(np.stack([state[w]["obs"]["grid"] for w in need_q]),
                                    np.stack([state[w]["obs"]["vec"] for w in need_q]))
                for w, a in zip(need_q, q.argmax(1)):
                    pipes[w].send(("step", int(a)))
            if time.time() - last > log_every:
                last = time.time()
                print(f"{done_jobs} rollouts, {len(res)} states, {last - t0:.0f} s", flush=True)
    finally:
        for p in pipes:
            try:
                p.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for pr in procs:
            pr.join(5)
    return res


def summarize(deaths, res, horizons, rollouts, window):
    """Per horizon and agent: eligible deaths, share with >= 1 surviving
    rollout, share with >= 25% surviving; also over the deaths eligible at
    every horizon within the window (the control horizon left out)."""
    full = [di for di in range(len(deaths))
            if all(res.get((di, k), {}).get("eligible") for k in horizons if k <= window)]
    rows = []
    for k in horizons:
        for subset, dis in (("eligible", range(len(deaths))), ("all horizons", full)):
            keys = [(di, k) for di in dis if res.get((di, k), {}).get("eligible")]
            row = {"horizon": k, "subset": subset, "deaths": len(keys)}
            for name in NAMES + ("either",):
                surv = []
                for key in keys:
                    r = res[key]
                    if name == "either":
                        s = [sum(t is None for t in r[n]) / max(len(r[n]), 1) for n in NAMES]
                        surv.append((max(s) > 0, max(s) >= 0.25))
                    else:
                        f = sum(t is None for t in r[name]) / max(len(r[name]), 1)
                        surv.append((f > 0, f >= 0.25))
                row[f"{name}_any"] = float(np.mean([a for a, _ in surv])) if surv else None
                row[f"{name}_q25"] = float(np.mean([b for _, b in surv])) if surv else None
                if name != "either":
                    row[f"{name}_mean_survival"] = float(np.mean(
                        [sum(t is None for t in res[key][name]) / len(res[key][name])
                         for key in keys])) if keys else None
            g = [res[key]["greedy_death_tick"] for key in keys]
            expect = k if k <= window else None  # a control horizon: no death in the window
            row["greedy_dies_on_time"] = float(np.mean([x == expect for x in g])) if g else None
            rows.append(row)
    return rows, full


def plot(rows, horizons, out, window, rollouts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    styles = {"agent": ("C0", "best_v04, eps 0.3"), "heuristic": ("C1", "heuristic, eps 0.2"),
              "either": ("C2", "either")}
    for ax, metric, title in ((axes[0], "any", "at least one rollout survives"),
                              (axes[1], "q25", ">= 25% of rollouts survive")):
        for subset, ls in (("eligible", "-"), ("all horizons", "--")):
            rs = [r for r in rows if r["subset"] == subset]
            for name, (c, lab) in styles.items():
                ax.plot([r["horizon"] for r in rs], [100 * r[f"{name}_{metric}"] for r in rs],
                        ls, color=c, marker="o", ms=4,
                        label=lab if subset == "eligible" else None)
        ax.set_title(title)
        ax.set_xlabel(f"ticks before the death (> {window}: control)")
        ax.set_xticks(horizons)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("% of deaths")
    axes[0].legend(fontsize=8)
    n = max(r["deaths"] for r in rows if r["subset"] == "eligible")
    fig.suptitle(f"Avoidability of best_v04's deaths: {rollouts} rollouts per state, survive "
                 f"{window} ticks (solid: deaths eligible at that horizon, up to {n}; "
                 f"dashed: deaths eligible at every horizon)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=110)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--prefixes", default=os.path.join(ROOT, "data", "death_prefixes_v04.json"))
    p.add_argument("--checkpoint", default=os.path.join(ROOT, "runs", "best_v04", "best.pt"))
    p.add_argument("--deaths", type=int, default=300)
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--horizons", type=int, nargs="+", default=[20, 40, 80, 120, 160, 400])
    p.add_argument("--rollouts", type=int, default=40)
    p.add_argument("--window", type=int, default=200)
    p.add_argument("--eps-agent", type=float, default=0.3)
    p.add_argument("--eps-heuristic", type=float, default=0.2)
    p.add_argument("--workers", type=int, default=36)
    p.add_argument("--out", required=True)
    p.add_argument("--plot", default=None)
    args = p.parse_args()
    import torch
    from train_eval import Scorer
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    obs_cfg = (ck.get("env_config") or {}).get("obs") or {}
    scorer = Scorer(args.checkpoint, "cuda" if torch.cuda.is_available() else "cpu")
    with open(args.prefixes) as f:
        prefixes = json.load(f)["prefixes"]
    deaths = select(prefixes, args.deaths, args.horizons, args.sample_seed)
    t0 = time.time()
    res = run(scorer, obs_cfg, deaths, args.horizons, args.rollouts, args.window,
              (args.eps_agent, args.eps_heuristic), args.workers)
    rows, full = summarize(deaths, res, args.horizons, args.rollouts, args.window)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "seconds": round(time.time() - t0), "summary": rows,
                   "full_deaths": len(full),
                   "deaths": [{k: v for k, v in d.items() if k != "actions"} for d in deaths],
                   "results": [{"death": di, "horizon": k, **{x: v for x, v in r.items() if x != "best"}}
                               for (di, k), r in res.items()]},
                  f)
    print("| horizon | subset | deaths | agent >=1 | agent >=25% | heur >=1 | heur >=25% "
          "| either >=1 | either >=25% | greedy dies on time |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| T-{r['horizon']} | {r['subset']} | {r['deaths']} | " + " | ".join(
            f"{100 * r[c]:.0f}%" if r[c] is not None else "-" for c in (
                "agent_any", "agent_q25", "heuristic_any", "heuristic_q25", "either_any",
                "either_q25", "greedy_dies_on_time")) + " |")
    if args.plot:
        plot(rows, args.horizons, args.plot, args.window, args.rollouts)
        print("wrote", args.plot)


if __name__ == "__main__":
    main()
