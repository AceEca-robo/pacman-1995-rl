#!/usr/bin/env python3
"""Train DQN on Pacman1995-v0.

    python scripts/train.py --config configs/dqn.yaml --seed 0 --run-name dqn0
    python scripts/train.py --run-name dqn0 --resume

Writes TensorBoard logs, checkpoint.pt (+ buffer.npz), best.pt and
evals.jsonl to runs/<run-name>/. A config may name a "base" config whose
values it overrides.
"""

import argparse
import json
import os
import sys
import time

import gymnasium as gym
import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agents.dqn import DQNLearner, NStep, make_buffer, pack_grid, pack_grids  # noqa: E402
from agents.heuristic_agent import HeuristicAgent  # noqa: E402
from env.pacman_env import PacmanEnv, load_env_config  # noqa: E402


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = cfg.pop("base", None)
    if base is None:
        return cfg
    merged = load_config(os.path.join(os.path.dirname(path), base))
    for k, v in cfg.items():
        merged[k] = {**merged[k], **v} if isinstance(v, dict) and isinstance(merged.get(k), dict) else v
    return merged


def epsilon(cfg, steps):
    frac = min(1.0, steps / cfg["epsilon_steps"])
    return cfg["epsilon_start"] + frac * (cfg["epsilon_end"] - cfg["epsilon_start"])


def per_beta(cfg, steps):
    per = cfg["per"]
    frac = min(1.0, steps / (per["beta_steps"] or cfg["total_steps"]))
    return per["beta_start"] + frac * (per["beta_end"] - per["beta_start"])


class Collector:
    """Steps the vector env, turns steps into n-step transitions in the
    buffer and logs finished episodes."""

    def __init__(self, envs, cfg, buffer, writer, seed):
        self.envs, self.buffer, self.writer = envs, buffer, writer
        self.n = envs.num_envs
        self.obs, _ = envs.reset(seed=seed)
        self.cur = pack_grids(self.obs["grid"])
        self.nsteps = [NStep(cfg["n_step"], cfg["gamma"]) for _ in range(self.n)]
        self.ep_reward = np.zeros(self.n)
        self.ep_len = np.zeros(self.n, int)
        self.ep_level = np.ones(self.n, int)

    def step(self, actions, log_step, prefix="episode"):
        """Returns the number of episodes that ended."""
        obs = self.obs
        nobs, rew, term, trunc, info = self.envs.step(actions)
        packed_next = pack_grids(nobs["grid"])
        ended_count = 0
        for i in range(self.n):
            ended = term[i] or trunc[i]
            if ended:  # nobs holds the next episode's first obs
                last_packed = pack_grid(info["final_obs"][i]["grid"])
                last_vec = info["final_obs"][i]["vec"]
            else:
                last_packed, last_vec = packed_next[i], nobs["vec"][i]
            for t in self.nsteps[i].push(self.cur[i], obs["vec"][i], actions[i], rew[i],
                                         last_packed, last_vec, term[i], ended):
                self.buffer.add(*t)
            self.ep_reward[i] += rew[i]
            self.ep_len[i] += 1
            src = info["final_info"] if ended else info
            self.ep_level[i] = max(self.ep_level[i], src["level"][i])
            if ended:
                ended_count += 1
                self.writer.add_scalar(f"{prefix}/reward", self.ep_reward[i], log_step)
                self.writer.add_scalar(f"{prefix}/score", src["score"][i], log_step)
                self.writer.add_scalar(f"{prefix}/length", self.ep_len[i], log_step)
                self.writer.add_scalar(f"{prefix}/level_reached", self.ep_level[i], log_step)
                self.ep_reward[i], self.ep_len[i], self.ep_level[i] = 0.0, 0, info["level"][i]
        self.cur, self.obs = packed_next, nobs
        return ended_count


def warm_start(collector, cfg):
    """Fill the buffer with cfg["warm_start"] env steps of the heuristic agent.
    These steps do not count towards total_steps or the schedules."""
    agent = HeuristicAgent(cfg["warm_start_agent_config"], n_actions=collector.envs.single_action_space.n)
    done, t0 = 0, time.perf_counter()
    while done < cfg["warm_start"]:
        obs = collector.obs
        actions = np.array([agent.act({"grid": obs["grid"][i], "vec": obs["vec"][i]})
                            for i in range(collector.n)])
        collector.step(actions, done, prefix="warm_start")
        done += collector.n
    print(f"warm start: {done} heuristic steps, buffer {collector.buffer.size} "
          f"({time.perf_counter() - t0:.0f} s)", flush=True)


def evaluate(learner, envs, cfg):
    """Greedy episodes over a fixed seed set, the envs stepping in a batch."""
    seeds = list(range(cfg["eval_seed"], cfg["eval_seed"] + cfg["eval_episodes"]))
    results, live = [], []  # live: [env, obs, seed, reward, length, max level]
    for env in envs:
        if seeds:
            s = seeds.pop(0)
            obs, info = env.reset(seed=s)
            live.append([env, obs, s, 0.0, 0, info["level"]])
    while live:
        q = learner.q_values(pack_grids(np.stack([e[1]["grid"] for e in live])),
                             np.stack([e[1]["vec"] for e in live]))
        for e, a in zip(list(live), q.argmax(1)):
            obs, r, term, trunc, info = e[0].step(a)
            e[1], e[3], e[4], e[5] = obs, e[3] + r, e[4] + 1, max(e[5], info["level"])
            if term or trunc:
                results.append({"seed": e[2], "reward": e[3], "score": info["score"],
                                "length": e[4], "level": e[5]})
                if seeds:
                    e[2] = seeds.pop(0)
                    e[1], info = e[0].reset(seed=e[2])
                    e[3], e[4], e[5] = 0.0, 0, info["level"]
                else:
                    live.remove(e)
    return sorted(results, key=lambda x: x["seed"])


def best_metric(name, results):
    """Value to pick best.pt by. "mean_reward" (default since dqn4: a median
    ignores games stuck until the step limit while they are under half),
    "median_reward" / "median_score"; old configs say "reward" / "score"."""
    name = {"reward": "median_reward", "score": "median_score"}.get(name, name)
    stat, key = name.split("_")
    values = [r[key] for r in results]
    return float(np.mean(values) if stat == "mean" else np.median(values))


def save_npz(d, path):
    with open(path, "wb") as f:  # a file object: np.savez would append .npz to a name
        np.savez(f, **d)


def save_atomic(obj, path, saver):
    tmp = path + ".tmp"
    saver(obj, tmp)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "dqn.yaml"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", required=True)
    p.add_argument("--resume", action="store_true", help="continue runs/<run-name>/checkpoint.pt")
    args = p.parse_args()

    run_dir = os.path.join(ROOT, "runs", args.run_name)
    ck_path = os.path.join(run_dir, "checkpoint.pt")
    buf_path = os.path.join(run_dir, "buffer.npz")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.resume:
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        cfg, seed = ck["config"], ck["seed"]
        # options added after the run started keep their defaults (off)
        cfg = {**load_config(os.path.join(ROOT, "configs", "dqn.yaml")), **cfg}
    else:
        if os.path.exists(ck_path):
            sys.exit(f"{run_dir} already has a checkpoint; use --resume or another --run-name")
        cfg, seed, ck = load_config(args.config), args.seed, None
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg, f)

    torch.manual_seed(seed)
    env_cfg = load_env_config(cfg["env_config"])
    n_actions = env_cfg["actions"]
    # evaluation without shaping, so eval rewards compare across runs
    eval_env_cfg = {**env_cfg, "shaping": {**env_cfg["shaping"], "enabled": False}}
    learner = DQNLearner(cfg, device, n_actions)
    buffer = make_buffer(cfg)
    rng = np.random.default_rng(seed)
    steps = n_updates = episodes = 0
    best = -np.inf
    if ck:
        learner.load_state_dict(ck["learner"])
        steps, n_updates, episodes, best = ck["steps"], ck["n_updates"], ck["episodes"], ck["best"]
        rng.bit_generator.state = ck["rng"]
        torch.set_rng_state(ck["torch_rng"].cpu())  # map_location moved it to the device
        if os.path.exists(buf_path):
            with np.load(buf_path) as z:
                buffer.load_state_dict(z)
        print(f"resumed at {steps} steps, buffer {buffer.size}")
    writer = SummaryWriter(run_dir, purge_step=steps if ck else None)

    n = cfg["num_envs"]
    make = lambda: PacmanEnv(config=env_cfg)  # noqa: E731
    envs = gym.vector.SyncVectorEnv([make] * n, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    eval_envs = [PacmanEnv(config=eval_env_cfg) for _ in range(n)]
    # a resumed run must not replay the same games
    col = Collector(envs, cfg, buffer, writer, seed=seed * 1_000_000 + steps)
    if not ck and cfg["warm_start"]:
        warm_start(col, cfg)
    per = cfg["per"]["enabled"]
    min_buffer = min(cfg["learning_starts"], cfg["buffer_size"]) if ck else cfg["batch_size"]

    losses, qs = [], []
    update_credit = 0.0
    next_target = (steps // cfg["target_update"] + 1) * cfg["target_update"]
    next_log = (steps // cfg["log_every"] + 1) * cfg["log_every"]
    next_eval = (steps // cfg["eval_every"] + 1) * cfg["eval_every"]
    t_log, s_log = time.perf_counter(), steps
    print(f"run {run_dir}, device {device}, buffer {buffer.nbytes / 1e9:.2f} GB")

    try:
        while steps < cfg["total_steps"]:
            eps = epsilon(cfg, steps)
            actions = rng.integers(n_actions, size=n)
            greedy = rng.random(n) >= eps
            if greedy.any():
                q = learner.q_values(col.cur[greedy], col.obs["vec"][greedy])
                actions[greedy] = q.argmax(1)
            episodes += col.step(actions, steps)
            steps += n

            # after a resume without a saved buffer, refill it first
            if steps >= cfg["learning_starts"] and buffer.size >= min_buffer:
                update_credit += n / cfg["train_every"]
                beta = per_beta(cfg, steps) if per else None
                while update_credit >= 1:
                    batch, idx, weights = buffer.sample(rng, cfg["batch_size"], beta)
                    loss, mq, td = learner.update(batch, weights)
                    if per:
                        buffer.update_priorities(idx, td.cpu().numpy())
                    losses.append(loss)
                    qs.append(mq)
                    n_updates += 1
                    update_credit -= 1
            if steps >= next_target:
                learner.sync_target()
                next_target += cfg["target_update"]

            if steps >= next_log:
                now = time.perf_counter()
                sps = (steps - s_log) / (now - t_log)
                writer.add_scalar("train/epsilon", eps, steps)
                writer.add_scalar("perf/steps_per_sec", sps, steps)
                writer.add_scalar("train/buffer_size", buffer.size, steps)
                if per:
                    writer.add_scalar("train/per_beta", per_beta(cfg, steps), steps)
                msg = f"{steps:>9} steps  eps {eps:.3f}  {sps:,.0f} steps/s  episodes {episodes}"
                if losses:
                    loss_m = torch.stack(losses).mean().item()
                    q_m = torch.stack(qs).mean().item()
                    writer.add_scalar("train/loss", loss_m, steps)
                    writer.add_scalar("train/mean_q", q_m, steps)
                    msg += f"  loss {loss_m:.4f}  Q {q_m:.2f}  updates {n_updates}"
                    losses, qs = [], []
                print(msg, flush=True)
                next_log += cfg["log_every"]
                t_log, s_log = time.perf_counter(), steps

            if steps >= next_eval or steps >= cfg["total_steps"]:
                t0 = time.perf_counter()
                res = evaluate(learner, eval_envs, cfg)
                metric = best_metric(cfg["best_metric"], res)
                summary = {
                    "steps": steps,
                    "median_reward": float(np.median([r["reward"] for r in res])),
                    "mean_reward": float(np.mean([r["reward"] for r in res])),
                    "median_score": float(np.median([r["score"] for r in res])),
                    "mean_length": float(np.mean([r["length"] for r in res])),
                    "level1_cleared": float(np.mean([r["level"] >= 2 for r in res])),
                }
                for k, v in summary.items():
                    if k != "steps":
                        writer.add_scalar(f"eval/{k}", v, steps)
                with open(os.path.join(run_dir, "evals.jsonl"), "a") as f:
                    f.write(json.dumps({**summary, "episodes": res}) + "\n")
                print(f"eval @ {steps}: " + ", ".join(f"{k} {v:.2f}" for k, v in summary.items()
                                                       if k != "steps")
                      + f"  ({time.perf_counter() - t0:.0f} s)", flush=True)
                state = {"config": cfg, "seed": seed, "n_actions": n_actions,
                         "env_config": env_cfg, "learner": learner.state_dict(),
                         "steps": steps, "n_updates": n_updates, "episodes": episodes,
                         "best": max(best, metric), "rng": rng.bit_generator.state,
                         "torch_rng": torch.get_rng_state()}
                if metric > best:
                    best = metric
                    save_atomic({**state, "eval": summary}, os.path.join(run_dir, "best.pt"), torch.save)
                save_atomic(state, ck_path, torch.save)
                if cfg["checkpoint_buffer"]:
                    save_atomic(buffer.state_dict(), buf_path, save_npz)
                next_eval += cfg["eval_every"]
                t_log, s_log = time.perf_counter(), steps  # eval time is not training speed
    finally:
        envs.close()
        for e in eval_envs:
            e.close()
        writer.close()


if __name__ == "__main__":
    main()
