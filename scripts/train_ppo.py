#!/usr/bin/env python3
"""Train PPO on Pacman1995-v0 (or any Discrete gymnasium env for tests).

    python scripts/train_ppo.py --config configs/ppo_hunger.yaml --seed 0 --run-name ppo0
    python scripts/train_ppo.py --run-name ppo0 --resume

Same run layout as scripts/train.py: runs/<run-name>/config.yaml,
TensorBoard logs, evals.jsonl, checkpoint.pt, best.pt ("algo": "ppo"), and
log lines "<steps> steps ... steps/s" / "eval @ <steps>: ..." that the
supervisor reads. Evaluation: greedy (argmax of the policy), without
shaping and hunger_limit.
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
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from agents.ppo import PPOLearner, compute_gae  # noqa: E402
from env.pacman_env import PacmanEnv, load_env_config  # noqa: E402
from train import best_metric, evaluate, load_config, save_atomic  # noqa: E402


class RewardScaler:
    """Divides rewards by the running std of the discounted return, per env
    stream (like gymnasium's NormalizeReward). Only the learning signal is
    scaled: the env's rewards, logs and evaluation are untouched. Without it
    the value loss (returns of tens) swamps the policy gradient in the
    shared network and the clipped update barely moves the policy."""

    def __init__(self, n, gamma, eps=1e-8):
        self.ret = np.zeros(n)
        self.gamma, self.eps = gamma, eps
        self.count, self.mean, self.var = 1e-4, 0.0, 1.0

    def __call__(self, r, ended):
        self.ret = self.ret * self.gamma + r
        self._update(self.ret)
        self.ret[ended] = 0.0
        return (r / np.sqrt(self.var + self.eps)).astype(np.float32)

    def _update(self, x):
        b_mean, b_var, b_count = x.mean(), x.var(), len(x)
        delta, tot = b_mean - self.mean, self.count + b_count
        self.mean += delta * b_count / tot
        m2 = self.var * self.count + b_var * b_count + delta ** 2 * self.count * b_count / tot
        self.var, self.count = m2 / tot, tot

    def state_dict(self):
        return {"ret": self.ret, "count": self.count, "mean": self.mean, "var": self.var}

    def load_state_dict(self, d):
        self.ret, self.count, self.mean, self.var = d["ret"], d["count"], d["mean"], d["var"]


def make_envs(cfg, n, env_id=None):
    if env_id:  # tests: a plain gymnasium env
        fns = [lambda: gym.make(env_id)] * n
    else:
        env_cfg = load_env_config(cfg["env_config"])
        fns = [lambda: PacmanEnv(config=env_cfg)] * n
    return gym.vector.SyncVectorEnv(fns, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)


def train(cfg, seed, run_dir, resume=False, env_id=None, device=None, quiet=False):
    """Returns the list of (steps, episode return) of finished training episodes."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck_path = os.path.join(run_dir, "checkpoint.pt")
    ck = torch.load(ck_path, map_location=device, weights_only=False) if resume else None
    if ck:
        cfg, seed = ck["config"], ck["seed"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    n, T = cfg["num_envs"], cfg["n_steps"]
    envs = make_envs(cfg, n, env_id)
    n_actions = envs.single_action_space.n
    learner = PPOLearner(cfg, envs.single_observation_space, n_actions, device)
    scaler = RewardScaler(n, cfg["gamma"]) if cfg.get("normalize_reward", True) else None
    steps, episodes, best = 0, 0, -np.inf
    if ck:
        learner.load_state_dict(ck["learner"])
        steps, episodes, best = ck["steps"], ck["episodes"], ck["best"]
        rng.bit_generator.state = ck["rng"]
        torch.set_rng_state(ck["torch_rng"].cpu())
        if scaler and "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        print(f"resumed at {steps} steps")
    writer = SummaryWriter(run_dir, purge_step=steps if ck else None)
    eval_envs = []
    if not env_id:
        env_cfg = load_env_config(cfg["env_config"])
        eval_cfg = {**env_cfg, "shaping": {**env_cfg["shaping"], "enabled": False}, "hunger_limit": 0}
        eval_envs = [PacmanEnv(config=eval_cfg) for _ in range(min(n, 4))]

    obs, _ = envs.reset(seed=seed * 1_000_000 + steps)
    cur = learner.codec.store(obs)
    ep_ret, ep_len, ep_level = np.zeros(n), np.zeros(n, int), np.ones(n, int)
    ended_n = hungry_n = 0
    history = []
    next_log = (steps // cfg["log_every"] + 1) * cfg["log_every"]
    next_eval = (steps // cfg["eval_every"] + 1) * cfg["eval_every"]
    t_log, s_log = time.perf_counter(), steps
    stats = {}
    if not quiet:
        print(f"run {run_dir}, device {device}, {n} envs x {T} steps per update", flush=True)

    store_shapes = [x.shape[1:] for x in cur]
    try:
        while steps < cfg["total_steps"]:
            if cfg["anneal_lr"]:
                learner.set_lr(cfg["lr"] * max(0.0, 1 - steps / cfg["total_steps"]))
            b_obs = [np.zeros((T, n) + s, x.dtype) for s, x in zip(store_shapes, cur)]
            b_act = np.zeros((T, n), np.int64)
            b_logp, b_val, b_rew, b_end = (np.zeros((T, n), np.float32) for _ in range(4))
            for t in range(T):
                for k, x in enumerate(cur):
                    b_obs[k][t] = x
                a, logp, v = learner.act(cur)
                nobs, r_env, term, trunc, info = envs.step(a)
                ended = term | trunc
                r = scaler(np.asarray(r_env, np.float64), ended) if scaler else \
                    np.asarray(r_env, np.float32).copy()
                for i in np.flatnonzero(trunc & ~term):  # bootstrap a truncated episode
                    final = info["final_obs"][i]
                    one = ({"grid": final["grid"][None], "vec": final["vec"][None]}
                           if learner.codec.dict_obs else final[None])
                    r[i] += cfg["gamma"] * learner.value(learner.codec.store(one))[0]
                b_act[t], b_logp[t], b_val[t], b_rew[t], b_end[t] = a, logp, v, r, ended
                ep_ret += r_env
                ep_len += 1
                for i in np.flatnonzero(ended):
                    src = info["final_info"]
                    if "level" in src:
                        ep_level[i] = max(ep_level[i], src["level"][i])
                    episodes += 1
                    history.append((steps + (t + 1) * n, float(ep_ret[i])))
                    hunger = bool(src["hunger"][i]) if "hunger" in src else False
                    ended_n += 1
                    hungry_n += hunger
                    writer.add_scalar("episode/reward", ep_ret[i], steps)
                    writer.add_scalar("episode/length", ep_len[i], steps)
                    writer.add_scalar("episode/hunger", float(hunger), steps)
                    if "score" in src:
                        writer.add_scalar("episode/score", src["score"][i], steps)
                        writer.add_scalar("episode/level_reached", ep_level[i], steps)
                    ep_ret[i], ep_len[i] = 0.0, 0
                    ep_level[i] = info["level"][i] if "level" in info else 1
                if "level" in info:
                    ep_level = np.maximum(ep_level, np.where(ended, ep_level, info["level"]))
                cur = learner.codec.store(nobs)
            steps += T * n
            last_v = learner.value(cur)
            adv, ret = compute_gae(b_rew, b_val, b_end, last_v, cfg["gamma"], cfg["gae_lambda"])
            flat = lambda x: x.reshape((T * n,) + x.shape[2:])  # noqa: E731
            stats = learner.update(tuple(flat(x) for x in b_obs), flat(b_act), flat(b_logp),
                                   flat(adv), flat(ret), flat(b_val), rng)

            if steps >= next_log:
                now = time.perf_counter()
                sps = (steps - s_log) / (now - t_log)
                writer.add_scalar("perf/steps_per_sec", sps, steps)
                for k, v in stats.items():
                    writer.add_scalar(f"train/{k}", v, steps)
                msg = (f"{steps:>9} steps  {sps:,.0f} steps/s  episodes {episodes}  "
                       f"pi {stats['policy_loss']:.4f}  v {stats['value_loss']:.3f}  "
                       f"ent {stats['entropy']:.3f}  kl {stats['approx_kl']:.4f}  "
                       f"clip {stats['clipfrac']:.3f}  ev {stats['explained_variance']:.2f}")
                if ended_n:
                    msg += f"  hunger {hungry_n}/{ended_n}"
                    ended_n = hungry_n = 0
                if not quiet:
                    print(msg, flush=True)
                next_log += cfg["log_every"]
                t_log, s_log = time.perf_counter(), steps

            if eval_envs and (steps >= next_eval or steps >= cfg["total_steps"]):
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
                    "mean_food": float(np.mean([r["food"] for r in res])),
                }
                for k, v in summary.items():
                    if k != "steps":
                        writer.add_scalar(f"eval/{k}", v, steps)
                with open(os.path.join(run_dir, "evals.jsonl"), "a") as f:
                    f.write(json.dumps({**summary, "episodes": res}) + "\n")
                print(f"eval @ {steps}: " + ", ".join(f"{k} {v:.2f}" for k, v in summary.items()
                                                       if k != "steps")
                      + f"  ({time.perf_counter() - t0:.0f} s)", flush=True)
                state = {"algo": "ppo", "config": cfg, "seed": seed, "n_actions": n_actions,
                         "env_config": load_env_config(cfg["env_config"]),
                         "learner": learner.state_dict(), "steps": steps, "episodes": episodes,
                         "best": max(best, metric), "rng": rng.bit_generator.state,
                         "scaler": scaler.state_dict() if scaler else None,
                         "torch_rng": torch.get_rng_state()}
                if metric > best:
                    best = metric
                    save_atomic({**state, "eval": summary}, os.path.join(run_dir, "best.pt"),
                                torch.save)
                save_atomic(state, ck_path, torch.save)
                next_eval += cfg["eval_every"]
                t_log, s_log = time.perf_counter(), steps
    finally:
        envs.close()
        for e in eval_envs:
            e.close()
        writer.close()
    return history


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "ppo.yaml"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    run_dir = os.path.join(ROOT, "runs", args.run_name)
    if args.resume:
        train(None, None, run_dir, resume=True)
        return
    if os.path.exists(os.path.join(run_dir, "checkpoint.pt")):
        sys.exit(f"{run_dir} already has a checkpoint; use --resume or another --run-name")
    cfg = load_config(args.config)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    train(cfg, args.seed, run_dir)


if __name__ == "__main__":
    main()
