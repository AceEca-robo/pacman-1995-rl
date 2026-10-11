#!/usr/bin/env python3
"""Train DQN on Pacman1995-v0.

    python scripts/train.py --config configs/dqn.yaml --seed 0 --run-name dqn0
    python scripts/train.py --run-name dqn0 --resume

Writes TensorBoard logs, checkpoint.pt (+ buffer.npz), best.pt and
evals.jsonl to runs/<run-name>/. A config may name a "base" config whose
values it overrides.
"""

import argparse
import contextlib
import functools
import json
import os
import signal
import sys
import threading
import time

import gymnasium as gym
import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agents.bc import BCAgent  # noqa: E402
from agents.dqn import DQNLearner, NStep, make_buffer, n_planes, pack_grid, pack_grids  # noqa: E402
from agents.env_pool import EnvPool  # noqa: E402
from agents.learner_proc import LearnerProcess  # noqa: E402
from agents.shared_replay import (SharedReplay, StreamWriter, make_packed_env,  # noqa: E402
                                  remove_stale_shm)
from agents.heuristic_agent import HeuristicAgent  # noqa: E402
from env.pacman_env import (CH, PacmanEnv, grid_extras, load_env_config, obs_dims,  # noqa: E402
                             parse_start_level)


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


def start_level_at(cfg, steps):
    """The training envs' start_level at this step: the last entry of
    cfg["start_level_schedule"] ([[from_step, spec], ...]) whose from_step
    <= steps; None without a schedule (the env config's start_level)."""
    spec = None
    for from_step, s in cfg.get("start_level_schedule") or []:
        if steps >= from_step:
            spec = s
    return spec


def dqfd_schedule(cfg, steps):
    """(demo share of the batch, margin weight) at this step from
    cfg["dqfd"]["anneal"] ({from_step, steps, demo_frac: [start, end],
    lambda_margin: [start, end]}, linear from from_step over steps, then the
    end values); (None, None) without it (PER's share, the config's weight)."""
    an = (cfg.get("dqfd") or {}).get("anneal")
    if not an:
        return None, None
    frac = min(1.0, max(0.0, (steps - an["from_step"]) / an["steps"]))

    def lerp(ab):
        return None if ab is None else ab[0] + frac * (ab[1] - ab[0])
    return lerp(an.get("demo_frac")), lerp(an.get("lambda_margin"))


def expand_first_conv(learner_state):
    """In place: the first convolution of online, target and the Adam
    moments gets one more input channel (the new last grid plane) with zero
    weights, so the network computes what it did before."""
    old_shape = learner_state["online"]["conv.0.weight"].shape
    for net in ("online", "target"):
        w = learner_state[net]["conv.0.weight"]
        learner_state[net]["conv.0.weight"] = torch.cat([w, w.new_zeros(w[:, :1].shape)], 1)
    # conv.0.weight is the online network's first parameter: optimizer state 0
    st = learner_state["opt"]["state"].get(0, {})
    for k in ("exp_avg", "exp_avg_sq"):
        if k in st:
            assert st[k].shape == old_shape, (k, st[k].shape, old_shape)
            st[k] = torch.cat([st[k], st[k].new_zeros(st[k][:, :1].shape)], 1)


def per_beta(cfg, steps):
    per = cfg["per"]
    frac = min(1.0, steps / (per["beta_steps"] or cfg["total_steps"]))
    return per["beta_start"] + frac * (per["beta_end"] - per["beta_start"])


class Collector:
    """Steps the vector env, turns steps into n-step transitions in the
    buffer and logs finished episodes."""

    def __init__(self, envs, cfg, buffer, writer, seed, lock=None):
        self.envs, self.buffer, self.writer = envs, buffer, writer
        self.lock = lock or contextlib.nullcontext()  # guards the buffer (async_learner)
        self.n = envs.num_envs
        self.obs, info = envs.reset(seed=seed)
        self.cur = pack_grids(self.obs["grid"])
        self.nsteps = [NStep(cfg["n_step"], cfg["gamma"]) for _ in range(self.n)]
        self.ep_reward = np.zeros(self.n)
        self.ep_len = np.zeros(self.n, int)
        self.ep_start = np.array(info["level"], int)  # start level of the running episodes
        self.ep_level = self.ep_start.copy()
        self.ended = self.hungry = 0  # episodes ended / ended by hunger_limit, since last read
        cells = cfg.get("track_cells") or []
        self.track = (np.array([c[0] for c in cells]), np.array([c[1] for c in cells])) \
            if cells else None
        self.tracked = 0  # pacman steps on track_cells since last read
        self.prefixed = 0  # episodes started from an endgame prefix since last read
        self.ids = np.arange(self.n)  # envs waiting for an action (all, for vector envs)
        self.last_n = self.n  # env steps taken by the last step()

    def set_start_level(self, spec):
        self.envs.call("set_start_level", spec)

    def step(self, actions, log_step, prefix="episode"):
        """Returns the number of episodes that ended."""
        obs = self.obs
        if self.track:  # pacman's steps on the tracked cells (before this move)
            self.tracked += int(obs["grid"][:, CH["pacman"], self.track[1], self.track[0]].sum())
        nobs, rew, term, trunc, info = self.envs.step(actions)
        packed_next = pack_grids(nobs["grid"])
        ended_count = 0
        new = []  # this step's finished n-step transitions, added in one batch
        for i in range(self.n):
            ended = term[i] or trunc[i]
            if ended:  # nobs holds the next episode's first obs
                last_packed = pack_grid(info["final_obs"][i]["grid"])
                last_vec = info["final_obs"][i]["vec"]
            else:
                last_packed, last_vec = packed_next[i], nobs["vec"][i]
            new += self.nsteps[i].push(self.cur[i], obs["vec"][i], actions[i], rew[i],
                                       last_packed, last_vec, term[i], ended)
            self.ep_reward[i] += rew[i]
            self.ep_len[i] += 1
            src = info["final_info"] if ended else info
            self.ep_level[i] = max(self.ep_level[i], src["level"][i])
            if ended:
                ended_count += 1
                # same-step autoreset: info holds the next episode's reset info
                if "prefix" in info and info["_prefix"][i] and info["prefix"][i] >= 0:
                    self.prefixed += 1
                hunger = bool(src["hunger"][i]) if "hunger" in src else False
                self.ended += 1
                self.hungry += hunger
                self.writer.add_scalar(f"{prefix}/hunger", float(hunger), log_step)
                self.writer.add_scalar(f"{prefix}/reward", self.ep_reward[i], log_step)
                self.writer.add_scalar(f"{prefix}/score", src["score"][i], log_step)
                self.writer.add_scalar(f"{prefix}/length", self.ep_len[i], log_step)
                self.writer.add_scalar(f"{prefix}/level_reached", self.ep_level[i], log_step)
                self.writer.add_scalar(f"{prefix}/start_level", self.ep_start[i], log_step)
                self.writer.add_scalar(f"{prefix}/levels_cleared",
                                       self.ep_level[i] - self.ep_start[i], log_step)
                self.ep_reward[i], self.ep_len[i] = 0.0, 0
                self.ep_start[i] = self.ep_level[i] = info["level"][i]
        self.add(new)
        self.cur, self.obs = packed_next, nobs
        return ended_count

    def add(self, transitions):
        with self.lock:
            self.buffer.add_batch(transitions)


class FrameCollector(Collector):
    """learner_process: the envs return bit-packed grids
    (agents/shared_replay.py PackedObs) and every step goes into a
    StreamWriter (frames in the shared replay). Same episode logging."""

    def __init__(self, envs, cfg, stream, writer, seed):
        if cfg.get("track_cells"):
            raise ValueError("track_cells needs the float grid (not with learner_process)")
        self.envs, self.writer, self.lock = envs, writer, contextlib.nullcontext()
        self.buffer = stream  # .size: transitions written (warm_start's report)
        self.n = envs.num_envs
        self.obs, info = envs.reset(seed=seed)
        self.cur = self.obs["grid"]
        stream.begin(self.cur, self.obs["vec"])
        self.ep_reward = np.zeros(self.n)
        self.ep_len = np.zeros(self.n, int)
        self.ep_start = np.array(info["level"], int)
        self.ep_level = self.ep_start.copy()
        self.ended = self.hungry = 0
        self.track, self.tracked, self.prefixed = None, 0, 0
        self.ids, self.last_n = np.arange(self.n), self.n

    def step(self, actions, log_step, prefix="episode"):
        nobs, rew, term, trunc, info = self.envs.step(actions)
        ended = np.flatnonzero(term | trunc)
        fin = [info["final_obs"][i] for i in ended]
        self.buffer.step(actions, rew, nobs["grid"], nobs["vec"], term, trunc,
                         np.stack([f["grid"] for f in fin]) if fin else None,
                         np.stack([f["vec"] for f in fin]) if fin else None)
        self.ep_reward += rew
        self.ep_len += 1
        live = ~(term | trunc)  # ended envs: info["level"] is the next episode's
        self.ep_level[live] = np.maximum(self.ep_level[live], info["level"][live])
        for i in ended:
            src = info["final_info"]
            level = max(int(self.ep_level[i]), int(src["level"][i]))
            if "prefix" in info and info["_prefix"][i] and info["prefix"][i] >= 0:
                self.prefixed += 1
            hunger = bool(src["hunger"][i]) if "hunger" in src else False
            self.ended += 1
            self.hungry += hunger
            w = self.writer
            w.add_scalar(f"{prefix}/hunger", float(hunger), log_step)
            w.add_scalar(f"{prefix}/reward", self.ep_reward[i], log_step)
            w.add_scalar(f"{prefix}/score", src["score"][i], log_step)
            w.add_scalar(f"{prefix}/length", self.ep_len[i], log_step)
            w.add_scalar(f"{prefix}/level_reached", level, log_step)
            w.add_scalar(f"{prefix}/start_level", self.ep_start[i], log_step)
            w.add_scalar(f"{prefix}/levels_cleared", level - self.ep_start[i], log_step)
            self.ep_reward[i], self.ep_len[i] = 0.0, 0
            self.ep_start[i] = self.ep_level[i] = info["level"][i]
        self.cur, self.obs = nobs["grid"], nobs
        return len(ended)


class PoolCollector(FrameCollector):
    """FrameCollector over agents/env_pool.py EnvPool: step() sends actions
    to the envs in self.ids and returns once the next envs have results
    (they become self.ids); envs step independently, no barrier."""

    def __init__(self, pool, cfg, stream, writer, seed):
        if cfg.get("track_cells"):
            raise ValueError("track_cells needs the float grid (not with learner_process)")
        self.envs, self.writer, self.lock = pool, writer, contextlib.nullcontext()
        self.buffer = stream
        self.n = pool.n
        pool.drain()
        grids, vecs, infos = pool.reset(seed=seed)
        self.cur, self.obs = grids, {"vec": vecs}
        stream.begin(grids, vecs)
        self.ep_reward = np.zeros(self.n)
        self.ep_len = np.zeros(self.n, int)
        self.ep_start = np.array([inf["level"] for inf in infos], int)
        self.ep_level = self.ep_start.copy()
        self.ended = self.hungry = 0
        self.track, self.tracked, self.prefixed = None, 0, 0
        self.ids, self.last_n = np.arange(self.n), 0
        self.actions = np.zeros(self.n, np.int64)  # last action sent to each env

    def set_start_level(self, spec):
        self.envs.call_async("set_start_level", spec)

    def step(self, actions, log_step, prefix="episode", ids=None):
        """ids: send the actions to these envs only (default self.ids); the
        other ready envs stay parked."""
        pool = self.envs
        send = self.ids if ids is None else ids
        self.actions[send] = actions
        pool.send(send, actions)
        if not pool.busy.any():
            self.ids, self.last_n = np.zeros(0, np.int64), 0
            return 0
        ids, res = pool.recv()
        rew = np.array([x[0] for x in res])
        term = np.array([x[1] for x in res])
        trunc = np.array([x[2] for x in res])
        fin = [x[4] for x in res if x[4] is not None]
        grids, vecs = pool.grid[ids], pool.vec[ids]
        self.buffer.step(self.actions[ids], rew, grids, vecs, term, trunc,
                         np.stack([f[0] for f in fin]) if fin else None,
                         np.stack([f[1] for f in fin]) if fin else None, ids=ids)
        self.cur[ids], self.obs["vec"][ids] = grids, vecs
        self.ep_reward[ids] += rew
        self.ep_len[ids] += 1
        n_ended = 0
        w = self.writer
        for j, i in enumerate(ids):
            info, final = res[j][3], res[j][4]
            if final is None:
                self.ep_level[i] = max(self.ep_level[i], info["level"])
                continue
            n_ended += 1
            src = final[2]
            level = max(int(self.ep_level[i]), int(src["level"]))
            if info.get("prefix", -1) >= 0:
                self.prefixed += 1
            hunger = bool(src.get("hunger", False))
            self.ended += 1
            self.hungry += hunger
            w.add_scalar(f"{prefix}/hunger", float(hunger), log_step)
            w.add_scalar(f"{prefix}/reward", self.ep_reward[i], log_step)
            w.add_scalar(f"{prefix}/score", src["score"], log_step)
            w.add_scalar(f"{prefix}/length", self.ep_len[i], log_step)
            w.add_scalar(f"{prefix}/level_reached", level, log_step)
            w.add_scalar(f"{prefix}/start_level", self.ep_start[i], log_step)
            w.add_scalar(f"{prefix}/levels_cleared", level - self.ep_start[i], log_step)
            self.ep_reward[i], self.ep_len[i] = 0.0, 0
            self.ep_start[i] = self.ep_level[i] = info["level"]
        self.ids, self.last_n = ids, len(ids)
        return n_ended


class BackgroundLearner(threading.Thread):
    """async_learner: gradient steps in a thread next to the acting loop.
    The acting loop hands out update credit (env steps / train_every, as in
    the synchronous loop) and blocks while more than async_max_lag updates
    are pending, so the replay ratio stays that of train_every. The buffer
    is shared under buffer_lock; the learner's lock keeps the acting loop's
    forward passes from reading the weights in the middle of an optimizer
    step. Target syncs run in this thread between updates. Not deterministic
    (thread timing decides which transitions each update samples)."""

    def __init__(self, learner, buffer, cfg, rng):
        super().__init__(daemon=True)
        self.learner, self.buffer, self.cfg, self.rng = learner, buffer, cfg, rng
        self.per = cfg["per"]["enabled"]
        self.max_lag = cfg.get("async_max_lag") or 4
        self.buffer_lock = threading.Lock()
        learner.lock = threading.Lock()
        learner.enable_actor_stream()
        self.cv = threading.Condition()
        self.credit, self.beta, self.busy, self.sync_pending = 0.0, None, False, False
        self.stopping, self.error = False, None
        self.n_updates, self.losses, self.qs = 0, [], []

    def give(self, credit, beta):
        with self.cv:
            self.credit += credit
            self.beta = beta
            self.cv.notify_all()
            while self.credit >= self.max_lag + 1 and self.error is None:
                self.cv.wait()
        self.check()

    def request_target_sync(self):
        with self.cv:
            self.sync_pending = True
            self.cv.notify_all()

    def drain(self):
        """Wait until every whole update handed out so far is done and the
        thread is idle (before evaluating / saving)."""
        with self.cv:
            while (self.credit >= 1 or self.busy or self.sync_pending) and self.error is None:
                self.cv.wait()
        self.check()

    def take_stats(self):
        with self.cv:
            losses, qs, self.losses, self.qs = self.losses, self.qs, [], []
            return losses, qs, self.n_updates

    def check(self):
        if self.error is not None:
            raise RuntimeError("learner thread failed") from self.error

    def stop(self):
        with self.cv:
            self.stopping = True
            self.cv.notify_all()
        self.join()

    def _flush(self):
        """Apply the previous update's priorities (deferred by one update, so
        launching an update overlaps the GPU finishing the one before)."""
        if self.pending is not None:
            idx, td, done = self.pending
            self.pending = None
            if done is not None:
                done.synchronize()
            if self.per:
                with self.buffer_lock:
                    self.buffer.update_priorities(idx, td.numpy())

    def run(self):
        cuda = str(self.learner.device).startswith("cuda")
        self.pending = None
        try:
            while True:
                with self.cv:
                    idle = not self.stopping and self.credit < 1 and not self.sync_pending
                if idle:
                    self._flush()
                with self.cv:
                    while not self.stopping and self.credit < 1 and not self.sync_pending:
                        self.cv.wait()
                    if self.stopping:
                        return
                    sync, update, beta = self.sync_pending, self.credit >= 1, self.beta
                    self.sync_pending, self.busy = False, True
                if sync:
                    self.learner.sync_target()
                if update:
                    with self.buffer_lock:
                        batch, idx, weights = self.buffer.sample(self.rng, self.cfg["batch_size"], beta)
                    loss, mq, td = self.learner.update(batch, weights)
                    if cuda:
                        td, done = td.to("cpu", non_blocking=True), torch.cuda.Event()
                        done.record()
                    else:
                        td, done = td.cpu(), None
                    self._flush()
                    self.pending = (idx, td, done)
                with self.cv:
                    if update:
                        self.credit -= 1
                        self.n_updates += 1
                        self.losses.append(loss)
                        self.qs.append(mq)
                    # idle only once the last update's priorities are in
                    self.busy = self.credit < 1 and self.pending is not None
                    self.cv.notify_all()
                if self.busy:
                    self._flush()
                    with self.cv:
                        self.busy = False
                        self.cv.notify_all()
        except BaseException as e:  # noqa: BLE001 -- handed to the acting loop
            with self.cv:
                self.error = e
                self.cv.notify_all()


def warm_start(collector, cfg, device):
    """Fill the buffer with cfg["warm_start"] env steps of the warm start
    agent: the heuristic, or (cfg["warm_start_agent"] = a BC checkpoint) the
    BC policy, greedy, all envs in one forward pass. These steps do not count
    towards total_steps or the schedules."""
    kind = cfg.get("warm_start_agent") or "heuristic"
    if kind == "heuristic":
        agent = HeuristicAgent(cfg["warm_start_agent_config"],
                               n_actions=collector.envs.single_action_space.n)
    else:
        agent = BCAgent(os.path.join(ROOT, kind), device=device)
    done, t0 = 0, time.perf_counter()
    while done < cfg["warm_start"]:
        obs = collector.obs
        ids = collector.ids
        if kind == "heuristic":
            actions = np.array([agent.act({"grid": obs["grid"][i], "vec": obs["vec"][i]})
                                for i in ids])
        else:
            actions = agent.act_batch(collector.cur[ids], obs["vec"][ids])
        collector.step(actions, done, prefix="warm_start")
        done += collector.last_n
    print(f"warm start: {done} {kind} steps, buffer {collector.buffer.size} "
          f"({time.perf_counter() - t0:.0f} s)", flush=True)


def generate_demo(make_collector, cfg, replay, n_actions, seed, device):
    """DQfD: the demo agent (a BC checkpoint, greedy unless demo_epsilon)
    plays in the training envs until the demo streams hold
    dqfd.demo_transitions transitions (or a stream is full)."""
    dq = cfg["dqfd"]
    agent = BCAgent(os.path.join(ROOT, dq["demo_agent"]), device=device)
    stream = StreamWriter(replay, range(replay.n_demo), cfg["n_step"], cfg["gamma"])
    col = make_collector(stream, 10**9 + seed)
    rng = np.random.default_rng([seed, 7])
    t0, steps, eps = time.perf_counter(), 0, dq.get("demo_epsilon") or 0.0
    pool = isinstance(col, PoolCollector)
    while stream.size < dq["demo_transitions"]:
        # the pool steps envs unevenly: an env whose stream is full is parked
        # (vector envs step together: stop at the first full stream)
        ids = col.ids[stream.room()[col.ids]] if pool else col.ids
        if not pool and stream.full():
            break
        if pool and not len(ids) and not col.envs.busy.any():
            break  # every stream full
        actions = agent.act_batch(col.cur[ids], col.obs["vec"][ids]) if len(ids) \
            else np.zeros(0, np.int64)
        rand = rng.random(len(ids)) < eps
        actions[rand] = rng.integers(n_actions, size=int(rand.sum()))
        if pool:
            col.step(actions, steps, prefix="demo", ids=ids)
        else:
            col.step(actions, steps, prefix="demo")
        steps += col.last_n
        if steps % 100_000 < col.last_n:
            print(f"demo: {steps} steps, {stream.size} transitions "
                  f"({time.perf_counter() - t0:.0f} s)", flush=True)
    print(f"demo: {stream.size} transitions of {dq['demo_agent']} "
          f"({time.perf_counter() - t0:.0f} s)", flush=True)


def pretrain(bg, cfg, writer, evaluate_fn):
    """DQfD phase 0: dqfd.pretrain_updates updates on the demo transitions
    only (no env steps; the agent streams are still empty), target sync every
    pretrain_target_update updates. Returns the eval summary at the end."""
    dq = cfg["dqfd"]
    total, tu = dq["pretrain_updates"], dq["pretrain_target_update"]
    log_every = dq.get("pretrain_log_every") or 10000
    beta = cfg["per"]["beta_start"] if cfg["per"]["enabled"] else None
    t0, done = time.perf_counter(), bg.n_updates
    t_log, d_log = t0, done
    while done < total:
        chunk = min(tu - done % tu, total - done)
        bg.give(chunk, beta)  # returns with <= async_max_lag updates pending
        bg.drain()
        done = bg.n_updates
        if done % tu == 0:
            bg.request_target_sync()
        if done % log_every == 0 or done >= total:
            losses, qs, _ = bg.take_stats()
            now = time.perf_counter()
            ups = (done - d_log) / (now - t_log)
            msg = f"pretrain {done:>7} updates  {ups:,.0f} updates/s"
            if losses:
                loss_m, q_m = losses[0].item(), qs[0].item()
                writer.add_scalar("pretrain/loss", loss_m, done)
                writer.add_scalar("pretrain/mean_q", q_m, done)
                writer.add_scalar("pretrain/margin_loss", bg.last_margin_loss, done)
                msg += f"  loss {loss_m:.4f}  Q {q_m:.2f}  margin {bg.last_margin_loss:.4f}"
            print(msg, flush=True)
            t_log, d_log = now, done
        if dq.get("pretrain_eval_every") and done % dq["pretrain_eval_every"] == 0 \
                and done < total:
            evaluate_fn(f"pretrain@{done}")
    print(f"pretrain: {done} updates ({time.perf_counter() - t0:.0f} s)", flush=True)
    return evaluate_fn("pretrain")


def eval_tasks(cfg):
    """(seed, start level, stop at the first level up) per eval game:
    eval_episodes games from level 1 (to game over or eval_max_episode_steps),
    then eval_level_episodes games started on each of cfg["eval_levels"],
    stopped once the start maze is cleared (same seeds; with eval_seed_blocks
    maze lv on its own seeds eval_seed + (lv - 1) * eval_level_episodes ..,
    as evaluate.py --seed-blocks)."""
    seeds = range(cfg["eval_seed"], cfg["eval_seed"] + cfg["eval_episodes"])
    tasks = [(s, 1, False) for s in seeds]
    k = cfg.get("eval_level_episodes") or cfg["eval_episodes"]
    for lv in cfg.get("eval_levels") or []:
        s0 = cfg["eval_seed"] + ((lv - 1) * k if cfg.get("eval_seed_blocks") else 0)
        tasks += [(s, lv, True) for s in range(s0, s0 + k)]
    return tasks


def evaluate(learner, envs, cfg):
    """Greedy episodes over a fixed seed set (eval_tasks), the envs stepping
    in a batch. Results of the level-1 games come first (sorted by seed),
    then the per-maze games ("maze" set, sorted by maze and seed)."""
    tasks = eval_tasks(cfg)
    results, live = [], []  # live: [env, obs, task, reward, length, max level, food, start]

    def start(env, task):
        obs, info = env.reset(seed=task[0], options={"start_level": task[1]})
        return [env, obs, task, 0.0, 0, info["level"], 0, info["level"]]

    for env in envs:
        if tasks:
            live.append(start(env, tasks.pop(0)))
    while live:
        q = learner.q_values(pack_grids(np.stack([e[1]["grid"] for e in live]),
                                        live[0][0].grid_extras),
                             np.stack([e[1]["vec"] for e in live]))
        for e, a in zip(list(live), q.argmax(1)):
            obs, r, term, trunc, info = e[0].step(a)
            e[1], e[3], e[4], e[5] = obs, e[3] + r, e[4] + 1, max(e[5], info["level"])
            e[6] += info["events"]["eaten_dot"] + info["events"]["eaten_energizer"]
            seed, level, stop = e[2]
            cut = stop and cfg.get("eval_level_max_steps") and e[4] >= cfg["eval_level_max_steps"]
            if term or trunc or cut or (stop and e[5] > e[7]):
                res = {"seed": seed, "reward": e[3], "score": info["score"],
                       "length": e[4], "level": e[5], "food": e[6],
                       "levels_cleared": e[5] - e[7]}
                if stop:
                    res["maze"] = level
                results.append(res)
                if tasks:
                    e[:] = start(e[0], tasks.pop(0))
                else:
                    live.remove(e)
    return sorted(results, key=lambda x: (x.get("maze", 0), x["seed"]))


def best_metric(name, results, maze_results=()):
    """Value to pick best.pt by (higher is better).
    "level1_cleared" (the rule since v0.2): share of eval games that cleared
    level 1, ties broken by mean food eaten; returned as
    share * 1000 + mean food / 1000, which orders lexicographically (one game
    changes the share term by >= 1000 / episodes, food / 1000 stays < 1).
    "mean_reward" (dqn4..dqn10: a median ignores games stuck until the step
    limit while they are under half), "median_reward" / "median_score"; old
    configs say "reward" / "score".
    "levels_cleared_min_maze": the same mean, ties by the lowest per-maze
    clear share (needs eval_levels).
    "levels_cleared" (the rule since v0.3): mean levels cleared per eval game
    (evaluation starts on level 1), ties by mean food; returned as
    mean * 1e5 + mean food / 1000 (one game changes the mean by >= 1 /
    episodes, i.e. the first term by >= 1e5 / episodes, food / 1000 stays
    far below that)."""
    if name == "levels_cleared_min_maze":
        # rule of the BC -> RL runs: mean levels cleared from level 1, ties by
        # the lowest share of per-maze games clearing their maze (in [0, 1])
        mean = float(np.mean([r["levels_cleared"] for r in results]))
        shares = {}
        for r in maze_results:
            shares.setdefault(r["maze"], []).append(r["levels_cleared"] >= 1)
        return mean * 1e5 + min(float(np.mean(v)) for v in shares.values())
    if name == "levels_cleared":
        mean = float(np.mean([r["levels_cleared"] for r in results]))
        return mean * 1e5 + float(np.mean([r["food"] for r in results])) / 1000
    if name == "level1_cleared":
        share = float(np.mean([r["level"] >= 2 for r in results]))
        return share * 1000 + float(np.mean([r["food"] for r in results])) / 1000
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
    p.add_argument("--extend-steps", type=int, default=0,
                   help="with --resume: add this many steps to the run's total_steps")
    p.add_argument("--resume-from", metavar="CHECKPOINT",
                   help="start a new run (--config) from another run's checkpoint: weights, "
                        "optimizer, step counts and schedules continue (no replay)")
    p.add_argument("--pretrain-only", action="store_true",
                   help="dqfd: stop after phase 0 (checkpoint.pt; --resume continues with RL)")
    args = p.parse_args()

    # tmux kill-session (SIGHUP) / kill (SIGTERM): exit through the finally
    # blocks, which stop the learner process and unlink the shared memory
    def _exit(signum, frame):
        raise SystemExit(f"signal {signum}")
    signal.signal(signal.SIGTERM, _exit)
    signal.signal(signal.SIGHUP, _exit)
    stale = remove_stale_shm()  # left behind by a run killed harder (SIGKILL, OOM)
    if stale:
        print(f"removed {len(stale)} stale shared memory segments")

    run_dir = os.path.join(ROOT, "runs", args.run_name)
    ck_path = os.path.join(run_dir, "checkpoint.pt")
    buf_path = os.path.join(run_dir, "buffer.npz")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.resume:
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        cfg, seed = ck["config"], ck["seed"]
        # options added after the run started keep their defaults (off)
        cfg = {**load_config(os.path.join(ROOT, "configs", "dqn.yaml")), **cfg}
        if args.extend_steps:
            # continue a finished run: more steps, everything else as it was
            cfg["total_steps"] += args.extend_steps
            with open(os.path.join(run_dir, "config.yaml"), "w") as f:
                yaml.safe_dump(cfg, f)
            print(f"total_steps extended to {cfg['total_steps']}")
    else:
        if os.path.exists(ck_path):
            sys.exit(f"{run_dir} already has a checkpoint; use --resume or another --run-name")
        cfg, seed, ck = load_config(args.config), args.seed, None
        if args.resume_from:
            ck = torch.load(args.resume_from, map_location=device, weights_only=False)
            seed = ck["seed"]
            env_old = ck["config"]["env_config"]
            if cfg.get("resume_add_cell_safety"):
                # the old run's env config plus obs.cell_safety: one more input plane
                env_old = {**env_old, "obs": {**(env_old.get("obs") or {}), "cell_safety": True}}
            if cfg["network"] != ck["config"]["network"] or cfg["env_config"] != env_old:
                sys.exit(f"{args.resume_from}: network or env config differs from {args.config}")
            if cfg.get("resume_add_cell_safety"):
                expand_first_conv(ck["learner"])
            if ck["steps"] >= cfg["total_steps"]:
                sys.exit(f"{args.resume_from} is at {ck['steps']} steps >= total_steps")
            print(f"continuing {args.resume_from} ({ck['steps']} steps) as {args.run_name}")
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg, f)

    torch.manual_seed(seed)
    env_cfg = load_env_config(cfg["env_config"])
    n_actions = env_cfg["actions"]
    # evaluation without shaping, hunger_limit, prefixes or the endgame dot
    # bonus, so eval rewards compare across runs
    # (and always from level 1, so levels cleared compare too)
    eval_env_cfg = {**env_cfg, "shaping": {**env_cfg["shaping"], "enabled": False},
                    "hunger_limit": 0, "prefix_prob": 0.0, "endgame_dot": {"enabled": False},
                    "start_level": 1}
    if cfg.get("eval_max_episode_steps"):
        eval_env_cfg["max_episode_steps"] = cfg["eval_max_episode_steps"]
    n_channels, vec_dim = obs_dims(env_cfg)
    extras = grid_extras(env_cfg)
    proc = bool(cfg.get("learner_process"))
    if "cell_safety" in extras and not proc:
        sys.exit("obs.cell_safety needs learner_process: true")
    dq = cfg.get("dqfd") or {}
    if dq.get("enabled") and not proc:
        sys.exit("dqfd needs learner_process: true")
    if proc and cfg["warm_start"] and (cfg.get("warm_start_agent") or "heuristic") == "heuristic":
        sys.exit("learner_process: the warm start agent must be a BC checkpoint")
    rng = np.random.default_rng(seed)
    steps = n_updates = episodes = 0
    best = -np.inf
    init = {}  # initial weights: applied here, or in the learner process (proc)
    if not ck and cfg.get("init_checkpoint"):
        # fine-tune: start from another run's network (online and target);
        # optimizer, buffer, step counts and schedules start fresh
        ick = torch.load(os.path.join(ROOT, cfg["init_checkpoint"]), map_location="cpu",
                         weights_only=False)
        if ick.get("n_actions", 5) != n_actions or \
                obs_dims(ick.get("env_config") or {}) != (n_channels, vec_dim):
            sys.exit(f"{cfg['init_checkpoint']} does not match this env's actions/observation")
        init["online"] = ick["learner"]["online"]
        print(f"initialized from {cfg['init_checkpoint']} ({ick['steps']} steps)")
    if not ck and cfg.get("init_conv_from"):
        # the convolutions of a BC network (agents/bc.py, same layer names);
        # the vec, fc and Q heads keep their fresh initialization
        bc = torch.load(os.path.join(ROOT, cfg["init_conv_from"]), map_location="cpu",
                        weights_only=False)
        init["conv"] = {k[len("conv."):]: v for k, v in bc["net"].items() if k.startswith("conv.")}
        print(f"convolutions from {cfg['init_conv_from']}")
    if ck:
        init["learner"] = ck["learner"]
        steps, n_updates, episodes, best = ck["steps"], ck["n_updates"], ck["episodes"], ck["best"]
        rng.bit_generator.state = ck["rng"]
        torch.set_rng_state(ck["torch_rng"].cpu())  # map_location moved it to the device
        if args.resume_from:  # best.pt is this run's own
            best = -np.inf
    n = cfg["num_envs"]
    learner = buffer = replay = None
    if proc:
        # frames in shared memory: one ring per env, plus the demo streams
        demo_sizes = []
        if dq.get("enabled"):
            per_stream = int(np.ceil(dq["demo_transitions"] / n * 1.1)) + 64
            demo_sizes = [per_stream] * n
        replay = SharedReplay.create([cfg["buffer_size"] // n] * n, demo_sizes,
                                     n_planes(n_channels, extras), vec_dim)
    else:
        learner = DQNLearner(cfg, device, n_actions, n_channels, vec_dim)
        if "online" in init:
            learner.online.load_state_dict(init["online"])
            learner.target.load_state_dict(init["online"])
        if "conv" in init:
            learner.online.conv.load_state_dict(init["conv"])
            learner.target.conv.load_state_dict(init["conv"])
        if "learner" in init:
            learner.load_state_dict(init["learner"])
        buffer = make_buffer(cfg, n_channels, vec_dim)
        if ck and os.path.exists(buf_path):
            with np.load(buf_path) as z:
                buffer.load_state_dict(z)
    if ck:
        print(f"resumed at {steps} steps, buffer {buffer.size if buffer else 0}")
    writer = SummaryWriter(run_dir, purge_step=steps if ck else None)

    level_spec = start_level_at(cfg, steps)
    if level_spec is not None:
        parse_start_level(level_spec)  # fail now, not hours later
        env_cfg = {**env_cfg, "start_level": level_spec}
        for _, spec in cfg["start_level_schedule"]:
            parse_start_level(spec)
        print(f"start_level {level_spec}")
    if proc:  # the grid is packed in the env workers
        make = functools.partial(make_packed_env, env_cfg, n_planes(n_channels, extras))
    else:
        make = functools.partial(PacmanEnv, config=env_cfg)
    envs = bg = None
    eval_envs = []
    aff = cfg.get("cpu_affinity") or {}
    try:
        if aff.get("envs"):  # the env workers (and the games they start) inherit it
            os.sched_setaffinity(0, aff["envs"])
        if cfg.get("vector_env", "sync") == "pool":
            if not proc:
                sys.exit("vector_env pool needs learner_process: true")
            envs = EnvPool(make, n, n_planes(n_channels, extras), vec_dim, n_actions,
                           min_ready=cfg.get("pool_min_ready"))
        elif cfg.get("vector_env", "sync") == "async":
            # one process per env copy, observations through shared memory
            envs = gym.vector.AsyncVectorEnv([make] * n, shared_memory=True,
                                             autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
        else:
            envs = gym.vector.SyncVectorEnv([make] * n,
                                            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
        eval_envs = [PacmanEnv(config=eval_env_cfg) for _ in range(cfg.get("eval_envs") or n)]
        if aff.get("actor"):
            os.sched_setaffinity(0, aff["actor"])
        # a resumed run must not replay the same games
        if proc:
            bg = LearnerProcess(cfg, device, (n_actions, n_channels, vec_dim), replay, init,
                                seed=seed * 1_000_000 + steps + 1, n_updates=n_updates)
            actor, state_src = bg.actor, bg
        else:
            if cfg.get("async_learner"):
                bg = BackgroundLearner(learner, buffer, cfg, np.random.default_rng([seed, steps, 1]))
                bg.n_updates = n_updates
            actor = state_src = learner

        def save_state(extra=None):
            state = {"config": cfg, "seed": seed, "n_actions": n_actions,
                     "env_config": env_cfg, "learner": state_src.state_dict(),
                     "steps": steps, "n_updates": bg.n_updates if bg else n_updates,
                     "episodes": episodes, "best": best, "rng": rng.bit_generator.state,
                     "torch_rng": torch.get_rng_state(), **(extra or {})}
            return state

        def run_eval(label=None):
            """Evaluate the acting network, log, append to evals.jsonl;
            -> (summary, best_metric value)."""
            t0 = time.perf_counter()
            all_res = evaluate(actor, eval_envs, cfg)
            res = [r for r in all_res if "maze" not in r]  # games from level 1
            metric = best_metric(cfg["best_metric"], res, [r for r in all_res if "maze" in r])
            summary = {
                "steps": steps,
                "median_reward": float(np.median([r["reward"] for r in res])),
                "mean_reward": float(np.mean([r["reward"] for r in res])),
                "median_score": float(np.median([r["score"] for r in res])),
                "mean_length": float(np.mean([r["length"] for r in res])),
                "level1_cleared": float(np.mean([r["level"] >= 2 for r in res])),
                "mean_food": float(np.mean([r["food"] for r in res])),
                "mean_levels_cleared": float(np.mean([r["levels_cleared"] for r in res])),
                "median_levels_cleared": float(np.median([r["levels_cleared"] for r in res])),
            }
            for lv in cfg.get("eval_levels") or []:
                summary[f"maze{lv}_cleared"] = float(np.mean(
                    [r["levels_cleared"] >= 1 for r in all_res if r.get("maze") == lv]))
            if cfg.get("eval_levels"):
                summary["min_maze_cleared"] = min(summary[f"maze{lv}_cleared"]
                                                  for lv in cfg["eval_levels"])
                summary["mean_maze_cleared"] = float(np.mean(
                    [summary[f"maze{lv}_cleared"] for lv in cfg["eval_levels"]]))
            tag = "eval" if label is None else "pretrain_eval"
            x = steps if label is None else bg.n_updates
            for k, v in summary.items():
                if k != "steps":
                    writer.add_scalar(f"{tag}/{k}", v, x)
            row = {**summary, "episodes": all_res}
            if label is not None:
                row = {"phase": label, "updates": bg.n_updates, **row}
            with open(os.path.join(run_dir, "evals.jsonl"), "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"eval @ {steps if label is None else label}: "
                  + ", ".join(f"{k} {v:.2f}" for k, v in summary.items() if k != "steps")
                  + f"  ({time.perf_counter() - t0:.0f} s)", flush=True)
            return summary, metric

        if proc:
            cls = PoolCollector if isinstance(envs, EnvPool) else FrameCollector

            def make_collector(stream, col_seed):
                return cls(envs, cfg, stream, writer, seed=col_seed)

            stream = StreamWriter(replay, range(replay.n_demo, replay.n_demo + n),
                                  cfg["n_step"], cfg["gamma"])
            if dq.get("enabled"):
                demo_path = os.path.join(ROOT, dq["demo_file"]) if dq.get("demo_file") \
                    else os.path.join(run_dir, "demo.npz")
                if os.path.exists(demo_path):
                    replay.load_demo(demo_path)
                    print(f"demo: {replay.demo_transitions()} transitions from {demo_path}")
                else:
                    generate_demo(make_collector, cfg, replay, n_actions, seed, device)
                    replay.save_demo(demo_path)
                if not (ck and ck.get("pretrained")):
                    summary, _ = pretrain(bg, cfg, writer, lambda label: run_eval(label))
                    state = save_state({"pretrained": True, "eval": summary})
                    save_atomic(state, os.path.join(run_dir, "pretrain.pt"), torch.save)
                    save_atomic(state, ck_path, torch.save)
                if args.pretrain_only:
                    print("pretrain only: stopped before RL (--resume continues)")
                    return
            col = make_collector(stream, seed * 1_000_000 + steps)
            fill = lambda: min(stream.size, n * (cfg["buffer_size"] // n))  # noqa: E731
            nbytes = replay.nbytes
        else:
            col = Collector(envs, cfg, buffer, writer, seed=seed * 1_000_000 + steps,
                            lock=bg.buffer_lock if bg else None)
            fill = lambda: buffer.size  # noqa: E731
            nbytes = buffer.nbytes
        if not ck and cfg["warm_start"]:
            warm_start(col, cfg, device)
        per = cfg["per"]["enabled"]
        min_buffer = min(cfg["learning_starts"], cfg["buffer_size"]) if ck else cfg["batch_size"]

        dq_sched = bool(dq.get("anneal"))
        freeze_steps = cfg.get("freeze_conv_steps") or 0
        conv_frozen = steps < freeze_steps
        if conv_frozen:
            state_src.request_conv_trainable(False)
            print(f"convolutions frozen until {freeze_steps} steps")
        losses, qs = [], []
        update_credit = 0.0
        next_target = (steps // cfg["target_update"] + 1) * cfg["target_update"]
        next_log = (steps // cfg["log_every"] + 1) * cfg["log_every"]
        next_eval = (steps // cfg["eval_every"] + 1) * cfg["eval_every"]
        t_log, s_log = time.perf_counter(), steps
        t_act = t_env = 0.0  # acting loop: forward passes / env steps + replay writes
        n_iter = 0
        print(f"run {run_dir}, device {device}, buffer {nbytes / 1e9:.2f} GB"
              + f", {cfg.get('vector_env', 'sync')} envs x{n}"
              + (", learner process" if proc else ", async learner" if bg else ""))
        if bg:
            bg.start()

        while steps < cfg["total_steps"]:
            eps = epsilon(cfg, steps)
            ids = col.ids  # all envs, or (pool) those with results in
            actions = rng.integers(n_actions, size=len(ids))
            greedy = rng.random(len(ids)) >= eps
            t_a = time.perf_counter()
            if greedy.any():
                g = ids[greedy]
                q = actor.q_values(col.cur[g], col.obs["vec"][g])
                actions[greedy] = q.argmax(1)
            t_b = time.perf_counter()
            episodes += col.step(actions, steps)
            t_act += t_b - t_a
            t_env += time.perf_counter() - t_b
            n_iter += 1
            steps += col.last_n
            if conv_frozen and steps >= freeze_steps:
                conv_frozen = False
                state_src.request_conv_trainable(True)
                print(f"{steps:>9} steps  convolutions unfrozen", flush=True)
            spec = start_level_at(cfg, steps)
            if spec != level_spec:  # curriculum: from the envs' next reset on
                level_spec = spec
                col.set_start_level(spec)
                writer.add_text("train/start_level", str(spec), steps)
                print(f"{steps:>9} steps  start_level {spec}", flush=True)

            # after a resume without a saved buffer, refill it first
            if steps >= cfg["learning_starts"] and fill() >= min_buffer:
                beta = per_beta(cfg, steps) if per else None
                if bg:
                    if dq_sched:
                        bg.set_dqfd_schedule(*dqfd_schedule(cfg, steps))
                    bg.give(col.last_n / cfg["train_every"], beta)
                else:
                    update_credit += col.last_n / cfg["train_every"]
                while not bg and update_credit >= 1:
                    batch, idx, weights = buffer.sample(rng, cfg["batch_size"], beta)
                    loss, mq, td = learner.update(batch, weights)
                    if per:
                        buffer.update_priorities(idx, td.cpu().numpy())
                    losses.append(loss)
                    qs.append(mq)
                    n_updates += 1
                    update_credit -= 1
            if steps >= next_target:
                if bg:
                    bg.request_target_sync()
                else:
                    learner.sync_target()
                next_target += cfg["target_update"]

            if steps >= next_log:
                now = time.perf_counter()
                sps = (steps - s_log) / (now - t_log)
                writer.add_scalar("train/epsilon", eps, steps)
                writer.add_scalar("perf/steps_per_sec", sps, steps)
                writer.add_scalar("train/buffer_size", fill(), steps)
                if per:
                    writer.add_scalar("train/per_beta", per_beta(cfg, steps), steps)
                msg = f"{steps:>9} steps  eps {eps:.3f}  {sps:,.0f} steps/s  episodes {episodes}"
                if col.track is not None:
                    writer.add_scalar("train/tracked_cell_steps", col.tracked, steps)
                    msg += f"  pocket {col.tracked}"
                    col.tracked = 0
                if col.prefixed:
                    msg += f"  prefixed {col.prefixed}"
                    col.prefixed = 0
                if col.ended:
                    msg += f"  hunger {col.hungry}/{col.ended}"
                    col.ended = col.hungry = 0
                if proc:
                    wait, busy = bg.perf()
                    writer.add_scalar("perf/actor_wait", wait, steps)
                    writer.add_scalar("perf/learner_busy", busy, steps)
                    msg += (f"  wait {wait:.0%} busy {busy:.0%}  q {t_act / n_iter * 1e3:.2f}"
                            f" env {t_env / n_iter * 1e3:.2f} ms x{(steps - s_log) / n_iter:.1f}")
                t_act = t_env = 0.0
                n_iter = 0
                if bg:
                    losses, qs, n_updates = bg.take_stats()
                if losses:
                    loss_m = torch.stack(losses).mean().item()
                    q_m = torch.stack(qs).mean().item()
                    writer.add_scalar("train/loss", loss_m, steps)
                    writer.add_scalar("train/mean_q", q_m, steps)
                    msg += f"  loss {loss_m:.4f}  Q {q_m:.2f}  updates {n_updates}"
                    if dq.get("enabled"):
                        writer.add_scalar("train/margin_loss", bg.last_margin_loss, steps)
                        writer.add_scalar("train/demo_frac", bg.last_demo_frac, steps)
                        msg += f"  margin {bg.last_margin_loss:.4f}  demo {bg.last_demo_frac:.1%}"
                        if dq_sched:
                            lam = dqfd_schedule(cfg, steps)[1]
                            if lam is not None:
                                writer.add_scalar("train/lambda_margin", lam, steps)
                                msg += f"  lam_m {lam:.2f}"
                    losses, qs = [], []
                print(msg, flush=True)
                next_log += cfg["log_every"]
                t_log, s_log = time.perf_counter(), steps

            if steps >= next_eval or steps >= cfg["total_steps"]:
                if bg:  # a quiet learner while evaluating and saving
                    bg.drain()
                    n_updates = bg.n_updates
                summary, metric = run_eval()
                state = save_state({"best": max(best, metric),
                                    "pretrained": bool(dq.get("enabled"))})
                if metric > best:
                    best = metric
                    save_atomic({**state, "eval": summary}, os.path.join(run_dir, "best.pt"),
                                torch.save)
                save_atomic(state, ck_path, torch.save)
                if cfg["checkpoint_buffer"] and buffer is not None:
                    save_atomic(buffer.state_dict(), buf_path, save_npz)
                next_eval += cfg["eval_every"]
                t_log, s_log = time.perf_counter(), steps  # eval time is not training speed
    finally:
        if bg:
            bg.stop()
        if envs is not None:
            envs.close()
        for e in eval_envs:
            e.close()
        writer.close()
        if replay is not None:
            replay.close()


if __name__ == "__main__":
    main()
