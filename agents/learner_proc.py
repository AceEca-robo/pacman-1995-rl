"""The learner in its own process (scripts/train.py learner_process: true).

The main process acts (vector envs, n-step frames into the shared replay,
agents/shared_replay.py) with its own copy of the online network (Actor);
the learner process samples, updates and publishes the online weights into
shared memory every weight_sync_every updates. Hot-path coordination goes
through a small shared float64 array (one writer per field):

- the actor hands out update credit (env steps / train_every, as the
  synchronous loop) and waits while more than async_max_lag updates are
  pending, so the replay ratio stays that of train_every;
- target syncs are requested by count (target_update env steps apart) and
  run between updates, as in the learner thread.
Rare requests (state for a checkpoint, freezing the convolutions, stop) go
through a pipe. Not deterministic (process timing decides what each update
samples).
"""

import os
import time
import traceback

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from agents.dqn import DQNLearner, GridDecoder, QNetwork

# ctrl fields
CREDIT, BETA, SYNC_REQ, STOP = 0, 1, 2, 3  # actor
DONE, SYNC_DONE, FLUSHED, WVER, ERROR = 4, 5, 6, 7, 8  # learner
LOSS_SUM, Q_SUM, JE_SUM, STAT_N, BUSY_S, DEMO_SUM = 9, 10, 11, 12, 13, 14  # learner, running sums
DEMO_FRAC, LAM_M = 15, 16  # actor: DQfD schedules (-1 = off: PER share, config weight)
N_CTRL = 18


class Actor:
    """The acting copy of the online network (q_values as DQNLearner)."""

    def __init__(self, cfg, device, n_actions, n_channels, vec_dim):
        self.device = device
        if str(device).startswith("cuda"):
            # as DQNLearner (whose q_values acted in the other modes)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.net = QNetwork(cfg["network"], n_actions, vec_dim, n_channels).to(device).eval()
        self.net.requires_grad_(False)
        self.decode = GridDecoder(device, n_channels)
        self.n_params = sum(p.numel() for p in self.net.parameters())

    def _t(self, x):
        return torch.as_tensor(x, device=self.device)

    @torch.no_grad()
    def q_values(self, packed, vecs):
        return self.net(self.decode(self._t(packed)), self._t(vecs)).cpu().numpy()


def _attach_shm(name, shape, dtype):
    from multiprocessing import shared_memory
    shm = shared_memory.SharedMemory(name=name)
    return shm, np.ndarray(shape, dtype, buffer=shm.buf)


class LearnerProcess:
    """Main-process side; the interface of train.py's BackgroundLearner
    (give, request_target_sync, drain, take_stats, n_updates, check, stop)
    plus state_dict, request_conv_trainable and pull_weights."""

    def __init__(self, cfg, device, dims, replay, init, seed, n_updates=0):
        from agents.shared_replay import create_shm
        self.cfg, self.replay = cfg, replay
        self.max_lag = cfg.get("async_max_lag") or 4
        self.actor = Actor(cfg, device, *dims)
        self._shm = [create_shm(N_CTRL * 8), create_shm(self.actor.n_params * 4)]
        self.ctrl = np.ndarray(N_CTRL, np.float64, buffer=self._shm[0].buf)
        self.ctrl[:] = 0
        self.ctrl[DEMO_FRAC] = self.ctrl[LAM_M] = -1.0  # schedules off
        self.ctrl[DONE] = self.ctrl[FLUSHED] = self.ctrl[CREDIT] = n_updates
        self.weights = np.ndarray(self.actor.n_params, np.float32, buffer=self._shm[1].buf)
        self._wver = -1
        self._stats = self.ctrl.copy()
        self.last_margin_loss = 0.0
        self.last_demo_frac = 0.0  # share of demo samples in the batches
        self.wait_s = 0.0  # time the actor waited for the learner (give)
        self._perf = (time.perf_counter(), 0.0, 0.0)
        init = _to_cpu(init)  # no CUDA tensors through the process boundary
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(
            target=learner_main, daemon=True, name="learner",
            args=(cfg, device, dims, replay.spec, self._shm[0].name, self._shm[1].name,
                  self.actor.n_params, child, init, seed))
        self.proc.start()
        child.close()
        msg = self._recv(timeout=600)
        assert msg == "ready", msg
        self.pull_weights(force=True)

    # --- hot path ----------------------------------------------------------

    @property
    def n_updates(self):
        return int(self.ctrl[DONE])

    def start(self):  # BackgroundLearner compatibility: the process runs from __init__
        pass

    def check(self):
        if self.ctrl[ERROR]:
            msg = self.conn.recv() if self.conn.poll(5) else ("error", "(no message)")
            raise RuntimeError(f"learner process failed:\n{msg[1]}")
        if not self.proc.is_alive():
            raise RuntimeError(f"learner process died (exit code {self.proc.exitcode})")

    def give(self, credit, beta):
        if beta is not None:
            self.ctrl[BETA] = beta
        self.ctrl[CREDIT] += credit
        if self.ctrl[CREDIT] - self.ctrl[DONE] >= self.max_lag + 1:
            t0 = time.perf_counter()
            self._wait(lambda: self.ctrl[CREDIT] - self.ctrl[DONE] < self.max_lag + 1)
            self.wait_s += time.perf_counter() - t0
        self.pull_weights()

    def set_dqfd_schedule(self, demo_frac, lam_m):
        """Share of demo samples per batch and margin weight for the next
        updates (None: PER decides the share / the config's weight)."""
        self.ctrl[DEMO_FRAC] = -1.0 if demo_frac is None else demo_frac
        self.ctrl[LAM_M] = -1.0 if lam_m is None else lam_m

    def request_target_sync(self):
        self.ctrl[SYNC_REQ] += 1

    def drain(self):
        """Wait until every whole update handed out so far is done, its
        priorities are in and target syncs are done; then the newest weights."""
        self._wait(lambda: self.ctrl[DONE] >= np.floor(self.ctrl[CREDIT])
                   and self.ctrl[FLUSHED] >= self.ctrl[DONE]
                   and self.ctrl[SYNC_DONE] >= self.ctrl[SYNC_REQ]
                   and self.ctrl[WVER] % 2 == 0)
        self._request(("publish",))
        self.pull_weights(force=True)

    def _wait(self, cond, poll=0.0002):
        n = 0
        while not cond():
            n += 1
            if n % 1000 == 0:
                self.check()
            time.sleep(poll)
        self.check()

    def take_stats(self):
        """-> (losses, mean Qs, n_updates) since the last call, as tensors
        (BackgroundLearner's format: one entry with the means)."""
        cur = self.ctrl.copy()
        d = cur - self._stats
        self._stats = cur
        if d[STAT_N] < 1:
            return [], [], int(cur[DONE])
        self.last_margin_loss = d[JE_SUM] / d[STAT_N]
        self.last_demo_frac = d[DEMO_SUM] / d[STAT_N]
        return ([torch.tensor(d[LOSS_SUM] / d[STAT_N])], [torch.tensor(d[Q_SUM] / d[STAT_N])],
                int(cur[DONE]))

    def perf(self):
        """-> (share of wall time the actor waited for the learner, share the
        learner spent sampling and updating) since the last call."""
        now, busy = time.perf_counter(), self.ctrl[BUSY_S]
        t0, w0, b0 = self._perf
        self._perf = (now, self.wait_s, busy)
        dt = max(now - t0, 1e-9)
        return (self.wait_s - w0) / dt, (busy - b0) / dt

    def pull_weights(self, force=False):
        """Copy newly published weights into the actor (seqlock: an odd
        version is being written; a version change during the copy retries)."""
        while True:
            v = self.ctrl[WVER]
            if v % 2 or (v == self._wver and not force):
                if not force:
                    return False
                time.sleep(0.0002)
                self.check()
                continue
            flat = torch.from_numpy(self.weights.copy())
            if self.ctrl[WVER] == v:
                break
        vector_to_parameters(flat.to(self.actor.device), self.actor.net.parameters())
        self._wver = v
        return True

    # --- rare requests -----------------------------------------------------

    def _recv(self, timeout):
        if not self.conn.poll(timeout):
            self.check()
            raise RuntimeError("learner process does not answer")
        msg = self.conn.recv()
        if isinstance(msg, tuple) and msg and msg[0] == "error":
            raise RuntimeError(f"learner process failed:\n{msg[1]}")
        return msg

    def _request(self, msg, timeout=600):
        self.check()
        self.conn.send(msg)
        return self._recv(timeout)

    def state_dict(self):
        """The learner's state (online, target, optimizer) on the CPU."""
        return self._request(("state",))

    def request_conv_trainable(self, flag):
        self._request(("conv", flag))

    def stop(self):
        try:
            if self.proc.is_alive():
                self.ctrl[STOP] = 1
                self.proc.join(timeout=60)
            if self.proc.is_alive():
                self.proc.kill()
                self.proc.join()
        finally:
            for shm in self._shm:
                shm.close()
                shm.unlink()
            self._shm = []


def learner_main(cfg, device, dims, replay_spec, ctrl_name, weights_name, n_params, conn,
                 init, seed):
    """Learner process: register new replay slots, update when there is
    credit, publish weights, answer requests."""
    from agents.shared_replay import Sampler, SharedReplay
    shms, replay = [], None
    ctrl = None
    try:
        torch.set_num_threads(1)
        if (cfg.get("cpu_affinity") or {}).get("learner"):
            os.sched_setaffinity(0, cfg["cpu_affinity"]["learner"])
        shm, ctrl = _attach_shm(ctrl_name, (N_CTRL,), np.float64)
        shms.append(shm)
        shm, weights = _attach_shm(weights_name, (n_params,), np.float32)
        shms.append(shm)
        replay = SharedReplay.attach(replay_spec)
        torch.manual_seed(seed)
        learner = DQNLearner(cfg, device, *dims)
        if init.get("learner"):
            learner.load_state_dict(init["learner"])
        if init.get("online"):
            learner.online.load_state_dict(init["online"])
            learner.target.load_state_dict(init["online"])
        if init.get("conv"):
            learner.online.conv.load_state_dict(init["conv"])
            learner.target.conv.load_state_dict(init["conv"])
        per = cfg["per"]
        dq = cfg.get("dqfd") or {}
        sampler = Sampler(replay, per["alpha"] if per["enabled"] else 0.0, per["eps"],
                          dq.get("eps_demo", 0.0) if dq.get("enabled") else 0.0,
                          guard=cfg.get("replay_guard") or 1024, n=cfg["n_step"])
        rng = np.random.default_rng(seed)
        cuda = str(device).startswith("cuda")
        sync_every = cfg.get("weight_sync_every") or 50

        def publish():
            ctrl[WVER] += 1
            weights[:] = parameters_to_vector(learner.online.parameters()).detach().cpu().numpy()
            ctrl[WVER] += 1

        publish()
        conn.send("ready")
        pending = None  # (slots, |TD| on the CPU, event): priorities deferred by one update

        def flush():
            nonlocal pending
            if pending is not None:
                idx, td, done = pending
                pending = None
                if done is not None:
                    done.synchronize()
                sampler.update_priorities(idx, td.numpy())
            ctrl[FLUSHED] = ctrl[DONE]

        acc, acc_n = None, 0  # loss, Q, margin loss summed on the device
        demo_acc = 0.0  # demo share of the batches, summed on the CPU

        def stats_out():
            nonlocal acc, acc_n, demo_acc
            if acc_n:
                loss_s, q_s, je_s = acc.tolist()
                ctrl[LOSS_SUM] += loss_s
                ctrl[Q_SUM] += q_s
                ctrl[JE_SUM] += je_s
                ctrl[DEMO_SUM] += demo_acc
                demo_acc = 0.0
                ctrl[STAT_N] += acc_n
                acc, acc_n = None, 0

        idle = 0
        while not ctrl[STOP]:
            if conn.poll():
                msg = conn.recv()
                if msg[0] == "state":
                    flush()
                    conn.send(_to_cpu(learner.state_dict()))
                elif msg[0] == "conv":
                    learner.request_conv_trainable(msg[1])
                    conn.send("ok")
                elif msg[0] == "publish":
                    stats_out()
                    publish()
                    conn.send("ok")
            sampler.register()
            if ctrl[SYNC_DONE] < ctrl[SYNC_REQ]:  # before the next update, as the learner thread
                learner.sync_target()
                ctrl[SYNC_DONE] = ctrl[SYNC_REQ]
            if ctrl[CREDIT] - ctrl[DONE] >= 1 and sampler.n_valid >= cfg["batch_size"]:
                idle = 0
                t_busy = time.perf_counter()
                frac, lam = float(ctrl[DEMO_FRAC]), float(ctrl[LAM_M])
                batch, idx, w = sampler.sample(rng, cfg["batch_size"], float(ctrl[BETA]) or 0.4,
                                               frac if frac >= 0 else None)
                loss, mq, td, je = learner.update_frames(batch, w, lam if lam >= 0 else None)
                if cuda:
                    td, done = td.to("cpu", non_blocking=True), torch.cuda.Event()
                    done.record()
                else:
                    td, done = td.cpu(), None
                flush()  # the previous update's priorities
                pending = (idx, td, done)
                st = torch.stack([loss, mq, je]).detach()
                acc, acc_n = (st if acc is None else acc + st), acc_n + 1
                demo_acc += float(np.mean(batch["demo"]))
                ctrl[DONE] += 1
                ctrl[BUSY_S] += time.perf_counter() - t_busy
                if int(ctrl[DONE]) % sync_every == 0:
                    stats_out()
                    publish()
            else:
                flush()
                if acc_n:
                    stats_out()
                idle += 1
                if idle > 50:
                    time.sleep(0.0002)
    except BaseException:  # noqa: BLE001 -- handed to the main process
        err = traceback.format_exc()
        if ctrl is not None:
            ctrl[ERROR] = 1
        try:
            conn.send(("error", err))
        except Exception:  # noqa: BLE001
            pass
    finally:
        if replay is not None:
            replay.close()
        for shm in shms:
            shm.close()


def _to_cpu(x):
    if torch.is_tensor(x):
        return x.cpu()
    if isinstance(x, dict):
        return {k: _to_cpu(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_cpu(v) for v in x)
    return x
