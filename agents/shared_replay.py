"""Replay in shared memory for the learner process (scripts/train.py
learner_process: true; agents/learner_proc.py).

Frame storage: every observation is stored once (bit-packed as in
agents/dqn.py). The buffer is split into streams: one ring per training env
(agent streams) and, for DQfD, demo streams that are written once and never
overwritten. An env writes its frames into its stream in order; the slot of
s_t also holds the action a_t, the 1-step reward / discount and the n-step
return / discount / offset k (<= n) of the bootstrap frame s_{t+k}, which is
k slots further in the same stream. An episode's last observation gets a
frame of its own (valid 0: not a transition).

One writer per field across the two processes:
- the actor (StreamWriter, main process) writes frames and transition
  fields, then publishes per stream `written` (frames written) and `final`
  (frames whose fields are final);
- the learner (Sampler, learner process) owns the PER sum tree: it registers
  newly final slots (priority = the current max priority, 0 for a frame that
  is not a transition) and zeroes the priority of the `guard` slots ahead of
  the writer before the writer may overwrite them (published as `cleared`;
  the writer waits for it). A transition only needs frames written after
  it, so overwriting the slot of count c only invalidates the transition in
  that slot.
Counters are int64 stores (atomic on x86-64, and stores are not reordered
there), written after the data they publish.
"""

import os
import time
from collections import deque
from multiprocessing import shared_memory

import gymnasium as gym
import numpy as np

from agents.dqn import PACKED, SumTree, pack_grid, pack_grids

SHM_PREFIX = "pacrl_"  # segment names: pacrl_<creator pid>_<n>
_shm_count = 0


def create_shm(nbytes):
    """A new shared memory segment named after this process, so that one left
    behind by a killed run can be recognized (remove_stale_shm)."""
    global _shm_count
    _shm_count += 1
    return shared_memory.SharedMemory(
        name=f"{SHM_PREFIX}{os.getpid()}_{_shm_count}", create=True, size=max(1, nbytes))


def remove_stale_shm(path="/dev/shm"):
    """Unlink segments whose creating process is gone (a run killed before
    its cleanup). Returns the names removed."""
    gone = []
    try:
        names = os.listdir(path)
    except FileNotFoundError:
        return gone
    for name in names:
        if not name.startswith(SHM_PREFIX):
            continue
        try:
            pid = int(name[len(SHM_PREFIX):].split("_")[0])
            os.kill(pid, 0)
        except ProcessLookupError:
            try:
                os.unlink(os.path.join(path, name))
                gone.append(name)
            except FileNotFoundError:
                pass
        except (ValueError, PermissionError):
            pass
    return gone


FIELDS = {  # name: (per-slot shape suffix, dtype); obs / vec shapes are added
    "act": ((), np.uint8),
    "rew1": ((), np.float32),
    "disc1": ((), np.float32),  # gamma, 0 if terminated after this step
    "ret": ((), np.float32),    # n-step return
    "discn": ((), np.float32),  # gamma^k, 0 if terminated within the k steps
    "offn": ((), np.uint8),     # k: slots to the n-step bootstrap frame
    "valid": ((), np.uint8),    # 1: a final transition
}


class SharedReplay:
    """The shared arrays. create() in the main process, attach(spec) in the
    learner process; the creator unlinks the segments in close()."""

    def __init__(self, spec, create):
        self.spec, self._owner, self._shm = spec, create, []
        self.planes, self.vec_dim = spec["planes"], spec["vec_dim"]
        self.sizes = np.array(spec["sizes"], np.int64)
        self.bases = np.concatenate([[0], np.cumsum(self.sizes)[:-1]]).astype(np.int64)
        self.n_demo = spec["n_demo"]  # the first n_demo streams are demo streams
        self.demo_frames = int(self.sizes[:self.n_demo].sum())
        self.frames = int(self.sizes.sum())
        shapes = {"obs": ((self.frames, self.planes, PACKED), np.uint8),
                  "vec": ((self.frames, self.vec_dim), np.float32),
                  **{k: ((self.frames,) + s, d) for k, (s, d) in FIELDS.items()},
                  "written": ((len(self.sizes),), np.int64),
                  "final": ((len(self.sizes),), np.int64),
                  "cleared": ((len(self.sizes),), np.int64)}
        names = spec.setdefault("names", {})
        try:
            for k, (shape, dtype) in shapes.items():
                nbytes = max(1, int(np.prod(shape)) * np.dtype(dtype).itemsize)
                if create:
                    shm = create_shm(nbytes)
                    names[k] = shm.name
                else:
                    shm = shared_memory.SharedMemory(name=names[k])
                self._shm.append(shm)
                setattr(self, k, np.ndarray(shape, dtype, buffer=shm.buf))
        except BaseException:
            self.close()
            raise
        if create:
            self.written[:] = self.final[:] = 0
            # demo streams are never overwritten; agent rings: the first lap is free
            self.cleared[:] = self.sizes
            self.valid[:] = 0

    @classmethod
    def create(cls, agent_sizes, demo_sizes=(), planes=35, vec_dim=3):
        spec = {"sizes": [int(s) for s in list(demo_sizes) + list(agent_sizes)],
                "n_demo": len(demo_sizes), "planes": planes, "vec_dim": vec_dim}
        return cls(spec, create=True)

    @classmethod
    def attach(cls, spec):
        return cls(spec, create=False)

    @property
    def nbytes(self):
        return sum(s.size for s in self._shm)

    def close(self):
        for k in ["obs", "vec", "written", "final", "cleared", *FIELDS]:
            self.__dict__.pop(k, None)  # drop the views before closing the buffers
        for shm in self._shm:
            shm.close()
            if self._owner:
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
        self._shm = []

    def slot(self, stream, count):
        return self.bases[stream] + count % self.sizes[stream]

    # --- demo region on disk ---------------------------------------------

    DEMO_ARRAYS = ("obs", "vec", *FIELDS)

    def save_demo(self, path):
        """The demo streams (arrays and counters) to an .npz file."""
        d = {k: getattr(self, k)[:self.demo_frames] for k in self.DEMO_ARRAYS}
        d.update(sizes=self.sizes[:self.n_demo], written=self.written[:self.n_demo],
                 final=self.final[:self.n_demo])
        tmp = path + ".tmp"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **d)  # the bit planes are mostly zeros
        os.replace(tmp, path)

    def load_demo(self, path):
        with np.load(path) as z:
            if not np.array_equal(z["sizes"], self.sizes[:self.n_demo]):
                raise ValueError(f"{path}: demo streams {z['sizes']} != {self.sizes[:self.n_demo]}")
            for k in self.DEMO_ARRAYS:
                getattr(self, k)[:self.demo_frames] = z[k]
            self.written[:self.n_demo] = z["written"]
            self.final[:self.n_demo] = z["final"]

    def demo_transitions(self):
        return int(self.valid[:self.demo_frames].sum())


class StreamWriter:
    """Actor side: turns each env's steps into frames and n-step transitions
    in its stream (the same transitions as agents/dqn.py NStep)."""

    def __init__(self, replay, streams, n, gamma, timeout=600.0):
        self.r, self.streams = replay, np.asarray(streams, np.int64)
        self.n, self.gamma, self.timeout = n, gamma, timeout
        self.count = self.r.written[self.streams].copy()  # next frame's count per env
        self.pend = [deque() for _ in self.streams]  # (count, reward, terminal) waiting for n steps
        self.cur = None  # count of each env's current frame
        self.size = 0  # transitions finalized by this writer

    def _wait_cleared(self, need):
        """need: per env, the largest count about to be written."""
        s = self.streams
        if (need < self.r.cleared[s]).all():
            return
        if self.r.n_demo and (s < self.r.n_demo).any() and \
                (need >= self.r.cleared[s])[s < self.r.n_demo].any():
            raise RuntimeError("a demo stream is full")
        t0 = time.monotonic()
        while not (need < self.r.cleared[s]).all():
            if time.monotonic() - t0 > self.timeout:
                raise RuntimeError("the learner does not clear replay slots")
            time.sleep(0.0002)

    def _write_frames(self, envs, packed, vecs):
        """One new frame per listed env (valid 0); returns their counts."""
        counts = self.count[envs].copy()
        slots = self.r.bases[self.streams[envs]] + counts % self.r.sizes[self.streams[envs]]
        self.r.valid[slots] = 0
        self.r.obs[slots] = packed
        self.r.vec[slots] = vecs
        self.count[envs] += 1
        return counts, slots

    def begin(self, packed, vecs):
        """The envs' first observations (after reset)."""
        envs = np.arange(len(self.streams))
        self._wait_cleared(self.count.copy())
        self.cur, _ = self._write_frames(envs, packed, vecs)
        self._publish()

    def room(self, margin=2):
        """Per env: False once its demo (non-ring) stream has less than margin
        free frames."""
        s = self.streams
        return ~((self.count + margin > self.r.cleared[s]) & (s < self.r.n_demo))

    def full(self, margin=2):
        """True when any env's demo stream is full (room)."""
        return not self.room(margin).all()

    def step(self, actions, rewards, next_packed, next_vecs, term, trunc, final_packed,
             final_vecs, ids=None):
        """One step of the envs `ids` (default: all, in order). next_*: the
        observations after the step (the next episode's first one where the
        episode ended); final_*: the last observations of the ended episodes
        (rows of the ended envs, in ids order). Returns the number of
        transitions finalized."""
        r, g = self.r, self.gamma
        ids = np.arange(len(self.streams)) if ids is None else np.asarray(ids)
        term, trunc = np.asarray(term, bool), np.asarray(trunc, bool)
        ended = term | trunc
        # the counts written now: one frame per env, two where an episode ended
        need = self.count.copy()
        need[ids] += ended
        self._wait_cleared(need)
        streams = self.streams[ids]
        cur_slots = r.bases[streams] + self.cur[ids] % r.sizes[streams]
        r.act[cur_slots] = actions
        r.rew1[cur_slots] = rewards
        r.disc1[cur_slots] = np.where(term, 0.0, g)
        if ended.any():
            self._write_frames(ids[ended], final_packed, final_vecs)
        new_counts, _ = self._write_frames(ids, next_packed, next_vecs)
        done = 0
        for j, i in enumerate(ids):
            pend = self.pend[i]
            pend.append((self.cur[i], float(rewards[j])))
            if ended[j]:
                while pend:
                    done += self._finalize(i, pend, bool(term[j]))
            elif len(pend) == self.n:
                done += self._finalize(i, pend, False)
        self.cur[ids] = new_counts
        self.size += done
        self._publish()
        return done

    def _finalize(self, i, pend, terminated):
        """The oldest pending transition of env i: n-step return over the
        pending rewards; the bootstrap frame follows the last of them."""
        g = self.gamma
        ret = sum(g ** j * p[1] for j, p in enumerate(pend))
        k = len(pend)
        c, _ = pend.popleft()
        s = self.streams[i]
        slot = self.r.bases[s] + c % self.r.sizes[s]
        self.r.ret[slot] = ret
        self.r.discn[slot] = 0.0 if terminated else g ** k
        self.r.offn[slot] = k
        self.r.valid[slot] = 1
        return 1

    def _publish(self):
        s = self.streams
        first_pending = np.array([p[0][0] if p else c for p, c in zip(self.pend, self.cur)])
        self.r.written[s] = self.count
        self.r.final[s] = first_pending


class Sampler:
    """Learner side: PER over the final transitions of all streams
    (proportional, as agents/dqn.py PrioritizedReplayBuffer; alpha 0 =
    uniform). Demo transitions get eps_demo on top of |TD| + eps."""

    def __init__(self, replay, alpha, eps, eps_demo=0.0, guard=1024, n=3):
        self.r = replay
        self.alpha, self.eps, self.eps_demo = alpha, eps, eps_demo
        if (guard + n + 1 > self.r.sizes[self.r.n_demo:]).any():
            raise ValueError("agent streams must be longer than guard + n_step")
        self.guard = guard
        self.tree = SumTree(self.r.frames)
        self.max_priority = 1.0
        self.registered = np.zeros(len(self.r.sizes), np.int64)
        self.n_valid = 0
        for s in range(self.r.n_demo, len(self.r.sizes)):  # agent rings: guard ahead of the writer
            self.r.cleared[s] = self.r.written[s] + guard

    def register(self):
        """New final slots in, slots ahead of the writers out (one tree
        update for all streams). Returns the number of transitions added."""
        r = self.r
        final, written = r.final.copy(), r.written.copy()
        new = np.flatnonzero(final > self.registered)
        # agent rings: slots of counts [cleared, written + guard) still hold count - size
        target = written + self.guard
        clear = np.flatnonzero((target > r.cleared) & (np.arange(len(r.sizes)) >= r.n_demo))
        if not len(new) and not len(clear):
            return 0
        reg = [r.bases[s] + np.arange(self.registered[s], final[s]) % r.sizes[s] for s in new]
        old = [r.bases[s] + np.arange(max(r.cleared[s], r.sizes[s]), target[s]) % r.sizes[s]
               for s in clear if target[s] > r.sizes[s]]
        added = 0
        if old:
            old = np.concatenate(old)
            self.n_valid -= int((self.tree.get(old) > 0).sum())
            self.tree.set(old, 0.0)
        if reg:
            reg = np.concatenate(reg)
            v = r.valid[reg].astype(bool)
            added = int(v.sum())
            self.n_valid += added - int((self.tree.get(reg) > 0).sum())
            self.tree.set(reg, np.where(v, self.max_priority, 0.0))
        self.registered[new] = final[new]
        r.cleared[clear] = target[clear]
        return added

    def sample(self, rng, batch, beta):
        """-> (batch dict of numpy arrays, slots, importance weights)."""
        total = self.tree.total
        prefix = (np.arange(batch) + rng.random(batch)) * (total / batch)  # stratified
        idx = np.minimum(self.tree.find(prefix), self.r.frames - 1)
        p = self.tree.get(idx)
        bad = p <= 0  # float rounding at a boundary: take another sample's slot
        if bad.any():
            good = np.flatnonzero(~bad)
            idx[bad] = idx[good[rng.integers(len(good), size=int(bad.sum()))]]
            p = self.tree.get(idx)
        w = (self.n_valid * p / self.tree.total) ** -beta
        return self.gather(idx), idx, (w / w.max()).astype(np.float32)

    def gather(self, idx):
        r = self.r
        s = np.searchsorted(r.bases, idx, side="right") - 1
        base, size = r.bases[s], r.sizes[s]
        local = idx - base
        nxt1 = base + (local + 1) % size
        nxtn = base + (local + r.offn[idx]) % size
        return {"obs": r.obs[idx], "vec": r.vec[idx], "act": r.act[idx],
                "obs1": r.obs[nxt1], "vec1": r.vec[nxt1], "rew1": r.rew1[idx],
                "disc1": r.disc1[idx], "obsn": r.obs[nxtn], "vecn": r.vec[nxtn],
                "ret": r.ret[idx], "discn": r.discn[idx],
                "demo": idx < r.demo_frames}

    def update_priorities(self, idx, td_abs):
        """Skips slots zeroed since they were sampled (overwritten by now)."""
        live = self.tree.get(idx) > 0
        if not live.any():
            return
        idx, td = np.asarray(idx)[live], np.asarray(td_abs, np.float64)[live]
        demo = idx < self.r.demo_frames
        pr = (td + self.eps + self.eps_demo * demo) ** self.alpha
        self.tree.set(idx, pr)
        self.max_priority = max(self.max_priority, float(pr.max()))


# --- env side ----------------------------------------------------------------

INFO_KEYS = ("level", "score", "hunger", "prefix")


class PackedObs(gym.Wrapper):
    """Runs inside the vector env's worker processes: the grid comes back
    bit-packed (agents/dqn.py pack_grid; 3.3 kB instead of 64 kB through the
    shared memory) and info keeps only what the training loop reads."""

    def __init__(self, env, planes):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict({
            "grid": gym.spaces.Box(0, 255, (planes, PACKED), np.uint8),
            "vec": env.observation_space["vec"]})

    @staticmethod
    def _info(info):
        return {k: info[k] for k in INFO_KEYS if k in info}

    def _obs(self, obs):
        return {"grid": pack_grid(obs["grid"]), "vec": obs["vec"]}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._obs(obs), self._info(info)

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        return self._obs(obs), rew, term, trunc, self._info(info)


def make_packed_env(env_config, planes):
    from env.pacman_env import PacmanEnv
    return PackedObs(PacmanEnv(config=env_config), planes)


__all__ = ["SharedReplay", "StreamWriter", "Sampler", "PackedObs", "make_packed_env",
           "pack_grids"]
