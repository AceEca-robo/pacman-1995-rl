"""Envs in worker processes without a step barrier (learner_process runs,
cfg vector_env: pool).

gymnasium's AsyncVectorEnv waits for all envs at every step, so one env
restarting its game (~13 ms) or replaying an endgame prefix (~24 ms) stalls
all of them (a step is ~0.06 ms). Here each env steps on its own: the actor
sends actions to the envs that are ready and takes the results of whichever
envs finish next (at least min_ready of them, or whatever is ready after
max_wait seconds). Observations are bit-packed in the workers
(agents/shared_replay.py PackedObs) and come back through shared memory;
rewards, flags and the small info dict through each worker's pipe. Same-step
autoreset as the vector envs: after an episode ends the worker resets at
once and returns the last observation as final_obs.
"""

import multiprocessing as mp
import traceback
from multiprocessing import shared_memory
from multiprocessing.connection import wait

import numpy as np

from agents.dqn import PACKED


def _worker(i, make_env, conn, names, planes, vec_dim, n):
    shms = [shared_memory.SharedMemory(name=nm) for nm in names]
    grid = np.ndarray((n, planes, PACKED), np.uint8, buffer=shms[0].buf)
    vec = np.ndarray((n, vec_dim), np.float32, buffer=shms[1].buf)
    env = None
    try:
        env = make_env()
        while True:
            cmd, arg = conn.recv()
            if cmd == "step":
                obs, rew, term, trunc, info = env.step(arg)
                final = None
                if term or trunc:
                    final = (obs["grid"], obs["vec"], info)
                    obs, info = env.reset()
                grid[i], vec[i] = obs["grid"], obs["vec"]
                conn.send((float(rew), bool(term), bool(trunc), info, final))
            elif cmd == "reset":
                obs, info = env.reset(**arg)
                grid[i], vec[i] = obs["grid"], obs["vec"]
                conn.send(info)
            elif cmd == "call":
                name, args = arg
                conn.send(getattr(env.unwrapped, name)(*args))
            elif cmd == "call_async":  # no answer: may run between steps
                name, args = arg
                getattr(env.unwrapped, name)(*args)
            elif cmd == "close":
                break
    except (KeyboardInterrupt, EOFError):
        pass
    except BaseException:  # noqa: BLE001 -- handed to the main process
        conn.send(("__error__", traceback.format_exc()))
    finally:
        if env is not None:
            env.close()
        for s in shms:
            s.close()


class EnvPool:
    def __init__(self, make_env, n, planes, vec_dim, n_actions, min_ready=None, max_wait=0.002):
        self.n, self.n_actions = n, n_actions
        self.min_ready = min_ready or max(1, n // 2)
        self.max_wait = max_wait
        from agents.shared_replay import create_shm
        self._shm = [create_shm(n * planes * PACKED), create_shm(n * vec_dim * 4)]
        self.grid = np.ndarray((n, planes, PACKED), np.uint8, buffer=self._shm[0].buf)
        self.vec = np.ndarray((n, vec_dim), np.float32, buffer=self._shm[1].buf)
        ctx = mp.get_context("fork")
        self.conns, self.procs = [], []
        names = [s.name for s in self._shm]
        for i in range(n):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(i, make_env, child, names, planes, vec_dim, n),
                            daemon=True, name=f"env{i}")
            p.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(p)
        self.busy = np.zeros(n, bool)
        self._by_conn = {c: i for i, c in enumerate(self.conns)}

    def _recv(self, i):
        msg = self.conns[i].recv()
        if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "__error__":
            raise RuntimeError(f"env worker {i} failed:\n{msg[1]}")
        return msg

    def reset(self, seed=None):
        """Reset every env (seed + i); -> (grids, vecs, infos)."""
        for i, c in enumerate(self.conns):
            c.send(("reset", {"seed": None if seed is None else seed + i}))
        infos = [self._recv(i) for i in range(self.n)]
        return self.grid.copy(), self.vec.copy(), infos

    def send(self, ids, actions):
        for i, a in zip(ids, actions):
            self.conns[i].send(("step", int(a)))
        self.busy[ids] = True

    def recv(self):
        """Wait for the next results: at least min_ready envs (fewer if fewer
        are busy, or after max_wait once one is in). -> (ids, results);
        each result is (reward, terminated, truncated, info, final) with
        final = (grid, vec, info) of the ended episode or None."""
        busy = [self.conns[i] for i in np.flatnonzero(self.busy)]
        want = min(self.min_ready, len(busy))
        ready = wait(busy)
        if len(ready) < want:
            more = wait([c for c in busy if c not in ready], timeout=self.max_wait)
            ready += more
            while len(ready) < want and more:
                more = wait([c for c in busy if c not in ready], timeout=self.max_wait)
                ready += more
        ids = np.array(sorted(self._by_conn[c] for c in ready))
        res = [self._recv(i) for i in ids]
        self.busy[ids] = False
        return ids, res

    def call(self, name, *args):
        """env.unwrapped.name(*args) in every env (no step may be pending)."""
        assert not self.busy.any()
        for c in self.conns:
            c.send(("call", (name, args)))
        return [self._recv(i) for i in range(self.n)]

    def call_async(self, name, *args):
        """env.unwrapped.name(*args) in every env, after its pending step; no result."""
        for c in self.conns:
            c.send(("call_async", (name, args)))

    def drain(self):
        """Collect (and drop) the results of all pending steps."""
        while self.busy.any():
            self.recv()

    def close(self):
        for i, c in enumerate(self.conns):
            try:
                if self.busy[i]:
                    c.recv()
                c.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.kill()
                p.join()
        for c in self.conns:
            c.close()
        for s in self._shm:
            s.close()
            s.unlink()
        self._shm = []
