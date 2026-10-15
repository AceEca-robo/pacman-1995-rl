"""ctypes wrapper of sim/_csim.c: the C port, same API as PacmanSim, plus
SimBatch (many sims in one numpy buffer, stepped and observed in C).

The shared library is (re)built with gcc next to the source when missing or
older than _csim.c.
"""

import ctypes
import json
import os
import subprocess

import numpy as np

from .pacman_sim import BOARD_CELLS, HEIGHT, LEVELS, WIDTH, game_seed

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "_csim.c")
LIB = os.path.join(HERE, "_csim.so")
NCH = 21
CELLS = WIDTH * HEIGHT
INFO = ("lives", "level", "cell", "score", "eats", "dir")  # sim_info_many columns


def _load():
    if not os.path.exists(LIB) or os.path.getmtime(LIB) < os.path.getmtime(SRC):
        tmp = f"{LIB}.{os.getpid()}.tmp"
        subprocess.run(["gcc", "-O2", "-shared", "-fPIC", "-o", tmp, SRC], check=True)
        os.replace(tmp, LIB)
    lib = ctypes.CDLL(LIB)
    p, i, u = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32
    lib.sim_size.restype = i
    lib.sim_set_board.argtypes = [i, ctypes.c_char_p]
    lib.sim_reset.argtypes = [p, u, i]
    lib.sim_advance.argtypes = [p, i]
    lib.sim_advance_many.argtypes = [p, i, p]
    lib.sim_json.argtypes = [p, ctypes.c_char_p]
    lib.sim_json.restype = i
    lib.sim_obs.argtypes = [p, p, p, i]
    lib.sim_obs_many.argtypes = [p, i, p, p, i]
    lib.sim_info_many.argtypes = [p, i, p]
    for k in range(LEVELS):
        lib.sim_set_board(k, BOARD_CELLS[k])
    return lib


_lib = _load()
SIZE = _lib.sim_size()
_buf = ctypes.create_string_buffer(8192)


def _addr(a):
    return a.ctypes.data


class CSim:
    """One sim; the same API as sim.pacman_sim.PacmanSim."""

    def __init__(self, seed=None, level=1):
        self.buf = np.zeros(SIZE, np.uint8)
        if seed is not None:
            self.reset(seed, level)

    def reset(self, seed=0, level=1, raw_seed=None):
        _lib.sim_reset(_addr(self.buf), game_seed(seed) if raw_seed is None else raw_seed, level)
        return self.state()

    def advance(self, action):
        _lib.sim_advance(_addr(self.buf), int(action))

    def step(self, action):
        self.advance(action)
        return self.state()

    def clone(self):
        c = CSim.__new__(CSim)
        c.buf = self.buf.copy()
        return c

    def state_json(self):
        n = _lib.sim_json(_addr(self.buf), _buf)
        return _buf.raw[:n]

    def state(self):
        return json.loads(self.state_json())

    @property
    def lives(self):
        return int(self.info()[0])

    @property
    def done(self):
        return self.lives == 0

    def info(self):
        """int32 (6,), columns INFO: lives, level, pacman cell y * 33 + x,
        score, food left, pacman's direction (0..3 U/D/L/R, 4/5 none/still)."""
        out = np.zeros(len(INFO), np.int32)
        _lib.sim_info_many(_addr(self.buf), 1, _addr(out))
        return out

    def obs(self, cell_safety=True):
        """PacmanEnv's observation of the current state (obs.cell_safety on
        or off; the other optional planes are not ported)."""
        grid = np.empty((NCH + bool(cell_safety), HEIGHT, WIDTH), np.float32)
        vec = np.empty(3, np.float32)
        _lib.sim_obs(_addr(self.buf), _addr(grid), _addr(vec), int(bool(cell_safety)))
        return {"grid": grid, "vec": vec}


class SimBatch:
    """n sims in one (n, SIZE) uint8 array; rows copy like any numpy rows."""

    def __init__(self, n):
        self.buf = np.zeros((n, SIZE), np.uint8)

    def __len__(self):
        return len(self.buf)

    def set(self, i, sim):
        self.buf[i] = sim.buf

    def get(self, i):
        c = CSim.__new__(CSim)
        c.buf = self.buf[i].copy()
        return c

    def advance(self, actions):
        a = np.ascontiguousarray(actions, np.int32)
        assert len(a) == len(self.buf)
        _lib.sim_advance_many(_addr(self.buf), len(a), _addr(a))

    def info(self):
        """int32 (n, 6), columns INFO (see CSim.info)."""
        out = np.zeros((len(self.buf), len(INFO)), np.int32)
        _lib.sim_info_many(_addr(self.buf), len(self.buf), _addr(out))
        return out

    def obs(self, grid=None, vec=None, cell_safety=True):
        n = len(self.buf)
        nch = NCH + bool(cell_safety)
        if grid is None:
            grid = np.empty((n, nch, HEIGHT, WIDTH), np.float32)
            vec = np.empty((n, 3), np.float32)
        _lib.sim_obs_many(_addr(self.buf), n, _addr(grid), _addr(vec), int(bool(cell_safety)))
        return grid, vec
