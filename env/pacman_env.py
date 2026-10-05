"""Gymnasium environment for pacman 1.0 (1995) over the game's RL bridge.

The game (game/pacman --rl <sock>) connects to a Unix socket served here,
sends one JSON state per tick and blocks for one action line. See
game/rlbridge.cc for the protocol.

Observation is a Dict:
  grid: float32 (8, 23, 33), 0/1 planes, see CHANNELS
  vec:  float32 (3,) = [min(lives, 9) / 3, supertime_left / SUPERTIME, level / LEVELS]
"""

import ctypes
import json
import os
import select
import signal
import socket
import subprocess
import tempfile
import weakref

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(ROOT, "configs", "env_default.yaml")

HEIGHT, WIDTH = 23, 33  # BOARDHEIGHT, BOARDWIDTH in game/sizes.h
SUPERTIME = 50          # game/pac.h
LEVELS = 16             # game/pac.h; boards repeat (randomly) after that
MAX_LIVES = 9           # only for the obs bound; the game has no limit
ACTIONS = [b"U\n", b"D\n", b"L\n", b"R\n", b"N\n"]
CHANNELS = ("walls", "food", "superfood", "pacman",
            "ghost_normal", "ghost_hunted", "ghost_eyes", "bonus")
GHOST_CHANNEL = {"normal": 4, "hunted": 5, "eyes": 6}


def _load_config(config):
    with open(DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    if config is None:
        return cfg
    if isinstance(config, (str, os.PathLike)):
        with open(config) as f:
            config = yaml.safe_load(f)
    for key, value in config.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key] = {**cfg[key], **value}
        else:
            cfg[key] = value
    return cfg


def _die_with_parent():
    # Children get SIGTERM if the Python process dies without close().
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except OSError:
        pass


def _stop_game(res):
    for key in ("rfile", "conn"):
        if res.get(key) is not None:
            res[key].close()
            res[key] = None
    proc = res.get("game")
    if proc is not None:
        try:
            proc.wait(timeout=5)  # the game quits on its own when the socket closes
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        res["game"] = None


def _shutdown(res):
    _stop_game(res)
    proc = res.get("xvfb")
    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        res["xvfb"] = None


class PacmanEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=None, seed=0, display=None, render_mode=None):
        """config: path to yaml, dict of overrides or None (configs/env_default.yaml).
        seed: --seed of the first game process (reset(seed=...) overrides it).
        display: X display to run the game on; None starts a private Xvfb."""
        self.cfg = _load_config(config)
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(0.0, 1.0, (len(CHANNELS), HEIGHT, WIDTH), np.float32),
            # lives can exceed 3 through bonus lives, clipped at MAX_LIVES
            "vec": spaces.Box(np.zeros(3, np.float32),
                              np.array([MAX_LIVES / 3, 1.0, 1.0], np.float32), dtype=np.float32),
        })

        self._res = {}
        self._finalizer = weakref.finalize(self, _shutdown, self._res)
        self._display = display if display is not None else self._start_xvfb()
        self._binary = os.path.join(ROOT, self.cfg["game"]["binary"])
        self._state = None   # last state received
        self._steps = 0
        self._start_game(seed)

    # --- processes -------------------------------------------------------

    def _start_xvfb(self):
        rfd, wfd = os.pipe()
        self._res["xvfb"] = subprocess.Popen(
            ["Xvfb", "-displayfd", str(wfd), "-nolisten", "tcp",
             "-screen", "0", "800x600x24"],
            pass_fds=(wfd,), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            preexec_fn=_die_with_parent)
        os.close(wfd)
        out = b""
        try:
            while not out.endswith(b"\n"):
                ready, _, _ = select.select([rfd], [], [], 10)
                chunk = os.read(rfd, 64) if ready else b""
                if not chunk:
                    raise RuntimeError("Xvfb did not start")
                out += chunk
        finally:
            os.close(rfd)
        return ":" + out.decode().strip()

    def _start_game(self, seed):
        tmp = tempfile.mkdtemp(prefix="pacman-env-")
        path = os.path.join(tmp, "game.sock")
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            srv.bind(path)
            srv.listen(1)
            srv.settimeout(self.cfg["game"]["connect_timeout"])
            self._res["game"] = subprocess.Popen(
                [self._binary, "--rl", path, "--fast", "--seed", str(seed)],
                env={**os.environ, "DISPLAY": self._display},
                stdout=subprocess.DEVNULL, preexec_fn=_die_with_parent)
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                raise RuntimeError(f"game did not connect, exit code {self._res['game'].poll()}")
        finally:
            # once connected the socket file is not needed any more
            srv.close()
            if os.path.exists(path):
                os.unlink(path)
            os.rmdir(tmp)
        conn.settimeout(None)
        self._res["conn"] = conn
        self._res["rfile"] = conn.makefile("rb")
        self._seed = seed
        self._fresh = True   # no state read from this process yet

    def _restart_game(self, seed):
        _stop_game(self._res)
        self._start_game(seed)

    def _recv(self):
        line = self._res["rfile"].readline()
        if not line:
            raise RuntimeError(f"game closed the connection, exit code {self._res['game'].poll()}")
        self._fresh = False
        return json.loads(line)

    # --- gym API ---------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            if not (self._fresh and self._seed == seed):
                self._restart_game(seed)
        elif not self._fresh and not (self._state and self._state["done"]):
            # mid-game (truncated or reset early): the game cannot be reset
            # from outside, start a new process
            self._restart_game(int(self.np_random.integers(2**31 - 1)))
        if not self._fresh:
            # after done the game starts a new one by itself; the action
            # answering the final state is ignored by the game
            self._res["conn"].sendall(ACTIONS[4])
        self._state = self._recv()
        self._steps = 0
        return self._obs(self._state), self._info(self._state)

    def step(self, action):
        if self._state is None:
            raise RuntimeError("step() before reset()")
        self._res["conn"].sendall(ACTIONS[int(action)])
        prev, st = self._state, self._recv()
        self._state = st
        self._steps += 1

        r = self.cfg["reward"]
        reward = (r["k_score"] * (st["score"] - prev["score"])
                  + r["death_penalty"] * max(0, prev["lives"] - st["lives"])
                  + r["step_penalty"]
                  + r["level_bonus"] * max(0, st["level"] - prev["level"]))
        terminated = bool(st["done"])
        limit = self.cfg["max_episode_steps"]
        truncated = not terminated and limit is not None and self._steps >= limit
        return self._obs(st), float(reward), terminated, truncated, self._info(st)

    def close(self):
        self._finalizer()

    # --- state -> obs ----------------------------------------------------

    @staticmethod
    def _obs(st):
        cells = np.frombuffer("".join(st["grid"]).encode(), np.uint8).reshape(HEIGHT, WIDTH)
        grid = np.zeros((len(CHANNELS), HEIGHT, WIDTH), np.float32)
        grid[0] = (cells == ord("#")) | (cells == ord("-"))  # the gate stops pacman too
        grid[1] = cells == ord(".")
        grid[2] = cells == ord("o")
        p = st["pacman"]
        grid[3, p["y"], p["x"]] = 1
        for g in st["ghosts"]:
            grid[GHOST_CHANNEL[g["state"]], g["y"], g["x"]] = 1
        if st["bonus"]:
            grid[7, st["bonus"]["y"], st["bonus"]["x"]] = 1
        vec = np.array([min(st["lives"], MAX_LIVES) / 3, st["supertime_left"] / SUPERTIME,
                        min(st["level"], LEVELS) / LEVELS], np.float32)
        return {"grid": grid, "vec": vec}

    @staticmethod
    def _info(st):
        return {"score": st["score"], "lives": st["lives"], "level": st["level"]}
