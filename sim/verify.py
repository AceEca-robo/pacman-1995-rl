"""Compare a sim against the real game line by line (raw JSON bytes).

GameProc runs game/pacman --rl --fast --headless with the same --seed and
--level as PacmanEnv and returns each state line as the game sent it.
"""

import os
import socket
import subprocess
import tempfile

import numpy as np

from env.pacman_env import ACTIONS, _die_with_parent
from .pacman_sim import ROOT, PacmanSim, game_seed

BINARY = os.path.join(ROOT, "game", "pacman")


class GameProc:
    def __init__(self, seed, level=1, binary=BINARY, timeout=10.0):
        """seed: env seed (the game gets --seed game_seed(seed))."""
        tmp = tempfile.mkdtemp(prefix="pacman-sim-")
        path = os.path.join(tmp, "game.sock")
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            srv.bind(path)
            srv.listen(1)
            srv.settimeout(timeout)
            cmd = [binary, "--rl", path, "--seed", str(game_seed(seed)), "--fast", "--headless"]
            if level != 1:
                cmd += ["--level", str(level)]
            env = dict(os.environ)
            env.pop("DISPLAY", None)
            self.proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL,
                                         preexec_fn=_die_with_parent)
            self.conn, _ = srv.accept()
        finally:
            srv.close()
            if os.path.exists(path):
                os.unlink(path)
            os.rmdir(tmp)
        self.conn.settimeout(None)
        self.rfile = self.conn.makefile("rb")

    def recv(self):
        line = self.rfile.readline()
        if not line:
            raise RuntimeError("game closed the connection")
        return line

    def send(self, action):
        self.conn.sendall(ACTIONS[action])

    def close(self):
        self.rfile.close()
        self.conn.close()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()


def random_actions(seed, n, n_actions=5):
    return np.random.default_rng(seed).integers(0, n_actions, n).tolist()


def compare(seed, level, actions, sim_cls=PacmanSim):
    """Play actions in the game and in the sim; returns (steps compared,
    None, sim) or (index, (game line, sim line), sim) at the first
    difference. Index 0 is the state after reset."""
    game = GameProc(seed, level)
    try:
        sim = sim_cls()
        sim.reset(seed, level)
        a, b = game.recv(), sim.state_json()
        if a != b:
            return 0, (a, b), sim
        for i, act in enumerate(actions, 1):
            game.send(act)
            sim.advance(act)
            a, b = game.recv(), sim.state_json()
            if a != b:
                return i, (a, b), sim
        return len(actions) + 1, None, sim
    finally:
        game.close()


def diff_lines(a, b):
    """Human-readable difference of two state lines."""
    import json
    sa, sb = json.loads(a), json.loads(b)
    out = []
    for k in sa:
        if k == "grid":
            for j, (ra, rb) in enumerate(zip(sa[k], sb[k])):
                if ra != rb:
                    out.append(f"grid row {j}: game {ra!r} sim {rb!r}")
        elif sa[k] != sb[k]:
            out.append(f"{k}: game {sa[k]} sim {sb[k]}")
    return out
