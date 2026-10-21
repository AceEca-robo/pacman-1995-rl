"""One game in the app's window: the real game on a private Xvfb, played in
a worker thread by the planner ("watch"), the network alone ("dqn") or a
person ("play": the game runs with --rl-watch, the app forwards the arrow
keys with XTest, launcher/xkeys.py). The UI thread draws .snap, the latest
tick, and never waits for the game.

The networks are loaded once per process and kept (_AGENTS): a new session
starts only after the previous one has let go of them (`after`).
"""

import os
import signal
import threading
import time
import traceback

import numpy as np

from . import core

DIRS = "UDLR"
_AGENTS = {}   # (kind, config or checkpoint, device) -> (agent, NetProbe)


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _agent(kind, smooth=False):
    """(agent, probe) for "watch" (PlannerAgent), "dqn" or "play" (DQNAgent)."""
    from scripts.netviz import NetProbe
    device = _device()
    if kind == "watch":
        from scripts.play import PlannerAgent
        cfg = core.plan_config(8 if smooth else 32)
        key = (kind, cfg, device)
        if key not in _AGENTS:
            agent = PlannerAgent(cfg, device, trace=True)
            _AGENTS[key] = (agent, NetProbe(agent.net.net, agent.net.device))
    else:
        from agents.dqn import DQNAgent
        key = ("dqn", core.checkpoint(), device)
        if key not in _AGENTS:
            agent = DQNAgent(core.checkpoint(), epsilon=0.0, device=device)
            _AGENTS[key] = (agent, NetProbe(agent.net, agent.device))
    return _AGENTS[key]


class Session:
    def __init__(self, mode, level, smooth=False, net_view=False, seed=None, after=None):
        self.mode, self.level, self.smooth = mode, level, smooth
        self.net_view = net_view    # "play": run the network on every tick (key N)
        self.seed = int.from_bytes(os.urandom(2), "little") if seed is None else seed
        self.snap = None            # the latest tick, a dict (see _publish)
        self.version = 0            # bumped with every snap
        self.over = False           # the game ended (the last snap stays)
        self.error = None           # (exception text, traceback) if it failed
        self.has_net = mode != "play" or core.checkpoint() is not None
        self.obs_cfg = None
        self.game_display = None    # the private Xvfb, e.g. ":2"
        self.keys = None            # XKeys on it ("play")
        self.paused = False
        self._stop = threading.Event()
        self._game = None           # the game's Popen
        self._after = after
        self.thread = threading.Thread(target=self._run, name=f"session-{mode}", daemon=True)
        self.thread.start()

    # --- UI thread ------------------------------------------------------------

    def key(self, name, down):
        """An arrow from the app's window to the game ("play")."""
        if self.keys is not None and not self.paused:
            self.keys.press(name, down)

    def release_keys(self):
        if self.keys is not None:
            self.keys.release_all()

    def pause(self, on):
        """The agents wait before their next move; a person's game, which does
        not wait for anybody, is stopped with SIGSTOP."""
        if on == self.paused:
            return
        self.paused = on
        if self.mode == "play":
            self.release_keys()
            if self._game is not None and self._game.poll() is None:
                self._game.send_signal(signal.SIGSTOP if on else signal.SIGCONT)

    def stop(self):
        """Ask the worker to end; it closes the game and the Xvfb itself."""
        self._stop.set()
        self.pause(False)

    def running(self):
        return self.thread.is_alive()

    # --- worker thread ----------------------------------------------------------

    def _publish(self, **snap):
        self.snap = snap
        self.version += 1

    def _wait(self):
        while self.paused and not self._stop.is_set():
            time.sleep(0.02)
        return not self._stop.is_set()

    def _run(self):
        from PIL import ImageGrab
        from env.pacman_env import PacmanEnv
        from scripts.play import PrivateX, find_window, window_box
        from .xkeys import XKeys
        if self._after is not None:
            self._after.thread.join()
        xserver = env = None
        try:
            agent = probe = None
            if self.has_net:
                agent, probe = _agent(self.mode, self.smooth)
            overrides = getattr(agent, "env_overrides", {}) if agent is not None else {}
            self.obs_cfg = overrides.get("obs")
            if self._stop.is_set():
                return
            xserver = PrivateX()
            self.game_display = xserver.name
            # the same seed and level as the reset below: no second game process
            env = PacmanEnv(display=xserver.name, fast=False, watch=self.mode == "play",
                            seed=self.seed, config={"max_episode_steps": None,
                                                    "start_level": self.level, **overrides})
            obs, info = env.reset(seed=self.seed)
            self._game = env._res["game"]
            if self.mode == "watch":
                agent.reset(self.seed, self.level)
            elif self.mode == "dqn":
                agent.reset(seed=self.seed)
            box = window_box(xserver.name, find_window(xserver.name))
            if self.mode == "play":
                self.keys = XKeys(xserver.name)
                if os.environ.get("PACMAN_RL_DEBUG"):
                    print(f"game display {xserver.name}", flush=True)
            grab = lambda: np.asarray(ImageGrab.grab(  # noqa: E731
                bbox=box, xdisplay=xserver.name).convert("RGB"))
            tick = deaths = 0
            while not self._stop.is_set():
                img = grab()
                lives = info["lives"]
                if self.mode == "play":
                    pr = None
                    if probe is not None and self.net_view:
                        pr = probe(obs["grid"], obs["vec"])
                    d = env._state["pacman"]["dir"]
                    self._publish(obs=obs, info=info, img=img, pr=pr, played=None, trace=None,
                                  tick=tick, deaths=deaths, t_dec=None,
                                  human=DIRS.index(d) if d in DIRS else None)
                    obs, info = env.observe()
                    term = bool(env._state["done"])
                else:
                    if not self._wait():
                        break
                    t0 = time.perf_counter()
                    a = agent.act(obs)
                    t_dec = time.perf_counter() - t0
                    pr = probe(obs["grid"], obs["vec"], a)
                    self._publish(obs=obs, info=info, img=img, pr=pr, played=a,
                                  trace=getattr(agent, "last_trace", None), tick=tick,
                                  deaths=deaths, t_dec=t_dec, human=None)
                    if not self._wait():
                        break
                    obs, _, term, _, info = env.step(a)
                deaths += int(info["lives"] < lives)
                tick += 1
                if term:
                    snap = dict(self.snap, img=grab(), info=info, tick=tick, deaths=deaths)
                    self.over = True
                    self._publish(**snap)
                    break
        except BaseException as e:   # SystemExit too: play.py's helpers exit on errors
            if not self._stop.is_set():
                self.error = (f"{type(e).__name__}: {e}", traceback.format_exc())
        finally:
            if self.keys is not None:
                self.keys.close()
            if env is not None:
                env.close()
            if xserver is not None:
                xserver.close()
