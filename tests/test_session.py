"""launcher/session.py: games on a private Xvfb, played by a person (keys
forwarded with XTest) or by the network, and closed again."""

import os
import shutil
import subprocess
import time

import pytest

from launcher import core
from launcher.session import Session

pytestmark = pytest.mark.skipif(
    not all(shutil.which(p) for p in core.TOOLS) or not os.access(core.GAME, os.X_OK),
    reason="needs Xvfb, xdotool, xwininfo and the built game")


def wait(cond, timeout=60.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def games():
    return subprocess.run(["pgrep", "-x", "pacman"], capture_output=True).stdout.split()


def close(s):
    s.stop()
    s.thread.join(timeout=15)
    assert not s.thread.is_alive() and s.error is None


def test_your_game_takes_the_arrows():
    before = set(games())
    s = Session("play", level=2, net_view=False)
    try:
        assert wait(lambda: s.snap is not None and s.keys is not None), s.error
        assert s.snap["info"]["level"] == 2 and s.snap["pr"] is None
        start = s.snap["obs"]["grid"]
        s.key("Left", True)
        v = s.version
        assert wait(lambda: s.version >= v + 4, 10)
        s.key("Left", False)
        assert s.snap["human"] == 2 or (s.snap["obs"]["grid"] != start).any()   # 2: L
        # pause stops the game process itself: no ticks while paused
        s.pause(True)
        time.sleep(0.4)
        v = s.version
        time.sleep(0.8)
        assert s.version == v
        s.pause(False)
        assert wait(lambda: s.version > v, 5)
    finally:
        close(s)
    assert set(games()) <= before   # the game is gone, as is its Xvfb


@pytest.mark.skipif(core.checkpoint() is None, reason="needs best_v04")
def test_the_network_plays_and_stops():
    s = Session("dqn", level=3)
    try:
        assert wait(lambda: s.version >= 3), s.error
        snap = s.snap
        assert snap["info"]["level"] == 3 and snap["played"] in range(5)
        assert snap["pr"]["q"].shape == (5,) and snap["img"].ndim == 3
    finally:
        close(s)


class Recorder(Session):
    """Remembers when each state arrived and pacman's try_dir in it."""
    def _read_keys(self, st):
        self.states.append((time.monotonic(), st["pacman"]["try_dir"]))
        super()._read_keys(st)


def test_short_taps_are_not_lost():
    """50 taps of 30 ms, each in a direction other than the last, at random
    points of the game's 0.25 s tick: the game reads the keyboard once per
    tick, the latch keeps every tap down until it has."""
    import random
    rng = random.Random(1)
    Recorder.states = []
    s = Recorder("play", level=1, env_config={"ghosts": False})
    try:
        assert wait(lambda: s.snap is not None and s.keys is not None), s.error
        cycle, taps = ["Up", "Left", "Down", "Right"], []
        for i in range(50):
            name = cycle[i % 4]
            taps.append((time.monotonic(), name[0]))
            s.key(name, True)
            time.sleep(0.03)
            s.key(name, False)
            time.sleep(rng.uniform(0.3, 0.55))
        time.sleep(0.6)
        assert not s.over and s.error is None
    finally:
        close(s)
    # every tap differs from the one before: the try_dir the game reports,
    # with repeats dropped, has to go through all 50 of them in order
    t0 = taps[0][0]
    tried = [d for t, d in s.states if t > t0 and d != "S"]   # S: none asked for yet
    tried = [d for i, d in enumerate(tried) if i == 0 or d != tried[i - 1]]
    assert tried == [d for _, d in taps], f"tapped {''.join(d for _, d in taps)}, game {''.join(tried)}"
