"""Protocol tests for the game's RL bridge (game/rlbridge.cc).

The game always opens an X window, so the tests start it under its own
Xvfb display and are skipped when the binary or Xvfb is missing.
"""

import json
import os
import random
import shutil
import socket
import subprocess
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAME = os.path.join(ROOT, "game", "pacman")

pytestmark = pytest.mark.skipif(
    not os.path.exists(GAME) or shutil.which("Xvfb") is None,
    reason="needs built game/pacman and Xvfb",
)


@pytest.fixture(scope="module")
def display():
    disp = ":%d" % (90 + os.getpid() % 100)
    xvfb = subprocess.Popen(["Xvfb", disp, "-nolisten", "tcp"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.5)
    yield disp
    xvfb.terminate()
    xvfb.wait()


class Game:
    def __init__(self, display, tmp_path, seed=1, name="game", headless=False, args=()):
        path = str(tmp_path / f"{name}.sock")
        self.srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.srv.bind(path)
        self.srv.listen(1)
        self.srv.settimeout(10)
        env = dict(os.environ, DISPLAY=display)
        extra = []
        if headless:
            env.pop("DISPLAY")
            extra = ["--headless"]
        self.proc = subprocess.Popen(
            [GAME, "--rl", path, "--fast", "--seed", str(seed)] + extra + list(args), env=env)
        self.conn, _ = self.srv.accept()
        self.rfile = self.conn.makefile("rb")

    def state(self):
        line = self.rfile.readline()
        assert line, "game closed the connection"
        return json.loads(line)

    def act(self, a):
        self.conn.sendall((a + "\n").encode())

    def close(self):
        self.rfile.close()
        self.conn.close()
        self.srv.close()
        try:
            return self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            raise


def play(game, n, seed=0):
    rng = random.Random(seed)
    states = []
    for _ in range(n):
        states.append(game.state())
        game.act(rng.choice("UDLRN"))
    return states


def test_state_format(display, tmp_path):
    g = Game(display, tmp_path)
    st = g.state()
    g.close()
    assert len(st["grid"]) == 23
    assert all(len(row) == 33 for row in st["grid"])
    assert set("".join(st["grid"])) <= set("#.o -")
    assert st["pacman"]["dir"] in "UDLRS"
    assert st["pacman"]["try_dir"] in "UDLRS"  # S on the first tick of a life
    assert len(st["ghosts"]) == 4
    for gh in st["ghosts"]:
        assert gh["state"] in ("normal", "hunted", "eyes")
        assert gh["dir"] in "UDLRS"
    assert (st["lives"], st["level"], st["done"]) == (3, 1, False)


def test_same_seed_same_game(display, tmp_path):
    runs = []
    for i in range(2):
        g = Game(display, tmp_path, name=f"run{i}")
        runs.append(play(g, 500))
        g.close()
    assert runs[0] == runs[1]


def test_game_exits_when_socket_closes(display, tmp_path):
    g = Game(display, tmp_path)
    play(g, 10)
    g.state()
    g.close()  # raises if the game is still running after 10 s


def test_headless_same_states_without_x(display, tmp_path):
    runs = []
    for headless in (False, True):
        g = Game(display, tmp_path, name=f"h{headless}", headless=headless)
        runs.append(play(g, 20000))
        assert g.close() == 1  # the game's normal exit code
    assert runs[0] == runs[1]


def test_headless_needs_rl():
    env = {k: v for k, v in os.environ.items() if k != "DISPLAY"}
    r = subprocess.run([GAME, "--headless"], env=env, capture_output=True, timeout=10)
    assert r.returncode == 1 and b"--rl" in r.stderr


def test_level_needs_rl():
    env = {k: v for k, v in os.environ.items() if k != "DISPLAY"}
    r = subprocess.run([GAME, "--level", "3"], env=env, capture_output=True, timeout=10)
    assert r.returncode == 1 and b"--rl" in r.stderr


def test_level_flag(display, tmp_path):
    """--level n starts on level n (maze n) and stays there after a game over."""
    from env.pacman_env import maze_layouts
    g = Game(display, tmp_path, headless=True, args=["--level", "7"])
    states = play(g, 20000)
    g.close()
    layout = [r.replace(".", " ").replace("o", " ") for r in states[0]["grid"]]
    assert layout == maze_layouts()[6]
    assert states[0]["level"] == 7
    overs = [i for i, st in enumerate(states[:-1]) if st["done"]]
    assert overs, "random play should lose a game in 20000 ticks"
    after = states[overs[0] + 1]
    assert (after["level"], after["lives"], after["score"]) == (7, 3, 0)


def test_level_1_flag_changes_nothing(display, tmp_path):
    runs = []
    for i, args in enumerate(([], ["--level", "1"])):
        g = Game(display, tmp_path, name=f"l{i}", headless=True, args=args)
        runs.append(play(g, 3000))
        g.close()
    assert runs[0] == runs[1]
