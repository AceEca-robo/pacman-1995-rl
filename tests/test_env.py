import glob
import os
import shutil
import tempfile

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from env.pacman_env import ROOT, PacmanEnv

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, "game", "pacman")) or shutil.which("Xvfb") is None,
    reason="needs built game/pacman and Xvfb",
)


def alive(pid):
    return os.path.exists(f"/proc/{pid}")


def sock_dirs():
    return set(glob.glob(os.path.join(tempfile.gettempdir(), "pacman-env-*")))


@pytest.fixture
def env():
    e = PacmanEnv()
    yield e
    e.close()


def test_check_env(env):
    check_env(env, skip_render_check=True)


def rollout(seed, actions):
    e = PacmanEnv()
    out = []
    try:
        obs, _ = e.reset(seed=seed)
        out.append(obs)
        for a in actions:
            obs, r, term, trunc, info = e.step(a)
            out.append((obs, r, term, trunc, info))
            if term or trunc:
                obs, _ = e.reset()
                out.append(obs)
    finally:
        e.close()
    return out


def same(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    return a == b


def test_same_seed_same_actions(env):
    actions = np.random.default_rng(0).integers(5, size=2000).tolist()
    a = rollout(7, actions)
    b = rollout(7, actions)
    assert sum(1 for x in a if isinstance(x, tuple) and x[2]) >= 2  # crosses game overs
    assert same(a, b)


def test_reset_seed_restarts_game(env):
    first, _ = env.reset(seed=3)
    for a in [3, 3, 3, 0, 0]:
        env.step(a)
    again, info = env.reset(seed=3)
    assert same(first, again)
    assert (info["score"], info["lives"]) == (0, 3)


def test_truncation_and_reset_mid_game():
    e = PacmanEnv(config={"max_episode_steps": 20})
    try:
        e.reset(seed=1)
        truncs = [e.step(4)[3] for _ in range(20)]
        assert truncs == [False] * 19 + [True]
        _, info = e.reset()
        assert (info["score"], info["lives"], info["level"]) == (0, 3, 1)
    finally:
        e.close()


def test_close_leaves_nothing():
    before = sock_dirs()
    e = PacmanEnv()
    e.reset(seed=0)
    e.step(0)
    game, xvfb = e._res["game"].pid, e._res["xvfb"].pid
    e.close()
    e.close()  # idempotent
    assert not alive(game) and not alive(xvfb)  # waited for, so no zombies either
    assert sock_dirs() <= before


def test_1000_random_steps(env):
    obs, _ = env.reset(seed=11)
    env.action_space.seed(11)
    assert env.observation_space.contains(obs)
    for _ in range(1000):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert np.isfinite(r)
        assert obs["grid"][3].sum() == 1  # exactly one pacman
        if term or trunc:
            obs, info = env.reset()
            assert info["lives"] == 3
