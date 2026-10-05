import glob
import os
import tempfile

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from env.pacman_env import CH, CHANNELS, ROOT, PacmanEnv

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, "game", "pacman")),
    reason="needs built game/pacman",
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


def check_obs(obs):
    g = obs["grid"]
    assert g.shape[0] == len(CHANNELS)
    assert g[CH["pacman"]].sum() == 1
    pac_dirs = g[CH["pacman_up"]:CH["pacman_right"] + 1]
    assert pac_dirs.sum() <= 1 and np.all(pac_dirs.sum(0) <= g[CH["pacman"]])
    ghosts = g[CH["ghost_normal"]:CH["ghost_eyes"] + 1].sum(0)
    assert ghosts.sum() == 1  # 4 ghosts, 1/4 each
    ghost_dirs = g[CH["ghost_up"]:CH["ghost_right"] + 1].sum(0)
    assert np.all(ghost_dirs <= ghosts)  # directions only where ghosts are
    assert not np.any(g[CH["walls"]] * g[CH["gate"]])


def test_headless_needs_no_display(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    e = PacmanEnv()
    try:
        obs, _ = e.reset(seed=0)
        check_obs(obs)
        assert obs["grid"][CH["gate"]].sum() == 3
    finally:
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
    game = e._res["game"].pid
    e.close()
    e.close()  # idempotent
    assert not alive(game)  # waited for, so no zombie either
    assert sock_dirs() <= before


def test_1000_random_steps(env):
    obs, _ = env.reset(seed=11)
    env.action_space.seed(11)
    assert env.observation_space.contains(obs)
    for _ in range(1000):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert np.isfinite(r)
        check_obs(obs)
        if term or trunc:
            obs, info = env.reset()
            assert info["lives"] == 3
