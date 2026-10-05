import numpy as np
import pytest
import torch

from agents.dqn import (N_ACTIONS, GridDecoder, NStep, QNetwork, ReplayBuffer,
                        load_config, pack_grid)
from env.pacman_env import CH, CHANNELS, HEIGHT, WIDTH, PacmanEnv


def test_pack_roundtrip_on_real_obs():
    env = PacmanEnv()
    dec = GridDecoder("cpu")
    try:
        obs, _ = env.reset(seed=0)
        grids = [obs["grid"]]
        for a in np.random.default_rng(0).integers(5, size=300):
            obs, *_ = env.step(a)
            grids.append(obs["grid"])
    finally:
        env.close()
    packed = np.stack([pack_grid(g) for g in grids])
    assert packed.dtype == np.uint8 and packed.shape[1:] == (35, 95)
    assert np.array_equal(dec(torch.as_tensor(packed)).numpy(), np.stack(grids))


def test_pack_ghost_counts():
    g = np.zeros((len(CHANNELS), HEIGHT, WIDTH), np.float32)
    g[CH["ghost_normal"], 5, 5] = 1.0  # all four ghosts in one cell
    g[CH["ghost_left"], 5, 5] = 0.75
    g[CH["ghost_eyes"], 0, 0] = 0.25
    out = GridDecoder("cpu")(torch.as_tensor(pack_grid(g))[None])[0].numpy()
    assert np.array_equal(out, g)


def test_nstep_returns():
    ns = NStep(3, 0.5)
    o = lambda i: np.full(2, i)  # noqa: E731
    out = []
    for t in range(4):
        out += ns.push(o(t), o(t), t, 1.0, o(t + 1), o(t + 1), False, False)
    assert len(out) == 2
    obs, _, act, ret, nxt, _, disc = out[0]
    assert (obs[0], act, ret, nxt[0], disc) == (0, 0, 1.75, 3, 0.125)
    # terminal: flush the rest with shorter horizons and no bootstrap
    out = ns.push(o(4), o(4), 4, 2.0, o(5), o(5), True, True)
    assert [t[2] for t in out] == [2, 3, 4]
    assert [t[3] for t in out] == [1 + 0.5 + 0.5, 1 + 1.0, 2.0]
    assert all(t[6] == 0.0 for t in out) and all(t[4][0] == 5 for t in out)
    # truncation: bootstrap with gamma^k
    ns.push(o(0), o(0), 0, 1.0, o(1), o(1), False, False)
    out = ns.push(o(1), o(1), 1, 1.0, o(2), o(2), False, True)
    assert [t[6] for t in out] == [0.25, 0.5]


def test_buffer_memory_500k():
    buf = ReplayBuffer(500_000)  # np.zeros is lazy, nothing is touched
    assert buf.nbytes < 3.5e9


def test_network_shapes():
    net = QNetwork(load_config()["network"])
    q = net(torch.zeros(7, len(CHANNELS), HEIGHT, WIDTH), torch.zeros(7, 3))
    assert q.shape == (7, N_ACTIONS)
