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


def test_sumtree_proportional():
    from agents.dqn import SumTree
    t = SumTree(5)
    t.set([0, 1, 2, 3, 4], [1.0, 0.0, 3.0, 0.0, 6.0])
    assert t.total == 10.0
    assert list(t.find([0.5, 1.5, 3.9, 4.1, 9.99])) == [0, 2, 2, 4, 4]
    t.set([4, 4], [2.0, 2.0])  # duplicate indices
    assert t.total == 6.0


def test_per_sampling_and_priorities():
    from agents.dqn import PrioritizedReplayBuffer
    buf = PrioritizedReplayBuffer(100, alpha=1.0, eps=0.0)
    z = np.zeros((35, 95), np.uint8)
    for i in range(10):
        buf.add(z, np.zeros(3), i % 5, float(i), z, np.zeros(3), 0.99)
    buf.update_priorities(np.arange(10), np.r_[np.full(9, 1.0), 91.0])  # last: 91 of 100
    rng = np.random.default_rng(0)
    (_, _, _, ret, *_), idx, w = buf.sample(rng, 1000, beta=1.0)
    assert 0.85 < np.mean(idx == 9) < 0.97
    assert w.max() == 1.0 and np.allclose(w[idx == 9], (1 / 91) / (1 / 1))
    assert np.all(ret == idx)
    buf2 = PrioritizedReplayBuffer(100, alpha=1.0, eps=0.0)
    buf2.load_state_dict(buf.state_dict())
    assert buf2.tree.total == buf.tree.total and buf2.max_priority == 91.0
