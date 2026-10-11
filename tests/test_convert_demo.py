import os
import sys

import numpy as np
import pytest

from agents.dqn import pack_grid
from env.pacman_env import ROOT, PacmanEnv

sys.path.insert(0, os.path.join(ROOT, "tools"))
from convert_demo import recompute_nstep, safety_from_packed  # noqa: E402


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "game", "pacman")),
                    reason="needs built game/pacman")
def test_safety_from_packed_matches_the_env():
    env = PacmanEnv(config={"obs": {"cell_safety": True}, "start_level": 5})
    try:
        frames, vecs, planes = [], [], []
        rng = np.random.default_rng(5)
        obs, _ = env.reset(seed=5)
        for _ in range(600):
            frames.append(pack_grid(obs["grid"][:-1]))  # as stored without the plane
            vecs.append(obs["vec"])
            planes.append(obs["grid"][-1].ravel())
            obs, _, term, trunc, _ = env.step(int(rng.integers(5)))
            if term or trunc:
                obs, _ = env.reset()
    finally:
        env.close()
    got = safety_from_packed(np.stack(frames), np.stack(vecs))
    assert np.array_equal(got, np.stack(planes).astype(np.uint8))


def test_recompute_nstep_two_streams():
    # stream 0: an episode of 4 steps that terminates (frames 0..3, last obs
    # frame 4), then 3 steps of the next one (5..7, 8 = current obs; 7 still
    # pending);
    # stream 1: 2 steps then truncated (frame 2 = last obs), nothing after
    g = 0.5
    rew = np.array([1, 2, 3, 4, 0, 5, 6, 7, 0, 1, 1, 0], np.float32)
    valid = np.array([1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0], np.uint8)
    disc1 = np.array([.9, .9, .9, 0, 0, .9, .9, .9, 0, .9, .9, 0], np.float32)
    z = {"rew1": rew, "valid": valid, "disc1": disc1, "ret": np.zeros(12, np.float32),
         "discn": np.zeros(12, np.float32), "offn": np.zeros(12, np.uint8),
         "sizes": np.array([9, 3]), "final": np.array([7, 3]), "written": np.array([9, 3])}
    d1, ret, discn, offn, v = recompute_nstep(z, 2, g)
    assert list(v) == [1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0]   # 7 is still pending
    assert ret[0] == 1 + g * 2 and discn[0] == g ** 2 and offn[0] == 2
    assert ret[2] == 3 + g * 4 and discn[2] == 0 and offn[2] == 2   # terminates in the window
    assert ret[3] == 4 and discn[3] == 0 and offn[3] == 1
    assert ret[6] == 6 + g * 7 and discn[6] == g ** 2              # into the pending frames
    assert ret[10] == 1 and discn[10] == g and offn[10] == 1        # truncated: bootstraps
    assert d1[3] == 0 and d1[0] == np.float32(g)
