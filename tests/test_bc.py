import numpy as np
import torch

from agents.bc import BCAgent, BCNetwork
from agents.dqn import QNetwork, pack_grids
from env.pacman_env import CHANNELS, HEIGHT, WIDTH
from scripts.train_bc import class_weights

NET = {"conv_channels": [8, 16, 16], "conv_kernels": [3, 3, 3], "conv_strides": [1, 2, 2],
       "vec_hidden": 8, "hidden": 32}


def test_conv_state_dict_fits_qnetwork():
    bc, q = BCNetwork(NET), QNetwork(NET)
    conv = {k: v for k, v in bc.state_dict().items() if k.startswith("conv.")}
    q.load_state_dict({**q.state_dict(), **conv})
    x = torch.rand(2, len(CHANNELS), HEIGHT, WIDTH)
    assert torch.equal(bc.conv(x), q.conv(x))


def test_class_weights_auto():
    cfg = {"class_weights": "auto", "imbalance_ratio": 2.0, "min_class_share": 0.01}
    labels = np.array([0] * 60 + [1] * 20 + [2] * 10 + [3] * 10)
    w, shares = class_weights(cfg, labels, 5)
    assert np.allclose(shares[:4], [0.6, 0.2, 0.1, 0.1]) and shares[4] == 0
    assert np.allclose(w, [0.25 / 0.6, 0.25 / 0.2, 2.5, 2.5, 1.0])
    balanced = np.repeat(np.arange(4), 25)
    assert np.allclose(class_weights(cfg, balanced, 5)[0], 1.0)


def test_bc_agent_act_matches_batch(tmp_path):
    torch.manual_seed(0)
    net = BCNetwork(NET)
    path = tmp_path / "bc.pt"
    torch.save({"algo": "bc", "config": {"network": NET}, "n_actions": 5,
                "env_config": {"actions": 5}, "net": net.state_dict()}, path)
    agent = BCAgent(str(path), device="cpu")
    rng = np.random.default_rng(0)
    grids = (rng.random((3, len(CHANNELS), HEIGHT, WIDTH)) < 0.2).astype(np.float32)
    vecs = rng.random((3, 3)).astype(np.float32)
    single = [agent.act({"grid": g, "vec": v}) for g, v in zip(grids, vecs)]
    assert list(agent.act_batch(pack_grids(grids), vecs)) == single
