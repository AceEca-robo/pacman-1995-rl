"""Behaviour cloning of the heuristic agent (scripts/train_bc.py).

BCNetwork has QNetwork's trunk (same conv / vec / fc layers and parameter
names, so its convolutions can initialize a DQN, scripts/train.py
init_conv_from) and one linear head of action logits.
"""

import numpy as np
import torch
import torch.nn as nn

from agents.base import Agent
from agents.dqn import GridDecoder, N_ACTIONS, pack_grids
from env.pacman_env import CHANNELS, HEIGHT, WIDTH, obs_dims


class BCNetwork(nn.Module):
    def __init__(self, cfg, n_actions=N_ACTIONS, vec_dim=3, in_channels=len(CHANNELS)):
        super().__init__()
        layers, c = [], in_channels
        for out, k, s in zip(cfg["conv_channels"], cfg["conv_kernels"], cfg["conv_strides"]):
            layers += [nn.Conv2d(c, out, k, s, padding=k // 2), nn.ReLU()]
            c = out
        self.conv = nn.Sequential(*layers, nn.Flatten())
        with torch.no_grad():
            conv_out = self.conv(torch.zeros(1, in_channels, HEIGHT, WIDTH)).shape[1]
        self.vec = nn.Sequential(nn.Linear(vec_dim, cfg["vec_hidden"]), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(conv_out + cfg["vec_hidden"], cfg["hidden"]), nn.ReLU())
        self.head = nn.Linear(cfg["hidden"], n_actions)

    def forward(self, grid, vec):
        return self.head(self.fc(torch.cat([self.conv(grid), self.vec(vec)], 1)))


def load_shards(paths):
    """npz shards from tools/gen_bc_data.py -> dict of concatenated arrays
    (obs, vec, act, level)."""
    parts = {"obs": [], "vec": [], "act": [], "level": []}
    for path in paths:
        with np.load(path) as z:
            for k in parts:
                parts[k].append(z[k])
    return {k: np.concatenate(v) for k, v in parts.items()}


def load_bc(checkpoint, device):
    """-> (network in eval mode, checkpoint dict)."""
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    n_channels, vec_dim = obs_dims(ck.get("env_config") or {})
    net = BCNetwork(ck["config"]["network"], ck["n_actions"], vec_dim, n_channels).to(device)
    net.load_state_dict(ck["net"])
    return net.eval(), ck


class BCAgent(Agent):
    """Argmax (or eps-greedy) of the BC logits; act_batch for many envs."""

    def __init__(self, checkpoint, epsilon=0.0, device=None, seed=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net, ck = load_bc(checkpoint, device)
        self.n_actions = ck["n_actions"]
        self._env_overrides = {"actions": self.n_actions,
                               "obs": (ck.get("env_config") or {}).get("obs") or {}}
        self.device, self.epsilon = device, epsilon
        self.decode = GridDecoder(device, len(CHANNELS))
        self.rng = np.random.default_rng(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def act(self, obs) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        g = torch.as_tensor(obs["grid"], device=self.device)[None]
        v = torch.as_tensor(obs["vec"], device=self.device)[None]
        return int(self.net(g, v).argmax(1).item())

    @torch.no_grad()
    def act_batch(self, packed, vecs):
        """Greedy actions for packed grids (B, planes, PACKED) and vecs (B, 3)."""
        g = self.decode(torch.as_tensor(packed, device=self.device))
        v = torch.as_tensor(vecs, device=self.device)
        return self.net(g, v).argmax(1).cpu().numpy()

    def act_grids(self, grids, vecs):
        return self.act_batch(pack_grids(grids), vecs)
