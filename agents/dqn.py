"""Double + Dueling DQN with n-step returns for PacmanEnv.

Replay storage: every grid plane of PacmanEnv is either 0/1 or a count/4
(ghost planes), so a grid is stored bit-packed: one bit plane per binary
channel and three per count channel, 35 planes x 759 cells -> 3325 bytes
(the float32 grid is 63756 bytes). Decoding happens on the GPU per batch.
"""

import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from agents.base import Agent
from env.pacman_env import CHANNELS, HEIGHT, ROOT, WIDTH

DEFAULT_CONFIG = os.path.join(ROOT, "configs", "dqn.yaml")
N_ACTIONS = 5
CELLS = HEIGHT * WIDTH
COUNT_CH = [i for i, name in enumerate(CHANNELS) if name.startswith("ghost_")]  # values k/4
BINARY_CH = [i for i in range(len(CHANNELS)) if i not in COUNT_CH]
COUNT_BITS = 3  # k in 0..4
N_PLANES = len(BINARY_CH) + COUNT_BITS * len(COUNT_CH)
PACKED = (CELLS + 7) // 8


def load_config(path=None):
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


# --- observation packing ---------------------------------------------------

def pack_grids(grids):
    """float32 (B, C, H, W) grids -> uint8 (B, N_PLANES, PACKED)."""
    flat = grids.reshape(len(grids), len(CHANNELS), CELLS)
    counts = np.rint(flat[:, COUNT_CH] * 4).astype(np.uint8)
    bits = [flat[:, BINARY_CH] > 0]
    bits += [(counts >> b) & 1 for b in range(COUNT_BITS)]
    return np.packbits(np.concatenate(bits, 1).astype(np.uint8), axis=-1)


def pack_grid(grid):
    """float32 (C, H, W) grid -> uint8 (N_PLANES, PACKED)."""
    return pack_grids(grid[None])[0]


class GridDecoder:
    """uint8 (B, N_PLANES, PACKED) tensor -> float32 (B, C, H, W) on device."""

    def __init__(self, device):
        self.shifts = torch.arange(7, -1, -1, dtype=torch.uint8, device=device)
        self.weights = torch.tensor([2.0 ** b / 4 for b in range(COUNT_BITS)], device=device)
        order = BINARY_CH + COUNT_CH  # channel of each decoded plane group
        self.inverse = torch.tensor(np.argsort(order), device=device)

    def __call__(self, packed):
        b = packed.shape[0]
        bits = ((packed.unsqueeze(-1) >> self.shifts) & 1).reshape(b, N_PLANES, -1)[..., :CELLS]
        bits = bits.float()
        nb = len(BINARY_CH)
        counts = bits[:, nb:].reshape(b, COUNT_BITS, len(COUNT_CH), CELLS)
        counts = (counts * self.weights.view(1, -1, 1, 1)).sum(1)
        grid = torch.cat([bits[:, :nb], counts], 1)[:, self.inverse]
        return grid.reshape(b, len(CHANNELS), HEIGHT, WIDTH)


# --- replay ----------------------------------------------------------------

class ReplayBuffer:
    """n-step transitions (s, a, R, s', done, discount) in numpy ring buffers."""

    def __init__(self, capacity, vec_dim=3):
        self.capacity, self.size, self.pos = capacity, 0, 0
        self.obs = np.zeros((capacity, N_PLANES, PACKED), np.uint8)
        self.next_obs = np.zeros((capacity, N_PLANES, PACKED), np.uint8)
        self.vec = np.zeros((capacity, vec_dim), np.float32)
        self.next_vec = np.zeros((capacity, vec_dim), np.float32)
        self.act = np.zeros(capacity, np.uint8)
        self.ret = np.zeros(capacity, np.float32)
        self.discount = np.zeros(capacity, np.float32)  # gamma^k, 0 if terminated

    @property
    def nbytes(self):
        return sum(a.nbytes for a in (self.obs, self.next_obs, self.vec, self.next_vec,
                                      self.act, self.ret, self.discount))

    def add(self, obs, vec, act, ret, next_obs, next_vec, discount):
        i = self.pos
        self.obs[i], self.vec[i], self.act[i], self.ret[i] = obs, vec, act, ret
        self.next_obs[i], self.next_vec[i], self.discount[i] = next_obs, next_vec, discount
        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, rng, batch):
        idx = rng.integers(self.size, size=batch)
        return (self.obs[idx], self.vec[idx], self.act[idx], self.ret[idx],
                self.next_obs[idx], self.next_vec[idx], self.discount[idx])

    ARRAYS = ("obs", "next_obs", "vec", "next_vec", "act", "ret", "discount")

    def state_dict(self):
        """Only the filled part of the arrays."""
        d = {k: getattr(self, k)[:self.size] for k in self.ARRAYS}
        return {**d, "capacity": self.capacity, "size": self.size, "pos": self.pos}

    def load_state_dict(self, d):
        assert int(d["capacity"]) == self.capacity, "buffer_size differs from the checkpoint"
        self.size, self.pos = int(d["size"]), int(d["pos"])
        for k in self.ARRAYS:
            getattr(self, k)[:self.size] = d[k]


class NStep:
    """Turns one env's stream of steps into n-step transitions."""

    def __init__(self, n, gamma):
        self.n, self.gamma = n, gamma
        self.q = deque()

    def push(self, obs, vec, act, rew, next_obs, next_vec, terminated, ended):
        """Returns the list of finished transitions (tuples for ReplayBuffer.add)."""
        self.q.append((obs, vec, act, rew))
        out = []
        if ended:
            while self.q:
                out.append(self._emit(next_obs, next_vec, terminated))
        elif len(self.q) == self.n:
            out.append(self._emit(next_obs, next_vec, False))
        return out

    def _emit(self, next_obs, next_vec, terminated):
        ret = sum(self.gamma ** k * t[3] for k, t in enumerate(self.q))
        discount = 0.0 if terminated else self.gamma ** len(self.q)
        obs, vec, act, _ = self.q.popleft()
        return obs, vec, act, ret, next_obs, next_vec, discount


# --- network -----------------------------------------------------------------

class QNetwork(nn.Module):
    def __init__(self, cfg, vec_dim=3):
        super().__init__()
        layers, c = [], len(CHANNELS)
        for out, k, s in zip(cfg["conv_channels"], cfg["conv_kernels"], cfg["conv_strides"]):
            layers += [nn.Conv2d(c, out, k, s, padding=k // 2), nn.ReLU()]
            c = out
        self.conv = nn.Sequential(*layers, nn.Flatten())
        with torch.no_grad():
            conv_out = self.conv(torch.zeros(1, len(CHANNELS), HEIGHT, WIDTH)).shape[1]
        self.vec = nn.Sequential(nn.Linear(vec_dim, cfg["vec_hidden"]), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(conv_out + cfg["vec_hidden"], cfg["hidden"]), nn.ReLU())
        self.value = nn.Linear(cfg["hidden"], 1)
        self.adv = nn.Linear(cfg["hidden"], N_ACTIONS)

    def forward(self, grid, vec):
        h = self.fc(torch.cat([self.conv(grid), self.vec(vec)], 1))
        adv = self.adv(h)
        return self.value(h) + adv - adv.mean(1, keepdim=True)


# --- learner -----------------------------------------------------------------

class DQNLearner:
    """Speed (this is what keeps training above ~5k env steps/s on a laptop
    RTX 4060): fused Adam, TF32, optional bf16 autocast and torch.compile of
    the loss, online(s) and online(s') in one forward pass, no host syncs in
    update() (loss and Q come back as device tensors)."""

    def __init__(self, cfg, device):
        self.cfg, self.device = cfg, device
        cuda = str(device).startswith("cuda")
        if cuda:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.online = QNetwork(cfg["network"]).to(device)
        self.target = QNetwork(cfg["network"]).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.requires_grad_(False)
        self.opt = torch.optim.Adam(self.online.parameters(), lr=cfg["lr"], eps=cfg["adam_eps"],
                                    fused=cuda)
        self.decode = GridDecoder(device)
        self.amp = cuda and cfg.get("bf16", False)
        self._loss = torch.compile(self._loss_fn) if cfg.get("compile", False) else self._loss_fn

    def _t(self, x, dtype=None):
        return torch.as_tensor(x, device=self.device, dtype=dtype)

    @torch.no_grad()
    def q_values(self, packed, vecs):
        """packed: uint8 (B, N_PLANES, PACKED) numpy; returns numpy (B, N_ACTIONS).
        Plain fp32: entering torch.autocast costs ~0.1 ms (torch.cuda.is_available()
        on every enter), more than this tiny forward pass."""
        return self.online(self.decode(self._t(packed)), self._t(vecs)).cpu().numpy()

    def _loss_fn(self, obs2, vec2, act, ret, discount):
        """obs2/vec2: s and s' stacked; returns (Huber loss, mean max Q(s))."""
        n = act.shape[0]
        x = self.decode(obs2)
        with torch.autocast("cuda", torch.bfloat16, enabled=self.amp):
            q_both = self.online(x, vec2)
            with torch.no_grad():
                q_target = self.target(x[n:], vec2[n:])
        q_both, q_target = q_both.float(), q_target.float()
        q_all = q_both[:n]
        q = q_all.gather(1, act[:, None]).squeeze(1)
        with torch.no_grad():
            best = q_both[n:].argmax(1, keepdim=True)  # double DQN: online picks
            target = ret + discount * q_target.gather(1, best).squeeze(1)
        return F.smooth_l1_loss(q, target), q_all.detach().max(1).values.mean()

    def update(self, batch):
        obs, vec, act, ret, next_obs, next_vec, discount = batch
        loss, mean_q = self._loss(self._t(np.concatenate([obs, next_obs])),
                                  self._t(np.concatenate([vec, next_vec])),
                                  self._t(act, torch.long), self._t(ret), self._t(discount))
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg["grad_clip"], foreach=True)
        self.opt.step()
        return loss.detach(), mean_q

    def sync_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def state_dict(self):
        return {"online": self.online.state_dict(), "target": self.target.state_dict(),
                "opt": self.opt.state_dict()}

    def load_state_dict(self, d):
        self.online.load_state_dict(d["online"])
        self.target.load_state_dict(d["target"])
        self.opt.load_state_dict(d["opt"])


# --- agent for evaluation ----------------------------------------------------

class DQNAgent(Agent):
    """Greedy (or eps-greedy) policy from a checkpoint saved by scripts/train.py."""

    def __init__(self, checkpoint, epsilon=0.0, device=None, seed=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(checkpoint, map_location=device, weights_only=False)
        self.net = QNetwork(ck["config"]["network"]).to(device).eval()
        self.net.load_state_dict(ck["learner"]["online"])
        self.device, self.epsilon = device, epsilon
        self.rng = np.random.default_rng(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def act(self, obs) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(N_ACTIONS))
        g = torch.as_tensor(obs["grid"], device=self.device)[None]
        v = torch.as_tensor(obs["vec"], device=self.device)[None]
        return int(self.net(g, v).argmax(1).item())
