"""Double + Dueling DQN with n-step returns for PacmanEnv.

Replay storage: every grid plane of PacmanEnv is either 0/1 or a count/4
(ghost planes), so a grid is stored bit-packed: one bit plane per binary
channel and three per count channel, 35 planes x 759 cells -> 3325 bytes
(the float32 grid is 63756 bytes). The optional food-distance channel
(env obs.food_distance) holds d / D_max with integer d <= D_max = 86 and is
stored as 7 more bit planes of d (42 planes); the optional cell-safety
channel (env obs.cell_safety, 0/1) as one more plane after it, the
optional safe-exits channel (env obs.safe_exits, k / 4) as three bit planes
of k after that. Decoding happens on the GPU per batch.
"""

import contextlib
import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from agents.base import Agent
from env.pacman_env import CHANNELS, HEIGHT, ROOT, WIDTH, d_max, grid_extras, obs_dims

DEFAULT_CONFIG = os.path.join(ROOT, "configs", "dqn.yaml")
N_ACTIONS = 5  # default; the env's "actions" (4 or 5) decides
CELLS = HEIGHT * WIDTH
COUNT_CH = [i for i, name in enumerate(CHANNELS) if name.startswith("ghost_")]  # values k/4
BINARY_CH = [i for i in range(len(CHANNELS)) if i not in COUNT_CH]
COUNT_BITS = 3  # k in 0..4
N_PLANES = len(BINARY_CH) + COUNT_BITS * len(COUNT_CH)
DIST_BITS = 7  # food distance d in 0..D_max (86)
EXTRA_BITS = {"food_distance": DIST_BITS, "cell_safety": 1, "safe_exits": COUNT_BITS}
PACKED = (CELLS + 7) // 8


def default_extras(n_channels, extras=None):
    """The optional grid planes after CHANNELS (env grid_extras); None:
    food distance for each extra channel (all there was before cell_safety)."""
    if extras is None:
        extras = ("food_distance",) * (n_channels - len(CHANNELS))
    assert len(extras) == n_channels - len(CHANNELS), (n_channels, extras)
    return tuple(extras)


def cfg_extras(cfg):
    """grid_extras of a training config's env_config."""
    from env.pacman_env import load_env_config
    return grid_extras(load_env_config(cfg.get("env_config") or {}))


def n_planes(n_channels, extras=None):
    return N_PLANES + sum(EXTRA_BITS[e] for e in default_extras(n_channels, extras))


def large_margin(q_all, act, margin):
    """DQfD's supervised loss per sample: max_a [Q(s, a) + margin * (a != a_E)]
    - Q(s, a_E) >= 0; zero once Q(s, a_E) is at least margin above every
    other action's value."""
    q = q_all.gather(1, act[:, None]).squeeze(1)
    margins = margin * (1.0 - F.one_hot(act, q_all.shape[1]).to(q_all.dtype))
    return (q_all + margins).max(1).values - q


def frame_loss_weights(cfg):
    """(lam_1, lam_n, lam_margin, margin, l2) of the frame-batch loss: plain
    n-step DQN unless cfg["dqfd"]["enabled"]."""
    d = cfg.get("dqfd") or {}
    if not d.get("enabled"):
        return 0.0, 1.0, 0.0, 0.0, 0.0
    return (float(d["lambda_1step"]), float(d["lambda_nstep"]), float(d["lambda_margin"]),
            float(d["margin"]), float(d["l2"]))


def load_config(path=None):
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


# --- observation packing ---------------------------------------------------

def pack_grids(grids, extras=None):
    """float32 (B, C, H, W) grids -> uint8 (B, n_planes(C, extras), PACKED);
    C is 21 plus the optional planes `extras` (default_extras)."""
    c = grids.shape[1]
    extras = default_extras(c, extras)
    flat = grids.reshape(len(grids), c, CELLS)
    counts = np.rint(flat[:, COUNT_CH] * 4).astype(np.uint8)
    bits = [flat[:, BINARY_CH] > 0]
    bits += [(counts >> b) & 1 for b in range(COUNT_BITS)]
    for i, name in enumerate(extras):
        plane = flat[:, len(CHANNELS) + i:len(CHANNELS) + i + 1]
        if name == "food_distance":
            dist = np.rint(plane * d_max()).astype(np.uint8)
            bits += [(dist >> b) & 1 for b in range(DIST_BITS)]
        elif name == "safe_exits":  # k / 4, k in 0..4, as the ghost counts
            k = np.rint(plane * 4).astype(np.uint8)
            bits += [(k >> b) & 1 for b in range(COUNT_BITS)]
        else:
            bits.append(plane > 0)
    return np.packbits(np.concatenate(bits, 1).astype(np.uint8), axis=-1)


def pack_grid(grid, extras=None):
    """float32 (C, H, W) grid -> uint8 (n_planes, PACKED)."""
    return pack_grids(grid[None], extras)[0]


class GridDecoder:
    """uint8 (B, n_planes, PACKED) tensor -> float32 (B, C, H, W) on device."""

    def __init__(self, device, n_channels=len(CHANNELS), extras=None):
        self.n_channels = n_channels
        self.extras = default_extras(n_channels, extras)
        self.dist = bool(self.extras)
        if "food_distance" in self.extras:
            self.d_max = float(d_max())
            self.dist_weights = torch.tensor([2.0 ** b for b in range(DIST_BITS)], device=device)
        self.shifts = torch.arange(7, -1, -1, dtype=torch.uint8, device=device)
        self.weights = torch.tensor([2.0 ** b / 4 for b in range(COUNT_BITS)], device=device)
        order = BINARY_CH + COUNT_CH  # channel of each decoded plane group
        self.inverse = torch.tensor(np.argsort(order), device=device)

    def __call__(self, packed):
        b = packed.shape[0]
        bits = ((packed.unsqueeze(-1) >> self.shifts) & 1).reshape(b, packed.shape[1], -1)
        bits = bits[..., :CELLS].float()
        nb = len(BINARY_CH)
        counts = bits[:, nb:N_PLANES].reshape(b, COUNT_BITS, len(COUNT_CH), CELLS)
        counts = (counts * self.weights.view(1, -1, 1, 1)).sum(1)
        grid = torch.cat([bits[:, :nb], counts], 1)[:, self.inverse]
        if self.dist:
            planes, i = [grid], N_PLANES
            for name in self.extras:
                if name == "food_distance":  # same float32 division as the env: d / D_max
                    d = (bits[:, i:i + DIST_BITS] * self.dist_weights.view(1, -1, 1)).sum(
                        1, keepdim=True)
                    planes.append(d / self.d_max)
                elif name == "safe_exits":
                    planes.append((bits[:, i:i + COUNT_BITS] * self.weights.view(1, -1, 1)).sum(
                        1, keepdim=True))
                else:
                    planes.append(bits[:, i:i + 1])
                i += EXTRA_BITS[name]
            grid = torch.cat(planes, 1)
        return grid.reshape(b, self.n_channels, HEIGHT, WIDTH)


# --- replay ----------------------------------------------------------------

class ReplayBuffer:
    """n-step transitions (s, a, R, s', done, discount) in numpy ring buffers."""

    def __init__(self, capacity, vec_dim=3, planes=N_PLANES):
        self.capacity, self.size, self.pos = capacity, 0, 0
        self.obs = np.zeros((capacity, planes, PACKED), np.uint8)
        self.next_obs = np.zeros((capacity, planes, PACKED), np.uint8)
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

    def add_batch(self, transitions):
        """Several add() at once (same result as adding them one by one);
        returns the slots written."""
        k = len(transitions)
        if not k:
            return np.zeros(0, np.int64)
        idx = (self.pos + np.arange(k)) % self.capacity
        for name, col in zip(("obs", "vec", "act", "ret", "next_obs", "next_vec", "discount"),
                             zip(*transitions)):
            # a batch longer than the buffer: the last write to a slot wins, as with add()
            getattr(self, name)[idx] = np.asarray(col)
        self.pos = int((self.pos + k) % self.capacity)
        self.size = min(self.size + k, self.capacity)
        return idx

    def sample(self, rng, batch, beta=None):
        """Returns (transitions, indices, importance weights); weights are 1 here."""
        idx = rng.integers(self.size, size=batch)
        return self._get(idx), idx, np.ones(batch, np.float32)

    def update_priorities(self, idx, td_abs):
        pass

    def _get(self, idx):
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


class SumTree:
    """Binary tree of priority sums over a power-of-two number of leaves,
    vectorized updates and prefix-sum search."""

    def __init__(self, capacity):
        self.leaves = 1 << max(1, int(np.ceil(np.log2(capacity))))
        self.depth = int(np.log2(self.leaves))
        self.tree = np.zeros(2 * self.leaves, np.float64)

    @property
    def total(self):
        return self.tree[1]

    def get(self, idx):
        return self.tree[np.asarray(idx) + self.leaves]

    def set(self, idx, values):
        pos = np.asarray(idx, np.int64) + self.leaves
        self.tree[pos] = values
        pos = np.unique(pos // 2)
        while pos[0] >= 1:
            self.tree[pos] = self.tree[2 * pos] + self.tree[2 * pos + 1]
            pos = np.unique(pos // 2)

    def prefix_sum(self, n):
        """Sum of the priorities of leaves [0, n)."""
        lo, hi, s = self.leaves, self.leaves + n, 0.0
        while lo < hi:
            if lo & 1:
                s += self.tree[lo]
                lo += 1
            if hi & 1:
                hi -= 1
                s += self.tree[hi]
            lo, hi = lo // 2, hi // 2
        return s

    def find(self, prefix):
        """Leaf index where the running sum of priorities passes each prefix."""
        prefix = np.array(prefix, np.float64)
        node = np.ones(len(prefix), np.int64)
        for _ in range(self.depth):
            left = 2 * node
            right = prefix > self.tree[left]
            prefix -= self.tree[left] * right
            node = left + right
        return node - self.leaves


class PrioritizedReplayBuffer(ReplayBuffer):
    """Proportional PER (Schaul et al. 2016): P(i) ~ (|td_i| + eps)^alpha,
    new transitions get the current max priority, importance weights
    (N * P(i))^-beta normalized by the largest weight in the batch."""

    def __init__(self, capacity, alpha, eps, vec_dim=3, planes=N_PLANES):
        super().__init__(capacity, vec_dim, planes)
        self.alpha, self.eps = alpha, eps
        self.tree = SumTree(capacity)
        self.max_priority = 1.0

    def add(self, *transition):
        i = self.pos
        super().add(*transition)
        self.tree.set([i], self.max_priority)

    def add_batch(self, transitions):
        idx = super().add_batch(transitions)
        if len(idx):
            self.tree.set(np.unique(idx), self.max_priority)
        return idx

    def sample(self, rng, batch, beta=0.4):
        total = self.tree.total
        prefix = (np.arange(batch) + rng.random(batch)) * (total / batch)  # stratified
        idx = np.minimum(self.tree.find(prefix), self.size - 1)
        p = self.tree.get(idx) / total
        w = (self.size * p) ** -beta
        return self._get(idx), idx, (w / w.max()).astype(np.float32)

    def update_priorities(self, idx, td_abs):
        pr = (np.asarray(td_abs, np.float64) + self.eps) ** self.alpha
        self.tree.set(idx, pr)
        self.max_priority = max(self.max_priority, float(pr.max()))

    def state_dict(self):
        d = super().state_dict()
        return {**d, "priorities": self.tree.get(np.arange(self.size)),
                "max_priority": self.max_priority}

    def load_state_dict(self, d):
        super().load_state_dict(d)
        self.tree.set(np.arange(self.size), d["priorities"])
        self.max_priority = float(d["max_priority"])


def make_buffer(cfg, n_channels=len(CHANNELS), vec_dim=3):
    per = cfg.get("per", {})
    planes = n_planes(n_channels, cfg_extras(cfg) if "env_config" in cfg else None)
    if per.get("enabled"):
        return PrioritizedReplayBuffer(cfg["buffer_size"], per["alpha"], per["eps"], vec_dim, planes)
    return ReplayBuffer(cfg["buffer_size"], vec_dim, planes)


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
        self.value = nn.Linear(cfg["hidden"], 1)
        self.adv = nn.Linear(cfg["hidden"], n_actions)

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

    def __init__(self, cfg, device, n_actions=N_ACTIONS, n_channels=len(CHANNELS), vec_dim=3):
        self.cfg, self.device, self.n_actions = cfg, device, n_actions
        cuda = str(device).startswith("cuda")
        if cuda:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.online = QNetwork(cfg["network"], n_actions, vec_dim, n_channels).to(device)
        self.target = QNetwork(cfg["network"], n_actions, vec_dim, n_channels).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.requires_grad_(False)
        # frame batches (learner process, agents/shared_replay.py): loss weights
        self.lam_1, self.lam_n, self.lam_m, self.margin, l2 = frame_loss_weights(cfg)
        # L2 on the weights: loss l2 * ||theta||^2 / 2 = coupled weight decay l2
        self.opt = torch.optim.Adam(self.online.parameters(), lr=cfg["lr"], eps=cfg["adam_eps"],
                                    weight_decay=l2, fused=cuda)
        self.decode = GridDecoder(device, n_channels,
                                  cfg_extras(cfg) if "env_config" in cfg else None)
        self.amp = cuda and cfg.get("bf16", False)
        self._loss = torch.compile(self._loss_fn) if cfg.get("compile", False) else self._loss_fn
        self._floss = torch.compile(self._frame_loss_fn) if cfg.get("compile", False) \
            else self._frame_loss_fn
        # set by train.py's BackgroundLearner: forward passes for acting and
        # optimizer steps (in another thread) do not interleave
        self.lock = contextlib.nullcontext()
        self.act_stream = None  # enable_actor_stream()
        # cfg "fast_learner" (async_learner runs): online(s') in its own
        # no-grad pass (same math, ~5% less GPU time), cudnn autotuning (not
        # bitwise reproducible), bf16 forward passes for acting
        self.fast = bool(cfg.get("fast_learner")) and cuda
        if self.fast:
            torch.backends.cudnn.benchmark = True

    def enable_actor_stream(self):
        """q_values() on its own high-priority CUDA stream, so acting does not
        queue behind the learner's kernels (async_learner). Events keep a
        forward pass and an optimizer step from overlapping on the GPU: the
        forward waits for the last step enqueued before it, the next step for
        the last forward enqueued before it (both enqueued under self.lock)."""
        if not str(self.device).startswith("cuda"):
            return
        self.act_stream = torch.cuda.Stream(priority=-1)
        self._act_done, self._step_done = torch.cuda.Event(), torch.cuda.Event()

    def _t(self, x, dtype=None):
        return torch.as_tensor(x, device=self.device, dtype=dtype)

    @torch.no_grad()
    def q_values(self, packed, vecs):
        """packed: uint8 (B, N_PLANES, PACKED) numpy; returns numpy (B, n_actions).
        Plain fp32: entering torch.autocast costs ~0.1 ms (torch.cuda.is_available()
        on every enter), more than this tiny forward pass."""
        if self.act_stream is None:
            with self.lock:
                q = self.online(self.decode(self._t(packed)), self._t(vecs))
            return q.cpu().numpy()
        with torch.cuda.stream(self.act_stream):
            with self.lock:
                self.act_stream.wait_event(self._step_done)
                with torch.autocast("cuda", torch.bfloat16, enabled=self.fast):
                    q = self.online(self.decode(self._t(packed)), self._t(vecs)).float()
                self._act_done.record(self.act_stream)
            return q.cpu().numpy()

    def _loss_fn(self, obs2, vec2, act, ret, discount, weights):
        """obs2/vec2: s and s' stacked; returns (importance-weighted Huber loss,
        mean max Q(s), |TD error| per sample)."""
        n = act.shape[0]
        x = self.decode(obs2)
        with torch.autocast("cuda", torch.bfloat16, enabled=self.amp):
            if self.fast:  # online(s') without autograd: backward only through s
                q_all = self.online(x[:n], vec2[:n])
                with torch.no_grad():
                    q_next = self.online(x[n:], vec2[n:])
            else:
                q_both = self.online(x, vec2)
                q_all, q_next = q_both[:n], q_both[n:]
            with torch.no_grad():
                q_target = self.target(x[n:], vec2[n:])
        q_all, q_next, q_target = q_all.float(), q_next.float(), q_target.float()
        q = q_all.gather(1, act[:, None]).squeeze(1)
        with torch.no_grad():
            best = q_next.argmax(1, keepdim=True)  # double DQN: online picks
            target = ret + discount * q_target.gather(1, best).squeeze(1)
        loss = (weights * F.smooth_l1_loss(q, target, reduction="none")).mean()
        return loss, q_all.detach().max(1).values.mean(), (q.detach() - target).abs()

    def request_conv_trainable(self, flag):
        """Freeze / unfreeze the online network's convolutions at the start of
        the next update() (so it happens in the thread that runs updates)."""
        self._conv_request = flag

    def _frame_loss_fn(self, obs, vec, act, rew1, disc1, ret, discn, demo, weights, lam_m):
        """DQfD-style loss on frame batches (Hester et al. 2018): n-step double
        DQN TD (Huber) * lam_n + 1-step TD * lam_1 + large-margin
        classification * lam_m on demo samples only:
        max_a [Q(s, a) + margin * (a != a_demo)] - Q(s, a_demo).
        obs/vec: s, then s_{t+1} (only with lam_1), then s_{t+n}, stacked.
        lam_m: the margin weight as a 0-d tensor (scheduled values do not
        recompile the compiled loss).
        Returns (importance-weighted loss, mean max Q(s), |n-step TD| per
        sample, mean margin loss over the demo samples)."""
        n = act.shape[0]
        x = self.decode(obs)
        with torch.autocast("cuda", torch.bfloat16, enabled=self.amp):
            q_all = self.online(x[:n], vec[:n])
            with torch.no_grad():  # backward only through s
                q_next = self.online(x[n:], vec[n:])
                q_target = self.target(x[n:], vec[n:])
        q_all, q_next, q_target = q_all.float(), q_next.float(), q_target.float()
        q = q_all.gather(1, act[:, None]).squeeze(1)
        with torch.no_grad():  # double DQN: online picks, target evaluates
            boot = q_target.gather(1, q_next.argmax(1, keepdim=True)).squeeze(1)
            target_n = ret + discn * boot[-n:]
        loss = self.lam_n * F.smooth_l1_loss(q, target_n, reduction="none")
        if self.lam_1:
            target_1 = rew1 + disc1 * boot[:n].detach()
            loss = loss + self.lam_1 * F.smooth_l1_loss(q, target_1, reduction="none")
        je = large_margin(q_all, act, self.margin)
        if self.lam_m:
            loss = loss + lam_m * je * demo
        loss = (weights * loss).mean()
        je_demo = (je.detach() * demo).sum() / demo.sum().clamp(min=1.0)
        return loss, q_all.detach().max(1).values.mean(), (q.detach() - target_n).abs(), je_demo

    def update_frames(self, b, weights, lam_m=None):
        """b: a batch dict of agents/shared_replay.py Sampler; lam_m: the
        margin weight for this update (default the config's). Returns device
        tensors (loss, mean max Q, |n-step TD| per sample, demo margin loss)."""
        self._apply_conv_request()
        if {"cell_safety", "safe_exits"} & set(self.decode.extras):
            self._safety_check(b)
        frames = ("obs", "obs1", "obsn") if self.lam_1 else ("obs", "obsn")
        loss, mean_q, td, je = self._floss(
            self._t(np.concatenate([b[k] for k in frames])),
            self._t(np.concatenate([b["vec" + k[3:]] for k in frames])),
            self._t(b["act"], torch.long), self._t(b["rew1"]), self._t(b["disc1"]),
            self._t(b["ret"]), self._t(b["discn"]), self._t(b["demo"], torch.float32),
            self._t(weights), self._t(self.lam_m if lam_m is None else lam_m, torch.float32))
        self._step(loss)
        return loss.detach(), mean_q, td, je

    def _safety_check(self, b):
        """Prints, at the first update and every 100k after, the share of
        cells set in the batch's cell-safety plane and the safe-exits plane
        (where the config has them), agent and demo samples apart: shows the
        planes reach the learner."""
        self._n_frame_updates = getattr(self, "_n_frame_updates", -1) + 1
        if self._n_frame_updates % 100_000:
            return
        demo = np.asarray(b["demo"]) > 0
        i = N_PLANES
        for name in self.decode.extras:
            if name in ("cell_safety", "safe_exits"):
                bits = np.unpackbits(b["obs"][:, i:i + EXTRA_BITS[name]], axis=-1)[..., :CELLS]
                ones = (bits.max(1) > 0).sum(1)  # cells with a nonzero value
                parts = [f"{tag} {ones[m].mean():.1f} cells/frame ({(ones[m] > 0).mean():.0%} "
                         f"nonzero)" for tag, m in (("agent", ~demo), ("demo", demo)) if m.any()]
                print(f"{name} in the batch (update {self._n_frame_updates}): "
                      + ", ".join(parts), flush=True)
            i += EXTRA_BITS[name]

    def _apply_conv_request(self):
        flag = getattr(self, "_conv_request", None)
        if flag is not None:  # frozen parameters get no grad; Adam skips them
            self._conv_request = None
            self.online.conv.requires_grad_(flag)

    def update(self, batch, weights):
        """Returns device tensors (loss, mean max Q, |TD| per sample)."""
        self._apply_conv_request()
        obs, vec, act, ret, next_obs, next_vec, discount = batch
        loss, mean_q, td = self._loss(self._t(np.concatenate([obs, next_obs])),
                                      self._t(np.concatenate([vec, next_vec])),
                                      self._t(act, torch.long), self._t(ret), self._t(discount),
                                      self._t(weights))
        self._step(loss)
        return loss.detach(), mean_q, td

    def _step(self, loss):
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg["grad_clip"], foreach=True)
        with self.lock:
            if self.act_stream is not None:
                torch.cuda.current_stream().wait_event(self._act_done)
            self.opt.step()
            if self.act_stream is not None:
                self._step_done.record()

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
        self.n_actions = ck.get("n_actions", N_ACTIONS)  # runs before dqn3 had 5
        env_cfg = ck.get("env_config") or {}
        n_channels, vec_dim = obs_dims(env_cfg)
        # what the env must be configured with to produce this agent's inputs
        self._env_overrides = {"actions": self.n_actions, "obs": env_cfg.get("obs") or {}}
        self.net = QNetwork(ck["config"]["network"], self.n_actions, vec_dim,
                            n_channels).to(device).eval()
        self.net.load_state_dict(ck["learner"]["online"])
        self.device, self.epsilon = device, epsilon
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
