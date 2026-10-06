"""PPO (clipped surrogate, GAE) with a shared actor-critic network.

For PacmanEnv's Dict observation the trunk is the DQN's CNN over the grid
plus an MLP over vec (agents/dqn.py), grids stored bit-packed in rollouts.
For a plain Box observation (used by the CartPole sanity test) the trunk is
an MLP.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from agents.base import Agent
from agents.dqn import GridDecoder, pack_grids
from env.pacman_env import HEIGHT, WIDTH, obs_dims


def layer_init(layer, gain=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_space, n_actions, net_cfg):
        super().__init__()
        self.dict_obs = isinstance(obs_space, spaces.Dict)
        hidden = net_cfg["hidden"]
        if self.dict_obs:
            in_channels = obs_space["grid"].shape[0]
            layers, c = [], in_channels
            for out, k, s in zip(net_cfg["conv_channels"], net_cfg["conv_kernels"],
                                 net_cfg["conv_strides"]):
                layers += [layer_init(nn.Conv2d(c, out, k, s, padding=k // 2)), nn.ReLU()]
                c = out
            self.conv = nn.Sequential(*layers, nn.Flatten())
            with torch.no_grad():
                conv_out = self.conv(torch.zeros(1, in_channels, HEIGHT, WIDTH)).shape[1]
            vec_dim = obs_space["vec"].shape[0]
            self.vec = nn.Sequential(layer_init(nn.Linear(vec_dim, net_cfg["vec_hidden"])), nn.ReLU())
            trunk_in = conv_out + net_cfg["vec_hidden"]
        else:
            trunk_in = int(np.prod(obs_space.shape))
        self.fc = nn.Sequential(layer_init(nn.Linear(trunk_in, hidden)), nn.ReLU())
        if not self.dict_obs:  # a second layer for the small MLP case
            self.fc.append(layer_init(nn.Linear(hidden, hidden)))
            self.fc.append(nn.Tanh())
        self.pi = layer_init(nn.Linear(hidden, n_actions), gain=0.01)
        self.v = layer_init(nn.Linear(hidden, 1), gain=1.0)

    def forward(self, obs):
        """obs: (grid, vec) tensors for Dict spaces, else a flat tensor.
        Returns (logits, value)."""
        if self.dict_obs:
            grid, vec = obs
            h = self.fc(torch.cat([self.conv(grid), self.vec(vec)], 1))
        else:
            h = self.fc(obs.flatten(1))
        return self.pi(h), self.v(h).squeeze(1)


class ObsCodec:
    """Rollout storage format and device tensors for one observation space."""

    def __init__(self, obs_space, device):
        self.dict_obs = isinstance(obs_space, spaces.Dict)
        self.device = device
        self.decode = GridDecoder(device, obs_space["grid"].shape[0]) if self.dict_obs else None

    def store(self, obs):
        """Batched env observation -> compact numpy form (tuple)."""
        if self.dict_obs:
            return pack_grids(obs["grid"]), np.asarray(obs["vec"], np.float32)
        return (np.asarray(obs, np.float32),)

    def tensor(self, stored):
        if self.dict_obs:
            packed, vec = stored
            return (self.decode(torch.as_tensor(packed, device=self.device)),
                    torch.as_tensor(vec, device=self.device))
        return torch.as_tensor(stored[0], device=self.device)


def compute_gae(rewards, values, ended, last_value, gamma, lam):
    """rewards, values, ended: (T, N); ended[t] = the episode ended with step t
    (rewards of truncated steps already include gamma * V(final obs)).
    last_value: (N,) value of the observation after step T-1.
    Returns advantages and returns (T, N)."""
    T = len(rewards)
    adv = np.zeros_like(rewards)
    last = np.zeros(rewards.shape[1], np.float32)
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - ended[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + values


class PPOLearner:
    def __init__(self, cfg, obs_space, n_actions, device):
        self.cfg, self.device = cfg, device
        self.net = ActorCritic(obs_space, n_actions, cfg["network"]).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg["lr"], eps=cfg["adam_eps"])
        self.codec = ObsCodec(obs_space, device)

    @torch.no_grad()
    def act(self, stored, rng=None, greedy=False):
        """Returns actions, log-probs and values (numpy) for a batch."""
        logits, value = self.net(self.codec.tensor(stored))
        if greedy:
            a = logits.argmax(1)
        else:
            a = torch.distributions.Categorical(logits=logits).sample()
        logp = torch.distributions.Categorical(logits=logits).log_prob(a)
        return a.cpu().numpy(), logp.cpu().numpy(), value.float().cpu().numpy()

    @torch.no_grad()
    def value(self, stored):
        return self.net(self.codec.tensor(stored))[1].float().cpu().numpy()

    def q_values(self, packed, vecs):
        """Greedy-action scores for scripts/train.py's evaluate(): the logits."""
        with torch.no_grad():
            return self.net(self.codec.tensor((packed, vecs)))[0].float().cpu().numpy()

    def set_lr(self, lr):
        for g in self.opt.param_groups:
            g["lr"] = lr

    def update(self, obs, actions, logp_old, advantages, returns, values_old, rng):
        """obs: tuple of arrays with leading dim B (flattened rollout).
        Returns a dict of mean statistics."""
        cfg = self.cfg
        b = len(actions)
        mb = b // cfg["minibatches"]
        t = lambda x, dt=torch.float32: torch.as_tensor(x, device=self.device, dtype=dt)  # noqa: E731
        actions_t, logp_old_t = t(actions, torch.long), t(logp_old)
        adv_t, ret_t, v_old_t = t(advantages), t(returns), t(values_old)
        stats = {k: [] for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac")}
        for _ in range(cfg["epochs"]):
            perm = rng.permutation(b)
            for start in range(0, b - mb + 1, mb):
                idx = perm[start:start + mb]
                logits, value = self.net(self.codec.tensor(tuple(o[idx] for o in obs)))
                dist = torch.distributions.Categorical(logits=logits)
                idx_t = torch.as_tensor(idx, device=self.device)
                logp = dist.log_prob(actions_t[idx_t])
                ratio = (logp - logp_old_t[idx_t]).exp()
                adv = adv_t[idx_t]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg = -torch.min(ratio * adv,
                                ratio.clamp(1 - cfg["clip"], 1 + cfg["clip"]) * adv).mean()
                v_loss = 0.5 * F.mse_loss(value, ret_t[idx_t])
                entropy = dist.entropy().mean()
                loss = pg + cfg["vf_coef"] * v_loss - cfg["ent_coef"] * entropy
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg["max_grad_norm"])
                self.opt.step()
                with torch.no_grad():
                    log_ratio = logp - logp_old_t[idx_t]
                    stats["approx_kl"].append(((ratio - 1) - log_ratio).mean())
                    stats["clipfrac"].append(((ratio - 1).abs() > cfg["clip"]).float().mean())
                stats["policy_loss"].append(pg.detach())
                stats["value_loss"].append(v_loss.detach())
                stats["entropy"].append(entropy.detach())
        out = {k: torch.stack(v).mean().item() for k, v in stats.items()}
        var = np.var(returns)
        out["explained_variance"] = float(1 - np.var(returns - values_old) / var) if var > 0 else 0.0
        return out

    def state_dict(self):
        return {"net": self.net.state_dict(), "opt": self.opt.state_dict()}

    def load_state_dict(self, d):
        self.net.load_state_dict(d["net"])
        self.opt.load_state_dict(d["opt"])


class PPOAgent(Agent):
    """Greedy (argmax of the policy) agent from a scripts/train_ppo.py checkpoint."""

    def __init__(self, checkpoint, device=None, seed=None, sample=False):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(checkpoint, map_location=device, weights_only=False)
        self.n_actions = ck["n_actions"]
        env_cfg = ck.get("env_config") or {}
        n_channels, vec_dim = obs_dims(env_cfg)
        self._env_overrides = {"actions": self.n_actions, "obs": env_cfg.get("obs") or {}}
        obs_space = spaces.Dict({
            "grid": spaces.Box(0.0, 1.0, (n_channels, HEIGHT, WIDTH), np.float32),
            "vec": spaces.Box(0.0, 3.0, (vec_dim,), np.float32)})
        self.net = ActorCritic(obs_space, self.n_actions, ck["config"]["network"]).to(device).eval()
        self.net.load_state_dict(ck["learner"]["net"])
        self.device, self.sample = device, sample
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def act(self, obs) -> int:
        g = torch.as_tensor(obs["grid"], device=self.device)[None]
        v = torch.as_tensor(obs["vec"], device=self.device)[None]
        logits, _ = self.net((g, v))
        if self.sample:
            p = torch.softmax(logits, 1)[0].cpu().numpy()
            return int(self.rng.choice(len(p), p=p / p.sum()))
        return int(logits.argmax(1).item())


def load_agent(checkpoint, **kw):
    """DQNAgent or PPOAgent, from the checkpoint's "algo" (default dqn)."""
    ck_algo = torch.load(checkpoint, map_location="cpu", weights_only=False).get("algo", "dqn")
    if ck_algo == "ppo":
        return PPOAgent(checkpoint, seed=kw.get("seed"))
    from agents.dqn import DQNAgent
    return DQNAgent(checkpoint, **kw)

