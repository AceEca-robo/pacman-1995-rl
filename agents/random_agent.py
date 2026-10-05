import numpy as np

from agents.base import Agent


class RandomAgent(Agent):
    """Uniform over U/D/L/R(/N)."""

    def __init__(self, seed=None, n_actions=5):
        self.rng = np.random.default_rng(seed)
        self.n_actions = n_actions

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def act(self, obs) -> int:
        return int(self.rng.integers(self.n_actions))
