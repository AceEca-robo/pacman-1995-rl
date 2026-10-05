import numpy as np

from agents.base import Agent


class RandomAgent(Agent):
    """Uniform over U/D/L/R/N."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def act(self, obs) -> int:
        return int(self.rng.integers(5))
