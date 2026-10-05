"""Common agent interface: act(obs) -> action index for PacmanEnv."""

from abc import ABC, abstractmethod

# PacmanEnv action indices (env/pacman_env.py ACTIONS)
UP, DOWN, LEFT, RIGHT, NOOP = range(5)


class Agent(ABC):
    """n_actions: 5 (U/D/L/R/N) or 4 (U/D/L/R), the env's action space size."""

    n_actions = 5

    def reset(self, seed=None):
        """Called at the start of every episode."""

    @abstractmethod
    def act(self, obs) -> int:
        """obs: PacmanEnv observation dict; returns an action in range(n_actions)."""
