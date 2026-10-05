"""Common agent interface: act(obs) -> action index for PacmanEnv."""

from abc import ABC, abstractmethod

# PacmanEnv action indices (env/pacman_env.py ACTIONS)
UP, DOWN, LEFT, RIGHT, NOOP = range(5)


class Agent(ABC):
    def reset(self, seed=None):
        """Called at the start of every episode."""

    @abstractmethod
    def act(self, obs) -> int:
        """obs: PacmanEnv observation dict; returns an action in Discrete(5)."""
