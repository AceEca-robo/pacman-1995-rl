"""Common agent interface: act(obs) -> action index for PacmanEnv."""

from abc import ABC, abstractmethod

# PacmanEnv action indices (env/pacman_env.py ACTIONS)
UP, DOWN, LEFT, RIGHT, NOOP = range(5)


class Agent(ABC):
    """n_actions: 5 (U/D/L/R/N) or 4 (U/D/L/R), the env's action space size."""

    n_actions = 5

    @property
    def env_overrides(self):
        """Env config entries this agent needs (action set, extra obs parts);
        checkpoint agents set _env_overrides from their training env."""
        return getattr(self, "_env_overrides", {"actions": self.n_actions})

    def reset(self, seed=None):
        """Called at the start of every episode."""

    @abstractmethod
    def act(self, obs) -> int:
        """obs: PacmanEnv observation dict; returns an action in range(n_actions)."""
