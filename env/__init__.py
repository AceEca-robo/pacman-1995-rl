import gymnasium

from env.pacman_env import PacmanEnv

# Truncation is done by PacmanEnv itself (max_episode_steps in the config),
# so no TimeLimit wrapper here.
gymnasium.register(id="Pacman1995-v0", entry_point="env.pacman_env:PacmanEnv")

__all__ = ["PacmanEnv"]
