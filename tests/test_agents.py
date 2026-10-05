import numpy as np

from agents.base import DOWN, LEFT, NOOP, RIGHT, UP
from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent
from env.pacman_env import CH, CHANNELS

CFG = {"flee_distance": 5, "safety_margin": 1, "hunt": True,
       "hunt_min_supertime": 6, "hunt_max_distance": 12}


def make_obs(rows, supertime=0.0):
    """rows: strings over '#', ' ', '.', 'P' (pacman), 'G' (normal ghost), 'H' (hunted)."""
    h, w = len(rows), len(rows[0])
    grid = np.zeros((len(CHANNELS), h, w), np.float32)
    for y, row in enumerate(rows):
        for x, c in enumerate(row):
            name = {"#": "walls", ".": "food", "P": "pacman",
                    "G": "ghost_normal", "H": "ghost_hunted"}.get(c)
            if name:
                grid[CH[name], y, x] = 0.25 if name.startswith("ghost") else 1
    return {"grid": grid, "vec": np.array([1, supertime, 1 / 16], np.float32)}


def test_goes_to_nearest_food():
    obs = make_obs(["#######",
                    "#.  P #",
                    "#######"])
    assert HeuristicAgent(CFG).act(obs) == LEFT


def test_flees_from_close_ghost():
    obs = make_obs(["#########",
                    "#.  PG  #",
                    "#### ####",
                    "#       #",
                    "#########"])
    # food is to the left, the ghost right next to pacman: food path is not
    # safe (ghost is as close to it), so step away from the ghost
    assert HeuristicAgent(CFG).act(obs) in (LEFT, DOWN)


def test_hunts_when_super():
    obs = make_obs(["#######",
                    "#. P H#",
                    "#######"], supertime=0.5)
    assert HeuristicAgent(CFG).act(obs) == RIGHT
    obs["vec"][1] = 0.0  # no supertime left: go for food instead
    assert HeuristicAgent(CFG).act(obs) == LEFT


def test_noop_without_food():
    obs = make_obs(["#####",
                    "# P #",
                    "#####"])
    assert HeuristicAgent(CFG).act(obs) == NOOP


def test_random_agent_reproducible():
    a, b = RandomAgent(3), RandomAgent(3)
    xs = [a.act(None) for _ in range(50)]
    assert xs == [b.act(None) for _ in range(50)]
    assert set(xs) <= {UP, DOWN, LEFT, RIGHT, NOOP}
