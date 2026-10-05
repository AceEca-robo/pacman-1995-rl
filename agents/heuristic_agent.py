"""Hand-written baseline: BFS on the grid.

Priority each step:
  1. a normal ghost is closer than flee_distance -> go to the nearest food
     along cells pacman reaches safety_margin steps before any ghost; if there
     is none, step to the neighbour farthest from the ghosts;
  2. super, enough supertime and a hunted ghost within hunt_max_distance ->
     go for it;
  3. otherwise go to the nearest food or superfood.
Distances are BFS steps. Pacman cannot pass walls or the ghost house gate,
ghosts can pass the gate.
"""

import os
from collections import deque

import numpy as np
import yaml

from agents.base import DOWN, LEFT, NOOP, RIGHT, UP, Agent
from env.pacman_env import CH, ROOT, SUPERTIME

DEFAULT_CONFIG = os.path.join(ROOT, "configs", "heuristic.yaml")
MOVES = ((UP, 0, -1), (DOWN, 0, 1), (LEFT, -1, 0), (RIGHT, 1, 0))
INF = 1 << 30


class _Graph:
    """Neighbour lists of passable cells, flat indices y * W + x."""

    def __init__(self, passable):
        h, w = passable.shape
        self.n = h * w
        self.nbrs = [[] for _ in range(self.n)]
        for y, x in zip(*np.nonzero(passable)):
            for move, dx, dy in MOVES:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and passable[ny, nx]:
                    self.nbrs[y * w + x].append((move, ny * w + nx))

    def dist(self, sources):
        """BFS distance from the nearest source; INF where unreachable."""
        d = [INF] * self.n
        q = deque()
        for s in sources:
            if d[s] != 0:
                d[s] = 0
                q.append(s)
        while q:
            c = q.popleft()
            for _, nb in self.nbrs[c]:
                if d[nb] == INF:
                    d[nb] = d[c] + 1
                    q.append(nb)
        return d

    def first_move_to(self, start, is_target, allowed=None):
        """First move of a shortest path from start to the nearest target cell,
        using only allowed cells (all if None); None if no target reachable."""
        seen = {start: None}
        q = deque([start])
        while q:
            c = q.popleft()
            if c != start and is_target[c]:
                return seen[c]
            for move, nb in self.nbrs[c]:
                if nb not in seen and (allowed is None or allowed(nb)):
                    seen[nb] = move if c == start else seen[c]
                    q.append(nb)
        return None


class HeuristicAgent(Agent):
    def __init__(self, config=None, n_actions=5):
        if config is None or isinstance(config, (str, os.PathLike)):
            with open(config or DEFAULT_CONFIG) as f:
                config = yaml.safe_load(f)
        self.cfg = config
        self.n_actions = n_actions
        self._graphs = {}  # walls/gate layout -> (pacman graph, ghost graph)

    def _graphs_for(self, walls, gate):
        key = walls.tobytes() + gate.tobytes()
        if key not in self._graphs:
            self._graphs[key] = (_Graph(~(walls | gate)), _Graph(~walls))
        return self._graphs[key]

    def act(self, obs) -> int:
        g = obs["grid"]
        w = g.shape[2]
        walls, gate = g[CH["walls"]] > 0, g[CH["gate"]] > 0
        pac_g, ghost_g = self._graphs_for(walls, gate)
        flat = g.reshape(g.shape[0], -1)
        pac = int(np.flatnonzero(flat[CH["pacman"]])[0])
        normal = np.flatnonzero(flat[CH["ghost_normal"]]).tolist()
        hunted = flat[CH["ghost_hunted"]] > 0
        food = (flat[CH["food"]] > 0) | (flat[CH["superfood"]] > 0)
        supertime = obs["vec"][1] * SUPERTIME

        if normal:
            d_ghost = ghost_g.dist(normal)
            if d_ghost[pac] < self.cfg["flee_distance"]:
                d_pac = pac_g.dist([pac])
                margin = self.cfg["safety_margin"]
                move = pac_g.first_move_to(
                    pac, food, allowed=lambda c: d_pac[c] + margin < d_ghost[c])
                if move is not None:
                    return move
                options = pac_g.nbrs[pac]
                if not options:
                    return self._idle(g)
                return max(options, key=lambda mn: d_ghost[mn[1]])[0]

        if self.cfg["hunt"] and hunted.any() and supertime >= self.cfg["hunt_min_supertime"]:
            d_pac = pac_g.dist([pac])
            if min(d_pac[c] for c in np.flatnonzero(hunted)) <= self.cfg["hunt_max_distance"]:
                move = pac_g.first_move_to(pac, hunted)
                if move is not None:
                    return move

        move = pac_g.first_move_to(pac, food)
        return self._idle(g) if move is None else move

    def _idle(self, g):
        """Nothing to do: N, or without N keep pacman's current direction."""
        if self.n_actions == 5:
            return NOOP
        dirs = g[CH["pacman_up"]:CH["pacman_right"] + 1].reshape(4, -1).max(1)
        return int(dirs.argmax()) if dirs.any() else UP
