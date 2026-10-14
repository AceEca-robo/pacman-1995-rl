"""Gymnasium environment for pacman 1.0 (1995) over the game's RL bridge.

The game (game/pacman --rl <sock>) connects to a Unix socket served here,
sends one JSON state per tick and blocks for one action line. See
game/rlbridge.cc for the protocol.

By default the game runs with --headless (no X11 at all); pass display=":0"
to watch it in a window.

Observation is a Dict, see docs/observation.md:
  grid: float32 (21, 23, 33) planes in [0, 1], see CHANNELS
  vec:  float32 (3,) = [min(lives, 9) / 3, supertime_left / SUPERTIME, level / LEVELS]
"""

import ctypes
import json
import os
import signal
import socket
import subprocess
import tempfile
import weakref
from collections import deque

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(ROOT, "configs", "env_default.yaml")

HEIGHT, WIDTH = 23, 33  # BOARDHEIGHT, BOARDWIDTH in game/sizes.h
SUPERTIME = 50          # game/pac.h
LEVELS = 16             # game/pac.h; boards repeat (randomly) after that
MAX_LIVES = 9           # only for the obs bound; the game has no limit
ACTIONS = [b"U\n", b"D\n", b"L\n", b"R\n", b"N\n"]
CHANNELS = ("walls", "gate", "food", "superfood",
            "pacman", "pacman_up", "pacman_down", "pacman_left", "pacman_right",
            "try_up", "try_down", "try_left", "try_right",
            "ghost_normal", "ghost_hunted", "ghost_eyes",
            "ghost_up", "ghost_down", "ghost_left", "ghost_right",
            "bonus")
CH = {name: i for i, name in enumerate(CHANNELS)}
DIR_OFFSET = {"U": 0, "D": 1, "L": 2, "R": 3}  # "S" (still) sets no direction plane
GHOSTS = 4
EVENTS = ("eaten_dot", "eaten_energizer", "ghost_eaten", "level_up", "death",
          "bonus_eaten", "step")


PAC_START = (16, 17)    # PACX, PACY in game/pac.h
_D_MAX = None


def _passable_neighbours(cells):
    """Neighbour lists (flat indices) of cells pacman can enter: not a wall
    ('#') and not the ghost house gate ('-')."""
    free = [c not in "#-" for c in cells]
    nbrs = [[] for _ in cells]
    for c in range(len(cells)):
        if not free[c]:
            continue
        y, x = divmod(c, WIDTH)
        for nx, ny in ((x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)):
            if 0 <= nx < WIDTH and 0 <= ny < HEIGHT and free[ny * WIDTH + nx]:
                nbrs[c].append(ny * WIDTH + nx)
    return nbrs


def _bfs_dist(nbrs, start):
    dist = {start: 0}
    q = deque([start])
    while q:
        c = q.popleft()
        for nb in nbrs[c]:
            if nb not in dist:
                dist[nb] = dist[c] + 1
                q.append(nb)
    return dist


def maze_layouts():
    """The LEVELS boards from game/board.h as lists of 23 strings over '#', ' ',
    '-' (food removed: only walls and the gate matter here)."""
    import re
    with open(os.path.join(ROOT, "game", "board.h"), encoding="latin-1") as f:
        rows = re.findall(r'\{"([^"]{%d})"\}' % WIDTH, f.read())
    assert len(rows) == LEVELS * HEIGHT, len(rows)
    table = str.maketrans({"O": "#", "_": "-", ".": " ", "o": " "})
    return [[r.translate(table) for r in rows[i:i + HEIGHT]] for i in range(0, len(rows), HEIGHT)]


def d_max():
    """Largest BFS distance between two cells pacman can reach from its start,
    over all boards. Computed once per process (~2 s)."""
    global _D_MAX
    if _D_MAX is None:
        best = 0
        for board in maze_layouts():
            nbrs = _passable_neighbours("".join(board))
            component = _bfs_dist(nbrs, PAC_START[1] * WIDTH + PAC_START[0])
            best = max(best, max(max(_bfs_dist(nbrs, c).values()) for c in component))
        _D_MAX = best
    return _D_MAX


FOOD_DISTANCE_CH = len(CHANNELS)  # index of the optional food-distance plane
HUNGER_SCALE = 200                # steps_since_food is reported / this, clipped to 1
GRID_EXTRAS = ("food_distance", "cell_safety", "safe_exits")  # optional grid planes, in this order
SAFE_EXITS_RADIUS = 6   # obs.safe_exits: cells within this many steps of pacman
SAFE_EXITS_LENGTH = 8   # ... count the directions with a safe path at least this long


def grid_extras(env_cfg):
    """Names of the optional grid planes after CHANNELS that an env config
    switches on, in plane order."""
    obs = env_cfg.get("obs") or {}
    return tuple(name for name in GRID_EXTRAS if obs.get(name))


def obs_dims(env_cfg):
    """(grid channels, vec size) of the observation for an env config."""
    obs = env_cfg.get("obs") or {}
    return (len(CHANNELS) + len(grid_extras(env_cfg)),
            3 + bool(obs.get("steps_since_food")))


def _ghost_neighbours(cells):
    """Neighbour lists (flat indices) for ghosts: every cell but walls ('#');
    ghosts pass the house gate ('-') both ways in any state (game/ghost.cc
    only stops at classWall)."""
    free = [c != "#" for c in cells]
    nbrs = [[] for _ in cells]
    for c in range(len(cells)):
        if not free[c]:
            continue
        y, x = divmod(c, WIDTH)
        for nx, ny in ((x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)):
            if 0 <= nx < WIDTH and 0 <= ny < HEIGHT and free[ny * WIDTH + nx]:
                nbrs[c].append(ny * WIDTH + nx)
    return nbrs


def _bfs_levels(nbrs, sources, n):
    """BFS steps from the nearest of `sources` to every cell (list of n;
    -1 where unreachable)."""
    dist = [-1] * n
    frontier = []
    for c in sources:
        if dist[c] < 0:
            dist[c] = 0
            frontier.append(c)
    d = 0
    while frontier:
        d += 1
        nxt = []
        for c in frontier:
            for nb in nbrs[c]:
                if dist[nb] < 0:
                    dist[nb] = d
                    nxt.append(nb)
        frontier = nxt
    return dist


def cell_safety_field(pac_nbrs, ghost_nbrs, pacman, normal_ghosts, super_on):
    """Safety plane, float32 (H*W,): 1 on each cell pacman reaches (BFS,
    walls and the gate block) strictly before every normal ghost (BFS from
    all of them at once, only walls block), else 0; with super_on (supertime
    > 0) every cell pacman can reach is 1. pacman and normal_ghosts are flat
    cell indices. Hunted ghosts and eyes are no threat."""
    n = len(pac_nbrs)
    dp = _bfs_levels(pac_nbrs, [pacman], n)
    if super_on or not normal_ghosts:
        return (np.array(dp) >= 0).astype(np.float32)
    dg = _bfs_levels(ghost_nbrs, normal_ghosts, n)
    out = np.zeros(n, np.float32)
    for c in range(n):
        if dp[c] >= 0 and (dg[c] < 0 or dp[c] < dg[c]):
            out[c] = 1.0
    return out


def safe_exits_field(pac_nbrs, safe, pacman, radius=SAFE_EXITS_RADIUS,
                     length=SAFE_EXITS_LENGTH):
    """Safe exits plane, float32 (H*W,): for pacman's cell and every cell
    within radius steps of it (BFS, walls and the gate block), the number of
    directions out of the cell with a safe path at least `length` steps long,
    / 4; 0 elsewhere and on cells that are not safe themselves. safe: the
    cell-safety plane (cell_safety_field). A direction counts when its
    neighbour n is safe and some safe cell is length - 1 steps from n by a
    path of safe cells that does not go back through the cell (a shortest
    path, so a simple one; a loop of safe cells around a block counts once
    it is long enough). A dead end with a safe corridor out -> 1/4, a
    squeeze -> 0."""
    n = len(pac_nbrs)
    out = np.zeros(n, np.float32)
    region = [pacman]
    seen = {pacman}
    frontier = [pacman]
    for _ in range(radius):
        nxt = []
        for c in frontier:
            for nb in pac_nbrs[c]:
                if nb not in seen:
                    seen.add(nb)
                    nxt.append(nb)
        region += nxt
        frontier = nxt
    need = length - 1
    for c in region:
        if not safe[c]:
            continue
        k = 0
        for first in pac_nbrs[c]:
            if not safe[first]:
                continue
            visited = {c, first}
            front, d = [first], 0
            while front and d < need:
                d += 1
                nxt = []
                for x in front:
                    for nb in pac_nbrs[x]:
                        if nb not in visited and safe[nb]:
                            visited.add(nb)
                            nxt.append(nb)
                front = nxt
            if front and d >= need:
                k += 1
        out[c] = k / 4
    return out


def food_distance_field(cells, nbrs):
    """BFS steps from every cell pacman can reach to the nearest dot or
    energizer (multi-source BFS from the food); int array (H*W,), 0 on
    walls, the gate, cells pacman cannot reach, food cells, and everywhere
    when there is no food."""
    dist = np.zeros(len(cells), np.int64)
    q = deque(c for c, ch in enumerate(cells) if ch in ".o")
    seen = set(q)
    while q:
        c = q.popleft()
        for nb in nbrs[c]:
            if nb not in seen:
                seen.add(nb)
                dist[nb] = dist[c] + 1
                q.append(nb)
    return dist


def parse_start_level(spec):
    """start_level config value -> (lo, hi): an int n gives (n, n),
    "random:a-b" a level drawn uniformly from a..b at each reset."""
    if isinstance(spec, str) and spec.startswith("random:"):
        lo, hi = (int(x) for x in spec[len("random:"):].split("-"))
    elif isinstance(spec, (int, np.integer)) and not isinstance(spec, bool):
        lo = hi = int(spec)
    else:
        raise ValueError(f"start_level must be an int or 'random:a-b', not {spec!r}")
    if not 1 <= lo <= hi:
        raise ValueError(f"start_level {spec!r}: need 1 <= a <= b")
    return lo, hi


def load_env_config(config):
    """Full env config: configs/env_default.yaml updated by config (a path,
    a dict of overrides or None)."""
    return _load_config(config)


def _load_config(config):
    with open(DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    if config is None:
        return cfg
    if isinstance(config, (str, os.PathLike)):
        with open(config) as f:
            config = yaml.safe_load(f)
    for key, value in config.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key] = {**cfg[key], **value}
        else:
            cfg[key] = value
    return cfg


def _die_with_parent():
    # Children get SIGTERM if the Python process dies without close().
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except OSError:
        pass


def _stop_game(res):
    for key in ("rfile", "conn"):
        if res.get(key) is not None:
            res[key].close()
            res[key] = None
    proc = res.get("game")
    if proc is not None:
        try:
            proc.wait(timeout=5)  # the game quits on its own when the socket closes
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        res["game"] = None


class PacmanEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=None, seed=0, display=None, render_mode=None, fast=True):
        """config: path to yaml, dict of overrides or None (configs/env_default.yaml).
        seed: --seed of the first game process (reset(seed=...) overrides it).
        display: X display to show the game on; None runs it --headless.
        fast: run the game with --fast (no sleeping); False plays in real time
        (0.25 s per tick, half that while pacman is super), for watching."""
        self.cfg = _load_config(config)
        self.render_mode = render_mode
        n_actions = self.cfg.get("actions", 5)
        if n_actions not in (4, 5):
            raise ValueError("actions must be 4 (U/D/L/R) or 5 (U/D/L/R/N)")
        self.action_space = spaces.Discrete(n_actions)
        shaping = self.cfg.get("shaping") or {}
        self._shaping = shaping.get("enabled", False)
        self._k_dist, self._shaping_gamma = shaping.get("k_dist", 0.0), shaping.get("gamma", 0.99)
        self._neighbours = {}  # maze layout -> passable neighbour lists
        self._d_max = d_max() if self._shaping else None
        self._hunger_limit = self.cfg.get("hunger_limit") or 0
        self._hunger_mode = self.cfg.get("hunger_mode") or "terminate"
        if self._hunger_mode not in ("terminate", "truncate"):
            raise ValueError("hunger_mode must be 'terminate' or 'truncate'")
        endgame = self.cfg.get("endgame_dot") or {}
        self._endgame_k = endgame.get("k", 0.0) if endgame.get("enabled") else 0.0
        self._levels = parse_start_level(self.cfg.get("start_level", 1))
        self._prefix_prob = self.cfg.get("prefix_prob") or 0.0
        self._prefix_sets, self._prefix_weights = [], []  # one list of prefixes per file
        if self._prefix_prob:
            mix = self.cfg.get("prefix_mix") or [{"file": self.cfg["prefix_file"], "weight": 1.0}]
            for entry in mix:
                path = os.path.join(ROOT, entry["file"])
                with open(path) as f:
                    data = json.load(f)
                if data["n_actions"] != n_actions:
                    raise ValueError(f"{path} was recorded with {data['n_actions']} actions")
                self._prefix_sets.append(data["prefixes"])
                self._prefix_weights.append(float(entry["weight"]))
            w = np.array(self._prefix_weights)
            self._prefix_weights = w / w.sum()
        obs_cfg = self.cfg.get("obs") or {}
        self._food_distance = bool(obs_cfg.get("food_distance"))
        self._cell_safety = bool(obs_cfg.get("cell_safety"))
        self._safe_exits = bool(obs_cfg.get("safe_exits"))
        self.grid_extras = grid_extras(self.cfg)
        self._steps_since_food = bool(obs_cfg.get("steps_since_food"))
        if self._food_distance:
            self._d_max = d_max()
            self._field_cache = (None, None)  # (grid string, normalized field)
        n_channels, vec_dim = obs_dims(self.cfg)
        vec_high = [MAX_LIVES / 3, 1.0, 1.0] + [1.0] * (vec_dim - 3)
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(0.0, 1.0, (n_channels, HEIGHT, WIDTH), np.float32),
            # lives can exceed 3 through bonus lives, clipped at MAX_LIVES
            "vec": spaces.Box(np.zeros(vec_dim, np.float32),
                              np.array(vec_high, np.float32), dtype=np.float32),
        })

        self._res = {}
        self._finalizer = weakref.finalize(self, _stop_game, self._res)
        self._display = display
        self._fast = fast
        self._binary = os.path.join(ROOT, self.cfg["game"]["binary"])
        self._state = None   # last state received
        self._steps = 0
        self._start_game(seed, self._levels[0])

    # --- processes -------------------------------------------------------

    def _start_game(self, seed, level):
        tmp = tempfile.mkdtemp(prefix="pacman-env-")
        path = os.path.join(tmp, "game.sock")
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            srv.bind(path)
            srv.listen(1)
            srv.settimeout(self.cfg["game"]["connect_timeout"])
            # glibc treats srandom(0) as srandom(1), so shift by one to keep
            # env seeds 0 and 1 apart
            game_seed = seed % (2**31 - 1) + 1
            cmd = [self._binary, "--rl", path, "--seed", str(game_seed)]
            if self._fast:
                cmd.append("--fast")
            if not self.cfg.get("ghosts", True):
                cmd.append("--no-ghosts")
            if level != 1:
                cmd += ["--level", str(level)]
            env = dict(os.environ)
            if self._display is None:
                cmd.append("--headless")
                env.pop("DISPLAY", None)
            else:
                env["DISPLAY"] = self._display
            self._res["game"] = subprocess.Popen(
                cmd, env=env, stdout=subprocess.DEVNULL, preexec_fn=_die_with_parent)
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                raise RuntimeError(f"game did not connect, exit code {self._res['game'].poll()}")
        finally:
            # once connected the socket file is not needed any more
            srv.close()
            if os.path.exists(path):
                os.unlink(path)
            os.rmdir(tmp)
        conn.settimeout(None)
        self._res["conn"] = conn
        self._res["rfile"] = conn.makefile("rb")
        self._seed = seed
        self._level = level  # the process's start level (also after its game overs)
        self._fresh = True   # no state read from this process yet

    def _restart_game(self, seed, level):
        _stop_game(self._res)
        self._start_game(seed, level)

    def _recv(self):
        line = self._res["rfile"].readline()
        if not line:
            raise RuntimeError(f"game closed the connection, exit code {self._res['game'].poll()}")
        self._fresh = False
        return json.loads(line)

    # --- gym API ---------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """options={"start_level": n or "random:a-b"} overrides the config's
        start_level for this reset."""
        super().reset(seed=seed)
        spec = (options or {}).get("start_level")
        lo, hi = parse_start_level(spec) if spec is not None else self._levels
        level = lo if lo == hi else int(self.np_random.integers(lo, hi + 1))
        prefix = None
        if self._prefix_prob and self.np_random.random() < self._prefix_prob:
            k = 0 if len(self._prefix_sets) == 1 else \
                int(self.np_random.choice(len(self._prefix_sets), p=self._prefix_weights))
            ps = self._prefix_sets[k]
            prefix = ps[int(self.np_random.integers(len(ps)))]
        if prefix is not None:
            self.replay_prefix(prefix)
            return self._after_reset(int(prefix["seed"]))
        if seed is not None:
            if not (self._fresh and self._seed == seed and self._level == level):
                self._restart_game(seed, level)
        elif level != self._level or (
                not self._fresh and not (self._state and self._state["done"])):
            # another start level, or mid-game (truncated, reset early, also
            # after a hunger termination): the game cannot be reset from
            # outside, start a new process
            self._restart_game(int(self.np_random.integers(2**31 - 1)), level)
        if not self._fresh:
            # after done the game starts a new one by itself; the action
            # answering the final state is ignored by the game
            self._res["conn"].sendall(ACTIONS[4])
        self._state = self._recv()
        return self._after_reset(-1)

    def replay_prefix(self, prefix):
        """Replay a recorded opening of a game (same seed, same actions: the
        game is deterministic) and leave the env in its last state. The game
        starts on the prefix's "start_level" (default its "level", 1 if the
        file has none); the prefix ends on "level" with "lives" (default 3)
        and "food_left" food, and, where recorded, pacman on "pacman" [x, y]
        and the ghosts on "ghosts" [[x, y], ..]; a mismatch raises."""
        plevel = int(prefix.get("level", 1))
        slevel = int(prefix.get("start_level", plevel))
        if not (self._fresh and self._seed == prefix["seed"] and self._level == slevel):
            self._restart_game(prefix["seed"], slevel)
        self._state = self._recv()
        for a in prefix["actions"]:
            self._res["conn"].sendall(ACTIONS[a])
            self._state = self._recv()
        st = self._state
        left = sum(r.count(".") + r.count("o") for r in st["grid"])
        ok = left == prefix["food_left"] and st["lives"] == prefix.get("lives", 3) \
            and st["level"] == plevel
        if ok and "pacman" in prefix:
            ok = [st["pacman"]["x"], st["pacman"]["y"]] == list(prefix["pacman"])
        if ok and "ghosts" in prefix:
            ok = [[g["x"], g["y"]] for g in st["ghosts"]] == [list(g) for g in prefix["ghosts"]]
        if not ok:
            raise RuntimeError(f"prefix replay of seed {prefix['seed']} diverged")

    def set_start_level(self, spec):
        """New start_level (int or "random:a-b") from the next reset on."""
        self._levels = parse_start_level(spec)

    def _after_reset(self, prefix_seed):
        """prefix_seed: the replayed prefix's game seed, -1 without a prefix
        (always an int: gymnasium's vector envs stack info values)."""
        self._steps = 0
        self._hungry = 0  # steps since pacman last ate a dot or an energizer
        self._phi = self._potential(self._state) if self._shaping else 0.0
        info = self._info(self._state)
        info["prefix"] = prefix_seed
        return self._obs(self._state), info

    def step(self, action):
        if self._state is None:
            raise RuntimeError("step() before reset()")
        action = int(action)
        if not 0 <= action < self.action_space.n:
            raise ValueError(f"action {action} not in Discrete({self.action_space.n})")
        self._res["conn"].sendall(ACTIONS[action])
        prev, st = self._state, self._recv()
        self._state = st
        self._steps += 1

        events = self._events(prev, st)
        coef = self.cfg["reward"]
        reward = sum(coef[k] * n for k, n in events.items())
        endgame = 0.0
        if self._endgame_k and events["eaten_dot"]:
            # each dot is worth eaten_dot * (1 + k / max(food left, 1))
            left = sum(r.count(".") + r.count("o") for r in st["grid"])
            endgame = coef["eaten_dot"] * events["eaten_dot"] * self._endgame_k / max(left, 1)
            reward += endgame
        terminated = bool(st["done"])
        hunger = hunger_trunc = False
        ate = events["eaten_dot"] or events["eaten_energizer"]
        self._hungry = 0 if ate else self._hungry + 1
        if self._hunger_limit:
            if self._hungry >= self._hunger_limit and not terminated:
                # "terminate": ends the episode like a death; "truncate": a
                # time limit (truncated, no penalty: the value bootstraps).
                # The game itself goes on and is restarted by the next reset()
                hunger = True
                if self._hunger_mode == "truncate":
                    hunger_trunc = True
                else:
                    terminated = True
                    reward += coef["death"]
        shaping = self._shape(st, terminated) if self._shaping else 0.0
        reward += shaping
        limit = self.cfg["max_episode_steps"]
        truncated = not terminated and (hunger_trunc or (limit is not None
                                                         and self._steps >= limit))
        info = self._info(st)
        info["events"] = events
        info["shaping"] = shaping
        info["endgame_bonus"] = endgame
        info["hunger"] = hunger
        return self._obs(st), float(reward), terminated, truncated, info

    def close(self):
        self._finalizer()

    # --- state -> reward / obs -------------------------------------------

    def _shape(self, st, terminated):
        """gamma * Phi(s') - Phi(s), with Phi(terminal) = 0 (Ng et al. 1999)."""
        phi = 0.0 if terminated else self._potential(st)
        f = self._shaping_gamma * phi - self._phi
        self._phi = phi
        return f

    def _potential(self, st):
        """Phi(s) = k_dist * (D_max - d), d = BFS steps from pacman to the
        nearest dot or energizer (walls and the gate are not passable);
        0 without food. Phi >= 0, so standing still costs (gamma - 1) * Phi."""
        cells = "".join(st["grid"])
        layout = cells.replace(".", " ").replace("o", " ")
        nbrs = self._neighbours.get(layout)
        if nbrs is None:
            nbrs = self._neighbours[layout] = _passable_neighbours(cells)
        p = st["pacman"]
        start = p["y"] * WIDTH + p["x"]
        seen = {start}
        q = deque([(start, 0)])
        while q:
            c, d = q.popleft()
            if cells[c] in ".o":
                return self._k_dist * (self._d_max - d)
            for nb in nbrs[c]:
                if nb not in seen:
                    seen.add(nb)
                    q.append((nb, d + 1))
        return 0.0

    @staticmethod
    def _events(prev, st):
        """What happened between two consecutive states (counts)."""
        ev = dict.fromkeys(EVENTS, 0)
        ev["step"] = 1
        if st["level"] == prev["level"]:
            # the board is rebuilt on a new level; the first tick of a level
            # never eats (pacman starts still)
            before, after = "".join(prev["grid"]), "".join(st["grid"])
            ev["eaten_dot"] = before.count(".") - after.count(".")
            ev["eaten_energizer"] = before.count("o") - after.count("o")
        else:
            ev["level_up"] = st["level"] - prev["level"]
        # a ghost turns into eyes only by being eaten (normally from hunted;
        # also from normal when the energizer and the ghost meet in one tick)
        ev["ghost_eaten"] = sum(a["state"] != "eyes" and b["state"] == "eyes"
                                for a, b in zip(prev["ghosts"], st["ghosts"]))
        ev["death"] = max(0, prev["lives"] - st["lives"])
        b, p = prev["bonus"], st["pacman"]
        ev["bonus_eaten"] = int(b is not None and st["bonus"] is None
                                and (b["x"], b["y"]) == (p["x"], p["y"]))
        return ev

    def _obs(self, st):
        cells = np.frombuffer("".join(st["grid"]).encode(), np.uint8).reshape(HEIGHT, WIDTH)
        grid = np.zeros((len(CHANNELS) + len(self.grid_extras), HEIGHT, WIDTH), np.float32)
        grid[CH["walls"]] = cells == ord("#")
        grid[CH["gate"]] = cells == ord("-")
        grid[CH["food"]] = cells == ord(".")
        grid[CH["superfood"]] = cells == ord("o")
        p = st["pacman"]
        grid[CH["pacman"], p["y"], p["x"]] = 1
        if p["dir"] in DIR_OFFSET:
            grid[CH["pacman_up"] + DIR_OFFSET[p["dir"]], p["y"], p["x"]] = 1
        if p["try_dir"] in DIR_OFFSET:
            grid[CH["try_up"] + DIR_OFFSET[p["try_dir"]], p["y"], p["x"]] = 1
        for g in st["ghosts"]:
            grid[CH["ghost_" + g["state"]], g["y"], g["x"]] += 1 / GHOSTS
            if g["dir"] in DIR_OFFSET:
                grid[CH["ghost_up"] + DIR_OFFSET[g["dir"]], g["y"], g["x"]] += 1 / GHOSTS
        if st["bonus"]:
            grid[CH["bonus"], st["bonus"]["y"], st["bonus"]["x"]] = 1
        if self._food_distance:
            grid[FOOD_DISTANCE_CH] = self._food_field(st)
        if self._cell_safety or self._safe_exits:
            safe = self._safety_field(st)
            i = len(CHANNELS) + self._food_distance
            if self._cell_safety:
                grid[i] = safe
                i += 1
            if self._safe_exits:
                grid[i] = self._safe_exits_field(st, safe)
        vec = [min(st["lives"], MAX_LIVES) / 3, st["supertime_left"] / SUPERTIME,
               min(st["level"], LEVELS) / LEVELS]
        if self._steps_since_food:
            vec.append(min(self._hungry / HUNGER_SCALE, 1.0))
        return {"grid": grid, "vec": np.array(vec, np.float32)}

    def _food_field(self, st):
        """Food distance plane / D_max; recomputed only when the board changes."""
        cells = "".join(st["grid"])
        if self._field_cache[0] != cells:
            layout = cells.replace(".", " ").replace("o", " ")
            nbrs = self._neighbours.get(layout)
            if nbrs is None:
                nbrs = self._neighbours[layout] = _passable_neighbours(cells)
            dist = food_distance_field(cells, nbrs).astype(np.float32)
            field = (dist / np.float32(self._d_max)).reshape(HEIGHT, WIDTH)
            self._field_cache = (cells, field)
        return self._field_cache[1]

    def _safety_field(self, st):
        """Cell safety plane (cell_safety_field) of a state."""
        cells = "".join(st["grid"])
        layout = cells.replace(".", " ").replace("o", " ")
        nb = self._neighbours.get(("safety", layout))
        if nb is None:
            nb = self._neighbours[("safety", layout)] = (_passable_neighbours(layout),
                                                         _ghost_neighbours(layout))
        p = st["pacman"]
        ghosts = [g["y"] * WIDTH + g["x"] for g in st["ghosts"] if g["state"] == "normal"]
        field = cell_safety_field(nb[0], nb[1], p["y"] * WIDTH + p["x"], ghosts,
                                  st["supertime_left"] > 0)
        return field.reshape(HEIGHT, WIDTH)

    def _safe_exits_field(self, st, safe):
        """Safe exits plane (safe_exits_field) of a state, from its cell-safety plane."""
        cells = "".join(st["grid"])
        layout = cells.replace(".", " ").replace("o", " ")
        nb = self._neighbours[("safety", layout)]
        p = st["pacman"]
        return safe_exits_field(nb[0], safe.reshape(-1), p["y"] * WIDTH + p["x"]).reshape(
            HEIGHT, WIDTH)

    @staticmethod
    def _info(st):
        return {"score": st["score"], "lives": st["lives"], "level": st["level"]}
