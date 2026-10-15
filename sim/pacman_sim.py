"""PacmanSim: a clonable port of the game (game/), state for state.

The reference implementation in plain Python; sim/csim.py is the same
machine in C (sim/_csim.c). See docs/sim_spec.md for the rules. Names
follow the C++ sources (pac.cc main loop, pacman.cc, ghost.cc, ...).

One step() is one exchange with the game over the RL bridge: the action
answers the last state, the result is the next state the game sends (two
per main-loop tick while pacman is super). reset() returns the first
state, like PacmanEnv.reset().
"""

import os
import re

from .glibc_random import GlibcRandom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WIDTH, HEIGHT = 33, 23          # sizes.h
LEVELS = 16
GHOSTS = 4
SUPERTIME, DEADTIME, BONUSTIME = 50, 25, 50
FOODS = 168                     # board.h: food per board
FOODSCORE, GHOSTSCORE, BONUSSCORE = 10, 100, 1000
PACX, PACY = 16, 17
BONUSX, BONUSY = 16, 15
GHOST_START = ((15, 9), (17, 9), (15, 11), (17, 11))

UP, DOWN, LEFT, RIGHT, NONE, STILL = range(6)               # direc.h
GATE, WALL, FOOD, SUPERFOOD, BLANK = range(5)               # types.h
RANDM, HUNTER, HUNTED, EATEN = range(4)                     # ghost.h
DEAD, NORMAL, SUPER = range(3)                              # pacman.h
NO_BONUS, BONUS_POINT, BONUS_LIFE = range(3)

DX = (0, 0, -1, 1, 0, 0)
DY = (-1, 1, 0, 0, 0, 0)
# relative left / right of up, down, left, right (ghost.cc)
LEFT_OF = (LEFT, RIGHT, UP, DOWN)
RIGHT_OF = (RIGHT, LEFT, DOWN, UP)
# env action index (U, D, L, R, N) -> direction; "N" is none
ACTION_DIR = (UP, DOWN, LEFT, RIGHT, NONE)

DIRCHAR = "UDLRSS"
CELLCHAR = "-#.o "
GHOST_STATE = ("normal", "normal", "hunted", "eyes")


def game_seed(seed):
    """PacmanEnv's --seed for an env seed (glibc treats srandom(0) as 1)."""
    return seed % (2**31 - 1) + 1


def load_boards(path=os.path.join(ROOT, "game", "board.h")):
    """The 16 mazes from board.h as lists of 23 strings of 33."""
    src = open(path).read()
    start = src.index("boards[LEVELS]")
    rows = re.findall(r'"([^"]*)"', src[start:src.index("};", start)])
    assert len(rows) == LEVELS * HEIGHT and all(len(r) == WIDTH for r in rows)
    return [rows[k * HEIGHT:(k + 1) * HEIGHT] for k in range(LEVELS)]


_CODE = {"O": WALL, ".": FOOD, "o": SUPERFOOD, "_": GATE, " ": BLANK}
BOARDS = load_boards()
# per maze: cells as bytes, index y * WIDTH + x
BOARD_CELLS = [bytes(_CODE[c] for row in b for c in row) for b in BOARDS]
_GRID_TABLE = bytes.maketrans(bytes(range(5)), CELLCHAR.encode())


class Ghost:
    __slots__ = ("x", "y", "sx", "sy", "gox", "goy", "st", "d", "deadtime", "randmv")

    def __init__(self, sx, sy, randmv):
        self.x, self.y, self.sx, self.sy = sx, sy, sx, sy
        self.gox = self.goy = 0
        self.st, self.d, self.deadtime, self.randmv = RANDM, UP, 0, randmv

    def copy(self):
        g = Ghost.__new__(Ghost)
        g.x, g.y, g.sx, g.sy, g.gox, g.goy = self.x, self.y, self.sx, self.sy, self.gox, self.goy
        g.st, g.d, g.deadtime, g.randmv = self.st, self.d, self.deadtime, self.randmv
        return g


class PacmanSim:
    def __init__(self, seed=None, level=1):
        if seed is not None:
            self.reset(seed, level)

    # --- public API ------------------------------------------------------

    def reset(self, seed=0, level=1, raw_seed=None):
        """A fresh game process: seed as for PacmanEnv (env seed), or
        raw_seed = the game's --seed. Returns the first state."""
        self.rng = GlibcRandom(game_seed(seed) if raw_seed is None else raw_seed)
        self.start_level = level
        rnd = self.rng.random
        # Ghost constructors: randmv drawn once per process
        self.ghosts = [Ghost(sx, sy, rnd() % 8 + 4) for sx, sy in GHOST_START]
        self.ghostseaten = 0
        self.bon, self.bon_life = NO_BONUS, 0
        self.supertime = 0
        self._gd_start()
        self._level_start()
        self._tick()
        self.phase = 1
        return self.state()

    def step(self, action):
        """action: env index 0..4 = U, D, L, R, N. Returns the next state."""
        self._advance(ACTION_DIR[action])
        return self.state()

    def advance(self, action):
        """step() without building the state dict."""
        self._advance(ACTION_DIR[action])

    def clone(self):
        c = PacmanSim.__new__(PacmanSim)
        c.__dict__.update(self.__dict__)
        c.rng = self.rng.copy()
        c.cells = bytearray(self.cells)
        c.ghosts = [g.copy() for g in self.ghosts]
        return c

    @property
    def done(self):
        return self.lives == 0

    def grid(self):
        cells = bytes(self.cells).translate(_GRID_TABLE).decode()
        return [cells[j * WIDTH:(j + 1) * WIDTH] for j in range(HEIGHT)]

    def state(self):
        """The state in the format of the RL bridge's JSON (json.loads)."""
        return {
            "grid": self.grid(),
            "pacman": {"x": self.px, "y": self.py, "dir": DIRCHAR[self.pd],
                       "try_dir": DIRCHAR[self.try_d]},
            "ghosts": [{"x": g.x, "y": g.y, "dir": DIRCHAR[g.d], "state": GHOST_STATE[g.st]}
                       for g in self.ghosts],
            "bonus": None if self.bon == NO_BONUS else
            {"x": BONUSX, "y": BONUSY, "type": "point" if self.bon == BONUS_POINT else "life"},
            "score": self.score, "lives": self.lives, "level": self.level,
            "supertime_left": self.supertime if self.pst == SUPER else 0,
            "done": self.lives == 0,
        }

    def state_json(self):
        """The exact line rlbridge.cc sends, as bytes (with the newline)."""
        grid = ",".join('"%s"' % r for r in self.grid())
        ghosts = ",".join('{"x":%d,"y":%d,"dir":"%s","state":"%s"}'
                          % (g.x, g.y, DIRCHAR[g.d], GHOST_STATE[g.st]) for g in self.ghosts)
        if self.bon == NO_BONUS:
            bonus = "null"
        else:
            bonus = '{"x":%d,"y":%d,"type":"%s"}' % (
                BONUSX, BONUSY, "point" if self.bon == BONUS_POINT else "life")
        return ('{"grid":[%s],"pacman":{"x":%d,"y":%d,"dir":"%s","try_dir":"%s"},'
                '"ghosts":[%s],"bonus":%s,"score":%d,"lives":%d,"level":%d,'
                '"supertime_left":%d,"done":%s}\n' % (
                    grid, self.px, self.py, DIRCHAR[self.pd], DIRCHAR[self.try_d],
                    ghosts, bonus, self.score, self.lives, self.level,
                    self.supertime if self.pst == SUPER else 0,
                    "true" if self.lives == 0 else "false")).encode()

    # --- main loop (pac.cc) ---------------------------------------------

    def _advance(self, d):
        self.d = d
        if self.phase == 1:
            self._bonus_tick()
            if self.pst == SUPER:       # the extra run while super
                self._pac_go(self.d)
                self._check()
                self.phase = 2
                return
        if self.pst == DEAD:
            self.bon = NO_BONUS
            self._pac_start()
            for g in self.ghosts:
                self._ghost_start(g)
        if self.bonus:
            self.bonus -= 5             # bondwn
        if not (self.lives and self.eats):
            if self.eats:               # game over: a new game in this process
                self._gd_start()
            else:                       # level cleared
                self.score += self.bonus
                self._levelup()
                self.bonus, self.eats = 5000, FOODS
                if not self.lives:      # died on the last dot: outer loop
                    self._gd_start()
            self._level_start()
        self._tick()
        self.phase = 1

    def _tick(self):
        for g in self.ghosts:
            self._lookforpac(g)
        self._pac_go(self.d)
        for g in self.ghosts:
            self._ghost_go(g)
        self._check()

    def _bonus_tick(self):
        rnd = self.rng.random
        if self.bon == NO_BONUS and (rnd() & 4095) == 0:
            self.bon = BONUS_POINT if rnd() & 61440 else BONUS_LIFE
            self.bon_life = BONUSTIME
        if self.bon != NO_BONUS:
            self.bon_life -= 1
            if self.bon_life + 1 == 0:
                self.bon = NO_BONUS

    def _check(self):
        px, py = self.px, self.py
        if self.bon != NO_BONUS and px == BONUSX and py == BONUSY:
            if self.bon == BONUS_POINT:
                self.score += BONUSSCORE
            else:
                self.lives += 1
            self.bon = NO_BONUS
        for g in self.ghosts:
            if g.x == px and g.y == py:
                if self.pst == SUPER:
                    self._ghost_die(g)
                else:
                    self.pst = DEAD
                    if self.lives:
                        self.lives -= 1
                    break

    # --- gamedata.cc / board.cc -------------------------------------------

    def _gd_start(self):
        self.level, self.lives, self.score, self.bonus, self.eats = 1, 3, 0, 5000, FOODS
        self._setboardlevel()
        if self.start_level:
            self.level = self.start_level
            self._setboardlevel()

    def _levelup(self):
        self.level += 1
        self._setboardlevel()

    def _setboardlevel(self):
        if self.level > LEVELS:
            self.boardlevel = 1 + self.rng.random() % LEVELS
        else:
            self.boardlevel = self.level

    def _level_start(self):
        self.cells = bytearray(BOARD_CELLS[self.boardlevel - 1])
        self._pac_start()
        for g in self.ghosts:
            self._ghost_start(g)
        self.d = STILL

    # --- pacman.cc --------------------------------------------------------

    def _pac_start(self):
        self.px, self.py = PACX, PACY
        self.pst, self.pd, self.try_d, self.supertime = NORMAL, STILL, RIGHT, 0

    def _pac_go(self, dd):
        if dd != NONE:
            self.pd = self.try_d = dd
        if self.pst == DEAD:
            self.pd = STILL
            return
        if self.pst == SUPER:
            if self.supertime:
                self.supertime -= 1
            else:
                self.pst = NORMAL
        d = self.pd
        xx, yy = self.px + DX[d], self.py + DY[d]
        k = yy * WIDTH + xx
        w = self.cells[k]
        if w == WALL or w == GATE:
            self.pd = STILL
        elif w == FOOD:
            self.px, self.py = xx, yy
            self.cells[k] = BLANK
            if self.eats:
                self.eats -= 1
            else:
                raise RuntimeError("food")   # pacexit("food")
            self.score += FOODSCORE
        elif w == SUPERFOOD:
            self.px, self.py = xx, yy
            self.cells[k] = BLANK
            self.pst, self.supertime = SUPER, SUPERTIME
        else:
            self.px, self.py = xx, yy

    # --- ghost.cc ---------------------------------------------------------

    def _ghost_start(self, g):
        g.x, g.y, g.st, g.d = g.sx, g.sy, RANDM, UP
        self.ghostseaten = 0

    def _ghost_die(self, g):
        self.ghostseaten += 1
        self.score += GHOSTSCORE << (self.ghostseaten - 1)
        g.deadtime, g.st, g.x, g.y, g.d = DEADTIME, EATEN, g.sx, g.sy, STILL

    def _wall(self, x, y):
        return self.cells[y * WIDTH + x] == WALL

    def _otherdir(self, g):
        d = g.d
        if d > RIGHT:       # uninitialized sides: both are the ghost's cell
            return True
        a, b = LEFT_OF[d], RIGHT_OF[d]
        return not self._wall(g.x + DX[a], g.y + DY[a]) or \
            not self._wall(g.x + DX[b], g.y + DY[b])

    def _newdir(self, g):
        d = g.d
        if d > RIGHT:       # see docs/sim_spec.md: one draw, no direction
            self.rng.random()
            g.d = NONE
            return
        a, b = LEFT_OF[d], RIGHT_OF[d]
        if self._wall(g.x + DX[a], g.y + DY[a]):
            g.d = NONE if self._wall(g.x + DX[b], g.y + DY[b]) else b
        elif self._wall(g.x + DX[b], g.y + DY[b]):
            g.d = a
        else:
            g.d = b if self.rng.random() % 2 else a

    def _searchpac(self, g):
        px, py = self.px, self.py
        x, y = g.x, g.y
        if px != x and py != y:
            return False
        dx = (px > x) - (px < x)
        dy = (py > y) - (py < y)
        cells = self.cells
        while not (x == px and y == py) and cells[y * WIDTH + x] != WALL:
            x += dx
            y += dy
        if not (x == px and y == py):
            return False
        g.gox, g.goy = px, py
        if g.st == HUNTED:
            dx, dy = -dx, -dy
        if dx == 0 and dy == -1:
            g.d = UP
        elif dx == 0 and dy == 1:
            g.d = DOWN
        elif dx == 1 and dy == 0:
            g.d = RIGHT
        elif dx == -1 and dy == 0:
            g.d = LEFT
        return True

    def _lookforpac(self, g):
        if self.pst == SUPER:
            if g.st != EATEN:
                g.st = HUNTED
        elif g.st == HUNTED:
            g.st = RANDM
            self.ghostseaten = 0
        st = g.st
        if st == HUNTER:
            if self._searchpac(g) and g.x == self.px and g.y == self.py:
                g.d = STILL
        elif st == RANDM:
            if self._searchpac(g):
                g.st = HUNTER
        elif st == HUNTED:
            self._searchpac(g)

    def _ghost_go(self, g):
        if self.pst == SUPER:
            if g.st != EATEN:
                g.st = HUNTED
        elif g.st == HUNTED:
            g.st = RANDM
        st = g.st
        if st == HUNTER:
            if self._searchpac(g):
                if g.x == self.px and g.y == self.py:
                    g.d = STILL
            elif g.gox == g.x and g.goy == g.y:
                self._newdir(g)
                g.st = RANDM
        elif st == RANDM:
            if self._searchpac(g):
                g.st = HUNTER
            elif self._otherdir(g) and self.rng.random() % g.randmv == 0:
                self._newdir(g)
        elif st == HUNTED:
            self._searchpac(g)
        else:                           # eaten
            if g.deadtime:
                g.deadtime -= 1
            else:
                g.st, g.d = RANDM, UP
        d = g.d
        xx, yy = g.x + DX[d], g.y + DY[d]
        if self._wall(xx, yy):
            self._newdir(g)
            d = g.d
            xx, yy = g.x + DX[d], g.y + DY[d]
        g.x, g.y = xx, yy
