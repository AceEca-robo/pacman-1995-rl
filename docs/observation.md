# PacmanEnv observation

`Pacman1995-v0` (`env/pacman_env.py`) returns a `Dict` observation:

| key    | shape          | dtype   | range  |
|--------|----------------|---------|--------|
| `grid` | (17, 23, 33)   | float32 | [0, 1] |
| `vec`  | (3,)           | float32 | see below |

The board is 23 rows x 33 columns (`BOARDHEIGHT`/`BOARDWIDTH` in
`game/sizes.h`); `grid[c, y, x]` is cell (x, y) of channel c, y down.
Every second column/row of the original maze strings is a "between" cell,
so pacman and ghosts walk through blank cells between the dots.

## grid channels

| # | name           | value in a cell |
|---|----------------|-----------------|
| 0 | `walls`        | 1 if wall (`#`) |
| 1 | `gate`         | 1 if ghost house gate (`-`): pacman cannot pass, ghosts can |
| 2 | `food`         | 1 if a dot (`.`) |
| 3 | `superfood`    | 1 if superfood (`o`) |
| 4 | `pacman`       | 1 at pacman |
| 5 | `pacman_up`    | 1 at pacman if it is moving up |
| 6 | `pacman_down`  | same, down |
| 7 | `pacman_left`  | same, left |
| 8 | `pacman_right` | same, right |
| 9 | `ghost_normal` | (ghosts in state normal in the cell) / 4 |
| 10 | `ghost_hunted` | (ghosts in state hunted in the cell) / 4 |
| 11 | `ghost_eyes`  | (eaten ghosts in the cell) / 4 |
| 12 | `ghost_up`    | (ghosts in the cell moving up) / 4 |
| 13 | `ghost_down`  | same, down |
| 14 | `ghost_left`  | same, left |
| 15 | `ghost_right` | same, right |
| 16 | `bonus`       | 1 at the bonus (points or extra life, not distinguished) |

Notes:

- Directions come from the game's `dir` field. `S` (still: pacman blocked by
  a wall or just respawned, eaten ghosts) sets no direction plane, so all
  four direction planes are 0 there.
- Pacman's direction is what it is actually doing: after a blocked move it
  is still, not the requested direction.
- Ghost state `normal` covers both the game's random-walk and hunting
  states. `eyes` ghosts sit on their start cell in the ghost house until
  they revive.
- Ghost direction planes do not say which ghost (state) moves where; with
  several ghosts in one cell only the sums are known.

## vec

| # | value | range |
|---|-------|-------|
| 0 | `min(lives, 9) / 3` | [0, 3] |
| 1 | `supertime_left / 50` (`SUPERTIME` in `game/pac.h`), 0 when not super | [0, 1] |
| 2 | `min(level, 16) / 16` (16 = `LEVELS`; boards repeat randomly after that) | (0, 1] |

## Timing

One step = one game tick that reads input. While pacman is super it moves
twice per ghost move, so ghosts move every other step then.
