# Game rules as implemented (spec for sim/)

What `game/` actually does, tick by tick, written down so that `sim/` can
repeat it state for state. Everything below is read from the sources
(`pac.cc` main loop, `pacman.cc`, `ghost.cc`, `gamedata.cc`, `board.h/.cc`,
`bonus*.cc`, `rlbridge.cc`) and, where the C++ has undefined behaviour, from
the compiled object code (`objdump -d ghost.o`, g++ -O2).

## Board

- 16 mazes, `boards[16][23][33]` in `board.h`; x = column 0..32, y = row
  0..22. Codes: `O` wall, `.` food, `o` superfood (energizer), `_` gate
  ("special wall"), ` ` blank.
- Every maze has exactly 168 food and 4 superfood, a closed border (no
  tunnels, no out-of-range reads) and no dead-end cells.
- The gate (row 8, x 15..17) is the top of the ghost house (x 15..17,
  y 9..11). Pacman treats the gate as a wall; ghosts walk through it
  (only `classWall` blocks a ghost, also in the line-of-sight test).
- Eating a cell replaces it with blank. The board is rebuilt only when a
  level starts (new game or level cleared), not after a death.
- JSON grid characters: `#` wall, `.` food, `o` superfood, `-` gate, ` `.

## Game data (`gamedata.cc`)

- `start()` (new game): level 1, lives 3, score 0, bonus 5000, eats 168.
  `--level n` then calls `setlevel(n)`.
- `setboardlevel()`: boardlevel = level for level <= 16, otherwise
  `1 + random() % 16` (one `random()` call per level start beyond 16).
- Food: `foodeat()` (eats -= 1) and +10 score. Superfood gives no score
  and does not count: a level is cleared when 168 dots are eaten (eats ==
  0), whatever superfood is left.
- `bondwn()`: bonus -= 5 once per main-loop tick (not per sub-step) while
  bonus > 0. On a cleared level `score += bonus`, then `start2()`:
  level += 1 (setboardlevel), bonus 5000, eats 168. Lives and score stay.
- Ghost score: the n-th ghost eaten in one super phase gives
  100 * 2^(n-1) (`ghostseaten` is static, shared by all ghosts; reset by
  any `Ghost::start()` and by `lookforpac` when a hunted ghost turns
  random). Bonus point +1000, bonus life +1 life.

## Pacman (`pacman.cc`)

- start: (16, 17), state normal, d = still, try_d = right, supertime 0.
- `go(dd)`: if dd != none then d = try_d = dd (dd = still is possible: the
  main loop's d is still at the start of each level). If dead: d = still,
  return. If super: supertime > 0 ? supertime -= 1 : state = normal (and
  this go still moves). Then the next cell in d: wall or gate -> d = still,
  no move; food -> move, eat, eats -= 1, +10; superfood -> move, eat,
  state = super, supertime = 50; blank -> move. There is no turn buffer:
  an action that points into a wall stops pacman.
- "N" (none) keeps d; after a stop d is still, so N keeps pacman standing.
- `die()`: state dead, lives -= 1 (if > 0).

## Ghosts (`ghost.cc`)

States: randm, hunter, hunted, eaten. Start cells (15,9), (17,9), (15,11),
(17,11); start: state randm, d = up, `ghostseaten = 0`. Each ghost has a
fixed `randmv = random() % 8 + 4`, drawn once in the constructor (process
start), not on game or level restart.

- `next(d)`: up y-1, down y+1, left x-1, right x+1; none/still (and any
  other value) stay.
- `searchpac(d)`: if pacman shares the ghost's row or column, step from the
  ghost's cell towards pacman while the cell is not a wall; if pacman is
  reached: go = pacman's cell, d = the direction towards pacman (reversed
  if the ghost is hunted), return 1. Same cell: d unchanged, return 1.
- `newdir(d)` (as compiled): left/right relative to d (up: left/right,
  down: right/left, left: up/down, right: down/up). Left wall -> d = right
  (none if right is a wall too). Else right wall -> d = left. Else
  `random() % 2` ? right : left (one `random()` call).
- `otherdir(d)`: true if the relative left or right cell is not a wall.
- `lookforpac(pac)` (before pacman moves):
  - pacman super: state != eaten -> hunted. Not super: hunted -> randm and
    ghostseaten = 0.
  - hunter: searchpac; found on the same cell -> d = still.
  - randm: searchpac found -> hunter.
  - hunted: searchpac. eaten: nothing.
- `go(pac)` (after pacman's first move):
  - pacman super: state != eaten -> hunted. Not super: hunted -> randm
    (ghostseaten is not reset here).
  - hunter: searchpac found -> (same cell -> d = still). Not found and the
    ghost is on `go` -> newdir, state randm.
  - randm: searchpac found -> hunter. Else if otherdir(d) and
    `random() % randmv == 0` -> newdir.
  - hunted: searchpac.
  - eaten: deadtime > 0 ? deadtime -= 1 : (state randm, d = up).
  - Move: if next(d) is a wall: newdir, then move to next(d) without a
    second wall test (newdir only picks open sides, or none). Else move.
- `die()`: ghostseaten += 1, score as above, deadtime = 25, state eaten,
  back to the start cell, d = still. An eaten ghost sits 25 `go`s on its
  start cell and leaves (d = up) on the 26th.

### Undefined behaviour: d = none / still in otherdir and newdir

`newdir` returns none when both sides are walls, which happens whenever a
hunter reaches the last-seen cell in a straight corridor. The ghost then
is randm with d = none. `otherdir`/`newdir` have no `case` for none/still,
so their left/right are uninitialized. In the compiled code (ghost.o) the
switch is `if (d <= 3) {left = table[d]; right = 3 - d}`, otherwise left
and right keep whatever the registers held: the low 32 bits of pointers
(the caller's r15, `this` in rbx), i.e. values > 5, for which `next()`
stays. The observable behaviour is therefore:

- `otherdir(d > 3)` is true (both "sides" are the ghost's own cell, never a
  wall), so a randm ghost with d = none still draws `random() % randmv`
  every tick;
- `newdir(d > 3)` draws one `random()` and leaves d at a non-direction
  value (reported as "S", moves nowhere).

Such a ghost stands still until searchpac sees pacman (sets a real d), a
death or level start resets it. The sim keeps d = none for it.

## Bonus (`bonus.cc`, main loop)

- Position (16, 15). One bonus object at a time (`bon`), lifetime 50.
- Each tick after the first send: if no bonus: `if ((random() & 4095) ==
  0)` spawn: `random() & 61440` != 0 -> point, else life; lifetime = 50.
  Then if a bonus exists: `if (lifetime-- == 0)` it disappears (so it
  lives for 51 of these checks, counting the spawn tick).
- Eaten in `check` when pacman's cell equals its cell (before ghosts).
- A death removes it; a level change does not.

## Main loop, one tick (`pac.cc`)

```
A  for g in ghosts: g.lookforpac(pac)
   pac.go(d)
   for g in ghosts: g.go(pac)
   check()                 # bonus, then ghosts 0..3 until pacman dies
   SEND state; d = action  # "N" -> none
B  bonus spawn / countdown (random() calls, also on a death tick)
C  if pac super:           # also if it turned super in this tick's go
     pac.go(d); check(); SEND state; d = action
D  if pac dead: bonus gone, pac.start(), all ghosts start()
E  bondwn()
   loop while lives > 0 and not all food eaten
```

`check()`: if pacman is on the bonus it is eaten. Then for each ghost on
pacman's cell: super -> the ghost dies, else pacman dies and the loop
stops. Ghosts and pacman can swap cells without meeting (pacman moves,
then the ghosts, then the check).

When the tick loop ends: if food remains (lives == 0): game over, a new
game starts in the same process (`start()`, `setlevel(--level)`), random
state continues. Else score += bonus, `start2()`; then if lives > 0 the
next level starts (if pacman died on the last dot with no lives left the
level-up still happens, then the outer loop calls `start()` again). Each
level start: board rebuilt, pac.start(), ghosts start(), d = still.

The action answering a state with lives == 0 ("done") is read but has no
effect (d is set to still before the next tick).

## random() call order

1. `srandom(seed)` (`--seed`; the env passes `seed % (2^31 - 1) + 1`).
2. 4 calls: randmv of ghosts 0..3.
3. Per tick, in this order: in A, ghosts 0..3 `go` (randm: `% randmv`
   test, then maybe newdir's `% 2`; any state: newdir at a wall with both
   sides open; hunter at its go cell: newdir); in B the bonus draw(s);
   per level start beyond 16 one call in setboardlevel. `lookforpac`
   draws nothing.

glibc `random()` (TYPE_3): r[0] = seed (0 -> 1), r[i] = 16807 * r[i-1] mod
(2^31 - 1) for i = 1..30, r[i] = r[i-31] for i = 31..33, r[i] = r[i-3] +
r[i-31] mod 2^32 after that; output k is r[k + 344] >> 1 (310 values
dropped).

## JSON state (`rlbridge.cc`)

One line per SEND: grid (23 strings of 33), pacman {x, y, dir, try_dir}
(U/D/L/R, anything else "S"), ghosts [{x, y, dir, state}] with state
"hunted", "eyes" (eaten) or "normal" (randm and hunter), bonus {x, y,
type "point"/"life"} or null, score, lives, level, supertime_left
(supertime if super else 0), done (lives == 0).
