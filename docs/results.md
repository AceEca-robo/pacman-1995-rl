# Baseline results

`scripts/evaluate.py`, environment `configs/env_default.yaml`
(episode = one game of 3 lives, truncated at `max_episode_steps`).
"Level 1 cleared" = share of episodes that reached level 2.
Length is in env steps (game ticks).

| agent | episodes | seeds | mean score | median score | max score | level 1 cleared | mean length | truncated | time, s | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| random | 100 | 0..99 | 92 | 80 | 310 | 0% | 129 | 0% | 1.3 | 24f3bda |
| heuristic | 100 | 0..99 | 16999 | 11825 | 221375 | 82% | 1244 | 0% | 16.4 | 24f3bda |

## Notes

- Scores have a heavy tail from an original game rule: the ghost score
  doubles with every ghost eaten (100 * 2^(k-1)) and the counter is only
  reset when pacman stops being super. Pacman's supertime runs at two ticks
  per ghost tick, so eaten ghosts revive (DEADTIME 25 ghost ticks) while
  pacman is still super, and eating the next superfood extends it. The
  heuristic's best episode (seed 42, 221375 points) got 102400 for a single
  ghost. Median is the more telling number.
- The heuristic's level transitions confirm `level_bonus`: e.g. 1 -> 2 gives
  reward 0.1 * 2415 (score: food + remaining level bonus) + 50 - 0.01 = 291.49.
