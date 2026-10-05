# Baseline results

`scripts/evaluate.py`, environment `configs/env_default.yaml`
(episode = one game of 3 lives, truncated at `max_episode_steps`).
Reward is the env's event reward (see docs/observation.md), score the game's.
"Level 1 cleared" = share of episodes that reached level 2; "without death"
= reached it before losing any life. Length is in env steps (game ticks).

| agent | episodes | seeds | mean reward | median reward | mean score | median score | max score | level 1 cleared | level 1 without death | mean length | truncated | time, s | commit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random | 100 | 0..99 | -53.4 | -54.0 | 92 | 80 | 310 | 0% | 0% | 129 | 0% | 1.5 | bae470e |
| heuristic | 100 | 0..99 | 465.5 | 401.2 | 16999 | 11825 | 221375 | 82% | 21% | 1244 | 0% | 17.6 | bae470e |

## Notes

- Scores have a heavy tail from an original game rule: the ghost score
  doubles with every ghost eaten (100 * 2^(k-1)) and the counter is only
  reset when pacman stops being super. Pacman's supertime runs at two ticks
  per ghost tick, so eaten ghosts revive (DEADTIME 25 ghost ticks) while
  pacman is still super, and eating the next superfood extends it. The
  heuristic's best episode (seed 42, 221375 points) got 102400 for a single
  ghost. Median is the more telling number.
- Since bae470e the reward counts events (fixed 5 per ghost), so these
  chains no longer dominate the reward, only the score columns.
- Scores are identical to the previous (score-delta reward) measurement:
  the reward and observation changes did not change the games played.
