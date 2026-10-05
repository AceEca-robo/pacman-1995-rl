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
| dqn dqn0/best.pt | 100 | 0..99 | 61.9 | 61.8 | 1423 | 1360 | 2950 | 0% | 0% | 848 | 0% | 30.4 | 5d22ce4 |
| dqn dqn2/best.pt | 100 | 0..99 | 56.5 | 59.2 | 1483 | 1420 | 2670 | 0% | 0% | 1088 | 0% | 72.9 | 7f96361 |

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
- `dqn dqn0/best.pt`: Double + Dueling DQN, 3-step, `configs/dqn.yaml` as of
  fa5a7f5, seed 0, 5M env steps (~41 min at 1.8-3.5k steps/s, median 2.6k).
  best.pt is the 4.75M checkpoint (median reward 62.9 on the 20 training
  eval seeds 1000000..1000019); on the 100 seeds above it gets the same,
  so no overfitting to the eval seeds. Eval reward was still rising slowly
  at 5M (curves in `docs/eval.png`, `docs/training.png`). It has never
  cleared level 1. Over 30 games (seeds 0..29) it eats a median of 124 of
  the 168 dots (min 120, max 132), all 4 energizers and ~1 ghost, then
  loses its last life: the remaining ~40 dots are the hard part.
- `dqn dqn2/best.pt`: dqn0's setup plus 200k heuristic steps in the replay
  buffer before training (`configs/dqn_warm.yaml`), seed 0, 5M steps, run in
  parallel with dqn1. best.pt = 4.5M checkpoint. Ends level with dqn0
  (median reward 59.2 vs 61.8, score 1420 vs 1360, within seed noise for one
  seed each) but was much worse early: at 1M median eval reward -49.8 vs
  -12.3. The heuristic transitions first pushed mean Q to +13 while the
  agent still played randomly, and loss went up to 3.7 while its own data
  corrected that. Its games are longer (1088 vs 848 steps) for the same
  reward: more time spent without eating.
