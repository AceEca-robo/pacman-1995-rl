# Baseline results

`scripts/evaluate.py`, environment `configs/env_default.yaml`
(episode = one game of 3 lives, truncated at `max_episode_steps`).
Reward is the env's event reward (see docs/observation.md), score the game's.
"Level 1 cleared" = share of episodes that reached level 2; "without death"
= reached it before losing any life. Length is in env steps (game ticks).

| agent | episodes | seeds | mean reward | median reward | mean score | median score | max score | level 1 cleared | level 1 without death | mean length | truncated | time, s | commit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dqn dqn5/best_food.pt, eps 0 | 100 | 0..99 | 20.4 | 45.8 | 2426 | 1950 | 7480 | 0% | 0% | 3623 | 20% | 130.6 | c6a6df9 |
| dqn dqn5/best_food.pt, eps 0.05 | 100 | 0..99 | 67.0 | 68.0 | 1943 | 1900 | 4300 | 0% | 0% | 844 | 0% | 32.3 | c6a6df9 |
| dqn dqn5/best_food.pt, no ghosts, eps 0 | 3 | 0..2 | -94.0 | -94.0 | 980 | 980 | 980 | 0% | 0% | 10000 | 100% | 10.6 | c6a6df9 |
| dqn dqn5/best_food.pt, no ghosts, eps 0.05 | 20 | 0..19 | -82.6 | -82.0 | 1344 | 1105 | 3070 | 0% | 0% | 10000 | 100% | 68.7 | c6a6df9 |
