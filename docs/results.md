# Baseline results

`scripts/evaluate.py`, environment `configs/env_default.yaml`
(episode = one game of 3 lives, truncated at `max_episode_steps`).
Reward is the env's event reward (see docs/observation.md), score the game's.
"Level 1 cleared" = share of episodes that reached level 2; "without death"
= reached it before losing any life. Length is in env steps (game ticks).

<!-- summary:begin -->
## Summary of all runs

Final evaluation of each run's chosen checkpoint: 100 games, seeds 0..99, eps 0, no shaping or hunger_limit (dqn5 on: best_food.pt, the checkpoint with the most food in the supervisor's 20-game evaluations). Food = dots + energizers eaten per game (172 on level 1). Stuck = games that hit the 10000-step limit. Last column: the same checkpoint with eps 0.05.

| run | checkpoint | change | mean food | level 1 cleared | mean reward | median score | max score | stuck | steps after last food | eps 0.05: food / reward |
|---|---|---|---|---|---|---|---|---|---|---|
| dqn0 | best.pt (4.75M) | baseline DQN, 5M | 128.3 | 0% | 61.9 | 1360 | 2950 | 0% | 69 | - |
| dqn2 | best.pt (4.5M) | + heuristic warm start | 125.6 | 0% | 56.5 | 1420 | 2670 | 0% | 236 | - |
| dqn1 | best.pt (11.0M, by mean) | 20M steps, eps over 3M | 150.7 | 0% | 65.7 | 1605 | 2990 | 3% | 378 | - |
| dqn3b | best.pt (4.75M) | + distance shaping, 4 actions | 126.8 | 0% | 62.2 | 1490 | 3070 | 0% | 212 | - |
| dqn4 | best.pt (5.0M) | + hunger_limit 200, 4 actions | 112.9 | 0% | 43.0 | 1285 | 2450 | 0% | 172 | - |
| ppo0 | best.pt (2.0M) | PPO, hunger_limit 200, 4 actions | 99.8 | 0% | 35.7 | 1020 | 1330 | 0% | 6 | - |
| dqn5 | best_food.pt (10.0M) | dqn1 + food-distance channel + steps since food, 10M | 127.4 | 0% | 20.4 | 1950 | 7480 | 20% | 602 | 125.6 / 67.0 |
<!-- summary:end -->

## All evaluations

| agent | episodes | seeds | mean reward | median reward | mean score | median score | max score | level 1 cleared | level 1 without death | mean length | truncated | time, s | commit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dqn dqn5/best_food.pt, eps 0 | 100 | 0..99 | 20.4 | 45.8 | 2426 | 1950 | 7480 | 0% | 0% | 3623 | 20% | 130.6 | c6a6df9 |
| dqn dqn5/best_food.pt, eps 0.05 | 100 | 0..99 | 67.0 | 68.0 | 1943 | 1900 | 4300 | 0% | 0% | 844 | 0% | 32.3 | c6a6df9 |
| dqn dqn5/best_food.pt, no ghosts, eps 0 | 3 | 0..2 | -94.0 | -94.0 | 980 | 980 | 980 | 0% | 0% | 10000 | 100% | 10.6 | c6a6df9 |
| dqn dqn5/best_food.pt, no ghosts, eps 0.05 | 20 | 0..19 | -82.6 | -82.0 | 1344 | 1105 | 3070 | 0% | 0% | 10000 | 100% | 68.7 | c6a6df9 |
