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
| dqn dqn1/best.pt (11.0M, by mean) | 100 | 0..99 | 65.7 | 71.9 | 1696 | 1605 | 2990 | 0% | 0% | 1875 | 3% | 115.2 | a5cfce0 |
| dqn dqn1/best_median.pt | 100 | 0..99 | 49.1 | 70.1 | 1688 | 1580 | 3180 | 0% | 0% | 2924 | 15% | 95.2 | af91590 |
| dqn dqn1/best_median.pt, 50 games | 50 | 0..49 | 46.0 | 67.3 | 1724 | 1590 | 3180 | 0% | 0% | 3104 | 16% | 95.4 | a5cfce0 |
| dqn dqn1/best_median.pt, no ghosts | 50 | 0..49 | -143.0 | -143.0 | 530 | 530 | 530 | 0% | 0% | 10000 | 100% | 257.8 | a5cfce0 |
| dqn dqn1@08.00M | 20 | 0..19 | 58.7 | 63.8 | 1606 | 1480 | 2880 | 0% | 0% | 1634 | 0% | 35.1 | cbc1d8a |
| dqn dqn1@08.50M | 20 | 0..19 | 59.2 | 63.8 | 1552 | 1475 | 2430 | 0% | 0% | 1584 | 0% | 32.1 | af91590 |
| dqn dqn1@09.00M | 20 | 0..19 | 55.5 | 65.2 | 1620 | 1535 | 2410 | 0% | 0% | 1991 | 0% | 39.5 | af91590 |
| dqn dqn1@09.50M | 20 | 0..19 | 52.1 | 66.7 | 1638 | 1515 | 2580 | 0% | 0% | 2477 | 10% | 49.9 | af91590 |
| dqn dqn1@10.00M | 20 | 0..19 | 63.5 | 66.1 | 1632 | 1580 | 2580 | 0% | 0% | 1773 | 0% | 36.0 | af91590 |
| dqn dqn1@10.50M | 20 | 0..19 | 61.7 | 75.5 | 1592 | 1580 | 1890 | 0% | 0% | 2092 | 10% | 42.2 | af91590 |
| dqn dqn1@11.00M | 20 | 0..19 | 69.2 | 70.4 | 1666 | 1585 | 2590 | 0% | 0% | 1547 | 0% | 30.7 | af91590 |
| dqn dqn1@11.50M | 20 | 0..19 | 62.4 | 78.0 | 1600 | 1575 | 1890 | 0% | 0% | 2109 | 10% | 42.7 | af91590 |
| dqn dqn1@12.00M | 20 | 0..19 | 73.1 | 78.7 | 1624 | 1580 | 2040 | 0% | 0% | 1391 | 0% | 24.4 | af91590 |
| dqn dqn1@12.50M | 20 | 0..19 | 31.7 | 60.7 | 1732 | 1670 | 2880 | 0% | 0% | 4010 | 20% | 53.8 | af91590 |
| dqn dqn1@13.00M | 20 | 0..19 | 0.4 | -1.4 | 1874 | 1705 | 3780 | 0% | 0% | 5760 | 45% | 73.4 | af91590 |
| dqn dqn1@13.50M | 20 | 0..19 | 28.9 | 44.3 | 1716 | 1635 | 2780 | 0% | 0% | 4409 | 30% | 57.5 | af91590 |
| dqn dqn1@14.00M | 20 | 0..19 | 45.6 | 72.9 | 1654 | 1655 | 1890 | 0% | 0% | 3363 | 20% | 43.9 | af91590 |
| dqn dqn1@14.50M | 20 | 0..19 | 43.6 | 65.1 | 1652 | 1580 | 1880 | 0% | 0% | 3332 | 20% | 42.3 | af91590 |
| dqn dqn1@15.00M | 20 | 0..19 | 14.8 | 56.3 | 1710 | 1595 | 2490 | 0% | 0% | 5120 | 45% | 65.5 | af91590 |
| dqn dqn1@15.50M | 20 | 0..19 | 35.4 | 62.6 | 1651 | 1605 | 2200 | 0% | 0% | 3725 | 20% | 47.6 | af91590 |
| dqn dqn1@16.00M | 20 | 0..19 | 28.1 | 55.5 | 1668 | 1680 | 1910 | 0% | 0% | 4102 | 20% | 53.3 | af91590 |
| dqn dqn1@16.50M | 20 | 0..19 | 24.3 | 47.8 | 1700 | 1590 | 2190 | 0% | 0% | 4764 | 30% | 60.1 | af91590 |
| dqn dqn1@17.00M | 20 | 0..19 | 44.4 | 58.8 | 1685 | 1685 | 1890 | 0% | 0% | 3444 | 15% | 44.2 | af91590 |
| dqn dqn1@17.50M | 20 | 0..19 | 57.9 | 65.5 | 1672 | 1645 | 1900 | 0% | 0% | 2389 | 5% | 31.3 | af91590 |
| dqn dqn1@18.00M | 20 | 0..19 | 23.3 | 54.5 | 1690 | 1645 | 2150 | 0% | 0% | 4514 | 30% | 57.4 | af91590 |
| dqn dqn1@18.50M | 20 | 0..19 | 34.8 | 62.6 | 1725 | 1690 | 2280 | 0% | 0% | 4486 | 35% | 57.5 | af91590 |
| dqn dqn1@19.00M | 20 | 0..19 | 40.3 | 46.9 | 1652 | 1610 | 1890 | 0% | 0% | 3411 | 10% | 44.0 | af91590 |
| dqn dqn1@19.50M | 20 | 0..19 | 6.9 | -0.4 | 1716 | 1685 | 2250 | 0% | 0% | 6035 | 50% | 77.8 | af91590 |
| dqn dqn1@20.00M | 20 | 0..19 | 21.0 | 45.4 | 1755 | 1600 | 3210 | 0% | 0% | 4551 | 25% | 29.3 | af91590 |
| dqn dqn2/best.pt | 100 | 0..99 | 56.5 | 59.2 | 1483 | 1420 | 2670 | 0% | 0% | 1088 | 0% | 72.9 | 7f96361 |
| dqn dqn3b/best.pt | 100 | 0..99 | 62.2 | 62.4 | 1610 | 1490 | 3070 | 0% | 0% | 996 | 0% | 65.9 | af91590 |
| dqn dqn3b@01.50M | 20 | 0..19 | -0.1 | 0.6 | 691 | 720 | 760 | 0% | 0% | 510 | 0% | 10.5 | af91590 |
| dqn dqn3b@02.00M | 20 | 0..19 | 6.9 | 8.0 | 768 | 730 | 1380 | 0% | 0% | 519 | 0% | 11.0 | af91590 |
| dqn dqn3b@02.50M | 20 | 0..19 | 22.4 | 21.4 | 936 | 885 | 1280 | 0% | 0% | 510 | 0% | 10.4 | af91590 |
| dqn dqn3b@03.00M | 20 | 0..19 | 37.9 | 35.7 | 1102 | 1090 | 1430 | 0% | 0% | 590 | 0% | 12.1 | af91590 |
| dqn dqn3b@03.50M | 20 | 0..19 | 50.4 | 50.0 | 1264 | 1210 | 1870 | 0% | 0% | 708 | 0% | 14.5 | af91590 |
| dqn dqn3b@04.00M | 20 | 0..19 | 52.1 | 51.8 | 1366 | 1345 | 1940 | 0% | 0% | 874 | 0% | 17.7 | af91590 |
| dqn dqn3b@04.50M | 20 | 0..19 | 59.8 | 63.5 | 1572 | 1500 | 2820 | 0% | 0% | 1006 | 0% | 20.3 | af91590 |
| dqn dqn3b@05.00M | 20 | 0..19 | 61.3 | 63.2 | 1688 | 1605 | 2740 | 0% | 0% | 1061 | 0% | 14.2 | af91590 |
| dqn dqn4/best.pt | 100 | 0..99 | 43.0 | 46.1 | 1314 | 1285 | 2450 | 0% | 0% | 1073 | 0% | 36.8 | 8b2de66 |
| dqn dqn4@00.50M | 20 | 0..19 | -56.7 | -55.3 | 135 | 140 | 160 | 0% | 0% | 510 | 0% | 10.0 | 8b2de66 |
| dqn dqn4@01.00M | 20 | 0..19 | -44.1 | -42.5 | 190 | 200 | 280 | 0% | 0% | 256 | 0% | 5.4 | 8b2de66 |
| dqn dqn4@01.50M | 20 | 0..19 | 13.0 | 12.4 | 784 | 775 | 1070 | 0% | 0% | 284 | 0% | 6.0 | 8b2de66 |
| dqn dqn4@02.00M | 20 | 0..19 | 17.9 | 21.0 | 866 | 860 | 1170 | 0% | 0% | 483 | 0% | 9.8 | 8b2de66 |
| dqn dqn4@02.50M | 20 | 0..19 | 20.9 | 19.6 | 996 | 955 | 1550 | 0% | 0% | 701 | 0% | 14.3 | 8b2de66 |
| dqn dqn4@03.00M | 20 | 0..19 | 27.1 | 27.8 | 1066 | 1040 | 1410 | 0% | 0% | 876 | 0% | 16.2 | 8b2de66 |
| dqn dqn4@03.50M | 20 | 0..19 | 38.9 | 39.0 | 1223 | 1200 | 1530 | 0% | 0% | 949 | 0% | 18.6 | 8b2de66 |
| dqn dqn4@04.00M | 20 | 0..19 | 36.8 | 36.6 | 1217 | 1155 | 2140 | 0% | 0% | 963 | 0% | 18.9 | 8b2de66 |
| dqn dqn4@04.50M | 20 | 0..19 | 41.4 | 41.5 | 1272 | 1265 | 1960 | 0% | 0% | 928 | 0% | 12.1 | 8b2de66 |
| dqn dqn4@05.00M | 20 | 0..19 | 41.5 | 44.7 | 1314 | 1215 | 2450 | 0% | 0% | 1064 | 0% | 7.7 | 8b2de66 |
| dqn dqn5/best_food.pt, eps 0 | 100 | 0..99 | 20.4 | 45.8 | 2426 | 1950 | 7480 | 0% | 0% | 3623 | 20% | 130.6 | c6a6df9 |
| dqn dqn5/best_food.pt, eps 0.05 | 100 | 0..99 | 67.0 | 68.0 | 1943 | 1900 | 4300 | 0% | 0% | 844 | 0% | 32.3 | c6a6df9 |
| dqn dqn5/best_food.pt, no ghosts, eps 0 | 3 | 0..2 | -94.0 | -94.0 | 980 | 980 | 980 | 0% | 0% | 10000 | 100% | 10.6 | c6a6df9 |
| dqn dqn5/best_food.pt, no ghosts, eps 0.05 | 20 | 0..19 | -82.6 | -82.0 | 1344 | 1105 | 3070 | 0% | 0% | 10000 | 100% | 68.7 | c6a6df9 |
| dqn dqn6/best_food.pt, eps 0 | 100 | 0..99 | 51.1 | 56.7 | 1839 | 1685 | 5450 | 0% | 0% | 1806 | 0% | 67.0 | eb30990 |
| dqn dqn6/best_food.pt, eps 0.05 | 100 | 0..99 | 60.8 | 62.5 | 1673 | 1505 | 3350 | 0% | 0% | 1090 | 0% | 39.7 | eb30990 |
| dqn dqn6/best_food.pt, no ghosts, eps 0 | 3 | 0..2 | -178.0 | -178.0 | 200 | 200 | 200 | 0% | 0% | 10000 | 100% | 10.5 | eb30990 |
| dqn dqn6/best_food.pt, no ghosts, eps 0.05 | 20 | 0..19 | -122.2 | -123.0 | 784 | 730 | 1730 | 0% | 0% | 10000 | 100% | 65.3 | eb30990 |
| dqn dqn6@01.00M | 20 | 0..19 | -41.7 | -41.2 | 252 | 240 | 370 | 0% | 0% | 425 | 0% | 5.5 | 2b91308 |
| dqn dqn6@01.50M | 20 | 0..19 | -16.3 | -15.6 | 482 | 450 | 600 | 0% | 0% | 472 | 0% | 5.9 | 2b91308 |
| dqn dqn6@02.00M | 20 | 0..19 | 26.0 | 26.0 | 960 | 915 | 1270 | 0% | 0% | 582 | 0% | 7.6 | 2b91308 |
| dqn dqn6@02.50M | 20 | 0..19 | 32.9 | 33.2 | 1039 | 1020 | 1310 | 0% | 0% | 651 | 0% | 8.0 | 2b91308 |
| dqn dqn6@03.00M | 20 | 0..19 | 39.8 | 38.8 | 1130 | 1115 | 1460 | 0% | 0% | 637 | 0% | 8.4 | 2b91308 |
| dqn dqn6@03.50M | 20 | 0..19 | 33.4 | 42.9 | 1186 | 1205 | 1780 | 0% | 0% | 937 | 0% | 11.7 | 2b91308 |
| dqn dqn6@04.00M | 20 | 0..19 | 42.8 | 41.1 | 1300 | 1225 | 1790 | 0% | 0% | 950 | 0% | 11.6 | 2b91308 |
| dqn dqn6@04.50M | 20 | 0..19 | 39.4 | 38.2 | 1468 | 1350 | 2620 | 0% | 0% | 1249 | 0% | 15.0 | 2b91308 |
| dqn dqn6@05.00M | 20 | 0..19 | 37.8 | 41.0 | 1334 | 1235 | 1820 | 0% | 0% | 1346 | 0% | 16.3 | 2b91308 |
| dqn dqn6@05.50M | 20 | 0..19 | 39.0 | 41.6 | 1414 | 1320 | 2120 | 0% | 0% | 1383 | 0% | 16.3 | 2b91308 |
| dqn dqn6@06.00M | 20 | 0..19 | 53.8 | 58.4 | 1532 | 1460 | 3000 | 0% | 0% | 1188 | 0% | 14.2 | 2b91308 |
| dqn dqn6@06.50M | 20 | 0..19 | 57.0 | 62.4 | 1488 | 1520 | 1950 | 0% | 0% | 1000 | 0% | 12.7 | 2b91308 |
| dqn dqn6@07.00M | 20 | 0..19 | 54.6 | 55.6 | 1488 | 1355 | 2070 | 0% | 0% | 1150 | 0% | 14.1 | 2b91308 |
| dqn dqn6@07.50M | 20 | 0..19 | 56.9 | 59.0 | 1734 | 1715 | 2830 | 0% | 0% | 1313 | 0% | 15.9 | 2b91308 |
| dqn dqn6@08.00M | 20 | 0..19 | 43.7 | 51.9 | 1716 | 1595 | 2450 | 0% | 0% | 2043 | 0% | 23.8 | 2b91308 |
| dqn dqn6@08.50M | 20 | 0..19 | 56.1 | 60.0 | 1938 | 1725 | 2900 | 0% | 0% | 1646 | 0% | 19.2 | 2b91308 |
| dqn dqn6@09.00M | 20 | 0..19 | 53.4 | 57.0 | 1796 | 1675 | 2800 | 0% | 0% | 1756 | 0% | 20.9 | 2b91308 |
| dqn dqn6@09.50M | 20 | 0..19 | 52.8 | 56.4 | 1785 | 1700 | 3100 | 0% | 0% | 1773 | 0% | 21.1 | 2b91308 |
| dqn dqn6@10.00M | 20 | 0..19 | 47.1 | 56.9 | 1933 | 1800 | 3140 | 0% | 0% | 2111 | 0% | 15.5 | 2b91308 |
| ppo ppo0/best.pt | 100 | 0..99 | 35.7 | 36.2 | 1034 | 1020 | 1330 | 0% | 0% | 576 | 0% | 19.8 | 8b2de66 |
| ppo ppo0@00.50M | 20 | 0..19 | -28.1 | -28.1 | 352 | 350 | 470 | 0% | 0% | 271 | 0% | 5.7 | 8b2de66 |
| ppo ppo0@01.00M | 20 | 0..19 | 0.8 | 1.3 | 675 | 625 | 920 | 0% | 0% | 469 | 0% | 8.9 | 8b2de66 |
| ppo ppo0@01.50M | 20 | 0..19 | 29.2 | 29.1 | 998 | 980 | 1190 | 0% | 0% | 617 | 0% | 12.0 | 8b2de66 |
| ppo ppo0@02.00M | 20 | 0..19 | 34.6 | 36.2 | 1026 | 1010 | 1330 | 0% | 0% | 601 | 0% | 11.6 | 8b2de66 |
| ppo ppo0@02.50M | 20 | 0..19 | 41.3 | 41.0 | 1124 | 1120 | 1670 | 0% | 0% | 592 | 0% | 11.8 | 8b2de66 |
| ppo ppo0@03.00M | 20 | 0..19 | 38.4 | 39.9 | 1134 | 1125 | 1830 | 0% | 0% | 790 | 0% | 14.7 | 8b2de66 |
| ppo ppo0@03.50M | 20 | 0..19 | 27.7 | 29.4 | 1015 | 1040 | 1240 | 0% | 0% | 889 | 0% | 16.9 | 8b2de66 |
| ppo ppo0@04.00M | 20 | 0..19 | 26.3 | 31.3 | 1043 | 980 | 1740 | 0% | 0% | 951 | 0% | 18.1 | 8b2de66 |
| ppo ppo0@04.50M | 20 | 0..19 | 30.6 | 32.7 | 1015 | 990 | 1240 | 0% | 0% | 746 | 0% | 14.2 | 8b2de66 |
| ppo ppo0@05.00M | 20 | 0..19 | 27.4 | 29.2 | 1010 | 990 | 1240 | 0% | 0% | 893 | 0% | 10.6 | 8b2de66 |

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

## Diagnostics: standing still without ghosts (2026-10-06)

`configs/env_noghosts.yaml` (game `--no-ghosts`: the ghosts stay in their
house, pacman cannot die), no shaping, 5 actions, greedy. Without ghosts the
game is deterministic, so all games of one agent are the same game.

| agent | games | mean reward | median score | max score | level 1 cleared | food eaten | after last food | in 30+ gaps | stuck (10000 steps) |
|---|---|---|---|---|---|---|---|---|---|
| dqn1/best_median.pt (13.25M), ghosts | 50 | 46.0 | 1590 | 3180 | 0% | 152 | 836 | 74% | 16% |
| dqn1/best_median.pt (13.25M), no ghosts | 50 | -143.0 | 530 | 530 | 0% | 55 | 9886 | 99% | 100% |
| dqn1/best.pt (11.0M), no ghosts | 3 | -110.0 | 820 | 820 | 0% | 86 | 9809 | 98% | 100% |
| dqn0/best.pt, no ghosts | 3 | -176.0 | 220 | 220 | 0% | 23 | 9953 | 100% | 100% |

Without ghosts every agent stops for good, earlier than with ghosts: dqn1
13.25M eats 55 food items in 114 steps, then stands at (23, 21) against the
bottom wall choosing N, with four dots and an energizer next to it in the
same row, until the step limit. So it does not stand to hide from ghosts.
It looks like a fixed point of the greedy policy: when pacman stands, the
observation stops changing (nothing else moves without ghosts), so the same
action is chosen again forever. With ghosts their moves change the
observation and usually break the loop (8 of these 50 games with ghosts
still hit the limit). Caveat: ghosts frozen in the house are something the
network never saw during training.

## hunger_limit 200: dqn4 and ppo0 (2026-10-06)

Both: 4 actions, no shaping, hunger_limit 200 in training only (evaluation
without it), seed 0, 5M steps, run in parallel. best.pt by mean eval
reward. Final evaluation: 100 games, seeds 0..99. "Hunger": training
episodes that ended by hunger_limit.

| checkpoint | mean reward | median reward | median score | max score | level 1 cleared | food eaten | after last food | in 30+ gaps | stuck | hunger (training) |
|---|---|---|---|---|---|---|---|---|---|---|
| dqn0/best.pt (4.75M; 5 actions) | 61.9 | 61.8 | 1360 | 2950 | 0% | 128 | 69 | 54% | 0% | - |
| dqn1/best.pt (11.0M of 20M, by mean) | 65.7 | 71.9 | 1605 | 2990 | 0% | 152 | 378 | 65% | 3% | - |
| dqn3b/best.pt (4.75M; shaping, 4 actions) | 62.2 | 62.4 | 1490 | 3070 | 0% | 127 | 212 | 59% | 0% | - |
| dqn4/best.pt (5.0M; hunger 200, 4 actions) | 43.0 | 46.1 | 1285 | 2450 | 0% | 113 | 172 | 61% | 0% | 30% (4092/13598); last 1M 63% |
| ppo0/best.pt (2.0M; hunger 200, 4 actions) | 35.7 | 36.2 | 1020 | 1330 | 0% | 98 | 6 | 57% | 0% | 55% (6818/12455); last 1M 75% |

PPO is our own implementation (`agents/ppo.py`): 8 envs x 128 steps,
4 epochs x 4 minibatches, clip 0.2, GAE lambda 0.95, gamma 0.99, entropy
0.01, lr 2.5e-4 annealed to 0, the learning signal divided by the running
std of the discounted return.

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
