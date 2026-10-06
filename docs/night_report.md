# Night report, 2026-10-06 (01:10-03:20)

Runs: **dqn1** (dqn0's setup, 20M steps, epsilon 1.0 -> 0.05 over 3M; started the
evening before) and **dqn3b** (shaping Phi = 0.1 * (86 - d), Phi(terminal) = 0,
4 actions, 5M steps; started 01:20 after dqn3 with the old potential was
stopped). Both finished, no crashes. Supervisor (`scripts/supervisor.py`)
watched both from 01:34 and exited at 03:18. Plot of all evals: `docs/eval.png`.

## Summary

- Neither run cleared level 1 in any evaluation game. Best medians:
  dqn1/best.pt 70.1 reward, 1580 score, 152 food items per game (dqn0: 61.8,
  1360, 128).
- **The shaping did not fix standing still.** dqn3b eats the same amount of
  food as dqn0 (127 vs 128) and spends more of the game without eating:
  median 212 steps after its last food vs 69; 59% vs 54% of the game in
  stretches of 30+ steps without food.
- **dqn1 gets worse at this with more training.** From ~9M steps on, more
  and more evaluation games run to the 10000-step limit without the game
  ending: 0/20 at 9M, 6/20 at 13M (train.py's evals), 45-50% of supervisor
  eval games at 13M, 15M and 19.5M. Its median eval reward on seeds 0..19
  fell to -1.4 (13M) and -0.4 (19.5M) because the step penalty piles up.
  Loss stayed flat (0.37-0.40), mean Q kept rising (8.7 at 7M -> 18 at 20M).
- best.pt is chosen by median eval reward, which ignores stuck games while
  fewer than half get stuck: dqn1/best.pt (13.25M) gets stuck in 15/100
  games, so its mean reward (49.1) is below dqn0's (61.9).
- Speed: both runs dropped below 1.5k steps/s at times (details below). No
  new runs were started.

## Final evaluations (100 games, seeds 0..99, no shaping)

"After last food": median steps from the last eaten dot/energizer to the
end of the game. "In 30+ gaps": mean share of the game spent in stretches
of 30 or more steps without eating (the tail included). "Stuck": games
that hit the 10000-step limit.

| checkpoint | mean reward | median reward | median score | max score | level 1 cleared | food eaten | after last food | in 30+ gaps | stuck |
|---|---|---|---|---|---|---|---|---|---|
| dqn0/best.pt (4.75M, 5 actions, no shaping) | 61.9 | 61.8 | 1360 | 2950 | 0% | 128 | 69 | 54% | 0% |
| dqn1/best.pt (13.25M of 20M, eps over 3M) | 49.1 | 70.1 | 1580 | 3180 | 0% | 152 | 784 | 73% | 15% |
| dqn3b/best.pt (4.75M, shaping, 4 actions) | 62.2 | 62.4 | 1490 | 3070 | 0% | 127 | 212 | 59% | 0% |

For scale (20 games, below): the heuristic spends 12% of its game in such
stretches, 12 steps after its last food.

## What the agents do when they are not eating

On 10 games each of dqn0/best.pt and dqn3b/best.pt (seeds 0..9), in
stretches of 30+ steps without food:

- About half the steps pacman does not move at all (dqn0: 1679 still /
  2671 moving; dqn3b: 2699 / 2768). With 4 actions dqn3b stands by walking
  into a wall.
- When it moves, it goes back and forth over a few cells (4.8 visits per
  distinct cell for dqn0, 7.4 for dqn3b).
- The most visited cells are corners of the maze where the energizers were:
  (1,1), (1,5), (31,1), (19,21), (21,21).
- Each game had 37-51 dots left when it reached its last life.

## What did not go as planned

1. **dqn3 (old potential) was stopped** at 1.16M steps, as decided. Its last
   eval (1M): median reward -9.2, score 505. Files kept in `runs/dqn3/`.
2. **The supervisor started at 01:34, after both runs had passed some
   500k points.** train.py keeps only its latest checkpoint, so there was
   nothing left to evaluate for dqn1 0.5M..7.5M and dqn3b 0.5M..1M
   (logged as skipped). The training script's own evals (every 250k)
   cover those steps in the tables below.
3. **Speed below 1.5k steps/s.** dqn3b: 97 of its 500 log lines (every
   10k steps) were under 1.5k, the lowest 1,277. dqn1: 65 of 2000, 26 of them while dqn3b
   was running. The first dip (dqn3b 1,345 at 01:34) came while I ran the
   test suite and the reference evaluations next to both runs. Later dips
   line up with the supervisor's evaluations (10-78 s every few minutes,
   up to 50% stuck games of 10000 steps make them long). The supervisor
   logged only one of them, because it checks the latest log line every
   15 s and writes at most one note per 10 minutes. No runs were waiting to
   start, so nothing was held back.
4. **No crash happened, so the restart path was not used tonight.** It is
   covered by `tests/test_supervisor.py` (kill -> one --resume, second
   kill -> left down). After 03:18 nothing was watched any more.
5. **dqn1's late checkpoints are worse than its best.pt** in the way that
   matters here (stuck games); best.pt selection by median reward did not
   catch it.

## What I did not find out

- **Why the agent stops eating.** My guess, not verified: it hides. Ghosts
  only chase pacman when they see it along a straight line with no wall in
  between. A death costs -20, while the ~45 remaining dots are worth +1 each
  and are often far away. To check this, I would need to:
  - test whether the favourite corners are out of the ghosts' line of sight
    more often than other cells;
  - compare Q(stand) with Q(move towards food) in those states;
  - see whether a smaller death penalty or a per-life time limit changes it.
- **Why mean Q keeps rising in dqn1** (to ~18) while it gets stuck more.
  Possible causes, both unchecked: overestimation that Double DQN does not
  remove, or bootstrapping at the 10000-step truncation in training.
- **Whether any of this holds across seeds.** Every run is seed 0. The gaps
  between dqn0, dqn2 and dqn3b (61-62 median reward) are within what one
  seed can tell apart.
- dqn3b's mean Q stayed negative for most of the run (-8 at 0.26M, about
  -0.7 from 3.9M) although Phi >= 0. I did not look into it.
- The 30-step threshold for a "long" stretch without food is arbitrary.

### Reference agents (20 games, seeds 0..19, no shaping)

| | median reward | median score | max score | level 1 cleared | median length | food eaten | mean gap | longest gap | steps after last food | in 30+ gaps |
|---|---|---|---|---|---|---|---|---|---|---|
| heuristic | 404.4 | 13620 | 51130 | 80% | 1270 | 342 | 3.2 | 53 | 12 | 12% |
| dqn0/best.pt | 61.2 | 1380 | 2950 | 0% | 778 | 128 | 4.9 | 163 | 65 | 56% |
| dqn2/best.pt | 56.3 | 1370 | 2350 | 0% | 1024 | 128 | 4.0 | 154 | 378 | 66% |

### dqn1

Training-script evals (every 250k steps, 20 greedy games, seeds 1000000..1000019):

| steps | median reward | median score | max score | level 1 cleared | mean length |
|---|---|---|---|---|---|
| 0.25M | -46.4 | 165 | 250 | 0% | 206 |
| 0.50M | -56.5 | 140 | 180 | 0% | 419 |
| 0.75M | -54.8 | 130 | 180 | 0% | 441 |
| 1.00M | -39.2 | 260 | 360 | 0% | 394 |
| 1.25M | -28.9 | 335 | 450 | 0% | 322 |
| 1.50M | -15.0 | 480 | 580 | 0% | 310 |
| 1.75M | -18.9 | 440 | 610 | 0% | 361 |
| 2.00M | 7.5 | 725 | 1760 | 0% | 543 |
| 2.25M | 13.2 | 810 | 1110 | 0% | 664 |
| 2.50M | 31.0 | 1000 | 2010 | 0% | 594 |
| 2.75M | 42.7 | 1110 | 2350 | 0% | 694 |
| 3.00M | 41.4 | 1060 | 2280 | 0% | 722 |
| 3.25M | 35.8 | 1075 | 2140 | 0% | 880 |
| 3.50M | 52.0 | 1235 | 4250 | 0% | 782 |
| 3.75M | 57.6 | 1280 | 2200 | 0% | 670 |
| 4.00M | 54.2 | 1210 | 2290 | 0% | 708 |
| 4.25M | 55.0 | 1410 | 2550 | 0% | 1032 |
| 4.50M | 52.1 | 1325 | 2830 | 0% | 1110 |
| 4.75M | 54.4 | 1310 | 2630 | 0% | 1105 |
| 5.00M | 52.0 | 1285 | 2450 | 0% | 1401 |
| 5.25M | 54.1 | 1360 | 2360 | 0% | 1505 |
| 5.50M | 57.8 | 1490 | 2360 | 0% | 1466 |
| 5.75M | 52.2 | 1320 | 2700 | 0% | 1420 |
| 6.00M | 61.2 | 1370 | 2590 | 0% | 1041 |
| 6.25M | 60.7 | 1420 | 2620 | 0% | 1576 |
| 6.50M | 60.2 | 1370 | 2500 | 0% | 1499 |
| 6.75M | 60.2 | 1375 | 2980 | 0% | 1390 |
| 7.00M | 59.6 | 1475 | 2880 | 0% | 1335 |
| 7.25M | 63.6 | 1475 | 2370 | 0% | 1397 |
| 7.50M | 54.7 | 1480 | 2440 | 0% | 1889 |
| 7.75M | 63.3 | 1510 | 2560 | 0% | 1703 |
| 8.00M | 62.1 | 1470 | 2780 | 0% | 1766 |
| 8.25M | 61.8 | 1465 | 2680 | 0% | 1434 |
| 8.50M | 58.4 | 1475 | 2500 | 0% | 2179 |
| 8.75M | 67.8 | 1445 | 2550 | 0% | 1114 |
| 9.00M | 53.6 | 1515 | 2620 | 0% | 2013 |
| 9.25M | 67.8 | 1580 | 2690 | 0% | 2658 |
| 9.50M | 68.8 | 1580 | 2670 | 0% | 2362 |
| 9.75M | 70.7 | 1555 | 2910 | 0% | 1840 |
| 10.00M | 69.4 | 1595 | 3880 | 0% | 2446 |
| 10.25M | 71.5 | 1630 | 2980 | 0% | 2166 |
| 10.50M | 68.3 | 1580 | 2580 | 0% | 1828 |
| 10.75M | 69.1 | 1670 | 2190 | 0% | 3733 |
| 11.00M | 68.8 | 1675 | 2480 | 0% | 1731 |
| 11.25M | 70.2 | 1580 | 2780 | 0% | 2134 |
| 11.50M | 78.1 | 1635 | 4580 | 0% | 1995 |
| 11.75M | 60.9 | 1750 | 2180 | 0% | 3089 |
| 12.00M | 67.4 | 1580 | 3480 | 0% | 2592 |
| 12.25M | 75.7 | 1680 | 2880 | 0% | 1809 |
| 12.50M | 62.7 | 1680 | 2190 | 0% | 3670 |
| 12.75M | 72.2 | 1685 | 4600 | 0% | 2193 |
| 13.00M | 62.8 | 1605 | 3490 | 0% | 4284 |
| 13.25M | 79.0 | 1680 | 2950 | 0% | 3142 |
| 13.50M | 73.0 | 1685 | 2280 | 0% | 3654 |
| 13.75M | -0.5 | 1730 | 3080 | 0% | 5743 |
| 14.00M | 70.8 | 1735 | 3080 | 0% | 4321 |
| 14.25M | 67.3 | 1645 | 2990 | 0% | 3075 |
| 14.50M | 77.7 | 1680 | 2080 | 0% | 3004 |
| 14.75M | 64.9 | 1605 | 2190 | 0% | 3081 |
| 15.00M | -41.5 | 1680 | 3650 | 0% | 6592 |
| 15.25M | 76.0 | 1640 | 2510 | 0% | 3656 |
| 15.50M | 41.3 | 1645 | 2810 | 0% | 4378 |
| 15.75M | 65.1 | 1690 | 2190 | 0% | 3543 |
| 16.00M | 57.4 | 1685 | 4560 | 0% | 4575 |
| 16.25M | 72.4 | 1695 | 2180 | 0% | 1846 |
| 16.50M | 67.7 | 1680 | 3010 | 0% | 3533 |
| 16.75M | 38.6 | 1705 | 2250 | 0% | 4664 |
| 17.00M | 67.7 | 1690 | 2240 | 0% | 2569 |
| 17.25M | 55.3 | 1690 | 2180 | 0% | 5050 |
| 17.50M | 62.9 | 1690 | 2290 | 0% | 3061 |
| 17.75M | 50.8 | 1660 | 2250 | 0% | 4001 |
| 18.00M | 54.4 | 1735 | 2290 | 0% | 4605 |
| 18.25M | 42.4 | 1740 | 2290 | 0% | 4875 |
| 18.50M | 42.7 | 1685 | 2290 | 0% | 5504 |
| 18.75M | 63.4 | 1565 | 2250 | 0% | 4490 |
| 19.00M | 64.6 | 1690 | 2190 | 0% | 3244 |
| 19.25M | 55.7 | 1760 | 4550 | 0% | 3459 |
| 19.50M | 50.9 | 1610 | 4550 | 0% | 4775 |
| 19.75M | -5.9 | 1685 | 2260 | 0% | 5551 |
| 20.00M | 59.8 | 1690 | 4590 | 0% | 3607 |

Supervisor evals (every 500k steps, 20 greedy games, seeds 0..19, no shaping); gaps are steps between eaten food, medians over games:

| steps | median reward | median score | max score | level 1 cleared | median length | food eaten | mean gap | longest gap | steps after last food | in 30+ gaps |
|---|---|---|---|---|---|---|---|---|---|---|
| 8.00M | 63.8 | 1480 | 2880 | 0% | 1449 | 142 | 5.7 | 270 | 415 | 72% |
| 8.50M | 63.8 | 1475 | 2430 | 0% | 1338 | 141 | 4.6 | 174 | 476 | 69% |
| 9.00M | 65.2 | 1535 | 2410 | 0% | 1430 | 147 | 5.9 | 262 | 516 | 73% |
| 9.50M | 66.7 | 1515 | 2580 | 0% | 1215 | 146 | 5.5 | 189 | 340 | 69% |
| 10.00M | 66.1 | 1580 | 2580 | 0% | 1778 | 152 | 6.0 | 177 | 506 | 66% |
| 10.50M | 75.5 | 1580 | 1890 | 0% | 1116 | 152 | 4.6 | 126 | 425 | 60% |
| 11.00M | 70.4 | 1585 | 2590 | 0% | 1628 | 152 | 4.5 | 188 | 375 | 62% |
| 11.50M | 78.0 | 1575 | 1890 | 0% | 1230 | 151 | 4.6 | 144 | 264 | 57% |
| 12.00M | 78.7 | 1580 | 2040 | 0% | 1286 | 152 | 5.1 | 160 | 384 | 57% |
| 12.50M | 60.7 | 1670 | 2880 | 0% | 2150 | 152 | 4.7 | 146 | 1169 | 80% |
| 13.00M | -1.4 | 1705 | 3780 | 0% | 5122 | 152 | 4.4 | 140 | 4126 | 81% |
| 13.50M | 44.3 | 1635 | 2780 | 0% | 2765 | 152 | 4.0 | 132 | 1854 | 79% |
| 14.00M | 72.9 | 1655 | 1890 | 0% | 1720 | 152 | 4.2 | 132 | 788 | 72% |
| 14.50M | 65.1 | 1580 | 1880 | 0% | 1842 | 152 | 3.6 | 78 | 1290 | 76% |
| 15.00M | 56.3 | 1595 | 2490 | 0% | 2134 | 150 | 3.5 | 94 | 1634 | 74% |
| 15.50M | 62.6 | 1605 | 2200 | 0% | 1888 | 148 | 3.7 | 72 | 1323 | 72% |
| 16.00M | 55.5 | 1680 | 1910 | 0% | 2726 | 152 | 4.4 | 156 | 1770 | 81% |
| 16.50M | 47.8 | 1590 | 2190 | 0% | 3084 | 152 | 3.7 | 158 | 1628 | 81% |
| 17.00M | 58.8 | 1685 | 1890 | 0% | 2208 | 152 | 3.7 | 108 | 1510 | 78% |
| 17.50M | 65.5 | 1645 | 1900 | 0% | 1898 | 153 | 3.8 | 90 | 1015 | 76% |
| 18.00M | 54.5 | 1645 | 2150 | 0% | 2401 | 152 | 3.6 | 96 | 1586 | 78% |
| 18.50M | 62.6 | 1690 | 2280 | 0% | 2222 | 153 | 4.1 | 132 | 1537 | 74% |
| 19.00M | 46.9 | 1610 | 1890 | 0% | 2848 | 153 | 3.8 | 150 | 1666 | 75% |
| 19.50M | -0.4 | 1685 | 2250 | 0% | 6792 | 152 | 3.6 | 90 | 5920 | 85% |
| 20.00M | 45.4 | 1600 | 3210 | 0% | 3131 | 153 | 3.6 | 90 | 1656 | 76% |

Speed: median 2,006 steps/s over the run, min 1,314, last 50 log lines median 2,608.

### dqn3b

Training-script evals (every 250k steps, 20 greedy games, seeds 1000000..1000019):

| steps | median reward | median score | max score | level 1 cleared | mean length |
|---|---|---|---|---|---|
| 0.25M | -47.1 | 170 | 210 | 0% | 264 |
| 0.50M | -45.2 | 190 | 270 | 0% | 332 |
| 0.75M | -27.1 | 340 | 380 | 0% | 271 |
| 1.00M | -5.7 | 560 | 740 | 0% | 258 |
| 1.25M | 0.0 | 665 | 1050 | 0% | 426 |
| 1.50M | 0.8 | 720 | 1020 | 0% | 467 |
| 1.75M | 9.3 | 760 | 1310 | 0% | 491 |
| 2.00M | 5.7 | 740 | 990 | 0% | 510 |
| 2.25M | 17.9 | 845 | 1050 | 0% | 533 |
| 2.50M | 19.6 | 955 | 1280 | 0% | 573 |
| 2.75M | 22.5 | 870 | 1750 | 0% | 410 |
| 3.00M | 41.1 | 1110 | 1420 | 0% | 597 |
| 3.25M | 45.5 | 1250 | 2540 | 0% | 660 |
| 3.50M | 53.9 | 1380 | 2480 | 0% | 824 |
| 3.75M | 58.7 | 1425 | 2840 | 0% | 840 |
| 4.00M | 55.8 | 1315 | 1930 | 0% | 723 |
| 4.25M | 65.0 | 1435 | 2780 | 0% | 680 |
| 4.50M | 65.2 | 1540 | 2690 | 0% | 877 |
| 4.75M | 65.5 | 1565 | 2960 | 0% | 935 |
| 5.00M | 63.7 | 1655 | 2780 | 0% | 1086 |

Supervisor evals (every 500k steps, 20 greedy games, seeds 0..19, no shaping); gaps are steps between eaten food, medians over games:

| steps | median reward | median score | max score | level 1 cleared | median length | food eaten | mean gap | longest gap | steps after last food | in 30+ gaps |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.50M | 0.6 | 720 | 760 | 0% | 498 | 64 | 4.1 | 106 | 98 | 66% |
| 2.00M | 8.0 | 730 | 1380 | 0% | 474 | 72 | 5.9 | 172 | 36 | 55% |
| 2.50M | 21.4 | 885 | 1280 | 0% | 499 | 86 | 5.3 | 130 | 14 | 56% |
| 3.00M | 35.7 | 1090 | 1430 | 0% | 514 | 100 | 4.6 | 102 | 50 | 48% |
| 3.50M | 50.0 | 1210 | 1870 | 0% | 684 | 116 | 4.4 | 144 | 160 | 53% |
| 4.00M | 51.8 | 1345 | 1940 | 0% | 827 | 118 | 4.7 | 145 | 141 | 57% |
| 4.50M | 63.5 | 1500 | 2820 | 0% | 778 | 126 | 5.4 | 179 | 48 | 57% |
| 5.00M | 63.2 | 1605 | 2740 | 0% | 914 | 128 | 5.7 | 294 | 183 | 62% |

Speed: median 1,732 steps/s over the run, min 1,277, last 50 log lines median 1,718.

### Supervisor events

```
2026-10-06 01:34:12  dqn1: no checkpoint left for 500000..7500000 (15 points), skipped
2026-10-06 01:34:52  dqn3b: no checkpoint left for 500000..1000000 (2 points), skipped
2026-10-06 01:34:52  dqn3b: slow, 1345 steps/s (< 1500); no new runs while this lasts
2026-10-06 02:12:00  dqn3b: finished (5000000 steps)
2026-10-06 03:17:47  dqn1: finished (20000000 steps)
2026-10-06 03:18:06  all runs finished or failed; supervisor exits
```
