# Overnight / autonomous session, 2026-10-06 (until 19:00)

Continues `docs/night_report.md` (the first night). Rules: at most 2 runs in
parallel, `game/` untouched, disk/GPU checked, one `--resume` per crash.
Branch plan for dqn5 as given (SUCCESS / PARTIAL / FAILURE on 100 games,
eps = 0, checkpoint `best_food.pt`).

## Decisions

- **best_food.pt** (the checkpoint with the highest mean food eaten =
  dots + energizers per game): chosen from the supervisor's evaluations,
  every 500k steps, 20 games on seeds 0..19, eps = 0, no shaping/hunger. These
  are the only evaluations that record food per game, and the supervisor keeps
  a copy of each evaluated checkpoint (`runs/<run>/ck_<steps>.pt`). The same
  rule for every run; dqn5 (already running with the old supervisor) is picked
  afterwards from its `sv_eval_*.jsonl` files. `best.pt` (mean reward) stays as
  train.py writes it.
- Final evaluation on seeds 0..99 as for all earlier runs. 20 of these games
  (seeds 0..19) were also used to pick best_food.pt, so the 100-game numbers
  are slightly optimistic; the numbers on seeds 20..99 alone are reported too.
- "Food" in the criteria = mean `food_eaten` over the 100 games (eps = 0).

## Chronology

- 11:13 dqn5 started (`configs/dqn_fooddist.yaml`: dqn1 + food-distance
  channel + steps-since-food in vec, 10M steps, eps over 2M, seed 0),
  supervisor watching it.
- 12:05 instructions for the autonomous session received; dqn5 at 7.0M.
  Disk 21 GB free, GPU 255 MB of 8 GB used, 17 GB RAM available.
- 12:15 `scripts/pick_best_food.py` added (the rule above), and the
  supervisor now updates best_food.pt after each evaluation (for the runs it
  starts from now on). dqn5 so far, mean food on seeds 0..19 per checkpoint:
  16 (0.5M), 36, 46, 74, 101, 113, **123 (3.5M)**, 119, 123, 119, 119, 121,
  121 (6.5M): flat since 3.5M while the score keeps rising (ghosts).
- 12:30 dqn5 finished, no crash. Mean food per checkpoint kept rising
  slowly after the plateau: 121 (6.5M) .. 127 (10M). best_food.pt = 10M
  (126.9 on seeds 0..19). Evaluations at 8M, 9.5M, 10M had 10-20% of games
  stuck until the 10000-step limit. Final evaluation started.
- 12:42 **dqn5 final evaluation (best_food.pt = 10M), verdict FAILURE.**
  100 games eps 0: mean food 127.4, level 1 cleared 0%, mean reward 20.4,
  median score 1950, 20% of games stuck until 10000 steps. Seeds 20..99
  only: food 127.5, same verdict. eps 0.05: food 125.6, reward 67.0, no game
  stuck. Without ghosts it stops eating after ~102 (eps 0) / ~113 (eps 0.05)
  food items and stays until the limit.
- 12:44 FAILURE branch, step 1: the food-distance channel checked by hand in
  3 states of a dqn5 game (start, 120 and 60 food left) against the BFS of
  the heuristic agent (`agents/heuristic_agent.py` `_Graph`, an independent
  implementation): 0 mismatches over all 759 cells each time; printouts in
  `runs/dqn5_channel_check.txt`. The channel is correct, so no rerun of dqn5.
- 12:45 FAILURE branch, step 2: dqn6 = dqn5 + n_step 5 + gamma 0.995, 10M
  (`configs/dqn_fooddist_n5.yaml`). Decision for a case the plan does not
  name: if dqn6 is SUCCESS or PARTIAL, seeds 1 and 2 are run for dqn6 (as in
  the other branches); only FAILURE leads to seeds 1 and 2 of dqn1.
- 12:49 **Incident (my bug), fixed.** The first, failed run of
  `final_eval.py` for dqn5 (row label without the agent type) made
  `evaluate.py` crash while rewriting docs/results.md: it had already opened
  the file for writing, so the file was left with only its header. That
  truncated file went into commit cbf55c7. Restored from c6e0be7 (all 76
  earlier rows and the Notes / Diagnostics / hunger sections) plus the 4 dqn5
  rows; `evaluate.py` now builds the whole file first and replaces it
  atomically, accepts any row label, and a test checks that rows and "## "
  sections survive an update. The summary table (`scripts/summary_table.py`)
  is a "## Summary of all runs" section at the end of results.md.
- 13:58 dqn6 finished, no crash, no slowdown. Mean food on seeds 0..19 per
  500k checkpoint: 11, 26, 49, 92, 99, 104, 102, 111, 112, 114, 115, 122, 123, 123, 125, 129, 131, 131, 130, 132. Ahead of dqn5 at 2M (92 vs 74), behind it from
  3M to 7M, ahead at the end: best_food.pt = 10M, 131.8 (dqn5: 126.9).
  Final evaluation started.
- 14:01 **dqn6 final evaluation (best_food.pt = 10M), verdict FAILURE.**
  100 games eps 0: mean food 130.8, level 1 cleared 0%, mean reward 51.1,
  median score 1685, no game stuck. Seeds 20..99: food 130.6, FAILURE.
  eps 0.05: food 127.8, reward 60.8. Without ghosts it stops even earlier
  than dqn5 (21 food items at eps 0, 76 at eps 0.05).
- 14:02 Branch end: dqn6 also FAILURE, so no more new configs; seeds 1
  and 2 of dqn1 (`configs/dqn_long.yaml`, 20M steps each, in parallel:
  runs dqn1s1, dqn1s2), supervisor started first. For the three-seed table
  dqn1 seed 0 is re-picked by the same food rule: best_food.pt = 17.5M
  (152.7 on seeds 0..19; the supervisor's copies cover 8M..20M only, the
  night supervisor started late). Plain dqn1 already ate ~149 at 10M, more
  than dqn5/dqn6 (127/132) with the extra observation parts at the same step
  count (its epsilon decays over 3M instead of 2M).
- 14:13 dqn1 seed 0, best_food.pt (17.5M), 100 games: eps 0 food 151.6,
  level 1 0%, mean reward 42.9, 16% stuck; eps 0.05 food 146.4, mean reward
  85.4, none stuck (the best mean reward of all runs). Without ghosts it stops
  after 92 (eps 0) / 111 (eps 0.05) food items. dqn1s1/dqn1s2 at 1.2M,
  1.65-1.73k steps/s each while this evaluation ran next to them.
- 15:02 Seeds diverge. dqn1s1 follows seed 0 (mean food 141 at 6M on the
  supervisor's games). dqn1s2 is stuck at ~100 food since 2.5M; train.py's
  own evals of it went negative (mean reward -15 to +13 at 5-5.75M, mean
  length 2200-3600 steps: standing again). Both at ~6M, 1.76-1.91k
  steps/s, no crash.
- 17:03 dqn1s1 at 17.6M (best_food.pt = 11M, 165.2 on seeds 0..19),
  dqn1s2 at 16.5M (best 11.5M, 101.0), both ~1.5k steps/s, no crash. Final
  evaluations set to start by themselves when each run finishes.
- 17:40 dqn1s1 finished (no crash); best_food.pt = 19.5M (169.3 on seeds
  0..19). 100 games eps 0: **mean food 168.8 of 172** (by the criteria this
  is SUCCESS, also 168.7 on seeds 20..99), but **level 1 cleared 0%**, mean
  reward 56.1, 20% stuck. eps 0.05: food 159.4, mean reward 99.0 (the best
  mean reward of all runs), none stuck.
- 17:45 Where the last dots stay (dqn1s1 best_food.pt, seeds 0..9, fewest
  dots left in each game): 2, 4, 5, 2, 2, 2, 2, 2, 2, 2. In all 10 games the
  same two dots are left: (25, 19) and (27, 19), in the bottom-right pocket
  of the maze. A systematic blind spot, not chance.
- 17:48 dqn1s2 finished (no crash); best_food.pt = 18.5M. 100 games eps 0:
  food 101.5, mean reward -28.9, 6% stuck: FAILURE. It never got past ~100
  food after 2.5M.
- 17:50 Final: summary table in docs/results.md ("## Summary of all runs"),
  docs/training.png and docs/eval.png with all runs. dqn1 over 3 seeds
  (best_food.pt, 100 games, eps 0): food 140.6 +- 28.5, mean reward
  23.4 +- 37.4, median score 1668 +- 157, level 1 cleared 0% in every seed.
  Nothing is training any more.

## Ideas for later

(Not tried, as instructed.)

- dqn5 with eps 0.05 at evaluation never gets stuck and has 3x the mean
  reward of eps 0 (67 vs 20) with the same food: a little randomness, or
  any loop breaker, is enough to leave the fixed point. Worth testing as a
  deliberate policy (e.g. eps 0.01-0.05, or sampling when the observation
  repeats).
- The score keeps rising (ghosts) while food plateaus at ~125: the agent
  learns to hunt ghosts rather than to clear the board; a reward that values
  the last dots more (e.g. a bonus growing as food runs out) would target
  that directly.
- Seeds matter as much as any change tried: with the same config dqn1s1
  reaches ~165 food and dqn1s2 stays at ~100. Comparing configs on one seed
  (everything before dqn1s1/s2) cannot separate small effects; 3 seeds per
  config would be the minimum.
- The food-distance channel and steps-since-food did not help (dqn5/dqn6
  below plain dqn1 at the same steps). Untried variants: the channel without
  steps-since-food, or only near pacman (a local crop), and the effect of the
  shorter epsilon schedule (2M vs 3M) on its own.
- Picking checkpoints by 20 games on seeds 0..19 and evaluating on 0..99
  overlaps; a separate selection seed range would remove the bias (here the
  numbers on seeds 20..99 were within 0.5 food of the 100-game ones).
- Train the end of the level directly: start some episodes from boards with
  few dots left (needs a game option to remove dots), since that is the part
  no agent learned.

- The two dots at (25, 19) and (27, 19) that dqn1s1 never eats: check
  whether the agent ever visits that pocket in training (state visitation
  counts), and whether episodes starting there would fix it.

# Night session 2026-10-06 22:39 .. 2026-10-07 08:00: clearing level 1

Goal: clear level 1. Starting point: dqn1s1/best_food.pt eats 168.8 of 172
and leaves the same two dots (25,19), (27,19) in every checked game; seeds
differ (101-169). Plan as given: 1) endgame diagnostics, 2) endgame
prefixes (heuristic play up to <= 15 food left, replayed in PacmanEnv
resets), 3) fine-tunes dqn7 (prefixes) and dqn8 (prefixes + endgame dot
reward) from dqn1s1/best_food.pt, 4) branch on the result, 5) README draft.
Rules unchanged (<= 2 runs in parallel, game/ untouched, disk > 10 GB and
GPU < 6 GB before a launch, checkpoint_buffer false, one --resume per crash,
best_food.pt, 100-game evaluations with eps 0 and 0.05).

## Chronology (night 2)

- 22:39 Start. Disk 30 GB free, GPU 15 MB of 8 GB used, nothing running.
- 22:47 **Step 1, endgame diagnostics** (`scripts/diag_endgame.py`, 100
  games each, seeds 0..99, level 1 only; maps in `docs/leftover_<name>.png`,
  data in `runs/diag/endgame_<name>.json`). "Pocket" = the side loop
  (24..27, 19) + (27, 18) holding the two dots.

  | agent | level 1 cleared | mean food left | games with (25,19)/(27,19) left | pacman steps on them | steps in the pocket | cells left in 100% of games |
  |---|---|---|---|---|---|---|
  | heuristic | 82% | 3.3 | 7 / 7 | 95 / 94 | 462 | none |
  | dqn1s1 best_food, eps 0 | 0% | 3.2 | 100 / 100 | 0 / 0 | 0 | (25,19), (27,19) |
  | dqn1s1 best_food, eps 0.05 | 0% | 12.6 | 100 / 100 | 0 / 0 | 3 | (25,19), (27,19) |
  | dqn1 seed 0 best_food, eps 0 | 0% | 20.4 | 15 / 15 | 85 / 87 | 430 | (31,9), (31,7), (29,15), (27,15), (25,15) |
  | dqn1s2 best_food, eps 0 | 0% | 70.5 | 7 / 7 | 96 / 95 | 475 | (27,9), (27,7), (27,5), (27,3), (25,9) |

  Answers: the heuristic does eat (25,19)/(27,19) (left in 7% of games).
  dqn1s1 never enters the pocket with eps 0 (0 steps in 100 games) and only
  3 steps with eps 0.05. Seeds 0 and 2 do go there; they have blind spots of
  their own elsewhere (each leaves its own cells in all 100 games). So every
  seed learned one fixed route with a region it never visits.
- 22:45 **Step 2**: `tools/record_prefixes.py` kept 28 of 100 seeds (the
  heuristic lost a life before <= 15 food left in the other 72), prefixes of
  395..533 steps, `data/endgame_prefixes.json`. In 14 of the 28 a pocket dot
  is still on the board at the prefix end. PacmanEnv `prefix_prob` replays
  a random prefix on reset (tested: <= 15 food, 3 lives, level 1;
  prefix_prob 0 changes nothing). `endgame_dot` reward option (tested).
  train.py: `init_checkpoint`, `track_cells` (pocket steps in each log line),
  prefixed-episode count; evaluation without prefixes and bonus. 56 tests pass.
- 22:48 **Step 3** started: supervisor (dqn7, dqn8), then dqn7
  (`configs/dqn_endgame.yaml`: from dqn1s1/best_food.pt, prefix_prob 0.5,
  5M steps, eps 0.1 -> 0.05 over 1M, lr 5e-5) and dqn8
  (`configs/dqn_endgame_bonus.yaml`: dqn7 + endgame dot reward k = 20).
  Disk 30 GB free, GPU 15 MB used before the launch.

## Unclear points and how they were resolved (night 2)

- "<= 15 points left": counted as food items (dots + energizers) on the
  board, the same unit as the food metric.
- Fine-tune: online and target networks start from best_food.pt; the
  optimizer, replay buffer, step counter and schedules start fresh
  (learning_starts 20000 steps before the first update, as in dqn.yaml).
- dqn8 "remaining": food items left after the dot is eaten; the last dot
  gets 1 * (1 + 20 / 1) = 21. Only dots get the bonus, not energizers.
- Training episodes from a prefix start at the prefix end; the replayed
  steps are not given to the agent.
- README "patch of ~12 lines": the real size is 63 added / 4 removed lines
  in 10 original files (mostly argument parsing in arg.cc) plus the new
  rlbridge.cc/.h (157 lines); the README says that.
- README throughput "39k steps/s": measured on 2026-10-05 (git log, commit
  b13cbc3) but not recorded in docs/results.md, so the README says TODO
  until it is measured again and added there.

