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
- 22:50 **Crash, my bug.** dqn7 crashed at 0.2M steps, before its first
  checkpoint: `info["prefix"]` was None after plain resets and an int after
  prefixed ones, and gymnasium's SyncVectorEnv cannot stack both. The
  supervisor's one `--resume` failed at once (no checkpoint yet), so it left
  dqn7 down as designed. dqn8 had the same bug waiting.
- 22:52 Fixed (`prefix` = -1 without a prefix; a test with mixed resets in a
  vector env; 57 tests pass), both runs and the supervisor restarted from
  scratch with the same configs. The failed attempts are kept as
  `runs/dqn7_crashed`, `runs/dqn8_crashed`; about 3 minutes of training lost.
- 23:00-23:20 Both runs dipped just under 1.5k steps/s (20-line mean 1473-1494)
  while I recorded the GIF and the supervisor evaluated; no new runs were
  pending anyway.
- 23:49 dqn8 finished (5M, no crash). Supervisor evaluations (20 games,
  seeds 0..19, eps 0, no prefixes): **level 1 cleared 55% at 3.0M, 70% at
  3.5M, 65% at 4.0M, 55% at 4.5M and 5.0M, no stuck games** (0-5% before
  2.5M). best_food.pt = 3.5M (260.3 food per game: it goes on into level 2).
  dqn7 (prefixes, no endgame bonus) never cleared level 1 in these
  evaluations up to 4.5M. Pocket steps in training: dqn7 and dqn8 enter the
  pocket now (401 / 604 steps in the first ~1.7M; dqn1s1 had 0 in 100 games).
  Final 100-game evaluations started automatically.
- 23:52 **dqn8 final (best_food.pt = 3.5M), 100 games, no prefixes: SUCCESS.**
  eps 0: level 1 cleared **67%** (66% on seeds 20..99), mean food 246.2,
  mean reward 214.8, median score 5030, 1% stuck. eps 0.05: level 1 only
  4%, food 159.5, reward 102.4: random moves now cost the precise endgame.
  Without ghosts it still stops for good (32 food at eps 0).
- 23:58 **dqn7 final (best_food.pt = 4.0M): FAILURE** by the night-2
  criteria: eps 0 food 169.4 (< 170), level 1 0%, reward 40.4, 29% stuck;
  eps 0.05 food 161.7, level 1 0%. (final_eval.py prints "SUCCESS" by the
  afternoon's food >= 160 rule; the night-2 verdict is from
  `scripts/night2_verdict.py`.) Prefixes alone did not clear the level; the
  endgame dot reward did.
- 23:59 **Step 4, SUCCESS branch**: dqn8's config fine-tuned from dqn1 seed 0
  (dqn8s0) and seed 2 (dqn8s2) best_food.pt, in parallel, 5M steps each,
  supervisor started first. Disk 29 GB free, GPU 15 MB used.
- 00:58-01:07 **Step 4 results** (no crash; both runs dipped to 1.39-1.46k
  steps/s at times, 3 supervisor slow notes; no new runs were pending).
  100 games, eps 0, no prefixes:
  - dqn8s0 (from dqn1 seed 0, best_food.pt = 5.0M): FAILURE, food 153.9
    (start: 151.6), level 1 0%, reward 67.1, 6% stuck; eps 0.05: 143.4 / 0%.
  - dqn8s2 (from dqn1 seed 2, best_food.pt = 4.5M): FAILURE, food 103.2
    (start: 101.5), level 1 0%, reward -40.8, 15% stuck; eps 0.05: 102.5 / 0%.
  - Neither cleared level 1 in any supervisor evaluation either.
  - dqn8 setup over the three starting networks (seeds 0, 1, 2): food
    167.8 +- 59.2, level 1 cleared 67% / 0% / 0% (mean 22%). The recipe
    worked only from seed 1, whose blind spot was the pocket the prefixes
    happen to cover; seeds 0 and 2 miss other regions (step 1).
- 01:07 Throughput measured on the idle machine and added to
  docs/results.md: single env 32.4k steps/s (39k on 2026-10-05 with the
  smaller observation of the time).
- 01:13 Why step 4 failed, checked (analysis only, no new training):
  - Endgame maps of the fine-tunes (100 games, eps 0, `docs/leftover_dqn8*_eps0.png`):
    dqn8 now eats the pocket (its dots left in 3 / 1 games) and no cell is
    left in all games (most often (7,13), 20 games). dqn8s0 still leaves
    (31,11), (31,9), (31,7), (29,15), (27,15) in all 100 games, dqn8s2 still
    leaves (27,3..9), (25,9) in all 100: their blind spots did not move.
  - Prefix coverage (food still on those cells at the prefix end): seed 1's
    pocket in 14 of 28 prefixes, seed 0's cells in 11 of 28, seed 2's in 0
    of 28. So seed 2 never saw its region as an endgame, but seed 0 did and
    still did not learn it; coverage is not the whole story.
- 01:14 **Step 5**: README.md draft finished (results from docs/results.md,
  maps, dqn8 GIF, what worked / did not, known problem, future work). Open
  TODO: the author name for the authorship line. Nothing is training; the
  night's runs are done (step 4 SUCCESS branch completed; per the plan no
  further configs were started).

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
- dqn7's crash was a code bug before any checkpoint existed, so "one
  --resume per crash" could not apply. Conservative choice: fix, test and
  restart both runs from scratch (nothing to lose), and say so here.
- Step 4 "the same 5M from seed 0 and seed 2": only the starting network
  differs; the training seed stays 0 as for dqn8, the prefixes and all other
  settings are the same (`configs/dqn_endgame_bonus_from_s0/s2.yaml`).
- README throughput "39k steps/s": measured on 2026-10-05 (git log, commit
  b13cbc3) but not recorded in docs/results.md, so the README says TODO
  until it is measured again and added there.


## Ideas for later (night 2)

(Not tried, as instructed.)

- Endgame prefixes per agent: record openings from the agent's own games
  (or the heuristic's) that stop with its own blind-spot cells still full,
  so seeds 0 and 2 see their regions as endgames (seed 2's cells were in no
  prefix at all).
- dqn8 with eps 0.05 clears level 1 in 4% of games instead of 67%: check
  whether a smaller eps (0.01) keeps the fixed-point escapes without
  breaking the endgame, or randomness only when the observation repeats.
- The endgame dot reward without prefixes (dqn8 had both; dqn7 shows that
  prefixes alone do not clear the level, but the reward alone is untested).
- Train the dqn8 recipe from scratch rather than as a fine-tune, and with
  several training seeds from the same starting network.
- Level 2+: dqn8 eats ~246 food per game, so it plays on into level 2 (a
  different maze); nothing was measured there yet.

# Session 2026-10-07 14:50: dqn8 from 67% to >= 85% of level 1 clears

Goal: level 1 cleared in >= 85% of 100 greedy games (PARTIAL: >= 75%).
Checkpoints are picked on seeds 1000..1019 (supervisor evaluations, 20
games), final evaluations on seeds 0..99, so the two never overlap. Rules
as before (<= 2 runs in parallel, game/ untouched, a commit per step).

## Chronology (2026-10-07)

- 14:50 Start. Disk 21 GB free, GPU 15 MB used, nothing running.
- 14:52 **Step 1, dqn8's lost games** (`scripts/diag_failures.py`,
  best_food.pt, eps 0, seeds 0..99; `docs/dqn8_failures.png`,
  `runs/diag/failures_dqn8.json`): 67 cleared, **32 died** (all lives lost on
  level 1), **1 stuck** (seed 90: 3 food left, 9377 steps without food).
  - Every life was lost by meeting a ghost in normal state (distance 0); the
    death cells are spread over the maze (192 lives lost on level 1 in all
    games, at most 6 in one cell, (9, 11)).
  - The deaths come late: food left at the last death median 7, <= 15 in 25
    of the 32 died games, <= 4 in 15 of them. Even the cleared games lose
    lives (94 in 67 games; only 6 cleared without a death).
  - Food left in >= 5 failed games: yes, e.g. (7,13) in 20, (3,3) in 16,
    (23,5) in 12, (29,3) in 11, the row (13..19, 3) in 6-8. These are where
    the agent was going when the last life went, not never-visited cells:
    only one failed game ended stuck.
  - **Conclusion: the main cause is death, not blind spots.** So dqn10 =
    dqn9 + PER (alpha 0.6, beta 0.4 -> 1.0), per the plan.
- 14:54 `tools/record_prefixes.py --checkpoint`: dqn8's own greedy games on
  env seeds 2000..2599 up to <= 15 food left on level 1, no life lost: 189 of
  600 kept (411 lost a life first), 362..577 steps, `data/dqn8_prefixes.json`;
  replays checked (15 food, 3 lives).
- 14:56 **Step 2 started**: supervisor (dqn9, dqn10; evaluations on seeds
  1000..1019), dqn9 (`configs/dqn_own_prefixes.yaml`) and dqn10
  (`configs/dqn_own_prefixes_per.yaml`, + PER alpha 0.6, beta 0.4 -> 1.0).
  Both: from dqn8/best_food.pt, 5M steps, eps 0.05 -> 0.02 over 1M, lr 3e-5,
  prefix_prob 0.5 with dqn8's prefixes. Disk 21 GB free, GPU 15 MB.
- 15:47 dqn9 finished (no crash). Supervisor evaluations (seeds 1000..1019):
  level 1 cleared 70, 45, 55, 65, 85, 70, 60, 85, 70, 85% at 0.5..5M;
  best_food.pt = 5.0M (291.4 food).
- 15:57 **dqn9 final, 100 games, seeds 0..99: level 1 cleared 79%** (79% on
  seeds 20..99), food 274.4, reward 255.3, median score 6050, none stuck.
  eps 0.05: 12%. Without ghosts it still stops (86 food, eps 0).
- 15:59 dqn10 finished (no crash; 5 supervisor notes of 1.28-1.41k steps/s:
  PER is slower, nothing else was waiting). Evaluations: 50, 35, 75, 80, 70,
  90, 65, 80, .. % ; best_food.pt = 5.0M (307.1 food).
  **dqn10 final: level 1 cleared 81%** (81% on seeds 20..99), food 284.1,
  reward 267.0, median score 6335, none stuck; eps 0.05: 12%.
- **Step 2 verdict: both PARTIAL** (79%, 81%: >= 75%, < 85%). The best is
  dqn10, so step 3 = continue dqn10 with --resume for 5M more.
  `train.py --resume --extend-steps N` added (raises the run's total_steps;
  tested), since a plain --resume of a finished run does nothing.
- 16:04 **Step 3**: dqn10 resumed from 5.0M with --extend-steps 5000000
  (supervisor kept its evaluated points). Finished 16:52, no crash.
  Selection evaluations 5.5..10M: level 1 cleared 60, 85, 80, 85, 85, 75,
  90, **95 (9.0M)**, 80, 85%; mean food at most 304.9, so best_food.pt stayed
  at 5.0M (307.1).
- 16:54 Official final (best_food.pt rule): unchanged, the same 5.0M
  checkpoint, **81% -> PARTIAL after the extra 5M; per the plan: stop.**
- 16:56 Extra, labelled separately: dqn10's 9.0M checkpoint (best level 1
  clears on the selection seeds 1000..1019) on seeds 0..99: **85%** (84% on
  20..99), food 301.4, reward 280.7, 3% stuck; eps 0.05: 13%.
- Conclusion: own-game endgame prefixes (+ PER) moved dqn8 from 67% to 79-81%
  by the food rule; 85% is reached only when the checkpoint is picked by
  level clears, and only at the threshold (+-3.6 points on 100 games). The
  seed 1 / seed 2 repeats were not started (see decisions).
- 17:18 **Decision (user): the checkpoint rule is level 1 clears on seeds
  1000..1019, food the second key**, from v0.2 on; dqn10 = 9.0M, 85%. No
  seed 1 / seed 2 repeats. Fixed in code: `best_metric: level1_cleared`
  (train.py, train_ppo.py; evaluations now count food per game),
  `eval_seed: 1000` in configs/dqn.yaml and ppo.yaml, supervisor default
  `--eval-seed 1000`; README states the rule.

## Unclear points and decisions (2026-10-07)

- Overlap found in the night-2 setup: the heuristic prefixes were recorded
  on env seeds 0..99, the same seeds as the final evaluations, so dqn8's
  training started some episodes from openings of the evaluation games'
  seeds. The games diverge once the agent's moves differ (ghosts react to
  pacman, the game's random numbers are drawn every tick), so this is weak,
  but it is not a clean split. New prefixes in this session are recorded on
  env seeds 2000 and up, disjoint from 0..99 and 1000..1019.
- dqn9/dqn10 keep dqn8's endgame dot reward (k 20): they fine-tune dqn8 in
  its own environment, only the prefixes (and PER for dqn10) change.
- "eps 0.05 -> 0.02" without a length: decayed over the first 1M steps, as
  the night-2 fine-tunes did with 0.1 -> 0.05.
- dqn8's prefixes keep only games without a lost life (as for the heuristic
  prefixes; PacmanEnv checks 3 lives after a replay). That keeps 189 of 600
  and leaves out exactly the games where dqn8 dies early, which may matter
  given that deaths are the main cause.
- Step 3 continuation of dqn10: from its last checkpoint (5.0M, also its
  best_food.pt); the replay buffer was not saved (checkpoint_buffer false),
  so it refills for learning_starts (20k steps) and PER priorities start
  fresh; epsilon stays at 0.02 (its schedule ended at 1M).

- Checkpoint rule vs target: best_food.pt (most food) was the rule, but the
  target is level clears and food stopped tracking them once the agent eats
  into level 2 (more food also comes from playing on). The level-picked 9.0M
  checkpoint is reported separately and released next to the rule's pick;
  which rule counts is for the user to decide.
- Step 3 says "if one reached >= 85%, repeat the fine-tune with seeds 1 and
  2". By the rule in force none did (81%), so the conservative reading was to
  stop; the repeats (2 x 10M, about 2 hours) were not started.

# Session 2026-10-07 22:32: v0.3, all 16 mazes

Goal: an agent that plays on all 16 mazes and clears levels in a row.
Rules as before (<= 2 runs in parallel, this log, a commit per step, disk and
GPU checked, one --resume per crash). Checkpoints picked by mean levels
cleared per game from level 1 on seeds 1000..1019, final numbers on seeds
0..99.

## Chronology (2026-10-07/08)

- 22:32 Start. Disk 25 GB free, GPU 15 MB used, nothing running.
- 22:36 **Step 1**: `game/pacman --level N` (arg.cc/arg.h: parsed and taken
  out of the arguments like --seed; needs --rl; pac.cc: after both
  `da->start()` calls, `da->setlevel(N)`, i.e. level and boardlevel by
  `Gamedata::setboardlevel`). 9 lines, all "// RL bridge"; without --level
  nothing changes (tested: --level 1 gives the same 3000 states as no flag).
  Env: `start_level` (int or "random:a-b", drawn from the env's np_random at
  each reset, a new game process when the level changes; after a game over
  the game restarts on the same level), `reset(options={"start_level": ..})`,
  `set_start_level()`. Prefix replays always use level 1. Tests in
  test_rlbridge.py/test_env.py.
- 22:38 **Step 3** (done before step 2, which uses it): `evaluate.py
  --levels out.json` (games from level 1 + games from each of the 16 levels,
  in parallel CPU processes, games cut at 30000 ticks); train.py:
  `best_metric: levels_cleared` (mean levels cleared, ties by food; the
  evaluation always starts on level 1), `eval_max_episode_steps`,
  `start_level_schedule` (curriculum). The old `run()` gives the same
  heuristic row as before (100 games, identical numbers).
- 22:39 **Step 2**: heuristic, 50 games per level: 56-92%, no maze under
  30%; from level 1 median 1, mean 1.58 levels (table in results.md).
- 22:39 Width smoke test started: `dqn_mm_smoke` (standard width) and
  `dqn_mm_smoke_wide` (64,128,128 channels, FC 1024), each mm1's setup for
  2M steps, in parallel.
- 22:44 dqn10 (9.0M) on all mazes: 90% on maze 1, 0% on the 15 others, never
  clears level 2 after level 1 (mean 0.84 levels from level 1).
- 23:08 Smoke test done (2M steps each, both runs at the same time): standard
  width 1194 steps/s on average, wide 1134 steps/s, **wide 5% slower** (limit
  25%), so mm1 and mm2 use the wide network (`configs/dqn_mm_wide.yaml`). At
  2M neither clears a level (evaluation from level 1: 17 vs 26 food per game).
- 23:09 **Step 4 started**: mm1 (`configs/dqn_mm1.yaml`, start_level
  random:1-16) and mm2 (`configs/dqn_mm2.yaml`, random:1-4 / 1-8 from 10M /
  1-16 from 20M), seed 0 each, supervisor (`--every 5000000`, its rows go to
  `runs/mm_sv_results.md`, not results.md). Disk 25 GB free, GPU 776 MB.
- 23:41 mm1 1.9M, mm2 2.0M steps; speed fell from ~1.2k to ~950 steps/s per
  run (each training process holds one core at ~90%, load average 3 of 20
  CPUs, GPU 74%: the Python loop is the limit, not contention). At this pace
  30M ends around 07:30-07:45, later than the ~5-6 h in the plan. Plan: no
  restart; if the runs are not done by ~06:45, the final evaluation runs on
  their best.pt at that point (labelled with its step count) and is repeated
  if they finish in time. Disk 18 GB free (replay buffers saved, 3.3 GB each).
- 00:42 mm1 5.3M, mm2 5.5M (~950-990 steps/s; projected end of 30M ~07:50).
  Selection evals so far: no level cleared yet. mm2 (mazes 1-4) eats more on
  maze 1 (98-106 food at 3-5M) but its games stretch to 2500 ticks at 5M
  (wandering without food); mm1 110 food at 5M in short games (deaths).
  Disk 18 GB free, GPU 776 MB.
- 01:43 mm1 8.8M, mm2 9.1M. Selection evals: 0 levels cleared at every point
  so far; food per game from level 1: mm1 110-121 (5-8M), mm2 106-110
  (5-9M, games 1700-2500 ticks). Training episodes (TensorBoard, last 1M
  steps): not one cleared level in 1654 (mm1) / 940 (mm2) episodes, on any
  start maze. As expected from earlier from-scratch runs on maze 1 alone
  (dqn0/dqn1: 0% level 1; level clears came only with prefixes and
  fine-tuning in dqn7-dqn10). Disk 18 GB free.
- 02:44-02:55 Probe (not the final evaluation): levels suite on both runs'
  best.pt at that time (11.0M each; seeds 0..99 / 0..19 per level, 5 CPU
  workers each, ~8 and ~10 min; training slowed to ~880 steps/s meanwhile).
  Both clear 0% on all 16 mazes. Food per game by start maze: mm1 81-130 on
  every maze (even); mm2 107-126 on mazes 1-4, 29-62 on 5-8 (1M steps of
  training there so far), 8-18 on the unseen 9-16, the same "does not play"
  pattern as dqn10 off maze 1. JSON: `runs/levels/mm*_11M_probe.json`.
- 03:55 mm1 16.1M, mm2 16.4M, no crashes. Selection evals 12-16M: 0 levels;
  food mm1 125-133, mm2 112-115. Projected end of 30M: ~08:00, so the final
  evaluation will be on best.pt as of ~06:45 (about 26M), as planned at 23:41.
  Disk 17 GB free.
- 04:56 mm1 19.6M, mm2 19.9M (switches to random:1-16 at 20M), no crashes.
  Evals 17-19M: 0 levels; food mm1 133-136, mm2 110-130. Disk 17 GB free.
- 05:47 mm1 22.5M, mm2 22.7M; evals 20-25M: 0 levels, food mm1 132-139,
  mm2 129-134 (games up to 4600 ticks).
- 06:38 **Step 5 (final evaluation, early)**: best.pt of both runs = 24.0M
  (copied to `best_24000000.pt`), levels suite, 8 CPU workers each (13 min;
  training slowed to ~370 steps/s meanwhile). **mm1 and mm2: 0 levels cleared
  from level 1 (100 games), 0% on each of the 16 mazes (20 games each).**
  Food per game: mm1 95-141 on every maze, mm2 110-138 on mazes 1-6 and
  21-102 on 7-16. Over the whole run not one training episode cleared a level
  (43818 / 38427 episodes up to 25.8M). Comparison: dqn10 (9.0M) 90% on maze
  1, 0% elsewhere; heuristic 56-92% everywhere.
- **Verdict: v0.3 criterion not met** (needs >= 3 levels in a row on average
  or >= 50% on every maze; the best is 0). Per the plan: results.md and
  conclusions only, no tag, no release, no README section. The best of the
  two by the selection metric is a tie at 0; by food from level 1 mm1
  (135.9 vs 133.9) and by evenness over the mazes mm1.
- 06:55 Plots `docs/eval_levels.png`, `docs/levels_mm1.png`,
  `docs/levels_mm2.png`; results.md section. mm1/mm2 keep training to 30M
  (about 08:00-08:10, no crashes so far); their last evaluations are not in
  this report.

## Unclear points and decisions (2026-10-07/08)

- `--level N` without `--rl` exits with an error (like `--headless`): the
  plan says nothing may change without --rl, and an accepted but ignored flag
  would have changed which arguments colour.cc sees.
- Level > 16 is allowed (the game then draws a random maze per
  `setboardlevel`); the env's `random:a-b` is checked to be >= 1.
- "levels_cleared per game": highest level reached minus the start level,
  over a whole game (3 lives, until game over). The suite cuts a game at
  30000 ticks (`--max-steps`, also `eval_max_episode_steps` in training);
  training episodes keep the 10000-step limit.
- "share of the level cleared per level when starting on it" read as: the
  share of games that clear their start level (reach start + 1).
- Step 2 table: 50 games per level as asked, seeds 0..49; levels in a row
  from level 1 on 100 games (seeds 0..99, like the final evaluations).
- The selection metric (mean levels cleared, food second) stayed at 0 for
  every evaluation of both runs, so best.pt was in effect picked by mean
  food from level 1 on seeds 1000..1019.
- Smoke test: both widths ran at the same time with mm1's setup, so they
  shared the machine equally; the comparison is the mean of all 200 log
  lines (1194 vs 1134 steps/s).
- eval_every 1M instead of the configs' usual 500k: the runs were slower than
  the plan assumed (~1.0-1.2k steps/s), and 30 selection points were judged
  enough.
- **The runs could not finish 30M before 08:00** (~950 steps/s from 2M on;
  the Python loop holds one core per run). The final evaluation was done on
  best.pt as of 06:38 (24.0M for both, copied to `runs/mm*/best_24000000.pt`);
  the runs keep training to 30M after the report (~08:00). If a later
  evaluation beats 24M on the selection seeds, the numbers should be
  re-taken; the curves give no sign of a level clear coming.
- dqn10's row uses its 9.0M checkpoint (the v0.2 release pick).

## Ideas for later (v0.3)

(Not tried.)

- What made dqn8-dqn10 clear maze 1 was not more steps from scratch but the
  endgame curriculum (prefixes ending with <= 15 food left, then own-game
  prefixes) on top of the endgame dot reward. The multi-maze runs had the
  reward but no prefixes (as planned), and both stall at ~135 of 172 food,
  the same plateau as dqn0/dqn1 on maze 1 alone. Next: endgame prefixes on
  all 16 mazes (heuristic games per maze up to <= 15 food left, recorded with
  `--level`; the heuristic clears 56-92% of each maze, so prefixes are cheap).
- Fine-tune mm1 (it plays evenly on all mazes) with those prefixes, like
  dqn7 -> dqn10 did from dqn1.
- mm2's curriculum gave a better maze-1 player early, but on unseen mazes it
  played no better than dqn10 (probe at 11M), and its games grow long
  (wandering): a hunger limit or steps-since-food input might help there.
- Throughput: ~950 steps/s per run is the limit for 30M-step plans (8.5 h).
  AsyncVectorEnv with more envs, or moving the PER sum tree to numpy
  batched ops, would be the first places to look.

# Session 2026-10-08 11:42: v0.3 continued, endgame prefixes on 16 mazes

mm1 plays all 16 mazes but stalls at ~135 of 172 food, the plateau dqn1 had
on maze 1 before endgame prefixes + fine-tuning (dqn7 -> dqn10) broke it.
Same recipe on 16 mazes. Selection: mean levels cleared from level 1 on
seeds 1000..1019; final: seeds 0..99 and 20 games per maze. Criterion:
mean levels >= 1.0 from level 1 and >= 40% on each maze (PARTIAL: >= 25%
mean over mazes -> --resume the best +10M; FAILURE: < 10% -> stop).

## Chronology (2026-10-08)

- 11:42 Start. mm1 and mm2 finished 30M at 08:09, no crashes; 25-30M
  selection evals all 0 levels (mm1 food 132.9-139.2, mm2 131.2-135.5).
  **mm1's best.pt is 27.0M** (139.2 food vs 138.9 at 24.0M, levels 0 both:
  food decides), copied to `runs/mm1/best_27000000.pt`, the start of mm3/mm4.
  Disk 17 GB free, GPU 15 MB.
- 11:43 **Step 2**: `tools/record_prefixes.py --level 1-16 --keep 25
  --seed-start 3000 --seeds 400` (levels in parallel, 9 s):
  `data/endgame_prefixes_all.json`, 25 prefixes per maze (400), heuristic
  games up to <= 15 food left with no life lost; seeds tried per maze 62-154
  (maze 15 needs the most: the heuristic loses a life first in 129 of 154),
  prefix lengths 363..873 steps (one of 2835 on maze 13). Each prefix stores
  its "level"; PacmanEnv replays it with game --level of that level and
  checks food left, lives and level after the replay. All 400 replays
  checked; test_prefix_on_its_own_level.
- 11:46 **Step 3 started**: mm3 (`configs/dqn_mm3.yaml`, prefix_prob 0.5) and
  mm4 (`configs/dqn_mm4.yaml`, 0.7): from mm1 27.0M, 10M steps, eps 0.05 ->
  0.02 over 1M, lr 3e-5, PER, start_level random:1-16 for resets without a
  prefix, evals every 500k; supervisor (`--every 1000000`).
- 11:51 Speed 940-980 steps/s per run (>= 800), so the AsyncVectorEnv x8
  smoke test was not needed; SyncVectorEnv stays. Expected end ~14:50.
