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

## Ideas for later

(Not tried, as instructed.)
