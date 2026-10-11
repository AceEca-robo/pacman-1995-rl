#!/usr/bin/env python3
"""Watch training runs (each in a tmux session named like the run).

    python scripts/supervisor.py dqn1 dqn3b [--every 500000] [--episodes 20]

Every --every env steps of each run (as soon as the checkpoint for that step
is written) it copies runs/<run>/checkpoint.pt to ck_<steps>.pt, evaluates it
with scripts/evaluate.py (--episodes greedy episodes, seeds 0.., env without
shaping), which appends a row "dqn <run>@<steps>M" to docs/results.md, writes
per-episode stats to runs/<run>/sv_eval_<steps>.jsonl and redraws
docs/eval.png.

If a run's tmux session disappears before the run reached its total_steps,
it is restarted once with --resume (train.py or train_ppo.py, by the run's
"algo"); a second crash is only logged. A mean speed under --min-speed
steps/s over the last 20 log lines is logged (at most every 10 min). Everything goes to runs/supervisor.log;
state (restart counts, done evals) in runs/supervisor_state.json.

--train-eval scores the copies with scripts/train_eval.py on the run's own
eval protocol instead (games from level 1 + games on each maze, seeds
eval_seed..; train.py's evaluate()) and appends the summaries with "steps" to
runs/<run>/sv_evals.jsonl (no results.md row, no plot).

--extend applies the long-run rule when a run reaches its total_steps: if the
mean of the last two evaluations beats the mean of the two before on the
per-maze clear share or on levels cleared from level 1, --resume
--extend-steps --extend-rise; otherwise (a plateau confirmation) extend by
--extend-plateau, and stop at the --plateau-stop-th confirmation in a row.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pick_best_food import pick as pick_best_food  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv", "bin", "python")
RUNS = os.path.join(ROOT, "runs")
LOG = os.path.join(RUNS, "supervisor.log")
STATE = os.path.join(RUNS, "supervisor_state.json")
RESULTS = os.path.join(ROOT, "docs", "results.md")
EVAL_SEED = 1000  # first seed of the supervisor's evaluation games (--eval-seed)
PLOT_DIR = os.path.join(ROOT, "docs")
PLOT_RUNS = ["dqn0", "dqn2"]  # finished runs kept on docs/eval.png for reference


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def alive(session):
    return subprocess.run(["tmux", "has-session", "-t", session],
                          capture_output=True).returncode == 0


def run_log(run):
    path = os.path.join(RUNS, f"{run}.log")
    if not os.path.exists(path):
        return ""
    with open(path, errors="replace") as f:
        return f.read()


def total_steps(run):
    """None until train.py has written the run's config."""
    path = os.path.join(RUNS, run, "config.yaml")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return yaml.safe_load(f)["total_steps"]


_ck_cache = {}  # path -> (mtime, steps)


def read_steps(path):
    import torch
    try:
        mtime = os.path.getmtime(path)
    except FileNotFoundError:
        return None
    if path in _ck_cache and _ck_cache[path][0] == mtime:
        return _ck_cache[path][1]
    try:
        steps = torch.load(path, map_location="cpu", weights_only=False)["steps"]
    except Exception as e:  # being replaced right now; next poll
        log(f"could not read {path} ({e!r})")
        return None
    _ck_cache[path] = (mtime, steps)
    return steps


def sv_evals(run):
    path = os.path.join(RUNS, run, "sv_evals.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return sorted(rows, key=lambda r: r["steps"])


def train_eval(run, steps, ck):
    tmp = os.path.join(RUNS, run, "sv_evals.tmp.jsonl")
    if os.path.exists(tmp):
        os.remove(tmp)
    out = subprocess.run(
        [PY, os.path.join(ROOT, "scripts", "train_eval.py"),
         "--config", os.path.join(RUNS, run, "config.yaml"), "--checkpoint", ck,
         "--label", f"{run}@{steps}", "--out", tmp],
        capture_output=True, text=True, cwd=ROOT)
    if out.returncode != 0:
        log(f"{run}: train_eval.py failed at {steps}: {out.stderr.strip()[-500:]}")
        return
    with open(tmp) as f:
        row = {"steps": steps, **json.loads(f.read().strip().splitlines()[-1])}
    os.remove(tmp)
    with open(os.path.join(RUNS, run, "sv_evals.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    log(f"{run}: eval @ {steps}: per-maze mean {row['mean_maze_cleared']:.1%}, "
        f"min {row['min_maze_cleared']:.0%}, levels from level 1 "
        f"{row['mean_levels_cleared']:.2f}, food {row['mean_food']:.0f}")


def evaluate(run, steps, episodes, train_eval_mode=False):
    ck = os.path.join(RUNS, run, f"ck_{steps}.pt")
    shutil.copy(os.path.join(RUNS, run, "checkpoint.pt"), ck + ".tmp")
    os.replace(ck + ".tmp", ck)
    got = read_steps(ck)
    if got != steps:  # training replaced checkpoint.pt in between
        log(f"{run}: wanted the {steps} checkpoint, copied {got}; skipped")
        os.remove(ck)
        return
    if train_eval_mode:
        train_eval(run, steps, ck)
        return
    algo = run_algo(run)
    label = f"{algo} {run}@{steps / 1e6:05.2f}M"
    out = subprocess.run(
        [PY, os.path.join(ROOT, "scripts", "evaluate.py"), "--agent", algo, "--checkpoint", ck,
         "--episodes", str(episodes), "--seed", str(EVAL_SEED), "--label", label,
         "--episodes-out", os.path.join(RUNS, run, f"sv_eval_{steps}.jsonl"),
         "--results", RESULTS],
        capture_output=True, text=True, cwd=ROOT)
    if out.returncode != 0:
        log(f"{run}: evaluate.py failed at {steps}: {out.stderr.strip()[-500:]}")
        return
    log(f"{run}: eval @ {steps}: {out.stdout.strip().splitlines()[-1]}")
    best = pick_best_food(run)
    if best:
        log(f"{run}: best_food.pt = {best[0]} steps, mean food {best[1]:.1f}")


def plot(runs):
    have = [r for r in PLOT_RUNS if os.path.isdir(os.path.join(RUNS, r))]
    out = subprocess.run([PY, os.path.join(ROOT, "scripts", "plot_runs.py"), *have, *runs,
                          "--which", "eval", "--out-dir", PLOT_DIR],
                         capture_output=True, text=True, cwd=ROOT)
    if out.returncode != 0:
        log(f"plot_runs.py failed: {out.stderr.strip()[-300:]}")


def run_algo(run):
    """"dqn" or "ppo", from the run's config.yaml."""
    with open(os.path.join(RUNS, run, "config.yaml")) as f:
        return yaml.safe_load(f).get("algo", "dqn")


def restart(run, extend=0):
    script = "train_ppo.py" if run_algo(run) == "ppo" else "train.py"
    ext = f"--extend-steps {extend} " if extend else ""
    cmd = (f"cd {ROOT} && {PY} scripts/{script} --run-name {run} --resume {ext}"
           f"2>&1 | tee -a runs/{run}.log")
    subprocess.run(["tmux", "new-session", "-d", "-s", run, cmd], check=True)


SPEED_WINDOW = 20  # log lines


def recent_speed(text):
    """Mean steps/s over the last SPEED_WINDOW log lines (None until there are that many)."""
    m = re.findall(r"([\d,]+) steps/s", text)
    if len(m) < SPEED_WINDOW:
        return None
    return sum(int(x.replace(",", "")) for x in m[-SPEED_WINDOW:]) / SPEED_WINDOW


def main():
    global LOG, STATE, RESULTS, PLOT_DIR, EVAL_SEED
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs", nargs="+")
    p.add_argument("--every", type=int, default=500_000)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--poll", type=float, default=15.0, help="seconds between checks")
    p.add_argument("--min-speed", type=int, default=1500)
    p.add_argument("--eval-seed", type=int, default=1000,
                   help="first seed of the evaluation games (keep apart from the final eval)")
    p.add_argument("--results", default=RESULTS, help="markdown table evaluate.py updates")
    p.add_argument("--plot-dir", default=PLOT_DIR, help="where eval.png goes")
    p.add_argument("--log-dir", default=RUNS, help="supervisor.log and supervisor_state.json")
    p.add_argument("--train-eval", action="store_true",
                   help="evaluate with scripts/train_eval.py (the run's eval protocol)")
    p.add_argument("--extend", action="store_true", help="apply the extension rule (see above)")
    p.add_argument("--extend-rise", type=int, default=50_000_000)
    p.add_argument("--extend-plateau", type=int, default=30_000_000)
    p.add_argument("--plateau-stop", type=int, default=3)
    args = p.parse_args()
    LOG = os.path.join(args.log_dir, "supervisor.log")
    STATE = os.path.join(args.log_dir, "supervisor_state.json")
    RESULTS, PLOT_DIR, EVAL_SEED = args.results, args.plot_dir, args.eval_seed

    state = {}
    if os.path.exists(STATE):
        with open(STATE) as f:
            state = json.load(f)
    for run in args.runs:
        state.setdefault(run, {"restarts": 0, "status": "running", "evaluated": [],
                               "slow_logged_at": 0})

    def save():
        with open(STATE + ".tmp", "w") as f:
            json.dump(state, f, indent=1)
        os.replace(STATE + ".tmp", STATE)

    log(f"supervisor started for {', '.join(args.runs)}: eval every {args.every} steps, "
        f"{args.episodes} episodes on seeds {EVAL_SEED}..{EVAL_SEED + args.episodes - 1}")
    save()
    while any(state[r]["status"] == "running" for r in args.runs):
        for run in args.runs:
            st = state[run]
            if st["status"] != "running":
                continue
            try:
                check(run, st, args, save)
            except Exception as e:  # never let one bad poll stop the night
                log(f"{run}: supervisor error {e!r}")
        time.sleep(args.poll)
    plot(args.runs)
    log("all runs finished or failed; supervisor exits")


def check(run, st, args, save):
    text = run_log(run)
    total = total_steps(run)
    if total is None:
        return
    # PPO counts in rollouts of 1024 steps, so its last eval is at or just past total
    evals_at = [int(s) for s in re.findall(r"eval @ (\d+):", text)]
    finished = bool(evals_at) and max(evals_at) >= total

    steps = read_steps(os.path.join(RUNS, run, "checkpoint.pt"))
    due = [s for s in range(args.every, (steps or 0) + 1, args.every)
           if s not in st["evaluated"]]
    if due:
        # the first checkpoint at or after a point stands for it (DQN saves at
        # exact multiples, PPO at the first rollout past them)
        target = due[-1]
        fresh = steps - target < args.every // 2
        missed = due[:-1] if fresh else due
        if missed:  # only the latest checkpoint is kept by the training scripts
            log(f"{run}: no checkpoint left for {missed[0]}..{missed[-1]} "
                f"({len(missed)} points), skipped")
        if fresh:
            evaluate(run, steps, args.episodes, args.train_eval)
            if not args.train_eval:
                plot(args.runs)
        st["evaluated"].extend(due)
        save()

    speed = recent_speed(text)
    if speed is not None and speed < args.min_speed and time.time() - st["slow_logged_at"] > 600:
        log(f"{run}: slow, {speed:,.0f} steps/s over the last {SPEED_WINDOW} log lines "
            f"(< {args.min_speed}); no new runs while this lasts")
        st["slow_logged_at"] = time.time()  # at most every 10 min
        save()

    if alive(run):
        return
    extended_here = bool(st.get("extensions")) and st["extensions"][-1][0] == total
    if finished and args.extend and extended_here:
        log(f"{run}: the extension from {total} did not start (config.yaml not extended); "
            f"not restarting")
        st["status"] = "failed"
    elif finished and args.extend:
        extend_or_stop(run, st, total, args)
    elif finished:
        log(f"{run}: finished ({total} steps)")
        st["status"] = "finished"
    elif st["restarts"] == 0:
        tail = "\n    ".join(text.strip().splitlines()[-5:])
        log(f"{run}: CRASHED at checkpoint {steps}, restarting once with --resume; "
            f"log tail:\n    {tail}")
        st["restarts"] = 1
        restart(run)
    else:
        log(f"{run}: CRASHED again (checkpoint {steps}); not restarting")
        st["status"] = "failed"
    save()


def extend_or_stop(run, st, total, args):
    # checkpoints overshoot their step by up to one actor batch (50000003)
    rows = [r for r in sv_evals(run) if r["steps"] < total + args.every // 2]
    mm = [r["mean_maze_cleared"] for r in rows]
    lv = [r["mean_levels_cleared"] for r in rows]
    if len(rows) >= 4:
        def mean(x):
            return sum(x) / len(x)
        rise_m, rise_l = mean(mm[-2:]) > mean(mm[-4:-2]), mean(lv[-2:]) > mean(lv[-4:-2])
        what = (f"evals at {[r['steps'] // 1_000_000 for r in rows[-4:]]}M: per-maze mean "
                f"{mean(mm[-4:-2]):.3f} -> {mean(mm[-2:]):.3f}, levels "
                f"{mean(lv[-4:-2]):.3f} -> {mean(lv[-2:]):.3f}")
    else:
        rise_m = rise_l = True
        what = f"only {len(rows)} evaluations (< 4), counted as a rise"
    if rise_m or rise_l:
        st["plateau"] = 0
        n = args.extend_rise
        log(f"{run}: reached {total}; rise ({what}); extending by {n}")
    else:
        st["plateau"] = st.get("plateau", 0) + 1
        if st["plateau"] >= args.plateau_stop:
            log(f"{run}: reached {total}; plateau confirmation {st['plateau']} ({what}); STOP")
            st["status"] = "finished"
            return
        n = args.extend_plateau
        log(f"{run}: reached {total}; plateau confirmation {st['plateau']} ({what}); "
            f"extending by {n} anyway")
    st.setdefault("extensions", []).append([total, n])
    restart(run, n)


if __name__ == "__main__":
    sys.exit(main())
