import json
import os
import subprocess
import sys
import time

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def wait_for(cond, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.2)
    return False


def kill_run(name):
    subprocess.run(["pkill", "-9", "-f", "--", f"--run-name {name}"])


TINY = {
    "dqn": ("train.py", {"base": "dqn.yaml", "total_steps": 200000, "buffer_size": 5000,
                         "learning_starts": 500, "epsilon_steps": 2000, "eval_every": 1000,
                         "eval_episodes": 1, "log_every": 1000, "compile": False,
                         "checkpoint_buffer": False}),
    # PPO steps come in rollouts of 4 * 64 = 256, so checkpoints miss the 2000 marks
    "ppo": ("train_ppo.py", {"base": "ppo_hunger.yaml", "total_steps": 200000, "num_envs": 4,
                             "n_steps": 64, "eval_every": 1000, "eval_episodes": 1,
                             "log_every": 1000}),
}


@pytest.mark.parametrize("algo", ["dqn", "ppo"])
def test_supervisor_evaluates_and_restarts_once(tmp_path, algo):
    name = f"test-sv-{algo}-{os.getpid()}"
    run_dir = os.path.join(ROOT, "runs", name)
    script, tiny = TINY[algo]
    cfg = tmp_path / "tiny.yaml"
    cfg.write_text(yaml.safe_dump({
        **tiny, "base": os.path.relpath(os.path.join(ROOT, "configs", tiny["base"]), tmp_path)}))
    results = tmp_path / "results.md"
    sv = None
    try:
        subprocess.run(["tmux", "new-session", "-d", "-s", name,
                        f"cd {ROOT} && {PY} scripts/{script} --config {cfg} --run-name {name} "
                        f"2>&1 | tee -a runs/{name}.log"], check=True)
        sv = subprocess.Popen([PY, os.path.join(ROOT, "scripts", "supervisor.py"), name,
                               "--every", "2000", "--episodes", "1", "--poll", "0.5",
                               "--results", str(results), "--plot-dir", str(tmp_path),
                               "--log-dir", str(tmp_path)], stdout=subprocess.DEVNULL)
        log = tmp_path / "supervisor.log"
        assert wait_for(lambda: results.exists() and f"{algo} {name}@" in results.read_text(), 120)
        assert wait_for(lambda: (tmp_path / "eval.png").exists(), 60)
        assert any(f.startswith("sv_eval_") for f in os.listdir(run_dir))

        kill_run(name)  # first crash: restarted from the checkpoint
        assert wait_for(lambda: log.exists() and "restarting once" in log.read_text(), 60)
        assert wait_for(lambda: "resumed at" in open(f"{run_dir}.log").read(), 60)
        kill_run(name)  # second crash: left dead
        assert wait_for(lambda: "not restarting" in log.read_text(), 60)
        assert sv.wait(timeout=30) == 0
        state = json.loads((tmp_path / "supervisor_state.json").read_text())
        assert state[name]["status"] == "failed" and state[name]["restarts"] == 1
    finally:
        if sv and sv.poll() is None:
            sv.kill()
        subprocess.run(["tmux", "kill-session", "-t", name], capture_output=True)
        kill_run(name)
        subprocess.run(["rm", "-rf", run_dir, f"{run_dir}.log"])
