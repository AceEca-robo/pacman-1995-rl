import json
import re
import os
import signal
import subprocess
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINY = {"base": os.path.join(ROOT, "configs", "dqn.yaml"), "total_steps": 20000,
        "buffer_size": 5000, "learning_starts": 500, "epsilon_steps": 2000,
        "eval_every": 2000, "eval_episodes": 2, "log_every": 1000, "compile": False}


def test_checkpoint_and_resume(tmp_path):
    cfg = tmp_path / "tiny.yaml"
    cfg.write_text(yaml.safe_dump({**TINY, "base": os.path.relpath(TINY["base"], tmp_path)}))
    name = f"test-resume-{os.getpid()}"
    run_dir = os.path.join(ROOT, "runs", name)
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "train.py"), "--run-name", name]
    try:
        p = subprocess.Popen(cmd + ["--config", str(cfg)], stdout=subprocess.PIPE, text=True)
        evals = os.path.join(run_dir, "evals.jsonl")
        deadline = time.time() + 120
        while not os.path.exists(os.path.join(run_dir, "buffer.npz")) and time.time() < deadline:
            time.sleep(0.05)
        p.send_signal(signal.SIGKILL)  # crash after the first checkpoint
        p.wait()
        out = subprocess.run(cmd + ["--resume"], capture_output=True, text=True, timeout=300)
        assert out.returncode == 0, out.stderr
        m = re.search(r"resumed at (\d+) steps, buffer (\d+)", out.stdout)
        assert m, out.stdout
        at, size = int(m.group(1)), int(m.group(2))
        assert 2000 <= at < 20000 and at % 2000 == 0
        assert min(at, 5000) - 8 <= size <= min(at, 5000)  # n-1 steps per env still pending
        with open(evals) as f:
            steps = [json.loads(line)["steps"] for line in f]
        assert steps[-1] == 20000
        for f in ("best.pt", "checkpoint.pt", "buffer.npz"):
            assert os.path.exists(os.path.join(run_dir, f))
    finally:
        subprocess.run(["rm", "-rf", run_dir])


def test_warm_start_per_shaping_4_actions(tmp_path):
    cfg = tmp_path / "tiny_per.yaml"
    cfg.write_text(yaml.safe_dump({**TINY, "base": os.path.relpath(TINY["base"], tmp_path),
                                   "total_steps": 3000, "warm_start": 1000,
                                   "per": {"enabled": True},
                                   "env_config": {"actions": 4,
                                                  "shaping": {"enabled": True}}}))
    name = f"test-per-{os.getpid()}"
    run_dir = os.path.join(ROOT, "runs", name)
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "train.py"), "--run-name", name,
           "--config", str(cfg)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        assert out.returncode == 0, out.stderr
        m = re.search(r"warm start: (\d+) heuristic steps, buffer (\d+)", out.stdout)
        assert m and int(m.group(1)) == 1000 and 990 <= int(m.group(2)) <= 1000, out.stdout
        with open(os.path.join(run_dir, "evals.jsonl")) as f:
            assert json.loads(f.readlines()[-1])["steps"] == 3000
        import torch
        from agents.dqn import DQNAgent
        ck = torch.load(os.path.join(run_dir, "best.pt"), weights_only=False)
        assert ck["n_actions"] == 4 and ck["env_config"]["shaping"]["enabled"]
        assert DQNAgent(os.path.join(run_dir, "best.pt"), device="cpu").n_actions == 4
    finally:
        subprocess.run(["rm", "-rf", run_dir])


def test_resume_without_buffer(tmp_path):
    cfg = tmp_path / "tiny_nobuf.yaml"
    cfg.write_text(yaml.safe_dump({**TINY, "base": os.path.relpath(TINY["base"], tmp_path),
                                   "total_steps": 4000, "checkpoint_buffer": False}))
    name = f"test-nobuf-{os.getpid()}"
    run_dir = os.path.join(ROOT, "runs", name)
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "train.py"), "--run-name", name]
    try:
        p = subprocess.Popen(cmd + ["--config", str(cfg)], stdout=subprocess.PIPE, text=True)
        deadline = time.time() + 120
        while not os.path.exists(os.path.join(run_dir, "checkpoint.pt")) and time.time() < deadline:
            time.sleep(0.05)
        p.send_signal(signal.SIGKILL)
        p.wait()
        out = subprocess.run(cmd + ["--resume"], capture_output=True, text=True, timeout=300)
        assert out.returncode == 0, out.stderr
        assert re.search(r"resumed at \d+ steps, buffer 0", out.stdout), out.stdout
    finally:
        subprocess.run(["rm", "-rf", run_dir])
