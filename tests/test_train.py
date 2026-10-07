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


def test_best_metric():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from train import best_metric
    res = [{"reward": r, "score": s} for r, s in [(60, 1), (70, 2), (80, 3), (-200, 4), (-200, 5)]]
    assert best_metric("mean_reward", res) == -38.0
    assert best_metric("reward", res) == best_metric("median_reward", res) == 60.0
    assert best_metric("score", res) == 3.0
    # level1_cleared: the share of games past level 1 first, mean food breaks ties
    a = [{"level": 2, "food": 180}, {"level": 1, "food": 170}]
    b = [{"level": 2, "food": 300}, {"level": 1, "food": 160}]   # same share, more food
    c = [{"level": 2, "food": 175}, {"level": 2, "food": 175}]   # higher share, less food
    assert best_metric("level1_cleared", b) > best_metric("level1_cleared", a)
    assert best_metric("level1_cleared", c) > best_metric("level1_cleared", b)


def test_train_and_evaluate_with_extra_obs(tmp_path):
    cfg = tmp_path / "tiny_obs.yaml"
    cfg.write_text(yaml.safe_dump({**TINY, "base": os.path.relpath(TINY["base"], tmp_path),
                                   "total_steps": 2000, "eval_every": 1000,
                                   "checkpoint_buffer": False,
                                   "env_config": {"obs": {"food_distance": True,
                                                          "steps_since_food": True}}}))
    name = f"test-obs-{os.getpid()}"
    run_dir = os.path.join(ROOT, "runs", name)
    try:
        out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "train.py"),
                              "--run-name", name, "--config", str(cfg)],
                             capture_output=True, text=True, timeout=300)
        assert out.returncode == 0, out.stderr
        assert "buffer 0.00 GB" not in out.stdout
        ev = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "evaluate.py"),
                             "--agent", "dqn", "--checkpoint", os.path.join(run_dir, "best.pt"),
                             "--episodes", "1", "--epsilon", "0.05", "--results", ""],
                            capture_output=True, text=True, timeout=300)
        assert ev.returncode == 0, ev.stderr
    finally:
        subprocess.run(["rm", "-rf", run_dir])


def test_finetune_from_checkpoint_with_prefixes(tmp_path):
    base = {**TINY, "base": os.path.relpath(TINY["base"], tmp_path), "total_steps": 2000,
            "eval_every": 1000, "checkpoint_buffer": False}
    first = f"test-ft0-{os.getpid()}"
    second = f"test-ft1-{os.getpid()}"
    dirs = [os.path.join(ROOT, "runs", n) for n in (first, second)]
    train = [sys.executable, os.path.join(ROOT, "scripts", "train.py")]
    try:
        c0 = tmp_path / "c0.yaml"
        c0.write_text(yaml.safe_dump(base))
        assert subprocess.run(train + ["--run-name", first, "--config", str(c0)],
                              capture_output=True, timeout=300).returncode == 0
        c1 = tmp_path / "c1.yaml"
        c1.write_text(yaml.safe_dump({**base, "init_checkpoint": f"runs/{first}/best.pt",
                                      "track_cells": [[24, 19], [25, 19]],
                                      "env_config": {"prefix_prob": 1.0,
                                                     "endgame_dot": {"enabled": True, "k": 20}}}))
        out = subprocess.run(train + ["--run-name", second, "--config", str(c1)],
                             capture_output=True, text=True, timeout=300)
        assert out.returncode == 0, out.stderr
        assert f"initialized from runs/{first}/best.pt" in out.stdout
        assert "pocket" in out.stdout and "prefixed" in out.stdout
    finally:
        for d in dirs:
            subprocess.run(["rm", "-rf", d])


def test_resume_extend_steps(tmp_path):
    cfg = tmp_path / "tiny_ext.yaml"
    cfg.write_text(yaml.safe_dump({**TINY, "base": os.path.relpath(TINY["base"], tmp_path),
                                   "total_steps": 2000, "eval_every": 1000,
                                   "checkpoint_buffer": False}))
    name = f"test-ext-{os.getpid()}"
    run_dir = os.path.join(ROOT, "runs", name)
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "train.py"), "--run-name", name]
    try:
        assert subprocess.run(cmd + ["--config", str(cfg)], capture_output=True,
                              timeout=300).returncode == 0
        out = subprocess.run(cmd + ["--resume", "--extend-steps", "2000"], capture_output=True,
                             text=True, timeout=300)
        assert out.returncode == 0, out.stderr
        assert "total_steps extended to 4000" in out.stdout
        with open(os.path.join(run_dir, "evals.jsonl")) as f:
            assert json.loads(f.readlines()[-1])["steps"] == 4000
        with open(os.path.join(run_dir, "config.yaml")) as f:
            assert yaml.safe_load(f)["total_steps"] == 4000
        with open(os.path.join(run_dir, "evals.jsonl")) as f:
            last = json.loads(f.readlines()[-1])
        assert "mean_food" in last and all("food" in e for e in last["episodes"])
    finally:
        subprocess.run(["rm", "-rf", run_dir])
