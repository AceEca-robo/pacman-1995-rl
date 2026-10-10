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


def test_best_metric_levels_cleared():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from train import best_metric
    a = [{"levels_cleared": 1, "food": 900}, {"levels_cleared": 0, "food": 100}]
    b = [{"levels_cleared": 1, "food": 950}, {"levels_cleared": 0, "food": 100}]  # more food
    c = [{"levels_cleared": 1, "food": 300}, {"levels_cleared": 1, "food": 300}]  # more levels
    assert best_metric("levels_cleared", b) > best_metric("levels_cleared", a)
    assert best_metric("levels_cleared", c) > best_metric("levels_cleared", b)


def test_start_level_schedule(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from train import start_level_at
    sched = {"start_level_schedule": [[0, "random:1-4"], [100, "random:1-8"], [200, 5]]}
    assert start_level_at({}, 50) is None
    assert [start_level_at(sched, s) for s in (0, 99, 100, 199, 200, 10**9)] == \
        ["random:1-4", "random:1-4", "random:1-8", "random:1-8", 5, 5]
    # a tiny run through the schedule: the switch is logged, evals count levels
    cfg = tmp_path / "tiny_levels.yaml"
    cfg.write_text(yaml.safe_dump({**TINY, "base": os.path.relpath(TINY["base"], tmp_path),
                                   "total_steps": 4000, "best_metric": "levels_cleared",
                                   "eval_max_episode_steps": 300,
                                   "start_level_schedule": [[0, "random:2-3"], [2000, 7]]}))
    name = f"test-levels-{os.getpid()}"
    run_dir = os.path.join(ROOT, "runs", name)
    try:
        out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "train.py"),
                              "--run-name", name, "--config", str(cfg)],
                             capture_output=True, text=True, timeout=300)
        assert out.returncode == 0, out.stderr
        assert "start_level random:2-3" in out.stdout
        assert re.search(r"2000 steps  start_level 7", out.stdout), out.stdout
        with open(os.path.join(run_dir, "evals.jsonl")) as f:
            evals = [json.loads(line) for line in f]
        assert all("mean_levels_cleared" in e for e in evals)
        assert all(g["levels_cleared"] == g["level"] - 1 for e in evals for g in e["episodes"])
    finally:
        subprocess.run(["rm", "-rf", run_dir])


def test_eval_levels(tmp_path):
    """eval_levels adds games started on each listed maze (stopped at the
    first level up) and a maze<N>_cleared share per maze to the eval summary."""
    cfg = tmp_path / "tiny_levels.yaml"
    cfg.write_text(yaml.safe_dump({**TINY, "base": os.path.relpath(TINY["base"], tmp_path),
                                   "total_steps": 2000, "eval_levels": [2, 3],
                                   "eval_level_episodes": 2, "eval_envs": 3,
                                   "eval_max_episode_steps": 300}))
    name = f"test-eval-levels-{os.getpid()}"
    run_dir = os.path.join(ROOT, "runs", name)
    try:
        out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "train.py"),
                              "--run-name", name, "--config", str(cfg)],
                             capture_output=True, text=True, timeout=300)
        assert out.returncode == 0, out.stderr
        with open(os.path.join(run_dir, "evals.jsonl")) as f:
            ev = json.loads(f.readline())
        assert "maze2_cleared" in ev and "maze3_cleared" in ev
        eps = ev["episodes"]
        assert [e.get("maze") for e in eps] == [None, None, 2, 2, 3, 3]
        assert [e["seed"] for e in eps] == [1000, 1001] * 3
        assert all(e["level"] >= e.get("maze", 1) for e in eps)  # started on its maze
    finally:
        subprocess.run(["rm", "-rf", run_dir])


def test_async_envs_and_learner(tmp_path):
    """vector_env: async + async_learner train, evaluate, checkpoint and resume."""
    cfg = tmp_path / "tiny_async.yaml"
    cfg.write_text(yaml.safe_dump({**TINY, "base": os.path.relpath(TINY["base"], tmp_path),
                                   "total_steps": 4000, "num_envs": 8, "vector_env": "async",
                                   "async_learner": True, "target_update": 1000,
                                   "per": {"enabled": True}}))
    name = f"test-async-{os.getpid()}"
    run_dir = os.path.join(ROOT, "runs", name)
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "train.py"), "--run-name", name]
    try:
        out = subprocess.run(cmd + ["--config", str(cfg)], capture_output=True, text=True,
                             timeout=300)
        assert out.returncode == 0, out.stderr
        assert "async envs x8, async learner" in out.stdout
        n_upd = [int(m) for m in re.findall(r"updates (\d+)", out.stdout)]
        # credit = steps / train_every from learning_starts on, at most
        # async_max_lag (4) updates behind
        assert n_upd and 0 < n_upd[-1] <= (4000 - 500) // 16 + 1
        with open(os.path.join(run_dir, "evals.jsonl")) as f:
            assert [json.loads(line)["steps"] for line in f] == [2000, 4000]
        out = subprocess.run(cmd + ["--resume", "--extend-steps", "2000"], capture_output=True,
                             text=True, timeout=300)
        assert out.returncode == 0, out.stderr
        assert "resumed at 4000 steps" in out.stdout
    finally:
        subprocess.run(["rm", "-rf", run_dir])


def test_rl_from_bc(tmp_path):
    """init_conv_from copies a BC network's convolutions, freeze_conv_steps
    keeps them fixed (then they train), warm_start_agent plays the BC policy,
    levels_cleared_min_maze picks best.pt."""
    import torch
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from agents.bc import BCNetwork
    from train import best_metric, load_config
    net_cfg = load_config(os.path.join(ROOT, "configs", "dqn.yaml"))["network"]
    bc_path = tmp_path / "bc.pt"
    bc = BCNetwork(net_cfg)
    torch.save({"algo": "bc", "config": {"network": net_cfg}, "n_actions": 5,
                "env_config": {"actions": 5}, "net": bc.state_dict()}, bc_path)
    cfg = tmp_path / "tiny_bc.yaml"
    cfg.write_text(yaml.safe_dump({**TINY, "base": os.path.relpath(TINY["base"], tmp_path),
                                   "total_steps": 4000, "warm_start": 600,
                                   "warm_start_agent": str(bc_path), "init_conv_from": str(bc_path),
                                   "freeze_conv_steps": 2000, "per": {"enabled": True},
                                   "best_metric": "levels_cleared_min_maze",
                                   "eval_levels": [2, 3], "eval_level_episodes": 1,
                                   "eval_level_max_steps": 200, "eval_max_episode_steps": 300}))
    name = f"test-bc-{os.getpid()}"
    run_dir = os.path.join(ROOT, "runs", name)
    try:
        out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "train.py"),
                              "--run-name", name, "--config", str(cfg)],
                             capture_output=True, text=True, timeout=300)
        assert out.returncode == 0, out.stderr
        assert re.search(r"warm start: 600 \S+bc.pt steps", out.stdout), out.stdout
        assert "convolutions frozen until 2000" in out.stdout
        assert re.search(r"2000 steps  convolutions unfrozen", out.stdout), out.stdout
        with open(os.path.join(run_dir, "evals.jsonl")) as f:
            evals = [json.loads(line) for line in f]
        first, last = evals[0], evals[-1]
        assert "min_maze_cleared" in last
        assert all(g["length"] <= 200 for g in last["episodes"] if "maze" in g)
        ck = [torch.load(os.path.join(run_dir, f), weights_only=False) for f in ("checkpoint.pt",)]
        conv = {k: v for k, v in bc.state_dict().items() if k.startswith("conv.")}
        online = ck[0]["learner"]["online"]
        assert any(not torch.equal(online[k].cpu(), v) for k, v in conv.items())  # trained later
        assert first["steps"] == 2000
    finally:
        subprocess.run(["rm", "-rf", run_dir])
    maze = [{"maze": 2, "levels_cleared": 1}, {"maze": 3, "levels_cleared": 0}]
    res = [{"levels_cleared": 1}, {"levels_cleared": 0}]
    assert best_metric("levels_cleared_min_maze", res, maze) == 0.5 * 1e5
    maze2 = [{"maze": 2, "levels_cleared": 1}, {"maze": 3, "levels_cleared": 1}]
    assert best_metric("levels_cleared_min_maze", res, maze2) == 0.5 * 1e5 + 1


def test_conv_freeze_keeps_weights():
    import numpy as np
    import torch
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from agents.dqn import DQNLearner, N_PLANES, PACKED
    from train import load_config
    cfg = {**load_config(os.path.join(ROOT, "configs", "dqn.yaml")), "compile": False}
    learner = DQNLearner(cfg, "cpu")
    before = {k: v.clone() for k, v in learner.online.state_dict().items()}
    rng = np.random.default_rng(0)
    b = 8
    batch = (rng.integers(0, 256, (b, N_PLANES, PACKED), dtype=np.uint8),
             rng.random((b, 3), dtype=np.float32), rng.integers(0, 5, b),
             rng.random(b, dtype=np.float32),
             rng.integers(0, 256, (b, N_PLANES, PACKED), dtype=np.uint8),
             rng.random((b, 3), dtype=np.float32), np.full(b, 0.97, np.float32))
    learner.request_conv_trainable(False)
    learner.update(batch, np.ones(b, np.float32))
    after = learner.online.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before if k.startswith("conv."))
    assert not torch.equal(before["fc.0.weight"], after["fc.0.weight"])
    learner.request_conv_trainable(True)
    learner.update(batch, np.ones(b, np.float32))
    assert not torch.equal(before["conv.0.weight"], learner.online.state_dict()["conv.0.weight"])


def shm_segments():
    """Shared memory segments of train.py's learner_process runs."""
    return {p for p in os.listdir("/dev/shm") if p.startswith("pacrl_")}


def test_learner_process(tmp_path):
    """learner_process with AsyncVectorEnv: the update count follows the
    credit, evaluate, checkpoint, resume (the replay is refilled)."""
    cfg = tmp_path / "tiny_proc.yaml"
    cfg.write_text(yaml.safe_dump({**TINY, "base": os.path.relpath(TINY["base"], tmp_path),
                                   "total_steps": 4000, "num_envs": 4, "vector_env": "async",
                                   "learner_process": True, "target_update": 1000,
                                   "buffer_size": 4000, "replay_guard": 64,
                                   "checkpoint_buffer": False, "per": {"enabled": True}}))
    name = f"test-proc-{os.getpid()}"
    run_dir = os.path.join(ROOT, "runs", name)
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "train.py"), "--run-name", name]
    shm_before = shm_segments()
    try:
        out = subprocess.run(cmd + ["--config", str(cfg)], capture_output=True, text=True,
                             timeout=300)
        assert out.returncode == 0, out.stderr
        assert "async envs x4, learner process" in out.stdout
        n_upd = [int(m) for m in re.findall(r"updates (\d+)", out.stdout)]
        assert n_upd and (4000 - 500) // 16 - 5 <= n_upd[-1] <= (4000 - 500) // 16 + 1
        with open(os.path.join(run_dir, "evals.jsonl")) as f:
            assert [json.loads(line)["steps"] for line in f] == [2000, 4000]
        out = subprocess.run(cmd + ["--resume", "--extend-steps", "2000"], capture_output=True,
                             text=True, timeout=300)
        assert out.returncode == 0, out.stderr
        assert "resumed at 4000 steps" in out.stdout
        assert not shm_segments() - shm_before  # shared memory unlinked
    finally:
        subprocess.run(["rm", "-rf", run_dir])


def test_dqfd_pool(tmp_path):
    """DQfD on the env pool: demo of a BC agent, pretrain updates on it with
    evals, RL; a resume reloads demo.npz and skips the pretraining."""
    import torch
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from agents.bc import BCNetwork
    from train import load_config
    net_cfg = load_config(os.path.join(ROOT, "configs", "dqn.yaml"))["network"]
    bc_path = tmp_path / "bc.pt"
    torch.save({"algo": "bc", "config": {"network": net_cfg}, "n_actions": 5,
                "env_config": {"actions": 5}, "net": BCNetwork(net_cfg).state_dict()}, bc_path)
    cfg = tmp_path / "tiny_dqfd.yaml"
    cfg.write_text(yaml.safe_dump({
        **TINY, "base": os.path.relpath(TINY["base"], tmp_path), "total_steps": 3000,
        "num_envs": 4, "vector_env": "pool", "learner_process": True, "buffer_size": 4000,
        "replay_guard": 64, "checkpoint_buffer": False, "eval_every": 1500,
        "per": {"enabled": True}, "init_conv_from": str(bc_path), "epsilon_start": 0.02,
        "epsilon_end": 0.01, "dqfd": {
            "enabled": True, "demo_agent": str(bc_path), "demo_transitions": 600,
            "demo_epsilon": 0.1, "pretrain_updates": 200, "pretrain_target_update": 50,
            "pretrain_eval_every": 100, "pretrain_log_every": 50, "margin": 0.8,
            "lambda_margin": 1.0, "lambda_1step": 1.0, "lambda_nstep": 1.0, "l2": 1e-5,
            "eps_demo": 1.0}}))
    name = f"test-dqfd-{os.getpid()}"
    run_dir = os.path.join(ROOT, "runs", name)
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "train.py"), "--run-name", name]
    try:
        out = subprocess.run(cmd + ["--config", str(cfg)], capture_output=True, text=True,
                             timeout=300)
        assert out.returncode == 0, out.stderr
        m = re.search(r"demo: (\d+) transitions of", out.stdout)
        assert m and 600 <= int(m.group(1)) < 620, out.stdout
        assert re.search(r"pretrain +200 updates .* margin \d", out.stdout), out.stdout
        assert "pool envs x4, learner process" in out.stdout
        with open(os.path.join(run_dir, "evals.jsonl")) as f:
            rows = [json.loads(line) for line in f]
        assert [r.get("phase") for r in rows] == ["pretrain@100", "pretrain", None, None]
        # pool: steps advance by the envs that had results, evals at the first step >= k * 1500
        assert 1500 <= rows[2]["steps"] < 1520 and 3000 <= rows[3]["steps"] < 3020
        pre = torch.load(os.path.join(run_dir, "pretrain.pt"), weights_only=False)
        assert pre["pretrained"] and pre["steps"] == 0 and pre["n_updates"] == 200
        out = subprocess.run(cmd + ["--resume", "--extend-steps", "1500"], capture_output=True,
                             text=True, timeout=300)
        assert out.returncode == 0, out.stderr
        assert re.search(r"demo: \d+ transitions from .*demo.npz", out.stdout), out.stdout
        assert "pretrain " not in out.stdout
    finally:
        subprocess.run(["rm", "-rf", run_dir])


def test_signal_cleans_shared_memory(tmp_path):
    """SIGHUP (tmux kill-session) mid-run: the learner process stops and the
    shared memory is unlinked; a segment of a dead process is removed at the
    next start."""
    from agents.shared_replay import SHM_PREFIX, remove_stale_shm
    cfg = tmp_path / "tiny_sig.yaml"
    cfg.write_text(yaml.safe_dump({**TINY, "base": os.path.relpath(TINY["base"], tmp_path),
                                   "total_steps": 10**6, "num_envs": 4, "vector_env": "pool",
                                   "learner_process": True, "buffer_size": 4000,
                                   "replay_guard": 64, "checkpoint_buffer": False}))
    name = f"test-sig-{os.getpid()}"
    run_dir = os.path.join(ROOT, "runs", name)
    try:
        p = subprocess.Popen([sys.executable, os.path.join(ROOT, "scripts", "train.py"),
                              "--run-name", name, "--config", str(cfg)],
                             stdout=subprocess.PIPE, text=True)
        for line in p.stdout:
            if "steps/s" in line:  # learning
                break
        mine = {s for s in shm_segments() if s.startswith(f"{SHM_PREFIX}{p.pid}_")}
        assert mine
        p.send_signal(signal.SIGHUP)
        p.wait(timeout=60)
        assert not mine & shm_segments()
    finally:
        if p.poll() is None:
            p.kill()
        subprocess.run(["rm", "-rf", run_dir])
    # a dead creator's segment
    dead = subprocess.Popen(["true"])
    dead.wait()
    path = f"/dev/shm/{SHM_PREFIX}{dead.pid}_1"
    with open(path, "wb") as f:
        f.write(b"x")
    assert f"{SHM_PREFIX}{dead.pid}_1" in remove_stale_shm()
    assert not os.path.exists(path)
