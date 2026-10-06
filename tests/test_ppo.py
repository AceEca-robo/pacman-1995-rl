import os
import sys
import time

import numpy as np
import pytest

from agents.ppo import compute_gae

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def naive_gae(rewards, values, ended, last_value, gamma, lam):
    T, N = rewards.shape
    adv = np.zeros((T, N))
    for i in range(N):
        for t in range(T):
            a, discount = 0.0, 1.0
            for k in range(t, T):
                nxt = last_value[i] if k == T - 1 else values[k + 1, i]
                nonterminal = 1.0 - ended[k, i]
                delta = rewards[k, i] + gamma * nxt * nonterminal - values[k, i]
                a += discount * delta
                if ended[k, i]:
                    break
                discount *= gamma * lam
            adv[t, i] = a
    return adv


def test_gae_matches_naive():
    rng = np.random.default_rng(0)
    T, N = 17, 3
    r = rng.normal(size=(T, N)).astype(np.float32)
    v = rng.normal(size=(T, N)).astype(np.float32)
    ended = (rng.random((T, N)) < 0.15).astype(np.float32)
    last = rng.normal(size=N).astype(np.float32)
    adv, ret = compute_gae(r, v, ended, last, 0.99, 0.95)
    assert np.allclose(adv, naive_gae(r, v, ended, last, 0.99, 0.95), atol=1e-4)
    assert np.allclose(ret, adv + v)


def test_ppo_learns_cartpole(tmp_path):
    """Sanity check of the whole PPO loop on a task it must solve quickly."""
    from train_ppo import train
    cfg = {"num_envs": 8, "n_steps": 128, "epochs": 4, "minibatches": 4, "gamma": 0.99,
           "gae_lambda": 0.95, "clip": 0.2, "ent_coef": 0.01, "vf_coef": 0.5, "lr": 0.00025,
           "anneal_lr": True, "adam_eps": 1e-5, "max_grad_norm": 0.5, "total_steps": 80_000,
           "network": {"hidden": 64}, "eval_every": 10**9, "eval_episodes": 0,
           "eval_seed": 0, "best_metric": "mean_reward", "log_every": 10**9, "env_config": None}
    t0 = time.time()
    hist = train(cfg, 0, str(tmp_path), env_id="CartPole-v1", device="cpu", quiet=True)
    returns = [r for _, r in hist]
    first, last = np.mean(returns[:20]), np.mean(returns[-20:])
    print(f"CartPole: first 20 episodes {first:.0f}, last 20 {last:.0f}, {time.time() - t0:.0f} s")
    assert first < 40 and last > 200


def test_ppo_pacman_run_eval_resume(tmp_path):
    """Tiny PPO run on PacmanEnv: checkpoints, greedy eval, resume, loading."""
    import json
    import signal
    import subprocess
    import yaml
    name = f"test-ppo-{os.getpid()}"
    run_dir = os.path.join(ROOT, "runs", name)
    cfg = tmp_path / "tiny.yaml"
    cfg.write_text(yaml.safe_dump({
        "base": os.path.relpath(os.path.join(ROOT, "configs", "ppo_hunger.yaml"), tmp_path),
        "total_steps": 12000, "num_envs": 4, "n_steps": 64, "eval_every": 3000,
        "eval_episodes": 2, "log_every": 2000}))
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "train_ppo.py"), "--run-name", name]
    try:
        p = subprocess.Popen(cmd + ["--config", str(cfg)], stdout=subprocess.PIPE, text=True)
        deadline = time.time() + 120
        while not os.path.exists(os.path.join(run_dir, "checkpoint.pt")) and time.time() < deadline:
            time.sleep(0.05)
        p.send_signal(signal.SIGKILL)
        p.wait()
        out = subprocess.run(cmd + ["--resume"], capture_output=True, text=True, timeout=300)
        assert out.returncode == 0, out.stderr
        assert "resumed at" in out.stdout and "hunger" in out.stdout
        with open(os.path.join(run_dir, "evals.jsonl")) as f:
            assert json.loads(f.readlines()[-1])["steps"] >= 12000
        from agents.ppo import PPOAgent, load_agent
        agent = load_agent(os.path.join(run_dir, "best.pt"))
        assert isinstance(agent, PPOAgent) and agent.n_actions == 4
        ev = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "evaluate.py"),
                             "--agent", "ppo", "--checkpoint", os.path.join(run_dir, "best.pt"),
                             "--episodes", "1", "--results", ""],
                            capture_output=True, text=True, timeout=300)
        assert ev.returncode == 0 and "| ppo " in ev.stdout, ev.stderr
    finally:
        subprocess.run(["rm", "-rf", run_dir])
