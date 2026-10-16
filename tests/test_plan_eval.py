"""scripts/plan_eval.py: the planner end to end on a few short games (CPU)."""
import json
import os
import subprocess
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V04 = os.path.join(ROOT, "runs", "best_v04", "best.pt")


@pytest.mark.skipif(not os.path.exists(V04), reason="runs/best_v04/best.pt not present")
@pytest.mark.parametrize("mode,config,extra", [
    ("plan", "plan.yaml", {}), ("greedy", "plan.yaml", {}), ("plan", "plan_honest.yaml", {}),
    ("plan", "plan_honest.yaml", {"loop_breaker": {"stall_steps": 5, "active_steps": 3,
                                                   "penalty": 10.0}})])
def test_plan_eval_smoke(tmp_path, mode, config, extra):
    with open(os.path.join(ROOT, "configs", config)) as f:
        cfg = yaml.safe_load(f)
    cfg["plan"].update(horizon=6, rollouts=2, **extra)
    cfg["eval"].update(episodes=1, level_episodes=2, max_steps=40)
    cfg["run"].update(workers=1, games_per_worker=2, device="cpu")
    conf = tmp_path / "plan.yaml"
    conf.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "out.json"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "plan_eval.py"),
                        "--config", str(conf), "--mode", mode, "--levels", "1,7",
                        "--replay-check", "2", "--out", str(out)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.loads(out.read_text())
    assert len(res["games"]) == 4 and res["replay_check"]["diverged"] == 0
    assert all(len(g["actions"]) == g["length"] <= 40 for g in res["games"])
    if mode == "greedy":
        assert res["overrides"] == 0
    assert (res["breaker"] > 0) == bool(extra)


def test_reseed_keeps_visible_state():
    """sim_reseed_many: the state line is the same, the ghosts' future is not."""
    import numpy as np
    sys.path.insert(0, ROOT)
    from sim.csim import SimBatch, CSim
    sim = CSim(5, 3)
    for a in [2] * 30:
        sim.advance(a)
    b = SimBatch(3)
    for i in range(3):
        b.set(i, sim)
    b.reseed([11, 12, 11])
    assert all(b.get(i).state_json() == sim.state_json() for i in range(3))
    for _ in range(60):
        b.advance(np.full(3, 4))
    lines = [b.get(i).state_json() for i in range(3)]
    assert lines[0] == lines[2] and lines[0] != lines[1]


@pytest.mark.skipif(not os.path.exists(V04), reason="runs/best_v04/best.pt not present")
def test_plan_eval_jsonl_maze_only(tmp_path):
    """Every game goes to the JSONL as it ends; --summarize reads it back."""
    with open(os.path.join(ROOT, "configs", "plan_honest_lb.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["plan"].update(horizon=4, rollouts=2)
    cfg["eval"].update(episodes=1, level_episodes=2, max_steps=30)
    cfg["run"].update(workers=2, games_per_worker=1, device="cpu")
    conf = tmp_path / "plan.yaml"
    conf.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "out.json"
    script = os.path.join(ROOT, "scripts", "plan_eval.py")
    r = subprocess.run([sys.executable, script, "--config", str(conf), "--levels", "1,7",
                        "--maze-only", "--out", str(out)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-3000:]
    assert r.stdout.count("game ") >= 4 and "game 4/4" in r.stdout
    lines = (tmp_path / "out.jsonl").read_text().splitlines()
    games = [json.loads(x) for x in lines]
    assert len(games) == 4 and all(g["stop"] for g in games)
    res = json.loads(out.read_text())
    assert "from_level1" not in res and set(res["per_level"]) == {"1", "7"}
    r = subprocess.run([sys.executable, script, "--config", str(conf), "--summarize",
                        str(tmp_path / "out.jsonl")], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-3000:]
    assert '"games_played": 4' in r.stdout
