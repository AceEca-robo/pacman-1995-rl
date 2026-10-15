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
@pytest.mark.parametrize("mode", ["plan", "greedy"])
def test_plan_eval_smoke(tmp_path, mode):
    with open(os.path.join(ROOT, "configs", "plan.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["plan"].update(horizon=6, rollouts=2)
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
