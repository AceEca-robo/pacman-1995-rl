"""Distillation: scripts/distill_collect.py, train_distill.py, sim_final_eval.py (CPU)."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
V04 = os.path.join(ROOT, "runs", "best_v04", "best.pt")
needs_v04 = pytest.mark.skipif(not os.path.exists(V04), reason="runs/best_v04/best.pt not present")


def run(*args):
    r = subprocess.run([sys.executable, *args], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-3000:]
    return r.stdout


@needs_v04
def test_full_rollouts_keep_the_choice():
    from plan_eval import Planner, QNet, batch_advance, batch_info, batch_obs, load_cfg
    from sim.csim import CSim
    cfg = load_cfg(os.path.join(ROOT, "configs", "distill_collect.yaml"))
    cfg["plan"].update(horizon=8, rollouts=4)
    net = QNet(V04, "cpu")
    planner = Planner(net, cfg)
    sims = []
    for seed, level in ((1, 3), (2, 9), (3, 14), (4, 15)):
        s = CSim()
        s.reset(seed, level)
        sims.append(s.buf)
    rows = np.stack(sims)
    for step in range(30):
        info = batch_info(rows)
        q = net(*batch_obs(net, rows))
        seeds = [[i, 0, step] for i in range(len(rows))]
        a_lazy, st_lazy = planner.choose(rows, info, q, seeds, frac_dir=True)
        a_full, st_full = planner.choose(rows, info, q, seeds, full=True)
        assert (a_lazy == a_full).all()
        fd = st_full["frac_dir"]
        # lazy: the rolled-out directions agree with full, the others are NaN
        fl = st_lazy["frac_dir"]
        known = np.isfinite(fl)
        assert known.sum(1).min() >= 1 and np.allclose(fl[known], fd[known])
        assert np.isfinite(fd).all() and ((0 <= fd) & (fd <= 1)).all()
        # the first candidate's share is the same in both
        qd = q[:, :4].copy()
        for g in range(len(rows)):
            if info[g, 5] <= 3 and q[g, 4] > q[g, info[g, 5]]:
                qd[g, info[g, 5]] = q[g, 4]
        first = np.argsort(-qd, axis=1, kind="stable")[:, 0]
        assert np.allclose(fd[np.arange(len(rows)), first], st_lazy["frac_best"])
        batch_advance(rows, a_full)


def _chunk(n, start):
    from distill_collect import FIELDS
    d = {k: np.zeros(n, np.int64) for k in FIELDS}
    d["act"] = np.arange(start, start + n)
    return d


def test_merger_parts_and_rest(tmp_path):
    from distill_collect import Merger, save_npz
    os.makedirs(tmp_path / "chunks")
    for i, (n, s) in enumerate(((30, 0), (50, 30), (45, 80))):
        save_npz(str(tmp_path / "chunks" / f"s00_w00_{i:05d}.npz"), _chunk(n, s))
    m = Merger(str(tmp_path), 40)
    # a chunk a worker is still writing (2026-10-23: the merger read one and hung)
    with open(tmp_path / "chunks" / "s00_w01_00000.npz.tmp.npz", "wb") as f:
        f.write(b"PK\x03\x04 half a file")
    m.merge()
    assert m.m["parts"] == {"part_000.npz": 40, "part_001.npz": 40, "part_002.npz": 40}
    os.remove(tmp_path / "chunks" / "s00_w01_00000.npz.tmp.npz")
    assert m.counts() == (120, 5)
    m.merge(final=True)
    assert m.m["parts"]["part_003.npz"] == 5
    acts = np.concatenate([np.load(tmp_path / f"part_{i:03d}.npz")["act"] for i in range(4)])
    assert (acts == np.arange(125)).all()  # every state once, in order
    assert not os.listdir(tmp_path / "chunks")
    # a stop after the manifest entry but before deleting: the chunk goes on restart
    save_npz(str(tmp_path / "chunks" / "s01_w00_00000.npz"), _chunk(3, 0))
    m.m["consumed"] = ["s01_w00_00000.npz"]
    m.save_manifest()
    Merger(str(tmp_path), 40)
    assert not os.listdir(tmp_path / "chunks")


def test_move_and_val_mask():
    from train_distill import move, val_mask
    pdir = np.array([0, 2, 5, 3])
    assert (move(np.array([4, 4, 4, 1]), pdir) == [0, 2, 4, 1]).all()
    seeds = np.arange(100000, 140000)
    m = val_mask(seeds, 0.05)
    assert 0.04 < m.mean() < 0.06 and (val_mask(seeds, 0.05) == m).all()


@needs_v04
def test_collect_train_eval_end_to_end(tmp_path):
    with open(os.path.join(ROOT, "configs", "distill_collect.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["plan"].update(horizon=5, rollouts=2)
    cfg["collect"].update(out_dir=str(tmp_path / "data"), target_states=120, part_size=50,
                          flush_minutes=0.01, log_seconds=1, max_steps=30)
    cfg["run"].update(workers=2, games_per_worker=3, device="cpu")
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    run("scripts/distill_collect.py", "--config", str(tmp_path / "c.yaml"))
    man = json.loads((tmp_path / "data" / "manifest.json").read_text())
    assert sum(man["parts"].values()) >= 120 and man["parts"]["part_000.npz"] == 50
    z = np.load(tmp_path / "data" / "part_000.npz")
    assert z["obs"].dtype == np.uint8 and (z["act"] == z["played"]).all()
    assert (z["step"] < 30).all() and ((1 <= z["start_level"]) & (z["start_level"] <= 16)).all()
    # DAgger: a driver plays, the planner labels
    cfg["collect"].update(out_dir=str(tmp_path / "dagger"), driver="runs/best_v04/best.pt",
                          target_states=40)
    (tmp_path / "d.yaml").write_text(yaml.safe_dump(cfg))
    run("scripts/distill_collect.py", "--config", str(tmp_path / "d.yaml"))
    z = np.load(tmp_path / "dagger" / "part_000.npz")
    assert (z["played"] == z["q"].argmax(1)).all()

    with open(os.path.join(ROOT, "configs", "dist0.yaml")) as f:
        tc = yaml.safe_load(f)
    tc.update(data=[str(tmp_path / "data")], run_dir=str(tmp_path / "run"), device="cpu")
    tc["train"].update(epochs=1, batch_size=16, val_frac=0.3, log_every=2)
    (tmp_path / "t.yaml").write_text(yaml.safe_dump(tc))
    run("scripts/train_distill.py", "--config", str(tmp_path / "t.yaml"))
    tc["train"]["epochs"] = 2
    (tmp_path / "t.yaml").write_text(yaml.safe_dump(tc))
    out = run("scripts/train_distill.py", "--config", str(tmp_path / "t.yaml"), "--resume")
    assert "epoch 2 val" in out
    log = [json.loads(line) for line in (tmp_path / "run" / "log.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in log] == [0, 1, 2]
    from agents.dqn import DQNAgent
    agent = DQNAgent(str(tmp_path / "run" / "epoch_02.pt"), device="cpu")
    assert agent.n_actions == 5

    out_json = tmp_path / "eval" / "e.json"
    run("scripts/sim_final_eval.py", "--checkpoint", str(tmp_path / "run" / "best.pt"),
        "--label", "t", "--out", str(out_json), "--levels", "7", "--episodes", "2",
        "--device", "cpu")
    res = json.loads(out_json.read_text())
    assert len(res["games"]) == 2 and all(g["maze"] == 7 for g in res["games"])
    # resume with nothing left to play keeps the games
    run("scripts/sim_final_eval.py", "--checkpoint", str(tmp_path / "run" / "best.pt"),
        "--label", "t", "--out", str(out_json), "--levels", "7", "--episodes", "2",
        "--device", "cpu", "--resume")
    assert len(json.loads(out_json.read_text())["games"]) == 2
