"""scripts/netviz.py (play.py --viz) and the planner's rollout trace."""
import os
import sys

import numpy as np
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
V04 = os.path.join(ROOT, "runs", "best_v04", "best.pt")
needs_v04 = pytest.mark.skipif(not os.path.exists(V04), reason="runs/best_v04/best.pt not present")


def small_cfg(**plan):
    with open(os.path.join(ROOT, "configs", "plan_honest_lb.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["plan"].update(horizon=8, rollouts=3, **plan)
    return cfg


@needs_v04
def test_trace_keeps_the_choice():
    """trace=True rolls out all four directions but plays the same moves."""
    from scripts.plan_eval import Planner, QNet
    from sim.csim import CSim
    cfg = small_cfg(death_threshold=0.3)
    net = QNet(V04, "cpu")
    planner = Planner(net, cfg)
    sims = [CSim(s, lv) for s, lv in ((0, 3), (1, 9), (2, 14))]
    for t in range(40):
        for i, sim in enumerate(sims):
            o = sim.obs(net.cell_safety)
            q = net(o["grid"][None], o["vec"][None])
            args = (sim.buf[None], sim.info()[None], q, [[i, 0, t]])
            a0, _ = planner.choose(*args)
            a1, st = planner.choose(*args, trace=True)
            assert a0[0] == a1[0]
            tr = st["trace"]
            assert tr["cells"].shape == (1, 4, 3, 8) and tr["dead"].shape == (1, 4, 3)
            assert np.allclose(tr["frac"][0], tr["dead"][0].mean(1))
            # a rollout that lost a life has no cells from that step on
            for r in range(4):
                for k in range(3):
                    c = tr["cells"][0, r, k]
                    if not tr["dead"][0, r, k]:
                        assert (c >= 0).all()
                    else:
                        assert c[-1] == -1
            sim.advance(int(a0[0]))


@needs_v04
def test_viz_frame_headless():
    pytest.importorskip("pygame")
    from scripts.netviz import Viz, W, H
    from scripts.play import PlannerAgent
    cfgp = os.path.join(ROOT, "tests", "_viz_cfg.yaml")
    try:
        with open(cfgp, "w") as f:
            yaml.safe_dump(small_cfg(), f)
        agent = PlannerAgent(cfgp, "cpu", trace=True)
    finally:
        os.remove(cfgp)
    agent.reset(0, 9)
    viz = Viz(agent.net.net, "cpu", agent.env_overrides.get("obs"), headless=True)
    for t in range(3):
        obs = agent.sim.obs(agent.net.cell_safety)
        a = agent.act(obs)
        pr = viz.update(obs, a, agent.last_trace, f"tick {t}")
    img = viz.frame()
    assert img.shape == (H, W, 3) and img.std() > 10
    assert pr["q"].shape == (5,) and np.isclose(pr["q"].mean(), pr["V"], atol=1e-3)
    assert pr["saliency"].shape == (23, 33) and pr["saliency"].max() > 0
    assert len(pr["acts"]) == 3
