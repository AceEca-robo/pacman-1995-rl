import json

import numpy as np

import pytest

from env.pacman_env import PacmanEnv
from tools.record_death_prefixes import env_config, record


@pytest.fixture(scope="module")
def recorded():
    from agents.heuristic_agent import HeuristicAgent
    agent = HeuristicAgent()
    envs = [PacmanEnv(config=env_config({}, "random:1-16")) for _ in range(3)]
    try:
        prefixes, games = record(lambda obs: [agent.act(o) for o in obs], envs,
                                 range(5000, 5003), max_steps=3000)
    finally:
        for e in envs:
            e.close()
    return prefixes, games


def test_death_prefixes_recorded(recorded):
    prefixes, games = recorded
    assert len(games) == 3 and prefixes
    assert {p["before_death"] for p in prefixes} == {40, 20}
    assert len(prefixes) == 2 * sum(g["kept_deaths"] for g in games)
    for p in prefixes:
        assert p["level_tick"] >= 60 and p["steps"] == len(p["actions"])
        assert 1 <= p["lives"] <= 3 and 1 <= p["maze"] <= 16


def test_death_prefix_replay_reproduces_positions(recorded, tmp_path):
    prefixes, _ = recorded
    path = tmp_path / "d.json"
    path.write_text(json.dumps({"n_actions": 5, "prefixes": prefixes}))
    e = PacmanEnv(config={"prefix_prob": 1.0, "prefix_file": str(path)})
    try:
        for p in prefixes[:6]:
            e.replay_prefix(p)  # raises on any mismatch; checked here as well
            st = e._state
            assert [st["pacman"]["x"], st["pacman"]["y"]] == p["pacman"]
            assert [[g["x"], g["y"]] for g in st["ghosts"]] == p["ghosts"]
            assert (st["lives"], st["level"]) == (p["lives"], p["level"])
        # a reset through the prefix path: the episode starts at the prefix's end
        obs, info = e.reset(seed=1)
        assert info["prefix"] in {p["seed"] for p in prefixes}
        # a wrong position is caught
        bad = {**prefixes[0], "pacman": [0, 0]}
        with pytest.raises(RuntimeError):
            e.replay_prefix(bad)
    finally:
        e.close()


def test_prefix_mix_weights(recorded, tmp_path):
    prefixes, _ = recorded
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps({"n_actions": 5, "prefixes": prefixes[:1]}))
    b.write_text(json.dumps({"n_actions": 5, "prefixes": prefixes[1:2]}))
    e = PacmanEnv(config={"prefix_prob": 1.0, "prefix_mix": [{"file": str(a), "weight": 3},
                                                             {"file": str(b), "weight": 1}]})
    try:
        assert list(e._prefix_weights) == [0.75, 0.25]
        assert [len(s) for s in e._prefix_sets] == [1, 1]
    finally:
        e.close()


def test_avoidability_prefix_replays_to_the_death(recorded):
    """tools/avoidability.py cuts the T-20 actions k ticks before the death
    (no food check); the recording agent replayed from there dies exactly k
    ticks later."""
    from agents.heuristic_agent import HeuristicAgent
    from tools.avoidability import prefix_at, select
    prefixes, _ = recorded
    d = select(prefixes, 1, None, 0)[0]
    agent = HeuristicAgent()
    e = PacmanEnv(config=env_config({}, 1))
    try:
        for k in (20, 40):
            e.replay_prefix(prefix_at(d, k))
            assert (e._state["level"], e._state["lives"]) == (d["level"], d["lives"])
            obs, _ = e._after_reset(d["seed"])
            for t in range(1, k + 1):
                obs, _, _, _, info = e.step(agent.act(obs))
                if info["events"]["death"]:
                    break
            assert t == k and info["events"]["death"]
    finally:
        e.close()


def test_death_windows_cut_prefix(recorded, tmp_path):
    """tools/escape_demos.py life_tick and the env's prefix_mix
    before_death window: a prefix cut back to 40 ticks before the death is
    the recorded T-40 prefix; window resets stay in the death's life."""
    from env.pacman_env import cut_prefix
    from tools.escape_demos import life_tick
    prefixes, _ = recorded
    t20 = [p for p in prefixes if p["before_death"] == 20]
    t40 = {(p["seed"], p["steps"] + 40): p for p in prefixes if p["before_death"] == 40}
    wins = [{**p, "life_tick": life_tick(p)} for p in t20]
    assert all(w["life_tick"] <= w["level_tick"] for w in wins)
    e = PacmanEnv(config=env_config({}, 1))
    try:
        long = [w for w in wins if w["life_tick"] >= 40]
        assert long
        for p in long[:3]:
            e.replay_prefix(cut_prefix(p, 40))
            ref = t40[(p["seed"], p["steps"] + 20)]
            st = e._state
            assert [st["pacman"]["x"], st["pacman"]["y"]] == ref["pacman"]
            assert [[g["x"], g["y"]] for g in st["ghosts"]] == ref["ghosts"]
    finally:
        e.close()
    path = tmp_path / "w.json"
    path.write_text(json.dumps({"n_actions": 5, "prefixes": wins}))
    e = PacmanEnv(config={"prefix_prob": 1.0, "prefix_mix": [
        {"file": str(path), "weight": 1.0, "before_death": [30, 59]}]})
    try:
        assert all(p["life_tick"] >= 30 for p in e._prefix_sets[0])
        for i in range(4):
            _, info = e.reset(seed=i)  # replay_prefix raises on a wrong level / lives
            assert info["prefix"] in {p["seed"] for p in wins}
    finally:
        e.close()


def test_escape_demo_episodes(recorded, tmp_path):
    """tools/escape_demos.py build_demos: an episode replayed from a death
    prefix becomes demo frames with n-step fields, loadable as a demo group."""
    from agents.heuristic_agent import HeuristicAgent
    from agents.shared_replay import SharedReplay
    from env.pacman_env import load_env_config
    from tools.avoidability import prefix_at, select
    from tools.escape_demos import build_demos
    prefixes, _ = recorded
    deaths = select(prefixes, 2, None, 0)
    agent = HeuristicAgent()
    env_cfg = {**load_env_config({"obs": {"cell_safety": True, "safe_exits": True}}),
               "hunger_limit": 0}
    best = []
    e = PacmanEnv(config=env_cfg)
    try:  # the heuristic's own 20 steps from T-40: it dies only at T-0
        for di, d in enumerate(deaths):
            e.replay_prefix(prefix_at(d, 40))
            obs, _ = e._after_reset(d["seed"])
            acts, food = [], 0
            for _ in range(20):
                a = agent.act(obs)
                obs, _, _, _, info = e.step(a)
                acts.append(a)
                food += info["events"]["eaten_dot"] + info["events"]["eaten_energizer"]
            best.append((di, 40, {"actions": acts, "food": food,
                                  "random": [t % 3 == 0 for t in range(20)]}))
    finally:
        e.close()
    arrays, meta = build_demos(env_cfg, 5, 0.995, deaths, best, 20)
    assert int(arrays["valid"].sum()) == 40 and len(meta) == 2
    v = arrays["valid"].astype(bool)
    assert (arrays["offn"][v] >= 1).all() and (arrays["offn"][v] <= 5).all()
    assert np.allclose(arrays["disc1"][v], 0.995)  # no terminal: truncated episodes
    # the random flags sit on the episodes' transitions, in order
    want = [t % 3 == 0 for t in range(20)]
    assert arrays["rand"][v].tolist() == want + want and not arrays["rand"][~v].any()
    np.savez(tmp_path / "esc.npz", **arrays)
    r = SharedReplay.create([64], [300, int(arrays["sizes"][0])], arrays["obs"].shape[1],
                            arrays["vec"].shape[1], demo_groups=[1, 1])
    try:
        r.load_demo(str(tmp_path / "esc.npz"), group=1)
        assert r.demo_transitions(1) == 40 and r.demo_transitions(0) == 0
    finally:
        r.close()
