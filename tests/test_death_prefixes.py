import json

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
