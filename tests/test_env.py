import glob
import os
import tempfile

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from env.pacman_env import CH, CHANNELS, HEIGHT, ROOT, WIDTH, PacmanEnv, d_max, maze_layouts

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, "game", "pacman")),
    reason="needs built game/pacman",
)


def alive(pid):
    return os.path.exists(f"/proc/{pid}")


def sock_dirs():
    return set(glob.glob(os.path.join(tempfile.gettempdir(), "pacman-env-*")))


@pytest.fixture
def env():
    e = PacmanEnv()
    yield e
    e.close()


def check_obs(obs):
    g = obs["grid"]
    assert g.shape[0] == len(CHANNELS)
    assert g[CH["pacman"]].sum() == 1
    pac_dirs = g[CH["pacman_up"]:CH["pacman_right"] + 1]
    assert pac_dirs.sum() <= 1 and np.all(pac_dirs.sum(0) <= g[CH["pacman"]])
    ghosts = g[CH["ghost_normal"]:CH["ghost_eyes"] + 1].sum(0)
    assert ghosts.sum() == 1  # 4 ghosts, 1/4 each
    ghost_dirs = g[CH["ghost_up"]:CH["ghost_right"] + 1].sum(0)
    assert np.all(ghost_dirs <= ghosts)  # directions only where ghosts are
    assert not np.any(g[CH["walls"]] * g[CH["gate"]])


def test_headless_needs_no_display(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    e = PacmanEnv()
    try:
        obs, _ = e.reset(seed=0)
        check_obs(obs)
        assert obs["grid"][CH["gate"]].sum() == 3
    finally:
        e.close()


def test_check_env(env):
    check_env(env, skip_render_check=True)


def rollout(seed, actions):
    e = PacmanEnv()
    out = []
    try:
        obs, _ = e.reset(seed=seed)
        out.append(obs)
        for a in actions:
            obs, r, term, trunc, info = e.step(a)
            out.append((obs, r, term, trunc, info))
            if term or trunc:
                obs, _ = e.reset()
                out.append(obs)
    finally:
        e.close()
    return out


def same(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    return a == b


def test_same_seed_same_actions(env):
    actions = np.random.default_rng(0).integers(5, size=2000).tolist()
    a = rollout(7, actions)
    b = rollout(7, actions)
    assert sum(1 for x in a if isinstance(x, tuple) and x[2]) >= 2  # crosses game overs
    assert same(a, b)


def test_reset_seed_restarts_game(env):
    first, _ = env.reset(seed=3)
    for a in [3, 3, 3, 0, 0]:
        env.step(a)
    again, info = env.reset(seed=3)
    assert same(first, again)
    assert (info["score"], info["lives"]) == (0, 3)


def test_truncation_and_reset_mid_game():
    e = PacmanEnv(config={"max_episode_steps": 20})
    try:
        e.reset(seed=1)
        truncs = [e.step(4)[3] for _ in range(20)]
        assert truncs == [False] * 19 + [True]
        _, info = e.reset()
        assert (info["score"], info["lives"], info["level"]) == (0, 3, 1)
    finally:
        e.close()


def test_close_leaves_nothing():
    before = sock_dirs()
    e = PacmanEnv()
    e.reset(seed=0)
    e.step(0)
    game = e._res["game"].pid
    e.close()
    e.close()  # idempotent
    assert not alive(game)  # waited for, so no zombie either
    assert sock_dirs() <= before


def test_1000_random_steps(env):
    obs, _ = env.reset(seed=11)
    env.action_space.seed(11)
    assert env.observation_space.contains(obs)
    for _ in range(1000):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert np.isfinite(r)
        check_obs(obs)
        if term or trunc:
            obs, info = env.reset()
            assert info["lives"] == 3


def test_seeds_0_and_1_differ():
    actions = [3] * 300
    assert not same(rollout(0, actions), rollout(1, actions))


def test_events_match_game_score():
    """Score change per step = 10 per dot + 1000 per bonus point + level
    completion bonus + ghosts (100 * 2^k each, doubling within one super)."""
    from agents.heuristic_agent import HeuristicAgent

    agent = HeuristicAgent()
    e = PacmanEnv()
    totals = dict.fromkeys(["eaten_dot", "ghost_eaten", "level_up", "bonus_eaten"], 0)
    try:
        for seed in (9, 42):  # 9: eats a bonus point; 42: levels, long ghost chains
            obs, info = e.reset(seed=seed)
            while True:
                prev = e._state
                obs, r, term, trunc, info = e.step(agent.act(obs))
                ev, st = info["events"], e._state
                for k in totals:
                    totals[k] += ev[k]
                rest = st["score"] - prev["score"] - 10 * ev["eaten_dot"]
                if ev["bonus_eaten"] and prev["bonus"]["type"] == "point":
                    rest -= 1000
                if ev["level_up"]:
                    assert 0 <= rest <= 5000  # remaining level bonus
                elif ev["ghost_eaten"]:
                    assert rest >= 100 * ev["ghost_eaten"] and rest % 100 == 0
                else:
                    assert rest == 0, (ev, rest)
                expected = sum(e.cfg["reward"][k] * n for k, n in ev.items())
                assert r == pytest.approx(expected)
                if term or trunc:
                    break
    finally:
        e.close()
    assert all(totals.values()), totals  # every event type was exercised


def shaped_run(gamma, seed, agent=None, steps=3000):
    from agents.heuristic_agent import HeuristicAgent
    agent = agent or HeuristicAgent()
    e = PacmanEnv(config={"shaping": {"enabled": True, "k_dist": 0.1, "gamma": gamma}})
    try:
        obs, _ = e.reset(seed=seed)
        phis = [e._phi]
        shaping = []
        for _ in range(steps):
            obs, r, term, trunc, info = e.step(agent.act(obs))
            shaping.append(info["shaping"])
            phis.append(e._phi)
            if term or trunc:
                break
    finally:
        e.close()
    return np.array(shaping), np.array(phis)


def test_shaping_telescopes():
    # seed 42: the heuristic dies three times, clears levels (Phi = 0 without
    # food) and the game ends (Phi(terminal) = 0)
    f, phi = shaped_run(1.0, 42)
    assert (phi == 0).any() and (phi > 1).any()
    assert phi[-1] == 0.0  # terminal
    assert f.sum() == pytest.approx(phi[-1] - phi[0])
    g = 0.99
    f, phi = shaped_run(g, 42)
    disc = g ** np.arange(len(f))
    assert (disc * f).sum() == pytest.approx(g ** len(f) * phi[-1] - phi[0])


def bare_env(k=0.5, gamma=0.99):
    """PacmanEnv with only the shaping state, no game process."""
    e = PacmanEnv.__new__(PacmanEnv)
    e._k_dist, e._shaping_gamma, e._neighbours, e._d_max = k, gamma, {}, d_max()
    return e


def test_potential_bfs():
    e = bare_env()
    rows = ["#" * WIDTH] * HEIGHT
    rows[1] = "#" + " " * (WIDTH - 2) + "#"
    rows[2] = "#" * 5 + "-" + "#" * (WIDTH - 6)
    rows[3] = "#" * 5 + "." + "#" * (WIDTH - 6)
    st = {"grid": rows, "pacman": {"x": 1, "y": 1}}
    assert e._potential(st) == 0.0  # the only dot is behind the gate
    rows[1] = rows[1][:10] + "o" + rows[1][11:]
    assert e._potential(st) == 0.5 * (d_max() - 9)


def test_standing_still_far_from_food_is_negative():
    """With Phi >= 0 and gamma < 1, a step without moving costs
    (gamma - 1) * Phi <= 0 on top of the step penalty, at any distance."""
    from env.pacman_env import load_env_config
    step_penalty = load_env_config(None)["reward"]["step"]
    e = bare_env(k=0.1)
    rows = ["#" * WIDTH] * HEIGHT
    rows[1] = "#" + " " * (WIDTH - 3) + ".#"
    for x in (1, 10, 20, 30):  # 30 .. 1 cells from the dot
        st = {"grid": rows, "pacman": {"x": x, "y": 1}}
        e._phi = e._potential(st)
        total = e._shape(st, terminated=False) + step_penalty
        assert total < step_penalty / 2 < 0
    # the old Phi = -k * d paid for standing far away: (1 - gamma) * k * d
    assert (1 - 0.99) * 0.1 * 30 + step_penalty > 0


def test_maze_layouts_match_the_game():
    e = PacmanEnv()
    try:
        e.reset(seed=0)
        cells = [r.replace(".", " ").replace("o", " ") for r in e._state["grid"]]
    finally:
        e.close()
    assert cells == maze_layouts()[0]
    assert 40 < d_max() < WIDTH * HEIGHT


def test_four_actions():
    e = PacmanEnv(config={"actions": 4})
    try:
        assert e.action_space.n == 4
        check_env(e, skip_render_check=True)
        e.reset(seed=0)
        for a in [3, 3, 0, 1, 2] * 20:
            e.step(a)
        with pytest.raises(ValueError):
            e.step(4)
    finally:
        e.close()


def test_no_ghosts_stay_home():
    e = PacmanEnv(config={"ghosts": False})
    try:
        e.reset(seed=0)
        start = [(g["x"], g["y"]) for g in e._state["ghosts"]]
        for a in np.random.default_rng(0).integers(5, size=2000):
            _, _, term, trunc, info = e.step(a)
            assert [(g["x"], g["y"]) for g in e._state["ghosts"]] == start
            assert info["events"]["death"] == 0
            if term or trunc:
                break
    finally:
        e.close()


def test_hunger_limit_ends_standing_still():
    e = PacmanEnv(config={"hunger_limit": 15, "ghosts": False})
    try:
        e.reset(seed=0)
        for t in range(1, 16):
            _, r, term, trunc, info = e.step(0)  # up from the start cell: a wall
            assert e._state["pacman"]["dir"] == "S"
            if t < 15:
                assert not term and not info["hunger"]
        assert term and info["hunger"] and not trunc
        assert r == pytest.approx(-20.0 - 0.02)
        _, info = e.reset()  # the game was not over: a new one is started
        assert (info["score"], info["lives"]) == (0, 3)
    finally:
        e.close()


def test_hunger_counter_resets_on_food():
    e = PacmanEnv(config={"hunger_limit": 3, "ghosts": False})
    try:
        e.reset(seed=0)
        hungry = []
        for a in [3, 3, 3, 3, 3, 3]:  # right along the dots of the start row
            _, _, term, _, info = e.step(a)
            hungry.append(e._hungry)
            assert not info["hunger"]
        assert min(hungry) == 0
    finally:
        e.close()


def test_hunger_limit_zero_changes_nothing():
    actions = np.random.default_rng(3).integers(5, size=1500).tolist()
    runs = []
    for cfg in (None, {"hunger_limit": 0}):
        e = PacmanEnv(config=cfg)
        try:
            out = [e.reset(seed=4)[0]]
            for a in actions:
                obs, r, term, trunc, info = e.step(a)
                assert not info["hunger"]
                out.append((obs, r, term, trunc))
                if term or trunc:
                    out.append(e.reset()[0])
        finally:
            e.close()
        runs.append(out)
    assert same(runs[0], runs[1])


def field_env():
    e = PacmanEnv.__new__(PacmanEnv)
    e._neighbours, e._d_max, e._field_cache = {}, d_max(), (None, None)
    return e


def test_food_distance_channel_synthetic():
    e = field_env()
    rows = ["#" * WIDTH] * HEIGHT
    rows[1] = "#." + " " * 11 + "#" * (WIDTH - 13)          # corridor x=1..12, food at x=1
    rows[2] = "#" * 12 + " " + "#" * (WIDTH - 13)            # then down at x=12
    rows[3] = "#" * 12 + " " + "#" * (WIDTH - 13)
    f = e._food_field({"grid": rows}) * d_max()
    assert f[1, 1] == 0                                       # the food cell
    path = [f[1, x] for x in range(1, 13)] + [f[2, 12], f[3, 12]]
    assert np.allclose(path, np.arange(14), atol=1e-4)        # BFS steps, around the corner
    assert np.all(np.diff(path) > 0)                          # grows along the path
    assert f[0].sum() == 0 and f[2, 0] == 0                   # walls are 0
    rows[1] = rows[1].replace(".", " ")
    e._field_cache = (None, None)
    assert not e._food_field({"grid": rows}).any()            # no food: all 0


def test_food_distance_channel_real_board():
    from env.pacman_env import FOOD_DISTANCE_CH, _passable_neighbours
    e = PacmanEnv(config={"obs": {"food_distance": True}})
    try:
        obs, _ = e.reset(seed=0)
        rng = np.random.default_rng(0)
        for _ in range(300):
            obs, *_ = e.step(int(rng.integers(5)))
        assert obs["grid"].shape[0] == len(CHANNELS) + 1
        assert e.observation_space.contains(obs)
        cells = "".join(e._state["grid"])
        nbrs = _passable_neighbours(cells)
        d = np.rint(obs["grid"][FOOD_DISTANCE_CH].ravel() * d_max()).astype(int)
        for c, ch in enumerate(cells):
            if ch in ".o":
                assert d[c] == 0
            elif nbrs[c] and d[c] > 0:  # one step closer along a shortest path
                assert min(d[n] if cells[n] not in ".o" else 0 for n in nbrs[c]) == d[c] - 1
    finally:
        e.close()


def test_steps_since_food_in_vec():
    e = PacmanEnv(config={"obs": {"steps_since_food": True}, "ghosts": False})
    try:
        obs, _ = e.reset(seed=0)
        assert obs["vec"].shape == (4,) and obs["vec"][3] == 0
        for t in range(1, 251):
            obs, *_ = e.step(0)  # into the wall above the start: never eats
            assert obs["vec"][3] == pytest.approx(min(t / 200, 1.0))
        obs, _, _, _, info = e.step(3)  # right: a dot next to the start cell
        assert info["events"]["eaten_dot"] and obs["vec"][3] == 0
    finally:
        e.close()


def test_obs_flags_off_by_default():
    e = PacmanEnv()
    try:
        obs, _ = e.reset(seed=0)
        assert obs["grid"].shape[0] == len(CHANNELS) and obs["vec"].shape == (3,)
    finally:
        e.close()


def test_prefix_reset_starts_in_the_endgame():
    e = PacmanEnv(config={"prefix_prob": 1.0})
    try:
        seeds = set()
        for i in range(4):
            obs, info = e.reset(seed=i) if i == 0 else e.reset()
            st = e._state
            left = sum(r.count(".") + r.count("o") for r in st["grid"])
            assert left <= 15 and st["lives"] == 3 and st["level"] == 1
            assert info["prefix"] is not None
            assert obs["grid"][CH["food"]].sum() + obs["grid"][CH["superfood"]].sum() == left
            seeds.add(info["prefix"])
            for _ in range(20):
                obs, r, term, trunc, info = e.step(3)
                if term:
                    break
        assert len(seeds) > 1
    finally:
        e.close()


def test_prefix_prob_zero_changes_nothing():
    actions = np.random.default_rng(5).integers(5, size=800).tolist()
    runs = []
    for cfg in (None, {"prefix_prob": 0.0}):
        e = PacmanEnv(config=cfg)
        try:
            obs, info = e.reset(seed=6)
            assert info["prefix"] is None
            out = [obs] + [e.step(a)[:4] for a in actions]
        finally:
            e.close()
        runs.append(out)
    assert same(runs[0], runs[1])


def test_endgame_dot_reward_formula():
    e = PacmanEnv(config={"endgame_dot": {"enabled": True, "k": 20}, "prefix_prob": 1.0})
    try:
        e.reset(seed=0)
        seen = 0
        for _ in range(400):
            prev_left = sum(r.count(".") + r.count("o") for r in e._state["grid"])
            obs, r, term, trunc, info = e.step(int(np.random.default_rng(seen).integers(5)))
            ev = info["events"]
            left = sum(row.count(".") + row.count("o") for row in e._state["grid"])
            if ev["eaten_dot"]:
                seen += 1
                assert info["endgame_bonus"] == pytest.approx(20 / max(left, 1))
                assert left == prev_left - 1
            else:
                assert info["endgame_bonus"] == 0.0
            if term or trunc:
                e.reset()
        assert seen > 0
    finally:
        e.close()
    # the formula itself: dot reward 1 * (1 + 20 / max(left, 1))
    for left, total in ((15, 1 + 20 / 15), (1, 21.0), (0, 21.0)):
        assert 1.0 * (1 + 20 / max(left, 1)) == pytest.approx(total)
