"""Shared-memory frame replay (agents/shared_replay.py), the DQfD loss
(agents/dqn.py) and the env pool (agents/env_pool.py)."""

import numpy as np
import pytest
import torch

from agents.dqn import PACKED, NStep, large_margin
from agents.shared_replay import Sampler, SharedReplay, StreamWriter

PLANES = 4


def frame(k):
    """A packed frame whose first bytes encode the id k."""
    f = np.zeros((PLANES, PACKED), np.uint8)
    f[0, :8] = np.frombuffer(np.int64(k).tobytes(), np.uint8)
    return f


def frame_id(f):
    return int(np.frombuffer(f[0, :8].tobytes(), np.int64)[0])


@pytest.fixture
def replay():
    made = []

    def make(agent_sizes, demo_sizes=(), demo_groups=None):
        r = SharedReplay.create(agent_sizes, demo_sizes, PLANES, 3, demo_groups)
        made.append(r)
        return r
    yield make
    for r in made:
        r.close()


def drive(writer, n_envs, steps, rng, refs=None, ids_fn=None, after=None, start=0):
    """Random episodes (term / trunc) in each env; frame ids are unique.
    refs: per env NStep references collecting the expected transitions.
    ids_fn(t): the envs that step at t (default all). Returns the next id."""
    nxt, nsteps = start, {}
    cur = np.arange(nxt, nxt + n_envs)
    nxt += n_envs
    writer.begin(np.stack([frame(k) for k in cur]), cur[:, None].repeat(3, 1).astype(np.float32))
    for t in range(steps):
        ids = np.arange(n_envs) if ids_fn is None else ids_fn(t)
        if not len(ids):
            continue
        act = rng.integers(5, size=len(ids))
        rew = rng.normal(size=len(ids)).astype(np.float32)
        u = rng.random(len(ids))
        term, trunc = u < 0.04, (u >= 0.04) & (u < 0.07)
        nobs = np.arange(nxt, nxt + len(ids))
        nxt += len(ids)
        ended = term | trunc
        fin = np.arange(nxt, nxt + int(ended.sum()))
        nxt += len(fin)
        if refs is not None:
            fi = iter(fin)
            for j, i in enumerate(ids):
                last = next(fi) if ended[j] else nobs[j]
                ns = nsteps.setdefault(i, NStep(writer.n, writer.gamma))
                refs += [(int(o), int(a), float(r), int(nx), float(d)) for o, _, a, r, nx, _, d
                         in ns.push(cur[i], None, act[j], rew[j], last, None, term[j], ended[j])]
        writer.step(act, rew, np.stack([frame(k) for k in nobs]),
                    nobs[:, None].repeat(3, 1).astype(np.float32), term, trunc,
                    np.stack([frame(k) for k in fin]) if len(fin) else None,
                    fin[:, None].repeat(3, 1).astype(np.float32) if len(fin) else None,
                    ids=ids)
        cur[ids] = nobs
        if after:
            after()
    return nxt


def test_frames_match_nstep(replay):
    """Every final slot is exactly one NStep transition (s, a, R, s_{t+k},
    gamma^k / 0), with the 1-step fields of s_t -> s_{t+1}."""
    r = replay([5000] * 3)
    w = StreamWriter(r, [0, 1, 2], 3, 0.9)
    refs = []
    rng = np.random.default_rng(0)
    drive(w, 3, 300, rng, refs, ids_fn=lambda t: np.flatnonzero(rng.random(3) < 0.7))
    s = Sampler(r, 0.0, 1e-6, guard=64, n=3)
    s.register()
    slots = np.flatnonzero(r.valid)
    assert len(slots) == len(refs) == s.n_valid == w.size
    b = s.gather(slots)
    got = sorted((frame_id(b["obs"][j]), int(b["act"][j]), round(float(b["ret"][j]), 4),
                  frame_id(b["obsn"][j]), round(float(b["discn"][j]), 6)) for j in range(len(slots)))
    want = sorted((o, a, round(ret, 4), nx, round(d, 6)) for o, a, ret, nx, d in refs)
    assert got == want
    # 1-step: gamma unless terminated (then the n-step discount is 0 too)
    assert set(b["disc1"].tolist()) <= {0.0, float(np.float32(0.9))}
    assert np.all(b["discn"][b["disc1"] == 0] == 0)
    assert np.all(b["vec1"][:, 0] == np.array([frame_id(x) for x in b["obs1"]]))


def test_one_step_fields(replay):
    """obs1 is the frame written right after s_t (the episode's last
    observation at its end) and rew1 the step's reward."""
    r = replay([4000])
    w = StreamWriter(r, [0], 1, 0.9)  # n = 1: the n-step transition is the 1-step one
    refs = []
    drive(w, 1, 400, np.random.default_rng(1), refs)
    s = Sampler(r, 0.0, 1e-6, guard=64, n=1)
    s.register()
    slots = np.flatnonzero(r.valid)
    b = s.gather(slots)
    assert np.allclose(b["rew1"], b["ret"]) and np.allclose(b["disc1"], b["discn"])
    assert [frame_id(x) for x in b["obs1"]] == [frame_id(x) for x in b["obsn"]]


def test_ring_overwrite_keeps_transitions_consistent(replay):
    """Many laps over small rings: every sampled transition's frames are the
    ones it was written with (old slots leave the tree before reuse)."""
    r = replay([200, 200])
    w = StreamWriter(r, [0, 1], 3, 0.9, timeout=5)
    s = Sampler(r, 0.6, 1e-6, guard=16, n=3)
    rng = np.random.default_rng(2)
    checked = [0]

    def learn():
        s.register()
        if s.n_valid < 32:
            return
        b, idx, wts = s.sample(rng, 32, 0.4)
        o, o1, on = ([frame_id(x) for x in b[k]] for k in ("obs", "obs1", "obsn"))
        for j in range(32):
            # ids grow along a stream; an episode's frames are written in order
            assert o[j] < o1[j] <= on[j] and b["vec"][j, 0] == o[j]
            assert b["vecn"][j, 0] == on[j]
        s.update_priorities(idx, rng.random(32))
        checked[0] += 1

    drive(w, 2, 3000, rng, after=learn)
    assert checked[0] > 2900
    assert s.n_valid == int((s.tree.get(np.arange(r.frames)) > 0).sum())


def test_demo_stays(replay):
    """Demo streams are never overwritten and keep being sampled; their
    priorities carry eps_demo."""
    r = replay([100, 100], demo_sizes=[300])
    dw = StreamWriter(r, [0], 3, 0.9)
    drive(dw, 1, 250, np.random.default_rng(3))
    demo = r.obs[:r.demo_frames].copy()
    n_demo = r.demo_transitions()
    assert n_demo > 200
    s = Sampler(r, 0.6, 1e-6, eps_demo=1.0, guard=16, n=3)
    s.register()
    aw = StreamWriter(r, [1, 2], 3, 0.9, timeout=5)
    rng = np.random.default_rng(4)
    seen_demo = [0]

    def learn():
        s.register()
        b, idx, _ = s.sample(rng, 32, 0.4)
        assert np.array_equal(b["demo"], idx < r.demo_frames)
        seen_demo[0] += int(b["demo"].sum())
        s.update_priorities(idx, np.zeros(32))  # |TD| 0: demo keeps eps_demo
    drive(aw, 2, 2000, rng, after=learn, start=10**6)
    assert np.array_equal(r.obs[:r.demo_frames], demo)
    assert r.demo_transitions() == n_demo
    pr = s.tree.get(np.flatnonzero(r.valid[:r.demo_frames]))
    assert (pr > 0).all() and pr.min() >= (1.0 + 1e-6) ** 0.6 - 1e-9
    assert seen_demo[0] > 0
    with pytest.raises(RuntimeError, match="demo stream is full"):
        drive(dw, 1, 100, np.random.default_rng(5))


def test_sumtree_prefix_sum():
    from agents.dqn import SumTree
    rng = np.random.default_rng(0)
    t = SumTree(1000)
    v = rng.random(1000)
    t.set(np.arange(1000), v)
    for n in (0, 1, 2, 511, 512, 513, 999, 1000):
        assert t.prefix_sum(n) == pytest.approx(v[:n].sum(), abs=1e-9)


def test_demo_frac_sampling(replay):
    """A fixed demo share: exactly round(frac * batch) demo samples per
    batch, and the importance weights follow the two-part probabilities."""
    r = replay([200, 200], demo_sizes=[300])
    drive(StreamWriter(r, [0], 3, 0.9), 1, 250, np.random.default_rng(3))
    s = Sampler(r, 0.6, 1e-6, eps_demo=1.0, guard=16, n=3)
    aw = StreamWriter(r, [1, 2], 3, 0.9, timeout=5)
    drive(aw, 2, 150, np.random.default_rng(1), after=s.register, start=10**6)
    s.register()
    rng = np.random.default_rng(2)
    s.update_priorities(np.flatnonzero(s.tree.get(np.arange(r.frames)) > 0),
                        rng.random(s.n_valid) * 3)
    demo_mass = s.tree.prefix_sum(r.demo_frames)
    total = s.tree.total
    for frac in (0.0, 0.1, 0.4, 1.0):
        b, idx, w = s.sample(rng, 40, 1.0, demo_frac=frac)
        k = int(round(frac * 40))
        assert int(b["demo"].sum()) == k
        p = s.tree.get(idx)
        prob = np.where(b["demo"], (k / 40) * p / demo_mass, (1 - k / 40) * p / (total - demo_mass))
        want = 1.0 / (s.n_valid * prob)
        assert np.allclose(w, want / want.max(), rtol=1e-5)
    _, _, w = s.sample(rng, 40, 1.0)  # None: plain PER
    assert w.max() == pytest.approx(1.0)


def test_demo_save_load(replay, tmp_path):
    r = replay([64], demo_sizes=[200, 200])
    drive(StreamWriter(r, [0, 1], 3, 0.9), 2, 150, np.random.default_rng(6))
    r.save_demo(str(tmp_path / "demo.npz"))
    r2 = replay([64], demo_sizes=[200, 200])
    r2.load_demo(str(tmp_path / "demo.npz"))
    for k in ("obs", "vec", "act", "ret", "discn", "offn", "valid"):
        assert np.array_equal(getattr(r, k)[:r.demo_frames], getattr(r2, k)[:r2.demo_frames])
    assert np.array_equal(r.final[:2], r2.final[:2])
    with pytest.raises(ValueError):
        replay([64], demo_sizes=[100, 100]).load_demo(str(tmp_path / "demo.npz"))


def test_demo_groups_sampling(replay, tmp_path):
    """Two demo groups (BC demos, escape demos): each is loaded into its own
    streams, the demo samples split by demo_shares, and the importance
    weights follow the per-segment probabilities."""
    src = replay([64], demo_sizes=[300])
    drive(StreamWriter(src, [0], 3, 0.9), 1, 250, np.random.default_rng(7), start=5 * 10**5)
    src.save_demo(str(tmp_path / "escape.npz"))
    r = replay([200, 200], demo_sizes=[300, 300, 300], demo_groups=[2, 1])
    drive(StreamWriter(r, [0, 1], 3, 0.9), 2, 250, np.random.default_rng(3))
    r.load_demo(str(tmp_path / "escape.npz"), group=1)
    lo, hi = r.group_bounds[1], r.group_bounds[2]
    assert np.array_equal(r.obs[lo:hi], src.obs[:src.demo_frames])
    assert r.demo_transitions(1) == src.demo_transitions()
    assert r.demo_transitions() == r.demo_transitions(0) + r.demo_transitions(1)
    with pytest.raises(ValueError):
        Sampler(r, 0.6, 1e-6, demo_shares=[0.5, 0.4], guard=16, n=3)
    s = Sampler(r, 0.6, 1e-6, eps_demo=1.0, guard=16, n=3, demo_shares=[1 / 3, 2 / 3])
    drive(StreamWriter(r, [3, 4], 3, 0.9, timeout=5), 2, 150, np.random.default_rng(1),
          after=s.register, start=10**6)
    s.register()
    rng = np.random.default_rng(2)
    s.update_priorities(np.flatnonzero(s.tree.get(np.arange(r.frames)) > 0),
                        rng.random(s.n_valid) * 3)
    m = [s.tree.prefix_sum(int(b)) if b else 0.0 for b in r.group_bounds] + [s.tree.total]
    b, idx, w = s.sample(rng, 60, 1.0, demo_frac=0.15)  # 9 demo samples: 3 BC, 6 escape
    part = np.searchsorted(list(r.group_bounds) + [r.frames], idx, side="right") - 1
    assert np.bincount(part, minlength=3).tolist() == [3, 6, 51]
    assert np.array_equal(b["demo"], part < 2)
    ks = np.array([3, 6, 51])
    prob = ks[part] / 60 * s.tree.get(idx) / np.diff(m)[part]
    want = 1.0 / (s.n_valid * prob)
    assert np.allclose(w, want / want.max(), rtol=1e-5)
    seen = np.zeros(3, int)
    for _ in range(200):
        _, idx, w = s.sample(rng, 64, 0.4, demo_frac=0.15)
        seen += np.bincount(np.searchsorted(list(r.group_bounds) + [r.frames], idx,
                                            side="right") - 1, minlength=3)
        assert np.isfinite(w).all()
    assert seen.tolist() == [3 * 200, 7 * 200, 54 * 200]  # 10 demo: 3.33 / 6.67 -> 3 / 7


def test_large_margin():
    q = torch.tensor([[1.0, 2.0, 0.0], [3.0, 1.0, 2.5], [0.0, 0.0, 0.0]])
    act = torch.tensor([1, 0, 2])
    je = large_margin(q, act, 0.8)
    # row 0: max(1.8, 2.0, 0.8) - 2 = 0; row 1: max(3, 1.8, 3.3) - 3 = 0.3; row 2: 0.8
    assert torch.allclose(je, torch.tensor([0.0, 0.3, 0.8]))
    assert (large_margin(q, act, 0.0) >= 0).all()


def frame_batch(rng, b, planes, demo):
    from agents.dqn import N_PLANES
    p = N_PLANES
    obs = rng.integers(0, 256, (b, p, PACKED), dtype=np.uint8)
    return {"obs": obs, "obs1": rng.integers(0, 256, (b, p, PACKED), dtype=np.uint8),
            "obsn": rng.integers(0, 256, (b, p, PACKED), dtype=np.uint8),
            "vec": rng.random((b, 3), dtype=np.float32), "vec1": rng.random((b, 3), dtype=np.float32),
            "vecn": rng.random((b, 3), dtype=np.float32), "act": rng.integers(0, 5, b),
            "rew1": rng.random(b, dtype=np.float32), "disc1": np.full(b, 0.99, np.float32),
            "ret": rng.random(b, dtype=np.float32), "discn": np.full(b, 0.97, np.float32),
            "demo": demo}


def dqfd_cfg(**kw):
    import os
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "scripts"))
    from train import load_config
    cfg = {**load_config(os.path.join(root, "configs", "dqn.yaml")), "compile": False}
    cfg["dqfd"] = {"enabled": True, "lambda_1step": 1.0, "lambda_nstep": 1.0,
                   "lambda_margin": 1.0, "margin": 0.8, "l2": 1e-5, **kw}
    return cfg


def test_margin_loss_only_on_demo():
    """The margin term is added for demo samples only: with lam_n = lam_1 = 0
    the loss of a non-demo batch is 0, of a demo batch the mean margin loss."""
    from agents.dqn import DQNLearner
    torch.manual_seed(0)
    learner = DQNLearner(dqfd_cfg(lambda_1step=0.0, lambda_nstep=0.0), "cpu")
    rng = np.random.default_rng(0)
    b = frame_batch(rng, 16, None, np.zeros(16, bool))
    w = np.ones(16, np.float32)
    args = lambda bb: (  # noqa: E731
        learner._t(np.concatenate([bb["obs"], bb["obsn"]])),
        learner._t(np.concatenate([bb["vec"], bb["vecn"]])), learner._t(bb["act"], torch.long),
        learner._t(bb["rew1"]), learner._t(bb["disc1"]), learner._t(bb["ret"]),
        learner._t(bb["discn"]), learner._t(bb["demo"], torch.float32), learner._t(w),
        learner._t(np.float32(learner.lam_m)))
    loss, _, _, je = learner._frame_loss_fn(*args(b))
    assert float(loss.detach()) == 0.0 and float(je) == 0.0
    b["demo"] = np.ones(16, bool)
    loss, _, _, je = learner._frame_loss_fn(*args(b))
    with torch.no_grad():
        q = learner.online(learner.decode(learner._t(b["obs"])), learner._t(b["vec"]))
    want = large_margin(q, learner._t(b["act"], torch.long), 0.8).mean()
    assert float(loss.detach()) == pytest.approx(float(want), rel=1e-5)
    assert float(je) == pytest.approx(float(want), rel=1e-5)
    half = args(b)[:-1] + (learner._t(np.float32(learner.lam_m / 2)),)  # scheduled weight
    loss, _, _, _ = learner._frame_loss_fn(*half)
    assert float(loss.detach()) == pytest.approx(float(want) / 2, rel=1e-5)


def test_margin_loss_learns_demo_actions():
    """Updates on a demo batch make the demo actions the greedy ones."""
    from agents.dqn import DQNLearner
    torch.manual_seed(0)
    learner = DQNLearner(dqfd_cfg(), "cpu")
    rng = np.random.default_rng(1)
    b = frame_batch(rng, 32, None, np.ones(32, bool))
    for _ in range(150):
        learner.update_frames(b, np.ones(32, np.float32))
    with torch.no_grad():
        q = learner.online(learner.decode(learner._t(b["obs"])), learner._t(b["vec"]))
    assert (q.argmax(1).numpy() == b["act"]).mean() > 0.9


def test_env_pool_matches_env():
    """EnvPool steps give the same observations and rewards as the env
    itself (same seed and actions), envs stepping in any order."""
    import functools
    from agents.dqn import n_planes
    from agents.env_pool import EnvPool
    from agents.shared_replay import make_packed_env
    from env.pacman_env import PacmanEnv, load_env_config, obs_dims
    cfg = load_env_config(None)
    nc, vd = obs_dims(cfg)
    pool = EnvPool(functools.partial(make_packed_env, cfg, n_planes(nc)), 3, n_planes(nc), vd, 5,
                   min_ready=1)
    ref = [make_packed_env(cfg, n_planes(nc)) for _ in range(3)]
    try:
        grids, vecs, _ = pool.reset(seed=10)
        for i, e in enumerate(ref):
            o, _ = e.reset(seed=10 + i)
            assert np.array_equal(o["grid"], grids[i])
        rng = np.random.default_rng(0)
        ready, n = np.arange(3), 0
        while n < 300:
            acts = rng.integers(5, size=len(ready))
            pool.send(ready, acts)
            want = {}
            for i, a in zip(ready, acts):
                o, rw, te, tr, _ = ref[i].step(int(a))
                if te or tr:
                    o, _ = ref[i].reset()
                want[i] = (o, rw, te, tr)
            got_ids, res = pool.recv()
            pending = set(ready) - set(got_ids)
            while pending:  # min_ready 1: collect the rest
                more, mres = pool.recv()
                got_ids, res = np.concatenate([got_ids, more]), res + mres
                pending -= set(more)
            for i, x in zip(got_ids, res):
                o, rw, te, tr = want[i]
                assert (x[0], x[1], x[2]) == (pytest.approx(rw), te, tr)
                assert np.array_equal(pool.grid[i], o["grid"])
            ready, n = np.sort(got_ids), n + len(got_ids)
    finally:
        pool.close()
        for e in ref:
            e.close()
