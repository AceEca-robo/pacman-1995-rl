"""sim/ against the real game: byte-equal state lines (docs/sim_spec.md)."""
import json
import os
import subprocess
import sys

import pytest

from sim.pacman_sim import PacmanSim
from sim.verify import compare, random_actions

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAME = os.path.join(ROOT, "game", "pacman")
V04 = os.path.join(ROOT, "runs", "best_v04", "best.pt")
needs_game = pytest.mark.skipif(not os.path.exists(GAME), reason="game not built")


def verify(*args):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "verify_sim.py"), *args],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout


@needs_game
@pytest.mark.parametrize("policy", ["random", "sticky"])
def test_random_policies_50_games(policy):
    """The readiness criterion: 50 games x 3000 steps, start level random:1-16."""
    out = verify("--policy", policy, "--games", "50", "--ticks", "3000")
    assert "150000 steps compared, 0 diverged" in out


@needs_game
@pytest.mark.skipif(not os.path.exists(V04), reason="runs/best_v04/best.pt not present")
def test_best_v04_50_games():
    """best_v04's own games (ghosts eaten, levels cleared): 50 x 3000 steps."""
    out = verify("--policy", "agent", "--checkpoint", V04, "--games", "50", "--ticks", "3000")
    assert "150000 steps compared, 0 diverged" in out


@needs_game
def test_compare_detects_a_difference():
    class Off(PacmanSim):
        def _pac_go(self, dd):     # pacman ignores "left"
            super()._pac_go(dd if dd != 2 else 4)
    n, d, _ = compare(3, 1, [2] * 50, Off)
    assert d is not None and n <= 50


def test_state_matches_json_line():
    sim = PacmanSim()
    sim.reset(5, 7)
    for a in random_actions(5, 500):
        sim.advance(a)
        assert json.loads(sim.state_json()) == sim.state()


def test_clone_is_independent():
    sim = PacmanSim()
    sim.reset(11, 3)
    acts = random_actions(11, 2000)
    for a in acts[:300]:
        sim.advance(a)
    c = sim.clone()
    a_lines = []
    for a in acts[300:]:
        sim.advance(a)
        a_lines.append(sim.state_json())
    b_lines = []
    for a in acts[300:]:
        c.advance(a)
        b_lines.append(c.state_json())
    assert a_lines == b_lines
