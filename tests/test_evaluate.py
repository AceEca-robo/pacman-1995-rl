import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from evaluate import (HEADER, RESULTS_INTRO, RULE, deaths_per_1000_ticks,  # noqa: E402
                      update_results)


def res(label, value):
    return {"agent": label, "x": value}


def test_update_results_keeps_rows_and_sections(tmp_path):
    path = str(tmp_path / "results.md")
    with open(path, "w") as f:
        f.write(RESULTS_INTRO + HEADER + "\n" + RULE + "\n"
                + "| dqn a | 1 |\n| heuristic | 2 |\n| custom label | 3 |\n"
                + "\n## Notes\n\n- keep me\n\n## Summary of all runs\n\n| run | x |\n|---|---|\n| r | 1 |\n")
    update_results(path, res("dqn b", 4))
    update_results(path, res("dqn a", 5))  # replaces its own row
    text = open(path).read()
    for line in ("| dqn a | 5 |", "| dqn b | 4 |", "| heuristic | 2 |", "| custom label | 3 |",
                 "- keep me", "| r | 1 |"):
        assert line in text
    assert "| dqn a | 1 |" not in text and text.count(HEADER) == 1
    assert not os.path.exists(path + ".tmp")


def test_deaths_per_1000_ticks_pools_games():
    # 3 deaths in 4000 ticks: 0.75, not the mean of the per-game rates (2/100, 1/3900)
    games = [{"deaths": 2, "length": 100}, {"deaths": 1, "length": 3900}]
    assert deaths_per_1000_ticks(games) == 0.75
    assert deaths_per_1000_ticks([{"deaths": 0, "length": 500}]) == 0.0
    assert deaths_per_1000_ticks([]) == 0.0


def test_levels_suite_range_stop_at_clear(tmp_path):
    """--level-range limits the per-level games to those levels;
    --stop-at-clear ends them at the first level up (level-1 games play on)."""
    out = tmp_path / "lv.json"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "evaluate.py"),
                        "--agent", "heuristic", "--levels", str(out), "--episodes", "2",
                        "--level-episodes", "2", "--level-range", "1-3", "--stop-at-clear",
                        "--workers", "2", "--max-steps", "3000"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    res = json.loads(out.read_text())
    assert res["levels"] == [1, 2, 3] and sorted(res["per_level"]) == ["1", "2", "3"]
    for g in res["games"]:
        if g["start_level"] != 1:
            assert g["levels_cleared"] <= 1
    for lv in (2, 3):
        gs = [g for g in res["games"] if g["start_level"] == lv]
        assert res["per_level"][str(lv)]["deaths_per_1000_ticks"] == \
            1000 * sum(g["deaths"] for g in gs) / sum(g["length"] for g in gs)
    assert sorted((g["start_level"], g["seed"]) for g in res["games"]) == \
        [(1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]


def test_levels_suite_seed_blocks(tmp_path):
    """--seed-blocks: each level on its own block of seeds (level 1's block
    is the from-level-1 games' seeds)."""
    out = tmp_path / "lv.json"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "evaluate.py"),
                        "--agent", "heuristic", "--levels", str(out), "--episodes", "2",
                        "--level-episodes", "2", "--level-range", "1-3", "--stop-at-clear",
                        "--seed-blocks", "--seed", "10", "--workers", "2", "--max-steps", "3000"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    res = json.loads(out.read_text())
    assert res["seed_blocks"]
    assert sorted((g["start_level"], g["seed"]) for g in res["games"]) == \
        [(1, 10), (1, 11), (2, 12), (2, 13), (3, 14), (3, 15)]
