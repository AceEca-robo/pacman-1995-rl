import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from evaluate import HEADER, RESULTS_INTRO, RULE, update_results  # noqa: E402


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
    assert sorted((g["start_level"], g["seed"]) for g in res["games"]) == \
        [(1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]
