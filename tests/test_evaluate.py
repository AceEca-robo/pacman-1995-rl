import os
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
