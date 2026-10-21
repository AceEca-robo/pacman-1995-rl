"""scripts/play.py: --agent random, the baseline it records."""


def test_random_agent():
    from scripts.play import RandomAgent
    a, b = RandomAgent(3), RandomAgent(3)
    assert [a.act(None) for _ in range(50)] == [b.act(None) for _ in range(50)]
    assert set(a.act(None) for _ in range(200)) == set(range(5))
