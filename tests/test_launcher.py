import importlib
import os
import string
import subprocess
import sys

import pytest

from launcher import i18n

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def core(tmp_path, monkeypatch):
    """launcher.core with its XDG directories in tmp_path."""
    for var in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME"):
        monkeypatch.setenv(var, str(tmp_path / var))
    import launcher.core
    return importlib.reload(launcher.core)


def fields(s):
    return {f for _, f, _, _ in string.Formatter().parse(s) if f}


def test_every_string_in_every_language():
    for key, entry in i18n.STRINGS.items():
        assert set(entry) == set(i18n.LANGS), key
        kinds = {callable(v) for v in entry.values()}
        assert len(kinds) == 1, key
        if not callable(entry["en"]):
            # the same placeholders everywhere
            assert len({frozenset(fields(v)) for v in entry.values()}) == 1, key


def test_plural():
    words = ("вариант", "варианта", "вариантов")
    assert [i18n.plural("ru", n, words) for n in (1, 2, 5, 11, 12, 21, 22, 25, 32, 111)] == \
        ["вариант", "варианта", "вариантов", "вариантов", "вариантов", "вариант", "варианта",
         "вариантов", "варианта", "вариантов"]
    assert "8 варіантів" in i18n.t("uk", "watch_hint", k=8)
    assert "32 варіанти" in i18n.t("uk", "watch_hint", k=32)
    assert "32 варианта" in i18n.t("ru", "watch_hint", k=32)
    assert i18n.t("xx", "play") == i18n.t("en", "play")


def test_settings_roundtrip(core):
    assert core.load_settings() == core.DEFAULTS
    core.save_settings({"lang": "uk", "level": 7, "smooth": True, "human_net": True})
    assert core.load_settings() == {"lang": "uk", "level": 7, "smooth": True, "human_net": True}
    core.save_settings({"lang": "ru"})   # the rest: the defaults
    assert core.load_settings() == {**core.DEFAULTS, "lang": "ru"}
    with open(core.SETTINGS, "w") as f:
        f.write('{"lang": "de", "level": 40}')
    assert core.load_settings() == core.DEFAULTS
    with open(core.SETTINGS, "w") as f:
        f.write("not json")
    assert core.load_settings() == core.DEFAULTS


def test_check(core, monkeypatch):
    monkeypatch.setattr(core, "GAME", "/nonexistent/pacman")
    assert core.check("play", "ru") == i18n.t("ru", "err_no_game")
    monkeypatch.setattr(core, "GAME", sys.executable)
    monkeypatch.setattr(core.shutil, "which", lambda p: None if p == "xwininfo" else "/usr/bin/" + p)
    assert "sudo apt install x11-utils" in core.check("watch", "en")
    monkeypatch.setattr(core.shutil, "which", lambda p: "/usr/bin/" + p)
    monkeypatch.setattr(core, "checkpoint", lambda: None)
    assert core.check("play", "en") is None   # your game without the network's panels
    assert core.check("dqn", "uk") == i18n.t("uk", "err_no_checkpoint")


def test_plan_config_points_at_the_download(core, monkeypatch):
    import yaml
    monkeypatch.setattr(core, "checkpoint", lambda: core.USER_CHECKPOINT)
    os.makedirs(core.DATA_DIR)
    for k in (8, 32):
        with open(core.plan_config(k)) as f:
            cfg = yaml.safe_load(f)
        with open(os.path.join(ROOT, "configs", core.PLAN_CONFIGS[k])) as f:
            ref = yaml.safe_load(f)
        assert cfg["checkpoint"] == core.USER_CHECKPOINT
        assert cfg["plan"] == ref["plan"] and cfg["plan"]["rollouts"] == k


def test_explain(core):
    msg = core.explain(("SystemExit: tick 9: the sim no longer matches the game", "Traceback x"), "uk")
    assert msg == i18n.t("uk", "err_desync", log=core.LOG)
    assert open(core.LOG).read() == "Traceback x"
    msg = core.explain(("KeyError: 'x'\nmore", "Traceback y"), "en")
    assert "KeyError: 'x'" in msg and "more" not in msg and "Traceback" not in msg
    assert core.explain(("RuntimeError: game did not connect, exit code 1", ""), "ru") == \
        i18n.t("ru", "err_game_start", log=core.LOG)


def test_status_line_languages():
    from scripts.netviz import status_line
    assert status_line(3, 2, 2, 3, 0, 50, 0.0123) == \
        "tick 3   level 2 \u2192 maze 2   lives 3   deaths 0   score 50   12 ms / decision"
    assert status_line(3, 2, 2, 3, 0, 50, lang="ru").startswith("ход 3   уровень 2 \u2192 лабиринт 2")
    assert "мс на рішення" in status_line(3, 2, 2, 3, 0, 50, 0.01, lang="uk")


def test_download_rejects_a_bad_checksum(core, monkeypatch, tmp_path):
    src = tmp_path / "fake.pt"
    src.write_bytes(b"not the checkpoint")
    monkeypatch.setattr(core, "CHECKPOINT_URL", src.as_uri())
    seen = []
    with pytest.raises(ValueError):
        core.download_checkpoint(lambda d, t: seen.append(d))
    assert not os.path.exists(core.USER_CHECKPOINT)
    assert not os.path.exists(core.USER_CHECKPOINT + ".part")
    assert seen[0] == len(b"not the checkpoint") and seen[-1] is None

    import hashlib
    monkeypatch.setattr(core, "CHECKPOINT_SHA256", hashlib.sha256(src.read_bytes()).hexdigest())
    core.download_checkpoint(lambda d, t: None)
    assert open(core.USER_CHECKPOINT, "rb").read() == src.read_bytes()


def test_desktop_entry():
    from launcher.desktop import entry
    e = entry("/opt/x y/pacman-rl")
    assert 'Exec="/opt/x y/pacman-rl"' in e
    for lang in ("ru", "uk"):
        assert f"Name[{lang}]={i18n.t(lang, 'desktop_name')}" in e
        assert f"Comment[{lang}]=" in e


@pytest.mark.parametrize("lang", i18n.LANGS)
def test_screenshot(tmp_path, lang):
    pytest.importorskip("pygame")
    out = tmp_path / "menu.png"
    env = {**os.environ, "XDG_CONFIG_HOME": str(tmp_path)}
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "launcher.py"),
                    "--lang", lang, "--screenshot", str(out)], check=True, env=env, timeout=60)
    assert out.stat().st_size > 10000
    assert not (tmp_path / "pacman-rl").exists()   # a screenshot does not save settings
