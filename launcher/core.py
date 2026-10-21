"""The app's logic without the UI: paths, settings, the checkpoint and its
download, the checks before a game starts and what a failed game means.
scripts/launcher.py draws the window, launcher/session.py runs the games.
"""

import ctypes
import ctypes.util
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import urllib.request

from . import i18n

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAME = os.path.join(ROOT, "game", "pacman")
REPO = "github.com/AceEca-robo/pacman-1995-rl"

# best_v04 as published in the v0.4 release
CHECKPOINT_URL = f"https://{REPO}/releases/download/v0.4/best_v04.pt"
CHECKPOINT_SHA256 = "ad76723ce7ffb97a53cc915d55fdf3c098bf594c4f269ecd98a3bada372c3cae"
CHECKPOINT_SIZE = 118190300

# configs whose checkpoint is best_v04: K = 32 (the default) and K = 8
PLAN_CONFIGS = {32: "plan_honest_lb.yaml", 8: "plan_honest_k8.yaml"}

# program -> Debian / Ubuntu package (the private Xvfb, finding and placing the
# game's window on it)
TOOLS = {"Xvfb": "xvfb", "xdotool": "xdotool", "xwininfo": "x11-utils"}


def _xdg(var, default):
    return os.path.join(os.environ.get(var) or os.path.expanduser(default), "pacman-rl")


CONFIG_DIR = _xdg("XDG_CONFIG_HOME", "~/.config")
DATA_DIR = _xdg("XDG_DATA_HOME", "~/.local/share")
CACHE_DIR = _xdg("XDG_CACHE_HOME", "~/.cache")
SETTINGS = os.path.join(CONFIG_DIR, "settings.json")
LOG = os.path.join(CACHE_DIR, "last.log")
REPO_CHECKPOINT = os.path.join(ROOT, "runs", "best_v04", "best.pt")
USER_CHECKPOINT = os.path.join(DATA_DIR, "best_v04", "best.pt")


def version():
    try:
        return importlib.metadata.version("pacman-rl")
    except importlib.metadata.PackageNotFoundError:
        with open(os.path.join(ROOT, "pyproject.toml")) as f:
            return re.search(r'^version = "(.+)"', f.read(), re.M).group(1)


# --- settings ---------------------------------------------------------------

DEFAULTS = {"lang": i18n.DEFAULT, "level": 1, "smooth": False, "human_net": False}


def load_settings():
    s = dict(DEFAULTS)
    try:
        with open(SETTINGS) as f:
            s.update(json.load(f))
    except (OSError, ValueError):
        pass
    if s["lang"] not in i18n.LANGS:
        s["lang"] = i18n.DEFAULT
    if not isinstance(s["level"], int) or not 1 <= s["level"] <= 16:
        s["level"] = 1
    s["smooth"], s["human_net"] = bool(s["smooth"]), bool(s["human_net"])
    return s


def save_settings(s):
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        tmp = SETTINGS + ".tmp"
        with open(tmp, "w") as f:
            json.dump({k: s.get(k, v) for k, v in DEFAULTS.items()}, f, indent=1)
        os.replace(tmp, SETTINGS)
    except OSError:
        pass   # not remembering the choice is no reason to fail


# --- checkpoint ---------------------------------------------------------------

def checkpoint():
    """best_v04's best.pt: the repository's runs/ first, then the downloaded one."""
    for p in (REPO_CHECKPOINT, USER_CHECKPOINT):
        if os.path.isfile(p):
            return p
    return None


class Cancelled(Exception):
    pass


def download_checkpoint(progress, cancelled=lambda: False):
    """Download best_v04 over https into USER_CHECKPOINT. progress(done, total)
    is called per chunk, with done = None while the checksum is checked.
    Raises Cancelled, ValueError (checksum mismatch) or OSError."""
    os.makedirs(os.path.dirname(USER_CHECKPOINT), exist_ok=True)
    part = USER_CHECKPOINT + ".part"
    h = hashlib.sha256()
    try:
        req = urllib.request.Request(CHECKPOINT_URL, headers={"User-Agent": "pacman-rl-launcher"})
        with urllib.request.urlopen(req, timeout=30) as r, open(part, "wb") as f:
            total = int(r.headers.get("Content-Length") or CHECKPOINT_SIZE)
            done = 0
            while chunk := r.read(1 << 16):
                if cancelled():
                    raise Cancelled
                f.write(chunk)
                h.update(chunk)
                done += len(chunk)
                progress(done, total)
        progress(None, total)
        if h.hexdigest() != CHECKPOINT_SHA256:
            raise ValueError("sha256 mismatch")
        os.replace(part, USER_CHECKPOINT)
    finally:
        if os.path.exists(part):
            os.remove(part)


def plan_config(k):
    """The planner config for K rollouts with the checkpoint found here: the
    repository's config as is, or a copy pointing at the downloaded one."""
    src = os.path.join(ROOT, "configs", PLAN_CONFIGS[k])
    ck = checkpoint()
    if ck == REPO_CHECKPOINT:
        return src
    import yaml
    with open(src) as f:
        cfg = yaml.safe_load(f)
    cfg["checkpoint"] = ck   # absolute: play.py joins it to the repository root
    dst = os.path.join(DATA_DIR, PLAN_CONFIGS[k])
    with open(dst, "w") as f:
        f.write(f"# written by the launcher from configs/{PLAN_CONFIGS[k]}\n")
        yaml.safe_dump(cfg, f, sort_keys=False)
    return dst


# --- checks -------------------------------------------------------------------

def check(mode, lang):
    """The error message that stops `mode` (play / watch / dqn), or None.
    Every mode runs the game on a private Xvfb and grabs its window."""
    if not os.access(GAME, os.X_OK):
        return i18n.t(lang, "err_no_game")
    missing = [p for p in TOOLS if shutil.which(p) is None]
    if mode == "play" and ctypes.util.find_library("Xtst") is None:
        missing.append("libXtst")
    if missing:
        return i18n.t(lang, "err_no_tool", tools=", ".join(missing),
                      pkgs=" ".join(sorted({TOOLS.get(p, "libxtst6") for p in missing})))
    if mode == "play" and checkpoint() is None:
        return None   # without the network's panels
    for name in ("torch", "yaml", "PIL"):
        if importlib.util.find_spec(name) is None:
            return i18n.t(lang, "err_no_module", name={"yaml": "PyYAML", "PIL": "Pillow"}.get(name, name))
    if mode != "play" and checkpoint() is None:
        return i18n.t(lang, "err_no_checkpoint")
    return None


def explain(error, lang):
    """The menu's message for a session that failed: error is (text,
    traceback); the traceback goes to LOG."""
    text, tb = error
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(LOG, "w") as f:
            f.write(tb)
    except OSError:
        pass
    if "the sim no longer matches the game" in text:
        return i18n.t(lang, "err_desync", log=LOG)
    if "out of memory" in text.lower():
        return i18n.t(lang, "err_gpu_memory")
    if any(s in text for s in ("game did not connect", "Xvfb did not start", "window not found",
                               "game closed the connection")):
        return i18n.t(lang, "err_game_start", log=LOG)
    return i18n.t(lang, "err_crash", error=text.splitlines()[0][:160], log=LOG)
