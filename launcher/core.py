"""The launcher's logic without the UI: paths, settings, the checkpoint and
its download, the checks before a launch, the commands it runs and what
their logs mean. scripts/launcher.py draws the window.
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
import sys
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

# program -> Debian / Ubuntu package
TOOLS = {"Xvfb": "xvfb", "xdotool": "xdotool", "xwd": "x11-apps"}


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

DEFAULTS = {"lang": i18n.DEFAULT, "level": 1, "smooth": False}


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
    s["smooth"] = bool(s["smooth"])
    return s


def save_settings(s):
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        tmp = SETTINGS + ".tmp"
        with open(tmp, "w") as f:
            json.dump({k: s[k] for k in DEFAULTS}, f, indent=1)
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

def x11_ok():
    """An X server answers on $DISPLAY (the game is an X11 program)."""
    if not os.environ.get("DISPLAY"):
        return False
    lib = ctypes.util.find_library("X11")
    if not lib:
        return False
    x = ctypes.CDLL(lib)
    x.XOpenDisplay.restype = ctypes.c_void_p
    x.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x.XCloseDisplay.argtypes = [ctypes.c_void_p]
    d = x.XOpenDisplay(None)
    if not d:
        return False
    x.XCloseDisplay(d)
    return True


def cuda_available():
    """Asked in a child process: importing torch here would stall the menu."""
    import subprocess
    try:
        out = subprocess.run([sys.executable, "-c",
                              "import torch; print(int(torch.cuda.is_available()))"],
                             capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return False
    return out == "1"


def check(mode, lang):
    """The error message that stops `mode` (play / watch / dqn), or None."""
    if not x11_ok():
        return i18n.t(lang, "err_no_x11")
    if not os.access(GAME, os.X_OK):
        return i18n.t(lang, "err_no_game")
    if mode == "play":
        return None
    for name in ("torch", "yaml", "pygame", "PIL"):
        if importlib.util.find_spec(name) is None:
            return i18n.t(lang, "err_no_module", name={"yaml": "PyYAML", "PIL": "Pillow"}.get(name, name))
    if checkpoint() is None:
        return i18n.t(lang, "err_no_checkpoint")
    if mode == "watch":
        missing = [p for p in TOOLS if shutil.which(p) is None]
        if missing:
            return i18n.t(lang, "err_no_tool", tools=", ".join(missing),
                          pkgs=" ".join(sorted({TOOLS[p] for p in missing})))
    return None


# --- commands -----------------------------------------------------------------

def command(mode, level, smooth=False, cuda=True, seed=None):
    """The command line for a menu button."""
    if mode == "play":
        return [GAME, "--level", str(level)]
    seed = int.from_bytes(os.urandom(2), "little") if seed is None else seed
    play = [sys.executable, os.path.join(ROOT, "scripts", "play.py"),
            "--level", str(level), "--seed", str(seed)]
    if mode == "watch":
        cmd = play + ["--agent", "planner", "--viz", "--plan-config", plan_config(8 if smooth else 32)]
        return cmd + ([] if cuda else ["--device", "cpu"])
    if mode == "dqn":
        return play + ["--agent", "dqn", "--checkpoint", checkpoint(),
                       "--display", os.environ.get("DISPLAY", ":0")]
    raise ValueError(mode)


def explain(mode, code, log_text, lang):
    """What a finished child means for the menu: (kind, message), kind
    'ok' (message may be None) or 'error'."""
    if mode == "play":
        # closing the window kills the X connection: the game exits with 1,
        # so only a game that never opened its window is an error
        if code == 0 or "XIO" in log_text or "fatal IO error" in log_text:
            return "ok", None
    m = re.findall(r"score (\d+), level (\d+)", log_text)
    if code == 0:
        return "ok", (i18n.t(lang, "last_game", score=m[-1][0], level=m[-1][1]) if m else None)
    if "the sim no longer matches the game" in log_text:
        return "error", i18n.t(lang, "err_desync", log=LOG)
    if "CUDA out of memory" in log_text or "OutOfMemoryError" in log_text:
        return "error", i18n.t(lang, "err_gpu_memory")
    if "game did not connect" in log_text or "cannot open display" in log_text.lower():
        return "error", i18n.t(lang, "err_game_start", log=LOG)
    if mode == "play" and code != 0:
        return "error", i18n.t(lang, "err_game_start", log=LOG)
    return "error", i18n.t(lang, "err_crash", code=code, log=LOG)
