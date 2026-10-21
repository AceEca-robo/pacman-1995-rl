#!/usr/bin/env bash
# Install pacman-rl as a desktop app for the current user, from this checkout:
# checks the system packages, builds the game (game/) and the sim (sim/),
# creates .venv with the package, and puts the app (scripts/launcher.py: one
# window with the menu, your game and the network's) into the applications
# menu (~/.local/share/applications/pacman-rl.desktop, icon,
# ~/.local/bin/pacman-rl). Nothing needs root; missing packages are listed
# with the apt command that installs them. Safe to run again.
#
#   ./install.sh          # torch as pip resolves it (with CUDA on Linux)
#   ./install.sh --cpu    # torch's CPU-only build: a much smaller download
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
BIN="$HOME/.local/bin"
CPU=0
for a in "$@"; do
    case "$a" in
        --cpu) CPU=1 ;;
        -h|--help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $a (see --help)" >&2; exit 2 ;;
    esac
done
step() { printf '\n== %s\n' "$*"; }

step "checking system packages"
missing=()
need_cmd() { command -v "$1" >/dev/null 2>&1 || missing+=("$2"); }
need_file() { [ -e "$1" ] || missing+=("$2"); }
need_cmd gcc build-essential
need_cmd g++ build-essential
need_cmd make build-essential
need_cmd xmkmf xutils-dev
need_file /usr/include/X11/Xlib.h libx11-dev
need_file /usr/include/X11/extensions/Xext.h libxext-dev
need_file /usr/include/curses.h libncurses-dev
need_cmd Xvfb xvfb          # the app runs every game on a private Xvfb
need_cmd xdotool xdotool    # finds the game's window there
need_cmd xwininfo x11-utils # and its position
need_cmd xwd x11-apps       # play.py --record
ldconfig -p 2>/dev/null | grep -q 'libXtst\.so\.6' || missing+=(libxtst6)   # your keys to the game (XTest)
if ! command -v python3 >/dev/null 2>&1; then
    missing+=(python3 python3-venv)
elif ! python3 -c 'import sys; assert sys.version_info >= (3, 10)' 2>/dev/null; then
    echo "Python 3.10 or newer is needed, python3 is $(python3 -V 2>&1)" >&2; exit 1
elif ! python3 -c 'import ensurepip, venv' 2>/dev/null; then
    missing+=(python3-venv)
fi
if [ ${#missing[@]} -gt 0 ]; then
    mapfile -t missing < <(printf '%s\n' "${missing[@]}" | sort -u)
    echo "Missing: ${missing[*]}"
    echo "Install them and run ./install.sh again:"
    echo "  sudo apt install ${missing[*]}"
    exit 1
fi
echo "all there"

step "building the game (game/)"
log="$(mktemp)"
if ! (cd "$ROOT/game" && xmkmf && make) >"$log" 2>&1; then
    tail -n 30 "$log"; echo "the build failed, full log: $log" >&2; exit 1
fi
rm -f "$log"
echo "game/pacman"

step "Python environment (.venv)"
[ -x "$ROOT/.venv/bin/python" ] || python3 -m venv "$ROOT/.venv"
PY="$ROOT/.venv/bin/python"
"$PY" -m pip install --quiet --upgrade pip
if [ "$CPU" = 1 ]; then
    torch="$(sed -n 's/^ *"\(torch==[^"]*\)".*/\1/p' "$ROOT/pyproject.toml")"
    "$PY" -m pip install --quiet "$torch" --index-url https://download.pytorch.org/whl/cpu
fi
"$PY" -m pip install --quiet -e "$ROOT[viz]"
(cd "$ROOT" && "$PY" -c 'from sim.csim import CSim; CSim(0, 1)')   # builds sim/_csim.so
echo "$("$PY" -c 'import torch; print("torch", torch.__version__, "CUDA" if torch.cuda.is_available() else "CPU only")')"

step "desktop entry"
mkdir -p "$DATA/applications" "$DATA/icons/hicolor/scalable/apps" "$BIN"
install -m 644 "$ROOT/launcher/pacman-rl.svg" "$DATA/icons/hicolor/scalable/apps/pacman-rl.svg"
cat >"$BIN/pacman-rl" <<SH
#!/bin/sh
# pacman-rl launcher, written by $ROOT/install.sh
exec "$PY" "$ROOT/scripts/launcher.py" "\$@"
SH
chmod 755 "$BIN/pacman-rl"
(cd "$ROOT" && "$PY" -m launcher.desktop "$BIN/pacman-rl") >"$DATA/applications/pacman-rl.desktop"
command -v desktop-file-validate >/dev/null && desktop-file-validate "$DATA/applications/pacman-rl.desktop"
command -v update-desktop-database >/dev/null && update-desktop-database "$DATA/applications" 2>/dev/null || true
command -v gtk-update-icon-cache >/dev/null && \
    gtk-update-icon-cache -q -t "$DATA/icons/hicolor" 2>/dev/null || true
echo "$DATA/applications/pacman-rl.desktop -> $BIN/pacman-rl"

if [ -f "$ROOT/runs/best_v04/best.pt" ] || [ -f "$DATA/pacman-rl/best_v04/best.pt" ]; then
    ck="the network's weights are in place"
else
    ck="the launcher offers to download the network's weights (118 MB) on its first start"
fi
printf '\nDone. Start "Pacman 1995 RL" from the applications menu or run pacman-rl;\n%s.\n' "$ck"
