#!/usr/bin/env bash
# Remove what install.sh put outside this checkout: the desktop entry, the
# icon and ~/.local/bin/pacman-rl. The downloaded weights, the settings and
# the log stay unless --purge is given. .venv and the build stay in the
# checkout (git clean -xdf removes them).
#
#   ./uninstall.sh [--purge]
set -euo pipefail

DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
PURGE=0
for a in "$@"; do
    case "$a" in
        --purge) PURGE=1 ;;
        -h|--help) sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $a (see --help)" >&2; exit 2 ;;
    esac
done

rm -fv "$DATA/applications/pacman-rl.desktop" \
       "$DATA/icons/hicolor/scalable/apps/pacman-rl.svg" \
       "$HOME/.local/bin/pacman-rl"
command -v update-desktop-database >/dev/null && update-desktop-database "$DATA/applications" 2>/dev/null || true
command -v gtk-update-icon-cache >/dev/null && \
    gtk-update-icon-cache -q -t "$DATA/icons/hicolor" 2>/dev/null || true
if [ "$PURGE" = 1 ]; then
    rm -rfv "$DATA/pacman-rl" "${XDG_CONFIG_HOME:-$HOME/.config}/pacman-rl" \
            "${XDG_CACHE_HOME:-$HOME/.cache}/pacman-rl"
fi
echo "pacman-rl removed from the applications menu"
