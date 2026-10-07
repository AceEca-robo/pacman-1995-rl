#!/usr/bin/env python3
"""Per-maze numbers for the 16 boards of game/board.h: dots, energizers,
cells pacman can reach, dead ends (reachable cells with one open neighbour),
junctions (>= 3), and the longest shortest path from pacman's start.

    python scripts/maze_stats.py [--json out.json]

Ghost speed does not change with the level: Gamedata's level only picks the
board (game/gamedata.cc setboardlevel, game/pac.cc board->start).
"""

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from env.pacman_env import HEIGHT, LEVELS, PAC_START, WIDTH, _bfs_dist, _passable_neighbours  # noqa: E402


def boards():
    """The raw boards ('O' wall, '_' gate, '.' dot, 'o' energizer)."""
    with open(os.path.join(ROOT, "game", "board.h"), encoding="latin-1") as f:
        rows = re.findall(r'\{"([^"]{%d})"\}' % WIDTH, f.read())
    return [rows[i:i + HEIGHT] for i in range(0, LEVELS * HEIGHT, HEIGHT)]


def stats(board):
    cells = "".join(board).translate(str.maketrans({"O": "#", "_": "-"}))
    nbrs = _passable_neighbours(cells)
    reach = _bfs_dist(nbrs, PAC_START[1] * WIDTH + PAC_START[0])
    return {"dots": cells.count("."), "energizers": cells.count("o"),
            "reachable_cells": len(reach),
            "dead_ends": sum(len(nbrs[c]) == 1 for c in reach),
            "junctions": sum(len(nbrs[c]) >= 3 for c in reach),
            "farthest_from_start": max(reach.values())}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--json")
    args = p.parse_args()
    out = {lv + 1: stats(b) for lv, b in enumerate(boards())}
    keys = list(out[1])
    print("| level | " + " | ".join(keys) + " |")
    print("|---|" + "---|" * len(keys))
    for lv, s in out.items():
        print(f"| {lv} | " + " | ".join(str(s[k]) for k in keys) + " |")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
