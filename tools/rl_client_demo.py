#!/usr/bin/env python3
"""Smoke test for the game's RL bridge.

Starts a Unix socket server, launches ./game/pacman --rl <sock> --fast --seed 1,
prints the first 5 states, answers every state with a random action for
10 seconds, reports ticks per second and closes the socket (the game then
exits on its own).

Needs an X display (the game always opens its window); run under
`xvfb-run -a` on a headless machine.
"""

import json
import os
import random
import socket
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAME = os.path.join(ROOT, "game", "pacman")
ACTIONS = "UDLRN"
DURATION = 10.0
SHOW = 5


def show(i, st):
    print(f"--- state {i} ---")
    for row in st["grid"]:
        print("  |" + row + "|")
    rest = {k: v for k, v in st.items() if k != "grid"}
    print("  " + json.dumps(rest))


def main():
    rng = random.Random(0)
    tmp = tempfile.mkdtemp(prefix="pacman-rl-")
    path = os.path.join(tmp, "game.sock")
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(1)
    srv.settimeout(10)

    proc = subprocess.Popen([GAME, "--rl", path, "--fast", "--seed", "1"])
    try:
        conn, _ = srv.accept()
    except socket.timeout:
        proc.kill()
        sys.exit("game did not connect within 10 s")
    rfile = conn.makefile("rb")

    ticks = episodes = 0
    t0 = time.monotonic()
    while time.monotonic() - t0 < DURATION:
        line = rfile.readline()
        if not line:
            print("game closed the connection")
            break
        st = json.loads(line)
        if ticks < SHOW:
            show(ticks, st)
        ticks += 1
        if st["done"]:
            episodes += 1
        conn.sendall((rng.choice(ACTIONS) + "\n").encode())
    elapsed = time.monotonic() - t0

    print(f"--- {ticks} ticks in {elapsed:.1f} s: {ticks / elapsed:.0f} ticks/s, "
          f"{episodes} games over; last score {st['score']}, level {st['level']}")

    rfile.close()
    conn.close()
    srv.close()
    try:
        code = proc.wait(timeout=10)
        print(f"game exited with code {code}")
    except subprocess.TimeoutExpired:
        proc.kill()
        print("game did not exit after disconnect, killed")
    os.unlink(path)
    os.rmdir(tmp)


if __name__ == "__main__":
    main()
