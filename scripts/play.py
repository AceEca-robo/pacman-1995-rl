#!/usr/bin/env python3
"""Watch an agent play in the game's own window, in real time.

    python scripts/play.py --agent heuristic --seed 0 --display :0
    python scripts/play.py --agent dqn --checkpoint runs/dqn0/best.pt --seed 0 \\
        --display :0 --record docs/dqn0.gif --max-steps 400

The game runs without --fast (0.25 s per tick, half while pacman is super).
With --record, the game window is grabbed (xwd) once per tick, after the
game has drawn it, and written as a GIF with the measured tick durations.
Needs xdotool and xwd. The window must stay visible: X has no backing store
here, a covered part of the window is grabbed as whatever covers it (the
game raises itself when obscured).
"""

import argparse
import os
import struct
import subprocess
import sys
import time

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agents.heuristic_agent import HeuristicAgent  # noqa: E402
from env.pacman_env import PacmanEnv  # noqa: E402

WINDOW_NAME = r"^Pacman v\. 1\.0"  # PACTITLE in game/pac.h


def find_window(display, timeout=10.0):
    env = {**os.environ, "DISPLAY": display}
    deadline = time.time() + timeout
    while time.time() < deadline:
        out = subprocess.run(["xdotool", "search", "--name", WINDOW_NAME],
                             env=env, capture_output=True, text=True).stdout.split()
        if out:
            return out[-1]
        time.sleep(0.1)
    sys.exit(f"game window not found on {display}")


def grab(display, window):
    """Window contents as an RGB PIL image, via xwd (32 bits per pixel only)."""
    data = subprocess.run(["xwd", "-silent", "-id", window],
                          env={**os.environ, "DISPLAY": display},
                          capture_output=True, check=True).stdout
    h = struct.unpack(">25I", data[:100])  # XWDFileHeader, always big-endian
    header_size, width, height, byte_order = h[0], h[4], h[5], h[7]
    bpp, bytes_per_line, ncolors = h[11], h[12], h[19]
    if bpp != 32:
        sys.exit(f"xwd: {bpp} bits per pixel not supported, need a 24/32-bit display")
    start = header_size + 12 * ncolors  # XWDColor entries
    raw = data[start:start + bytes_per_line * height]
    mode = "BGRX" if byte_order == 0 else "XRGB"  # LSBFirst / MSBFirst
    return Image.frombuffer("RGB", (width, height), raw, "raw", mode, bytes_per_line, 1)


def make_agent(args):
    if args.agent == "heuristic":
        return HeuristicAgent()
    if not args.checkpoint:
        sys.exit("--agent dqn needs --checkpoint")
    from agents.dqn import DQNAgent
    return DQNAgent(args.checkpoint, epsilon=0.0, seed=args.seed)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--agent", choices=("heuristic", "dqn"), required=True)
    p.add_argument("--checkpoint", help="dqn only: best.pt or checkpoint.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--display", default=os.environ.get("DISPLAY", ":0"))
    p.add_argument("--record", help="write a GIF of the window here, e.g. docs/heuristic.gif")
    p.add_argument("--max-steps", type=int, default=None,
                   help="stop after this many ticks (a full heuristic game is ~1250, 5 min)")
    p.add_argument("--scale", type=float, default=1.0, help="GIF frame scale")
    args = p.parse_args()

    agent = make_agent(args)
    env = PacmanEnv(display=args.display, fast=False, config={"max_episode_steps": None})
    frames, durations = [], []
    try:
        obs, info = env.reset(seed=args.seed)
        agent.reset(seed=args.seed)
        window = find_window(args.display) if args.record else None
        total, steps, t_prev = 0.0, 0, time.perf_counter()
        while True:
            if window:
                # the game drew this tick before sending the state we just got
                img = grab(args.display, window)
                if args.scale != 1.0:
                    img = img.resize((round(img.width * args.scale), round(img.height * args.scale)),
                                     Image.NEAREST)
                frames.append(img.quantize(colors=64, method=Image.Quantize.FASTOCTREE))
                now = time.perf_counter()
                durations.append(now - t_prev)
                t_prev = now
            obs, r, term, trunc, info = env.step(agent.act(obs))
            total += r
            steps += 1
            if term or trunc or (args.max_steps and steps >= args.max_steps):
                break
        print(f"{steps} steps, reward {total:.1f}, score {info['score']}, level {info['level']}, "
              f"lives {info['lives']}{' (game over)' if term else ''}")
    finally:
        env.close()

    if args.record and frames:
        # frame i is shown until frame i+1 was grabbed; GIF delays are in 10 ms units
        delays = [max(20, round(d * 100) * 10) for d in durations[1:]] + [1000]
        os.makedirs(os.path.dirname(os.path.abspath(args.record)), exist_ok=True)
        frames[0].save(args.record, save_all=True, append_images=frames[1:], duration=delays,
                       loop=0, optimize=True, disposal=1)
        print(f"wrote {args.record}: {len(frames)} frames, "
              f"{os.path.getsize(args.record) / 1e6:.1f} MB, "
              f"median tick {np.median(durations[1:]) * 1000:.0f} ms")


if __name__ == "__main__":
    main()
