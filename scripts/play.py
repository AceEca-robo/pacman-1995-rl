#!/usr/bin/env python3
"""Watch an agent play in the game's own window, in real time.

    python scripts/play.py --agent heuristic --seed 0 --display :0
    python scripts/play.py --agent dqn --checkpoint runs/dqn0/best.pt --seed 0 \\
        --display :0 --record docs/dqn0.gif --max-steps 400
    xvfb-run -a -s "-screen 0 1280x1024x24" python scripts/play.py --agent dqn \
        --checkpoint runs/dqfd0/best.pt --level 7 --record docs/x.gif --max-steps 400
    python scripts/play.py --agent planner --seed 0 --display :0   # best_v04 + rollouts
    python scripts/play.py --agent planner --level 9 --viz --record docs/viz.mp4 --max-steps 400

--agent planner: scripts/plan_eval.py's planner (--plan-config, default
the honest variant with the loop breaker, configs/plan_honest_lb.yaml)
plays the game: a sim (sim/) follows the game in lock-step and the
planner's rollouts run in clones of it; the observation of the game is
checked against the sim's every tick.

The game runs without --fast (0.25 s per tick, half while pacman is super).
With --record, the game window is grabbed (xwd) once per tick, after the
game has drawn it, and written as a GIF with the measured tick durations.
Needs xdotool and xwd. The window must stay visible: X has no backing store
here, a covered part of the window is grabbed as whatever covers it (the
game raises itself when obscured).

--viz (dqn and planner): a second window (scripts/netviz.py, pygame) next to
the game shows the network's input planes, conv activations, saliency, Q /
V / A and the planner's rollouts, redrawn every tick for the state the
decision was made on. With --record both windows go side by side into the
file; a .mp4 is written by ffmpeg as 1920x1080 at 25 fps without sound
(the two windows centred, scaled down only if they do not fit; frames
repeated to keep the tick durations), anything else is a GIF. At the end
it prints the share of decisions that took longer than the game's tick
(0.25 s, 0.125 s while pacman is super): the game waits for the action, so
a late decision stretches that tick.
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
    """Window contents as an RGB PIL image, via xwd (24 or 32 bits per pixel)."""
    data = subprocess.run(["xwd", "-silent", "-id", window],
                          env={**os.environ, "DISPLAY": display},
                          capture_output=True, check=True).stdout
    h = struct.unpack(">25I", data[:100])  # XWDFileHeader, always big-endian
    header_size, width, height, byte_order = h[0], h[4], h[5], h[7]
    bpp, bytes_per_line, ncolors = h[11], h[12], h[19]
    if bpp not in (24, 32):
        sys.exit(f"xwd: {bpp} bits per pixel not supported, need a 24/32-bit display")
    start = header_size + 12 * ncolors  # XWDColor entries
    raw = data[start:start + bytes_per_line * height]
    # LSBFirst / MSBFirst; xwd gives 24 bpp once a pygame window (--viz) is open
    if bpp == 32:
        mode = "BGRX" if byte_order == 0 else "XRGB"
    else:
        mode = "BGR" if byte_order == 0 else "RGB"
    return Image.frombuffer("RGB", (width, height), raw, "raw", mode, bytes_per_line, 1)


def window_geometry(display, window):
    """(x, y, width, height) of an X window, via xdotool."""
    out = subprocess.run(["xdotool", "getwindowgeometry", "--shell", window],
                         env={**os.environ, "DISPLAY": display}, capture_output=True,
                         text=True).stdout
    g = dict(line.split("=", 1) for line in out.split() if "=" in line)
    return int(g["X"]), int(g["Y"]), int(g["WIDTH"]), int(g["HEIGHT"])


def screen_size(display):
    out = subprocess.run(["xdotool", "getdisplaygeometry"], env={**os.environ, "DISPLAY": display},
                         capture_output=True, text=True).stdout.split()
    return (int(out[0]), int(out[1])) if len(out) == 2 else (10 ** 5, 10 ** 5)


class Mp4Writer:
    """RGB frames with their durations -> H.264 mp4 at a fixed rate and frame
    size (ffmpeg), no audio track."""

    def __init__(self, path, size, fps=25, out=(1920, 1080)):
        self.fps, self.size, self.n, self.t, self.shown = fps, size, 0, 0.0, 0
        # centred on the background colour, scaled down only if larger than out
        s = min(1.0, out[0] / size[0], out[1] / size[1])
        w, h = 2 * round(size[0] * s / 2), 2 * round(size[1] * s / 2)
        vf = (f"scale={w}:{h}:flags=lanczos," if s < 1 else "") + \
            f"pad={out[0]}:{out[1]}:({out[0]}-iw)/2:({out[1]}-ih)/2:color=0x0e1016"
        self.proc = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-", "-vf", vf, "-an",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", path],
            stdin=subprocess.PIPE)

    def add(self, img, duration):
        """img (h, w, 3) uint8 shown for duration seconds."""
        self.t += duration
        k = round(self.t * self.fps) - self.shown
        data = np.ascontiguousarray(img).tobytes()
        for _ in range(max(k, 0)):
            self.proc.stdin.write(data)
        self.shown += max(k, 0)
        self.n += 1

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


def side_by_side(game, viz):
    """Game window (PIL) left, viz frame (array) right, on one dark canvas
    with even sides (H.264)."""
    g = np.asarray(game.convert("RGB"))
    h = max(g.shape[0], viz.shape[0])
    w = g.shape[1] + viz.shape[1]
    out = np.zeros((h + h % 2, w + w % 2, 3), np.uint8)
    out[:] = (14, 16, 22)
    out[:g.shape[0], :g.shape[1]] = g
    out[:viz.shape[0], g.shape[1]:g.shape[1] + viz.shape[1]] = viz
    return out


def make_agent(args):
    if args.agent == "heuristic":
        return HeuristicAgent()
    if not args.checkpoint:
        sys.exit("--agent dqn needs --checkpoint")
    from agents.dqn import DQNAgent
    return DQNAgent(args.checkpoint, epsilon=0.0, seed=args.seed)


class PlannerAgent:
    """scripts/plan_eval.py's planner on one game: a sim mirrors the game."""

    def __init__(self, config, device=None, trace=False):
        from scripts.plan_eval import Planner, QNet, Stall, load_cfg
        from sim.csim import CSim
        self.cfg = load_cfg(config)
        self.net = QNet(os.path.join(ROOT, self.cfg["checkpoint"]), device or self.cfg["run"]["device"])
        self.planner = Planner(self.net, self.cfg)
        self.env_overrides = self.net.env_overrides
        self.CSim, self.Stall = CSim, Stall
        self.trace, self.last_trace = trace, None   # trace: rollouts for --viz

    def reset(self, seed, level=1):
        self.sim = self.CSim(seed, level)
        self.seed, self.level, self.t = seed, level, 0
        self.stall = self.Stall(self.cfg)
        self.prev = self.sim.info()

    def act(self, obs):
        o = self.sim.obs(self.net.cell_safety)
        if not (np.array_equal(o["grid"], obs["grid"]) and np.array_equal(o["vec"], obs["vec"])):
            sys.exit(f"tick {self.t}: the sim no longer matches the game")
        q = self.net(o["grid"][None], o["vec"][None])
        a, st = self.planner.choose(self.sim.buf[None], self.sim.info()[None], q,
                                    [[self.seed, self.level, self.t]], [self.stall.visits()],
                                    trace=self.trace)
        if self.trace:
            self.last_trace = {k: (v[0] if isinstance(v, np.ndarray) else v)
                               for k, v in st["trace"].items()}
        a = int(a[0])
        self.sim.advance(a)
        info = self.sim.info()
        self.stall.update(int(info[2]), info[1] != self.prev[1] or info[4] < self.prev[4])
        self.prev, self.t = info, self.t + 1
        return a


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--agent", choices=("heuristic", "dqn", "planner"), required=True)
    p.add_argument("--checkpoint", help="dqn only: best.pt or checkpoint.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--display", default=os.environ.get("DISPLAY", ":0"))
    p.add_argument("--record", help="write a GIF of the window here, e.g. docs/heuristic.gif")
    p.add_argument("--max-steps", type=int, default=None,
                   help="stop after this many ticks (a full heuristic game is ~1250, 5 min)")
    p.add_argument("--scale", type=float, default=1.0, help="GIF frame scale")
    p.add_argument("--level", type=int, default=None, help="start on this level (maze), 1-16")
    p.add_argument("--plan-config", default=os.path.join(ROOT, "configs", "plan_honest_lb.yaml"),
                   help="planner only: a scripts/plan_eval.py config")
    p.add_argument("--device", help="planner only: torch device (default: the config's)")
    p.add_argument("--viz", action="store_true",
                   help="dqn / planner: the network view in a second window (scripts/netviz.py)")
    args = p.parse_args()
    if args.viz and args.agent == "heuristic":
        sys.exit("--viz needs a network: --agent dqn or planner")

    agent = (PlannerAgent(args.plan_config, args.device, trace=args.viz)
             if args.agent == "planner" else make_agent(args))
    env = PacmanEnv(display=args.display, fast=False,
                    config={"max_episode_steps": None, **agent.env_overrides})
    if args.viz:
        return play_viz(args, agent, env)
    frames, durations = [], []
    try:
        obs, info = env.reset(seed=args.seed, options=(
            {"start_level": args.level} if args.level else None))
        if args.agent == "planner":
            agent.reset(args.seed, args.level or 1)
        else:
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


def play_viz(args, agent, env):
    """main() with --viz: the game, the network view next to it, and (with
    --record) both side by side in one file."""
    from scripts.netviz import H as VIZ_H, W as VIZ_W, Viz, status_line
    net, device = (agent.net.net, agent.net.device) if args.agent == "planner" else \
        (agent.net, agent.device)
    mp4 = bool(args.record) and args.record.lower().endswith(".mp4")
    writer, frames, durations, t_dec, t_viz = None, [], [], [], []
    t_agent, ticks, supers, deaths = [], [], [], 0   # per tick: our time, wall time, super
    os.environ["DISPLAY"] = args.display
    try:
        obs, info = env.reset(seed=args.seed, options=(
            {"start_level": args.level} if args.level else None))
        if args.agent == "planner":
            agent.reset(args.seed, args.level or 1)
        else:
            agent.reset(seed=args.seed)
        window = find_window(args.display)
        x, y, w, _ = window_geometry(args.display, window)
        # right of the game window, but on the screen
        sw, sh = screen_size(args.display)
        vx = min(x + w + 8, max(0, sw - VIZ_W))
        viz = Viz(net, device, agent.env_overrides.get("obs"), pos=(vx, max(0, min(y, sh - VIZ_H))))
        total, steps, t_prev, prev_img = 0.0, 0, time.perf_counter(), None
        t_obs = time.perf_counter()
        while True:
            game_img = grab(args.display, window) if args.record else None
            t0 = time.perf_counter()
            a = agent.act(obs)
            t1 = time.perf_counter()
            viz.update(obs, a, getattr(agent, "last_trace", None),
                       status_line(steps, info["level"], viz.maze_of(obs["grid"]), info["lives"],
                                   deaths, info["score"], t1 - t0, t_viz[-1] if t_viz else None))
            t_dec.append(t1 - t0)
            t_viz.append(time.perf_counter() - t1)
            if args.record:
                img = side_by_side(game_img, viz.frame())
                now = time.perf_counter()
                if prev_img is not None:
                    if mp4:
                        if writer is None:
                            os.makedirs(os.path.dirname(os.path.abspath(args.record)), exist_ok=True)
                            writer = Mp4Writer(args.record, (prev_img.shape[1], prev_img.shape[0]))
                        writer.add(prev_img, now - t_prev)
                    else:
                        durations.append(now - t_prev)
                if not mp4:
                    pil = Image.fromarray(img)
                    if args.scale != 1.0:
                        pil = pil.resize((round(pil.width * args.scale),
                                          round(pil.height * args.scale)), Image.NEAREST)
                    frames.append(pil.quantize(colors=128, method=Image.Quantize.FASTOCTREE))
                prev_img, t_prev = img, now
            lives = info["lives"]
            t_agent.append(time.perf_counter() - t_obs)
            obs, r, term, trunc, info = env.step(a)
            now = time.perf_counter()
            ticks.append(now - t_obs)
            t_obs = now
            # the game slept for 0.25 / (1 + super) before sending this state
            supers.append(bool(obs["vec"][1] > 0))
            deaths += int(info["lives"] < lives)
            total += r
            steps += 1
            if term or trunc or viz.closed or (args.max_steps and steps >= args.max_steps):
                break
        tot = np.array(t_dec) + np.array(t_viz)
        print(f"{steps} steps, reward {total:.1f}, score {info['score']}, level {info['level']}, "
              f"lives {info['lives']}{' (game over)' if term else ''}")
        print(f"decision {np.mean(t_dec) * 1000:.0f} ms + viz {np.mean(t_viz) * 1000:.0f} ms per "
              f"tick (p95 of the sum {np.percentile(tot, 95) * 1000:.0f} ms): "
              f"{1 / np.mean(tot):.1f} decisions/s possible")
        t_agent, ticks, supers = np.array(t_agent), np.array(ticks), np.array(supers)
        for name, m, tick in (("normal", ~supers, 0.25), ("super", supers, 0.125)):
            if m.any():
                print(f"{name} ticks ({tick * 1000:.0f} ms): {m.sum()}, decision + viz"
                      f"{' + grab' if args.record else ''} longer than the tick: "
                      f"{np.mean(t_agent[m] > tick):.1%} (max {t_agent[m].max() * 1000:.0f} ms), "
                      f"ticks stretched by > 10 ms: {np.mean(ticks[m] > tick + 0.01):.1%}, "
                      f"wall time median {np.median(ticks[m]) * 1000:.0f} ms, "
                      f"max {ticks[m].max() * 1000:.0f} ms")
    finally:
        env.close()
    if writer is not None:
        writer.add(prev_img, 1.0)
        writer.close()
        print(f"wrote {args.record}: {writer.n} ticks, {writer.shown} frames at {writer.fps} fps, "
              f"{os.path.getsize(args.record) / 1e6:.1f} MB")
    elif args.record and frames:
        delays = [max(20, round(d * 100) * 10) for d in durations] + [1000]
        os.makedirs(os.path.dirname(os.path.abspath(args.record)), exist_ok=True)
        frames[0].save(args.record, save_all=True, append_images=frames[1:], duration=delays,
                       loop=0, optimize=True, disposal=1)
        print(f"wrote {args.record}: {len(frames)} frames, "
              f"{os.path.getsize(args.record) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
