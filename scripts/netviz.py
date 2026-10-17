#!/usr/bin/env python3
"""What the network "thinks": a pygame window next to the game (play.py --viz).

Panels, all of the state the decision was made on:
  1. input: walls + food, ghosts by state, the cell-safety plane (a network
     input), safe exits (computed from it with env.pacman_env's
     safe_exits_field; best_v04 does not get this plane);
  2. activations: the 16 most active maps of each conv layer (forward
     hooks on the ReLUs), each tile scaled to its own max;
  3. saliency: |d Q(chosen action) / d input| summed over the channels, on a
     copy of the maze;
  4. output: Q of U / D / L / R / N (N: keep going), the played action in
     colour, greedy argmax marked; V and the advantages A - mean(A) of the
     dueling heads (Q = V + A - mean A);
  5. planner: the K rollouts of every direction on a copy of the maze (red:
     lost a life), dropped directions crossed out, death share per
     direction;
  6. status line: tick, level, lives, score, ms per decision.

    python scripts/netviz.py --bench --frames 300     # sim only, no game: decisions/s with viz
    python scripts/netviz.py --snapshot docs/viz_frame.png --level 9 --ticks 120
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from env.pacman_env import CH, HEIGHT, WIDTH, _passable_neighbours, safe_exits_field  # noqa: E402

W, H = 1280, 800
BG = (14, 16, 22)
PANEL = (24, 27, 36)
FG = (225, 228, 235)
DIM = (130, 136, 150)
ACCENT = (255, 204, 0)
RED = (235, 64, 52)
GREEN = (80, 200, 120)
WALL = (40, 70, 170)
DIR_NAMES = ("U", "D", "L", "R", "N")
DIR_COLORS = ((90, 170, 255), (255, 150, 60), (190, 110, 255), (60, 210, 200), (170, 170, 170))
GHOST_COLORS = {"normal": (235, 64, 52), "hunted": (70, 110, 255), "eyes": (240, 240, 240)}


def lut(name):
    from matplotlib import colormaps
    return (colormaps[name](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)


class NetProbe:
    """One forward + backward pass of the Q network on the decision's state:
    conv activations (hooks), V, A, Q and the saliency of an action."""

    def __init__(self, net, device):
        import torch
        self.torch, self.net, self.device = torch, net, device
        self.acts, self.heads, self.on = {}, {}, False
        relus = [m for m in net.conv if isinstance(m, torch.nn.ReLU)]
        self.layer_names = [f"conv{i + 1}" for i in range(len(relus))]
        for name, m in zip(self.layer_names, relus):
            m.register_forward_hook(self._hook(self.acts, name))
        net.value.register_forward_hook(self._hook(self.heads, "V"))
        net.adv.register_forward_hook(self._hook(self.heads, "A"))

    def _hook(self, store, name):
        def f(_m, _i, out):
            if self.on:
                store[name] = out
        return f

    def __call__(self, grid, vec, action=None):
        """grid (C, 23, 33), vec; action None: the greedy one. Returns a dict
        of numpy arrays."""
        torch = self.torch
        g = torch.from_numpy(grid[None]).to(self.device).requires_grad_(True)
        v = torch.from_numpy(vec[None]).to(self.device)
        self.on = True
        try:
            with torch.enable_grad():
                q = self.net(g, v)
                a = int(q.argmax(1)) if action is None else int(action)
                q[0, a].backward()
        finally:
            self.on = False
        A = self.heads["A"][0].detach()
        return {"q": q[0].detach().cpu().numpy(), "V": float(self.heads["V"][0, 0].detach()),
                "A": (A - A.mean()).cpu().numpy(), "action": a,
                "saliency": g.grad[0].abs().sum(0).cpu().numpy(),
                "acts": {k: x[0].detach().cpu().numpy() for k, x in self.acts.items()}}


class Viz:
    def __init__(self, net, device, obs_cfg=None, pos=None, headless=False):
        if pos is not None:
            os.environ["SDL_VIDEO_WINDOW_POS"] = f"{pos[0]},{pos[1]}"
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame
        self.pg = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("pacman: network view")
        self.font = {s: pygame.font.Font(None, s) for s in (18, 22, 26, 30, 36)}
        self.probe = NetProbe(net, device)
        self.heat, self.vir = lut("inferno"), lut("viridis")
        self.nbrs = {}
        self.closed = False
        obs_cfg = obs_cfg or {}
        # the cell-safety plane comes after CHANNELS and the food distance plane
        self.safety_ch = (CH["bonus"] + 1 + bool(obs_cfg.get("food_distance"))
                          if obs_cfg.get("cell_safety") else None)

    def safety(self, grid):
        if self.safety_ch is None:
            return np.zeros((HEIGHT, WIDTH), np.float32)
        return grid[self.safety_ch]

    # --- drawing helpers ------------------------------------------------------

    def text(self, s, pos, size=22, color=FG, anchor="topleft"):
        if size not in self.font:
            self.font[size] = self.pg.font.Font(None, size)
        img = self.font[size].render(s, True, color)
        r = img.get_rect(**{anchor: pos})
        self.screen.blit(img, r)
        return r

    def panel(self, rect, title):
        pg = self.pg
        pg.draw.rect(self.screen, PANEL, rect, border_radius=6)
        self.text(title, (rect[0] + 10, rect[1] + 6), 26, ACCENT)

    def blit_array(self, rgb, pos, size):
        """rgb (h, w, 3) uint8 -> scaled (nearest) to size at pos."""
        surf = self.pg.surfarray.make_surface(np.ascontiguousarray(rgb.transpose(1, 0, 2)))
        self.screen.blit(self.pg.transform.scale(surf, size), pos)

    def maze_rgb(self, grid, dim=1.0):
        img = np.zeros((HEIGHT, WIDTH, 3), np.float32) + np.array(BG, np.float32)
        img[grid[CH["walls"]] > 0] = WALL
        img[grid[CH["gate"]] > 0] = (200, 120, 160)
        return (img * dim).astype(np.uint8)

    def draw_maze(self, grid, rect, dim=1.0, heat=None):
        """A copy of the maze in rect (x, y, w, h); heat (23, 33) in 0..1 is
        blended over it. Returns the cell size and origin."""
        pg = self.pg
        x0, y0, w, h = rect
        cs = min(w / WIDTH, h / HEIGHT)
        ox, oy = x0 + (w - cs * WIDTH) / 2, y0 + (h - cs * HEIGHT) / 2
        img = self.maze_rgb(grid, dim).astype(np.float32)
        if heat is not None:
            hc = self.heat[(np.clip(heat, 0, 1) * 255).astype(int)].astype(np.float32)
            alpha = np.clip(heat, 0, 1)[..., None] ** 0.7 * 0.9
            img = img * (1 - alpha) + hc * alpha
        self.blit_array(img.astype(np.uint8), (ox, oy), (round(cs * WIDTH), round(cs * HEIGHT)))
        c = lambda y, x: (int(ox + (x + 0.5) * cs), int(oy + (y + 0.5) * cs))  # noqa: E731
        for y, x in zip(*np.nonzero(grid[CH["food"]])):
            pg.draw.circle(self.screen, (230, 210, 170), c(y, x), max(1, cs / 9))
        for y, x in zip(*np.nonzero(grid[CH["superfood"]])):
            pg.draw.circle(self.screen, (255, 230, 190), c(y, x), max(2, cs / 3.5))
        for st in ("normal", "hunted", "eyes"):
            for y, x in zip(*np.nonzero(grid[CH["ghost_" + st]])):
                pg.draw.circle(self.screen, GHOST_COLORS[st], c(y, x), cs * 0.42)
                if st == "eyes":
                    pg.draw.circle(self.screen, (20, 20, 120), c(y, x), cs * 0.18)
        for y, x in zip(*np.nonzero(grid[CH["pacman"]])):
            pg.draw.circle(self.screen, ACCENT, c(y, x), cs * 0.45)
        return cs, ox, oy

    # --- panels -----------------------------------------------------------------

    def safe_exits(self, grid):
        walls = grid[CH["walls"]].reshape(-1) > 0
        gate = grid[CH["gate"]].reshape(-1) > 0
        layout = "".join("#" if w else "-" if g else " " for w, g in zip(walls, gate))
        nb = self.nbrs.get(layout)
        if nb is None:
            nb = self.nbrs[layout] = _passable_neighbours(layout)
        pac = np.flatnonzero(grid[CH["pacman"]].reshape(-1))
        if not len(pac) or self.safety_ch is None:
            return np.zeros((HEIGHT, WIDTH), np.float32)
        safe = self.safety(grid).reshape(-1)
        return safe_exits_field(nb, safe, int(pac[0])).reshape(HEIGHT, WIDTH)

    def draw_inputs(self, grid, rect):
        x0, y0, w, h = rect
        self.panel(rect, "1  INPUT")
        tw, th = (w - 30) // 2, (h - 40) // 2
        tiles = []
        wf = self.maze_rgb(grid)
        wf[grid[CH["food"]] > 0] = (230, 210, 170)
        wf[grid[CH["superfood"]] > 0] = (255, 255, 255)
        wf[grid[CH["pacman"]] > 0] = ACCENT
        tiles.append(("walls + food", wf))
        gh = self.maze_rgb(grid, 0.6)
        for st in ("normal", "hunted", "eyes"):
            gh[grid[CH["ghost_" + st]] > 0] = GHOST_COLORS[st]
        gh[grid[CH["pacman"]] > 0] = ACCENT
        tiles.append(("ghosts: normal / hunted / eyes", gh))
        safe = self.safety(grid)
        sf = self.maze_rgb(grid, 0.6)
        sf[safe > 0] = GREEN
        sf[grid[CH["pacman"]] > 0] = ACCENT
        tiles.append(("cell safety (input)", sf))
        ex = self.safe_exits(grid)
        se = self.maze_rgb(grid, 0.6)
        se[ex > 0] = self.vir[(ex[ex > 0] * 255).astype(int)]
        se[(grid[CH["pacman"]] > 0) & (ex == 0)] = ACCENT
        tiles.append(("safe exits (derived)", se))
        for i, (name, img) in enumerate(tiles):
            tx, ty = x0 + 10 + (i % 2) * (tw + 10), y0 + 36 + (i // 2) * (th + 2)
            self.text(name, (tx, ty), 18, DIM)
            cs = min(tw / WIDTH, (th - 18) / HEIGHT)
            self.blit_array(img, (tx, ty + 17), (int(cs * WIDTH), int(cs * HEIGHT)))

    def draw_acts(self, acts, rect):
        x0, y0, w, h = rect
        self.panel(rect, "2  ACTIVATIONS (top 16 maps per conv layer)")
        rows = len(self.probe.layer_names)
        rh = (h - 40) // rows
        for r, name in enumerate(self.probe.layer_names):
            a = acts[name]
            top = np.argsort(-a.mean((1, 2)))[:16]
            ty = y0 + 36 + r * rh
            self.text(f"{name}  {a.shape[0]}x{a.shape[1]}x{a.shape[2]}", (x0 + 10, ty), 18, DIM)
            tw = (w - 20 - 7 * 4) // 8
            th = min(int(tw * a.shape[1] / a.shape[2]), (rh - 22) // 2 - 3)
            for i, ch in enumerate(top):
                m = a[ch] / (a[ch].max() + 1e-6)
                img = self.vir[(m * 255).astype(int)]
                self.blit_array(img, (x0 + 10 + (i % 8) * (tw + 4), ty + 18 + (i // 8) * (th + 3)),
                                (tw, th))

    def draw_saliency(self, grid, sal, action, rect):
        self.panel(rect, f"3  SALIENCY  |dQ({DIR_NAMES[action]})/d input|")
        s = sal / (np.percentile(sal, 99.5) + 1e-9)
        self.draw_maze(grid, (rect[0] + 10, rect[1] + 36, rect[2] - 20, rect[3] - 44), 0.7, s)

    def draw_q(self, pr, played, rect):
        x0, y0, w, h = rect
        self.panel(rect, "4  OUTPUT  Q per action")
        q, A = pr["q"], pr["A"]
        greedy = int(np.argmax(q))
        lo, hi = min(q.min(), 0), max(q.max(), 0)
        span = hi - lo + 1e-6
        bx, bw, gap = x0 + 18, 40, 12
        top, bot = y0 + 70, y0 + h - 70
        zero = float(bot - (0 - lo) / span * (bot - top))
        for i in range(5):
            x = bx + i * (bw + gap)
            yv = float(bot - (q[i] - lo) / span * (bot - top))
            col = DIR_COLORS[i] if i == played else (70, 76, 92)
            self.pg.draw.rect(self.screen, col, (x, min(yv, zero), bw, max(2, abs(zero - yv))),
                              border_radius=3)
            self.text(f"{q[i]:.1f}", (x + bw / 2, min(yv, zero) - 4), 22,
                      FG if i == played else DIM, "midbottom")
            self.text(DIR_NAMES[i], (x + bw / 2, bot + 6), 30, FG if i == played else DIM, "midtop")
            if i == greedy:
                self.text("greedy", (x + bw / 2, bot + 32), 18, ACCENT, "midtop")
        self.pg.draw.line(self.screen, DIM, (bx - 6, zero), (bx + 5 * (bw + gap) - gap + 6, zero), 1)
        vx = bx + 5 * (bw + gap) + 10
        self.text("dueling heads", (vx, top - 20), 20, DIM)
        self.text(f"V = {pr['V']:.1f}", (vx, top + 4), 30, FG)
        for i in range(5):
            self.text(f"A {DIR_NAMES[i]}  {A[i]:+.2f}", (vx, top + 40 + i * 26), 22,
                      DIR_COLORS[i] if i == played else DIM)
        self.text("Q = V + A - mean(A)", (vx, top + 40 + 5 * 26 + 6), 18, DIM)

    def draw_planner(self, grid, tr, played, rect):
        x0, y0, w, h = rect
        if tr is None:
            self.panel(rect, "5  PLANNER  (off: --agent dqn)")
            self.draw_maze(grid, (x0 + 10, y0 + 36, w - 20, h - 44), 0.8)
            return
        K, Hh = tr["cells"].shape[1], tr["cells"].shape[2]
        self.panel(rect, f"5  PLANNER  K={K} rollouts x H={Hh} steps per direction")
        lw = 150
        cs, ox, oy = self.draw_maze(grid, (x0 + 10, y0 + 36, w - lw - 30, h - 44), 0.55)
        pg = self.pg
        pac = np.flatnonzero(grid[CH["pacman"]].reshape(-1))
        start = int(pac[0]) if len(pac) else None
        pt = lambda c: (ox + (c % WIDTH + 0.5) * cs, oy + (c // WIDTH + 0.5) * cs)  # noqa: E731
        # the rollouts that lost a life last, on top of the others
        order = sorted(((r, k) for r in range(4) for k in range(K)),
                       key=lambda rk: bool(tr["dead"][rk]))
        for r, k in order:
            col = DIR_COLORS[int(tr["act"][r])]
            cells = tr["cells"][r, k]
            path = ([start] if start is not None else []) + [int(c) for c in cells if c >= 0]
            # the K rollouts of a direction often share cells: spread them
            # a little so that every one of them stays visible
            j = (k - (K - 1) / 2) / max(K, 2) * 0.7
            lox, loy = ox + j * cs, oy + j * cs
            pts = []
            for c in path:
                if pts and abs(c % WIDTH - pts[-1][0]) + abs(c // WIDTH - pts[-1][1]) > 1:
                    self._line(pts, RED if tr["dead"][r, k] else col, cs, lox, loy)
                    pts = []
                if not pts or pts[-1] != (c % WIDTH, c // WIDTH):
                    pts.append((c % WIDTH, c // WIDTH))
            self._line(pts, RED if tr["dead"][r, k] else col, cs, lox, loy)
            if tr["dead"][r, k] and path:
                x, y = pt(path[-1])
                pg.draw.circle(self.screen, RED, (x + j * cs, y + j * cs), max(3, cs * 0.3), 2)
        if start is not None:
            pg.draw.circle(self.screen, ACCENT, pt(start), cs * 0.5)
            pg.draw.circle(self.screen, BG, pt(start), cs * 0.5, 2)
        # legend: direction, death share, dropped
        lx, ly = x0 + w - lw - 10, y0 + 50
        self.text("death share", (lx, ly), 20, DIM)
        self.text(f"drop if > {tr['threshold']:.2f}", (lx, ly + 20), 18, DIM)
        for r in range(4):
            a = int(tr["act"][r])
            name = DIR_NAMES[a] if a < 4 else "N"
            f = tr["frac"][r]
            dropped = f > tr["threshold"]
            yy = ly + 56 + r * 44
            col = DIR_COLORS[a]
            rr = self.text(f"{name}  {f:4.0%}", (lx, yy), 30, col if not dropped else DIM)
            if dropped:
                pg.draw.line(self.screen, RED, (rr.left - 2, rr.centery), (rr.right + 2, rr.centery), 3)
            tag = "played" if a == played else ("" if tr["needed"][r] else "shown only")
            if tag:
                self.text(tag, (lx + 4, yy + 26), 16, ACCENT if a == played else DIM)
        self.text("red: lost a life", (lx, ly + 56 + 4 * 44 + 6), 18, RED)

    def _line(self, pts, col, cs, ox, oy):
        if len(pts) >= 2:
            self.pg.draw.lines(self.screen, col, False,
                               [(ox + (x + 0.5) * cs, oy + (y + 0.5) * cs) for x, y in pts], 2)

    def draw_status(self, s):
        self.pg.draw.rect(self.screen, PANEL, (0, 0, W, 40))
        self.text(s, (14, 20), 30, FG, "midleft")

    # --- frame ------------------------------------------------------------------

    def update(self, obs, played, trace=None, status=""):
        """Draw one tick: obs the decision's observation, played the action,
        trace the planner's stats["trace"] for this game (or None)."""
        pg = self.pg
        for e in pg.event.get():
            if e.type == pg.QUIT:
                self.closed = True
        grid, vec = obs["grid"], obs["vec"]
        pr = self.probe(grid, vec, played)
        self.screen.fill(BG)
        self.draw_status(status)
        g = 8
        self.draw_inputs(grid, (g, 48, 410, 320))
        self.draw_acts(pr["acts"], (g, 376, 410, H - 376 - g))
        self.draw_saliency(grid, pr["saliency"], played, (426, 48, 416, 320))
        self.draw_planner(grid, trace, played, (426, 376, 846, H - 376 - g))
        self.draw_q(pr, played, (850, 48, W - 850 - g, 320))
        pg.display.flip()
        return pr

    def frame(self):
        """The window's pixels, (H, W, 3) uint8."""
        return self.pg.surfarray.array3d(self.screen).transpose(1, 0, 2).copy()

    def save(self, path):
        self.pg.image.save(self.screen, path)


def status_line(tick, level, lives, score, t_decision, t_viz=None):
    s = f"tick {tick}    level {level}    lives {lives}    score {score}    " \
        f"{t_decision * 1000:.0f} ms / decision"
    if t_viz is not None:
        s += f"    (+{t_viz * 1000:.0f} ms drawing)"
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plan-config", default=os.path.join(ROOT, "configs", "plan_honest_lb.yaml"))
    p.add_argument("--level", type=int, default=9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ticks", type=int, default=300, help="decisions to play in the sim")
    p.add_argument("--snapshot", help="save the last frame here (PNG)")
    p.add_argument("--device", default=None)
    args = p.parse_args()
    from scripts.play import PlannerAgent
    agent = PlannerAgent(args.plan_config, args.device, trace=True)
    agent.reset(args.seed, args.level)
    viz = Viz(agent.net.net, agent.net.device, agent.env_overrides.get("obs"), headless=True)
    t_dec, t_viz = [], []
    for t in range(args.ticks):
        obs = agent.sim.obs(agent.net.cell_safety)
        t0 = time.perf_counter()
        a = agent.act(obs)
        t1 = time.perf_counter()
        i = agent.prev
        viz.update(obs, a, agent.last_trace, status_line(t, i[1], i[0], i[3], t1 - t0,
                                                         t_viz[-1] if t_viz else None))
        t_viz.append(time.perf_counter() - t1)
        t_dec.append(t1 - t0)
        if i[0] == 0:
            break
    tot = np.array(t_dec) + np.array(t_viz)
    print(f"{len(t_dec)} ticks: decision {np.mean(t_dec) * 1000:.0f} ms, viz {np.mean(t_viz) * 1000:.0f} ms,"
          f" total {np.mean(tot) * 1000:.0f} ms (p95 {np.percentile(tot, 95) * 1000:.0f}) ->"
          f" {1 / np.mean(tot):.1f} decisions/s")
    if args.snapshot:
        viz.save(args.snapshot)
        print("saved", args.snapshot)


if __name__ == "__main__":
    main()
