#!/usr/bin/env python3
"""The app: one 1920x1080 pygame window (F11: full screen) that starts with
the menu and shows every mode in place:

    Play yourself              the game, the arrows forwarded to it (N: the
                               network's view of your game, it does not steer)
    Watch the neural network   the planner and the network view (K = 32, or 8
                               in smooth mode), as play.py --agent planner --viz
    Network without planner    the network alone, as play.py --agent dqn --viz

Each game runs on a private Xvfb and its window is grabbed into the centre
(launcher/session.py). Esc goes back to the menu from any mode (the game
and its Xvfb are closed), Esc in the menu quits; P pauses with a menu over
the game (continue / back to the menu / quit). English, Russian or
Ukrainian (launcher/i18n.py) on every screen, switched in the menu's or the
pause menu's corner; the language, the level and the switches are kept in
~/.config/pacman-rl/settings.json. Without runs/best_v04/best.pt the menu
offers to download it from the v0.4 release into ~/.local/share/pacman-rl/.
A failed game's traceback goes to ~/.cache/pacman-rl/last.log, the menu
says what happened in plain words. play.py stays the terminal's tool.

    python scripts/launcher.py
    python scripts/launcher.py --lang uk --screenshot docs/launcher.png   # the menu, no window
    PACMAN_RL_DEBUG=1 python scripts/launcher.py   # prints keys, slow frames, the game's display
"""

import argparse
import math
import os
import signal
import sys
import threading
import time
import webbrowser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from launcher import core, i18n  # noqa: E402
from launcher.i18n import t  # noqa: E402

W, H = 1920, 1080
# the menu is laid out on 960 x 600 and drawn K times larger
K = 1.8
OX = round((W - 960 * K) / 2)
# scripts/netviz.py's palette
BG = (14, 16, 22)
PANEL = (24, 27, 36)
PANEL_HI = (34, 38, 52)
FG = (225, 228, 235)
DIM = (130, 136, 150)
FAINT = (100, 105, 118)
ACCENT = (255, 204, 0)
RED = (235, 64, 52)
GREEN = (80, 200, 120)
LINK = (110, 170, 255)
TITLE = "pacman 1.0 (1995)"
ARROWS = {"K_UP": "Up", "K_DOWN": "Down", "K_LEFT": "Left", "K_RIGHT": "Right"}
MODES = ("play", "watch", "dqn")
PAUSE_ITEMS = ("pause_continue", "pause_menu", "pause_quit")


class Download:
    """best_v04's download in a thread; the menu polls its fields."""

    def __init__(self):
        self.done = self.total = 0
        self.verifying = False
        self.result = None   # None while running, then "ok", "cancelled", "hash" or the error
        self.cancel = False
        threading.Thread(target=self.run, daemon=True).start()

    def progress(self, done, total):
        if done is None:
            self.verifying = True
        else:
            self.done, self.total = done, total

    def run(self):
        try:
            core.download_checkpoint(self.progress, lambda: self.cancel)
            self.result = "ok"
        except core.Cancelled:
            self.result = "cancelled"
        except ValueError:
            self.result = "hash"
        except Exception as e:   # network errors come as several types
            self.result = str(getattr(e, "reason", None) or e) or type(e).__name__


class App:
    def __init__(self, lang=None, headless=False):
        os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
        os.environ.setdefault("SDL_VIDEO_X11_WMCLASS", "pacman-rl")   # StartupWMClass
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        import pygame
        self.pg = pygame
        pygame.init()
        flags = 0
        if not headless:
            flags = pygame.SCALED
            sizes = pygame.display.get_desktop_sizes()
            if sizes and (sizes[0][0] < W or sizes[0][1] < H):
                flags |= pygame.FULLSCREEN   # scaled down to the screen
        self.screen = pygame.display.set_mode((W, H), flags)
        pygame.display.set_caption(TITLE)
        try:
            pygame.display.set_icon(pygame.image.load(os.path.join(ROOT, "launcher", "pacman-rl.svg")))
        except (pygame.error, FileNotFoundError):
            pass
        pygame.key.set_repeat()   # held arrows: one KEYDOWN, one KEYUP
        self.fonts = {}
        self.settings = core.load_settings()
        self.save = not headless
        if lang:
            self.settings["lang"] = lang
        self.version = core.version()
        self.message = None      # (kind, text) under the menu's buttons
        self.download = None
        self.offer = core.checkpoint() is None and not headless
        self.cuda = None         # unknown until torch answers (imported in a thread)
        if not headless:
            threading.Thread(target=self._check_cuda, daemon=True).start()
        self.buttons = {}        # name -> screen rect of the last frame
        self.hover = None
        self.quit = False
        self.screen_name = "menu"
        self.session = None
        self.pause_sel = 0
        self.record = {"games": 0, "score": 0, "level": 0}   # your finished games
        self.viz = None
        self.dirty = True
        self.drawn = -1          # the session version on the screen

    def _check_cuda(self):
        try:
            import torch
            self.cuda = torch.cuda.is_available()
        except Exception:
            self.cuda = False
        self.dirty = True

    @property
    def lang(self):
        return self.settings["lang"]

    def t(self, key, **kw):
        return t(self.lang, key, **kw)

    def set(self, key, value):
        self.settings[key] = value
        if self.save:
            core.save_settings(self.settings)
        self.dirty = True

    # --- drawing helpers ----------------------------------------------------

    def font(self, size):
        size = round(size)
        if size not in self.fonts:
            self.fonts[size] = self.pg.font.Font(None, size)
        return self.fonts[size]

    def text(self, s, pos, size=22, color=FG, anchor="topleft"):
        img = self.font(size).render(s, True, color)
        r = img.get_rect(**{anchor: (round(pos[0]), round(pos[1]))})
        self.screen.blit(img, r)
        return r

    def wrap(self, s, size, width):
        f, lines, line = self.font(size), [], ""
        for word in s.split(" "):
            cand = f"{line} {word}".strip()
            if line and f.size(cand)[0] > width:
                lines.append(line)
                line = word
            else:
                line = cand
        return lines + [line]

    def paragraph(self, s, pos, size, color, width, gap=4):
        x, y = pos
        for line in self.wrap(s, size, width):
            r = self.text(line, (x, y), size, color)
            y = r.bottom + gap
        return y

    def box(self, name, rect, enabled=True, fill=PANEL, radius=10, selected=False):
        """A clickable rounded rect; returns whether it is highlighted."""
        pg = self.pg
        rect = pg.Rect(rect)
        hot = enabled and (selected or (self.hover is not None and rect.collidepoint(self.hover)))
        pg.draw.rect(self.screen, PANEL_HI if hot else fill, rect, border_radius=radius)
        if hot:
            pg.draw.rect(self.screen, ACCENT, rect, 3, border_radius=radius)
        if enabled:
            self.buttons[name] = rect
        return hot

    def pacman(self, center, r, mouth=0.25):
        """The icon: a yellow circle with a wedge cut out."""
        pg = self.pg
        pg.draw.circle(self.screen, ACCENT, center, r)
        a = math.pi * mouth
        pg.draw.polygon(self.screen, BG, [
            center, (center[0] + 2 * r * math.cos(a), center[1] - 2 * r * math.sin(a)),
            (center[0] + 2 * r * math.cos(a), center[1] + 2 * r * math.sin(a))])

    # the menu's 960 x 600 layout on the screen
    def R(self, x, y, w, h):
        return self.pg.Rect(round(OX + x * K), round(y * K), round(w * K), round(h * K))

    def P(self, x, y):
        return OX + x * K, y * K

    def lang_switch(self, right, top, enabled=True):
        x = right
        for code in reversed(i18n.LANGS):
            r = self.pg.Rect(0, 0, 80, 50)
            r.topright = (x, top)
            on = code == self.lang
            self.box(f"lang:{code}", r, enabled, radius=10)
            if on:
                self.pg.draw.rect(self.screen, ACCENT, r, 3, border_radius=10)
            self.text(i18n.LABELS[code], r.center, 40, ACCENT if on else DIM, "center")
            x = r.left - 10

    def checkbox(self, name, rect, on, label, hint, enabled=True):
        pg = self.pg
        self.box(name, rect, enabled, fill=BG, radius=8)
        cb = pg.Rect(0, 0, 32, 32)
        cb.midleft = (rect.x + 14, rect.centery)
        pg.draw.rect(self.screen, FG if enabled else FAINT, cb, 3, border_radius=6)
        if on:
            pg.draw.rect(self.screen, ACCENT, cb.inflate(-12, -12), border_radius=3)
        r = self.text(label, (cb.right + 16, rect.centery), 36, FG if enabled else FAINT, "midleft")
        self.text(hint, (r.right + 16, rect.centery + 2), 30, DIM if enabled else FAINT, "midleft")

    def shade(self, alpha=170):
        s = self.pg.Surface((W, H), self.pg.SRCALPHA)
        s.fill((0, 0, 0, alpha))
        self.screen.blit(s, (0, 0))

    # --- the menu -------------------------------------------------------------

    def draw_menu(self):
        pg = self.pg
        self.screen.fill(BG)
        modal = self.offer or self.download is not None
        self.pacman(self.P(66, 64), 24 * K)
        self.text(TITLE, self.P(104, 64), 60 * K, ACCENT, "midleft")
        self.paragraph(self.t("subtitle"), self.P(40, 104), 24 * K, DIM, 880 * K)
        self.lang_switch(round(OX + 920 * K), round(26 * K), not modal)

        k = 8 if self.settings["smooth"] else 32
        items = (("play", self.t("play"), self.t("play_hint")),
                 ("watch", self.t("watch"), self.t("watch_hint", k=k)),
                 ("dqn", self.t("dqn"), self.t("dqn_hint")))
        y = 148
        for name, label, hint in items:
            r = self.R(40, y, 540, 74)
            self.box(name, r, not modal)
            self.text(label, (r.x + 22 * K, r.y + 10 * K), 32 * K, FG)
            self.paragraph(hint, (r.x + 22 * K, r.y + 40 * K), 19 * K, DIM, r.w - 44 * K, 2)
            y += 74 + 4
            if name == "play":
                self.checkbox("human_net", self.R(40, y, 540, 26), self.settings["human_net"],
                              self.t("human_net"), self.t("human_net_short"), not modal)
                y += 26 + 8
            elif name == "watch":
                self.checkbox("smooth", self.R(40, y, 540, 26), self.settings["smooth"],
                              self.t("smooth"), self.t("smooth_short"), not modal)
                y += 26 + 8

        # right column: level, keys
        px = 610
        pg.draw.rect(self.screen, PANEL, self.R(px, 148, 310, 128), border_radius=18)
        self.text(self.t("level"), self.P(px + 20, 160), 26 * K, DIM)
        lv = self.settings["level"]
        for name, sign, cx in (("level-", "-", px + 60), ("level+", "+", px + 250)):
            r = self.R(0, 0, 44, 44)
            r.center = self.P(cx, 214)
            self.box(name, r, not modal, fill=BG, radius=r.w // 2)
            self.text(sign, r.center, 40 * K, FG, "center")
        self.text(str(lv), self.P(px + 155, 216), 64 * K, ACCENT, "center")
        self.paragraph(self.t("level_hint"), self.P(px + 20, 246), 18 * K, FAINT, 270 * K)

        r = self.R(px, 290, 310, 156)
        pg.draw.rect(self.screen, PANEL, r, border_radius=18)
        self.text(self.t("keys_title"), self.P(px + 20, 302), 26 * K, DIM)
        yy = 330 * K
        for line in self.t("keys_list").split("|"):
            yy = self.paragraph(line, (r.x + 20 * K, yy), 18 * K, FG, 270 * K, 2) + 6
        if self.cuda is False:
            self.paragraph(self.t("cpu_note"), self.P(px, 456), 17 * K, FAINT, 310 * K, 2)

        if self.message:
            kind, s = self.message
            self.paragraph(s, self.P(40, 494), 21 * K, {"error": RED, "ok": GREEN}.get(kind, FG),
                           540 * K)

        pg.draw.line(self.screen, PANEL_HI, self.P(40, 556), self.P(920, 556), 2)
        link = self.text(core.REPO, self.P(40, 576), 20 * K, LINK, "midleft")
        if not modal:
            self.buttons["repo"] = link
            if self.hover and link.collidepoint(self.hover):
                pg.draw.line(self.screen, LINK, link.bottomleft, link.bottomright, 2)
        self.text(self.t("version", v=self.version), self.P(520, 576), 20 * K, FAINT, "center")
        self.text(self.t("license"), self.P(920, 576), 20 * K, FAINT, "midright")
        if modal:
            self.draw_download()

    def draw_download(self):
        pg = self.pg
        self.shade()
        p = self.R(150, 150, 660, 300)
        pg.draw.rect(self.screen, PANEL, p, border_radius=20)
        pg.draw.rect(self.screen, ACCENT, p, 3, border_radius=20)
        self.text(self.t("dl_title"), (p.x + 30 * K, p.y + 26 * K), 36 * K, ACCENT)
        size = round(core.CHECKPOINT_SIZE / 1e6)
        path = core.USER_CHECKPOINT.replace(os.path.expanduser("~"), "~", 1)
        self.paragraph(self.t("dl_text", size=size, path=path), (p.x + 30 * K, p.y + 72 * K),
                       22 * K, FG, 600 * K)
        d = self.download
        buttons = []
        if d is None or d.result not in (None, "ok"):
            if d is not None and d.result != "cancelled":
                err = (self.t("dl_err_hash") if d.result == "hash"
                       else self.t("dl_err_net", error=d.result))
                self.paragraph(err, (p.x + 30 * K, p.y + 170 * K), 20 * K, RED, 600 * K)
            buttons = [("dl_yes", self.t("dl_yes" if d is None else "dl_retry")),
                       ("dl_no", self.t("dl_no"))]
        elif d.result is None:
            bar = pg.Rect(p.x + 30 * K, p.y + 180 * K, 600 * K, 18 * K)
            pg.draw.rect(self.screen, BG, bar, border_radius=bar.h // 2)
            frac = d.done / d.total if d.total else 0.0
            if frac:
                pg.draw.rect(self.screen, ACCENT, (bar.x, bar.y, max(bar.h, bar.w * frac), bar.h),
                             border_radius=bar.h // 2)
            s = (self.t("dl_verifying") if d.verifying else
                 self.t("dl_progress", done=f"{d.done / 1e6:.0f}", total=f"{d.total / 1e6:.0f}"))
            self.text(s, (bar.x, bar.bottom + 10 * K), 20 * K, DIM)
            buttons = [("dl_cancel", self.t("dl_cancel"))]
        x = p.right - 30 * K
        for name, label in reversed(buttons):
            w = self.font(26 * K).size(label)[0] + 40 * K
            r = pg.Rect(0, 0, w, 42 * K)
            r.bottomright = (x, p.bottom - 22 * K)
            self.box(name, r, fill=BG if name != "dl_yes" else (60, 52, 10))
            self.text(label, r.center, 26 * K, ACCENT if name == "dl_yes" else FG, "center")
            x = r.left - 12 * K

    # --- a game ----------------------------------------------------------------

    def get_viz(self):
        if self.viz is None:
            from scripts.netviz import Viz
            self.viz = Viz(None, None, screen=self.screen, lang=self.lang)
        self.viz.lang = self.lang
        return self.viz

    def draw_game_screen(self):
        from scripts.netviz import status_line
        s = self.session
        snap = s.snap
        viz = self.get_viz()
        if snap is None:
            self.screen.fill(BG)
            n = int(time.time() * 4) % 2
            self.pacman((W // 2 - 300, H // 2), 40, 0.25 if n else 0.05)
            self.text(self.t("starting"), (W // 2 - 230, H // 2), 60, FG, "midleft")
            self.text(self.t("status_keys"), (W // 2, H - 60), 30, DIM, "center")
            return
        viz.configure(s.obs_cfg)
        obs, info = snap["obs"], snap["info"]
        maze = viz.maze_of(obs["grid"])
        status = status_line(snap["tick"], info["level"], maze, info["lives"], snap["deaths"],
                             info["score"], snap["t_dec"], lang=self.lang)
        if s.mode == "play":
            viz.caption = "viz_caption_human"
            keys = self.t("status_keys_human")
            if snap["pr"] is not None and s.net_view:
                viz.draw(obs, snap["pr"], snap["pr"]["action"], status=status, game=snap["img"],
                         human=snap["human"], bottom=self.draw_controls, keys=keys)
            else:
                self.screen.fill(BG)
                viz.draw_status(status, keys)
                viz.draw_game(obs["grid"], snap["img"], viz.CENTRE)
                self.draw_your_game(viz.LEFT_TOP, info, maze)
                self.draw_how(viz.LEFT_BOTTOM)
                self.draw_net_hint(viz.RIGHT_TOP)
                self.draw_record(viz.RIGHT_BOTTOM)
                self.draw_controls(viz.BOTTOM)
        else:
            viz.caption = "viz_caption_net"
            viz.draw(obs, snap["pr"], snap["played"], snap["trace"], status, snap["img"],
                     keys=self.t("status_keys"))
        if s.over:
            self.draw_game_over(info)

    def panel(self, rect, title):
        x, y, w, h = rect
        self.pg.draw.rect(self.screen, PANEL, rect, border_radius=6)
        self.text(title, (x + 10, y + 6), 26, ACCENT)

    def draw_your_game(self, rect, info, maze):
        x, y, w, h = rect
        self.panel(rect, self.t("game_title"))
        rows = (("lbl_score", info["score"]), ("lbl_lives", info["lives"]),
                ("lbl_level", info["level"]), ("lbl_maze", maze))
        for i, (key, v) in enumerate(rows):
            cx = x + 24 + (i % 2) * (w // 2)
            cy = y + 50 + (i // 2) * 120
            self.text(self.t(key), (cx, cy), 30, DIM)
            self.text(str(v), (cx, cy + 30), 84, ACCENT if key == "lbl_score" else FG)

    def draw_how(self, rect):
        x, y, w, h = rect
        self.panel(rect, self.t("how_title"))
        yy = y + 46
        for part in self.t("how_text").split("|"):
            yy = self.paragraph(part, (x + 20, yy), 28, FG, w - 40, 4) + 14

    def draw_net_hint(self, rect):
        x, y, w, h = rect
        self.panel(rect, self.t("viz_output_human"))
        s = self.t("net_hint") if self.session.has_net else self.t("net_missing")
        self.keycap("N", (x + 20, y + 56), 60)
        self.paragraph(s, (x + 100, y + 60), 30, FG, w - 120, 4)

    def draw_record(self, rect):
        x, y, w, h = rect
        self.panel(rect, self.t("rec_title"))
        rec = self.record
        rows = (("rec_games", rec["games"]), ("rec_best", rec["score"]), ("rec_level", rec["level"]))
        for i, (key, v) in enumerate(rows):
            yy = y + 56 + i * 76
            self.text(self.t(key), (x + 24, yy + 30), 32, DIM, "midleft")
            self.text(str(v), (x + w - 30, yy + 30), 64, ACCENT if key == "rec_best" else FG,
                      "midright")

    def keycap(self, label, pos, size, wide=False, arrow=None):
        pg = self.pg
        r = pg.Rect(pos[0], pos[1], round(size * (1.8 if wide else 1)), size)
        pg.draw.rect(self.screen, (44, 48, 62), r, border_radius=size // 6)
        pg.draw.rect(self.screen, DIM, r, 2, border_radius=size // 6)
        if arrow is not None:
            self.get_viz().arrow(arrow, r.center, size * 0.6, FG)
        else:
            self.text(label, r.center, size * 0.6, FG, "center")
        return r

    def draw_controls(self, rect):
        """The bottom row of your game: the keys, large."""
        x, y, w, h = rect
        self.panel(rect, self.t("controls_title"))
        s = 90
        cx, cy = x + 60, y + 70
        self.keycap("", (cx + s + 10, cy), s, arrow=0)
        for i, d in enumerate((2, 1, 3)):
            self.keycap("", (cx + i * (s + 10), cy + s + 10), s, arrow=d)
        self.paragraph(self.t("ctl_arrows"), (cx, cy + 2 * s + 40), 30, FG, 3 * s + 60)
        col = x + 560
        for i, (cap, key) in enumerate((("N", "ctl_n"), ("P", "ctl_p"), ("Esc", "ctl_esc"),
                                        ("F11", "ctl_f11"))):
            px, py = col + (i % 2) * 660, y + 70 + (i // 2) * 140
            r = self.keycap(cap, (px, py), s, wide=len(cap) > 1)
            self.paragraph(self.t(key), (r.right + 24, py + 26), 34, FG, 640 - r.w - 40)

    def draw_game_over(self, info):
        pg = self.pg
        r = pg.Rect(0, 0, 1000, 170)
        r.center = (W // 2, 320)
        sh = pg.Surface(r.size, pg.SRCALPHA)
        sh.fill((14, 16, 22, 225))
        self.screen.blit(sh, r)
        pg.draw.rect(self.screen, ACCENT, r, 3, border_radius=14)
        self.text(self.t("game_over", score=info["score"], level=info["level"]),
                  (r.centerx, r.y + 58), 56, ACCENT, "center")
        self.text(self.t("game_over_keys"), (r.centerx, r.y + 122), 36, FG, "center")

    def draw_pause(self):
        pg = self.pg
        self.shade(150)
        p = pg.Rect(0, 0, 640, 470)
        p.center = (W // 2, H // 2)
        pg.draw.rect(self.screen, PANEL, p, border_radius=20)
        pg.draw.rect(self.screen, ACCENT, p, 3, border_radius=20)
        self.text(self.t("paused"), (p.x + 40, p.y + 34), 72, ACCENT)
        self.lang_switch(p.right - 30, p.y + 40)
        for i, key in enumerate(PAUSE_ITEMS):
            r = pg.Rect(p.x + 40, p.y + 140 + i * 100, p.w - 80, 80)
            self.box(key, r, selected=i == self.pause_sel, radius=12, fill=BG)
            self.text(self.t(key), r.center, 48, FG, "center")

    # --- actions ----------------------------------------------------------------

    def start(self, mode):
        if mode != "play" and core.checkpoint() is None:
            self.screen_name = "menu"
            self.offer = True   # the download dialog says why
            return
        err = core.check(mode, self.lang)
        if err:
            self.screen_name = "menu"
            self.message = ("error", err)
            return
        from launcher.session import Session
        self.message = None
        self.session = Session(mode, self.settings["level"], self.settings["smooth"],
                               net_view=self.settings["human_net"], after=self.session)
        self.screen_name = "game"
        self.drawn = -1

    def back_to_menu(self):
        if self.session is not None:
            self.session.release_keys()
            self.session.stop()
        self.screen_name = "menu"
        self.dirty = True

    def act(self, name):
        """A pause menu item."""
        if name == "pause_continue":
            self.session.pause(False)
        elif name == "pause_menu":
            self.back_to_menu()
        elif name == "pause_quit":
            self.quit = True

    def click(self, pos):
        hit = next((n for n, r in self.buttons.items() if r.collidepoint(pos)), None)
        if hit is None:
            return
        s = self.settings
        if hit.startswith("lang:"):
            if self.message and self.message[0] == "error":
                self.message = None   # it was in the old language
            self.set("lang", hit[5:])
        elif hit == "level-":
            self.set("level", max(1, s["level"] - 1))
        elif hit == "level+":
            self.set("level", min(16, s["level"] + 1))
        elif hit in ("smooth", "human_net"):
            self.set(hit, not s[hit])
        elif hit == "repo":
            webbrowser.open(f"https://{core.REPO}")
        elif hit == "dl_yes":
            self.offer = False
            self.download = Download()
        elif hit == "dl_no":
            self.offer, self.download = False, None
        elif hit == "dl_cancel":
            self.download.cancel = True
        elif hit in MODES:
            self.start(hit)
        elif hit in PAUSE_ITEMS:
            self.act(hit)
        self.dirty = True

    def key(self, e):
        pg = self.pg
        s = self.session
        self.dirty = True
        if e.key == pg.K_F11:
            pg.display.toggle_fullscreen()
            return
        if self.screen_name == "menu":
            if e.key == pg.K_ESCAPE:
                if self.offer or (self.download and self.download.result is not None):
                    self.offer, self.download = False, None
                elif self.download is None:
                    self.quit = True
            elif self.offer or self.download is not None:
                return
            elif e.key in (pg.K_LEFT, pg.K_MINUS, pg.K_KP_MINUS):
                self.set("level", max(1, self.settings["level"] - 1))
            elif e.key in (pg.K_RIGHT, pg.K_PLUS, pg.K_EQUALS, pg.K_KP_PLUS):
                self.set("level", min(16, self.settings["level"] + 1))
            return
        arrow = next((v for k, v in ARROWS.items() if e.key == getattr(pg, k)), None)
        if e.key == pg.K_ESCAPE:
            self.back_to_menu()
        elif s.over:
            if e.key in (pg.K_RETURN, pg.K_KP_ENTER, pg.K_SPACE):
                self.start(s.mode)
        elif s.paused:
            if e.key == pg.K_p:
                s.pause(False)
            elif e.key in (pg.K_UP, pg.K_DOWN):
                self.pause_sel = (self.pause_sel + (1 if e.key == pg.K_DOWN else -1)) % 3
            elif e.key in (pg.K_RETURN, pg.K_KP_ENTER, pg.K_SPACE):
                self.act(PAUSE_ITEMS[self.pause_sel])
        elif e.key == pg.K_p and s.snap is not None:
            s.pause(True)
            self.pause_sel = 0
        elif e.key == pg.K_n and s.mode == "play" and s.has_net:
            s.net_view = not s.net_view
            self.set("human_net", s.net_view)
        elif arrow and s.mode == "play":
            s.key(arrow, True)

    def key_up(self, e):
        s = self.session
        if self.screen_name == "game" and s is not None and s.mode == "play":
            name = next((v for k, v in ARROWS.items() if e.key == getattr(self.pg, k)), None)
            if name:
                s.key(name, False)

    def poll(self):
        s = self.session
        if s is not None and s.mode == "play" and s.over and not getattr(s, "recorded", False):
            s.recorded = True
            info, rec = s.snap["info"], self.record
            rec["games"] += 1
            rec["score"], rec["level"] = max(rec["score"], info["score"]), max(rec["level"], info["level"])
        if self.screen_name == "game" and s is not None:
            if s.error is not None:
                self.message = ("error", core.explain(s.error, self.lang))
                s.error = None
                self.back_to_menu()
            elif s.version != self.drawn:
                self.dirty = True
            elif s.snap is None:
                self.dirty = True   # the waiting animation
        d = self.download
        if d is not None and d.result == "ok":
            self.download = None
            self.message = ("ok", self.t("dl_done"))
            self.dirty = True
        elif d is not None and d.result is None:
            self.dirty = True

    def draw(self):
        self.buttons = {}
        if self.screen_name == "menu":
            self.draw_menu()
        else:
            self.drawn = self.session.version
            self.draw_game_screen()
            if self.session.paused:
                self.draw_pause()

    def close(self):
        if self.session is not None:
            self.session.release_keys()
            self.session.stop()
            self.session.thread.join(timeout=8)
        if self.download is not None:
            self.download.cancel = True
        self.pg.quit()

    def run(self):
        pg = self.pg
        debug = bool(os.environ.get("PACMAN_RL_DEBUG"))
        while not self.quit:
            # wait at most 15 ms for an event: keys reach the game at once, the
            # screen is redrawn when a tick (or the mouse, or a key) changed it
            events = [pg.event.wait(15)] + pg.event.get()
            t0 = time.perf_counter()
            for e in events:
                if debug and e.type in (pg.KEYDOWN, pg.KEYUP):
                    print(f"{time.perf_counter():.3f} {pg.event.event_name(e.type)} "
                          f"{pg.key.name(e.key)}", flush=True)
                if e.type == pg.QUIT:
                    self.quit = True
                elif e.type == pg.MOUSEMOTION:
                    self.hover = e.pos
                    if self.screen_name == "menu" or (self.session and self.session.paused):
                        self.dirty = True
                elif e.type == pg.MOUSEBUTTONDOWN and e.button == 1:
                    self.click(e.pos)
                elif e.type == pg.MOUSEWHEEL and self.screen_name == "menu" and not self.offer:
                    self.set("level", min(16, max(1, self.settings["level"] + (1 if e.y > 0 else -1))))
                elif e.type == pg.KEYDOWN:
                    self.key(e)
                elif e.type == pg.KEYUP:
                    self.key_up(e)
                elif e.type == pg.WINDOWFOCUSLOST and self.session is not None:
                    self.session.release_keys()   # a key let go elsewhere must not stay held
                elif e.type in (pg.WINDOWEXPOSED, pg.WINDOWSHOWN, pg.WINDOWRESTORED):
                    self.dirty = True
            self.poll()
            if self.dirty:
                self.dirty = False
                self.draw()
                pg.display.flip()
            if debug and time.perf_counter() - t0 > 0.03:
                print(f"{time.perf_counter():.3f} slow frame {(time.perf_counter() - t0) * 1000:.0f} ms",
                      flush=True)
        self.close()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--lang", choices=i18n.LANGS, help="this language (default: the saved one)")
    p.add_argument("--screenshot", help="render the menu once into this PNG, without a window")
    p.add_argument("--dialog", action="store_true", help="with --screenshot: the download dialog")
    args = p.parse_args()
    try:
        app = App(args.lang, headless=bool(args.screenshot))
    except Exception as e:   # no display at all: say so where it can be seen
        msg = t(core.load_settings()["lang"], "err_no_window")
        print(f"{msg}\n({e})", file=sys.stderr)
        if not args.screenshot:   # started from the desktop: stderr is not seen
            try:
                import subprocess
                subprocess.run(["notify-send", "pacman-rl", msg], check=False)
            except OSError:
                pass
        sys.exit(1)
    if args.screenshot:
        app.offer = args.dialog
        app.cuda = True
        app.draw()
        app.pg.image.save(app.screen, args.screenshot)
        app.pg.quit()
        return
    signal.signal(signal.SIGTERM, lambda *_: setattr(app, "quit", True))
    signal.signal(signal.SIGINT, lambda *_: setattr(app, "quit", True))
    app.run()


if __name__ == "__main__":
    main()
