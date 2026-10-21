#!/usr/bin/env python3
"""The app's menu: a 960x600 window in the style of the network view
(scripts/netviz.py) with three buttons, each starting a child process and
returning here when its window is closed:

    Play yourself              game/pacman --level N (arrows, q quits)
    Watch the neural network   play.py --agent planner --viz (K = 32, or 8 in smooth mode)
    Network without planner    play.py --agent dqn --checkpoint best_v04 on this display

English, Russian or Ukrainian (launcher/i18n.py), the choice, the level and
the smooth switch are kept in ~/.config/pacman-rl/settings.json. Without
runs/best_v04/best.pt the menu offers to download it from the v0.4 release
into ~/.local/share/pacman-rl/. The children's output goes to
~/.cache/pacman-rl/last.log.

    python scripts/launcher.py
    python scripts/launcher.py --lang uk --screenshot docs/launcher.png   # no window
"""

import argparse
import os
import signal
import subprocess
import sys
import threading
import webbrowser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from launcher import core, i18n  # noqa: E402
from launcher.i18n import t  # noqa: E402

W, H = 960, 600
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


class Launcher:
    def __init__(self, lang=None, headless=False):
        os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
        os.environ.setdefault("SDL_VIDEO_X11_WMCLASS", "pacman-rl")   # StartupWMClass
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        import pygame
        self.pg = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption(TITLE)
        icon = os.path.join(ROOT, "launcher", "pacman-rl.svg")
        try:
            pygame.display.set_icon(pygame.image.load(icon))
        except (pygame.error, FileNotFoundError):
            pass
        self.fonts = {}
        self.settings = core.load_settings()
        self.save = not headless
        if lang:
            self.settings["lang"] = lang
        self.version = core.version()
        self.child = self.child_mode = None
        self.stopped = False
        self.message = None   # (kind, text): kind error / ok / info
        self.download = None
        self.offer = core.checkpoint() is None and not headless
        self.cuda = None      # unknown until the check below answers
        if not headless:
            threading.Thread(target=lambda: setattr(self, "cuda", core.cuda_available()),
                             daemon=True).start()
        self.buttons = {}     # name -> rect of the last frame, for clicks
        self.hover = None
        self.quit = False

    @property
    def lang(self):
        return self.settings["lang"]

    def t(self, key, **kw):
        return t(self.lang, key, **kw)

    # --- drawing helpers ----------------------------------------------------

    def font(self, size):
        if size not in self.fonts:
            self.fonts[size] = self.pg.font.Font(None, size)
        return self.fonts[size]

    def text(self, s, pos, size=22, color=FG, anchor="topleft"):
        img = self.font(size).render(s, True, color)
        r = img.get_rect(**{anchor: pos})
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

    def box(self, name, rect, enabled=True, fill=PANEL, radius=10):
        """A clickable rounded rect; returns whether the mouse is over it."""
        pg = self.pg
        rect = pg.Rect(rect)
        hot = enabled and self.hover is not None and rect.collidepoint(self.hover)
        pg.draw.rect(self.screen, PANEL_HI if hot else fill, rect, border_radius=radius)
        if hot:
            pg.draw.rect(self.screen, ACCENT, rect, 2, border_radius=radius)
        if enabled:
            self.buttons[name] = rect
        return hot

    def pacman(self, center, r, mouth=0.25):
        """The icon: a yellow circle with a wedge cut out."""
        import math
        pg = self.pg
        pg.draw.circle(self.screen, ACCENT, center, r)
        a = math.pi * mouth
        pts = [center, (center[0] + 2 * r * math.cos(a), center[1] - 2 * r * math.sin(a)),
               (center[0] + 2 * r * math.cos(a), center[1] + 2 * r * math.sin(a))]
        pg.draw.polygon(self.screen, BG, pts)

    # --- the frame ------------------------------------------------------------

    def draw(self):
        pg = self.pg
        self.buttons = {}
        self.screen.fill(BG)
        busy = self.child is not None
        modal = self.offer or self.download is not None

        # title, subtitle, language switch
        self.pacman((66, 64), 24)
        self.text(TITLE, (104, 64), 60, ACCENT, "midleft")
        self.paragraph(self.t("subtitle"), (40, 104), 24, DIM, 880)
        x = W - 40
        for code in reversed(i18n.LANGS):
            r = pg.Rect(0, 0, 44, 28)
            r.topright = (x, 26)
            on = code == self.lang
            self.box(f"lang:{code}", r, not modal, fill=PANEL, radius=6)
            if on:
                pg.draw.rect(self.screen, ACCENT, r, 2, border_radius=6)
            self.text(i18n.LABELS[code], r.center, 22, ACCENT if on else DIM, "center")
            x = r.left - 6

        # the three buttons
        k = 8 if self.settings["smooth"] else 32
        items = (("play", self.t("play"), self.t("play_hint")),
                 ("watch", self.t("watch"), self.t("watch_hint", k=k)),
                 ("dqn", self.t("dqn"), self.t("dqn_hint")))
        y = 150
        for name, label, hint in items:
            r = pg.Rect(40, y, 540, 88)
            enabled = not busy and not modal
            self.box(name, r, enabled)
            running = busy and self.child_mode == name
            self.text(label, (r.x + 22, r.y + 14), 34,
                      ACCENT if running else (FG if enabled else FAINT))
            self.paragraph(hint, (r.x + 22, r.y + 46), 20, DIM if enabled else FAINT, r.w - 44, 2)
            y = r.bottom + 14

        # right column: level, smooth mode
        px = 610
        pg.draw.rect(self.screen, PANEL, (px, 150, 310, 128), border_radius=10)
        self.text(self.t("level"), (px + 20, 164), 26, DIM)
        lv = self.settings["level"]
        for name, sign, cx in (("level-", "-", px + 60), ("level+", "+", px + 250)):
            r = pg.Rect(0, 0, 44, 44)
            r.center = (cx, 218)
            self.box(name, r, not busy and not modal, fill=BG, radius=22)
            self.text(sign, r.center, 40, FG, "center")
        self.text(str(lv), (px + 155, 220), 64, ACCENT, "center")
        self.paragraph(self.t("level_hint"), (px + 20, 248), 18, FAINT, 270)

        r = pg.Rect(px, 292, 310, 106)
        self.box("smooth", r, not busy and not modal)
        cb = pg.Rect(r.x + 20, r.y + 18, 22, 22)
        pg.draw.rect(self.screen, FG, cb, 2, border_radius=4)
        if self.settings["smooth"]:
            pg.draw.rect(self.screen, ACCENT, cb.inflate(-8, -8), border_radius=2)
        self.text(self.t("smooth"), (cb.right + 12, cb.centery), 26, FG, "midleft")
        self.paragraph(self.t("smooth_hint"), (r.x + 20, r.y + 52), 18, DIM, 270, 2)
        if self.cuda is False:
            self.paragraph(self.t("cpu_note"), (px, 408), 18, FAINT, 310, 2)

        # status: the running child or the last message
        sy = 470
        if busy:
            self.paragraph(self.t("running"), (40, sy), 24, FG, 700)
            r = pg.Rect(0, 0, 160, 40)
            r.topright = (W - 40, sy - 6)
            self.box("stop", r, not modal)
            self.text(self.t("stop"), r.center, 26, FG, "center")
        elif self.message:
            kind, s = self.message
            self.paragraph(s, (40, sy), 22, {"error": RED, "ok": GREEN}.get(kind, FG), 880)

        # footer
        pg.draw.line(self.screen, PANEL_HI, (40, 556), (W - 40, 556))
        link = self.text(core.REPO, (40, 576), 20, LINK, "midleft")
        self.buttons["repo"] = link
        if self.hover and link.collidepoint(self.hover) and not modal:
            pg.draw.line(self.screen, LINK, link.bottomleft, link.bottomright)
        self.text(self.t("version", v=self.version), (W // 2 + 40, 576), 20, FAINT, "center")
        self.text(self.t("license"), (W - 40, 576), 20, FAINT, "midright")

        if modal:
            self.draw_download()

    def draw_download(self):
        pg = self.pg
        shade = pg.Surface((W, H), pg.SRCALPHA)
        shade.fill((0, 0, 0, 170))
        self.screen.blit(shade, (0, 0))
        p = pg.Rect(0, 0, 660, 300)
        p.center = (W // 2, H // 2)
        pg.draw.rect(self.screen, PANEL, p, border_radius=12)
        pg.draw.rect(self.screen, ACCENT, p, 2, border_radius=12)
        self.text(self.t("dl_title"), (p.x + 30, p.y + 26), 36, ACCENT)
        size = round(core.CHECKPOINT_SIZE / 1e6)
        path = core.USER_CHECKPOINT.replace(os.path.expanduser("~"), "~", 1)
        self.paragraph(self.t("dl_text", size=size, path=path), (p.x + 30, p.y + 72), 22, FG, 600)

        d = self.download
        buttons = []
        if d is None or d.result not in (None, "ok"):
            if d is not None and d.result != "cancelled":
                err = (self.t("dl_err_hash") if d.result == "hash"
                       else self.t("dl_err_net", error=d.result))
                self.paragraph(err, (p.x + 30, p.y + 170), 20, RED, 600)
            buttons = [("dl_yes", self.t("dl_yes" if d is None else "dl_retry")),
                       ("dl_no", self.t("dl_no"))]
        elif d.result is None:
            bar = pg.Rect(p.x + 30, p.y + 180, 600, 18)
            pg.draw.rect(self.screen, BG, bar, border_radius=9)
            frac = d.done / d.total if d.total else 0.0
            if frac:
                pg.draw.rect(self.screen, ACCENT, (bar.x, bar.y, max(18, bar.w * frac), bar.h),
                             border_radius=9)
            s = (self.t("dl_verifying") if d.verifying else
                 self.t("dl_progress", done=f"{d.done / 1e6:.0f}", total=f"{d.total / 1e6:.0f}"))
            self.text(s, (bar.x, bar.bottom + 10), 20, DIM)
            buttons = [("dl_cancel", self.t("dl_cancel"))]
        x = p.right - 30
        for name, label in reversed(buttons):
            w = self.font(26).size(label)[0] + 40
            r = pg.Rect(0, 0, w, 42)
            r.bottomright = (x, p.bottom - 22)
            self.box(name, r, fill=BG if name != "dl_yes" else (60, 52, 10))
            self.text(label, r.center, 26, ACCENT if name == "dl_yes" else FG, "center")
            x = r.left - 12

    # --- actions --------------------------------------------------------------

    def click(self, pos):
        hit = next((n for n, r in self.buttons.items() if r.collidepoint(pos)), None)
        if hit is None:
            return
        s = self.settings
        if hit.startswith("lang:"):
            s["lang"] = hit[5:]
            self.message = None   # it was in the old language
        elif hit == "level-":
            s["level"] = max(1, s["level"] - 1)
        elif hit == "level+":
            s["level"] = min(16, s["level"] + 1)
        elif hit == "smooth":
            s["smooth"] = not s["smooth"]
        elif hit == "repo":
            webbrowser.open(f"https://{core.REPO}")
        elif hit == "stop":
            self.stop_child()
        elif hit == "dl_yes":
            self.offer = False
            self.download = Download()
        elif hit == "dl_no":
            self.offer, self.download = False, None
        elif hit == "dl_cancel":
            self.download.cancel = True
        elif hit in ("play", "watch", "dqn"):
            self.start(hit)
        if self.save:
            core.save_settings(s)

    def start(self, mode):
        if mode != "play" and core.checkpoint() is None:
            self.offer = True   # the download dialog says why
            return
        err = core.check(mode, self.lang)
        if err:
            self.message = ("error", err)
            return
        cuda = self.cuda if self.cuda is not None else core.cuda_available()
        try:
            cmd = core.command(mode, self.settings["level"], self.settings["smooth"], cuda)
            os.makedirs(core.CACHE_DIR, exist_ok=True)
            with open(core.LOG, "w") as log:
                log.write(" ".join(cmd) + "\n\n")
                log.flush()
                self.child = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                              stdin=subprocess.DEVNULL,
                                              env={**os.environ, "PYTHONUNBUFFERED": "1"})
        except (OSError, ValueError) as e:
            self.message = ("error", self.t("err_launch", error=e))
            return
        self.child_mode, self.stopped, self.message = mode, False, None

    def stop_child(self):
        if self.child is not None and self.child.poll() is None:
            self.stopped = True
            # play.py ends the game cleanly on SIGINT; the game itself on SIGTERM
            self.child.send_signal(signal.SIGTERM if self.child_mode == "play" else signal.SIGINT)

    def poll(self):
        if self.child is not None and self.child.poll() is not None:
            code = self.child.returncode
            try:
                with open(core.LOG, errors="replace") as f:
                    log = f.read()
            except OSError:
                log = ""
            kind, msg = core.explain(self.child_mode, code, log, self.lang)
            if self.stopped and kind == "error":
                kind, msg = "ok", None
            self.message = (kind, msg) if msg else None
            self.child = self.child_mode = None
        d = self.download
        if d is not None and d.result == "ok":
            self.download = None
            self.message = ("ok", self.t("dl_done"))

    def close(self):
        if self.child is not None:
            self.stop_child()
            try:
                self.child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.child.kill()
        if self.download is not None:
            self.download.cancel = True
        self.pg.quit()

    def run(self):
        pg = self.pg
        clock = pg.time.Clock()
        while not self.quit:
            for e in pg.event.get():
                if e.type == pg.QUIT:
                    self.quit = True
                elif e.type == pg.MOUSEMOTION:
                    self.hover = e.pos
                elif e.type == pg.MOUSEBUTTONDOWN and e.button == 1:
                    self.click(e.pos)
                elif e.type == pg.MOUSEWHEEL and self.child is None and not self.offer:
                    s = self.settings
                    s["level"] = min(16, max(1, s["level"] + (1 if e.y > 0 else -1)))
                    if self.save:
                        core.save_settings(s)
                elif e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE:
                    if self.offer or (self.download and self.download.result is not None):
                        self.offer, self.download = False, None
                    elif self.child is None:
                        self.quit = True
            self.poll()
            self.draw()
            pg.display.flip()
            clock.tick(30)
        self.close()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--lang", choices=i18n.LANGS, help="this language (default: the saved one)")
    p.add_argument("--screenshot", help="render the menu once into this PNG, without a window")
    p.add_argument("--dialog", action="store_true", help="with --screenshot: the download dialog")
    args = p.parse_args()
    try:
        app = Launcher(args.lang, headless=bool(args.screenshot))
    except Exception as e:   # no display at all: say so where it can be seen
        msg = t(core.load_settings()["lang"], "err_no_x11")
        print(f"{msg}\n({e})", file=sys.stderr)
        if not args.screenshot:   # started from the desktop: stderr is not seen
            try:
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
