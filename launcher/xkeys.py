"""Hold and release keys on another X display through the XTEST extension
(libXtst, what `xdotool keydown` / `keyup` use), without a process per key.

The game reads the keyboard with XQueryKeymap, the server's key state, once
per tick, so a key has to be held down on the server: events sent to its
window (XSendEvent, `xdotool key --window`) do not change that state and the
game never sees them. On the game's private Xvfb nothing else uses the
keyboard.
"""

import ctypes
import ctypes.util


class XKeys:
    def __init__(self, display):
        x11 = ctypes.CDLL(ctypes.util.find_library("X11") or "libX11.so.6")
        xtst = ctypes.CDLL(ctypes.util.find_library("Xtst") or "libXtst.so.6")
        x11.XOpenDisplay.restype = ctypes.c_void_p
        x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        x11.XStringToKeysym.restype = ctypes.c_ulong
        x11.XStringToKeysym.argtypes = [ctypes.c_char_p]
        x11.XKeysymToKeycode.restype = ctypes.c_ubyte
        x11.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        x11.XFlush.argtypes = [ctypes.c_void_p]
        x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
        x11.XQueryKeymap.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        xtst.XTestFakeKeyEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int,
                                           ctypes.c_ulong]
        self.x11, self.xtst = x11, xtst
        self.dpy = x11.XOpenDisplay(display.encode())
        if not self.dpy:
            raise RuntimeError(f"cannot open display {display}")
        self.codes, self.held = {}, set()

    def code(self, name):
        """name: an X keysym name, e.g. "Up", "Left", "q"."""
        if name not in self.codes:
            sym = self.x11.XStringToKeysym(name.encode())
            self.codes[name] = self.x11.XKeysymToKeycode(self.dpy, sym)
        return self.codes[name]

    def press(self, name, down=True):
        if self.dpy is None:
            return
        self.xtst.XTestFakeKeyEvent(self.dpy, self.code(name), int(down), 0)
        self.x11.XFlush(self.dpy)
        (self.held.add if down else self.held.discard)(name)

    def release_all(self):
        for name in list(self.held):
            self.press(name, False)

    def is_down(self, name):
        """The server's state of the key (what the game's XQueryKeymap sees)."""
        buf = ctypes.create_string_buffer(32)
        self.x11.XQueryKeymap(self.dpy, buf)
        c = self.code(name)
        return bool(buf.raw[c // 8] >> (c % 8) & 1)

    def close(self):
        if self.dpy is not None:
            self.release_all()
            self.x11.XCloseDisplay(self.dpy)
            self.dpy = None
