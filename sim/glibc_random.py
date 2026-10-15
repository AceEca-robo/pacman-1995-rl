"""Exact copy of glibc random()/srandom() (the default TYPE_3 generator).

r[0] = seed (0 -> 1), r[i] = 16807 * r[i-1] mod (2^31 - 1) for i = 1..30
(Schrage's method, as glibc), r[i] = r[i-31] for i = 31..33, then
r[i] = r[i-3] + r[i-31] mod 2^32; output k is r[k + 344] >> 1.
"""

DEG, SEP = 31, 3
MASK = 0xFFFFFFFF


class GlibcRandom:
    __slots__ = ("r", "i")

    def __init__(self, seed=1):
        self.seed(seed)

    def seed(self, seed):
        seed &= MASK
        if seed == 0:
            seed = 1
        # glibc stores the seed in an int32_t
        word = seed - (1 << 32) if seed >= 1 << 31 else seed
        r = [word]
        for _ in range(1, DEG):
            hi = int(word / 127773)            # C division truncates
            lo = word - hi * 127773
            word = 16807 * lo - 2836 * hi
            if word < 0:
                word += 2147483647
            r.append(word)
        r += r[:SEP]                           # r[31..33] = r[0..2]
        # ring of the last 31 values (r[3..33]): r[i - 31] sits at i % 31
        self.r = [v & MASK for v in r[DEG:] + r[SEP:DEG]]
        self.i = DEG + SEP
        for _ in range(DEG * 10):
            self.random()

    def random(self):
        r, i = self.r, self.i
        k = i % DEG
        v = (r[k] + r[(i - SEP) % DEG]) & MASK
        r[k] = v
        self.i = i + 1
        return v >> 1

    def getstate(self):
        return (tuple(self.r), self.i)

    def setstate(self, st):
        self.r, self.i = list(st[0]), st[1]

    def copy(self):
        c = GlibcRandom.__new__(GlibcRandom)
        c.r, c.i = list(self.r), self.i
        return c
