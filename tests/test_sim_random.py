import ctypes

import pytest

from sim.glibc_random import GlibcRandom

libc = ctypes.CDLL("libc.so.6")
libc.random.restype = ctypes.c_long


@pytest.mark.parametrize("seed", [0, 1, 2, 12345, 2**31 - 2, 2**31 - 1, 4000000000])
def test_matches_libc(seed):
    libc.srandom(ctypes.c_uint(seed))
    g = GlibcRandom(seed)
    assert [g.random() for _ in range(10000)] == [libc.random() for _ in range(10000)]


def test_copy_is_independent():
    g = GlibcRandom(7)
    for _ in range(5):
        g.random()
    c = g.copy()
    a = [g.random() for _ in range(100)]
    assert [c.random() for _ in range(100)] == a
