#!/usr/bin/env python3
"""Convert a DQfD demo file (SharedReplay.save_demo) for another run setup.

    python tools/convert_demo.py runs/dqfd_demo/demo.npz out.npz \
        --add-cell-safety --n-step 5 --gamma 0.995

--add-cell-safety appends the cell-safety bit plane (env obs.cell_safety)
to every frame, computed from the frame itself: walls, gate, pacman and
the normal ghosts' cells from the packed planes, supertime from vec[1]
(the same as the env's plane, see tests/test_convert_demo.py).
--n-step / --gamma recompute the 1-step discounts and the n-step returns,
discounts and offsets of every transition from the stored 1-step rewards
(the frames of an episode are consecutive in their stream; an episode's
last observation is a frame with valid 0). Transitions at the end of a
stream whose window is not complete are dropped (valid 0).
"""

import argparse
import multiprocessing
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agents.dqn import BINARY_CH, CELLS, COUNT_BITS, COUNT_CH, N_PLANES  # noqa: E402
from env.pacman_env import (CH, HEIGHT, WIDTH, _ghost_neighbours,  # noqa: E402
                            _passable_neighbours, cell_safety_field)

_nbrs = {}


def _plane(bits, ch):
    """Decoded 0/1 cells (B, CELLS) of a binary channel."""
    return bits[:, BINARY_CH.index(ch)]


def safety_from_packed(packed, vecs):
    """uint8 (B, N_PLANES + ..., PACKED) frames, float32 (B, vec) -> uint8
    (B, CELLS) cell-safety planes."""
    bits = np.unpackbits(packed[:, :N_PLANES], axis=-1)[:, :, :CELLS]
    walls, gate, pac = (_plane(bits, CH[n]) for n in ("walls", "gate", "pacman"))
    j = COUNT_CH.index(CH["ghost_normal"])
    nb = len(BINARY_CH)
    normal = sum(bits[:, nb + b * len(COUNT_CH) + j] for b in range(COUNT_BITS)) > 0
    out = np.zeros((len(packed), CELLS), np.uint8)
    for i in range(len(packed)):
        layout = np.where(walls[i] > 0, "#", np.where(gate[i] > 0, "-", " "))
        key = "".join(layout)
        if key not in _nbrs:
            _nbrs[key] = (_passable_neighbours(key), _ghost_neighbours(key))
        p = np.flatnonzero(pac[i])
        if not len(p):  # no pacman on the board (should not happen)
            continue
        out[i] = cell_safety_field(*_nbrs[key], int(p[0]), np.flatnonzero(normal[i]).tolist(),
                                   bool(vecs[i, 1] > 0))
    return out


def _work(args):
    packed, vecs = args
    s = safety_from_packed(packed, vecs)
    return np.packbits(s, axis=-1)[:, None]


def recompute_nstep(z, n, gamma):
    """New (disc1, ret, discn, offn, valid) arrays for n-step n, discount gamma."""
    rew1, valid = z["rew1"], z["valid"]
    disc1 = np.where(z["disc1"] > 0, np.float32(gamma), np.float32(0)).astype(np.float32)
    ret = np.zeros_like(z["ret"])
    discn = np.zeros_like(z["discn"])
    offn = np.zeros_like(z["offn"])
    new_valid = np.zeros_like(valid)
    bases = np.concatenate([[0], np.cumsum(z["sizes"])[:-1]])
    for s, base in enumerate(bases):
        # frames below fin have final fields (valid 0 there: an episode's last
        # observation); fin..end are one ongoing episode (pending transitions
        # and the current observation)
        fin = base + int(z["final"][s])
        end = base + int(z["written"][s])
        for c in range(base, fin):
            if not valid[c]:
                continue
            r, k, term = 0.0, 0, False
            while True:
                r += gamma ** k * float(rew1[c + k])
                k += 1
                nxt = c + k
                if nxt >= end:
                    k = -1  # the window runs past the frames written
                    break
                if nxt < fin and not valid[nxt]:  # the episode ended after this step
                    term = z["disc1"][c + k - 1] == 0
                    break
                if k == n:
                    break
            if k < 0:
                continue
            ret[c] = r
            discn[c] = 0.0 if term else gamma ** k
            offn[c] = k
            new_valid[c] = 1
    return disc1, ret, discn, offn, new_valid


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--add-cell-safety", action="store_true")
    p.add_argument("--n-step", type=int, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    with np.load(args.src) as f:
        z = {k: f[k] for k in f.files}
    if args.n_step:
        if args.gamma is None:
            sys.exit("--n-step needs --gamma")
        z["disc1"], z["ret"], z["discn"], z["offn"], new_valid = \
            recompute_nstep(z, args.n_step, args.gamma)
        print(f"n-step {args.n_step}, gamma {args.gamma}: {int(new_valid.sum())} transitions "
              f"(was {int(z['valid'].sum())})", flush=True)
        z["valid"] = new_valid
    if args.add_cell_safety:
        obs, vec = z["obs"], z["vec"]
        chunk = 20000
        jobs = [(obs[i:i + chunk], vec[i:i + chunk]) for i in range(0, len(obs), chunk)]
        with multiprocessing.get_context("fork").Pool(args.workers) as pool:
            planes = []
            for i, pl in enumerate(pool.imap(_work, jobs)):
                planes.append(pl)
                print(f"safety: {min((i + 1) * chunk, len(obs))}/{len(obs)}", flush=True)
        z["obs"] = np.concatenate([obs, np.concatenate(planes)], 1)
    tmp = args.dst + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **z)
    os.replace(tmp, args.dst)
    print(f"wrote {args.dst}: obs {z['obs'].shape}")


if __name__ == "__main__":
    main()
