#!/usr/bin/env python3
"""Collect planner-labelled states for distillation (docs/overnight.md, "distill").

    python scripts/distill_collect.py --config configs/distill_collect.yaml
    python scripts/distill_collect.py --config configs/distill_collect.yaml --resume

Games in the sim (sim/csim.py) from a start level drawn uniformly from
collect.start_levels, played in lock-step batches as in scripts/plan_eval.py.
At every state the honest planner (plan_eval.Planner, all four directions
rolled out: Planner.choose(full=True)) chooses a move. Without a driver the
planner's move is played (step 1 of the distill plan); with driver: <ckpt>
(DAgger) the driver network's greedy move is played and the planner only
labels the state.

Per state: obs (packed grid, agents.dqn.pack_grids, + vec), the planner's
action (env index U/D/L/R/N), the played action, the death share of each
direction U/D/L/R, Q of the planner's network (5), and bookkeeping (game
seed, step in the game, level, start level, pacman's direction, all-dropped
and loop-breaker flags).

Storage (out_dir): workers write their buffer to chunks/ every
collect.flush_minutes (and at the end) atomically; the main process merges
chunks into part_NNN.npz of exactly collect.part_size states (the last part
may be smaller) and records consumed chunks in manifest.json before deleting
them. A stop loses at most flush_minutes of collection plus the games in
progress (their already flushed states stay). Ctrl-C / SIGTERM: every
worker flushes its buffer at once and the merger writes the full parts (the
rest stays in chunks/ for --resume). --resume counts what is in
the parts and chunks and continues with a new session (new game seeds) until
collect.target_states. progress.log gets a line every log_seconds.
"""

import argparse
import glob
import json
import multiprocessing
import os
import signal
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from plan_eval import Planner, QNet, Stall, batch_advance, batch_info, batch_obs, load_cfg  # noqa: E402

FIELDS = ("obs", "vec", "act", "played", "frac", "q", "seed", "step", "level", "start_level",
          "pdir", "all_dropped", "breaker")
SESSION_STRIDE = 10_000_000   # game seeds: seed_base + session * stride + worker * 1e6 + j
WORKER_STRIDE = 1_000_000


def game_task(cfg, session, worker, j):
    """(seed, start level) of the worker's j-th game in this session."""
    c = cfg["collect"]
    seed = int(c["seed_base"]) + session * SESSION_STRIDE + worker * WORKER_STRIDE + j
    lo, hi = c["start_levels"]
    level = int(np.random.default_rng([seed, 7]).integers(lo, hi + 1))
    return seed, level


def save_npz(path, arrays):
    """Write atomically (tmp + rename) so a stop never leaves half a file."""
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


class Buffer:
    def __init__(self):
        self.rows = {k: [] for k in FIELDS}
        self.n = 0

    def add(self, **kw):
        for k in FIELDS:
            self.rows[k].append(kw[k])
        self.n += len(kw["act"])

    def take(self):
        out = {k: np.concatenate(v) for k, v in self.rows.items()}
        self.__init__()
        return out


def collect_worker(cfg, out_dir, session, worker, counter, device):
    """Play games until the shared counter of produced states (flushed or
    still in a worker's buffer) reaches the target; flush the buffer to
    chunks/ every flush_minutes."""
    from agents.dqn import pack_grids
    from env.pacman_env import grid_extras
    from sim.csim import SIZE, CSim
    c = cfg["collect"]
    target, G = int(c["target_states"]), int(cfg["run"]["games_per_worker"])
    max_steps, flush_s = int(c["max_steps"]), 60 * float(c["flush_minutes"])
    net = QNet(os.path.join(ROOT, cfg["checkpoint"]), device)
    extras = grid_extras(net.env_overrides)
    driver = None
    if c.get("driver"):
        driver = QNet(os.path.join(ROOT, c["driver"]), device)
        assert driver.cell_safety == net.cell_safety
    planner = Planner(net, cfg)
    full = not c.get("lazy", False)
    buf, n_chunk, last_flush = Buffer(), 0, time.time()
    games, rows, j = [], np.zeros((0, SIZE), np.uint8), 0

    def flush():
        nonlocal n_chunk, last_flush
        last_flush = time.time()
        if not buf.n:
            return
        data = buf.take()
        name = f"s{session:02d}_w{worker:02d}_{n_chunk:05d}.npz"
        save_npz(os.path.join(out_dir, "chunks", name), data)
        n_chunk += 1

    while counter.value < target and not _STOP["flag"]:
        while len(games) < G:
            seed, level = game_task(cfg, session, worker, j)
            j += 1
            sim = CSim()
            sim.reset(seed, level)
            info = sim.info()
            games.append({"seed": seed, "start_level": level, "length": 0, "lives": int(info[0]),
                          "level": level, "eats": int(info[4]), "stall": Stall(cfg)})
            rows = np.concatenate([rows, sim.buf[None]])
        info = batch_info(rows)
        grid, vec = batch_obs(net, rows)
        q = net(grid, vec)
        seeds = [[g["seed"], g["start_level"], g["length"]] for g in games]
        act, st = planner.choose(rows, info, q, seeds, [g["stall"].visits() for g in games],
                                 full=full, frac_dir=True)
        played = act if driver is None else driver(grid, vec).argmax(1).astype(np.int32)
        buf.add(obs=pack_grids(grid, extras), vec=vec, act=act.astype(np.int8),
                played=played.astype(np.int8), frac=st["frac_dir"].astype(np.float32),
                q=q.astype(np.float32), seed=np.array([g["seed"] for g in games], np.int64),
                step=np.array([g["length"] for g in games], np.int32),
                level=info[:, 1].astype(np.int16),
                start_level=np.array([g["start_level"] for g in games], np.int16),
                pdir=info[:, 5].astype(np.int8), all_dropped=st["all_dropped"],
                breaker=st["breaker"])
        with counter.get_lock():
            counter.value += len(games)
        batch_advance(rows, played)
        info = batch_info(rows)
        keep = []
        for i, g in enumerate(games):
            g["length"] += 1
            lives, level, eats = int(info[i, 0]), int(info[i, 1]), int(info[i, 4])
            g["stall"].update(int(info[i, 2]), level != g["level"] or eats < g["eats"])
            g["lives"], g["eats"], g["level"] = lives, eats, max(g["level"], level)
            if lives > 0 and g["length"] < max_steps:
                keep.append(i)
        games = [games[i] for i in keep]
        rows = rows[keep]
        if time.time() - last_flush >= flush_s:
            flush()
    flush()
    return worker


_COUNTER = {}
_STOP = {"flag": False}


def _on_signal(signum, frame):
    _STOP["flag"] = True


def _init_counter(counter):
    _COUNTER["c"] = counter
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)


def _worker(args):
    cfg, out_dir, session, worker, device = args
    return collect_worker(cfg, out_dir, session, worker, _COUNTER["c"], device)


class Merger:
    """chunks/*.npz -> part_NNN.npz of part_size states, in chunk-name order."""

    def __init__(self, out_dir, part_size):
        self.dir, self.part_size = out_dir, int(part_size)
        self.manifest_path = os.path.join(out_dir, "manifest.json")
        self.m = {"parts": {}, "consumed": [], "sessions": []}
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path) as f:
                self.m = json.load(f)
        # a stop between a part's manifest entry and the chunk deletion
        for name in self.m["consumed"]:
            p = os.path.join(out_dir, "chunks", name)
            if os.path.exists(p):
                os.remove(p)
        for p in glob.glob(os.path.join(out_dir, "**", "*.tmp.npz"), recursive=True):
            os.remove(p)

    def save_manifest(self):
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.m, f, indent=1)
        os.replace(tmp, self.manifest_path)

    def chunk_paths(self):
        # not the *.tmp.npz a worker is still writing (save_npz renames it when done)
        return sorted(p for p in glob.glob(os.path.join(self.dir, "chunks", "*.npz"))
                      if not p.endswith(".tmp.npz"))

    def counts(self):
        """(states in parts, states in chunks not merged yet)."""
        in_parts = sum(self.m["parts"].values())
        in_chunks = 0
        for p in self.chunk_paths():
            with np.load(p) as z:
                in_chunks += len(z["act"])
        return in_parts, in_chunks

    def merge(self, final=False):
        """Write every full part the chunks allow (with final also the rest)."""
        paths = self.chunk_paths()
        sizes = []
        for p in paths:
            with np.load(p) as z:
                sizes.append(len(z["act"]))
        while paths and (sum(sizes) >= self.part_size or final):
            take, n = [], 0
            while paths and n < self.part_size:
                take.append(paths.pop(0))
                n += sizes.pop(0)
            data = {k: [] for k in FIELDS}
            for p in take:
                with np.load(p) as z:
                    for k in FIELDS:
                        data[k].append(z[k])
            data = {k: np.concatenate(v) for k, v in data.items()}
            rest = {k: v[self.part_size:] for k, v in data.items()}
            data = {k: v[:self.part_size] for k, v in data.items()}
            idx = len(self.m["parts"])
            name = f"part_{idx:03d}.npz"
            save_npz(os.path.join(self.dir, name), data)
            if len(rest["act"]):
                # the rest goes back as a chunk that sorts before the others
                rname = f"r{idx:03d}_rest.npz"
                save_npz(os.path.join(self.dir, "chunks", rname), rest)
                paths.insert(0, os.path.join(self.dir, "chunks", rname))
                sizes.insert(0, len(rest["act"]))
            self.m["parts"][name] = len(data["act"])
            self.m["consumed"] = [os.path.basename(p) for p in take]
            self.save_manifest()
            for p in take:
                os.remove(p)
            self.m["consumed"] = []
            self.save_manifest()


def log(path, msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    try:
        print(line, flush=True)
    except OSError:   # a closed stdout (e.g. a killed tee) must not stop the merger
        pass
    with open(path, "a") as f:
        f.write(line + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    p.add_argument("--resume", action="store_true", help="continue a stopped collection")
    p.add_argument("--target", type=int, help="override collect.target_states")
    p.add_argument("--workers", type=int, help="override run.workers")
    args = p.parse_args()
    cfg = load_cfg(args.config)
    if args.target:
        cfg["collect"]["target_states"] = args.target
    if args.workers:
        cfg["run"]["workers"] = args.workers
    c = cfg["collect"]
    out_dir = os.path.join(ROOT, c["out_dir"])
    if os.path.exists(os.path.join(out_dir, "manifest.json")) and not args.resume:
        sys.exit(f"{out_dir} has a collection already; use --resume to continue it")
    os.makedirs(os.path.join(out_dir, "chunks"), exist_ok=True)
    merger = Merger(out_dir, c["part_size"])
    logf = os.path.join(out_dir, "progress.log")
    session = len(merger.m["sessions"])
    in_parts, in_chunks = merger.counts()
    have = in_parts + in_chunks
    target = int(c["target_states"])
    merger.m["sessions"].append({"started": time.strftime("%Y-%m-%d %H:%M:%S"), "had": have,
                                 "config": args.config})
    merger.save_manifest()
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        import yaml
        yaml.safe_dump(cfg, f)
    log(logf, f"session {session}: {have} states already ({in_parts} in parts), target {target}")
    if have >= target:
        merger.merge(final=True)
        log(logf, "target reached; parts: " + ", ".join(merger.m["parts"]))
        return
    ctx = multiprocessing.get_context("spawn")
    counter = ctx.Value("q", have)
    nw = int(cfg["run"]["workers"])
    t0, n0, last_log = time.time(), have, 0.0
    with ctx.Pool(nw, initializer=_init_counter, initargs=(counter,)) as pool:
        res = pool.map_async(_worker, [(cfg, out_dir, session, w, cfg["run"]["device"])
                                       for w in range(nw)])
        signal.signal(signal.SIGINT, _on_signal)
        signal.signal(signal.SIGTERM, _on_signal)
        stop_logged = False
        while not res.ready():
            if _STOP["flag"] and not stop_logged:
                stop_logged = True
                log(logf, "stop requested: workers flush their buffers")
                for pid in [w.pid for w in pool._pool]:
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            res.wait(10)
            merger.merge()
            if time.time() - last_log >= float(c["log_seconds"]):
                last_log = time.time()
                n = counter.value
                rate = (n - n0) / max(time.time() - t0, 1e-9)
                eta = (target - n) / rate / 3600 if rate > 0 else float("inf")
                log(logf, f"{n} / {target} states ({len(merger.m['parts'])} parts written), "
                          f"{rate:.1f} states/s this session, ETA {eta:.1f} h")
        res.get()  # re-raise a worker's exception
    if _STOP["flag"]:
        merger.merge()
        a, b = merger.counts()
        log(logf, f"stopped: {a} states in {len(merger.m['parts'])} parts, {b} in chunks/ "
                  f"(continue with --resume)")
        return
    merger.merge(final=True)
    n = sum(merger.m["parts"].values())
    log(logf, f"done: {n} states in {len(merger.m['parts'])} parts, "
              f"{(n - n0) / (time.time() - t0):.1f} states/s this session")


if __name__ == "__main__":
    main()
