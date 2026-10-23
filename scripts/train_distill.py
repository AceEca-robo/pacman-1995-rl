#!/usr/bin/env python3
"""Distil the planner into the network (docs/overnight.md, "distill").

    python scripts/train_distill.py --config configs/dist0.yaml
    python scripts/train_distill.py --config configs/dist0.yaml --resume

Data: part_*.npz from scripts/distill_collect.py (every dir in cfg data).
The network is best_v04's QNetwork (initialised from init_checkpoint) plus
an optional auxiliary linear head on the hidden layer that predicts the
death share of each direction U/D/L/R.

Loss per sample (a = the planner's action, Q = the network's 5 outputs):
    ce * cross_entropy(Q / temperature, a)
  + margin_weight * (max_b [Q_b + margin * (b != a)] - Q_a)      (DQfD)
  + aux_death * BCE(sigmoid(aux), death shares)                  (optional)

Validation: the games whose seed hashes into val_frac (5%: a game is
either all train or all val, its states are correlated). Metrics: accuracy
(exact action), move accuracy (N counts as pacman's current direction, the
move it makes), agreement of best_v04's greedy action with the labels on
the same states (from the stored Q, the reference), aux MAE.

Checkpoints in run_dir: epoch_NN.pt after every epoch and best.pt (highest
val move accuracy) in train.py's DQN format (learner.online; plays with
scripts/evaluate.py, train_eval.py, plan_eval.py, play.py), state.pt for
--resume (written after every epoch and every checkpoint_minutes inside
an epoch: model, aux head, optimizer, epoch, position in the epoch's
permutation; Ctrl-C / SIGTERM writes it at the current step and exits).
log.jsonl: one line per epoch.
"""

import argparse
import glob
import json
import os
import signal
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agents.dqn import GridDecoder, QNetwork, large_margin  # noqa: E402
from env.pacman_env import grid_extras, obs_dims  # noqa: E402

N_ACTION = 4
LOAD = ("obs", "vec", "act", "frac", "q", "pdir", "seed")


class DistillNet(nn.Module):
    """QNetwork plus an auxiliary head (death share logits of U/D/L/R)."""

    def __init__(self, net_cfg, n_actions, vec_dim, n_channels):
        super().__init__()
        self.q = QNetwork(net_cfg, n_actions, vec_dim, n_channels)
        self.aux = nn.Linear(net_cfg["hidden"], 4)

    def forward(self, grid, vec):
        q = self.q
        h = q.fc(torch.cat([q.conv(grid), q.vec(vec)], 1))
        adv = q.adv(h)
        return q.value(h) + adv - adv.mean(1, keepdim=True), self.aux(h)


def load_parts(dirs):
    paths = []
    for d in dirs:
        found = sorted(glob.glob(os.path.join(ROOT, d, "part_*.npz")))
        if not found:
            sys.exit(f"no part_*.npz in {d}")
        paths += found
    sizes = []
    for p in paths:
        with np.load(p) as z:
            sizes.append(len(z["act"]))
            shapes = {k: (z[k].shape[1:], z[k].dtype) for k in LOAD}
    out = {k: np.empty((sum(sizes),) + s, t) for k, (s, t) in shapes.items()}
    i = 0
    for p, n in zip(paths, sizes):
        with np.load(p) as z:
            for k in LOAD:
                out[k][i:i + n] = z[k]
        i += n
    return out, paths


def val_mask(seeds, frac):
    """A fixed hash of the game seed: the same split for every run."""
    h = (seeds.astype(np.uint64) * np.uint64(2654435761)) % np.uint64(2**32)
    return h.astype(np.float64) / 2**32 < frac


def move(a, pdir):
    """Action -> the move it makes: N is pacman's current direction (when
    pacman stands, pdir > 3, N stays N)."""
    a = a.copy()
    n = (a == N_ACTION) & (pdir <= 3)
    a[n] = pdir[n]
    return a


def save_dqn_format(path, template, net, meta):
    sd = {k: v.detach().cpu().clone() for k, v in net.q.state_dict().items()}
    ck = {"config": template["config"], "n_actions": template["n_actions"],
          "env_config": template["env_config"], "seed": template.get("seed"),
          "learner": {"online": sd, "target": sd}, "algo": "dqn",
          "aux_head": {k: v.detach().cpu().clone() for k, v in net.aux.state_dict().items()},
          "distill": meta}
    tmp = path + ".tmp"
    torch.save(ck, tmp)
    os.replace(tmp, path)


class Trainer:
    def __init__(self, cfg, device):
        self.cfg, self.device = cfg, device
        t = cfg["train"]
        self.template = torch.load(os.path.join(ROOT, cfg["init_checkpoint"]), map_location="cpu",
                                   weights_only=False)
        env_cfg = self.template.get("env_config") or {}
        n_channels, vec_dim = obs_dims(env_cfg)
        self.net = DistillNet(self.template["config"]["network"], self.template["n_actions"],
                              vec_dim, n_channels)
        init = self.template["learner"]["online"]
        self.net.q.load_state_dict(init)
        if "aux_head" in self.template:
            self.net.aux.load_state_dict(self.template["aux_head"])
        self.net.to(device)
        self.decode = GridDecoder(device, n_channels, grid_extras(env_cfg))
        self.opt = torch.optim.Adam(self.net.parameters(), lr=float(t["lr"]),
                                    eps=float(t.get("adam_eps", 1e-8)))
        if str(device).startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def batch(self, data, idx):
        g = self.decode(torch.as_tensor(data["obs"][idx], device=self.device))
        v = torch.as_tensor(data["vec"][idx], device=self.device)
        return g, v

    def losses(self, data, idx):
        lc = self.cfg["loss"]
        g, v = self.batch(data, idx)
        a = torch.as_tensor(data["act"][idx].astype(np.int64), device=self.device)
        q, aux = self.net(g, v)
        ce = F.cross_entropy(q / float(lc.get("temperature", 1.0)), a)
        margin = large_margin(q, a, float(lc["margin"])).mean()
        total = float(lc["ce"]) * ce + float(lc["margin_weight"]) * margin
        out = {"ce": ce, "margin": margin}
        if float(lc.get("aux_death", 0)):
            f = torch.as_tensor(data["frac"][idx], device=self.device)
            known = ~torch.isnan(f)   # lazy collection: directions not rolled out
            bce = F.binary_cross_entropy_with_logits(aux, torch.nan_to_num(f), reduction="none")
            out["aux"] = (bce * known).sum() / known.sum().clamp(min=1)
            total = total + float(lc["aux_death"]) * out["aux"]
        out["loss"] = total
        return out

    @torch.no_grad()
    def evaluate(self, data, idx, bs=2048):
        self.net.eval()
        pred, mae = [], []
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            q, aux = self.net(*self.batch(data, j))
            pred.append(q.argmax(1).cpu().numpy())
            f = torch.as_tensor(data["frac"][j], device=self.device)
            mae.append((torch.sigmoid(aux) - f).abs().flatten().cpu().numpy())
        self.net.train()
        pred = np.concatenate(pred)
        act, pdir = data["act"][idx].astype(np.int64), data["pdir"][idx].astype(np.int64)
        ref = data["q"][idx].argmax(1)
        return {"acc": float(np.mean(pred == act)),
                "move_acc": float(np.mean(move(pred, pdir) == move(act, pdir))),
                "ref_move_acc": float(np.mean(move(ref, pdir) == move(act, pdir))),
                "aux_mae": float(np.nanmean(np.concatenate(mae)))}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    t = cfg["train"]
    run_dir = os.path.join(ROOT, cfg["run_dir"])
    state_path = os.path.join(run_dir, "state.pt")
    if os.path.exists(state_path) and not args.resume:
        sys.exit(f"{run_dir} has a run already; use --resume to continue it")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(t["seed"]))
    data, paths = load_parts(cfg["data"])
    is_val = val_mask(data["seed"], float(t["val_frac"]))
    tr_idx, va_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    if t.get("max_val"):
        va_idx = va_idx[np.random.default_rng(0).permutation(len(va_idx))[:int(t["max_val"])]]
        va_idx.sort()
    print(f"{len(data['act'])} states from {len(paths)} parts: {len(tr_idx)} train, "
          f"{len(va_idx)} val ({len(np.unique(data['seed'][is_val]))} val games), device {device}",
          flush=True)
    tr = Trainer(cfg, device)
    bs, epochs = int(t["batch_size"]), int(t["epochs"])
    steps_per_epoch = len(tr_idx) // bs
    epoch, pos, best = 0, 0, -1.0
    if args.resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location=device, weights_only=False)
        tr.net.load_state_dict(st["net"])
        tr.opt.load_state_dict(st["opt"])
        epoch, pos, best = st["epoch"], st["pos"], st["best"]
        torch.set_rng_state(st["torch_rng"])
        print(f"resumed at epoch {epoch} step {pos}/{steps_per_epoch}", flush=True)
    meta = {"data": cfg["data"], "init": cfg["init_checkpoint"], "states": int(len(data["act"]))}

    def save_state(e, i):
        tmp = state_path + ".tmp"
        torch.save({"net": tr.net.state_dict(), "opt": tr.opt.state_dict(), "epoch": e, "pos": i,
                    "best": best, "torch_rng": torch.get_rng_state()}, tmp)
        os.replace(tmp, state_path)

    logf = os.path.join(run_dir, "log.jsonl")
    if epoch == 0 and pos == 0:
        ev = tr.evaluate(data, va_idx)
        print(f"epoch 0 (init) val {json.dumps(ev)}", flush=True)
        with open(logf, "a") as f:
            f.write(json.dumps({"epoch": 0, "val": ev, "time": time.strftime("%H:%M:%S")}) + "\n")
    ck_s = 60 * float(t["checkpoint_minutes"])
    stop = {"flag": False}

    def on_signal(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    while epoch < epochs:
        perm = tr_idx[np.random.default_rng([int(t["seed"]), epoch]).permutation(len(tr_idx))]
        sums, n, t0, last_ck = {}, 0, time.time(), time.time()
        for i in range(pos, steps_per_epoch):
            idx = np.sort(perm[i * bs:(i + 1) * bs])
            out = tr.losses(data, idx)
            tr.opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            if t.get("grad_clip"):
                nn.utils.clip_grad_norm_(tr.net.parameters(), float(t["grad_clip"]))
            tr.opt.step()
            for k, v in out.items():
                sums[k] = sums.get(k, 0.0) + v.detach()
            n += 1
            if (i + 1) % int(t.get("log_every", 1000)) == 0:
                msg = " ".join(f"{k} {float(v) / n:.4f}" for k, v in sums.items())
                print(f"epoch {epoch + 1} step {i + 1}/{steps_per_epoch} {msg} "
                      f"{n / (time.time() - t0):.0f} it/s", flush=True)
            if time.time() - last_ck >= ck_s or stop["flag"]:
                save_state(epoch, i + 1)
                last_ck = time.time()
                if stop["flag"]:
                    print(f"stopped at epoch {epoch + 1} step {i + 1}; state.pt written "
                          f"(continue with --resume)", flush=True)
                    return
        epoch, pos = epoch + 1, 0
        ev = tr.evaluate(data, va_idx)
        row = {"epoch": epoch, "train": {k: float(v) / max(n, 1) for k, v in sums.items()},
               "val": ev, "time": time.strftime("%H:%M:%S"), "epoch_s": round(time.time() - t0)}
        save_dqn_format(os.path.join(run_dir, f"epoch_{epoch:02d}.pt"), tr.template, tr.net,
                        {**meta, "epoch": epoch, "val": ev})
        if ev["move_acc"] > best:
            best = ev["move_acc"]
            save_dqn_format(os.path.join(run_dir, "best.pt"), tr.template, tr.net,
                            {**meta, "epoch": epoch, "val": ev})
            row["best"] = True
        save_state(epoch, 0)
        with open(logf, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"epoch {epoch} val {json.dumps(ev)}{' best' if row.get('best') else ''}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
