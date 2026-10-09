#!/usr/bin/env python3
"""Behaviour cloning: train agents/bc.py's BCNetwork on heuristic labels.

    python scripts/train_bc.py --config configs/bc.yaml --run-name bc0
    python scripts/train_bc.py --config configs/bc_dagger1.yaml --run-name bc0_dagger1

Cross-entropy, Adam, minibatches drawn without replacement every epoch.
After each epoch: validation accuracy (all labels and per class); the epoch
with the best validation accuracy is saved as runs/<run>/bc_best.pt, the
last as bc_last.pt; per-epoch numbers go to runs/<run>/log.jsonl.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agents.bc import BCNetwork, load_bc, load_shards  # noqa: E402
from agents.dqn import GridDecoder  # noqa: E402
from env.pacman_env import CHANNELS  # noqa: E402


def expand(globs):
    paths = []
    for g in globs:
        found = sorted(glob.glob(os.path.join(ROOT, g)))
        if not found:
            sys.exit(f"no files match {g}")
        paths += found
    return paths


def class_weights(cfg, labels, n_actions):
    """(weights, shares); see configs/bc.yaml class_weights."""
    shares = np.bincount(labels, minlength=n_actions) / len(labels)
    w = np.ones(n_actions)
    if cfg["class_weights"] == "auto":
        main = shares >= cfg["min_class_share"]
        if shares[main].max() / shares[main].min() > cfg["imbalance_ratio"]:
            w[main] = shares[main].mean() / shares[main]
    return w, shares


@torch.no_grad()
def evaluate(net, decode, data, device, n_actions, batch=4096):
    net.eval()
    pred = []
    for i in range(0, len(data["act"]), batch):
        g = decode(torch.as_tensor(data["obs"][i:i + batch], device=device))
        v = torch.as_tensor(data["vec"][i:i + batch], device=device)
        pred.append(net(g, v).argmax(1).cpu().numpy())
    net.train()
    pred = np.concatenate(pred)
    ok = pred == data["act"]
    per_class = {a: float(ok[data["act"] == a].mean()) for a in range(n_actions)
                 if (data["act"] == a).any()}
    per_level = {int(lv): float(ok[data["level"] == lv].mean()) for lv in np.unique(data["level"])}
    return float(ok.mean()), per_class, per_level


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "bc.yaml"))
    p.add_argument("--run-name", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_dir = os.path.join(ROOT, "runs", args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    t0 = time.perf_counter()
    train = load_shards(expand(cfg["train_data"]))
    val = load_shards(expand(cfg["val_data"]))
    n, na = len(train["act"]), cfg["n_actions"]
    print(f"train {n} ({train['obs'].nbytes / 1e9:.1f} GB), val {len(val['act'])}, "
          f"loaded in {time.perf_counter() - t0:.0f} s", flush=True)
    w, shares = class_weights(cfg, train["act"], na)
    print("label shares " + " ".join(f"{s:.3f}" for s in shares)
          + "  class weights " + " ".join(f"{x:.2f}" for x in w), flush=True)

    net = BCNetwork(cfg["network"], na).to(device)
    if cfg.get("init_checkpoint"):
        init, _ = load_bc(os.path.join(ROOT, cfg["init_checkpoint"]), device)
        net.load_state_dict(init.state_dict())
        print(f"initialized from {cfg['init_checkpoint']}")
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    decode = GridDecoder(device, len(CHANNELS))
    weight = torch.tensor(w, dtype=torch.float32, device=device)
    ck_base = {"algo": "bc", "config": cfg, "n_actions": na, "env_config": {"actions": na},
               "class_weights": w.tolist(), "label_shares": shares.tolist()}
    best = -1.0
    bs = cfg["batch_size"]
    acc0, _, _ = evaluate(net, decode, val, device, na)
    print(f"epoch 0: val acc {acc0:.4f}", flush=True)
    for epoch in range(1, cfg["epochs"] + 1):
        t1 = time.perf_counter()
        perm = rng.permutation(n)
        losses = []
        for i in range(0, n - bs + 1, bs):
            idx = np.sort(perm[i:i + bs])  # sorted: faster gathers
            g = decode(torch.as_tensor(train["obs"][idx]).to(device, non_blocking=True))
            v = torch.as_tensor(train["vec"][idx], device=device)
            a = torch.as_tensor(train["act"][idx], device=device, dtype=torch.long)
            with torch.autocast("cuda", torch.bfloat16, enabled=device == "cuda"):
                logits = net(g, v)
            loss = F.cross_entropy(logits.float(), a, weight=weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.detach())
        train_loss = torch.stack(losses).mean().item()
        acc, per_class, per_level = evaluate(net, decode, val, device, na)
        rec = {"epoch": epoch, "train_loss": train_loss, "val_acc": acc,
               "val_acc_per_class": per_class, "val_acc_per_level": per_level,
               "seconds": round(time.perf_counter() - t1, 1)}
        with open(os.path.join(run_dir, "log.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"epoch {epoch}: loss {train_loss:.4f}  val acc {acc:.4f}  per class "
              + " ".join(f"{k}:{v:.3f}" for k, v in per_class.items())
              + f"  ({rec['seconds']:.0f} s)", flush=True)
        state = {**ck_base, "net": net.state_dict(), "epoch": epoch, "val_acc": acc,
                 "val_acc_per_class": per_class, "val_acc_per_level": per_level}
        torch.save(state, os.path.join(run_dir, "bc_last.pt"))
        if acc > best:
            best = acc
            torch.save(state, os.path.join(run_dir, "bc_best.pt"))
    print(f"best val acc {best:.4f}")


if __name__ == "__main__":
    main()
