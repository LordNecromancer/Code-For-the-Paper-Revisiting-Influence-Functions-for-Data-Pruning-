"""Stage 4: train a fresh model on a saved subset and record test/val accuracy.

  python -m dpif.train --config configs/cifar10_resnet18.yaml --subset runs/.../subsets/x.json --seed 0
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .data import build_data
from .models import build_model
from .training import accuracy, train_model
from .utils import get_device, load_config, load_json, save_json, seed_everything


def result_path(results_dir, tag, ratio, seed, budget="epochs"):
    suffix = "" if budget == "epochs" else f"_{budget}"
    return Path(results_dir) / f"{tag.replace('/', '_')}_r{ratio:.2f}_s{seed}{suffix}.json"


def run(cfg, subset_file, seed, results_dir, device):
    sub = load_json(subset_file)
    budget = cfg["train"].get("budget", "epochs")
    out = result_path(results_dir, sub["tag"], sub["ratio"], seed, budget)
    if out.exists():
        return out
    seed_everything(seed)
    data = build_data(cfg["data"], device)
    idx = np.asarray(sub["indices"], dtype=np.int64)
    w = np.asarray(sub["weights"], dtype=np.float64)
    syn = sub.get("synthetic")
    if syn:  # merged examples (Method 1): append to the training tensor as extra rows
        z = np.load(syn["file"])
        N = len(data.x_train)
        data.x_train = torch.cat([data.x_train, torch.from_numpy(z["x"]).to(device)])
        data.y_train = torch.cat([data.y_train, torch.from_numpy(z["y"]).long().to(device)])
        idx = np.concatenate([idx, N + np.arange(len(z["y"]))])
        w = np.concatenate([w, z["w"]])
    print(f"[train] {sub['tag']} r={sub['ratio']:.2f} n={len(idx)} seed={seed} budget={budget}")
    model = build_model(cfg["model"], data.num_classes)
    t0 = time.time()
    train_model(model, data, idx, None if np.allclose(w, 1) else w, cfg["train"], device, seed)
    res = dict(method=sub["method"], tag=sub["tag"], ratio=sub["ratio"], achieved_ratio=sub["info"]["achieved_ratio"],
               seed=seed, budget=budget, n_train=len(idx), test_acc=accuracy(model, data, "test", device),
               val_acc=accuracy(model, data, "val", device), val_a_acc=accuracy(model, data, "val_a", device),
               val_b_acc=accuracy(model, data, "val_b", device), train_time_s=time.time() - t0,
               select_time_s=sub["info"].get("select_time_s"), emb=sub.get("emb"), opts=sub.get("opts"))
    save_json(res, out)
    print(f"[train] test acc {res['test_acc']:.4f}  val acc {res['val_acc']:.4f}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    run_dir = Path(args.run_dir or f"runs/{cfg['name']}")
    run(cfg, args.subset, args.seed, run_dir / "results", get_device(args.device))


if __name__ == "__main__":
    main()
