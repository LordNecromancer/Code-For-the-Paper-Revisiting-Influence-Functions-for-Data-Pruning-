"""Diagnose why pair-based pruning does or does not work for a scoring run. No GPU training.

For the final model and every early snapshot (suffix _e<k>) it reports:
  nc1               within-class / between-class feature variance (-> 0 under neural collapse;
                    then nearest neighbours in feature space are close to random within a class)
  saturated_1e-4    fraction of pool examples with 1 - p_y < 1e-4 (influence carries little signal)
  pair_dist_*       cosine distance of matched pairs (median, 10th pct) vs random same-class pairs
  pixel_ratio       pixel distance of matched pairs / random same-class pairs (needs --config;
                    ~1.0 means the "near-duplicates" are no closer in pixels than random images)
  rho_infl_loss     Spearman(param_norm, loss); ~1 means influence just re-ranks by loss
  agree_infl_loss   fraction of pairs where influence and loss drop the same member (0.5 = unrelated)

  python -m dpif.diagnose runs/cifar10_r18 --seed 0 --config configs/cifar10_resnet18.yaml
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from .embed import load_scores
from .influence import softmax_parts
from .pairing import find_pairs
from .utils import load_config, run_paths, save_json


def _rank(x):
    return np.argsort(np.argsort(x))


def spearman(a, b):
    return float(np.corrcoef(_rank(a), _rank(b))[0, 1])


def nc1(F, y):
    mu = F.mean(0)
    within = between = 0.0
    for c in np.unique(y):
        Fc = F[y == c]
        mc = Fc.mean(0)
        within += ((Fc - mc) ** 2).sum()
        between += len(Fc) * ((mc - mu) ** 2).sum()
    return float(within / max(between, 1e-12))


def _cos_dist(A, B):
    A = A / np.linalg.norm(A, axis=1, keepdims=True).clip(1e-12)
    B = B / np.linalg.norm(B, axis=1, keepdims=True).clip(1e-12)
    return 1 - (A * B).sum(1)


def analyse(scores, suffix, y, rng, pixels=None, k=10, device="cpu"):
    F = scores["features" + suffix].astype(np.float32)
    logits = torch.as_tensor(scores["logits" + suffix])
    _, om = softmax_parts(logits)
    om_y = om[torch.arange(len(y)), torch.as_tensor(y)].numpy()
    infl = scores["param_norm" + suffix]
    loss = scores["loss" + suffix]
    pairs, d = find_pairs(F, y, np.arange(len(y)), k=k, device=device)
    # random same-class pairs for reference
    ru, rv = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        h = len(idx) // 2
        ru.append(idx[:h]); rv.append(idx[h:2 * h])
    ru, rv = np.concatenate(ru), np.concatenate(rv)
    rd = _cos_dist(F[ru], F[rv])
    u, v = pairs[:, 0], pairs[:, 1]
    out = dict(
        nc1=nc1(F, y),
        saturated_1e4=float((om_y < 1e-4).mean()),
        train_loss_mean=float(loss.mean()),
        pair_dist_median=float(np.median(d)), pair_dist_p10=float(np.quantile(d, 0.1)),
        random_pair_dist_median=float(np.median(rd)),
        rho_infl_loss=spearman(infl, loss),
        agree_infl_loss=float(((infl[u] <= infl[v]) == (loss[u] <= loss[v])).mean()),
    )
    if "el2n" in scores:
        out["rho_infl_el2n"] = spearman(infl, scores["el2n"])
    if pixels is not None:
        pd = lambda a, b: np.sqrt(((pixels[a] - pixels[b]) ** 2).reshape(len(a), -1).sum(1))
        out["pixel_ratio"] = float(np.median(pd(u, v)) / np.median(pd(ru, rv)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None, help="enables the pixel-distance check (loads the dataset)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    scores = load_scores(args.run_dir, args.seed)
    y = scores["labels"]
    rng = np.random.default_rng(0)
    pixels = None
    if args.config:
        from .data import build_data
        data = build_data(load_config(args.config)["data"], torch.device("cpu"))
        pixels = (data.x_train[torch.as_tensor(scores["pool_idx"])].float() / 255).numpy()
    suffixes = [""] + sorted({k[len("features"):] for k in scores if k.startswith("features_e")},
                             key=lambda s: int(s[2:]))
    rows = {}
    for sfx in suffixes:
        name = "final" if sfx == "" else f"epoch {sfx[2:]}"
        rows[name] = analyse(scores, sfx, y, rng, pixels, device=args.device)
    keys = list(next(iter(rows.values())).keys())
    print(f"{'':24s}" + "".join(f"{n:>14s}" for n in rows))
    for k in keys:
        print(f"{k:24s}" + "".join(f"{rows[n][k]:14.4f}" for n in rows))
    save_json(rows, run_paths(args.run_dir, args.seed)["seed"] / "diagnosis.json")


if __name__ == "__main__":
    main()
