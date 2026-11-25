"""Stage 2: compute and cache an embedding for pairing.

  python -m dpif.embed --config configs/cifar10_resnet18.yaml --seed 0 --kind imagenet_r50
"""
from __future__ import annotations

import argparse

import numpy as np

from .data import build_data
from .embeddings import SEED_INDEPENDENT, compute_embedding
from .utils import get_device, load_config, run_paths


def embedding_path(run_dir, seed, kind):
    p = run_paths(run_dir, seed)
    return (p["run"] if kind in SEED_INDEPENDENT else p["seed"]) / f"emb_{kind}.npy"


def load_scores(run_dir, seed) -> dict:
    sd = run_paths(run_dir, seed)["score"]
    with np.load(sd / "scores.npz") as z:
        out = {k: z[k] for k in z.files}
    extra = sd / "scores_extra.npz"
    if extra.exists():  # added later by `python -m dpif.rescore`
        with np.load(extra) as z:
            out.update({k: z[k] for k in z.files})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kind", required=True)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    run_dir = args.run_dir or f"runs/{cfg['name']}"
    out = embedding_path(run_dir, args.seed, args.kind)
    if out.exists():
        print(f"[embed] exists: {out}")
        return
    device = get_device(args.device)
    data = build_data(cfg["data"], device)
    scores = load_scores(run_dir, args.seed) if args.kind.startswith("self") else None
    kw = cfg.get("embed", {}).get(args.kind, {})
    emb = compute_embedding(args.kind, data, device, scores=scores, seed=args.seed, **kw)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, emb.astype(np.float32))
    print(f"[embed] {args.kind}: {emb.shape} -> {out}")


if __name__ == "__main__":
    main()
