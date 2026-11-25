"""Stage 3: choose a subset and write it as JSON (global indices + loss weights).

  python -m dpif.prune --run-dir runs/cifar10_r18 --seed 0 --method pair_influence --ratio 0.4 \
      --emb self --opt T_q=0.9
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import yaml

from .embed import embedding_path, load_scores
from .methods import ALL_METHODS, EMB_METHODS, select
from .utils import run_paths, save_json


def make_tag(method, emb, opts, tag=None):
    if tag:
        return tag
    t = method
    if method in EMB_METHODS or method == "merge":
        t += f"@{emb}"
    if opts:
        t += "[" + ",".join(f"{k}={v}" for k, v in sorted(opts.items()) if k != "device") + "]"
    return t


def subset_path(run_dir, seed, tag, ratio):
    safe = tag.replace("/", "_").replace(" ", "")
    return run_paths(run_dir, seed)["subsets"] / f"{safe}_r{ratio:.2f}.json"


def run(run_dir, seed, method, ratio, emb_kind="self", opts=None, tag=None, device="cpu"):
    opts = dict(opts or {})
    tag = make_tag(method, emb_kind, opts, tag)
    out = subset_path(run_dir, seed, tag, ratio)
    if out.exists():
        return out
    assert method in ALL_METHODS, method
    scores = load_scores(run_dir, seed)
    scores["features"] = scores["features"].astype(np.float32)
    emb = None
    if method in EMB_METHODS:
        p = embedding_path(run_dir, seed, emb_kind)
        if not p.exists():
            raise FileNotFoundError(f"{p} missing; run `python -m dpif.embed --kind {emb_kind}` first")
        emb = np.load(p)
    t0 = time.time()
    keep, w, info = select(method, ratio, scores, scores["labels"], emb=emb, seed=seed, opts={**opts, "device": device})
    info.update(select_time_s=time.time() - t0)
    save_json(dict(method=method, tag=tag, ratio=ratio, seed=seed, emb=emb_kind if method in EMB_METHODS else None,
                   opts=opts, indices=scores["pool_idx"][keep], weights=w, info=info), out)
    print(f"[prune] {tag} r={ratio:.2f} kept {len(keep)}/{len(scores['labels'])} "
          f"(achieved {info['achieved_ratio']:.3f}) -> {out}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", required=True, choices=sorted(ALL_METHODS))
    ap.add_argument("--ratio", type=float, required=True)
    ap.add_argument("--emb", default="self")
    ap.add_argument("--opt", nargs="*", default=[], help="method options k=v, e.g. T_q=0.9 k=10")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    opts = {k: yaml.safe_load(v) for k, v in (o.split("=", 1) for o in args.opt)}
    run(args.run_dir, args.seed, args.method, args.ratio, args.emb, opts, args.tag, args.device)


if __name__ == "__main__":
    main()
