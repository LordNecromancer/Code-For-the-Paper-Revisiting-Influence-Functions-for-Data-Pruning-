"""Run a whole grid (seeds x methods x ratios). Every stage skips outputs that already exist,
so an interrupted sweep can simply be restarted. Use --shard i/n to split training jobs
across n processes/GPUs (e.g. CUDA_VISIBLE_DEVICES=1 python -m dpif.sweep ... --shard 1/2).

  python -m dpif.sweep sweeps/cifar10_main.yaml
  python -m dpif.sweep sweeps/cifar10_main.yaml --set train.epochs=30 --name cifar10_dev   # quick dev pass

Optional sweep keys:
  train_set: [model.arch=vgg11_cifar]   overrides applied ONLY when training on the subsets
                                        (cross-architecture transfer); results go to results_<suffix>/
  results_suffix: vgg11                 required with train_set
  method entries with `method: merge` run Method 1 (dpif.merge) instead of dpif.prune
  rescore: true                         add directed influence scores (dpif.rescore) after scoring
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml

from . import embed as embed_cli
from . import merge as merge_cli
from . import prune as prune_cli
from . import rescore as rescore_cli
from . import score as score_cli
from . import train as train_cli
from .methods import EMB_METHODS
from .utils import get_device, load_config, run_paths


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides applied to every stage")
    ap.add_argument("--name", default=None, help="override experiment name (use with --set!)")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--seeds", default=None, help="comma-separated seed override, e.g. 1 or 0,2 "
                    "(run different seeds on different machines, then copy results/ together)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop cleanly (exit 0) before starting a new stage once this many hours have passed; "
                         "use it on Kaggle/Colab so the session ends before the platform's limit")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    sw = yaml.safe_load(open(args.sweep))
    overrides = list(sw.get("set", [])) + list(args.set)
    name = args.name or sw.get("name")
    if name:
        overrides.append(f"name={name}")
    if args.set and not args.name and not all(o.startswith("data.root=") for o in args.set):
        print("[sweep] warning: --set without --name writes into the default run directory")
    cfg = load_config(sw["config"], overrides)
    train_cfg = load_config(sw["config"], overrides + list(sw.get("train_set", [])))
    if sw.get("train_set") and not sw.get("results_suffix"):
        raise SystemExit("train_set requires results_suffix (keeps transfer results separate)")
    run_dir = Path(sw.get("run_dir", f"runs/{cfg['name']}"))
    seeds, ratios = sw["seeds"], sw["ratios"]
    if args.seeds:
        seeds = [int(x) for x in str(args.seeds).split(",")]
    k, n_shards = map(int, args.shard.split("/"))
    device = get_device(args.device)
    common = ["--config", sw["config"], "--run-dir", str(run_dir), "--device", args.device, "--set", *overrides]

    def say(*a):
        print("[sweep]", *a, flush=True)

    t_start = time.time()

    def out_of_time():
        if args.max_hours is not None and (time.time() - t_start) / 3600 > args.max_hours:
            say(f"time budget of {args.max_hours} h reached; stopping cleanly (rerun to resume)")
            return True
        return False

    jobs = []
    for seed in seeds:
        # stage 1 + 2 (only shard 0 does these, other shards wait for files to appear)
        if not (run_paths(run_dir, seed)["score"] / "scores.npz").exists():
            if out_of_time():
                return
            say(f"score seed={seed}")
            if not args.dry_run and k == 0:
                score_cli.main(["--seed", str(seed), *common])
        if sw.get("rescore") and not (run_paths(run_dir, seed)["score"] / "scores_extra.npz").exists():
            if out_of_time():
                return
            say(f"rescore seed={seed}")
            if not args.dry_run and k == 0:
                rescore_cli.main(["--seed", str(seed), *common])
        embs = {m.get("emb", "self") for m in sw["methods"] if m["method"] in EMB_METHODS or m["method"] == "merge"}
        for kind in sorted(embs):
            if not embed_cli.embedding_path(run_dir, seed, kind).exists():
                if out_of_time():
                    return
                say(f"embed {kind} seed={seed}")
                if not args.dry_run and k == 0:
                    embed_cli.main(["--seed", str(seed), "--kind", kind, *common])
        for m in sw["methods"]:
            for r in m.get("ratios", ratios):  # per-method ratio list overrides the global one
                jobs.append((seed, m, r))

    results_dir = run_dir / (f"results_{sw['results_suffix']}" if sw.get("results_suffix") else "results")
    for j, (seed, m, r) in enumerate(jobs):
        if j % n_shards != k:
            continue
        opts = dict(m.get("opts", {}))
        tag = prune_cli.make_tag(m["method"], m.get("emb", "self"), opts, m.get("tag"))
        if m["method"] == "merge":
            tag = prune_cli.make_tag("merge", m.get("emb", "self"), opts, m.get("tag"))
        res = train_cli.result_path(results_dir, tag, r, seed, train_cfg["train"].get("budget", "epochs"))
        if res.exists():
            continue
        if out_of_time():
            return
        say(f"[{j + 1}/{len(jobs)}] {tag} r={r} seed={seed}")
        if args.dry_run:
            continue
        if m["method"] == "merge":
            sub = merge_cli.run(cfg, run_dir, seed, r, opts, m.get("emb", "self"), m.get("tag"), device=device)
        else:
            sub = prune_cli.run(run_dir, seed, m["method"], r, m.get("emb", "self"), opts, m.get("tag"),
                                device=str(device))
        train_cli.run(train_cfg, sub, seed, results_dir, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    say("done")


if __name__ == "__main__":
    main()
