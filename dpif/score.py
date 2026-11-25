"""Stage 1: train on the full pool once per seed and record every per-example score.

This run doubles as the full-data baseline (written to results/ as method "full").

  python -m dpif.score --config configs/cifar10_resnet18.yaml --seed 0
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import build_data, eval_loader
from .influence import lastlayer_influence
from .models import build_model
from .training import accuracy, predict, train_model
from .utils import get_device, load_config, run_paths, save_json, seed_everything


class ScoreHooks:
    """Forgetting events (Toneva et al.) on the augmented minibatches, EL2N at a fixed epoch."""

    def __init__(self, data, device, el2n_epoch: int, snapshot_epochs=(), checkpoint_epochs=(), ckpt_dir=None):
        n = len(data.x_train)
        self.checkpoint_epochs = set(int(e) for e in checkpoint_epochs)
        self.ckpt_dir = ckpt_dir
        self.snapshot_epochs = set(int(e) for e in snapshot_epochs)
        self.snapshots = {}  # epoch -> dict of pool/val logits+features (CPU tensors)
        self.prev = torch.zeros(n, dtype=torch.bool, device=device)
        self.ever = torch.zeros(n, dtype=torch.bool, device=device)
        self.forget = torch.zeros(n, dtype=torch.int32, device=device)
        self.data, self.device, self.el2n_epoch = data, device, el2n_epoch
        self.el2n = None

    def on_batch(self, idx, logits, y):
        correct = logits.argmax(1) == y
        self.forget[idx] += (self.prev[idx] & ~correct).int()
        self.prev[idx] = correct
        self.ever[idx] |= correct

    def on_epoch_end(self, epoch, model):
        if epoch in self.checkpoint_epochs and self.ckpt_dir is not None:
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, self.ckpt_dir / f"ckpt_e{epoch}.pt")
        if epoch in self.snapshot_epochs:
            # early-training view: features/logits before the network collapses each class
            lg, ft, _ = predict(model, eval_loader(self.data, "train", self.data.pool_idx), self.device)
            snap = {"logits": lg, "features": ft}
            if len(self.data.val_idx):
                vl, vf, _ = predict(model, eval_loader(self.data, "train", self.data.val_idx), self.device)
                snap.update(val_logits=vl, val_features=vf)
            self.snapshots[epoch] = snap
        if epoch == self.el2n_epoch:
            logits, _, y = predict(model, eval_loader(self.data, "train", self.data.pool_idx), self.device)
            p = torch.softmax(logits, 1)
            self.el2n = (p - F.one_hot(y, p.shape[1]).float()).norm(dim=1).numpy()

    def forgetting_scores(self, pool_idx):
        f = self.forget.float().cpu().numpy()
        never = ~self.ever.cpu().numpy()
        f[never] = f.max() + 1  # never-learned examples count as most forgettable
        return f[pool_idx]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides a.b=value")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    device = get_device(args.device)
    paths = run_paths(args.run_dir or f"runs/{cfg['name']}", args.seed)
    paths["score"].mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    data = build_data(cfg["data"], device)
    print(f"[score] {cfg['name']} seed={args.seed} pool={len(data.pool_idx)} val={len(data.val_idx)} device={device}")
    model = build_model(cfg["model"], data.num_classes)
    el2n_epoch = int(cfg.get("score", {}).get("el2n_epoch", 20))
    snapshot_epochs = [int(e) for e in cfg.get("score", {}).get("snapshot_epochs", [])]
    checkpoint_epochs = [int(e) for e in cfg.get("score", {}).get("checkpoint_epochs", [])]
    state_file = paths["score"] / "train_state.npz"
    model_file = paths["score"] / "model.pt"
    if state_file.exists() and model_file.exists():
        # an earlier attempt finished training but crashed later: reuse the trained model
        print(f"[score] resuming from saved model ({model_file}); skipping training")
        model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
        model.to(device)
        with np.load(state_file) as z:
            el2n, forgetting = z["el2n"], z["forgetting"]
            test_acc, val_acc, train_time = float(z["test_acc"]), float(z["val_acc"]), float(z["train_time"])
            snapshots = {}
            for k in z.files:
                if k.startswith("snap"):
                    ep, key = k[4:].split("_", 1)
                    snapshots.setdefault(int(ep), {})[key] = torch.from_numpy(z[k])
    else:
        hooks = ScoreHooks(data, device, el2n_epoch, snapshot_epochs, checkpoint_epochs, paths["score"])
        t0 = time.time()
        train_model(model, data, data.pool_idx, None, cfg["train"], device, args.seed, hooks=hooks)
        train_time = time.time() - t0
        test_acc = accuracy(model, data, "test", device)
        val_acc = accuracy(model, data, "val", device)
        el2n = hooks.el2n
        if el2n is None:
            print(f"[score] warning: el2n_epoch={el2n_epoch} not reached; using final-model EL2N")
            lg, _, yy = predict(model, eval_loader(data, "train", data.pool_idx), device)
            pp = torch.softmax(lg, 1)
            el2n = (pp - F.one_hot(yy, pp.shape[1]).float()).norm(dim=1).numpy()
        forgetting = hooks.forgetting_scores(data.pool_idx)
        snapshots = hooks.snapshots
        torch.save(model.state_dict(), model_file)
        snap_arrays = {f"snap{ep}_{k}": v.numpy() for ep, d in snapshots.items() for k, v in d.items()}
        np.savez(state_file, el2n=el2n, forgetting=forgetting, test_acc=test_acc, val_acc=val_acc,
                 train_time=train_time, **snap_arrays)
    print(f"[score] full-data test acc {test_acc:.4f}  val acc {val_acc:.4f}")

    t1 = time.time()
    logits, feats, y = predict(model, eval_loader(data, "train", data.pool_idx), device)
    loss = F.cross_entropy(logits, y, reduction="none").numpy()
    val = None
    if len(data.val_idx):
        vl, vf, vy = predict(model, eval_loader(data, "train", data.val_idx), device)
        val = (vf, vl, vy)
    icfg = cfg.get("influence", {})
    inf = lastlayer_influence(feats, logits, y, data.num_classes, damping=float(icfg.get("damping", 0.01)),
                              mode=icfg.get("mode", "auto"), exact_max_params=int(icfg.get("exact_max_params", 12000)),
                              val=val, device=device)
    score_time = time.time() - t1
    print(f"[score] influence mode={inf['_mode']} params={inf['_num_params']} "
          f"min_eig/lambda={inf['_min_eig_over_lambda']:.2e} ({score_time:.0f}s)")

    # same scores at each early snapshot, stored with an _e<epoch> suffix
    snap_out = {}
    for ep, d in sorted(snapshots.items()):
        sval = (d["val_features"], d["val_logits"], val[2]) if val is not None and "val_features" in d else None
        si = lastlayer_influence(d["features"], d["logits"], y, data.num_classes,
                                 damping=float(icfg.get("damping", 0.01)), mode=icfg.get("mode", "auto"),
                                 exact_max_params=int(icfg.get("exact_max_params", 12000)), val=sval, device=device)
        snap_out[f"features_e{ep}"] = d["features"].numpy()
        snap_out[f"logits_e{ep}"] = d["logits"].numpy()
        snap_out[f"loss_e{ep}"] = F.cross_entropy(d["logits"], y, reduction="none").numpy()
        for k, v in si.items():
            if not k.startswith("_"):
                snap_out[f"{k}_e{ep}"] = v
        print(f"[score] snapshot epoch {ep}: influence min_eig/lambda={si['_min_eig_over_lambda']:.2e}")

    arrays = dict(
        pool_idx=data.pool_idx, labels=y.numpy(), loss=loss, logits=logits.numpy(), el2n=el2n,
        forgetting=forgetting, features=feats.numpy(),
        **{k: v for k, v in inf.items() if not k.startswith("_")},
        **snap_out,
    )
    np.savez_compressed(paths["score"] / "scores.npz", **arrays)
    meta = dict(config=cfg, seed=args.seed, test_acc=test_acc, val_acc=val_acc, train_time_s=train_time,
                score_time_s=score_time, influence_mode=inf["_mode"], influence_params=inf["_num_params"],
                influence_min_eig_over_lambda=inf["_min_eig_over_lambda"], influence_numerics="float64-gram-v2")
    save_json(meta, paths["score"] / "meta.json")
    save_json(dict(method="full", tag="full", ratio=0.0, achieved_ratio=0.0, seed=args.seed, test_acc=test_acc,
                   val_acc=val_acc, val_a_acc=accuracy(model, data, "val_a", device),
                   val_b_acc=accuracy(model, data, "val_b", device), n_train=len(data.pool_idx), train_time_s=train_time),
              paths["results"] / f"full_r0.00_s{args.seed}.json")


if __name__ == "__main__":
    main()
