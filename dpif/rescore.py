"""Add validation-DIRECTED influence scores to an existing scoring run (no retraining).

Size-based influence ||H^-1 g|| behaves like a hardness score (it is the gradient norm up to the
Hessian's conditioning) and was indistinguishable from the loss in tiers 2b/2c. Directed scores
ask a different question: does training on z make the model better or worse on held-out data?

Written to score/scores_extra.npz (scores.npz is never modified). Half A of the validation split
defines the direction; half B stays clean for judging methods that use these scores.

  val_removal_A[_e<k>]  last-layer influence-function estimate of the change in val-A loss when z
                        is removed (Koh & Liang 2017); > 0 means removing z hurts.
  tracin_val[_early]    full-network TracIn (Pruthi et al. 2020) towards val-A:
                        sum_t lr_t * <grad l(z, theta_t), grad L_valA(theta_t)> over saved checkpoints
                        (all, or epochs <= early_max). > 0 means training on z lowers val-A loss.
                        Each checkpoint costs one backward pass on val-A plus one forward-mode
                        derivative per training example (torch.func.jvp), so no Hessian is needed.

  python -m dpif.rescore --config C --seed S [--run-dir R] [--set ...]
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp

from .data import build_data, eval_loader
from .embed import load_scores
from .influence import lastlayer_influence
from .models import build_model
from .training import predict
from .utils import get_device, load_config, run_paths


def directional_losses(model, x, y, tangent: dict, params: dict, buffers: dict):
    """Per-example d/de l(z, theta + e*tangent) at e=0, i.e. <grad_theta l(z), tangent>."""
    def f(p):
        return F.cross_entropy(functional_call(model, (p, buffers), (x,)), y, reduction="none")
    _, d = jvp(f, (params,), (tangent,))
    return d


def lr_at_epoch(cfg_train: dict, epoch: int) -> float:
    """Learning rate at the end of `epoch` under the repo's cosine schedule (no warm-up)."""
    lr0, E = float(cfg_train["lr"]), float(cfg_train["epochs"])
    return 0.5 * lr0 * (1 + math.cos(math.pi * min(epoch, E) / E))


def tracin_val(model, data, ckpts: dict, cfg_train: dict, device, val_a, early_max=20, batch=256, log=print):
    pool = data.pool_idx
    total = torch.zeros(len(pool), dtype=torch.float64)
    early = torch.zeros(len(pool), dtype=torch.float64)
    for ep, path in sorted(ckpts.items()):
        t0 = time.time()
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.to(device).eval()
        model.zero_grad(set_to_none=True)
        for xb, yb, _ in eval_loader(data, "train", val_a, 512):
            (F.cross_entropy(model(xb), yb, reduction="sum") / len(val_a)).backward()
        params = {k: p.detach() for k, p in model.named_parameters()}
        tangent = {k: p.grad.detach().clone() for k, p in model.named_parameters()}
        buffers = {k: b for k, b in model.named_buffers()}
        d = torch.cat([directional_losses(model, xb, yb, tangent, params, buffers).double().cpu()
                       for xb, yb, _ in eval_loader(data, "train", pool, batch)])
        w = lr_at_epoch(cfg_train, ep)
        total += w * d
        if ep <= early_max:
            early += w * d
        log(f"  [rescore] tracin checkpoint epoch {ep}: lr {w:.4f}, mean <g_i,g_val> {d.mean():.3e} ({time.time() - t0:.0f}s)")
    return total.numpy(), early.numpy()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    run_dir = args.run_dir or f"runs/{cfg['name']}"
    sd = run_paths(run_dir, args.seed)["score"]
    out_file = sd / "scores_extra.npz"
    if out_file.exists() and not args.force:
        print(f"[rescore] exists: {out_file}")
        return
    device = get_device(args.device)
    data = build_data(cfg["data"], device)
    val_a, _ = data.val_halves
    assert len(val_a), "rescore needs a validation split (data.val_size > 0)"
    scores = load_scores(run_dir, args.seed)
    y = torch.as_tensor(scores["labels"])
    C = data.num_classes
    icfg = cfg.get("influence", {})
    kw = dict(damping=float(icfg.get("damping", 0.01)), mode=icfg.get("mode", "auto"),
              exact_max_params=int(icfg.get("exact_max_params", 12000)), device=device)
    extra = {}
    model = build_model(cfg["model"], C)

    # signed last-layer influence towards val-A, final model
    model.load_state_dict(torch.load(sd / "model.pt", map_location=device, weights_only=True))
    model.to(device)
    vl, vf, vy = predict(model, eval_loader(data, "train", val_a), device)
    r = lastlayer_influence(torch.as_tensor(scores["features"]), torch.as_tensor(scores["logits"]), y, C,
                            val=(vf, vl, vy), **kw)
    extra["val_removal_A"] = r["val_removal"]

    # same at each early snapshot (val features were saved with the snapshot)
    pos_a = np.searchsorted(data.val_idx, val_a)  # val_a's rows within the saved val arrays
    yv = data.y_train[torch.as_tensor(val_a, device=data.y_train.device)].cpu()
    st = sd / "train_state.npz"
    if st.exists():
        with np.load(st) as z:
            eps = sorted(int(k[4:].split("_", 1)[0]) for k in z.files if k.startswith("snap") and k.endswith("_val_features"))
            for ep in eps:
                r = lastlayer_influence(torch.as_tensor(scores[f"features_e{ep}"]), torch.as_tensor(scores[f"logits_e{ep}"]),
                                        y, C, val=(torch.as_tensor(z[f"snap{ep}_val_features"][pos_a]),
                                                   torch.as_tensor(z[f"snap{ep}_val_logits"][pos_a]), yv), **kw)
                extra[f"val_removal_A_e{ep}"] = r["val_removal"]

    # full-network TracIn towards val-A over saved checkpoints
    ckpts = {int(p.stem.split("_e")[1]): p for p in sd.glob("ckpt_e*.pt")}
    if ckpts:
        early_max = int(cfg.get("score", {}).get("tracin_early_max", 20))
        tv, te = tracin_val(build_model(cfg["model"], C), data, ckpts, cfg["train"], device, val_a, early_max)
        extra["tracin_val"] = tv
        extra["tracin_val_early"] = te
    else:
        print("[rescore] no checkpoints (score.checkpoint_epochs) -> TracIn scores skipped")
    np.savez_compressed(out_file, **extra)
    print(f"[rescore] wrote {sorted(extra)} -> {out_file}")


if __name__ == "__main__":
    main()
