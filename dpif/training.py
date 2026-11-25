"""Training and inference. Identical for every method, so comparisons are apples to apples."""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import DataBundle, Loader, eval_loader


def _optimizer(model, cfg):
    name = cfg.get("optimizer", "sgd").lower()
    lr, wd = float(cfg["lr"]), float(cfg.get("weight_decay", 0.0))
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=float(cfg.get("momentum", 0.9)),
                               nesterov=bool(cfg.get("nesterov", True)), weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(name)


def use_channels_last(device) -> bool:
    """channels_last only pays off with tensor cores (compute capability >= 7.0, e.g. T4/V100/
    RTX 20xx+). On Pascal cards such as the GTX 1080 it made training 4-10x slower."""
    return device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 7


def total_steps(n_subset: int, n_pool: int, cfg: dict) -> int:
    """`budget: epochs` -> same number of epochs over the subset (fewer steps; standard in the
    pruning literature). `budget: iterations` -> same number of steps as full-data training."""
    bs, ep = int(cfg["batch_size"]), float(cfg["epochs"])
    n = n_pool if cfg.get("budget", "epochs") == "iterations" else n_subset
    return max(1, int(round(ep * math.ceil(n / bs))))


def train_model(model, data: DataBundle, indices, weights, cfg: dict, device, seed: int, hooks=None, log=print):
    """Train `model` on global `indices`; `weights` (same order) are per-example loss weights."""
    indices = np.asarray(indices)
    model.to(device)
    cl = use_channels_last(device)
    if cl:
        model = model.to(memory_format=torch.channels_last)
    bs = int(cfg["batch_size"])
    steps = total_steps(len(indices), len(data.pool_idx), cfg)
    warm = int(float(cfg.get("warmup_epochs", 0)) * math.ceil(len(indices) / bs))

    # per-example weight lookup by global index, normalised to mean 1 over the subset
    wfull = torch.ones(len(data.x_train), device=device)
    if weights is not None:
        w = torch.as_tensor(np.asarray(weights, dtype=np.float32), device=device)
        wfull[torch.as_tensor(indices, device=device)] = w / w.mean()

    opt = _optimizer(model, cfg)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loader = Loader(data.x_train, data.y_train, indices, bs, shuffle=True, augment=data.augment,
                    mean=data.mean, std=data.std, image_size=data.image_size, seed=seed)

    step, epoch, t0 = 0, 0, time.time()
    while step < steps:
        model.train()
        run_loss, run_n = 0.0, 0
        for xb, yb, ib in loader:
            if cl:
                xb = xb.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(xb)
            lvec = F.cross_entropy(logits.float(), yb, reduction="none")
            loss = (lvec * wfull[ib]).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if hooks is not None:
                hooks.on_batch(ib, logits.detach(), yb)
            run_loss += loss.item() * len(yb)
            run_n += len(yb)
            step += 1
            if step >= steps:
                break
        epoch += 1
        if hooks is not None:
            hooks.on_epoch_end(epoch, model)
        if epoch % max(1, int(cfg.get("log_every", 10))) == 0 or step >= steps:
            log(f"  epoch {epoch} step {step}/{steps} loss {run_loss / max(1, run_n):.4f} ({time.time() - t0:.0f}s)")
    return model


@torch.no_grad()
def predict(model, loader, device):
    """Returns (logits, features, labels) as float32 CPU tensors, in loader order."""
    model.eval()
    cl = use_channels_last(device)
    L, H, Y = [], [], []
    for xb, yb, _ in loader:
        if cl:
            xb = xb.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out, h = model(xb, return_features=True)
        L.append(out.float().cpu()); H.append(h.float().cpu()); Y.append(yb.cpu())
    return torch.cat(L), torch.cat(H), torch.cat(Y)


def accuracy(model, data: DataBundle, split: str, device, indices=None, bs=1024) -> float:
    if split.startswith("val") and len(data.val_idx) == 0:
        return float("nan")
    if split == "val":
        idx = data.val_idx
    elif split in ("val_a", "val_b"):
        idx = data.val_halves[0 if split == "val_a" else 1]
    else:
        idx = indices
    logits, _, y = predict(model, eval_loader(data, "test" if split == "test" else "train", idx, bs), device)
    return (logits.argmax(1) == y).float().mean().item()
