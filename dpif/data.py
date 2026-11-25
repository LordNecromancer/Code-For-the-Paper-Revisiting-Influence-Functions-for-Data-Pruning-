"""Datasets held entirely on the device as uint8 tensors, with GPU-side augmentation.

CIFAR-sized datasets fit easily in GPU memory (CIFAR-10 train = 150 MB as uint8), and
doing crop/flip/resize on the GPU removes the CPU data-loading bottleneck, which matters
when the study needs hundreds of training runs.

Index conventions used throughout the repo
------------------------------------------
* "global index": row in x_train (0..N-1). Subset files store global indices.
* "pool": the training indices available for pruning (train minus the held-out val split).
  All per-example score arrays are aligned to `pool_idx` order ("pool position").
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "mnist": ((0.1307,) * 3, (0.3081,) * 3),
    "synthetic": ((0.5,) * 3, (0.25,) * 3),
}
IMAGENET_STATS = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


@dataclass
class DataBundle:
    x_train: torch.Tensor  # uint8 (N, 3, H, W), on device
    y_train: torch.Tensor  # int64 (N,), on device
    x_test: torch.Tensor
    y_test: torch.Tensor
    num_classes: int
    pool_idx: np.ndarray  # global indices available for pruning (sorted)
    val_idx: np.ndarray  # global indices held out for model selection (sorted)
    mean: tuple
    std: tuple
    image_size: int | None  # None = native resolution
    augment: bool
    name: str

    @property
    def val_halves(self):
        """(val_a, val_b): disjoint class-balanced halves of val_idx. A is used to compute
        validation-directed influence; B is then the uncontaminated set for judging those methods."""
        if not hasattr(self, "_val_halves"):
            y = self.y_train[torch.as_tensor(self.val_idx, device=self.y_train.device)].cpu().numpy()
            g = np.random.default_rng(12345)
            a = []
            for c in np.unique(y):
                pos = np.flatnonzero(y == c)
                a.append(g.choice(pos, len(pos) // 2, replace=False))
            a = np.sort(np.concatenate(a)) if a else np.array([], dtype=np.int64)
            b = np.setdiff1d(np.arange(len(self.val_idx)), a)
            self._val_halves = (self.val_idx[a], self.val_idx[b])
        return self._val_halves

    @property
    def pool_labels(self) -> np.ndarray:
        return self.y_train[torch.as_tensor(self.pool_idx, device=self.y_train.device)].cpu().numpy()


# --------------------------------------------------------------------------- loading
def _make_synthetic(n_train=600, n_test=200, num_classes=4, size=32, dup_frac=0.3, seed=0):
    """Learnable toy data with planted near-duplicates (for smoke tests only)."""
    g = np.random.default_rng(seed)
    protos = g.random((num_classes, 3, size, size))

    def sample(n):
        y = g.integers(0, num_classes, n)
        x = 0.55 * protos[y] + 0.45 * g.random((n, 3, size, size))
        return x, y

    xtr, ytr = sample(n_train)
    n_dup = int(dup_frac * n_train)
    src = g.choice(n_train - n_dup, n_dup, replace=False)
    xtr[n_train - n_dup:] = np.clip(xtr[src] + 0.01 * g.standard_normal(xtr[src].shape), 0, 1)
    ytr[n_train - n_dup:] = ytr[src]
    xte, yte = sample(n_test)
    to_u8 = lambda a: torch.from_numpy((a * 255).round().astype(np.uint8))
    return to_u8(xtr), torch.from_numpy(ytr), to_u8(xte), torch.from_numpy(yte), num_classes


def _load_raw(name: str, root: str):
    if name == "synthetic":
        return _make_synthetic()
    import torchvision

    if name in ("cifar10", "cifar100"):
        cls = torchvision.datasets.CIFAR10 if name == "cifar10" else torchvision.datasets.CIFAR100
        tr, te = cls(root, train=True, download=True), cls(root, train=False, download=True)
        f = lambda d: torch.from_numpy(d.data).permute(0, 3, 1, 2).contiguous()
        return f(tr), torch.tensor(tr.targets), f(te), torch.tensor(te.targets), (10 if name == "cifar10" else 100)
    if name == "mnist":
        tr = torchvision.datasets.MNIST(root, train=True, download=True)
        te = torchvision.datasets.MNIST(root, train=False, download=True)
        f = lambda d: d.data.unsqueeze(1).repeat(1, 3, 1, 1).contiguous()
        return f(tr), tr.targets.clone(), f(te), te.targets.clone(), 10
    raise ValueError(f"unknown dataset {name!r}")


def stratified_split(labels: np.ndarray, val_size: int, seed: int):
    """Class-balanced held-out validation split. Returns (pool_idx, val_idx), both sorted."""
    n = len(labels)
    if val_size <= 0:
        return np.arange(n), np.array([], dtype=np.int64)
    g = np.random.default_rng(seed)
    classes = np.unique(labels)
    per = val_size // len(classes)
    val = []
    for c in classes:
        idx = np.flatnonzero(labels == c)
        val.append(g.choice(idx, min(per, len(idx)), replace=False))
    val = np.sort(np.concatenate(val))
    pool = np.setdiff1d(np.arange(n), val)
    return pool, val


def build_data(cfg: dict, device: torch.device) -> DataBundle:
    name = cfg["dataset"]
    xtr, ytr, xte, yte, nc = _load_raw(name, cfg.get("root", "./data"))
    pool, val = stratified_split(ytr.numpy(), int(cfg.get("val_size", 0)), int(cfg.get("split_seed", 0)))
    mean, std = IMAGENET_STATS if cfg.get("normalize", "dataset") == "imagenet" else STATS[name]
    return DataBundle(
        x_train=xtr.to(device), y_train=ytr.long().to(device),
        x_test=xte.to(device), y_test=yte.long().to(device),
        num_classes=nc, pool_idx=pool, val_idx=val, mean=mean, std=std,
        image_size=cfg.get("image_size"), augment=bool(cfg.get("augment", True)), name=name,
    )


# --------------------------------------------------------------------------- augmentation
def random_crop(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Per-sample random crop with zero padding, fully on device."""
    B, C, H, W = x.shape
    xp = F.pad(x, (pad, pad, pad, pad))
    oy = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    ox = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    rows = (oy[:, None] + torch.arange(H, device=x.device))[:, None, :, None].expand(B, C, H, W + 2 * pad)
    xp = xp.gather(2, rows)
    cols = (ox[:, None] + torch.arange(W, device=x.device))[:, None, None, :].expand(B, C, H, W)
    return xp.gather(3, cols)


def random_flip(x: torch.Tensor) -> torch.Tensor:
    m = torch.rand(x.shape[0], device=x.device) < 0.5
    x = x.clone()
    x[m] = x[m].flip(3)
    return x


def preprocess(xb_u8, *, augment, mean, std, image_size):
    x = xb_u8.float().div_(255.0)
    if augment:
        x = random_flip(random_crop(x, pad=max(1, x.shape[-1] // 8)))
    if image_size is not None and image_size != x.shape[-1]:
        x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
    m = torch.tensor(mean, device=x.device).view(1, -1, 1, 1)
    s = torch.tensor(std, device=x.device).view(1, -1, 1, 1)
    return (x - m) / s


class Loader:
    """Iterates over `indices` (global) of an on-device uint8 tensor. Yields (x, y, idx)."""

    def __init__(self, x, y, indices, batch_size, *, shuffle, augment, mean, std, image_size, seed=0):
        self.x, self.y = x, y
        self.indices = torch.as_tensor(np.asarray(indices), dtype=torch.long)
        self.bs, self.shuffle, self.augment = batch_size, shuffle, augment
        self.mean, self.std, self.image_size = mean, std, image_size
        self.g = torch.Generator().manual_seed(seed)

    def __len__(self):
        return math.ceil(len(self.indices) / self.bs)

    def __iter__(self):
        idx = self.indices
        if self.shuffle:
            idx = idx[torch.randperm(len(idx), generator=self.g)]
        for s in range(0, len(idx), self.bs):
            b = idx[s:s + self.bs].to(self.x.device)
            xb = preprocess(self.x[b], augment=self.augment, mean=self.mean, std=self.std, image_size=self.image_size)
            yield xb, self.y[b], b


def eval_loader(data: DataBundle, split: str, indices=None, batch_size=1024, mean=None, std=None, image_size="data"):
    x, y = (data.x_test, data.y_test) if split == "test" else (data.x_train, data.y_train)
    if indices is None:
        indices = np.arange(len(x))
    size = data.image_size if image_size == "data" else image_size
    if size and size >= 128:
        # Local edit (Claude, 2026-11-21): batch 1024 at 224 px does not fit in 8 GB (GTX 1080 OOM).
        # Eval mode only (BatchNorm uses running stats), so outputs do not depend on batch size.
        batch_size = min(batch_size, 128)
    return Loader(
        x, y, indices, batch_size, shuffle=False, augment=False,
        mean=mean or data.mean, std=std or data.std,
        image_size=size,
    )
