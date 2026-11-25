"""Embeddings used to find near-duplicates. All arrays are aligned to pool order.

  self          penultimate features of the scoring model (no extra cost)
  self_e<k>     the same, from the snapshot at epoch k (needs score.snapshot_epochs)
  pixels        raw pixels downsampled to 16x16 (sanity baseline)
  ae            convolutional autoencoder trained on the whole pool (the thesis design, with the
                training-loop bug fixed: the original loop re-used the same 100 images)
  imagenet_r50  torchvision ResNet-50 (ImageNet-1k) pooled features
  dinov2_s14    DINOv2 ViT-S/14 CLS features (self-supervised; via torch.hub)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import IMAGENET_STATS, DataBundle, Loader, eval_loader

SEED_INDEPENDENT = {"pixels", "imagenet_r50", "dinov2_s14"}


@torch.no_grad()
def _extract(net, data: DataBundle, device, image_size, stats, bs=256):
    net.eval().to(device)
    out = []
    for xb, _, _ in eval_loader(data, "train", data.pool_idx, bs, mean=stats[0], std=stats[1], image_size=image_size):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out.append(net(xb).float().cpu())
    return torch.cat(out).numpy()


class ConvAE(nn.Module):
    def __init__(self, latent=256, size=32):
        super().__init__()
        self.s = size // 8
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(), nn.Flatten(), nn.Linear(128 * self.s * self.s, latent))
        self.dec_fc = nn.Linear(latent, 128 * self.s * self.s)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(), nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid())

    def forward(self, x):
        z = self.enc(x)
        return self.dec(self.dec_fc(z).view(-1, 128, self.s, self.s)), z


def fit_autoencoder(data: DataBundle, device, seed=0, latent=256, epochs=30, bs=256, lr=1e-3, log=print):
    """Train the conv autoencoder on the whole pool ([0,1] pixels). Returns the model."""
    torch.manual_seed(seed)
    size = (data.x_train.shape[-1] + 7) // 8 * 8
    ae = ConvAE(latent, size).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    loader = Loader(data.x_train, data.y_train, data.pool_idx, bs, shuffle=True, augment=False,
                    mean=(0.0,) * 3, std=(1.0,) * 3, image_size=size, seed=seed)
    for ep in range(epochs):
        tot = 0.0
        for xb, _, _ in loader:
            rec, _ = ae(xb)
            loss = F.mse_loss(rec, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            log(f"  AE epoch {ep + 1}: mse {tot / len(data.pool_idx):.5f}")
    ae.size = size
    return ae.eval()


def train_autoencoder(data: DataBundle, device, seed=0, log=print, **kw):
    ae = fit_autoencoder(data, device, seed=seed, log=log, **kw)

    class Enc(nn.Module):
        def forward(self, x):
            return ae.enc(x)
    return _extract(Enc(), data, device, ae.size, ((0.0,) * 3, (1.0,) * 3))


def compute_embedding(kind: str, data: DataBundle, device, scores: dict | None = None, seed=0, log=print, **kw):
    if kind == "self":
        return scores["features"].astype(np.float32)
    if kind.startswith("self_e"):  # features from an early-training snapshot of the scoring run
        return scores["features_" + kind.split("_", 1)[1]].astype(np.float32)
    if kind == "pixels":
        x = data.x_train[torch.as_tensor(data.pool_idx, device=data.x_train.device)].float() / 255
        return F.adaptive_avg_pool2d(x, 16).flatten(1).cpu().numpy()
    if kind == "ae":
        return train_autoencoder(data, device, seed=seed, log=log, **kw)
    if kind == "imagenet_r50":
        import torchvision
        m = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        m.fc = nn.Identity()
        return _extract(m, data, device, 224, IMAGENET_STATS)
    if kind == "dinov2_s14":
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        return _extract(m, data, device, 224, IMAGENET_STATS)
    raise ValueError(f"unknown embedding {kind!r}")
