"""Classifiers = backbone (-> penultimate features) + a single linear head `fc`.

Influence is computed on `fc` only (last-layer influence, as in the original thesis and in
Yang et al., ICLR 2023), so every model must expose `embed(x)` and `fc`.
Note: the original TensorFlow code used Dense(32) -> Dense(C). Here the head is one linear
layer on the backbone features, which makes the last-layer Hessian exact and cheap.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short = nn.Sequential()
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.short(x))


class ResNetCIFAR(nn.Module):
    """Standard CIFAR ResNet (3x3 stem, no max-pool). blocks=(2,2,2,2), width=64 -> ResNet-18."""

    def __init__(self, blocks=(2, 2, 2, 2), width=64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        layers, cin = [], width
        for i, nb in enumerate(blocks):
            cout = width * (2 ** i)
            for j in range(nb):
                layers.append(BasicBlock(cin, cout, stride=2 if (j == 0 and i > 0) else 1))
                cin = cout
        self.layers = nn.Sequential(*layers)
        self.out_dim = cin

    def forward(self, x):
        x = self.layers(F.relu(self.bn1(self.conv1(x))))
        return F.adaptive_avg_pool2d(x, 1).flatten(1)


class VGGCIFAR(nn.Module):
    """VGG-11 with BatchNorm for 32x32 inputs (cross-architecture transfer test)."""
    CFG = (64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M")

    def __init__(self, width_mult=1.0):
        super().__init__()
        layers, cin = [], 3
        for v in self.CFG:
            if v == "M":
                layers.append(nn.MaxPool2d(2))
            else:
                c = int(v * width_mult)
                layers += [nn.Conv2d(cin, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
                cin = c
        self.features = nn.Sequential(*layers)
        self.out_dim = cin

    def forward(self, x):
        return F.adaptive_avg_pool2d(self.features(x), 1).flatten(1)


class Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(feat_dim, num_classes)

    def embed(self, x):
        return self.backbone(x)

    def forward(self, x, return_features: bool = False):
        h = self.embed(x)
        out = self.fc(h)
        return (out, h) if return_features else out


def build_model(cfg: dict, num_classes: int) -> Classifier:
    arch = cfg.get("arch", "resnet18_cifar")
    if arch == "resnet18_cifar":
        bb = ResNetCIFAR(tuple(cfg.get("blocks", (2, 2, 2, 2))), int(cfg.get("width", 64)))
        return Classifier(bb, bb.out_dim, num_classes)
    if arch == "vgg11_cifar":
        bb = VGGCIFAR(float(cfg.get("width_mult", 1.0)))
        return Classifier(bb, bb.out_dim, num_classes)
    if arch in ("resnet18", "resnet50"):
        import torchvision

        weights = None
        if cfg.get("pretrained", False):
            weights = "IMAGENET1K_V1" if arch == "resnet18" else "IMAGENET1K_V2"
        m = getattr(torchvision.models, arch)(weights=weights)
        d = m.fc.in_features
        m.fc = nn.Identity()
        return Classifier(m, d, num_classes)
    raise ValueError(f"unknown arch {arch!r}")
