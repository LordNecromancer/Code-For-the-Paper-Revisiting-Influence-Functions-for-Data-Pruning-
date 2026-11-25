"""Near-duplicate pairing inside each class.

Replaces the original O(n^2) "first match under sigma" loop with:
  1. exact k-nearest neighbours within each class (blocked matmul on the device), then
  2. greedy matching over candidate edges in ascending distance, so every point is in at
     most one pair and the closest pairs are formed first (a 1/2-approximation to the
     optimal min-distance matching).
Distances: cosine distance (1 - cos, in [0, 2]) on L2-normalised embeddings by default,
which makes sigma comparable across embedding types; 'l2' is also available.
"""
from __future__ import annotations

import numpy as np
import torch


def knn_edges(emb: np.ndarray, labels: np.ndarray, positions: np.ndarray, k: int = 10,
              metric: str = "cosine", device="cpu", block: int = 4096):
    """Candidate edges (i, j, dist) among `positions`, within class. i, j are positions."""
    device = torch.device(device)
    I, J, Dd = [], [], []
    for c in np.unique(labels[positions]):
        pos = positions[labels[positions] == c]
        if len(pos) < 2:
            continue
        X = torch.as_tensor(emb[pos], dtype=torch.float32, device=device)
        if metric == "cosine":
            X = torch.nn.functional.normalize(X, dim=1)
        sq = (X * X).sum(1)
        kk = min(k, len(pos) - 1)
        for s in range(0, len(pos), block):
            xb = X[s:s + block]
            sim = xb @ X.T
            if metric == "cosine":
                dist = 1.0 - sim
            else:
                dist = (sq[s:s + block, None] + sq[None, :] - 2 * sim).clamp_min(0).sqrt()
            r = torch.arange(len(xb), device=device)
            dist[r, r + s] = float("inf")
            d, j = dist.topk(kk, dim=1, largest=False)
            I.append(np.repeat(pos[s:s + len(xb)], kk))
            J.append(pos[j.cpu().numpy().ravel()])
            Dd.append(d.cpu().numpy().ravel())
    if not I:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    return np.concatenate(I), np.concatenate(J), np.concatenate(Dd)


def greedy_matching(i: np.ndarray, j: np.ndarray, d: np.ndarray, n: int):
    """Greedy min-distance matching. Returns pairs (m, 2) and distances (m,), ascending."""
    lo, hi = np.minimum(i, j), np.maximum(i, j)
    key = lo.astype(np.int64) * n + hi
    order = np.lexsort((d, key))  # by key, then distance
    key_s = key[order]
    first = np.ones(len(order), bool)
    first[1:] = key_s[1:] != key_s[:-1]
    order = order[first]
    order = order[np.argsort(d[order], kind="stable")]
    matched = np.zeros(n, bool)
    pairs, dists = [], []
    for e in order:
        u, v = lo[e], hi[e]
        if not matched[u] and not matched[v]:
            matched[u] = matched[v] = True
            pairs.append((u, v)); dists.append(d[e])
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2), np.asarray(dists, dtype=np.float64)


def find_pairs(emb, labels, positions, k=10, metric="cosine", device="cpu"):
    i, j, d = knn_edges(emb, labels, positions, k=k, metric=metric, device=device)
    return greedy_matching(i, j, d, len(labels))
