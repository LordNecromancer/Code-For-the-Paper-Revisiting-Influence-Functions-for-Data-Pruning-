"""Pruning methods. Every method maps (scores, embeddings, labels, ratio) -> kept pool positions
and per-example loss weights. All positions refer to the pool (see data.py).

Proposed (paper's Method 2, made exact-ratio and extended):
  pair_influence      pair near-duplicates; in each pair drop the lower-influence member,
                      only if its influence <= T (T = quantile `T_q` of the pool's scores)
  pair_influence_rw   same, but the kept twin inherits the dropped one's weight
                      ("remove-and-reweight": total mass is conserved)
  pair_valinf         in each pair drop the member whose removal raises val loss least
                      (signed, validation-directed influence; opts valinf_key, e.g. val_removal_A_e10)
  pair_loss + loss_key=tracin_val*   drop the member whose gradient is least aligned with the
                      validation gradient over checkpoints (full-network TracIn; from dpif.rescore)
  pair_merge_rw       remove-and-reweight, but pairs are processed in ascending order of the
                      first-order merge influence ||H^-1 (g_u - g_v)|| instead of embedding
                      distance (the quantity bounded in the paper's proposition)
Option keep=easy (pair_influence / pair_loss / pair_valinf) reverses the choice and keeps the EASIER
twin; Sorscher et al. (2022) predict that keeping easy examples wins once data becomes scarce.
Controls (isolate what the influence choice adds; same pairs, same order):
  pair_random         drop a random member (SemDeDup-style near-duplicate removal)
  pair_loss           drop the lower-loss member
Cluster generalisation (thesis future work: "subsets of similar data instead of pairs"):
  cluster_rw          per-class k-means with exactly n_keep clusters in total; keep one representative
                      per cluster (rep = influence | medoid | random), weighted by cluster size
  cluster             same, unweighted (isolates the effect of reweighting)
Ablation:
  influence_global    drop the globally lowest-influence points, no pairing
Baselines:
  random, el2n (Paul et al. 2021), forgetting (Toneva et al. 2019),
  moderate (Xia et al. 2023), ccs (Zheng et al. 2023)

Exact pruning ratio: pairs are processed in ascending distance (closest duplicates first),
so sigma is set implicitly by the ratio. Pairing can remove at most half of the alive points
per round, so higher ratios re-pair the survivors for up to `max_rounds` rounds.
The original absolute (sigma, T) rule is available via opts sigma=..., T=... .
"""
from __future__ import annotations

import warnings

import numpy as np

from .influence import pair_merge_scores
from .pairing import find_pairs

PAIR_METHODS = {"pair_influence", "pair_influence_rw", "pair_valinf", "pair_random", "pair_loss", "pair_merge_rw"}
CLUSTER_METHODS = {"cluster", "cluster_rw"}
SCORE_METHODS = {"random", "el2n", "forgetting", "influence_global", "moderate", "ccs"}
EMB_METHODS = PAIR_METHODS | CLUSTER_METHODS
ALL_METHODS = EMB_METHODS | SCORE_METHODS


def _keep_top(score: np.ndarray, n_keep: int, rng) -> np.ndarray:
    """Keep the n_keep highest scores; ties broken at random."""
    jitter = rng.random(len(score)) * 1e-9 * (np.nanstd(score) + 1e-12)
    return np.sort(np.argsort(-(score + jitter), kind="stable")[:n_keep])


def _ccs(score, labels, n_keep, rng, beta=0.0, bins=50):
    """Coverage-centric coreset selection (Zheng et al., ICLR 2023), Alg. 1. Higher = harder."""
    order = np.argsort(-score)
    n_hard = int(beta * len(score))
    cand = np.sort(order[n_hard:])
    s = score[cand]
    edges = np.linspace(s.min(), s.max() + 1e-12, bins + 1)
    strata = [cand[(s >= lo) & (s < hi)] for lo, hi in zip(edges[:-1], edges[1:])]
    strata = [x for x in strata if len(x)]
    keep, budget = [], n_keep
    while strata and budget > 0:
        strata.sort(key=len)
        st = strata.pop(0)
        take = min(len(st), budget // (len(strata) + 1) if strata else budget)
        keep.append(rng.choice(st, take, replace=False))
        budget -= take
    return np.sort(np.concatenate(keep)) if keep else np.zeros(0, int)


def _moderate(feats, labels, n_keep):
    """Moderate-DS (Xia et al., ICLR 2023): keep points whose distance to the class centre
    is closest to the per-class median."""
    dev = np.empty(len(labels))
    for c in np.unique(labels):
        m = labels == c
        d = np.linalg.norm(feats[m] - feats[m].mean(0), axis=1)
        dev[m] = np.abs(d - np.median(d))
    return np.sort(np.argsort(dev, kind="stable")[:n_keep])


def _kmeans_cos(X, k, iters=25, seed=0):
    """Spherical k-means on L2-normalised rows of X (torch). Returns cluster assignment."""
    import torch
    n = len(X)
    if k >= n:
        return torch.arange(n, device=X.device)
    g = torch.Generator().manual_seed(seed)
    C = X[torch.randperm(n, generator=g)[:k].to(X.device)].clone()
    for _ in range(iters):
        assign = (X @ C.T).argmax(1)
        newC = torch.zeros_like(C).index_add_(0, assign, X)
        counts = torch.bincount(assign, minlength=k)
        empty = counts == 0
        if empty.any():  # re-seed empty clusters with the points worst served by their centroid
            fit = (X * C[assign]).sum(1)
            newC[empty] = X[fit.argsort()[: int(empty.sum())]]
        C = torch.nn.functional.normalize(newC, dim=1)
    return (X @ C.T).argmax(1)


def _cluster_select(n_keep, emb, labels, scores, rng, opts):
    import torch
    dev = opts.get("device", "cpu")
    rep = opts.get("rep", "influence")
    infl = scores[opts.get("influence_key", "param_norm")]
    n = len(labels)
    classes, counts = np.unique(labels, return_counts=True)
    quota = counts * n_keep / n  # largest-remainder apportionment -> exactly n_keep in total
    kc = np.floor(quota).astype(int)
    kc[np.argsort(-(quota - kc))[: n_keep - kc.sum()]] += 1
    kc = np.clip(kc, 1, counts)
    keep, wts = [], []
    for c, k in zip(classes, kc):
        pos = np.flatnonzero(labels == c)
        X = torch.nn.functional.normalize(torch.as_tensor(emb[pos], dtype=torch.float32, device=dev), dim=1)
        assign_t = _kmeans_cos(X, int(k), int(opts.get("iters", 25)), int(rng.integers(1 << 31)))
        assign = assign_t.cpu().numpy()
        if rep == "medoid":
            cent = torch.zeros(int(assign.max()) + 1, X.shape[1], device=dev).index_add_(0, assign_t, X)
            crit = (X * torch.nn.functional.normalize(cent, dim=1)[assign_t]).sum(1).cpu().numpy()
        elif rep == "influence":
            crit = infl[pos]
        else:
            crit = rng.random(len(pos))
        size = np.bincount(assign)
        order = np.lexsort((-crit, assign))  # by cluster, best first
        rank = np.empty(len(pos), int)
        starts = np.r_[0, np.cumsum(np.bincount(assign[order]))[:-1]]
        rank[order] = np.arange(len(pos)) - np.repeat(starts, np.bincount(assign[order]))[: len(pos)]
        # one rep per non-empty cluster first, then extra reps from the largest clusters
        pick = np.lexsort((-crit, -size[assign], rank))[:k]
        n_in = np.bincount(assign[pick], minlength=len(size))
        keep.append(pos[pick])
        wts.append(size[assign[pick]] / n_in[assign[pick]])
    keep, w = np.concatenate(keep), np.concatenate(wts).astype(float)
    if opts.get("max_weight") is not None:
        w = np.minimum(w, float(opts["max_weight"]))
    o = np.argsort(keep)
    return keep[o], w[o]


def _pair_prune(method, n_remove, emb, labels, scores, rng, opts):
    n = len(labels)
    infl = scores[opts.get("influence_key", "param_norm")]
    thr = np.quantile(infl, float(opts.get("T_q", 1.0)))
    sigma = opts.get("sigma")
    if opts.get("T") is not None:
        thr = float(opts["T"])
    alive = np.ones(n, bool)
    weight = np.ones(n)
    removed, rounds = 0, 0
    reweight = method in ("pair_influence_rw", "pair_merge_rw") or bool(opts.get("reweight", False))
    cache = {}
    for rounds in range(1, int(opts.get("max_rounds", 10)) + 1):
        pairs, dists = find_pairs(emb, labels, np.flatnonzero(alive), k=int(opts.get("k", 10)),
                                  metric=opts.get("metric", "cosine"), device=opts.get("device", "cpu"))
        if method == "pair_merge_rw" and len(pairs):
            import torch
            ms = pair_merge_scores(torch.as_tensor(scores["features"]), torch.as_tensor(scores["logits"]),
                                   torch.as_tensor(labels), scores["logits"].shape[1], pairs,
                                   damping=float(opts.get("damping", 0.01)), device=opts.get("device", "cpu"),
                                   _cache=cache)
            order = np.argsort(ms, kind="stable")
            pairs, dists = pairs[order], dists[order]
        progress = False
        easy = opts.get("keep", "hard") == "easy"  # Sorscher et al. 2022: keep easy when data is scarce
        for (u, v), dist in zip(pairs, dists):
            if removed >= n_remove or (sigma is not None and dist > float(sigma)):
                break
            if method == "pair_random":
                victim = u if rng.random() < 0.5 else v
            elif method == "pair_loss":
                ls = scores[opts.get("loss_key", "loss")]
                victim = u if (ls[u] <= ls[v]) != easy else v
            elif method == "pair_valinf":
                vr = scores[opts.get("valinf_key", "val_removal")]
                victim = u if (vr[u] <= vr[v]) != easy else v
            else:  # pair_influence(_rw), pair_merge_rw
                victim = u if (infl[u] <= infl[v]) != easy else v
                if not easy and infl[victim] > thr:
                    continue
            twin = v if victim == u else u
            alive[victim] = False
            if reweight:
                weight[twin] += weight[victim]
            removed += 1
            progress = True
        if removed >= n_remove or not progress or sigma is not None:
            break
    keep = np.flatnonzero(alive)
    return keep, (weight[keep] if reweight else np.ones(len(keep))), {"rounds": rounds}


def _per_class_quotas(labels, n_keep):
    """Largest-remainder apportionment of n_keep over classes, proportional to class size."""
    classes, counts = np.unique(labels, return_counts=True)
    quota = counts * n_keep / len(labels)
    k = np.floor(quota).astype(int)
    k[np.argsort(-(quota - k))[: n_keep - k.sum()]] += 1
    return classes, np.minimum(k, counts)


def select(method: str, ratio: float, scores: dict, labels: np.ndarray, emb=None, seed: int = 0,
           opts: dict | None = None):
    """Returns (keep_positions, weights, info). `ratio` = fraction of the pool to remove."""
    opts = dict(opts or {})
    rng = np.random.default_rng(seed)
    n = len(labels)
    n_remove = int(round(ratio * n))
    n_keep = n - n_remove
    info = {}
    if opts.get("per_class") and method in SCORE_METHODS:
        # class-balanced variant: run the same score-based rule inside each class with a fixed quota
        sub_opts = {k: v for k, v in opts.items() if k != "per_class"}
        classes, quotas = _per_class_quotas(labels, n_keep)
        keep = []
        for c, q in zip(classes, quotas):
            pos = np.flatnonzero(labels == c)
            sub_scores = {k: (v[pos] if isinstance(v, np.ndarray) and len(v) == n else v) for k, v in scores.items()}
            k_c, _, _ = select(method, 1 - q / len(pos), sub_scores, labels[pos], None, int(rng.integers(1 << 31)), sub_opts)
            if len(k_c) != q:  # rounding inside select can differ by one; fix to the exact quota
                k_c = k_c[:q] if len(k_c) > q else np.union1d(k_c, rng.choice(np.setdiff1d(np.arange(len(pos)), k_c), q - len(k_c), replace=False))
            keep.append(pos[k_c])
        keep = np.sort(np.concatenate(keep))
        info["achieved_ratio"] = 1 - len(keep) / n
        info["per_class"] = True
        return keep, np.ones(len(keep)), info
    if method == "random":
        keep = np.sort(rng.choice(n, n_keep, replace=False))
    elif method == "el2n":
        keep = _keep_top(scores["el2n"], n_keep, rng)
    elif method == "forgetting":
        keep = _keep_top(scores["forgetting"], n_keep, rng)
    elif method == "influence_global":
        keep = _keep_top(scores[opts.get("influence_key", "param_norm")], n_keep, rng)
    elif method == "moderate":
        keep = _moderate(scores["features"], labels, n_keep)
    elif method == "ccs":
        beta = opts.get("beta")
        if beta is None:  # rough default schedule; tune on the validation split
            beta = 0.0 if ratio <= 0.5 else (0.1 if ratio <= 0.7 else 0.3)
        keep = _ccs(scores[opts.get("ccs_score", "el2n")], labels, n_keep, rng, float(beta), int(opts.get("bins", 50)))
        info["beta"] = beta
    elif method in CLUSTER_METHODS:
        if emb is None:
            raise ValueError(f"{method} needs embeddings")
        keep, w = _cluster_select(n_keep, emb, labels, scores, rng, opts)
        if method == "cluster":
            w = np.ones(len(keep))
    elif method in PAIR_METHODS:
        if emb is None:
            raise ValueError(f"{method} needs embeddings")
        keep, w, info = _pair_prune(method, n_remove, emb, labels, scores, rng, opts)
        if len(keep) > n_keep and opts.get("sigma") is None:
            warnings.warn(f"{method}: reached only {1 - len(keep) / n:.3f} of requested ratio {ratio}")
        info["achieved_ratio"] = 1 - len(keep) / n
        return keep, w, info
    else:
        raise ValueError(f"unknown method {method!r}")
    info["achieved_ratio"] = 1 - len(keep) / n
    return keep, (w if method in CLUSTER_METHODS else np.ones(len(keep))), info
