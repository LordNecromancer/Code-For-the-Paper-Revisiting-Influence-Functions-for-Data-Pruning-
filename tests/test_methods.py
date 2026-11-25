import numpy as np
import pytest

from dpif.methods import ALL_METHODS, select
from dpif.pairing import find_pairs


def _toy(n=300, C=3, d=8, seed=0):
    g = np.random.default_rng(seed)
    labels = g.integers(0, C, n)
    emb = g.standard_normal((n, d))
    # plant exact near-duplicate pairs: i and i+150 share class and embedding (+tiny noise)
    labels[150:] = labels[:150]
    emb[150:] = emb[:150] + 1e-4 * g.standard_normal((150, d))
    scores = dict(
        loss=g.random(n), el2n=g.random(n), forgetting=g.integers(0, 5, n).astype(float),
        param_norm=g.random(n), self=g.random(n), val_removal=g.standard_normal(n),
        features=g.standard_normal((n, d)), logits=g.standard_normal((n, C)),
    )
    return emb, labels, scores


def test_pairs_valid_and_find_planted_duplicates():
    emb, labels, _ = _toy()
    pairs, d = find_pairs(emb, labels, np.arange(len(labels)), k=5)
    flat = pairs.ravel()
    assert len(set(flat)) == len(flat), "a point appears in two pairs"
    assert np.all(labels[pairs[:, 0]] == labels[pairs[:, 1]]), "cross-class pair"
    assert np.all(np.diff(d) >= 0), "pairs not sorted by distance"
    planted = {(i, i + 150) for i in range(150)}
    found = {tuple(sorted(p)) for p in pairs[:150]}
    assert len(found & planted) == 150


@pytest.mark.parametrize("method", sorted(ALL_METHODS))
@pytest.mark.parametrize("ratio", [0.2, 0.45, 0.7])
def test_exact_ratio_and_valid_output(method, ratio):
    emb, labels, scores = _toy()
    keep, w, info = select(method, ratio, scores, labels, emb=emb, seed=1)
    n = len(labels)
    assert len(np.unique(keep)) == len(keep) and keep.min() >= 0 and keep.max() < n
    assert len(keep) == n - int(round(ratio * n)), (method, len(keep), info)
    assert len(w) == len(keep) and np.all(w > 0)


def test_reweight_conserves_mass_and_pairs_planted_first():
    emb, labels, scores = _toy()
    keep, w, _ = select("pair_influence_rw", 0.3, scores, labels, emb=emb, seed=0)
    assert np.isclose(w.sum(), len(labels))
    removed = np.setdiff1d(np.arange(len(labels)), keep)
    # at 30% (90 points) everything removed should come from a planted duplicate pair
    assert np.all([(r < 150 and r + 150 in keep) or (r >= 150 and r - 150 in keep) for r in removed])


def test_influence_rule_drops_lower_influence_member():
    emb, labels, scores = _toy()
    keep, _, _ = select("pair_influence", 0.2, scores, labels, emb=emb, seed=0)
    removed = np.setdiff1d(np.arange(len(labels)), keep)
    pi = scores["param_norm"]
    for r in removed:
        twin = r + 150 if r < 150 else r - 150
        assert pi[r] <= pi[twin]


def test_threshold_mode_original_rule():
    emb, labels, scores = _toy()
    keep, _, info = select("pair_influence", 0.9, scores, labels, emb=emb, seed=0,
                           opts=dict(sigma=1e-3, T=0.5))
    removed = np.setdiff1d(np.arange(len(labels)), keep)
    assert np.all(scores["param_norm"][removed] <= 0.5)
    assert len(removed) <= 150


def test_merge_scores_match_bruteforce_and_vanish_for_exact_duplicates():
    import torch
    from dpif.influence import _aug, exact_hessian, _damped_inverse, pair_merge_scores
    g = np.random.default_rng(0)
    n, d, C = 60, 4, 3
    feats = torch.tensor(g.standard_normal((n, d)), dtype=torch.float32)
    labels = torch.tensor(g.integers(0, C, n))
    feats[30:] = feats[:30]; labels[30:] = labels[:30]          # exact duplicates i <-> i+30
    W = torch.tensor(g.standard_normal((C, d + 1)), dtype=torch.float32)
    logits = _aug(feats) @ W.T
    pairs = np.array([[0, 30], [1, 31], [2, 5], [7, 11]])
    for mode in ["exact", "kfac"]:
        ms = pair_merge_scores(feats, logits, labels, C, pairs, mode=mode)
        assert np.allclose(ms[:2], 0, atol=1e-5), ms
        assert np.all(ms[2:] > 0)
    a = _aug(feats); p = torch.softmax(logits, 1)
    e = p - torch.nn.functional.one_hot(labels, C).float()
    Hinv = _damped_inverse(exact_hessian(a, p), 0.01).float()
    gvec = lambda i: (e[i][:, None] * a[i][None, :]).reshape(-1)
    ms = pair_merge_scores(feats, logits, labels, C, pairs, mode="exact")
    for k, (u, v) in enumerate(pairs):
        np.testing.assert_allclose(ms[k], (Hinv @ (gvec(u) - gvec(v))).norm().item(), rtol=1e-4)


def test_keep_easy_reverses_the_choice():
    emb, labels, scores = _toy()
    hard, _, _ = select("pair_influence", 0.2, scores, labels, emb=emb, seed=0)
    easy, _, _ = select("pair_influence", 0.2, scores, labels, emb=emb, seed=0, opts={"keep": "easy"})
    pi = scores["param_norm"]
    for r in np.setdiff1d(np.arange(len(labels)), easy):
        twin = r + 150 if r < 150 else r - 150
        assert pi[r] >= pi[twin]  # easy mode drops the harder twin
    assert len(hard) == len(easy) and not np.array_equal(hard, easy)


@pytest.mark.parametrize("method", ["random", "el2n", "ccs", "forgetting", "moderate", "influence_global"])
@pytest.mark.parametrize("ratio", [0.3, 0.7, 0.97])
def test_per_class_option_gives_exact_balanced_quotas(method, ratio):
    g = np.random.default_rng(3)
    n, C = 600, 6
    labels = np.repeat(np.arange(C), n // C)
    scores = dict(el2n=g.random(n), forgetting=g.random(n), param_norm=g.random(n), features=g.standard_normal((n, 4)),
                  loss=g.random(n))
    # make one class systematically "hard" so the global rule would be unbalanced
    scores["el2n"][labels == 0] += 1.0
    keep, w, info = select(method, ratio, scores, labels, seed=0, opts={"per_class": True})
    assert len(keep) == n - int(round(ratio * n)) and len(np.unique(keep)) == len(keep)
    counts = np.bincount(labels[keep], minlength=C)
    assert counts.max() - counts.min() <= 1, counts
