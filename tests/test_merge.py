import numpy as np
import torch

from dpif.influence import _aug, hessian_factor, hinv_norm_of_combination, pair_merge_scores
from dpif.merge import featmatch, hybrid, shared_crop_flip
from dpif.methods import select


def test_hybrid_identity_and_range():
    x = torch.rand(4, 3, 16, 16)
    assert torch.allclose(hybrid(x, x), x, atol=1e-5)
    y = hybrid(x, torch.rand(4, 3, 16, 16))
    assert y.min() >= 0 and y.max() <= 1


def test_shared_augmentation_is_shared():
    a = torch.rand(8, 3, 16, 16)
    xa, ya = shared_crop_flip([a, a.clone()], pad=2)
    assert torch.equal(xa, ya) and xa.shape == a.shape


def test_general_merge_norm_matches_pair_scores():
    g = np.random.default_rng(0)
    n, d, C = 80, 5, 3
    feats = torch.tensor(g.standard_normal((n, d)), dtype=torch.float32)
    labels = torch.tensor(g.integers(0, C, n))
    logits = torch.tensor(g.standard_normal((n, C)), dtype=torch.float32)
    pairs = np.array([[0, 1], [5, 9], [20, 33]])
    e = torch.softmax(logits, 1) - torch.nn.functional.one_hot(labels, C).float()
    a = _aug(feats)
    u, v = torch.as_tensor(pairs[:, 0]), torch.as_tensor(pairs[:, 1])
    one = torch.ones(len(pairs))
    for mode in ["exact", "kfac"]:
        ref = pair_merge_scores(feats, logits, labels, C, pairs, mode=mode)
        fac = hessian_factor(feats, logits, labels, C, mode=mode)
        got = hinv_norm_of_combination(fac, [(one, e[u], a[u]), (-one, e[v], a[v])]).numpy()
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-6)
        # "keep u with weight 2" == 2 g_u - g_u - g_v == g_u - g_v
        got2 = hinv_norm_of_combination(fac, [(2 * one, e[u], a[u]), (-one, e[u], a[u]), (-one, e[v], a[v])]).numpy()
        np.testing.assert_allclose(got2, ref, rtol=1e-4, atol=1e-6)


def test_featmatch_reduces_feature_gap():
    from dpif.models import build_model

    class D:  # minimal stand-in for DataBundle
        image_size, mean, std = None, (0.5,) * 3, (0.25,) * 3

    torch.manual_seed(0)
    model = build_model({"arch": "resnet18_cifar", "blocks": [1, 1, 1, 1], "width": 8}, 4).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    xu, xv = torch.rand(6, 3, 16, 16), torch.rand(6, 3, 16, 16)
    norm = lambda x: (x - 0.5) / 0.25
    with torch.no_grad():
        target = (model.embed(norm(xu)) + model.embed(norm(xv))) / 2
        gap0 = (model.embed(norm((xu + xv) / 2)) - target).norm(dim=1).mean()
    x = featmatch(model, D(), xu, xv, steps=60, lr=0.02, augment=False)
    with torch.no_grad():
        gap1 = (model.embed(norm(x)) - target).norm(dim=1).mean()
    assert gap1 < 0.5 * gap0, (gap0, gap1)
    assert x.min() >= 0 and x.max() <= 1


def test_cluster_rw_conserves_mass():
    g = np.random.default_rng(1)
    n = 400
    labels = g.integers(0, 4, n)
    emb = g.standard_normal((n, 6))
    scores = {"param_norm": g.random(n)}
    for rep in ["influence", "medoid", "random"]:
        keep, w, _ = select("cluster_rw", 0.6, scores, labels, emb=emb, seed=0, opts={"rep": rep})
        assert len(keep) == n - 240 and np.isclose(w.sum(), n), (rep, len(keep), w.sum())
