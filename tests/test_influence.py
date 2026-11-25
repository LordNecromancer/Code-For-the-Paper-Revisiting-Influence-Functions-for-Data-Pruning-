import numpy as np
import torch

from dpif.influence import _aug, exact_hessian, lastlayer_influence


def _problem(n=40, d=5, C=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(n, d, generator=g, dtype=torch.float64)
    y = torch.randint(0, C, (n,), generator=g)
    W = torch.randn(C, d + 1, generator=g, dtype=torch.float64) * 0.5
    return h, y, W


def _loss(Wflat, a, y, C):
    return torch.nn.functional.cross_entropy(a @ Wflat.view(C, -1).T, y)


def test_hessian_matches_autograd():
    h, y, W = _problem()
    C = W.shape[0]
    a = _aug(h)
    H_auto = torch.autograd.functional.hessian(lambda w: _loss(w, a, y, C), W.reshape(-1))
    p = torch.softmax(a @ W.T, 1)
    H = exact_hessian(a.float(), p.float())
    assert torch.allclose(H, H_auto, atol=1e-5), (H - H_auto).abs().max()


def test_scores_match_autograd():
    h, y, W = _problem()
    C, n = W.shape[0], len(y)
    a = _aug(h)
    logits = a @ W.T
    # validation set
    hv, yv, _ = _problem(n=25, seed=1)
    av = _aug(hv)
    damping = 0.05
    res = lastlayer_influence(h, logits, y, C, damping=damping, mode="exact",
                              val=(hv, av @ W.T, yv))

    w = W.reshape(-1).clone().requires_grad_(True)
    H = torch.autograd.functional.hessian(lambda q: _loss(q, a, y, C), w.detach())
    H = H + damping * torch.diagonal(H).mean() * torch.eye(len(H), dtype=H.dtype)
    Hinv = torch.linalg.inv(H)
    gval = torch.autograd.grad(_loss(w, av, yv, C), w)[0]
    for i in [0, 7, 19]:
        gi = torch.autograd.grad(_loss(w, a[i:i + 1], y[i:i + 1], C), w)[0]
        x = Hinv @ gi
        np.testing.assert_allclose(res["param_norm"][i], x.norm().item(), rtol=1e-3)
        np.testing.assert_allclose(res["self"][i], (gi @ x).item(), rtol=1e-3)
        np.testing.assert_allclose(res["val_removal"][i], (gval @ x).item() / n, rtol=1e-3, atol=1e-7)


def test_val_removal_predicts_retraining():
    """First-order removal effect should track true leave-one-out retraining (convex case)."""
    h, y, W0 = _problem(n=200, d=4, C=3, seed=3)
    C, n = 3, 200
    a = _aug(h)
    hv, yv, _ = _problem(n=100, d=4, C=3, seed=4)
    av = _aug(hv)
    lam = 1e-2  # L2 makes the problem strongly convex; include it in the Hessian

    def fit(mask):
        w = torch.zeros(C * (a.shape[1]), dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([w], max_iter=500, tolerance_grad=1e-12, tolerance_change=1e-14,
                                line_search_fn="strong_wolfe")
        idx = torch.nonzero(mask).squeeze(1)

        def closure():
            opt.zero_grad()
            # objective normalised by full n, so removal = dropping a 1/n term (influence convention)
            l = torch.nn.functional.cross_entropy(a[idx] @ w.view(C, -1).T, y[idx], reduction="sum") / n
            l = l + 0.5 * lam * (w ** 2).sum()
            l.backward()
            return l
        opt.step(closure)
        return w.detach()

    full = torch.ones(n, dtype=torch.bool)
    w = fit(full)
    vloss = lambda q: _loss(q, av, yv, C).item()
    base = vloss(w)
    H = torch.autograd.functional.hessian(lambda q: _loss(q, a, y, C), w) + lam * torch.eye(len(w), dtype=torch.float64)
    Hinv = torch.linalg.inv(H)
    wq = w.clone().requires_grad_(True)
    gval = torch.autograd.grad(_loss(wq, av, yv, C), wq)[0]
    pred, true = [], []
    for i in range(0, 200, 10):
        gi = torch.autograd.grad(_loss(wq, a[i:i + 1], y[i:i + 1], C), wq)[0]
        pred.append((gval @ Hinv @ gi).item() / n)
        m = full.clone(); m[i] = False
        true.append(vloss(fit(m)) - base)
    r = np.corrcoef(pred, true)[0, 1]
    assert r > 0.95, r


def test_kfac_runs_and_correlates():
    h, y, W = _problem(n=300, d=8, C=4, seed=2)
    logits = _aug(h) @ W.T
    ex = lastlayer_influence(h, logits, y, 4, mode="exact")
    kf = lastlayer_influence(h, logits, y, 4, mode="kfac")
    assert kf["_mode"] == "kfac" and np.all(np.isfinite(kf["param_norm"]))
    r = np.corrcoef(np.argsort(np.argsort(ex["param_norm"])), np.argsort(np.argsort(kf["param_norm"])))[0, 1]
    assert r > 0.5, r


def test_confident_model_does_not_crash_and_is_accurate():
    """Regression: a very confident model (100-epoch ResNet-18, train loss ~0.002) made the float32
    Hessian indefinite and Cholesky failed. Scores must now match a float64 reference."""
    from dpif.influence import softmax_parts
    torch.manual_seed(0)
    n, d, C = 3000, 64, 10
    for scale in [20.0, 40.0]:
        y = torch.randint(0, C, (n,))
        h = torch.relu(torch.randn(n, d)) * 3
        logits = scale * torch.nn.functional.one_hot(y, C).float() + 0.5 * torch.randn(n, C)
        res = lastlayer_influence(h, logits, y, C, mode="exact", damping=0.01)
        assert res["_min_eig_over_lambda"] > -1e-3, res["_min_eig_over_lambda"]
        # reference: textbook formulas in float64 with exact 1 - p (mpmath-free: logits are moderate)
        a = _aug(h.double()); p, om = softmax_parts(logits)
        H = exact_hessian(a, p, om)
        H = H + 0.01 * torch.diagonal(H).mean() * torch.eye(len(H), dtype=torch.float64)
        Hinv = torch.linalg.inv(H)
        e = p.clone(); e[torch.arange(n), y] = -om[torch.arange(n), y]
        for i in [0, 11, 999]:
            g = (e[i][:, None] * a[i][None, :]).reshape(-1)
            np.testing.assert_allclose(res["param_norm"][i], (Hinv @ g).norm().item(), rtol=2e-3)
        kf = lastlayer_influence(h, logits, y, C, mode="kfac", damping=0.01)
        assert np.all(np.isfinite(kf["param_norm"]))


def test_confident_model_matches_independent_textbook_reference():
    """Independent check: per-example kron(diag p - p p^T, a a^T) and e = p - onehot, naive float64.
    At this confidence (1 - p_y ~ 1e-8) naive float64 is still accurate, float32 is not."""
    torch.manual_seed(1)
    n, d, C = 1500, 24, 6
    y = torch.randint(0, C, (n,))
    h = torch.relu(torch.randn(n, d)) * 3
    logits = 20.0 * torch.nn.functional.one_hot(y, C).float() + 0.5 * torch.randn(n, C)
    res = lastlayer_influence(h, logits, y, C, mode="exact", damping=0.01)
    a = _aug(h.double())
    p = torch.softmax(logits.double(), 1)
    S = torch.diag_embed(p) - p[:, :, None] * p[:, None, :]
    H = torch.zeros(C * (d + 1), C * (d + 1), dtype=torch.float64)
    for i in range(n):
        H += torch.kron(S[i], torch.outer(a[i], a[i]))
    H /= n
    H += 0.01 * torch.diagonal(H).mean() * torch.eye(len(H), dtype=torch.float64)
    Hinv = torch.linalg.inv(H)
    e = p - torch.nn.functional.one_hot(y, C).double()
    ref = torch.stack([(Hinv @ torch.kron(e[i], a[i])).norm() for i in range(n)]).numpy()
    np.testing.assert_allclose(res["param_norm"], ref, rtol=1e-4)
    # rank agreement is what the method uses; must be essentially perfect
    assert np.corrcoef(np.argsort(np.argsort(res["param_norm"])), np.argsort(np.argsort(ref)))[0, 1] > 0.9999


def test_tracin_directional_derivative_matches_autograd():
    from dpif.models import build_model
    from dpif.rescore import directional_losses
    torch.manual_seed(0)
    model = build_model({"arch": "resnet18_cifar", "blocks": [1, 1, 1, 1], "width": 4}, 3).eval()
    x, y = torch.randn(5, 3, 8, 8), torch.randint(0, 3, (5,))
    params = {k: p.detach() for k, p in model.named_parameters()}
    buffers = {k: b for k, b in model.named_buffers()}
    tangent = {k: torch.randn_like(p) for k, p in params.items()}
    d = directional_losses(model, x, y, tangent, params, buffers)
    for i in range(5):
        model.zero_grad()
        torch.nn.functional.cross_entropy(model(x[i:i + 1]), y[i:i + 1]).backward()
        ref = sum((p.grad * tangent[k]).sum() for k, p in model.named_parameters())
        np.testing.assert_allclose(d[i].item(), ref.item(), rtol=1e-4, atol=1e-6)
