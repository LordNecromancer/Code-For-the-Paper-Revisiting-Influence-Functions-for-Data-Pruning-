r"""Closed-form last-layer influence for a softmax-cross-entropy linear head.

Setup. Head parameters W in R^{C x D} (bias folded in: a = [h, 1], D = d + 1), flattened
row-major so theta index = c*D + j. For example i with probabilities p_i and one-hot y_i:

    gradient   g_i = (p_i - y_i) (x) a_i                      (Kronecker product)
    Hessian    H   = (1/n) sum_i (diag(p_i) - p_i p_i^T) (x) a_i a_i^T

Numerics (all in float64). For confident models 1 - p_y is far below float32 resolution, which
broke both the Hessian (indefinite -> Cholesky failure) and the gradients (p_y rounded to 1 gave
g a component along the Hessian's null direction, which H^-1 then amplified by 1/lambda). So:
  * 1 - p_c is computed directly from the logits (log-sum-exp over the other classes);
  * H is assembled as a Gram matrix  sum_i sum_k (B_i[:,k] (x) a_i)(...)^T  with
    B B^T = diag(p) - p p^T, B = diag(sqrt p) - p sqrt(p)^T, so it is PSD by construction;
  * the damped inverse uses an eigendecomposition with negative rounding residue clipped at 0.

H is only positive SEMI-definite (adding the same vector to every class row leaves the
softmax unchanged), so we always damp: H + lambda I, lambda = damping * mean(diag H).
(The preprint's "positive definite by assumption" needs this correction.)

Scores returned (all aligned to the input rows):
  param_norm   ||H^{-1} g_i||           -- the scalar used in the thesis code (norm of I_up,params)
  self         g_i^T H^{-1} g_i          -- self-influence (leave-one-out change of z_i's own loss)
  val_removal  (1/n) g_val^T H^{-1} g_i  -- first-order change in mean validation loss if z_i is
                                            removed (> 0: removal hurts; < 0: removal helps)
"""
from __future__ import annotations

import torch

_LAST_DIAG: dict = {}


def _aug(h: torch.Tensor) -> torch.Tensor:
    return torch.cat([h, torch.ones(h.shape[0], 1, dtype=h.dtype, device=h.device)], 1)


def softmax_parts(logits: torch.Tensor, batch: int = 4096):
    """float64 (p, 1 - p) with 1 - p_c computed as exp(lse_{k != c} z_k - lse z): accurate even when
    p_c rounds to 1 in floating point."""
    z = logits.double()
    lse = torch.logsumexp(z, 1, keepdim=True)
    p = torch.exp(z - lse)
    C = z.shape[1]
    eye = torch.eye(C, dtype=torch.bool, device=z.device)
    om = torch.empty_like(p)
    for s in range(0, len(z), max(1, batch // C)):
        zb = z[s:s + batch // C or 1].unsqueeze(1).expand(-1, C, C).masked_fill(eye, float("-inf"))
        om[s:s + zb.shape[0]] = torch.exp(torch.logsumexp(zb, 2) - lse[s:s + zb.shape[0]])
    return p, om


def _residuals(logits, labels, num_classes):
    """float64 (p, e) with e = p - onehot(y); e_y = -(1 - p_y) computed stably, so sum_c e_c = 0."""
    p, om = softmax_parts(logits)
    y = labels.long().to(p.device)
    e = p.clone()
    e.scatter_(1, y[:, None], -om.gather(1, y[:, None]))
    return p, e


def exact_hessian(a: torch.Tensor, p: torch.Tensor, om: torch.Tensor | None = None, batch: int = 256) -> torch.Tensor:
    """(1/n) sum_i (diag p_i - p_i p_i^T) (x) a_i a_i^T in float64, assembled as a Gram matrix
    (PSD by construction). `om` = 1 - p computed stably (see softmax_parts); if omitted, 1 - p."""
    a, p = a.double(), p.double()
    om = (1 - p) if om is None else om.double()
    n, D = a.shape
    C = p.shape[1]
    P = C * D
    H = torch.zeros(P, P, dtype=torch.float64, device=a.device)
    sp = p.sqrt()
    for s in range(0, n, batch):
        pb, spb, omb, ab = p[s:s + batch], sp[s:s + batch], om[s:s + batch], a[s:s + batch]
        B = -pb[:, :, None] * spb[:, None, :]  # B[c, k] = -p_c sqrt(p_k)
        idx = torch.arange(C, device=a.device)
        B[:, idx, idx] = spb * omb  # B[c, c] = sqrt(p_c) (1 - p_c)
        V = (B.transpose(1, 2)[:, :, :, None] * ab[:, None, None, :]).reshape(-1, P)  # rows: B[:, k] (x) a
        H += V.T @ V
    return H / n


def _damped_inverse(M: torch.Tensor, damping: float) -> torch.Tensor:
    """(M_+ + lambda I)^{-1}, M_+ = PSD part of M, lambda = damping * mean(diag M); float64.
    The most negative eigenvalue relative to lambda is recorded (should be ~0)."""
    M = M.double()
    M = 0.5 * (M + M.T)
    lam = damping * torch.diagonal(M).mean().clamp_min(1e-300)
    evals, evecs = torch.linalg.eigh(M)
    _LAST_DIAG["min_eig_over_lambda"] = float(evals.min() / lam)
    _LAST_DIAG["lambda"] = float(lam)
    return (evecs / (evals.clamp_min(0) + lam)) @ evecs.T


@torch.no_grad()
def hessian_factor(feats, logits, labels, num_classes, *, damping=0.01, mode="auto", exact_max_params=12000,
                   device="cpu"):
    """(mode, H_lambda^{-1}) for 'exact' or (mode, (S^-1, A^-1)) for 'kfac'; float64."""
    device = torch.device(device)
    a = _aug(feats.double().to(device))
    p, om = softmax_parts(logits.to(device))
    n, D = a.shape
    if mode == "auto":
        mode = "exact" if num_classes * D <= exact_max_params else "kfac"
    if mode == "exact":
        return mode, _damped_inverse(exact_hessian(a, p, om), damping)
    if mode != "kfac":
        raise ValueError(mode)
    S = -(p.T @ p) / n
    S.diagonal().copy_((p * om).mean(0))  # E[p_c (1 - p_c)] without cancellation
    A = a.T @ a / n
    Si = _damped_inverse(S, damping)
    m1 = _LAST_DIAG["min_eig_over_lambda"]
    Ai = _damped_inverse(A, damping)
    _LAST_DIAG["min_eig_over_lambda"] = min(m1, _LAST_DIAG["min_eig_over_lambda"])
    return mode, (Si, Ai)


@torch.no_grad()
def lastlayer_influence(feats, logits, labels, num_classes, *, damping=0.01, mode="auto",
                        exact_max_params=12000, val=None, device="cpu", batch=2048) -> dict:
    """feats (n,d), logits (n,C), labels (n,). `val` = optional (feats, logits, labels) of a
    held-out set, used for `val_removal`. Returns dict of numpy arrays + metadata."""
    device = torch.device(device)
    mode, fac = hessian_factor(feats, logits, labels, num_classes, damping=damping, mode=mode,
                               exact_max_params=exact_max_params, device=device)
    a = _aug(feats.double().to(device))
    _, e = _residuals(logits.to(device), labels.to(device), num_classes)
    n, D = a.shape
    P = num_classes * D
    g_val = None
    if val is not None and len(val[0]) > 0:
        av = _aug(val[0].double().to(device))
        _, ev = _residuals(val[1].to(device), val[2].to(device), num_classes)
        g_val = (ev.T @ av) / len(av)  # (C, D): mean validation-loss gradient

    pn, sf, vr = [], [], []
    if mode == "exact":
        Hinv = fac
        v = (Hinv @ g_val.reshape(-1)) if g_val is not None else None
        for s in range(0, n, batch):
            G = (e[s:s + batch, :, None] * a[s:s + batch, None, :]).reshape(-1, P)
            X = G @ Hinv  # rows = H^{-1} g_i (Hinv symmetric)
            pn.append(X.norm(dim=1)); sf.append((G * X).sum(1))
            if v is not None:
                vr.append(G @ v / n)
    else:  # H ~= S (x) A  ->  H^{-1}(e (x) a) = (S^-1 e) (x) (A^-1 a)
        Si, Ai = fac
        Se, Aa = e @ Si, a @ Ai
        pn = [Se.norm(dim=1) * Aa.norm(dim=1)]
        sf = [(e * Se).sum(1) * (a * Aa).sum(1)]
        if g_val is not None:
            vr = [((e @ (Si @ g_val @ Ai)) * a).sum(1) / n]
    out = {"param_norm": torch.cat(pn).float().cpu().numpy(), "self": torch.cat(sf).float().cpu().numpy()}
    if vr:
        out["val_removal"] = torch.cat(vr).float().cpu().numpy()
    out["_mode"] = mode
    out["_num_params"] = P
    out["_min_eig_over_lambda"] = _LAST_DIAG.get("min_eig_over_lambda")
    return out


@torch.no_grad()
def hinv_norm_of_combination(factor, terms, batch=2048):
    """||H^{-1} sum_t c_t (e_t (x) a_t)|| per row. `terms` = list of (coef (m,), e (m,C), a_aug (m,D)).
    Used for the first-order merge influence of replacing (u, v) by a synthetic point:
        terms = [(w, e_syn, a_syn), (-1, e_u, a_u), (-1, e_v, a_v)]."""
    mode, fac = factor
    dt = fac.dtype if mode == "exact" else fac[0].dtype
    terms = [(c.to(dt), e.to(dt), a.to(dt)) for c, e, a in terms]
    m = terms[0][1].shape[0]
    out = []
    for s in range(0, m, batch):
        sl = slice(s, s + batch)
        if mode == "exact":
            G = sum(c[sl, None] * (e[sl, :, None] * a[sl, None, :]).reshape(e[sl].shape[0], -1) for c, e, a in terms)
            out.append((G @ fac).norm(dim=1))
        else:
            Si, Ai = fac
            xs = [(c[sl], e[sl] @ Si, a[sl] @ Ai) for c, e, a in terms]
            sq = 0
            for c1, x1, y1 in xs:
                for c2, x2, y2 in xs:
                    sq = sq + c1 * c2 * (x1 * x2).sum(1) * (y1 * y2).sum(1)
            out.append(sq.clamp_min(0).sqrt())
    return torch.cat(out) if out else torch.zeros(0, dtype=dt)


@torch.no_grad()
def pair_merge_scores(feats, logits, labels, num_classes, pairs, *, damping=0.01, mode="auto",
                      exact_max_params=12000, device="cpu", batch=4096, _cache=None):
    """First-order effect of merging pair (u, v) into one point (drop v, give u weight 2):
    ||H^{-1} (g_u - g_v)||  -- symmetric in u, v. This is the "influence of merging" from the
    thesis' Method 1, restricted to merging into an existing (real) example."""
    device = torch.device(device)
    key = (mode, damping)
    if _cache is not None and key in _cache:
        fac = _cache[key]
    else:
        fac = hessian_factor(feats, logits, labels, num_classes, damping=damping, mode=mode,
                             exact_max_params=exact_max_params, device=device)
        if _cache is not None:
            _cache[key] = fac
    if len(pairs) == 0:
        return torch.zeros(0).numpy()
    a = _aug(feats.double().to(device))
    _, e = _residuals(logits.to(device), labels.to(device), num_classes)
    pr = torch.as_tensor(pairs, device=device, dtype=torch.long)
    one = torch.ones(len(pr), dtype=torch.float64, device=device)
    return hinv_norm_of_combination(fac, [(one, e[pr[:, 0]], a[pr[:, 0]]), (-one, e[pr[:, 1]], a[pr[:, 1]])],
                                    batch=batch).float().cpu().numpy()
