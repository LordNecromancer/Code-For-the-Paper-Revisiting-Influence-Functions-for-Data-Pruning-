"""Method 1 revisited: replace near-duplicate pairs by ONE synthetic example.

Same pairs as Method 2 (within-class kNN + greedy matching). Each selected pair (u, v) is
removed and replaced by a merged example x~ with label y_u and loss weight `weight`
(default 2 = mass conserving; 1 = the thesis setting). Merging m pairs removes m points net,
so one round reaches at most ~50% pruning.

Merge operators (`op`), from trivial to advanced:
  keep       keep the higher-influence real member with weight 2  (= Method 2 + reweighting)
  avg        pixel average
  infl_avg   influence-weighted pixel average (more weight on the more influential image)
  hybrid     low-pass(higher-influence image) + high-pass(other), Gaussian filters in Fourier space
  ae         decode the mean latent code of a conv autoencoder trained on the whole pool
  featmatch  optimise x~ (initialised at the pixel average) so the scoring network's penultimate
             features and logits of x~ match the AVERAGE of the pair's, under shared random
             crops/flips (siamese augmentation, as in dataset condensation) for transfer.
             Matching features+logits makes x~'s last-layer gradient ~ (g_u + g_v)/2, so with
             weight 2 the first-order merge influence is ~0.

Pair selection (`select`):
  distance          closest pairs first
  merge_influence   synthesise candidates for the `cand_factor * m` closest pairs, then keep the m
                    with the smallest first-order merge influence ||H^-1 (w g(x~) - g_u - g_v)||.
                    This solves the thesis' integer program exactly: maximising the number of
                    merges under sum(merge influence) <= rho is a sort, so no ILP solver is needed.

  python -m dpif.merge --config C --seed S --ratio 0.3 --opt op=featmatch select=merge_influence
"""
from __future__ import annotations

import argparse
import math
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .data import build_data
from .embed import embedding_path, load_scores
from .embeddings import ConvAE, fit_autoencoder
from .influence import _aug, _residuals, hessian_factor, hinv_norm_of_combination
from .models import build_model
from .pairing import find_pairs
from .prune import make_tag, subset_path
from .utils import get_device, load_config, run_paths, save_json

OPS = {"keep", "avg", "infl_avg", "hybrid", "ae", "featmatch"}


# --------------------------------------------------------------------------- helpers
def _normalize(x01, data):
    if data.image_size is not None and data.image_size != x01.shape[-1]:
        x01 = F.interpolate(x01, size=(data.image_size,) * 2, mode="bilinear", align_corners=False)
    m = torch.tensor(data.mean, device=x01.device).view(1, -1, 1, 1)
    s = torch.tensor(data.std, device=x01.device).view(1, -1, 1, 1)
    return (x01 - m) / s


def shared_crop_flip(xs, pad):
    """Apply the SAME random crop+flip (per row) to every tensor in xs (differentiable)."""
    B, C, H, W = xs[0].shape
    dev = xs[0].device
    oy = torch.randint(0, 2 * pad + 1, (B,), device=dev)
    ox = torch.randint(0, 2 * pad + 1, (B,), device=dev)
    flip = torch.rand(B, device=dev) < 0.5
    rows = (oy[:, None] + torch.arange(H, device=dev))[:, None, :, None].expand(B, C, H, W + 2 * pad)
    cols = (ox[:, None] + torch.arange(W, device=dev))[:, None, None, :].expand(B, C, H, W)
    out = []
    for x in xs:
        y = F.pad(x, (pad,) * 4).gather(2, rows).gather(3, cols)
        out.append(torch.where(flip[:, None, None, None], y.flip(3), y))
    return out


def _tv(x):
    return (x[..., 1:, :] - x[..., :-1, :]).abs().mean() + (x[..., :, 1:] - x[..., :, :-1]).abs().mean()


def hybrid(x_low, x_high, sigma=0.08):
    """Gaussian low-pass of x_low + high-pass of x_high (sigma in cycles/pixel)."""
    H, W = x_low.shape[-2:]
    fy = torch.fft.fftfreq(H, device=x_low.device)[:, None]
    fx = torch.fft.fftfreq(W, device=x_low.device)[None, :]
    G = torch.exp(-(fx ** 2 + fy ** 2) / (2 * sigma ** 2))
    out = torch.fft.ifft2(torch.fft.fft2(x_low) * G + torch.fft.fft2(x_high) * (1 - G)).real
    return out.clamp(0, 1)


def featmatch(model, data, xu, xv, *, steps=200, lr=0.01, tv=0.0, prox=0.0, logit_w=0.1, augment=True, log=None):
    """Optimise merged images whose features/logits match the pair average (inputs in [0,1])."""
    pad = max(1, xu.shape[-1] // 8)
    with torch.no_grad():
        lu, fu = model(_normalize(xu, data), return_features=True)
        lv, fv = model(_normalize(xv, data), return_features=True)
        fscale = ((fu + fv) / 2).pow(2).sum(1).mean().clamp_min(1e-8)
        lscale = ((lu + lv) / 2).pow(2).sum(1).mean().clamp_min(1e-8)
    x0 = (xu + xv) / 2
    x = x0.clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    for t in range(steps):
        if augment:
            xa, ua, va = shared_crop_flip([x, xu, xv], pad)
            with torch.no_grad():
                lu, fu = model(_normalize(ua, data), return_features=True)
                lv, fv = model(_normalize(va, data), return_features=True)
        else:
            xa = x
        l, f = model(_normalize(xa, data), return_features=True)
        loss = ((f - (fu + fv) / 2).pow(2).sum(1).mean() / fscale
                + logit_w * (l - (lu + lv) / 2).pow(2).sum(1).mean() / lscale)
        if tv:
            loss = loss + tv * _tv(x)
        if prox:
            loss = loss + prox * (x - x0).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            x.clamp_(0, 1)
    if log is not None:
        log(f"    featmatch final loss {loss.item():.4f}")
    return x.detach()


def load_scoring_model(cfg, run_dir, seed, num_classes, device):
    model = build_model(cfg["model"], num_classes)
    model.load_state_dict(torch.load(run_paths(run_dir, seed)["score"] / "model.pt", map_location=device, weights_only=True))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def _feats_logits(model, data, x01, bs=512):
    L, Fh = [], []
    for s in range(0, len(x01), bs):
        l, f = model(_normalize(x01[s:s + bs], data), return_features=True)
        L.append(l.float()); Fh.append(f.float())
    return torch.cat(L), torch.cat(Fh)


# --------------------------------------------------------------------------- synthesis
def synthesize(op, cand, gidx, data, scores, opts, model=None, ae=None, bs=256, log=print):
    """cand: (m,2) pool positions. Returns merged images in [0,1] (m,3,H,W) on device, plus the
    index of the real member kept (for op=keep) or -1."""
    infl = scores[opts.get("influence_key", "param_norm")]
    outs, kept = [], np.full(len(cand), -1)
    for s in range(0, len(cand), bs):
        c = cand[s:s + bs]
        u, v = c[:, 0], c[:, 1]
        xu = data.x_train[torch.as_tensor(gidx[u], device=data.x_train.device)].float() / 255
        xv = data.x_train[torch.as_tensor(gidx[v], device=data.x_train.device)].float() / 255
        hi_u = torch.as_tensor(infl[u] >= infl[v], device=xu.device)[:, None, None, None]
        if op == "keep":
            outs.append(torch.where(hi_u, xu, xv))
            kept[s:s + bs] = np.where(infl[u] >= infl[v], u, v)
        elif op == "avg":
            outs.append((xu + xv) / 2)
        elif op == "infl_avg":
            lam = torch.as_tensor(infl[u] / (infl[u] + infl[v] + 1e-12), device=xu.device, dtype=xu.dtype)
            outs.append(lam[:, None, None, None] * xu + (1 - lam[:, None, None, None]) * xv)
        elif op == "hybrid":
            outs.append(hybrid(torch.where(hi_u, xu, xv), torch.where(hi_u, xv, xu), float(opts.get("sigma_f", 0.08))))
        elif op == "ae":
            size = ae.size
            rs = lambda x: x if x.shape[-1] == size else F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
            with torch.no_grad():
                z = (ae.enc(rs(xu)) + ae.enc(rs(xv))) / 2
                rec = ae.dec(ae.dec_fc(z).view(-1, 128, ae.s, ae.s))
            outs.append(F.interpolate(rec, size=xu.shape[-2:], mode="bilinear", align_corners=False) if rec.shape[-1] != xu.shape[-1] else rec)
        elif op == "featmatch":
            outs.append(featmatch(model, data, xu, xv, steps=int(opts.get("steps", 200)), lr=float(opts.get("lr", 0.01)),
                                  tv=float(opts.get("tv", 0.0)), prox=float(opts.get("prox", 0.0)),
                                  logit_w=float(opts.get("logit_w", 0.1)), augment=bool(opts.get("augment", True)),
                                  log=log if s == 0 else None))
        else:
            raise ValueError(f"unknown merge op {op!r}")
    return torch.cat(outs), kept


def run(cfg, run_dir, seed, ratio, opts, emb_kind="self", tag=None, device=None, log=print):
    opts = dict(opts or {})
    device = device or get_device()
    tag = make_tag("merge", emb_kind, opts, tag)
    out = subset_path(run_dir, seed, tag, ratio)
    if out.exists():
        return out
    op, select = opts.get("op", "avg"), opts.get("select", "distance")
    w = float(opts.get("weight", 2.0))
    assert op in OPS, op
    t0 = time.time()
    scores = load_scores(run_dir, seed)
    labels, gidx = scores["labels"], scores["pool_idx"]
    n = len(labels)
    emb = np.load(embedding_path(run_dir, seed, emb_kind))
    data = build_data(cfg["data"], device)
    pairs, dists = find_pairs(emb, labels, np.arange(n), k=int(opts.get("k", 10)),
                              metric=opts.get("metric", "cosine"), device=device)
    m = int(round(ratio * n))
    if m > len(pairs):
        warnings.warn(f"merge: only {len(pairs)} pairs available for {m} merges; ratio capped")
        m = len(pairs)
    n_cand = m if select == "distance" else min(len(pairs), int(math.ceil(float(opts.get("cand_factor", 1.5)) * m)))
    cand = pairs[:n_cand]

    need_model = op == "featmatch" or select == "merge_influence"
    model = load_scoring_model(cfg, run_dir, seed, data.num_classes, device) if need_model else None
    ae = None
    if op == "ae":
        ae_path = run_paths(run_dir, seed)["seed"] / "ae.pt"
        kw = cfg.get("embed", {}).get("ae", {})
        if ae_path.exists():
            ck = torch.load(ae_path, map_location=device, weights_only=True)
            ae = ConvAE(int(ck["latent"]), int(ck["size"])).to(device)
            ae.load_state_dict(ck["state"])
            ae.size = int(ck["size"])
            ae.eval()
        else:
            ae = fit_autoencoder(data, device, seed=seed, log=log, **kw)
            torch.save({"state": ae.state_dict(), "latent": int(kw.get("latent", 256)), "size": ae.size}, ae_path)
    log(f"[merge] {tag} r={ratio:.2f}: synthesising {len(cand)} candidates (op={op}, select={select})")
    X, kept = synthesize(op, cand, gidx, data, scores, opts, model=model, ae=ae, log=log)

    info = {"op": op, "select": select, "weight": w, "n_candidates": len(cand)}
    if select == "merge_influence":
        dev = torch.device(device)
        C = data.num_classes
        feats = torch.as_tensor(scores["features"], dtype=torch.float32)
        logits = torch.as_tensor(scores["logits"])
        fac = hessian_factor(feats, logits, torch.as_tensor(labels), C, damping=float(opts.get("damping", 0.01)), device=dev)
        ls, fs = _feats_logits(model, data, X)
        y = torch.as_tensor(labels[cand[:, 0]], device=dev)
        _, e_s = _residuals(ls, y, C)
        _, ep = _residuals(logits.to(dev), torch.as_tensor(labels, device=dev), C)
        ap = _aug(feats.to(dev).double())
        u, v = torch.as_tensor(cand[:, 0], device=dev), torch.as_tensor(cand[:, 1], device=dev)
        s = hinv_norm_of_combination(fac, [(torch.full((len(cand),), w, device=dev), e_s, _aug(fs.double())),
                                           (-torch.ones(len(cand), device=dev), ep[u], ap[u]),
                                           (-torch.ones(len(cand), device=dev), ep[v], ap[v])]).cpu().numpy()
        order = np.argsort(s, kind="stable")[:m]
        info["merge_influence_selected_mean"] = float(s[order].mean())
        info["merge_influence_all_mean"] = float(s.mean())
    else:
        order = np.arange(m)
    sel = cand[order]
    Xsel = X[torch.as_tensor(order, device=X.device)]

    removed = np.zeros(n, bool)
    removed[sel.ravel()] = True
    keep_pos = np.flatnonzero(~removed)
    real_idx, real_w = gidx[keep_pos], np.ones(len(keep_pos))
    synth = None
    if op == "keep":  # no synthetic images: re-add the kept real member with weight w
        k = kept[order]
        real_idx = np.concatenate([real_idx, gidx[k]])
        real_w = np.concatenate([real_w, np.full(len(k), w)])
    else:
        sp = out.with_name(out.stem + "_synth.npz")
        sp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(sp, x=(Xsel.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy(),
                            y=labels[sel[:, 0]], w=np.full(len(sel), w), pairs=gidx[sel])
        synth = {"file": str(sp), "n": int(len(sel))}
    n_final = len(real_idx) + (len(sel) if synth else 0)
    info.update(achieved_ratio=1 - n_final / n, select_time_s=time.time() - t0)
    save_json(dict(method="merge", tag=tag, ratio=ratio, seed=seed, emb=emb_kind, opts=opts, indices=real_idx,
                   weights=real_w, synthetic=synth, info=info), out)
    log(f"[merge] kept {len(real_idx)} real + {len(sel) if synth else 0} synthetic of {n} "
        f"(achieved {info['achieved_ratio']:.3f}, {info['select_time_s']:.0f}s) -> {out}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ratio", type=float, required=True)
    ap.add_argument("--emb", default="self")
    ap.add_argument("--opt", nargs="*", default=[])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    opts = {k: yaml.safe_load(v) for k, v in (o.split("=", 1) for o in args.opt)}
    run(cfg, args.run_dir or f"runs/{cfg['name']}", args.seed, args.ratio, opts, args.emb, args.tag,
        get_device(args.device))


if __name__ == "__main__":
    main()
