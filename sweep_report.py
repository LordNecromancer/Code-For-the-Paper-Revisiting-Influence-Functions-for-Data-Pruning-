"""Live, crash-safe progress report for a sweep. Safe to run while the sweep is running; rebuilds
everything from files on disk, so it can be re-run any time (e.g. after a shutdown).

  python sweep_report.py sweeps/tier1_decision.yaml --log logs/tier1_decision.log --report REPORT_tier1.md

Writes into runs/<name>/ (never into results/):
  results.csv, table_val_acc_epochs.md, plot_val_acc_epochs.png, figures/merge_*.png, <report>.md
Decision tiers are compared on val_acc only (CLAUDE.md protocol 1); no test_acc table is produced.
Every file is written to a temp name and then atomically replaced.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import yaml

from dpif import aggregate
from dpif.prune import make_tag
from dpif.train import result_path
from dpif.utils import load_config, run_paths

# Tier-1 decision questions: (question, [(label, tag_a, tag_b)]) -> val_acc(a) - val_acc(b) per ratio
TIER1_QUESTIONS = [
    ("Q1 influence vs random choice inside the same pairs",
     [("pair_influence − pair_random", "pair_influence@self", "pair_random@self")]),
    ("Q2 does reweighting help?",
     [("pair_influence_rw − pair_influence", "pair_influence_rw@self", "pair_influence@self")]),
    ("Q3 merge-ordered pairs / clusters vs plain reweighting",
     [("pair_merge_rw − pair_influence_rw", "pair_merge_rw@self", "pair_influence_rw@self"),
      ("cluster_rw − pair_influence_rw", "cluster_rw@self[rep=influence]", "pair_influence_rw@self")]),
    ("Q4 can merging (Method 1) beat keeping a real image?",
     [("merge avg − merge keep", "merge@self[op=avg]", "merge@self[op=keep]"),
      ("merge featmatch − merge keep", "merge@self[op=featmatch]", "merge@self[op=keep]")]),
    ("Q5 sanity: above random and a standard baseline?",
     [("el2n − random", "el2n", "random"),
      ("pair_influence − random", "pair_influence@self", "random"),
      ("pair_influence_rw − random", "pair_influence_rw@self", "random")]),
]


def atomic_write_text(path: Path, text: str):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def expected_jobs(sw, cfg, run_dir):
    seeds, ratios = sw["seeds"], sw["ratios"]
    rdir = run_dir / (f"results_{sw['results_suffix']}" if sw.get("results_suffix") else "results")
    budget = cfg["train"].get("budget", "epochs")
    jobs = []
    for s in seeds:
        for m in sw["methods"]:
            for r in m.get("ratios", ratios):
                tag = make_tag(m["method"], m.get("emb", "self"), dict(m.get("opts", {})), m.get("tag"))
                jobs.append(dict(seed=s, tag=tag, ratio=r, done=result_path(rdir, tag, r, s, budget).exists()))
    return jobs


def log_scan(log: Path):
    if not log or not log.exists():
        return dict(current=None, issues=[], featmatch=[], start=None, finished=None)
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    current = next((l for l in reversed(lines) if l.startswith("[sweep] [")), None)
    seen = {}  # normalised line -> [first occurrence, count]
    for l in lines:
        if re.search(r"Traceback|Error|warning|Warning|nan\b|NaN", l):
            key = re.sub(r"\d+", "#", l.strip())[:160]
            seen.setdefault(key, [l.strip()[:300], 0])[1] += 1
    issues = [first + (f"  (×{n})" if n > 1 else "") for first, n in seen.values()]
    fm = [float(m.group(1)) for l in lines if (m := re.search(r"featmatch final loss ([\d.]+)", l))]
    start = next((l for l in lines if l.startswith("[launcher]") and "start" in l), None)
    finished = next((l for l in reversed(lines) if l.startswith("[launcher]") and "exit code" in l), None)
    return dict(current=current, issues=issues, featmatch=fm, start=start, finished=finished)


def gpu_time(run_dir: Path, seeds, df):
    score = 0.0
    for s in seeds:
        mp = run_paths(run_dir, s)["score"] / "meta.json"
        if mp.exists():
            m = json.load(open(mp))
            score += m.get("train_time_s", 0) + m.get("score_time_s", 0)
    train = float(df.loc[df.tag != "full", "train_time_s"].sum()) if len(df) and "train_time_s" in df else 0.0
    select = 0.0
    for s in seeds:
        for f in glob.glob(str(run_paths(run_dir, s)["subsets"] / "*.json")):
            select += json.load(open(f)).get("info", {}).get("select_time_s", 0) or 0
    return score, train, select


def merge_figures(run_dir: Path, cfg, seeds, n=8):
    """8 example pairs per merge operator: original u | original v | merged (CLAUDE.md requirement)."""
    out = []
    files = [f for s in seeds for f in sorted(glob.glob(str(run_paths(run_dir, s)["subsets"] / "*_synth.npz")))]
    if not files:
        return out
    figdir = run_dir / "figures"
    figdir.mkdir(exist_ok=True)
    xtr = None
    for f in files:
        png = figdir / (Path(f).name.replace("_synth.npz", "_examples.png"))
        out.append(png)
        if png.exists():
            continue
        if xtr is None:
            import torchvision
            ds = cfg["data"]["dataset"]
            cls = {"cifar10": torchvision.datasets.CIFAR10, "cifar100": torchvision.datasets.CIFAR100}.get(ds)
            if cls is None:
                return out
            xtr = cls(cfg["data"].get("root", "./data"), train=True, download=False).data  # (N,32,32,3) uint8
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        z = np.load(f)
        x, pairs = z["x"], z["pairs"]
        idx = np.random.default_rng(0).choice(len(pairs), size=min(n, len(pairs)), replace=False)
        fig, axes = plt.subplots(len(idx), 3, figsize=(4.2, 1.45 * len(idx)))
        for row, k in enumerate(idx):
            u, v = pairs[k]
            for col, (img, title) in enumerate([(xtr[u], "u"), (xtr[v], "v"), (x[k].transpose(1, 2, 0), "merged")]):
                ax = axes[row, col]
                ax.imshow(img, interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                if row == 0:
                    ax.set_title(title, fontsize=8)
        fig.suptitle(Path(f).stem.replace("_synth", ""), fontsize=7)
        fig.tight_layout()
        tmp = png.with_name(png.stem + ".tmp.png")
        fig.savefig(tmp, dpi=150)
        plt.close(fig)
        os.replace(tmp, png)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--log", default=None)
    ap.add_argument("--report", default=None, help="file name inside runs/<name>/, default REPORT_<sweep>.md")
    args = ap.parse_args(argv)

    sw = yaml.safe_load(open(args.sweep))
    overrides = list(sw.get("set", []))
    if sw.get("name"):
        overrides.append(f"name={sw['name']}")
    cfg = load_config(sw["config"], overrides)
    run_dir = Path(sw.get("run_dir", f"runs/{cfg['name']}"))
    run_dir.mkdir(parents=True, exist_ok=True)
    report = run_dir / (args.report or f"REPORT_{Path(args.sweep).stem}.md")
    log = log_scan(Path(args.log) if args.log else None)
    jobs = expected_jobs(sw, cfg, run_dir)
    n_done = sum(j["done"] for j in jobs)

    rdir = "results" + (f"_{sw['results_suffix']}" if sw.get("results_suffix") else "")
    has_results = bool(glob.glob(str(run_dir / rdir / "*.json")))
    df = aggregate.load(run_dir, rdir) if has_results else None
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    status = "FINISHED" if log["finished"] else ("running" if n_done < len(jobs) else "all jobs done")
    L = [f"# {Path(args.sweep).stem} — `{run_dir.as_posix()}`", "",
         f"_Auto-generated by `sweep_report.py`, last updated {now}. Status: **{status}**, "
         f"{n_done}/{len(jobs)} pruned-training jobs done._", ""]
    if log["start"]:
        L.append(f"- Launch: `{log['start']}`")
    if log["finished"]:
        L.append(f"- End: `{log['finished']}`")
    elif log["current"]:
        L.append(f"- Currently running: `{log['current']}`")
    L.append(f"- Config: `{sw['config']}` with overrides `{sw.get('set', [])}`; seeds {sw['seeds']}")
    L.append("")

    if df is not None and len(df):
        out_csv = run_dir / ("results.csv" if rdir == "results" else f"{rdir}/results.csv")
        tmp = out_csv.with_name(out_csv.name + ".tmp")
        df.to_csv(tmp, index=False)
        os.replace(tmp, out_csv)
        t = aggregate.table(df, "val_acc", "epochs")
        md = t.to_markdown()
        atomic_write_text(run_dir / "table_val_acc_epochs.md", md + "\n")
        png = run_dir / "plot_val_acc_epochs.png"
        tmp_png = run_dir / "plot_val_acc_epochs.tmp.png"
        aggregate.plot(df, tmp_png, "val_acc", "epochs")
        os.replace(tmp_png, png)
        L += ["## Validation accuracy (%), mean ± std over seeds", "", md, "",
              f"Plot: `{png.as_posix()}`. Column = fraction of the 45k pool removed. "
              "Decisions use val_acc only; test_acc is kept in results.csv for final reporting.", ""]

        # mechanical comparisons for tier 1
        if Path(args.sweep).stem == "tier1_decision":
            piv = df[df.tag != "full"].groupby(["tag", "ratio"])["val_acc"].mean().unstack()
            L += ["## Decision comparisons (val_acc difference, percentage points)", "",
                  "_Single seed, 30 epochs: differences under ~0.3–0.5 pp are within run-to-run noise._", ""]
            for q, comps in TIER1_QUESTIONS:
                L.append(f"**{q}**")
                for label, a, b in comps:
                    if a in piv.index and b in piv.index:
                        cells = [f"{int(round(100 * r))}%: {100 * (piv.loc[a, r] - piv.loc[b, r]):+.2f}"
                                 for r in piv.columns if not (math.isnan(piv.loc[a, r]) or math.isnan(piv.loc[b, r]))]
                        L.append(f"- {label}: " + (", ".join(cells) if cells else "no common ratio yet"))
                    else:
                        L.append(f"- {label}: pending")
                L.append("")

        # per-run detail
        L += ["## Runs", "", "| tag | ratio | achieved | n_train | val_acc | train min |", "|---|---|---|---|---|---|"]
        for _, r in df.sort_values(["tag", "ratio", "seed"]).iterrows():
            L.append(f"| {r['tag']} | {r['ratio']:.2f} | {r.get('achieved_ratio', float('nan')):.3f} | "
                     f"{int(r.get('n_train', 0))} | {100 * r['val_acc']:.2f} | {r.get('train_time_s', 0) / 60:.1f} |")
        L.append("")

    # pending jobs
    pend = [j for j in jobs if not j["done"]]
    if pend:
        L += ["## Pending jobs", ""] + [f"- {j['tag']} r={j['ratio']} seed={j['seed']}" for j in pend] + [""]

    # time
    if df is not None:
        sc, tr, se = gpu_time(run_dir, sw["seeds"], df)
        L += ["## GPU time", "",
              f"- Scoring runs (full-data training + influence): {sc / 3600:.2f} h",
              f"- Pruned-subset training: {tr / 3600:.2f} h",
              f"- Selection / merge synthesis: {se / 60:.1f} min",
              f"- **Total so far: {(sc + tr + se) / 3600:.2f} h**", ""]

    # anomalies
    an = []
    if df is not None and len(df):
        d = df[df.tag != "full"]
        bad = d[(d.ratio - d.achieved_ratio).abs() > 0.005]
        for _, r in bad.iterrows():
            an.append(f"achieved ratio {r.achieved_ratio:.3f} ≠ requested {r.ratio:.2f} for {r.tag}")
        for _, r in df[df.val_acc.isna() | df.test_acc.isna()].iterrows():
            an.append(f"NaN accuracy in {r.tag} r={r.ratio}")
        full = df[df.tag == "full"].val_acc.mean()
        beat = d[d.val_acc >= full]
        for _, r in beat.iterrows():
            an.append(f"{r.tag} r={r.ratio:.2f} val_acc {100 * r.val_acc:.2f} ≥ full data {100 * full:.2f}")
        rnd = d[d.tag == "random"].groupby("ratio").val_acc.mean().sort_index()
        if len(rnd) > 1 and not rnd.is_monotonic_decreasing:
            an.append(f"random does not degrade monotonically with ratio: {dict((100 * rnd).round(2))}")
    if log["featmatch"]:
        hi = [x for x in log["featmatch"] if x >= 1.0]
        an.append(f"featmatch final losses: n={len(log['featmatch'])}, min {min(log['featmatch']):.3f}, "
                  f"max {max(log['featmatch']):.3f}" + (f" — {len(hi)} ≥ 1.0: optimisation NOT working" if hi else " (all < 1.0, OK)"))
    an += [f"log: `{x}`" for x in log["issues"]]
    L += ["## Checks and anomalies", ""] + ([f"- {a}" for a in an] if an else ["- none"]) + [""]

    figs = merge_figures(run_dir, cfg, sw["seeds"])
    if figs:
        L += ["## Merged-image examples (u | v | merged)", ""] + [f"- `{p.as_posix()}`" for p in figs] + [""]

    if (run_dir / "INTERPRETATION.md").exists():
        L += ["## Interpretation", "", (run_dir / "INTERPRETATION.md").read_text(encoding="utf-8"), ""]

    atomic_write_text(report, "\n".join(L))
    print(f"[report] {n_done}/{len(jobs)} done -> {report}")


if __name__ == "__main__":
    main()
