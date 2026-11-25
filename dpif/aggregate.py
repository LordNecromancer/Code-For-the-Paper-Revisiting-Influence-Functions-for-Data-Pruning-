"""Collect result JSONs -> CSV, a mean ± std markdown table, and an accuracy-vs-ratio plot.

  python -m dpif.aggregate runs/cifar10_r18 [--metric test_acc] [--budget epochs]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd


def load(run_dir, results="results") -> pd.DataFrame:
    rows = [json.load(open(f)) for f in glob.glob(str(Path(run_dir) / results / "*.json"))]
    df = pd.DataFrame(rows)
    if "budget" not in df:
        df["budget"] = "epochs"
    df["budget"] = df["budget"].fillna("epochs")
    return df


def table(df: pd.DataFrame, metric="test_acc", budget="epochs") -> pd.DataFrame:
    d = df[(df.tag == "full") | (df.budget == budget)].copy()
    d.loc[d.tag == "full", "tag"] = "full data"
    g = d.groupby(["tag", "ratio"])[metric].agg(["mean", "std", "count"]).reset_index()
    cell = lambda r: f"{100 * r['mean']:.2f} ± {100 * (r['std'] if r['count'] > 1 else 0):.2f}" + (
        f" (n={int(r['count'])})" if r["count"] < 3 else "")
    g["cell"] = g.apply(cell, axis=1)
    t = g.pivot(index="tag", columns="ratio", values="cell")
    t.columns = [f"{int(round(100 * c))}%" for c in t.columns]
    order = ["full data"] + sorted(i for i in t.index if i != "full data")
    return t.reindex([i for i in order if i in t.index]).fillna("")


def plot(df, path, metric="test_acc", budget="epochs"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df[(df.tag != "full") & (df.budget == budget)]
    g = d.groupby(["tag", "ratio"])[metric].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tag, s in g.groupby("tag"):
        ax.errorbar(100 * s["ratio"], 100 * s["mean"], yerr=100 * s["std"].fillna(0), marker="o", capsize=3, label=tag)
    full = df[df.tag == "full"][metric]
    if len(full):
        ax.axhline(100 * full.mean(), color="k", ls="--", lw=1, label="full data")
    ax.set_xlabel("pruned fraction of training pool (%)"); ax.set_ylabel(f"{metric} (%)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=200)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--metric", default="test_acc")
    ap.add_argument("--budget", default="epochs")
    ap.add_argument("--results", default="results", help="results subdir, e.g. results_vgg11")
    args = ap.parse_args(argv)
    df = load(args.run_dir, args.results)
    out = Path(args.run_dir) / ("" if args.results == "results" else args.results)
    df.to_csv(out / "results.csv", index=False)
    t = table(df, args.metric, args.budget)
    md = t.to_markdown()
    (out / f"table_{args.metric}_{args.budget}.md").write_text(md + "\n")
    print(md)
    plot(df, out / f"plot_{args.metric}_{args.budget}.png", args.metric, args.budget)


if __name__ == "__main__":
    main()
