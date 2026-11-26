"""Build results/paired_differences_seed_level.csv from the released per-run CSVs in results/.

Each row is one paired difference  A - B  (percentage points) for one run, seed, pruning ratio and metric.
A and B are always trained from the same scoring run (same seed, same pool) and, for pair-based methods,
on the same near-duplicate pairs, so the difference isolates the choice of method.

    python make_paired_differences.py            # reads results/*.csv, writes results/paired_differences_seed_level.csv

Metrics: test_acc and val_acc for every run. For runs/cifar10_r18_dir (validation-directed influence, which is
computed from validation half A) only val_b_acc and test_acc are reported, because val_acc includes half A.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parent / "results"

# DINOv2 pairs, epoch-10 scores
INF = "pair_influence@dinov2_s14[influence_key=param_norm_e10]"
INF_EASY = "pair_influence@dinov2_s14[influence_key=param_norm_e10,keep=easy]"
LOSS = "pair_loss@dinov2_s14[loss_key=loss_e10]"
LOSS_EASY = "pair_loss@dinov2_s14[keep=easy,loss_key=loss_e10]"
RND = "pair_random@dinov2_s14"

# run -> (dataset, model, metrics, [(A, B, comparison label), ...])
SPEC = {
    "cifar10_r18": ("CIFAR-10", "ResNet-18", ["test_acc", "val_acc"], [
        ("pair_influence@self", "pair_random@self", "pair_influence - pair_random (self pairs)"),
        ("pair_influence@self", "pair_loss@self", "pair_influence - pair_loss (self pairs)"),
        ("pair_influence_rw@self", "pair_influence@self", "pair_influence_rw - pair_influence (self pairs)"),
        ("el2n[per_class=True]", "el2n", "el2n class-balanced - el2n"),
        ("ccs[per_class=True]", "ccs", "ccs class-balanced - ccs"),
    ]),
    "cifar10_r18_snap": ("CIFAR-10", "ResNet-18", ["test_acc", "val_acc"], [
        (INF, RND, "pair_influence - pair_random (DINOv2 pairs, e10)"),
        (INF, LOSS, "pair_influence - pair_loss (DINOv2 pairs, e10)"),
        (INF, "pair_loss@dinov2_s14[loss_key=el2n]", "pair_influence - pair_loss[el2n] (DINOv2 pairs, e10)"),
        ("pair_influence@dinov2_s14", RND, "pair_influence[final-epoch scores] - pair_random (DINOv2 pairs)"),
        (INF_EASY, INF, "keep easy - keep hard, influence (DINOv2 pairs, e10)"),
        (INF_EASY, RND, "keep easy (influence) - pair_random (DINOv2 pairs)"),
        (LOSS_EASY, LOSS, "keep easy - keep hard, loss (DINOv2 pairs, e10)"),
        (LOSS_EASY, RND, "keep easy (loss) - pair_random (DINOv2 pairs)"),
        ("pair_influence@self_e10[influence_key=param_norm_e10]", "pair_random@self_e10",
         "pair_influence - pair_random (self_e10 pairs)"),
        ("pair_influence@self_e10[influence_key=param_norm_e10]", "pair_loss@self_e10[loss_key=loss_e10]",
         "pair_influence - pair_loss (self_e10 pairs)"),
    ]),
    "cifar10_r18_dir": ("CIFAR-10", "ResNet-18", ["test_acc", "val_b_acc"], [
        ("pair_valinf@dinov2_s14[valinf_key=val_removal_A]", RND, "valinf (final) - pair_random (DINOv2 pairs)"),
        ("pair_valinf@dinov2_s14[valinf_key=val_removal_A_e10]", RND, "valinf (e10) - pair_random (DINOv2 pairs)"),
        ("pair_loss@dinov2_s14[loss_key=tracin_val]", RND, "TracIn-val - pair_random (DINOv2 pairs)"),
        ("pair_loss@dinov2_s14[loss_key=tracin_val_early]", RND, "TracIn-val early - pair_random (DINOv2 pairs)"),
        ("pair_valinf@dinov2_s14[valinf_key=val_removal_A]", INF, "valinf (final) - pair_influence (DINOv2 pairs)"),
        ("pair_valinf@dinov2_s14[valinf_key=val_removal_A_e10]", INF, "valinf (e10) - pair_influence (DINOv2 pairs)"),
        ("pair_loss@dinov2_s14[loss_key=tracin_val]", INF, "TracIn-val - pair_influence (DINOv2 pairs)"),
        ("pair_loss@dinov2_s14[loss_key=tracin_val_early]", INF, "TracIn-val early - pair_influence (DINOv2 pairs)"),
        (INF, RND, "pair_influence - pair_random (DINOv2 pairs, e10)"),
        (INF, LOSS, "pair_influence - pair_loss (DINOv2 pairs, e10)"),
    ]),
    "cifar100_r18_snap": ("CIFAR-100", "ResNet-18", ["test_acc", "val_acc"], [
        (INF, RND, "pair_influence - pair_random (DINOv2 pairs, e10)"),
        (INF, LOSS, "pair_influence - pair_loss (DINOv2 pairs, e10)"),
        (INF_EASY, INF, "keep easy - keep hard, influence (DINOv2 pairs, e10)"),
        (INF_EASY, RND, "keep easy (influence) - pair_random (DINOv2 pairs)"),
        (LOSS_EASY, LOSS, "keep easy - keep hard, loss (DINOv2 pairs, e10)"),
        (LOSS_EASY, RND, "keep easy (loss) - pair_random (DINOv2 pairs)"),
        ("pair_influence@self", "pair_random@self", "pair_influence - pair_random (self pairs)"),
        ("el2n[per_class=True]", "el2n", "el2n class-balanced - el2n"),
        ("ccs[per_class=True]", "ccs", "ccs class-balanced - ccs"),
    ]),
    "cifar10_ft_r50": ("CIFAR-10", "ResNet-50 (fine-tuned)", ["test_acc", "val_acc"], [
        ("pair_influence@dinov2_s14", "pair_random@dinov2_s14", "pair_influence - pair_random (DINOv2 pairs)"),
        ("pair_influence@dinov2_s14", "pair_loss@dinov2_s14", "pair_influence - pair_loss (DINOv2 pairs)"),
        ("pair_influence@self", "pair_random@self", "pair_influence - pair_random (self pairs)"),
        ("pair_influence@self", "pair_loss@self", "pair_influence - pair_loss (self pairs)"),
    ]),
}


def main():
    rows, missing = [], []
    for run, (dataset, model, metrics, comps) in SPEC.items():
        d = pd.read_csv(RES / f"{run}.csv")
        d["ratio"] = d["ratio"].round(2)
        for a, b, label in comps:
            if a not in set(d.tag) or b not in set(d.tag):
                missing.append(f"{run}: {label}")
                continue
            for m in metrics:
                p = d[d.tag.isin([a, b])].pivot_table(index=["ratio", "seed"], columns="tag", values=m).dropna()
                for (ratio, seed), r in p.iterrows():
                    rows.append(dict(dataset=dataset, model=model, run=run, comparison=label, method_a=a, method_b=b,
                                     metric=m, ratio=ratio, seed=int(seed), acc_a=round(100 * r[a], 2),
                                     acc_b=round(100 * r[b], 2), diff_pp=round(100 * (r[a] - r[b]), 2)))
    out = pd.DataFrame(rows).sort_values(["run", "comparison", "metric", "ratio", "seed"])
    out.to_csv(RES / "paired_differences_seed_level.csv", index=False)
    print(f"wrote {len(out)} rows -> {RES / 'paired_differences_seed_level.csv'}")
    for m in missing:
        print("skipped (method not in CSV):", m)


if __name__ == "__main__":
    main()
