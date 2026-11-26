# Revisiting Influence Functions for Data Pruning

Code, configurations, sweep definitions and released results for the paper *Revisiting Influence Functions
for Data Pruning*.

The method pairs near-duplicate training examples within each class in an embedding space (DINOv2 or the
network's own features). In each pair, it drops one member based on its last-layer influence. Controls on
the same pairs (`pair_random`, `pair_loss`) and standard baselines (`random`, `el2n`, `ccs`, `moderate`)
run through the same training pipeline.

## Contents

| path | what it is |
|---|---|
| `dpif/` | the pipeline: scoring, influence, embeddings, pairing, pruning, training, aggregation |
| `configs/` | model and training configurations (ResNet-18 on CIFAR-10/100; fine-tuned ResNet-50) |
| `sweeps/` | sweep definitions (method × ratio × seed grids) behind every result |
| `results/<run>.csv` | one row per trained model, for each experiment directory (see below) |
| `results/paired_differences_seed_level.csv` | seed-level paired differences A − B on the same scoring run and pairs |
| `make_paired_differences.py` | rebuilds the paired-differences file from `results/*.csv` |
| `figures/cifar10_t1/` | merge examples (original pair and merged image) for the `avg` and `featmatch` operators, from the tier-1 run |
| `make_paper_tables.py`, `make_paper_tables_c100.py` | build the paper tables from a full local `runs/` directory |
| `tests/` | unit tests, including influence checks against autograd |
| `colab/`, `kaggle/` | notebooks to run single seeds on Colab or Kaggle |

## Install

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install torch torchvision                           # a CUDA build matching your driver
pip install -r requirements.txt
python -m pytest -q tests                               # ~30 s on CPU
python -m dpif.sweep sweeps/smoke.yaml --device cpu     # ~1 min end-to-end on synthetic data
```

CIFAR-10/100 download automatically into `data/`. DINOv2 ViT-S/14 loads through `torch.hub`.

## Reproducing the tables

### Option A: from the released CSVs (minutes, CPU)

Every accuracy in the paper is a row of `results/<run>.csv`, with columns `method, tag, ratio, seed,
test_acc, val_acc, ...`. Accuracies are fractions, and `ratio` is the fraction of the 45,000-image pool
removed. Mean ± std tables are a group-by over `tag` and `ratio`:

```python
import pandas as pd
d = pd.read_csv("results/cifar100_r18_snap.csv")
print((d.groupby(["tag", "ratio"])["test_acc"].agg(["mean", "std", "count"]) * 100).round(2))
```

To rebuild the seed-level paired differences:

```bash
python make_paired_differences.py      # -> results/paired_differences_seed_level.csv
```

Each row of the paired-differences file is `diff_pp = acc_a − acc_b` in percentage points, for one run,
comparison, metric, ratio and seed. Methods A and B always share the same scoring run (same seed and pool),
and for pair methods the same pairs. The file reproduces the paper's Table 3: rows with `metric == "test_acc"`,
averaged over seeds (and over CIFAR-10 and CIFAR-100 for the from-scratch rows). Two cell means fall exactly
on a rounding boundary (−1.305 and +0.045); the paper rounds half away from zero, so Python's default
rounding may show them 0.01 closer to zero (−1.30 and +0.04).

### Option B: from scratch (GPU)

Each experiment directory `runs/<run>/` is filled by the sweeps listed below. `dpif.sweep` runs the scoring
model, embeddings, pruning and final training, and resumes after an interruption. Seeds are independent:
`--seeds 0` runs one seed, so seeds can run on separate machines.

```bash
python -m dpif.sweep sweeps/tier2_cifar10_main.yaml --device cuda     # optionally: --seeds 0
python -m dpif.aggregate runs/cifar10_r18                             # -> runs/cifar10_r18/results.csv, table, plot
python make_paper_tables.py; python make_paper_tables_c100.py         # -> runs/PAPER_TABLES_*.md
```

| run (= `results/<run>.csv`) | dataset / model | sweeps |
|---|---|---|
| `cifar10_r18` | CIFAR-10, ResNet-18 | `tier2_cifar10_main`, `local/exp4_balanced_c10` |
| `cifar10_r18_snap` | CIFAR-10, ResNet-18 (epoch-10 snapshot scores) | `tier2b_diagnosis`, `local/tier2b_plus`, `local/tier2c_confirm`, `tier4c_keep_easy`, `local/exp5_keep_easy_loss_c10` |
| `cifar10_r18_dir` | CIFAR-10, ResNet-18 (validation-directed influence) | `tier2d_directed` |
| `cifar100_r18_snap` | CIFAR-100, ResNet-18 | `tier4_cifar100_final`, `tier4c_keep_easy_c100`, `local/exp4_balanced_c100`, `local/exp5_keep_easy_loss_c100` |
| `cifar10_ft_r50` | CIFAR-10, ImageNet-pretrained ResNet-50, fine-tuned 3 epochs at 224×224 | `tier5_finetune` |
| `cifar10_t1` | CIFAR-10, ResNet-18. **Exploratory: one seed, 30 epochs, earlier influence implementation (before the float64 fixes); reported in Table 8 only.** | `tier1_decision` |

Run the sweeps of one directory in the order listed: later sweeps reuse the scores and embeddings that the
first one computes. Sweep paths are relative to `sweeps/`. Other files in `sweeps/` (`tier3_ablations`,
`tier4_cifar100_main`, `tier4b_*`, `tier5_transfer`, `extra/`) are development passes or unused options.
They are not part of the reported results. Rerunning `tier1_decision` with the current code uses the
float64 influence implementation, so its influence-based rows can differ slightly from `results/cifar10_t1.csv`.

Measured cost on a GTX 1080: about 31 s per epoch for ResNet-18 on the 45k pool, and 100 epochs per model (the fine-tuning setting uses 3 epochs).

## Evaluation protocol

* Official test sets. 5,000 class-balanced training images are held out as a validation split. Every
  method prunes the same remaining 45,000-image pool.
* Within each setting, every method uses the same training recipe (ResNet-18: 100 epochs, cosine schedule, the same
  augmentation; fine-tuning: ResNet-50, 3 epochs). Pruning ratios are exact.
* Three seeds. Each seed trains its own scoring model, prunes and trains, so seed variance includes the
  pruning step.
* Choices were made on `val_acc` only, and `test_acc` is reported. For validation-directed methods
  (`cifar10_r18_dir`), scores use validation half A, so they are judged on `val_b_acc` (half B).
* Influence is computed in closed form for the final linear layer: the exact damped Hessian on CIFAR-10
  ResNet-18, and K-FAC on CIFAR-100 and ResNet-50.

## License

MIT, see `LICENSE`.
