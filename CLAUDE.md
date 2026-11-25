# Project context for Claude Code

This repo re-runs the experiments for a journal revision of an MSc-thesis paper on
influence-guided near-duplicate data pruning. Read `README.md` (pipeline, protocol) and
`PAPER_PLAN.md` (what each sweep is for) before doing anything.

## Environment

* Use a dedicated environment: `python -m venv .venv && source .venv/bin/activate`, then
  install a CUDA build of PyTorch that matches the installed driver (check `nvidia-smi`),
  then `pip install -r requirements.txt`.
* Confirm the GPU is visible: `python -c "import torch; print(torch.cuda.is_available())"`.
* Run `python -m pytest -q tests` and the smoke sweep before any real run.

## How to run long jobs

* Training runs take minutes to hours. Launch sweeps as detached background processes, with
  output going to a log under `logs/`, so they survive the session ending. On Linux use `tmux`;
  on Windows use a detached process (e.g. `Start-Process`, or `start /b` with redirection).
  Make sure the PC will not sleep during a sweep.
* Don't sit polling a running job. Report what was launched, where the log is, and the
  estimated finish time, then stop. Check progress when asked.
* Sweeps resume automatically. After a crash, fix the cause and relaunch the same command.
* Seeds are independent. `--seeds 1` runs only seed 1, so seeds can run on different machines
  (this PC, Colab via `colab/run_sweep.ipynb`, Kaggle). When the user brings results JSONs back
  from another machine, copy them into `runs/<name>/results/` (never overwrite existing files),
  then aggregate.

## GPU notes (GTX 1080, Pascal)

* `dpif/training.py` enables channels_last only on GPUs with compute capability >= 7.0, so on
  the GTX 1080 it is off automatically (it was 4-10x slower there). No manual edit is needed.
* Pascal has weak native FP16. Re-time 2 epochs with `train.amp=true` vs `false` under the new
  default layout. If FP32 is faster, pass `--set train.amp=false` to every sweep on this
  machine. That is safe: AMP changes speed, not the recipe.

## Experiment protocol (do not break these)

1. **Never choose hyperparameters or methods using `test_acc`.** Tuning (T_q, damping,
   embedding, k) uses `val_acc` only, except methods that use directed influence (computed from val
   half A), which are judged on `val_b_acc` only. Test accuracy is only read for final reporting.
2. Never delete or overwrite files in `runs/*/results/`. If a run must be redone, use a new
   experiment name (`--name ...`).
3. If you change any code under `dpif/` that affects training, scoring or selection, existing
   results become stale. Stop and tell the user before changing such code mid-study. Never
   mix results from before and after a change in one table.
4. Development passes with reduced epochs must use their own `--name` (e.g. `cifar10_dev`).
5. Keep the training recipe identical across methods. Don't tune training per method.
6. When something fails, report the actual error and the fix. Don't silently work around it,
   for example by dropping a method or a seed.

## Order of work (tiers, by importance; see sweeps/tier*.yaml)

0. After updating the code: tests, the smoke sweep on GPU, the AMP timing above.
1. `tier1_decision` (~4 h): 1 seed, 30 epochs, under its own name `cifar10_t1`. Afterwards,
   report a val_acc table and answer the five questions listed at the top of the YAML. Then
   **stop**: the user decides which two proposed variants go into tier 2.
2. `tier2_cifar10_main` (~11 h per seed on this GPU): the headline table. The methods are already
   set from the tier-1 decision. This PC runs `--seeds 0`; seeds 1 and 2 run on Kaggle
   (`kaggle/run_sweep_kaggle.ipynb`, T4 x2, one seed per GPU).
2b. **Tier 3 is ON HOLD.** Tier 2 seed 0 showed no pair_influence advantage at 100 epochs.
   First run `python -m dpif.diagnose runs/cifar10_t1 --seed 0 --config configs/cifar10_resnet18.yaml`
   and the same for `runs/cifar10_r18` (CPU only, minutes) and report both tables side by side.
   Then run `tier2b_diagnosis` (~4.5 h, seed 0) and report against the decision rule written in
   its YAML. If DINOv2 cannot be loaded (torch.hub), substitute `imagenet_r50` and say so.
2c. Done: `sweeps/local/tier2c_confirm.yaml` (3 seeds). Design for the size-based methods is frozen.
2d. `tier2d_directed` (~6 h, seed 0): validation-directed influence (signed last-layer influence and
   full-network TracIn). Judge ONLY on `val_b_acc` (aggregate with `--metric val_b_acc`), because
   val half A is used to compute these scores. Report against the decision rule in the YAML.
   If DINOv2's embedding already exists in runs/cifar10_r18_snap/, copying `emb_dinov2_s14.npy`
   into runs/cifar10_r18_dir/ is fine (same pool); otherwise it is recomputed in minutes.
4. `tier4_cifar100_final` (in sweeps/, replaces any sweeps/local copy): CIFAR-100 confirmation.
   It saves checkpoints and runs `dpif.rescore`, so directed methods can be appended later.
   Done (3 seeds, `runs/cifar100_r18_snap`). `tier4_cifar100_main` is OBSOLETE: do not launch it.
3. `tier3_ablations` is retired; its questions were answered by tiers 1, 2b and 2c.
4b. `tier4b_cifar100_self_e10` (~4 h): CIFAR-100 epoch-10 self pairs (no outside model).
4c. `tier4c_keep_easy` + `tier4c_keep_easy_c100` (~4 h together): keep the EASIER twin at 50/70%.
   The prediction (keep-easy beats keep-hard at 70%) was stated in the YAML before running.
   Report paired keep-easy minus keep-hard differences per seed, per dataset.
5. `sweeps/local/tier5_finetune.yaml` (~10 h): fine-tuning regime, after 4b and 4c.
5. `tier5_transfer` and the sweeps in `sweeps/extra/`, only if the user asks.
Before each tier, give the user an updated time estimate based on measured epoch times.

## Code versions

`runs/*/seed*/score/meta.json` records `influence_numerics`. All seeds in one table must have the
same value (currently `float64-gram-v2`). Scores from before v2 (tier 1) are fine for tier 1's
direction-finding but must not be mixed into tier 2+ tables.

## Results from Kaggle / Colab

The user downloads the Kaggle output (a `runs/` folder). Merge it into the local `runs/` with
no-clobber copying (never overwrite local files), keeping each seed's scores and embeddings
(tier 3 reuses them), then aggregate. Check that every seed used the same config: same epochs,
same pool size in the score meta.json. Report any mismatch instead of mixing results.

## Merged images (Method 1)

* `dpif.merge` writes `<subset>_synth.npz` (uint8 images, labels, weights, source pairs).
  After each Method 1 run, save a figure of 8 example pairs and their merges per operator to
  `runs/<name>/figures/` (original u, original v, merged). The paper needs this figure, and it
  is the quickest way to spot a broken operator.
* `featmatch` logs its final loss for the first batch. If it is not clearly below 1.0, the
  optimization is not working; report it rather than carrying on.

## Updating the code

When the user provides a new version of the repo, replace everything except `runs/`, `logs/`,
`data/`, `.venv`, `sweeps/local/` and your own helper scripts (keep any launcher or report
scripts you created), then run the tests. New versions keep existing methods' behavior
identical unless the release notes say otherwise, so earlier results stay valid.

## Reporting

After each sweep finishes, run `python -m dpif.aggregate runs/<name>` and give the user:
the markdown table, the plot path, total GPU time, and anything anomalous (high variance,
methods failing to reach the target ratio, warnings in the logs).
