# Revision plan for the journal version

Working title: **Influence-Guided Near-Duplicate Pruning: Removing, Reweighting, and Merging
Redundant Training Examples**

## Status after tier 1 (1 seed, 30 epochs, val_acc; to be confirmed by tier 2)

* **Core claim holds:** inside the same near-duplicate pairs, dropping the lower-influence member
  beats a random choice: +0.1 / +1.7 / +2.7 points at 30 / 50 / 70% pruning. It also beats random
  pruning at every ratio (+0.9 / +1.9 / +4.9).
* **Reweighting does not help** (−0.1 / −0.5 / −2.7 vs plain). It ties random choice within pairs,
  as the first-order symmetry remark below predicts. At 70%, multi-round mass transfer leaves most
  kept points with weight 4.
* **Merge-ordered pairs, clusters and merging (Method 1) all lose** to plain pair_influence. Even
  feature-matched merges lose 1.8 points to keeping a real image at 30%, and they carry
  high-frequency artefacts. That is a clean negative result.
* So the paper centres on **plain pair_influence**, and its theory is **Proposition 1**: dropping the
  lower-influence member of each pair greedily minimises the per-removal term of the |R|·T/n bound.
  Proposition 2 and reweighting become an analysis section that explains *why* the member choice
  matters only when mass is not transferred, backed by the empirical tie.
* Tier 2 adds `pair_loss`, the control that tests whether influence beats the per-example loss.

## Story in one paragraph (original plan; revise after tier 2)

Near-duplicate removal (SemDeDup-style) decides *which pairs* are redundant but chooses the
member to drop arbitrarily. We ask whether last-layer influence functions make that choice
better, and show how to make the removal provably gentle. Thresholded removal bounds the
first-order parameter shift by |R|·T/n. Moving the dropped example's weight to its twin
(remove-and-reweight) makes the shift vanish as twins become identical. The quantity that
controls this shift is the influence of *merging* the pair. That links back to the thesis's
first method: Method 1 failed because synthetic merged images were poor. Restricting the merge
target to a real example keeps the idea and removes the failure mode.

## Section-by-section

1. **Introduction.** Drop the scaling-law framing and the line "advances the mathematical
   foundations". Contributions: (i) the exact-ratio pairing method; (ii) Propositions 1–2;
   (iii) a controlled study that isolates the influence choice from the pairing; (iv) a
   negative result on synthesizing merged examples.
2. **Related work.** Add near-duplicate removal (SemDeDup; D4), recent score-based pruning
   (CCS, Moderate-DS, D2 pruning, TDDS, MoSo), and influence-function reliability (Basu et al.
   2021; Koh et al. 2019 on group effects). Shorten the per-method paragraphs.
3. **Method.** Pairing (kNN + greedy matching), the choice rule, threshold T, multi-round
   pairing for exact ratios, remove-and-reweight, merge-ordered pairs. Define the influence
   scalar precisely: ‖H_λ⁻¹ g‖ on the final layer, with damping.
4. **Theory.** Propositions 1–2 below, with the first-order and last-layer caveats stated plainly.
5. **Experiments.** Main: CIFAR-10/100, ResNet-18 from scratch, 3 seeds, 10–70%.
   Secondary: the thesis's fine-tuning setting, now with a random baseline. Ablations: choice
   rule, embedding, T, reweighting, equal-compute budget. Cost table: scoring and selection time.
6. **Method 1 revisited: merging pairs.** Use the same pairs and weights, and vary only the merge
   operator: keep-one (= Method 2), pixel average, influence-weighted average, hybrid, autoencoder
   latent average, and feature-matched synthesis. Pairs are chosen by distance or by the thesis's
   merge-influence rule. Two points to state: the thesis's integer program reduces to sorting by merge
   influence, and feature matching drives the first-order merge influence toward zero by construction.
   Whatever the outcome, this section answers "did Method 1 fail because the merges were naive?"
   If `featmatch` beats `keep`, merging becomes a second positive contribution. If simple operators
   lose to `keep`, that is a clean, explained negative result. The old theorem goes to an appendix
   only if its proof is repaired; otherwise cut it.
6b. **From pairs to clusters** (`cluster_rw`): a weighted k-means coreset that generalizes Method 2.
   Compare representative rules (influence, medoid, random) and weighting (on, off, capped).
6c. **Cross-architecture transfer:** subsets chosen with ResNet-18, trained with VGG-11.
7. **Limitations.** First-order and last-layer approximations; quadratic pairing cost within
   a class (fine at CIFAR scale; needs ANN search at web scale); influence fragility in deep nets.

## Theory

Setup: θ̂ minimises (1/n)Σₖ wₖ ℓ(zₖ, θ) over the final linear layer (features a = [h, 1]
fixed), with wₖ = 1. Let H_λ = H + λI. Changing the weights by Δw gives, to first order,
Δθ ≈ −(1/n) H_λ⁻¹ Σₖ Δwₖ gₖ, where gₖ = (pₖ − yₖ) ⊗ aₖ.

**Proposition 1 (thresholded removal).** Remove a set R in which every point satisfies
‖H_λ⁻¹ gⱼ‖ ≤ T. Then ‖Δθ‖ ≤ |R|·T/n, and the first-order change in any loss with gradient
norm ≤ G is at most G·|R|·T/n.
*Proof:* Δwⱼ = −1 for j ∈ R, then apply the triangle inequality. ∎
This is what the threshold T guarantees. The bound does not shrink as twins get closer.

**Proposition 2 (remove-and-reweight).** Remove j ∈ R and add its weight to a same-class
twin t(j), with ‖a_j − a_{t(j)}‖ ≤ σ. Then
‖Δθ‖ ≤ (1/n)‖H_λ⁻¹‖ Σⱼ ‖g_{t(j)} − gⱼ‖ ≤ |R|·L·σ / (n·λ),
with L = ½‖W‖₂·maxₖ‖aₖ‖ + √2.
*Proof:* g_t − g_j = (r_t − r_j) ⊗ a_t + r_j ⊗ (a_t − a_j), where r = softmax(Wa) − y.
The softmax Jacobian has spectral norm ≤ ½, so ‖r_t − r_j‖ ≤ ½‖W‖₂σ. Also ‖r_j‖ ≤ √2, and
‖H_λ⁻¹‖ ≤ 1/λ. ∎

Remarks:
* For exact duplicates (σ = 0), remove-and-reweight leaves the training objective *exactly*
  unchanged, for the whole network and not only the last layer. Plain removal does not.
* The middle term, ‖H_λ⁻¹(g_t − g_j)‖, is the influence of merging j into t. `pair_merge_rw`
  orders pairs by it, which minimises the per-pair term of the bound greedily.
* Under reweighting, which member of a pair is dropped does not matter to first order (the
  term is symmetric). If `pair_random` ≈ `pair_influence_rw` in the experiments, that is
  predicted by the theory and is worth reporting as a finding.
* The σ in Prop. 2 is Euclidean in penultimate-feature space. Hence the `metric: l2` ablation.

## Experiments → paper

| artefact | sweep |
|---|---|
| (internal) choice of proposed variants | `tier1_decision` |
| Table 1 / Fig. 1: CIFAR-10 accuracy vs ratio, mean ± std | `tier2_cifar10_main` |
| Table 2: ablations (choice rule, pairing, embedding, T, clusters, merge operators) | `tier3_ablations` |
| Table 1 (second half): CIFAR-100 | `tier4_cifar100_main` |
| Table 3: VGG-11 transfer (optional) | `tier5_transfer` |
| Fig.: merged-image examples per operator | `_synth.npz` files from tier 1/3 |
| Table 4: cost (score + select time vs training time) | timings in result JSONs |

## Text fixes regardless of results

* Remove the old Table 2 (its numbers come from Yang et al.'s different setup).
* Data description: official test sets and a validation split. Remove the "70% of data" text.
* Fix signs and indices in Eqs. 7, 11, 12 (or drop them with the old theorem). Fix the
  inconsistent m in the theorem. Replace "Hessian positive definite by assumption" with damping.
* Conclusion: remove "without any loss of accuracy". Abstract: remove the foundations claim.
* Fix the method description: nearest-neighbour matching, exact ratio, influence-scalar definition.
* Release the code (this repo) and mention it in the paper. TMLR reviewers value this.

## Citations to add (verify the bibliographic details before submission)

* Abbas et al., *SemDeDup: Data-efficient learning at web-scale through semantic deduplication*, 2023.
* Tirumala et al., *D4: Improving LLM pretraining via document de-duplication and diversification*, NeurIPS 2023.
* Zheng et al., *Coverage-centric coreset selection for high pruning rates*, ICLR 2023.
* Xia et al., *Moderate coreset: a universal method of data selection for real-world data-efficient deep learning*, ICLR 2023.
* Maharana et al., *D2 pruning: message passing for balancing diversity and difficulty in data pruning*, ICLR 2024.
* Zhang et al., *Spanning training progress: temporal dual-depth scoring (TDDS) for enhanced dataset pruning*, CVPR 2024.
* Tan et al., *Data pruning via moving-one-sample-out*, NeurIPS 2023.
* Koh et al., *On the accuracy of influence functions for measuring group effects*, NeurIPS 2019.
* Basu et al., *Influence functions in deep learning are fragile*, ICLR 2021.
* Martens & Grosse, *Optimizing neural networks with Kronecker-factored approximate curvature*, ICML 2015.
* Oquab et al., *DINOv2: learning robust visual features without supervision*, 2023.

## Venue

First choice: **TMLR**. It judges correctness and evidence, it is open access with no fees,
and a careful study with a clean bound fits well. Alternatives: Neurocomputing, Pattern
Recognition Letters, Neural Computing and Applications. Decide after Table 1 exists. If
influence clearly beats `pair_random`, aim higher. If it does not, the reweighting result and
the honest negative result still make a solid TMLR paper.

## Open decisions before submission

* Authorship: decide with Dr. Sajedi (thesis supervisor; currently only in the acknowledgments).
* Affiliation: the work was done at the University of Tehran. List the current affiliation as a
  present address.
* The arXiv preprint can stay up. Update it to the new version when submitting.
