# E2 reliability-gated controller — exploratory implementation v1

## Scope and scientific status

This archive supplies **new files only**. It does not alter E0, E1-A, E1-B,
the registered E1-B protocol, or the journal code. It is **not** evidence that
E2 has been executed or that the controller improves retrieval. Run it on the
existing GPU repository and report only observed results.

### Objective

`L = L_align + lambda_geo(t) L_geo + lambda_topo(t) L_topo`

For each *training* minibatch, compute detached embeddings and three diagnostics:

* Alignment maturity: mean positive cross-lingual cosine minus mean hardest
  nonmatching cosine in the batch.
* Global concentration: mean squared norm of the mean embedding vector,
  averaged across source and target (larger means more concentration).
* Neighborhood reliability: fraction of overlapping source/target top-5
  within-language neighbors of each paired example (diagonal excluded).

Apply exponential moving average (decay 0.9), then fixed piecewise-linear
ramp functions. Maturity ramp [0,0.30] multiplies both gates. Geometry gate
also uses concentration ramp [0.05,0.20]. Topology gate also uses overlap
ramp [0.15,0.45]. A 20-step linear warmup multiplies both. Multiply each
result by the pair-specific fixed E1-B coefficient (geometry .0005; topology
.0005 en-fr, .0004 en-ru, .0002 en-it/es). **Thresholds are heuristic, not
calibrated or proven predictive.** All gate decisions use training batches;
validation is computed only after training and test is never used.

No claim of representation-state predictive validity should be made based on
these gates alone. The gate computation adds batchwise similarity calculations
and is not a free computational improvement.

## Installation (from repository root)

Extract the supplied archive **in** `/workspace/Multilingual-Thesis` using
`tar -xzf e2_reliability_gated_v1.tar.gz` (it contains only new files).
Do not extract over existing files if you already have custom E2 work.

```bash
python -m pytest -q tests/test_e2_gates.py
python thesis/experiments/run_e2_gated.py --mode preflight --pair en-it --seed 1 --device cuda
python thesis/experiments/run_e2_gated.py --mode smoke-test --pair en-it --seed 1 --device cuda
python thesis/experiments/run_e2_gated.py --mode run --pair en-it --seed 1 --device cuda --approve-full
```

**Do not start full run before the smoke output and gate are inspected.**
Results go to `thesis/results/e2/reliability_gated_v1/{smoke,full}/en-it_seed1/`.
Every output directory is non-overwriting. Full training uses all training
batches and complete validation. Compare against E1-B `control`, `full`,
`geo`, and `topo` **for the same pair/seed** only after confirming checkpoint
hash, batch schedule hash, steps, and evaluation semantics match.

## Evaluation and thesis reporting

A single seed per pair is an **engineering feasibility demonstration**, not a
five-seed effectiveness result. The minimum meaningful comparative evaluation
uses matched seeds, paired Top-1 (primary), Top-5/MeanRank/MarginMean
(secondary), means and sample SD, and sign counts. If the fixed thresholds
were informed by E1-B results, label the E2 evaluation **exploratory** and do
not present E1-B data as independent controller-validation evidence. If
only two pairs are run, explicitly report limited generality. No test-set
selection or post-hoc threshold tuning.

The exact thesis title is **Reliability-Gated Structural Regularization for
Multilingual Sentence Embeddings**. Describe the journal as the fixed-weight
starting point and E2 as the separate controller extension. Distinguish
completed empirical evidence from future or unexecuted experiments.
