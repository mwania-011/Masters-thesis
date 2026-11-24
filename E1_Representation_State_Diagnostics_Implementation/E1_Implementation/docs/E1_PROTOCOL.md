# E1 protocol — draft, not approved for GPU training

## Aim
Test whether representation-state measurements **preceding** a structural regularization intervention predict its marginal validation retrieval benefit versus a matched alignment-only continuation. No E2 controller is implemented.

## E1-A: measurement
Paired source/target validation embedding rows share translation IDs. Both sides are L2-normalized (not mean-centered) before cosine analysis. Alignment: source-to-target Top-1, positive cosine, maximum off-diagonal hard-negative cosine, and their difference. Top-1 ties use lowest index. Geometry: pairwise mean cosine, norm of embedding mean, eigenvalue max/sum of sample covariance after centering normalized vectors, exp of eigenvalue entropy (effective rank), and log mean exp(-2 squared Euclidean distance) over uniformly sampled *distinct* pairs. Local: overlap of monolingual k-NN ID sets across aligned source/target sentence IDs, plus Gaussian-perturbation stability. k-NN excludes self. These are candidate diagnostics, not validated indicators of reliability.

Alignment *maturity* requires a longitudinal series and is not inferred from one snapshot. For every diagnostic, report sample size, random seed, k, perturbation scale, and uncertainty from repeated independent validation samples.

## E1-B: interventions — requires original repository
From a single E0 training checkpoint, fork control, geometry-only, topology-only, and both branches. Restore **model, optimizer, scheduler, all RNG streams and data order**. Validate matching initial-state hashes, identical training batches and learning rate schedule. Only change lambda_geo and lambda_topo. Proposed initial values are 0 or 0.0005 to match E0, subject to verifying actual loss scaling. Compute delta Top-1 relative to control and paired effects on other validation metrics. No held-out test is accessed for selection. **Existing E0 best/last checkpoints may not supply all desired epochs; do not reconstruct missing checkpoints.**

## E1-C: prediction
Rows are intervention outcomes, predictors are pre-intervention diagnostics only. All branches from one source checkpoint belong to one group. Cross-validation must separate groups and use training-only feature scaling and model fitting. Include epoch-only, pair-only and intercept-only baselines, leave-one-pair-out validation where sample sizes permit, paired uncertainty intervals, and multiple-comparison handling. The provided `grouped_cv` helper is an initial Ridge-vs-intercept baseline, not the full confirmatory analysis. Set beneficial/neutral/harmful margins *before* looking at confirmatory outcomes. Correlation alone is not causal evidence.

## Execution matrix
`python -m e1.cli plan --output results/e1_execution_matrix.csv` creates an 80-row *provisional* matrix (4 pairs x 5 seeds x 4 branches) with unresolved checkpoint and duration fields. It is **not runnable** until the original repository has been inspected, source checkpoint/epoch choices fixed, and GPU budget approved.

## Safeguards
Do not modify `thesis/results/e0`, its logs, source scripts, or audit manifest. Do not run full-scale GPU training. An independent E0 backup is still unconfirmed. The recovery of `en-fr_seed1_full` from epoch 9 to 10 must be documented with provenance and exact runner hashes before confirmatory execution.
