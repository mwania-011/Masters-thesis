# Frozen prospective thesis-validation plan (proposed, NOT executed)

## Primary questions
1. Does a pre-intervention diagnostic predict the *held-out* paired benefit of the corresponding structural objective?
2. Does representation-state-dependent control outperform constant and time-matched regularization at equivalent training budgets?
3. Do any effects persist under convergence-based training and across random seeds?

## Training and selection
- Use the same processed train/validation/test partitions, model architecture, tokenization, and five seeds for every comparator.
- Train independent baselines and controller variants under a common maximum budget of 40 epochs (a ceiling, not a claim that 40 are necessary).
- Prespecify validation Top-1 as the primary early-stopping endpoint, with minimum 5 epochs, patience 5 complete validations, and a minimum meaningful improvement of 0.0001 in fractional accuracy.
- Save per-epoch curves, best checkpoint, optimizer state, coefficient trace, wall time, and GPU memory; test only once after validation-based selection.
- Also report fixed-update-budget comparisons, to separate training-time differences from final quality.
- For long-horizon matched continuations, freeze the same initial checkpoint, shuffled-data schedule, and training budget; evaluate at 1, 5, 10 epochs and at stopping. These are distinct from independently trained baselines.

## Prospective intervention-prediction test
- Prespecify diagnostic features and benefit thresholds using only a development partition of pair-seed units.
- Validate predictive association and direction on untouched held-out units; report correlation, sign-prediction performance, calibration and uncertainty.
- Five seeds and four language pairs provide limited independent units: avoid treating per-batch measurements as independent replications.

## Controller mechanism comparison
- Alignment only; geometry only; neighborhood only; combined fixed coefficients.
- Existing diagnostic-modulated controller; fixed coefficients matched to its average exposure; time-dependent diagnostic-independent schedule; shuffled diagnostic control; calibrated selective controller.
- The last three require implementing new runners; they do not exist in the verified archive.
- Match regularization exposure and optimization steps where applicable; freeze all tuning on validation data.

## Analysis
- Primary: paired Top-1 difference in percentage points with 95% CIs and five-seed distribution.
- Secondary: Top-5, mean rank, alignment margin, uniformity, effective rank, neighborhood overlap, perturbation stability, runtime and GPU memory.
- Apply a prespecified multiple-comparison correction to confirmatory tests; distinguish exploratory results.
- Analyze convergence curves, coefficient activation, thresholds and failure conditions.

## Go/no-go
DO NOT claim convergence or controller superiority based on the existing one-epoch continuation alone.
DO NOT rerun or overwrite archived reference, diagnostic, intervention, or controller results.
DO NOT launch unimplemented comparators or rewrite frozen runners in place.
