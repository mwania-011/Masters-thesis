# E1-B Analysis Protocol — Version 1

## Status
Registered before E1-B execution for en-fr, en-it and en-es.
English–Russian results and exploratory correlations were already examined
before this registration and must be disclosed as prior exploratory evidence.

## Experimental unit
One language-pair × seed checkpoint. Four matched continuations:
control, geo, topo, full.

## Training protocol
Continue from the frozen epoch-10 E0 baseline for one full epoch.
Restore checkpoint model, optimizer, and RNG states.
Use matched minibatch schedules within each experimental unit.
Keep the existing E1-B runner and pair-specific coefficients unchanged.

## Primary endpoint
Top-1 intervention response:
delta_top1 = Top-1(intervention) - Top-1(control).

## Secondary endpoints
delta_top5 = Top-5(intervention) - Top-5(control).
delta_meanrank = MeanRank(control) - MeanRank(intervention).
delta_margin = MarginMean(intervention) - MarginMean(control).
Positive values represent favorable changes for all endpoints.

## Diagnostic families
Alignment maturity:
alignment_positive_cosine, alignment_hard_negative_cosine,
alignment_margin, e0_top1.

Global geometry:
source_anisotropy, target_anisotropy,
source_effective_rank, target_effective_rank,
source_spectral_concentration, target_spectral_concentration,
source_uniformity, target_uniformity.

Neighborhood reliability:
cross_lingual_overlap_at_10,
perturbation_stability_at_10, neighborhood_sensitivity.

Retain original E1-A diagnostic definitions.
Treat deterministically related diagnostics as redundant.

## Analysis
Join E1-A and E1-B on (language_pair, seed).
Require exact checkpoint SHA-256 agreement.
Report individual seed responses, means, sample SDs and sign counts.
Report each language pair separately before pooled descriptions.
Use descriptive Spearman correlations with ties handled by average ranks.
Inspect directions and stability across language pairs.
Do not infer predictive validity from in-sample correlations.
Do not select diagnostics, thresholds, or controller coefficients
based on the largest observed correlations.

## Multiplicity and inference
All diagnostic-response correlations are exploratory.
No unadjusted significance claims from multiple screening tests.
The experimental unit for seed-level analysis is the paired checkpoint.
The three intervention responses from one seed are dependent.
Report effect sizes and uncertainty transparently.

## Controller design boundary
No reliability-gated controller is fitted or evaluated in E1-B.
Freeze controller rules before testing them on held-out data.
A held-out language pair or seed must not influence controller design.
Choose development/evaluation separation before controller fitting.
If the available data cannot support credible holdout evaluation,
collect additional independent data or describe the controller as exploratory.

## Integrity requirements
Never overwrite E0, E1-A, seed-1 pilot, or completed E1-B outputs.
Require historical checkpoint provenance and matched initial states.
Require passing seed-specific smoke gate before a full run.
Require complete full-epoch schedules and finite loss/metric audits.
Stop on any failure; retain failed-run evidence.
Preserve validation/test separation; do not tune on test outcomes.

## Prior information disclosure
Before registration, en-ru five-seed results were inspected.
The en-ru diagnostic-response correlation matrix was also inspected.
Thus this protocol is prospective for remaining language pairs only.
It is not a preregistration of the completed en-ru analysis.
