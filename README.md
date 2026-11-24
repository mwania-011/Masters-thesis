# Reliability-Gated Extension — executable research starter

**Status:** Core implementation and smoke tests completed locally on 2026-11-19. **No GPU training or measured multilingual retrieval results have been generated.** Previously prepared prospective tables/figures are synthetic planning artifacts only.

## Fast start on your GPU server

```bash
cd /workspace/Multilingual-Thesis
# Transfer and unzip rg_extension_impl.zip into this directory, then:
cd rg_extension_impl
python -m pip install -r requirements.txt
python -m unittest -v test_core
```

Prepare one JSONL file per split, with each line like:

```json
{"source":"An example English sentence.","target":"Une phrase française d'exemple."}
```

Directory structure: `DATA/en-fr/train.jsonl`, `DATA/en-fr/validation.jsonl`, likewise `en-ru`, `en-it`, `en-es`. Use the thesis's existing **exact same** split assignments; do not generate random splits. Check data leakage and record split hashes. Do not put held-out test examples in calibration or controller development.

### Quick one-pair smoke experiment (GPU)

```bash
python experiment.py --train DATA/en-fr/train.jsonl --validation DATA/en-fr/validation.jsonl \
  --pair en-fr --method calibrated --epochs 1 --seed 1 --output RUNS/en-fr/calibrated
python analyze.py --runs RUNS --out RUNS/figures
```

### Matched one-epoch continuation from thesis checkpoints

The `--checkpoint` expects a **state dict compatible with the Encoder class** in `experiment.py` (or a dictionary with a `model` state dict). Your original thesis checkpoint may use different module names, projection layers or model wrappers: **verify and write a checkpoint adapter before launching large runs**. `strict=True` intentionally prevents silently loading incompatible checkpoints. A matched comparison must start all branches from exactly the same checkpoint and data order, with optimizer state restored if continuing the original experiment. This starter currently restores model weights only; it does **not** restore optimizer/RNG state. Therefore its continuation results must **not** be described as a strict reproduction of thesis continuations without adding that restoration.

```bash
EPOCHS=1 SEEDS='1' bash run_matrix.sh DATA RUNS CHECKPOINTS
```

The full matrix is 4 language pairs × 5 seeds × 9 configurations = **180 training runs** and is unlikely to finish in one hour on a single GPU. For a one-hour budget, run one pair, one seed, alignment-only versus calibrated gated, then scale later.

### Full training

Without `--checkpoint`, training initializes a new XLM-R base encoder and projection. Use a full comparable training budget (thesis reference: 10 epochs) and a matched set of seeds. `EPOCHS=10 bash run_matrix.sh DATA RUNS` will be expensive.

## Figures and tables from measured runs

`analyze.py` reads actual `history.json` and `traces.json` files. It generates:
- Figure 4: validation Top-1 convergence with approximate confidence bands when multiple seeds are available.
- Figure 5: diagnostic and effective-coefficient traces.
- Figure 7: measured final Top-1 versus wall-clock GPU hours (not normalized for GPU hardware).
- `measured_epoch_results.csv`, `measured_final_results.csv`.

Figure 3 requires **paired intervention-probe outcomes** collected on disjoint calibration/test splits, in `probes.csv` with columns `margin,concentration,overlap,delta_top1,split`. Each row should represent a diagnostic snapshot before a matched intervention and the resulting Top-1 difference versus a no-intervention control at an identical training budget. Do not derive the outcome from the diagnostics themselves. Run:

```bash
python analyze.py --runs RUNS --out RUNS/figures --probes probes.csv
```

This fits logistic models on the calibration split and evaluates AUROC and Brier score on the held-out test split. **The online controller currently uses preset bounded diagnostic thresholds (.55), not the learned logistic predictor.** A scientifically valid predictive controller needs an additional implementation stage that trains the benefit predictor on held-out probes and integrates its calibrated probabilities into the training decision. Do not claim that stage is implemented yet.

## Methodology notes

- First-token XLM-R encoder and 768→768→256 projection with GELU, LayerNorm and unit normalization.
- Alignment uses bidirectional cross entropy, smoothing and hard-negative margin.
- Geometric loss uses squared Gram similarities excluding self and positive translation pairs.
- Neighborhood loss compares within-language softmax similarity distributions.
- Diagnostics: alignment margin, mean-vector concentration, top-5 neighborhood overlap; EMA beta .9, warm-up 20.
- Selective controller independently zeroes geometric and neighborhood coefficients when diagnostic ramps are below configured thresholds.
- These choices are a thesis-aligned **starter**, not a verified bit-for-bit reproduction. Check against original code and data preprocessing before treating any outputs as comparable.
- GradNorm/DWA and fully calibrated learned intervention predictor are **not implemented** in this starter. No 2-point gain can be promised.

## Safety / reproducibility

Save raw logs and checkpoints outside ephemeral containers; keep your existing research files untouched. Avoid committing/pushing until your preferred milestone, but make a separate off-server backup. Never mix prospective simulated data with actual measured logs.
