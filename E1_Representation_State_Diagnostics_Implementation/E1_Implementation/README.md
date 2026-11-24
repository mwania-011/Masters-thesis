# E1 Representation-State Diagnostics — standalone implementation (partial)

**Status:** Working repository-independent E1-A diagnostics and E1-B bookkeeping / E1-C grouped predictive helper. **Not** integrated with the actual remote E0 training repository. **Not** an end-to-end E1 GPU training system. No GPU runs performed.

## Setup
Requires Python >=3.10, numpy, PyTorch; predictive helper optionally requires scikit-learn. From package root:

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
PYTHONPATH=. python -m e1.cli plan --output results/e1_execution_matrix.csv
PYTHONPATH=. python -m e1.cli diagnose --npz validation_embeddings.npz --output results/e1_diagnostics.json --k 10 --seed 1
```

`validation_embeddings.npz` must contain **only** `source` and `target` float arrays of shape `[n,d]`, aligned row-wise by translation ID, drawn from the **validation** split. The loader does not have access to E0 dataset manifests, so the operator must verify the split and ordering; the CLI does not independently authenticate that claim. Never feed held-out test embeddings into threshold design. `--output` refuses to overwrite existing files or to write into a directory named `e0`.

## Contents
- `e1/diagnostics.py`: alignment, geometry, neighborhood measurements
- `e1/interventions.py`: branch state checks, deltas, blocked execution plan
- `e1/predictive.py`: group-disjoint Ridge cross-validation helper
- `e1/cli.py`: offline validation embedding evaluation and provisional plan
- `tests/test_e1.py`: deterministic unit tests
- `docs/E1_PROTOCOL.md`: methodological draft and blocked research decisions

## Repository-specific integration remaining
1. Inspect actual `thesis/experiments/run_e0.py`, training implementation, tokenizer, data splits and checkpoint schema on `/workspace/Multilingual-Thesis`.
2. Implement encoder checkpoint loading and embedding extraction with **identical E0 evaluation semantics**.
3. Verify E0 checkpoints include optimizer, scheduler, RNG, and data-order state; implement and test matched continuation with exact branch equality and fixed training batches.
4. Freeze intervention points, number of epochs, coefficient scaling, predictive baselines and confidence intervals; estimate GPU costs.
5. Execute CPU and GPU integration smoke tests; obtain authorization before full E1 GPU training.

This package is safe to stage in a new `thesis/experiments/e1` directory, but do not overwrite existing code or treat the provisional execution matrix as an approved experiment plan.
