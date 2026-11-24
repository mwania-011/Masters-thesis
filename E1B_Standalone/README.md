# E1-B standalone matched-continuation runner

**Frozen E0 is never edited.** Copy `thesis/experiments/run_e1b.py` and `tests/test_e1b.py` into the matching paths in `/workspace/Multilingual-Thesis`. The runner imports the existing E0 journal model, dataset and evaluator.

## Steps

```bash
cd /workspace/Multilingual-Thesis
PYTHONPATH="$PWD" pytest -q tests/test_e1b.py
PYTHONPATH="$PWD" python thesis/experiments/run_e1b.py --mode preflight --pair en-ru --seed 1 --device cuda --steps 2
PYTHONPATH="$PWD" python thesis/experiments/run_e1b.py --mode smoke-test --pair en-ru --seed 1 --device cuda --steps 2
```

The smoke test runs **all four branches twice**, checking exact replay of model/optimizer hashes, validation metrics, and loss/gradient logs. It writes a gate only if all checks pass. Review the results before authorizing any full continuation.

**Full training is locked by default**. The following command is provided for a *later* explicitly approved experiment, **not to run now**:

```bash
PYTHONPATH="$PWD" python thesis/experiments/run_e1b.py --mode run --pair en-ru --seed 1 --device cuda --approve-full
```

The full mode additionally requires a passed smoke gate for the **same pair, seed, device, checkpoint hash and runner code hash**. One `--run` command executes four branches, not all 80. Each branch receives the same original checkpoint, the same saved RNG state, and the same precomputed shuffled sample indices. It runs exactly one additional epoch by default, using full validation. The matched control is the comparator. No held-out test data are accessed.

## Limitations and safety

- Source checkpoint files are loaded using `torch.load(..., weights_only=False)`; only load trusted, locally generated E0 files.
- The preflight validates the model, optimizer and provenance against the frozen E0 checkpoint. It does not train or write results.
- The smoke test uses the first 128 validation examples for a **software check**, not for publishable retrieval results.
- CUDA deterministic mode may reject nondeterministic kernels rather than silently proceeding. Such a failure is a legitimate failed gate.
- The original topology loss can internally clamp/fallback to zero. This runner preserves its definition and logs the resulting loss, but cannot distinguish an internal zero fallback from a mathematically zero value. Add explicit loss-health instrumentation before interpreting topology responses as scientific evidence.
- The runner computes fixed coefficients from `e0_config.py`, rather than from E0's epoch curriculum or E1-A diagnostics.
- The full mode currently uses one additional epoch; a different duration requires a separately reviewed design. `--steps` is limited to at most one epoch.
- Files are created exclusively under `thesis/results/e1/b_interventions/{smoke,full}`. Existing output directories are never overwritten.
- The first smoke test may take significant time because it loads XLM-R eight times and performs eight validation passes. It does not change E0.
