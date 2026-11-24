#!/usr/bin/env bash
set -euo pipefail

cd /workspace/Multilingual-Thesis

export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

ROOT="thesis/results/e3/ablations_v2"
mkdir -p "$ROOT/launch_logs"

PAIRS=(en-ru en-fr en-it en-es)
SEEDS=(1 2 3 4 5)
BRANCHES=(
  continuous
  geometry_only
  neighborhood_only
  selective_neighborhood
  selective_geometry
  joint_selective
  alignment_only
)

RUNNER="thesis.experiments.run_e3_ablations"

validate_result() {
    python - "$1" "$2" "$3" "$4" <<'PY'
import json
import sys

path, branch, pair, seed = sys.argv[1:]
with open(path) as f:
    r = json.load(f)

assert r["status"] == "PASS", path
assert r["branch"] == branch, path
assert r["pair"] == pair, path
assert int(r["seed"]) == int(seed), path

result = r["result"]
assert result["optimizer_steps"] > 0, path
assert result["model_sha256"], path
assert result["optimizer_sha256"], path
assert "Top-1" in result["validation"], path
assert "Top-5" in result["validation"], path
assert len(result["trace"]) == result["optimizer_steps"], path

print("VERIFIED:", path)
PY
}

for pair in "${PAIRS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for branch in "${BRANCHES[@]}"; do

      stem="${pair}_seed${seed}_${branch}"
      directory="${ROOT}/full/${branch}/${pair}_seed${seed}"
      result="${directory}/results.json"
      smoke="${ROOT}/smoke/${branch}/${pair}_seed${seed}/gate.json"

      echo
      echo "========================================"
      echo "EXPERIMENT: $stem"
      echo "========================================"

      if [[ -f "$result" ]]; then
        validate_result "$result" "$branch" "$pair" "$seed"
        echo "SKIP completed experiment"
        continue
      fi

      if [[ -d "$directory" ]]; then
        echo "ERROR: Incomplete output directory: $directory"
        echo "Preserving it for investigation."
        exit 1
      fi

      echo "PREFLIGHT"

      python -m "$RUNNER" \
        --mode preflight \
        --pair "$pair" \
        --seed "$seed" \
        --branch "$branch" \
        --device cuda \
        > "${ROOT}/launch_logs/${stem}_preflight.log" 2>&1

      if [[ ! -f "$smoke" ]]; then
        echo "SMOKE TEST"

        python -m "$RUNNER" \
          --mode smoke-test \
          --pair "$pair" \
          --seed "$seed" \
          --branch "$branch" \
          --device cuda \
          2>&1 | tee "${ROOT}/launch_logs/${stem}_smoke.log"
      else
        echo "Existing smoke gate found; full runner will verify it."
      fi

      echo "FULL TRAINING"

      python -m "$RUNNER" \
        --mode run \
        --pair "$pair" \
        --seed "$seed" \
        --branch "$branch" \
        --device cuda \
        --approve-full \
        2>&1 | tee "${ROOT}/launch_logs/${stem}_full.log"

      validate_result "$result" "$branch" "$pair" "$seed"

      echo "COMPLETED: $stem"

    done
  done
done

echo
echo "========================================"
echo "ALL EXPERIMENTS COMPLETED"
echo "========================================"

python - <<'PY'
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

root = Path("thesis/results/e3/ablations_v2/full")
records = defaultdict(dict)

for path in sorted(root.glob("*/*/results.json")):
    r = json.loads(path.read_text())
    if r["status"] != "PASS":
        raise RuntimeError(f"Non-PASS result: {path}")
    key = (r["pair"], r["branch"])
    seed = int(r["seed"])
    if seed in records[key]:
        raise RuntimeError(f"Duplicate seed: {key} {seed}")
    records[key][seed] = r["result"]["validation"]["Top-1"] * 100

print("\nPAIR | BRANCH | N | MEAN TOP-1 (%) | SD (%)")

for (pair, branch), seed_values in sorted(records.items()):
    values = np.array(list(seed_values.values()), dtype=float)
    print(
        f"{pair:6s} | {branch:24s} | "
        f"{len(values):d} | "
        f"{values.mean():.4f} | "
        f"{values.std(ddof=1):.4f}"
    )

print("\nPAIRED TOP-1 DIFFERENCES VS CONTINUOUS")
for (pair, branch), seed_values in sorted(records.items()):
    if branch == "continuous":
        continue
    baseline = records.get((pair, "continuous"), {})
    common = sorted(set(seed_values) & set(baseline))
    if len(common) != 5:
        raise RuntimeError(f"Expected five paired seeds: {pair}, {branch}")
    differences = np.array(
        [seed_values[s] - baseline[s] for s in common],
        dtype=float
    )
    print(
        f"{pair:6s} | {branch:24s} | "
        f"mean difference={differences.mean():+.4f} pp | "
        f"SD={differences.std(ddof=1):.4f} pp"
    )

expected = 4 * 5 * 7
actual = sum(len(v) for v in records.values())
print(f"\nTOTAL: {actual}/{expected}")
assert actual == expected, "Incomplete experiment matrix"
PY
