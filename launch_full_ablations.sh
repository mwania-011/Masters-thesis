#!/usr/bin/env bash
set -euo pipefail
cd /workspace/Multilingual-Thesis
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p thesis/results/e3/ablations_v2/launch_logs

# Start with matched EN-RU seed 1; add pairs/seeds only after reviewing these results.
PAIRS=(en-ru)
SEEDS=(1)
BRANCHES=(continuous geometry_only neighborhood_only selective_neighborhood selective_geometry joint_selective alignment_only)

for pair in "${PAIRS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for branch in "${BRANCHES[@]}"; do
      stem="${pair}_seed${seed}_${branch}"
      out="thesis/results/e3/ablations_v2/full/${branch}/${pair}_seed${seed}/results.json"
      if [[ -f "$out" ]]; then
        python - "$out" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r['status']=='PASS', 'Existing run is not PASS: '+sys.argv[1]
PY
        echo "SKIP verified existing $stem"
        continue
      fi
      echo "=== $stem: PREFLIGHT ==="
      python -m thesis.experiments.run_e3_ablations --mode preflight --pair "$pair" --seed "$seed" --branch "$branch" --device cuda \
        > "thesis/results/e3/ablations_v2/launch_logs/${stem}_preflight.log" 2>&1
      smoke="thesis/results/e3/ablations_v2/smoke/${branch}/${pair}_seed${seed}/gate.json"
      if [[ ! -f "$smoke" ]]; then
        echo "=== $stem: SMOKE ==="
        python -m thesis.experiments.run_e3_ablations --mode smoke-test --pair "$pair" --seed "$seed" --branch "$branch" --device cuda \
          2>&1 | tee "thesis/results/e3/ablations_v2/launch_logs/${stem}_smoke.log"
      fi
      echo "=== $stem: FULL ==="
      python -m thesis.experiments.run_e3_ablations --mode run --pair "$pair" --seed "$seed" --branch "$branch" --device cuda --approve-full \
        2>&1 | tee "thesis/results/e3/ablations_v2/launch_logs/${stem}_full.log"
    done
  done
done
python - <<'PY'
import json
from pathlib import Path
root=Path('thesis/results/e3/ablations_v2/full')
print('\nBRANCH\tPAIR\tSEED\tTOP1%\tTOP5%\tGEO_ACTIVE%\tTOPO_ACTIVE%')
for p in sorted(root.glob('*/*/results.json')):
    r=json.loads(p.read_text()); t=r['result']['trace']; v=r['result']['validation']; n=len(t)
    print(f"{r['branch']}\t{r['pair']}\t{r['seed']}\t{100*v['Top-1']:.4f}\t{100*v['Top-5']:.4f}\t{100*sum(x['lambda_geo']>0 for x in t)/n:.2f}\t{100*sum(x['lambda_topo']>0 for x in t)/n:.2f}")
PY
