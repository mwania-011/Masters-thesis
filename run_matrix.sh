#!/usr/bin/env bash
set -euo pipefail
# Usage: bash run_matrix.sh /path/to/data /path/to/output [checkpoint-directory]
DATA="${1:?provide dataset root}"; OUT="${2:?provide output root}"; CKPT="${3:-}"
EPOCHS="${EPOCHS:-1}"; SEEDS="${SEEDS:-1 2 3 4 5}"
for pair in en-fr en-ru en-it en-es; do
  train="$DATA/$pair/train.jsonl"; val="$DATA/$pair/validation.jsonl"
  [[ -f "$train" && -f "$val" ]] || { echo "Missing $train or $val" >&2; exit 2; }
  for seed in $SEEDS; do
    checkpoint_args=()
    if [[ -n "$CKPT" ]]; then
      file="$CKPT/$pair/seed_$seed.pt"
      [[ -f "$file" ]] || { echo "Missing $file" >&2; exit 2; }
      checkpoint_args=(--checkpoint "$file")
    fi
    for method in alignment fixed original calibrated; do
      python experiment.py --train "$train" --validation "$val" --pair "$pair" --seed "$seed" \
        --method "$method" --epochs "$EPOCHS" --output "$OUT/$pair/seed_$seed/$method" "${checkpoint_args[@]}"
    done
    for ablate in margin geometry neighborhood ema selection; do
      python experiment.py --train "$train" --validation "$val" --pair "$pair" --seed "$seed" \
        --method calibrated --ablate "$ablate" --epochs "$EPOCHS" \
        --output "$OUT/$pair/seed_$seed/ablate_$ablate" "${checkpoint_args[@]}"
    done
  done
done
python analyze.py --runs "$OUT" --out "$OUT/figures"
