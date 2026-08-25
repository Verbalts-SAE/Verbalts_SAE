#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

"${SAE_PYTHON:-python}" -m sae.training.train_sae \
  --activations results/sae_retrain/activations/train_layer1_t5_45.pt \
  --valid-activations results/sae_retrain/activations/valid_layer1_t5_45.pt \
  --output-dir results/sae_retrain/models/t5_45 \
  --t-range 5 45 \
  --input-dim 64 \
  --latent-dim 512 \
  --topk-k 48 \
  --aux-lambda 0.01 \
  --dead-tolerance 500 \
  --lr 0.0001 \
  --batch-size 4096 \
  --epochs 100 \
  --warmup-steps 500 \
  --eval-interval 10 \
  --seed 42
