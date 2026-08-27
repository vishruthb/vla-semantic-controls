#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LIBERO_CONFIG_PATH="$PWD/.cache/libero"

uv run --frozen python src/evaluate.py \
  --task-ids 0 \
  --episodes-per-task 2 \
  --batch-size 2 \
  --render-episodes-per-task 1 \
  --output results/smoke_metrics.json
