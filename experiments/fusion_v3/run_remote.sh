#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHON=/home/amax/anaconda3/envs/torch230_cu123_py310/bin/python
if [ -e outputs/fusion_v3 ]; then
  printf '%s\n' 'Refusing to overwrite existing fusion_v3 results.'
  exit 1
fi
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  printf '%s\n' 'Select one inspected free GPU explicitly.'
  exit 1
fi
"$PYTHON" -m pytest tests/test_causal_intervals.py tests/test_multihorizon_fusion.py \
  tests/test_physics_forecast.py tests/test_forecast_dispatch.py \
  tests/test_balanced_storage.py tests/test_trajectory_audit.py -q
"$PYTHON" -u scripts/run_multihorizon_experiment.py --workers 3
