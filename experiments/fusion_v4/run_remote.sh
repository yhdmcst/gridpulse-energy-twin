#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
PYTHON=/home/amax/anaconda3/envs/torch230_cu123_py310/bin/python
if [ -e outputs/fusion_v4 ]; then
  printf '%s\n' 'Refusing to overwrite existing fusion_v4 results.'
  exit 1
fi
"$PYTHON" -m pytest tests/test_tail_control.py tests/test_forecast_dispatch.py \
  tests/test_balanced_storage.py tests/test_trajectory_audit.py -q
"$PYTHON" -u scripts/run_tail_experiment.py --workers 3
