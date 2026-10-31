#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHON=/home/amax/anaconda3/envs/torch230_cu123_py310/bin/python
OUT=outputs/fusion_v1
if [ -e "$OUT" ]; then
  printf '%s\n' 'Refusing to overwrite an existing experiment.'
  exit 1
fi
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m pytest \
  tests/test_energy_twin.py tests/test_twin_drift.py \
  tests/test_twin_validation.py tests/test_physics_forecast.py -q
mkdir -p "$OUT/logs"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
for gpu in 0 2; do
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
  if [ "$used" -gt 4000 ]; then
    printf 'GPU %s is occupied; aborting without disturbing its jobs.\n' "$gpu"
    exit 1
  fi
done
(
  for seed in 42 43 44; do
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u scripts/run_fusion_experiment.py \
      --task train --seed "$seed" > "$OUT/logs/data_seed${seed}.log" 2>&1
  done
) &
data_pid=$!
(
  for seed in 42 43 44; do
    CUDA_VISIBLE_DEVICES=2 "$PYTHON" -u scripts/run_fusion_experiment.py \
      --task train --physics --seed "$seed" > "$OUT/logs/physics_seed${seed}.log" 2>&1
  done
) &
physics_pid=$!
CUDA_VISIBLE_DEVICES="" "$PYTHON" -u scripts/run_fusion_experiment.py \
  --task baseline > "$OUT/logs/baseline.log" 2>&1 &
baseline_pid=$!
status=0
wait "$data_pid" || status=1
wait "$physics_pid" || status=1
wait "$baseline_pid" || status=1
if [ "$status" -ne 0 ]; then
  printf '%s\n' 'One or more runs failed; retained logs and partial artifacts.'
  exit 1
fi
CUDA_VISIBLE_DEVICES="" "$PYTHON" -u scripts/run_fusion_experiment.py \
  --task diagnostics > "$OUT/logs/diagnostics.log" 2>&1
CUDA_VISIBLE_DEVICES="" "$PYTHON" -u scripts/run_fusion_experiment.py \
  --task summary > "$OUT/logs/summary.log" 2>&1
printf '%s\n' 'FUSION_V1_COMPLETE'
