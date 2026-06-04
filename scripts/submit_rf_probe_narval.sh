#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16}"
ACCOUNT="${NARVAL_SLURM_ACCOUNT:-def-rsadve}"
PARTITION="${NARVAL_SLURM_PARTITION:-gpu}"
GPUS_PER_NODE="${NARVAL_GPUS_PER_NODE:-}"
TIME="${NARVAL_RF_PROBE_TIME:-0:30:00}"
MEM="${NARVAL_RF_PROBE_MEM:-24G}"
CPUS="${NARVAL_RF_PROBE_CPUS:-4}"

cd "${PROJECT_ROOT}"
mkdir -p logs
sbatch \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --gres="gpu:${GPUS_PER_NODE}" \
  --time="${TIME}" \
  --mem="${MEM}" \
  --cpus-per-task="${CPUS}" \
  slurm/narval/probe_rf_impairments.sbatch
