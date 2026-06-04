#!/bin/bash
#SBATCH --job-name=upair-rf-probe
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16}"
VENV_PATH="${VENV_PATH:-/home/rsadve1/scratch/.venvUPAIR}"
cd "${PROJECT_ROOT}"
source "${VENV_PATH}/bin/activate"
export TF_CPP_MIN_LOG_LEVEL=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
python scripts/probe_sionna_rf_impairments.py \
  --config configs/twc_comprehensive_mu32_base.yaml \
  --num-users "${UPAIR_RF_PROBE_USERS:-1}" \
  --batch-size "${UPAIR_RF_PROBE_BATCH:-2}" \
  --deep
