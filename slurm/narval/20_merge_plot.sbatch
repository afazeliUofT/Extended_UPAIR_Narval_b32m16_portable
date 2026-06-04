#!/usr/bin/env bash
#SBATCH --job-name=upair-narval-merge
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16}"
VENV_PATH="${VENV_PATH:-/home/rsadve1/scratch/.venvUPAIR}"
cd "${PROJECT_ROOT}"
mkdir -p logs
if [[ "${NARVAL_LOAD_MODULES:-1}" == "1" ]]; then
  if [[ -f /etc/profile.d/modules.sh ]]; then source /etc/profile.d/modules.sh; fi
  if command -v module >/dev/null 2>&1; then
    module --force purge || true
    module load "${NARVAL_STDENV_MODULE:-StdEnv/2023}" || true
    module load "${NARVAL_GCC_MODULE:-gcc/12.3}" || true
    module load "${NARVAL_PYTHON_MODULE:-python/3.11.5}" || true
  fi
fi
export PYTHONNOUSERSITE=1 PIP_NO_USER=1 MPLBACKEND=Agg
unset PYTHONPATH
source "${VENV_PATH}/bin/activate"
python scripts/merge_comprehensive_mu32_results.py
python scripts/make_comprehensive_mu32_plots.py
