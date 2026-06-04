#!/usr/bin/env bash
set -euo pipefail

# Convenience launcher for the first RF-impaired campaign:
#   Optuna/training: mixed clean + waveform-domain CFO/phase-noise
#   Evaluation: impaired 3-user, 1-DMRS, seed 7 by default

export PROJECT_ROOT="${PROJECT_ROOT:-/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16}"
export VENV_PATH="${VENV_PATH:-/home/rsadve1/scratch/.venvUPAIR}"
export UPAIR_COMPREHENSIVE_CONFIG="${UPAIR_COMPREHENSIVE_CONFIG:-configs/twc_comprehensive_mu32_rf_mixed_u3_1dmrs.yaml}"
export UPAIR_OPTUNA_CONFIG="${UPAIR_OPTUNA_CONFIG:-${UPAIR_COMPREHENSIVE_CONFIG}}"
export UPAIR_DMRS_CASES="${UPAIR_DMRS_CASES:-1dmrs}"
export UPAIR_EVAL_USERS="${UPAIR_EVAL_USERS:-3}"
export NARVAL_PUBLICATION_SEEDS="${NARVAL_PUBLICATION_SEEDS:-7}"
export OPTUNA_STUDY_PREFIX="${OPTUNA_STUDY_PREFIX:-pilot_rx16p12_rf_mixed_1dmrs}"
export UPAIR_OPTUNA_BEST_STUDY_PREFIX="${UPAIR_OPTUNA_BEST_STUDY_PREFIX:-${OPTUNA_STUDY_PREFIX}}"

bash "${PROJECT_ROOT}/scripts/submit_optuna_then_train_eval_u3_narval.sh"
