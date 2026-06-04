#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16}"
VENV_PATH="${VENV_PATH:-/home/rsadve1/scratch/.venvUPAIR}"
ACCOUNT="${NARVAL_SLURM_ACCOUNT:-def-rsadve_gpu}"
GPUS_PER_NODE="${NARVAL_GPUS_PER_NODE:-}"
SEEDS="${NARVAL_PUBLICATION_SEEDS:-7}"
VARIANTS="${UPAIR_VARIANTS:-main_d96_b4_r2,shallow_d96_b2_r2,deep_d96_b6_r2,narrow_d64_b4_r2,wide_d128_b4_r2,wide_deep_d128_b6_r2,mlpwide_d96_b4_r4}"
DMRS_CASES="${UPAIR_DMRS_CASES:-1dmrs}"
EVAL_USERS="${UPAIR_EVAL_USERS:-3}"
ARRAY_CONCURRENCY="${NARVAL_EVAL_ONLY_ARRAY_CONCURRENCY:-3}"
EVAL_WALLTIME="${NARVAL_EVAL_ONLY_WALLTIME:-12:00:00}"
cd "${PROJECT_ROOT}"
mkdir -p logs
count_csv(){ local value="$1"; local count=0; IFS=',' read -r -a arr <<< "$value"; for item in "${arr[@]}"; do [[ -n "$(echo "$item" | tr -d '[:space:]')" ]] && ((count+=1)); done; echo "$count"; }
seed_count=$(count_csv "${SEEDS}"); variant_count=$(count_csv "${VARIANTS}"); dmrs_count=$(count_csv "${DMRS_CASES}")
last_index=$((seed_count * variant_count * dmrs_count - 1))
if (( last_index < 0 )); then echo "[SUBMIT] ERROR: empty seeds/variants/DMRS cases" >&2; exit 2; fi
account_args=(); if [[ -n "${ACCOUNT}" ]]; then account_args=(--account="${ACCOUNT}"); fi
gpu_args=(); if [[ -n "${NARVAL_GRES:-gpu:1}" ]]; then gpu_args=(--gres="${NARVAL_GRES:-gpu:1}"); elif [[ -n "${GPUS_PER_NODE}" ]]; then gpu_args=(--gpus-per-node="${GPUS_PER_NODE}"); fi
echo "[SUBMIT] eval-only config: ${UPAIR_COMPREHENSIVE_CONFIG:-configs/twc_comprehensive_mu32_eval_safe.yaml}"
echo "[SUBMIT] seeds=${SEEDS} variants=${VARIANTS} dmrs_cases=${DMRS_CASES} eval_users=${EVAL_USERS} array=0-${last_index}"
sbatch \
  "${account_args[@]}" "${gpu_args[@]}" \
  --time="${EVAL_WALLTIME}" \
  --array="0-${last_index}%${ARRAY_CONCURRENCY}" \
  --export="ALL,UPAIR_PUBLICATION_SEEDS=${SEEDS},PROJECT_ROOT=${PROJECT_ROOT},VENV_PATH=${VENV_PATH},UPAIR_VARIANTS=${VARIANTS},UPAIR_DMRS_CASES=${DMRS_CASES},UPAIR_EVAL_USERS=${EVAL_USERS},UPAIR_TF_GPU_ALLOCATOR=${UPAIR_TF_GPU_ALLOCATOR:-default}" \
  slurm/narval/12_eval_only_1dmrs_u3_optuna_best_safe.sbatch
