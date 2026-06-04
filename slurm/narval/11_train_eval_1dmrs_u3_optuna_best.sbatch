#!/usr/bin/env bash
#SBATCH --job-name=upair-rx16p12-optbest
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-6%7
#SBATCH --signal=B:TERM@600
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
    if [[ -n "${NARVAL_CUDA_MODULE:-cuda/12.2}" ]]; then module load "${NARVAL_CUDA_MODULE:-cuda/12.2}" || true; fi
  fi
fi

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PIP_NO_USER=1
# Use TensorFlow default allocator by default; allow override with UPAIR_TF_GPU_ALLOCATOR.
if [[ "${UPAIR_TF_GPU_ALLOCATOR:-default}" == "default" || -z "${UPAIR_TF_GPU_ALLOCATOR:-}" ]]; then
  unset TF_GPU_ALLOCATOR
else
  export TF_GPU_ALLOCATOR="${UPAIR_TF_GPU_ALLOCATOR}"
fi
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
export MPLBACKEND=Agg
unset PYTHONPATH
source "${VENV_PATH}/bin/activate"

DEFAULT_VARIANTS="main_d96_b4_r2,shallow_d96_b2_r2,deep_d96_b6_r2,narrow_d64_b4_r2,wide_d128_b4_r2,wide_deep_d128_b6_r2,mlpwide_d96_b4_r4"
IFS=',' read -r -a variants <<< "${UPAIR_VARIANTS:-${DEFAULT_VARIANTS}}"
IFS=',' read -r -a dmrs_cases <<< "${UPAIR_DMRS_CASES:-1dmrs}"
IFS=',' read -r -a seeds <<< "${UPAIR_PUBLICATION_SEEDS:-${UPAIR_TRAIN_SEED:-7}}"
idx="${SLURM_ARRAY_TASK_ID}"
variant_count="${#variants[@]}"; case_count="${#dmrs_cases[@]}"; seed_count="${#seeds[@]}"
tasks_per_seed=$((case_count * variant_count)); total_tasks=$((seed_count * tasks_per_seed))
if (( idx >= total_tasks )); then echo "[SLURM] task=${idx} outside grid ${total_tasks}; exiting."; exit 0; fi
seed_idx=$((idx / tasks_per_seed)); within_seed_idx=$((idx % tasks_per_seed)); case_idx=$((within_seed_idx / variant_count)); variant_idx=$((within_seed_idx % variant_count))
dmrs_case="$(echo "${dmrs_cases[$case_idx]}" | tr -d '[:space:]')"
variant="$(echo "${variants[$variant_idx]}" | tr -d '[:space:]')"
seed="$(echo "${seeds[$seed_idx]}" | tr -d '[:space:]')"
eval_seed=$((seed + 1000))
eval_users="${UPAIR_EVAL_USERS:-3}"

use_optuna="${UPAIR_USE_OPTUNA_BEST_1DMRS:-auto}"
if [[ "${use_optuna}" == "auto" ]]; then
  if [[ "${dmrs_case}" == "1dmrs" ]]; then use_optuna=1; else use_optuna=0; fi
fi

echo "[SLURM] task=${idx}/${total_tasks} seed=${seed} eval_seed=${eval_seed} dmrs_case=${dmrs_case} eval_users=${eval_users} variant=${variant} use_optuna=${use_optuna}"
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi || true; fi

extra_args=(--eval-users "${eval_users}")
[[ "${UPAIR_COMPREHENSIVE_EVAL_ONLY:-0}" == "1" ]] && extra_args+=(--eval-only)
[[ "${UPAIR_COMPREHENSIVE_FORCE:-0}" == "1" ]] && extra_args+=(--force)
if [[ "${use_optuna}" == "1" ]]; then extra_args+=(--use-optuna-best-1dmrs); fi
if [[ "${UPAIR_REQUIRE_OPTUNA_BEST:-1}" == "1" ]]; then extra_args+=(--require-optuna-best); fi
extra_args+=(--optuna-best-storage-dir "${UPAIR_OPTUNA_BEST_STORAGE_DIR:-${PROJECT_ROOT}/optuna}")
extra_args+=(--optuna-best-study-prefix "${UPAIR_OPTUNA_BEST_STUDY_PREFIX:-clean_b32_iso_u34610_1dmrs_stageC}")
if [[ -n "${UPAIR_COMPREHENSIVE_EXTRA_ARGS:-}" ]]; then extra_args+=(${UPAIR_COMPREHENSIVE_EXTRA_ARGS}); fi

child_pid=""
forward_term(){ echo "[SLURM] received signal; forwarding TERM to Python worker ${child_pid}" >&2; [[ -n "${child_pid}" ]] && kill -TERM "${child_pid}" 2>/dev/null || true; [[ -n "${child_pid}" ]] && wait "${child_pid}" || true; exit 143; }
trap forward_term TERM INT
python scripts/run_comprehensive_mu32_ablation.py \
  --config "${UPAIR_COMPREHENSIVE_CONFIG:-configs/twc_comprehensive_mu32_base.yaml}" \
  --seeds "${seed}" \
  --dmrs-cases "${dmrs_case}" \
  --variants "${variant}" \
  --no-global-summary \
  "${extra_args[@]}" &
child_pid=$!
wait "${child_pid}"
