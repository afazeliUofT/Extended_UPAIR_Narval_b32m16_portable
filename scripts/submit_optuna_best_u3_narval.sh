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
ARRAY_CONCURRENCY="${NARVAL_OPTUNA_BEST_ARRAY_CONCURRENCY:-7}"
TRAIN_WALLTIME="${NARVAL_OPTUNA_BEST_WALLTIME:-12:00:00}"
MERGE_WALLTIME="${NARVAL_OPTUNA_BEST_MERGE_WALLTIME:-02:00:00}"

cd "${PROJECT_ROOT}"
mkdir -p logs
export PROJECT_ROOT VENV_PATH TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}" MPLBACKEND=Agg
export UPAIR_TF_GPU_ALLOCATOR="${UPAIR_TF_GPU_ALLOCATOR:-default}"
if [[ "${UPAIR_TF_GPU_ALLOCATOR}" == "default" || -z "${UPAIR_TF_GPU_ALLOCATOR}" ]]; then
  unset TF_GPU_ALLOCATOR
else
  export TF_GPU_ALLOCATOR="${UPAIR_TF_GPU_ALLOCATOR}"
fi
IFS=',' read -r -a seed_list <<< "${SEEDS}"
IFS=',' read -r -a variant_list <<< "${VARIANTS}"
IFS=',' read -r -a dmrs_list <<< "${DMRS_CASES}"
tasks_per_seed=$((${#variant_list[@]} * ${#dmrs_list[@]}))
array_max=$((tasks_per_seed - 1))
account_args=(); [[ -n "${ACCOUNT}" ]] && account_args=(--account="${ACCOUNT}")
gpu_args=(); if [[ -n "${NARVAL_GRES:-gpu:1}" ]]; then gpu_args=(--gres="${NARVAL_GRES:-gpu:1}"); elif [[ -n "${GPUS_PER_NODE}" ]]; then gpu_args=(--gpus-per-node="${GPUS_PER_NODE}"); fi

echo "[SUBMIT] project=${PROJECT_ROOT}"
echo "[SUBMIT] venv=${VENV_PATH} account=${ACCOUNT:-none} gpu_args=${gpu_args[*]}"
echo "[SUBMIT] seeds=${SEEDS} variants=${VARIANTS} dmrs_cases=${DMRS_CASES} eval_users=${EVAL_USERS} array=0-${array_max}"
echo "[SUBMIT] Optuna-best source: ${UPAIR_OPTUNA_BEST_STORAGE_DIR:-${PROJECT_ROOT}/optuna}; study_prefix=${UPAIR_OPTUNA_BEST_STUDY_PREFIX:-clean_b32_iso_u34610_1dmrs_stageC}; missing files are rejected by default; set UPAIR_REQUIRE_OPTUNA_BEST=0 only for an intentional fallback"
echo "[SUBMIT] TensorFlow allocator: UPAIR_TF_GPU_ALLOCATOR=${UPAIR_TF_GPU_ALLOCATOR} TF_GPU_ALLOCATOR=${TF_GPU_ALLOCATOR:-<unset/default>}"

prev_dependency=""
last_train_job=""
for seed in "${seed_list[@]}"; do
  seed="$(echo "${seed}" | tr -d '[:space:]')"
  [[ -z "${seed}" ]] && continue
  seed_args=(
    --parsable
    "${account_args[@]}"
    "${gpu_args[@]}"
    --time="${TRAIN_WALLTIME}"
    --array="0-${array_max}%${ARRAY_CONCURRENCY}"
    --export="ALL,UPAIR_PUBLICATION_SEEDS=${seed},UPAIR_VARIANTS=${VARIANTS},UPAIR_DMRS_CASES=${DMRS_CASES},UPAIR_EVAL_USERS=${EVAL_USERS},UPAIR_USE_OPTUNA_BEST_1DMRS=${UPAIR_USE_OPTUNA_BEST_1DMRS:-auto},UPAIR_OPTUNA_BEST_STORAGE_DIR=${UPAIR_OPTUNA_BEST_STORAGE_DIR:-${PROJECT_ROOT}/optuna},UPAIR_OPTUNA_BEST_STUDY_PREFIX=${UPAIR_OPTUNA_BEST_STUDY_PREFIX:-clean_b32_iso_u34610_1dmrs_stageC},UPAIR_REQUIRE_OPTUNA_BEST=${UPAIR_REQUIRE_OPTUNA_BEST:-1},PROJECT_ROOT=${PROJECT_ROOT},VENV_PATH=${VENV_PATH}"
  )
  [[ -n "${prev_dependency}" ]] && seed_args+=(--dependency="${prev_dependency}")
  train_job=$(sbatch "${seed_args[@]}" slurm/narval/11_train_eval_1dmrs_u3_optuna_best.sbatch)
  echo "[SUBMIT] seed ${seed} train/eval array job=${train_job}"
  last_train_job="${train_job}"
  prev_dependency="afterok:${train_job}"
done

if [[ -z "${last_train_job}" ]]; then echo "[SUBMIT] ERROR: no seed jobs submitted" >&2; exit 2; fi
merge_job=$(sbatch --parsable "${account_args[@]}" "${partition_args[@]}" --time="${MERGE_WALLTIME}" --dependency="afterok:${last_train_job}" --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},VENV_PATH=${VENV_PATH}" slurm/narval/20_merge_plot.sbatch)
echo "[SUBMIT] merge/plot job=${merge_job}"
echo "[SUBMIT] monitor: squeue -u \"$USER\""
