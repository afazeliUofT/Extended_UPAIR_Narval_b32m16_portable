#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16}"
VENV_PATH="${VENV_PATH:-/home/rsadve1/scratch/.venvUPAIR}"
ACCOUNT="${NARVAL_SLURM_ACCOUNT:-def-rsadve_gpu}"
GPUS_PER_NODE="${NARVAL_GPUS_PER_NODE:-}"
SEEDS="${NARVAL_PUBLICATION_SEEDS:-7}"
ARRAY_CONCURRENCY="${NARVAL_OPTUNA_BEST_ARRAY_CONCURRENCY:-7}"
OPTUNA_ARRAY_CONCURRENCY="${NARVAL_OPTUNA_ARRAY_CONCURRENCY:-7}"
OPTUNA_WALLTIME="${NARVAL_OPTUNA_WALLTIME:-12:00:00}"
TRAIN_WALLTIME="${NARVAL_OPTUNA_BEST_WALLTIME:-12:00:00}"
MERGE_WALLTIME="${NARVAL_OPTUNA_BEST_MERGE_WALLTIME:-02:00:00}"
VARIANTS="${UPAIR_VARIANTS:-main_d96_b4_r2,shallow_d96_b2_r2,deep_d96_b6_r2,narrow_d64_b4_r2,wide_d128_b4_r2,wide_deep_d128_b6_r2,mlpwide_d96_b4_r4}"
EVAL_USERS="${UPAIR_EVAL_USERS:-3}"
DMRS_CASES="${UPAIR_DMRS_CASES:-1dmrs}"
COMPREHENSIVE_CONFIG="${UPAIR_COMPREHENSIVE_CONFIG:-configs/twc_comprehensive_mu32_base.yaml}"
OPTUNA_CONFIG="${UPAIR_OPTUNA_CONFIG:-${COMPREHENSIVE_CONFIG}}"
OPTUNA_STAGE="${OPTUNA_STAGE:-A}"
OPTUNA_STUDY_PREFIX="${OPTUNA_STUDY_PREFIX:-clean_b32_iso_u34610_1dmrs_stageA}"
UPAIR_OPTUNA_BEST_STUDY_PREFIX="${UPAIR_OPTUNA_BEST_STUDY_PREFIX:-clean_b32_iso_u34610_1dmrs_stageC}"
UPAIR_REQUIRE_OPTUNA_BEST="${UPAIR_REQUIRE_OPTUNA_BEST:-1}"

cd "${PROJECT_ROOT}"
mkdir -p logs optuna
export PROJECT_ROOT VENV_PATH UPAIR_VARIANTS="${VARIANTS}" UPAIR_EVAL_USERS="${EVAL_USERS}" UPAIR_DMRS_CASES="${DMRS_CASES}" UPAIR_COMPREHENSIVE_CONFIG="${COMPREHENSIVE_CONFIG}" UPAIR_OPTUNA_CONFIG="${OPTUNA_CONFIG}"
export OPTUNA_TF_GPU_ALLOCATOR="${OPTUNA_TF_GPU_ALLOCATOR:-default}"
export UPAIR_TF_GPU_ALLOCATOR="${UPAIR_TF_GPU_ALLOCATOR:-default}"
if [[ "${OPTUNA_TF_GPU_ALLOCATOR}" == "default" || -z "${OPTUNA_TF_GPU_ALLOCATOR}" ]]; then
  unset TF_GPU_ALLOCATOR
else
  export TF_GPU_ALLOCATOR="${OPTUNA_TF_GPU_ALLOCATOR}"
fi
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export MPLBACKEND=Agg

account_args=()
if [[ -n "${ACCOUNT}" ]]; then account_args=(--account="${ACCOUNT}"); fi
partition_args=()
if [[ -n "${NARVAL_SLURM_PARTITION:-}" ]]; then partition_args=(--partition="${NARVAL_SLURM_PARTITION}"); fi
gpu_args=()
if [[ -n "${NARVAL_GRES:-gpu:1}" ]]; then
  gpu_args=(--gres="${NARVAL_GRES:-gpu:1}")
elif [[ -n "${GPUS_PER_NODE}" ]]; then
  gpu_args=(--gpus-per-node="${GPUS_PER_NODE}")
fi

IFS=',' read -r -a variant_list <<< "${VARIANTS}"
variant_count=0
for v in "${variant_list[@]}"; do [[ -n "$(echo "$v" | tr -d '[:space:]')" ]] && ((variant_count+=1)); done
if (( variant_count <= 0 )); then echo "[SUBMIT] ERROR: no variants selected" >&2; exit 2; fi

IFS=',' read -r -a dmrs_list <<< "${DMRS_CASES}"
dmrs_count=0
for c in "${dmrs_list[@]}"; do [[ -n "$(echo "$c" | tr -d '[:space:]')" ]] && ((dmrs_count+=1)); done
if (( dmrs_count <= 0 )); then echo "[SUBMIT] ERROR: no DMRS cases selected" >&2; exit 2; fi

IFS=',' read -r -a seed_list <<< "${SEEDS}"
seed_count=0
for s in "${seed_list[@]}"; do [[ -n "$(echo "$s" | tr -d '[:space:]')" ]] && ((seed_count+=1)); done
if (( seed_count <= 0 )); then echo "[SUBMIT] ERROR: no seeds selected" >&2; exit 2; fi

optuna_last_index=$((variant_count - 1))
train_last_index=$((variant_count * dmrs_count * seed_count - 1))

echo "[SUBMIT] project=${PROJECT_ROOT}"
echo "[SUBMIT] venv=${VENV_PATH}"
echo "[SUBMIT] account=${ACCOUNT:-none} partition=${NARVAL_SLURM_PARTITION:-auto}"
echo "[SUBMIT] gpu_args=${gpu_args[*]}"
echo "[SUBMIT] seeds=${SEEDS} variants=${VARIANTS} dmrs_cases=${DMRS_CASES} eval_users=${EVAL_USERS}"
echo "[SUBMIT] comprehensive_config=${COMPREHENSIVE_CONFIG}"
echo "[SUBMIT] optuna_config=${OPTUNA_CONFIG}"
echo "[SUBMIT] optuna_stage=${OPTUNA_STAGE} optuna_study_prefix=${OPTUNA_STUDY_PREFIX}"
echo "[SUBMIT] optuna defaults: train_batch=${OPTUNA_TRAIN_BATCH_SIZE:-32} val_batch=${OPTUNA_VALIDATION_BATCH_SIZE:-32} val_microbatch=${OPTUNA_VALIDATION_MICROBATCH_SIZE:-16} train_user_weights=${OPTUNA_TRAIN_USER_COUNT_WEIGHTS:-1,3,6,10} objective_min_step=${OPTUNA_OBJECTIVE_MIN_STEP:-1000}"
echo "[SUBMIT] optuna_best_prefix=${UPAIR_OPTUNA_BEST_STUDY_PREFIX} require_best=${UPAIR_REQUIRE_OPTUNA_BEST}"
echo "[SUBMIT] optuna array=0-${optuna_last_index}; train/eval array=0-${train_last_index}"

dependency_args=()
if [[ "${NARVAL_SKIP_OPTUNA:-0}" != "1" ]]; then
  optuna_job=$(sbatch --parsable \
    "${account_args[@]}" \
    "${gpu_args[@]}" \
    --time="${OPTUNA_WALLTIME}" \
    --array="0-${optuna_last_index}%${OPTUNA_ARRAY_CONCURRENCY}" \
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},VENV_PATH=${VENV_PATH},UPAIR_VARIANTS=${VARIANTS},UPAIR_OPTUNA_CONFIG=${OPTUNA_CONFIG},OPTUNA_STAGE=${OPTUNA_STAGE},OPTUNA_STUDY_PREFIX=${OPTUNA_STUDY_PREFIX},OPTUNA_TF_GPU_ALLOCATOR=${OPTUNA_TF_GPU_ALLOCATOR}" \
    slurm/narval/optuna_1dmrs_structures.sbatch)
  echo "[SUBMIT] optuna array job=${optuna_job}"
  dependency_args=(--dependency="afterok:${optuna_job}")
else
  echo "[SUBMIT] Optuna skipped because NARVAL_SKIP_OPTUNA=1; train/eval will require optuna/ stage-C artifacts unless UPAIR_REQUIRE_OPTUNA_BEST is changed."
fi

train_job=$(sbatch --parsable \
  "${account_args[@]}" \
  "${partition_args[@]}" \
  "${gpu_args[@]}" \
  --time="${TRAIN_WALLTIME}" \
  "${dependency_args[@]}" \
  --array="0-${train_last_index}%${ARRAY_CONCURRENCY}" \
  --export="ALL,UPAIR_PUBLICATION_SEEDS=${SEEDS},PROJECT_ROOT=${PROJECT_ROOT},VENV_PATH=${VENV_PATH},UPAIR_VARIANTS=${VARIANTS},UPAIR_EVAL_USERS=${EVAL_USERS},UPAIR_DMRS_CASES=${DMRS_CASES},UPAIR_COMPREHENSIVE_CONFIG=${COMPREHENSIVE_CONFIG},UPAIR_OPTUNA_BEST_STUDY_PREFIX=${UPAIR_OPTUNA_BEST_STUDY_PREFIX},UPAIR_REQUIRE_OPTUNA_BEST=${UPAIR_REQUIRE_OPTUNA_BEST},UPAIR_TF_GPU_ALLOCATOR=${UPAIR_TF_GPU_ALLOCATOR}" \
  slurm/narval/11_train_eval_1dmrs_u3_optuna_best.sbatch)
echo "[SUBMIT] train/eval array job=${train_job}"

merge_job=$(sbatch --parsable \
  "${account_args[@]}" \
  --time="${MERGE_WALLTIME}" \
  --dependency="afterok:${train_job}" \
  --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},VENV_PATH=${VENV_PATH}" \
  slurm/narval/20_merge_plot.sbatch)
echo "[SUBMIT] merge/plot job=${merge_job}"
echo "[SUBMIT] monitor: squeue -u \"$USER\""
