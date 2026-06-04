#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16}"
VENV_PATH="${VENV_PATH:-/home/rsadve1/scratch/.venvUPAIR}"
ACCOUNT="${NARVAL_SLURM_ACCOUNT:-def-rsadve_gpu}"
GPUS_PER_NODE="${NARVAL_GPUS_PER_NODE:-}"
ARRAY_CONCURRENCY="${NARVAL_OPTUNA_ARRAY_CONCURRENCY:-7}"
OPTUNA_WALLTIME="${NARVAL_OPTUNA_WALLTIME:-12:00:00}"
OPTUNA_STAGE="$(echo "${OPTUNA_STAGE:-A}" | tr '[:lower:]' '[:upper:]')"
VARIANTS="${UPAIR_VARIANTS:-main_d96_b4_r2,shallow_d96_b2_r2,deep_d96_b6_r2,narrow_d64_b4_r2,wide_d128_b4_r2,wide_deep_d128_b6_r2,mlpwide_d96_b4_r4}"

cd "${PROJECT_ROOT}"
mkdir -p logs optuna

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

count_csv(){ local value="$1"; local count=0; IFS=',' read -r -a arr <<< "$value"; for item in "${arr[@]}"; do [[ -n "$(echo "$item" | tr -d '[:space:]')" ]] && ((count+=1)); done; echo "$count"; }
variant_count=$(count_csv "${VARIANTS}")
if (( variant_count <= 0 )); then echo "[SUBMIT] ERROR: no variants selected" >&2; exit 2; fi
last_index=$((variant_count - 1))

export PROJECT_ROOT VENV_PATH UPAIR_VARIANTS="${VARIANTS}" OPTUNA_STAGE
# Use isolated one-trial TensorFlow workers by default.  Do not pass through
# TF_GPU_ALLOCATOR=cuda_malloc_async unless explicitly requested.
export OPTUNA_TRIAL_PROCESS_ISOLATION="${OPTUNA_TRIAL_PROCESS_ISOLATION:-1}"
export OPTUNA_TF_GPU_ALLOCATOR="${OPTUNA_TF_GPU_ALLOCATOR:-default}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export MPLBACKEND=Agg
if [[ "${OPTUNA_TF_GPU_ALLOCATOR}" == "default" || -z "${OPTUNA_TF_GPU_ALLOCATOR}" ]]; then
  unset TF_GPU_ALLOCATOR
else
  export TF_GPU_ALLOCATOR="${OPTUNA_TF_GPU_ALLOCATOR}"
fi

case "${OPTUNA_STAGE}" in
  A) default_prefix="clean_b32_iso_u34610_1dmrs_stageA"; default_target="${OPTUNA_STAGE_A_TRIALS:-30}"; default_steps="${OPTUNA_STAGE_A_STEPS:-6000}" ;;
  B) default_prefix="clean_b32_iso_u34610_1dmrs_stageB"; default_target="${OPTUNA_STAGE_B_TRIALS:-8}"; default_steps="${OPTUNA_STAGE_B_STEPS:-10000}" ;;
  C) default_prefix="clean_b32_iso_u34610_1dmrs_stageC"; default_target="${OPTUNA_STAGE_C_TRIALS:-3}"; default_steps="${OPTUNA_STAGE_C_STEPS:-20000}" ;;
  *) echo "[SUBMIT] ERROR: OPTUNA_STAGE must be A, B, or C; got ${OPTUNA_STAGE}" >&2; exit 2 ;;
esac
export OPTUNA_STUDY_PREFIX="${OPTUNA_STUDY_PREFIX:-${default_prefix}}"
export OPTUNA_TARGET_TOTAL_TRIALS="${OPTUNA_TARGET_TOTAL_TRIALS:-${default_target}}"
export OPTUNA_N_TRIALS_PER_JOB="${OPTUNA_N_TRIALS_PER_JOB:-${default_target}}"
export OPTUNA_STEPS="${OPTUNA_STEPS:-${default_steps}}"

if [[ "${OPTUNA_STAGE}" == "B" ]]; then
  export OPTUNA_SOURCE_STUDY_PREFIX="${OPTUNA_SOURCE_STUDY_PREFIX:-clean_b32_iso_u34610_1dmrs_stageA}"
elif [[ "${OPTUNA_STAGE}" == "C" ]]; then
  export OPTUNA_SOURCE_STUDY_PREFIX="${OPTUNA_SOURCE_STUDY_PREFIX:-clean_b32_iso_u34610_1dmrs_stageB}"
fi

echo "[SUBMIT] project=${PROJECT_ROOT}"
echo "[SUBMIT] venv=${VENV_PATH}"
echo "[SUBMIT] account=${ACCOUNT:-none} partition=${NARVAL_SLURM_PARTITION:-auto}"
echo "[SUBMIT] gpu_args=${gpu_args[*]}"
echo "[SUBMIT] Optuna clean 1-DMRS stage=${OPTUNA_STAGE} variants=${VARIANTS}"
echo "[SUBMIT] study_prefix=${OPTUNA_STUDY_PREFIX} target=${OPTUNA_TARGET_TOTAL_TRIALS} steps=${OPTUNA_STEPS}"
echo "[SUBMIT] train_batch=${OPTUNA_TRAIN_BATCH_SIZE:-32} val_batch=${OPTUNA_VALIDATION_BATCH_SIZE:-32} val_microbatch=${OPTUNA_VALIDATION_MICROBATCH_SIZE:-16} train_user_weights=${OPTUNA_TRAIN_USER_COUNT_WEIGHTS:-1,3,6,10} val_user_weights=${OPTUNA_VAL_USER_COUNT_WEIGHTS:-1,2,3,4} objective_min_step=${OPTUNA_OBJECTIVE_MIN_STEP:-1000}"
echo "[SUBMIT] array=0-${last_index} concurrency=${ARRAY_CONCURRENCY} walltime=${OPTUNA_WALLTIME}"
echo "[SUBMIT] trial_process_isolation=${OPTUNA_TRIAL_PROCESS_ISOLATION} OPTUNA_TF_GPU_ALLOCATOR=${OPTUNA_TF_GPU_ALLOCATOR} TF_GPU_ALLOCATOR=${TF_GPU_ALLOCATOR:-<unset/default>}"
echo "[SUBMIT] resume: completed/pruned trials stay in optuna/*.db; ResourceExhausted trials are FAIL and do not count; interrupted RUNNING trials resume from train_state.json."

job_id=$(sbatch --parsable \
  "${account_args[@]}" \
  "${partition_args[@]}" \
  "${gpu_args[@]}" \
  --time="${OPTUNA_WALLTIME}" \
  --array="0-${last_index}%${ARRAY_CONCURRENCY}" \
  --export=ALL \
  slurm/narval/optuna_1dmrs_structures.sbatch)

echo "[SUBMIT] optuna job=${job_id}"
echo "[SUBMIT] monitor: squeue -u \"$USER\""
