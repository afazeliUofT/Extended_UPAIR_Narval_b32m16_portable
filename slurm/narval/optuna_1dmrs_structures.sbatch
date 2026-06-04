#!/usr/bin/env bash
#SBATCH --job-name=optuna-1dmrs
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
mkdir -p logs optuna

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

export PYTHONNOUSERSITE=1 PIP_NO_USER=1 PYTHONUNBUFFERED=1
# Default Optuna runs use TensorFlow's default GPU allocator and one fresh worker
# process per trial.  Do not inherit a previous TF_GPU_ALLOCATOR=cuda_malloc_async
# setting unless explicitly requested, because the uploaded logs showed CUDA async
# mempool reservation poisoning across many sequential trials.
if [[ "${OPTUNA_TF_GPU_ALLOCATOR:-default}" == "default" || -z "${OPTUNA_TF_GPU_ALLOCATOR:-}" ]]; then
  unset TF_GPU_ALLOCATOR
else
  export TF_GPU_ALLOCATOR="${OPTUNA_TF_GPU_ALLOCATOR}"
fi
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
export MPLBACKEND=Agg
unset PYTHONPATH
unset PIP_USER
source "${VENV_PATH}/bin/activate"

DEFAULT_VARIANTS="main_d96_b4_r2,shallow_d96_b2_r2,deep_d96_b6_r2,narrow_d64_b4_r2,wide_d128_b4_r2,wide_deep_d128_b6_r2,mlpwide_d96_b4_r4"
IFS=',' read -r -a variants <<< "${UPAIR_VARIANTS:-${DEFAULT_VARIANTS}}"
idx="${SLURM_ARRAY_TASK_ID}"
if (( idx >= ${#variants[@]} )); then echo "[OPTUNA-SLURM] task=${idx} outside variant grid ${#variants[@]}; exiting."; exit 0; fi
variant="$(echo "${variants[$idx]}" | tr -d '[:space:]')"
stage="$(echo "${OPTUNA_STAGE:-A}" | tr '[:lower:]' '[:upper:]')"

case "${stage}" in
  A)
    default_prefix="clean_b32_iso_u34610_1dmrs_stageA"
    default_steps="${OPTUNA_STAGE_A_STEPS:-6000}"
    default_target="${OPTUNA_STAGE_A_TRIALS:-30}"
    default_source_top_k="0"
    source_args=()
    ;;
  B)
    default_prefix="clean_b32_iso_u34610_1dmrs_stageB"
    default_steps="${OPTUNA_STAGE_B_STEPS:-10000}"
    default_target="${OPTUNA_STAGE_B_TRIALS:-8}"
    default_source_top_k="${OPTUNA_STAGE_B_SOURCE_TOP_K:-8}"
    source_prefix="${OPTUNA_SOURCE_STUDY_PREFIX:-clean_b32_iso_u34610_1dmrs_stageA}"
    source_study_name="${source_prefix}_${variant}"
    source_storage="${OPTUNA_SOURCE_STORAGE:-sqlite:///${PROJECT_ROOT}/optuna/${source_study_name}.db}"
    source_args=(--source-study-name "${source_study_name}" --source-storage "${source_storage}")
    ;;
  C)
    default_prefix="clean_b32_iso_u34610_1dmrs_stageC"
    default_steps="${OPTUNA_STAGE_C_STEPS:-20000}"
    default_target="${OPTUNA_STAGE_C_TRIALS:-3}"
    default_source_top_k="${OPTUNA_STAGE_C_SOURCE_TOP_K:-3}"
    source_prefix="${OPTUNA_SOURCE_STUDY_PREFIX:-clean_b32_iso_u34610_1dmrs_stageB}"
    source_study_name="${source_prefix}_${variant}"
    source_storage="${OPTUNA_SOURCE_STORAGE:-sqlite:///${PROJECT_ROOT}/optuna/${source_study_name}.db}"
    source_args=(--source-study-name "${source_study_name}" --source-storage "${source_storage}")
    ;;
  *)
    echo "[OPTUNA-SLURM] ERROR: OPTUNA_STAGE must be A, B, or C; got ${stage}" >&2
    exit 2
    ;;
esac

study_prefix="${OPTUNA_STUDY_PREFIX:-${default_prefix}}"
study_name="${study_prefix}_${variant}"
storage="sqlite:///${PROJECT_ROOT}/optuna/${study_name}.db"

n_trials="${OPTUNA_N_TRIALS_PER_JOB:-${default_target}}"
target_trials="${OPTUNA_TARGET_TOTAL_TRIALS:-${default_target}}"
steps="${OPTUNA_STEPS:-${default_steps}}"
source_top_k="${OPTUNA_SOURCE_TOP_K:-${default_source_top_k}}"

extra_args=()
if [[ "${OPTUNA_DISABLE_PRUNING:-0}" == "1" ]]; then extra_args+=(--disable-pruning); fi
if [[ "${OPTUNA_CONSTANT_LIAR:-0}" == "1" ]]; then extra_args+=(--constant-liar); fi
if [[ "${OPTUNA_VAL_MEMORY_CLEANUP_EVERY_BATCH:-1}" == "0" ]]; then
  extra_args+=(--no-val-memory-cleanup-every-batch)
else
  extra_args+=(--val-memory-cleanup-every-batch)
fi

printf '[OPTUNA-SLURM] task=%s variant=%s stage=%s\n' "${SLURM_ARRAY_TASK_ID}" "${variant}" "${stage}"
printf '[OPTUNA-SLURM] study=%s storage=%s\n' "${study_name}" "${storage}"
printf '[OPTUNA-SLURM] target=%s n_trials_this_job=%s steps=%s source_top_k=%s\n' "${target_trials}" "${n_trials}" "${steps}" "${source_top_k}"
if [[ "${OPTUNA_TRIAL_PROCESS_ISOLATION:-1}" == "0" ]]; then
  optuna_driver="scripts/run_optuna_1dmrs_structure.py"
else
  optuna_driver="scripts/run_optuna_1dmrs_structure_isolated.py"
fi
printf '[OPTUNA-SLURM] driver=%s trial_process_isolation=%s TF_GPU_ALLOCATOR=%s\n' "${optuna_driver}" "${OPTUNA_TRIAL_PROCESS_ISOLATION:-1}" "${TF_GPU_ALLOCATOR:-<unset/default>}"
if [[ ${#source_args[@]} -gt 0 ]]; then printf '[OPTUNA-SLURM] source=%s\n' "${source_args[*]}"; fi
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi || true; fi

child_pid=""
forward_term(){ echo "[OPTUNA-SLURM] received signal; forwarding TERM to Python worker ${child_pid}" >&2; [[ -n "${child_pid}" ]] && kill -TERM "${child_pid}" 2>/dev/null || true; [[ -n "${child_pid}" ]] && wait "${child_pid}" || true; exit 143; }
trap forward_term TERM INT
python "${optuna_driver}" \
  --config "${UPAIR_OPTUNA_CONFIG:-configs/twc_comprehensive_mu32_base.yaml}" \
  --variant "${variant}" \
  --study-name "${study_name}" \
  --storage "${storage}" \
  --stage "${stage}" \
  --n-trials "${n_trials}" \
  --target-total-trials "${target_trials}" \
  --steps "${steps}" \
  --source-top-k "${source_top_k}" \
  "${source_args[@]}" \
  --train-batch-size "${OPTUNA_TRAIN_BATCH_SIZE:-32}" \
  --validation-batch-size "${OPTUNA_VALIDATION_BATCH_SIZE:-32}" \
  --validation-microbatch-size "${OPTUNA_VALIDATION_MICROBATCH_SIZE:-16}" \
  --train-user-count-weights "${OPTUNA_TRAIN_USER_COUNT_WEIGHTS:-1,3,6,10}" \
  --val-user-counts "${OPTUNA_VAL_USER_COUNTS:-1,2,3,4}" \
  --val-user-count-weights "${OPTUNA_VAL_USER_COUNT_WEIGHTS:-1,2,3,4}" \
  --eval-every "${OPTUNA_EVAL_EVERY:-1000}" \
  --checkpoint-every "${OPTUNA_CHECKPOINT_EVERY:-1000}" \
  --log-every "${OPTUNA_LOG_EVERY:-100}" \
  --val-steps "${OPTUNA_VAL_STEPS:-96}" \
  --val-ebno-db="${OPTUNA_VAL_EBNO_DB:--4,-2,0,2,4}" \
  --objective-metric "${OPTUNA_OBJECTIVE_METRIC:-hybrid_nmse_ratio}" \
  --objective-aggregation "${OPTUNA_OBJECTIVE_AGGREGATION:-recent_mean}" \
  --objective-recent-k "${OPTUNA_OBJECTIVE_RECENT_K:-2}" \
  --objective-min-step "${OPTUNA_OBJECTIVE_MIN_STEP:-1000}" \
  --pruner "${OPTUNA_PRUNER:-percentile}" \
  --pruner-percentile "${OPTUNA_PRUNER_PERCENTILE:-25}" \
  --pruner-startup-trials "${OPTUNA_PRUNER_STARTUP_TRIALS:-10}" \
  --pruner-min-trials "${OPTUNA_PRUNER_MIN_TRIALS:-5}" \
  --prune-warmup-steps "${OPTUNA_PRUNE_WARMUP_STEPS:-1000}" \
  --prune-interval-steps "${OPTUNA_PRUNE_INTERVAL_STEPS:-1000}" \
  --tpe-startup-trials "${OPTUNA_TPE_STARTUP_TRIALS:-12}" \
  --seed "${OPTUNA_SEED:-7}" \
  "${extra_args[@]}" &
child_pid=$!
wait "${child_pid}"
