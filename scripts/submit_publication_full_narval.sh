#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16}"
VENV_PATH="${VENV_PATH:-/home/rsadve1/scratch/.venvUPAIR}"
ACCOUNT="${NARVAL_SLURM_ACCOUNT:-def-rsadve_gpu}"
GPUS_PER_NODE="${NARVAL_GPUS_PER_NODE:-}"
SEEDS="${NARVAL_PUBLICATION_SEEDS:-7}"
VARIANTS="${UPAIR_VARIANTS:-main_d96_b4_r2,shallow_d96_b2_r2,deep_d96_b6_r2,narrow_d64_b4_r2,wide_d128_b4_r2,wide_deep_d128_b6_r2,mlpwide_d96_b4_r4}"
DMRS_CASES="${UPAIR_DMRS_CASES:-1dmrs,2dmrs}"
EVAL_USERS="${UPAIR_EVAL_USERS:-}"
ARRAY_CONCURRENCY="${NARVAL_PUBLICATION_ARRAY_CONCURRENCY:-12}"
TRAIN_WALLTIME="${NARVAL_PUBLICATION_WALLTIME:-12:00:00}"
MERGE_WALLTIME="${NARVAL_PUBLICATION_MERGE_WALLTIME:-02:00:00}"
cd "${PROJECT_ROOT}"
mkdir -p logs
export PROJECT_ROOT VENV_PATH UPAIR_PUBLICATION_SEEDS="${SEEDS}" UPAIR_VARIANTS="${VARIANTS}" UPAIR_DMRS_CASES="${DMRS_CASES}"
[[ -n "${EVAL_USERS}" ]] && export UPAIR_EVAL_USERS="${EVAL_USERS}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export MPLBACKEND=Agg
IFS=',' read -r -a seed_list <<< "${SEEDS}"
IFS=',' read -r -a variant_list <<< "${VARIANTS}"
IFS=',' read -r -a dmrs_list <<< "${DMRS_CASES}"
tasks_per_seed=$((${#variant_list[@]} * ${#dmrs_list[@]}))
array_max=$((tasks_per_seed - 1))
account_args=(); if [[ -n "${ACCOUNT}" ]]; then account_args=(--account="${ACCOUNT}"); fi
gpu_args=(); if [[ -n "${NARVAL_GRES:-gpu:1}" ]]; then gpu_args=(--gres="${NARVAL_GRES:-gpu:1}"); elif [[ -n "${GPUS_PER_NODE}" ]]; then gpu_args=(--gpus-per-node="${GPUS_PER_NODE}"); fi
echo "[SUBMIT] project=${PROJECT_ROOT}"
echo "[SUBMIT] venv=${VENV_PATH} account=${ACCOUNT:-none} gpu_args=${gpu_args[*]}"
echo "[SUBMIT] seeds=${SEEDS} variants=${VARIANTS} dmrs_cases=${DMRS_CASES} eval_users=${EVAL_USERS:-config} array=0-${array_max}"
prev_dependency=""
last_merge_job=""
for seed in "${seed_list[@]}"; do
  seed="$(echo "${seed}" | tr -d '[:space:]')"; [[ -z "${seed}" ]] && continue
  seed_args=(--parsable "${account_args[@]}" "${gpu_args[@]}" --time="${TRAIN_WALLTIME}" --array="0-${array_max}%${ARRAY_CONCURRENCY}" --export="ALL,UPAIR_PUBLICATION_SEEDS=${seed},UPAIR_VARIANTS=${VARIANTS},UPAIR_DMRS_CASES=${DMRS_CASES},UPAIR_EVAL_USERS=${EVAL_USERS},PROJECT_ROOT=${PROJECT_ROOT},VENV_PATH=${VENV_PATH}")
  [[ -n "${prev_dependency}" ]] && seed_args+=(--dependency="${prev_dependency}")
  seed_job=$(sbatch "${seed_args[@]}" slurm/narval/10_train_eval_array.sbatch)
  echo "[SUBMIT] seed ${seed} train/eval array job=${seed_job}"
  merge_job=$(sbatch --parsable "${account_args[@]}" "${partition_args[@]}" --time="${MERGE_WALLTIME}" --dependency="afterok:${seed_job}" --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},VENV_PATH=${VENV_PATH}" slurm/narval/20_merge_plot.sbatch)
  echo "[SUBMIT] seed ${seed} merge/plot job=${merge_job}"
  last_merge_job="${merge_job}"; prev_dependency="afterok:${merge_job}"
done
if [[ -z "${last_merge_job}" ]]; then echo "[SUBMIT] ERROR: no seed jobs submitted" >&2; exit 2; fi
echo "[SUBMIT] final merge/plot job=${last_merge_job}"
echo "[SUBMIT] monitor: squeue -u \"$USER\""
