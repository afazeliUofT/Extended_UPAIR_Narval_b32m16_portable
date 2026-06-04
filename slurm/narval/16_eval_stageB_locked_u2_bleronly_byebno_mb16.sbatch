#!/usr/bin/env bash
#SBATCH --job-name=upair-narval-u2-eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --signal=B:TERM@600

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16}"
VENV_PATH="${VENV_PATH:-/home/rsadve1/scratch/.venvUPAIR}"

cd "${PROJECT_ROOT}"
mkdir -p logs tmp_eval_configs

if [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
fi

if command -v module >/dev/null 2>&1; then
  module --force purge || true
  module load StdEnv/2023 || true
  module load gcc/12.3 || true
  module load python/3.11.5 || true
  module load cuda/12.2 || true
fi

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PIP_NO_USER=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=1
export MPLBACKEND=Agg

# Evaluation-only allocator. No training is performed in this job.
export TF_GPU_ALLOCATOR="${UPAIR_EVAL_TF_GPU_ALLOCATOR:-cuda_malloc_async}"

# Force local source tree.
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

source "${VENV_PATH}/bin/activate"

VARIANT="${UPAIR_VARIANTS:?UPAIR_VARIANTS must be set to exactly one variant}"
SEED="${UPAIR_PUBLICATION_SEEDS:-7}"
DMRS_CASE="${UPAIR_DMRS_CASES:-1dmrs}"
EVAL_USERS="${UPAIR_EVAL_USERS:-2}"
BASE_CONFIG="${UPAIR_COMPREHENSIVE_CONFIG:-${PROJECT_ROOT}/configs/twc_comprehensive_mu32_base_narval_stageB_locked_u2_bleronly_compiled_mb16.yaml}"
PREFIX="${UPAIR_OPTUNA_BEST_STUDY_PREFIX:-clean_b32_iso_u34610_1dmrs_stageB_locked}"
OPTUNA_DIR="${UPAIR_OPTUNA_BEST_STORAGE_DIR:-${PROJECT_ROOT}/optuna}"

echo "[NARVAL-U2] job=${SLURM_JOB_ID:-NA} node=${SLURMD_NODENAME:-NA}"
echo "[NARVAL-U2] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[NARVAL-U2] VENV_PATH=${VENV_PATH}"
echo "[NARVAL-U2] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-NA}"
echo "[NARVAL-U2] EVAL_TF_GPU_ALLOCATOR=${TF_GPU_ALLOCATOR}"
echo "[NARVAL-U2] VARIANT=${VARIANT}"
echo "[NARVAL-U2] EVAL_USERS=${EVAL_USERS}"
echo "[NARVAL-U2] BASE_CONFIG=${BASE_CONFIG}"
echo "[NARVAL-U2] OPTUNA_PREFIX=${PREFIX}"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[NARVAL-U2] GPU identity check:"
  nvidia-smi --query-gpu=name,memory.total,mig.mode.current,uuid --format=csv || true
fi

echo
echo "============================================================"
echo "[NARVAL-U2] Checking trained checkpoint"
echo "============================================================"

python - "$VARIANT" <<'PY_CHECK'
import json
import sys
from pathlib import Path

variant = sys.argv[1]
run_dir = Path("TWC_plots_comprehensive/runs_rx16/seed7/1dmrs") / variant
state_path = run_dir / "metrics" / "train_state.json"
best_path = run_dir / "checkpoints" / "best.weights.h5"

print("run_dir:", run_dir)
print("state_path:", state_path)
print("best_path:", best_path)
print("best_exists:", best_path.exists())

if not state_path.exists():
    print("ERROR: missing train_state.json. This eval-only job requires a completed trained model.")
    raise SystemExit(10)

state = json.load(open(state_path, "r", encoding="utf-8"))
print("latest_step:", state.get("latest_step", state.get("completed_step")))
print("total_steps:", state.get("total_steps", state.get("steps")))
print("training_complete:", state.get("training_complete"))
print("best_val:", state.get("best_val"))

if not best_path.exists():
    print("ERROR: missing best.weights.h5. This eval-only job requires trained weights.")
    raise SystemExit(11)
PY_CHECK

echo
echo "============================================================"
echo "[NARVAL-U2] BLER-only compiled 2-user eval by Eb/N0"
echo "============================================================"

export UPAIR_EVAL_IGNORE_SHARED_CSV=1
export UPAIR_COMPREHENSIVE_EVAL_ONLY=1
export UPAIR_COMPREHENSIVE_FORCE=0

mapfile -t EBNO_LIST < <(
  python - "$BASE_CONFIG" <<'PY_EBNO_LIST'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
vals = cfg.get("system", {}).get("ebno_db_eval", [-5,-4,-3,-2,-1,0,0.5])
for v in vals:
    print(float(v))
PY_EBNO_LIST
)

for EBNO in "${EBNO_LIST[@]}"; do
  echo
  echo "============================================================"
  echo "[NARVAL-U2] evaluating Eb/N0=${EBNO} dB for 2 users in a fresh Python process"
  echo "============================================================"

  TMP_CFG="tmp_eval_configs/${VARIANT}_u${EVAL_USERS}_stageB_locked_bleronly_ebno_${EBNO}_job${SLURM_JOB_ID:-manual}.yaml"

  python - "$BASE_CONFIG" "$TMP_CFG" "$EBNO" "$EVAL_USERS" <<'PY_TMP_CFG'
import sys, yaml

src, dst, ebno, eval_users = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
cfg = yaml.safe_load(open(src, "r", encoding="utf-8"))

cfg.setdefault("system", {})
cfg["system"]["ebno_db_eval"] = [ebno]
cfg["system"]["batch_size_eval"] = 64

cfg.setdefault("multiuser", {})
cfg["multiuser"]["fixed_num_users"] = eval_users
cfg["multiuser"]["eval_num_users"] = [eval_users]

cfg.setdefault("evaluation", {})
cfg["evaluation"]["nmse_receivers"] = []
cfg["evaluation"]["logical_batch_size"] = 64
cfg["evaluation"]["receiver_microbatch_size"] = 16
cfg["evaluation"]["stream_eval_microbatches"] = True
cfg["evaluation"]["compiled_receiver_error_counts"] = True
cfg["evaluation"]["receiver_call_jit_compile"] = False
cfg["evaluation"]["memory_cleanup_every_batches"] = 1
cfg["evaluation"]["memory_cleanup_every_microbatch"] = True
cfg["evaluation"]["save_example_batch"] = False
cfg["evaluation"]["resume"] = True
cfg["evaluation"]["log_latency"] = True
cfg["evaluation"]["log_gpu_memory"] = True

with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(dst)
PY_TMP_CFG

  python scripts/run_comprehensive_mu32_ablation.py \
    --config "${TMP_CFG}" \
    --seeds "${SEED}" \
    --dmrs-cases "${DMRS_CASE}" \
    --variants "${VARIANT}" \
    --eval-users "${EVAL_USERS}" \
    --eval-only \
    --use-optuna-best-1dmrs \
    --require-optuna-best \
    --optuna-best-storage-dir "${OPTUNA_DIR}" \
    --optuna-best-study-prefix "${PREFIX}" \
    --no-global-summary

  echo "[NARVAL-U2] finished Eb/N0=${EBNO} dB for u${EVAL_USERS}; Python process exited, GPU memory released."
done

echo
echo "[NARVAL-U2] completed 2-user evaluation for ${VARIANT}"
