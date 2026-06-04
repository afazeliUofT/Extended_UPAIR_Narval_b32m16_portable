#!/usr/bin/env bash
#SBATCH --job-name=upair-narval-stageB-tr-eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=24:00:00
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
# Training should use the default TensorFlow allocator on A100 40GB.
# cuda_malloc_async can reserve almost the whole CUDA mempool during training.
if [[ "${UPAIR_TRAIN_TF_GPU_ALLOCATOR:-default}" == "default" || -z "${UPAIR_TRAIN_TF_GPU_ALLOCATOR:-}" ]]; then
  unset TF_GPU_ALLOCATOR
else
  export TF_GPU_ALLOCATOR="${UPAIR_TRAIN_TF_GPU_ALLOCATOR}"
fi

unset PYTHONPATH
source "${VENV_PATH}/bin/activate"

VARIANT="${UPAIR_VARIANTS:?UPAIR_VARIANTS must be set to exactly one variant}"
SEED="${UPAIR_PUBLICATION_SEEDS:-7}"
DMRS_CASE="${UPAIR_DMRS_CASES:-1dmrs}"
EVAL_USERS="${UPAIR_EVAL_USERS:-3}"
BASE_CONFIG="${UPAIR_COMPREHENSIVE_CONFIG:-${PROJECT_ROOT}/configs/twc_comprehensive_mu32_base_narval_stageB_locked_bleronly_compiled_mb16.yaml}"
PREFIX="${UPAIR_OPTUNA_BEST_STUDY_PREFIX:-clean_b32_iso_u34610_1dmrs_stageB_locked}"
OPTUNA_DIR="${UPAIR_OPTUNA_BEST_STORAGE_DIR:-${PROJECT_ROOT}/optuna}"

echo "[NARVAL-COMBINED] job=${SLURM_JOB_ID:-NA} node=${SLURMD_NODENAME:-NA}"
echo "[NARVAL-COMBINED] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[NARVAL-COMBINED] VENV_PATH=${VENV_PATH}"
echo "[NARVAL-COMBINED] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-NA}"
echo "[NARVAL-COMBINED] TRAIN_TF_GPU_ALLOCATOR=${TF_GPU_ALLOCATOR:-<unset/default>}"
echo "[NARVAL-COMBINED] VARIANT=${VARIANT}"
echo "[NARVAL-COMBINED] BASE_CONFIG=${BASE_CONFIG}"
echo "[NARVAL-COMBINED] OPTUNA_PREFIX=${PREFIX}"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[NARVAL-COMBINED] GPU identity check:"
  nvidia-smi --query-gpu=name,memory.total,mig.mode.current,uuid --format=csv || true
fi

echo
echo "============================================================"
echo "[NARVAL-COMBINED] PHASE 1: train/resume with Stage-B locked params; skip unsafe built-in final eval"
echo "============================================================"

unset UPAIR_COMPREHENSIVE_EVAL_ONLY
export UPAIR_COMPREHENSIVE_SKIP_FINAL_EVAL=1
export UPAIR_COMPREHENSIVE_FORCE=0

python scripts/run_comprehensive_mu32_ablation.py \
  --config "${BASE_CONFIG}" \
  --seeds "${SEED}" \
  --dmrs-cases "${DMRS_CASE}" \
  --variants "${VARIANT}" \
  --eval-users "${EVAL_USERS}" \
  --use-optuna-best-1dmrs \
  --require-optuna-best \
  --optuna-best-storage-dir "${OPTUNA_DIR}" \
  --optuna-best-study-prefix "${PREFIX}" \
  --no-global-summary

echo
echo "============================================================"
echo "[NARVAL-COMBINED] Checking whether training is complete"
echo "============================================================"

set +e
python - "$VARIANT" <<'PY_TRAIN_CHECK'
import json
import sys
from pathlib import Path

variant = sys.argv[1]
run_dir = Path("TWC_plots_comprehensive/runs_rx16/seed7/1dmrs") / variant
state_path = run_dir / "metrics" / "train_state.json"
best_path = run_dir / "checkpoints" / "best.weights.h5"

print("state_path:", state_path)
print("best_path:", best_path)
print("best_exists:", best_path.exists())

if not state_path.exists():
    print("TRAINING_INCOMPLETE: train_state.json not found")
    raise SystemExit(10)

state = json.load(open(state_path, "r", encoding="utf-8"))
latest = state.get("latest_step", state.get("completed_step"))
total = state.get("total_steps", state.get("steps"))
complete = state.get("training_complete")

print("latest_step:", latest)
print("total_steps:", total)
print("training_complete:", complete)
print("best_val:", state.get("best_val"))

if not best_path.exists():
    print("TRAINING_INCOMPLETE: best.weights.h5 not found")
    raise SystemExit(10)

if complete is True:
    raise SystemExit(0)

try:
    if latest is not None and total is not None and int(latest) >= int(total):
        print("training_complete inferred from latest_step >= total_steps")
        raise SystemExit(0)
except Exception:
    pass

print("TRAINING_INCOMPLETE: training_complete is not True")
raise SystemExit(10)
PY_TRAIN_CHECK
rc=$?
set -e

if [[ "${rc}" == "10" ]]; then
  echo "[NARVAL-COMBINED] Training is not complete yet. Resubmit the same task later."
  exit 0
elif [[ "${rc}" != "0" ]]; then
  echo "[NARVAL-COMBINED] Training state check failed rc=${rc}"
  exit "${rc}"
fi

echo
echo "============================================================"
echo "[NARVAL-COMBINED] PHASE 2: BLER-only compiled eval by Eb/N0"
echo "============================================================"

unset UPAIR_COMPREHENSIVE_SKIP_FINAL_EVAL
export UPAIR_EVAL_IGNORE_SHARED_CSV=1
export UPAIR_COMPREHENSIVE_EVAL_ONLY=1
export UPAIR_COMPREHENSIVE_FORCE=0

# Evaluation uses cuda_malloc_async because each Eb/N0 is run in a fresh process.
export TF_GPU_ALLOCATOR="${UPAIR_EVAL_TF_GPU_ALLOCATOR:-cuda_malloc_async}"
echo "[NARVAL-COMBINED] EVAL_TF_GPU_ALLOCATOR=${TF_GPU_ALLOCATOR}"

mapfile -t EBNO_LIST < <(
  python - "$BASE_CONFIG" <<'PY_EBNO_LIST'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
vals = cfg.get("system", {}).get("ebno_db_eval", [-4,-3,-2,-1,0,1,2,3,4])
for v in vals:
    print(float(v))
PY_EBNO_LIST
)

for EBNO in "${EBNO_LIST[@]}"; do
  echo
  echo "============================================================"
  echo "[NARVAL-COMBINED] evaluating Eb/N0=${EBNO} dB in a fresh Python process"
  echo "============================================================"

  TMP_CFG="tmp_eval_configs/${VARIANT}_u${EVAL_USERS}_stageB_locked_bleronly_ebno_${EBNO}_job${SLURM_JOB_ID:-manual}.yaml"

  python - "$BASE_CONFIG" "$TMP_CFG" "$EBNO" <<'PY_TMP_CFG'
import sys, yaml

src, dst, ebno = sys.argv[1], sys.argv[2], float(sys.argv[3])
cfg = yaml.safe_load(open(src, "r", encoding="utf-8"))

cfg.setdefault("system", {})
cfg["system"]["ebno_db_eval"] = [ebno]
cfg["system"]["batch_size_eval"] = 64

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

# Perfect CSI behavior intentionally unchanged.

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

  echo "[NARVAL-COMBINED] finished Eb/N0=${EBNO} dB; Python process exited, GPU memory released."
done

echo
echo "[NARVAL-COMBINED] all training/eval phases finished for ${VARIANT}"
