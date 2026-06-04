#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16}"
VENV_PATH="${VENV_PATH:-/home/rsadve1/scratch/.venvUPAIR}"
STDENV_MODULE="${NARVAL_STDENV_MODULE:-${STDENV_MODULE:-StdEnv/2023}}"
GCC_MODULE="${NARVAL_GCC_MODULE:-${GCC_MODULE:-gcc/12.3}}"
PYTHON_MODULE="${NARVAL_PYTHON_MODULE:-${PYTHON_MODULE:-python/3.11.5}}"
CUDA_MODULE="${NARVAL_CUDA_MODULE:-${CUDA_MODULE:-cuda/12.2}}"
PIP_INDEX_FLAGS="${UPAIR_PIP_INDEX_FLAGS:---no-index}"
SIONNA_NO_RT_PACKAGE="${SIONNA_NO_RT_PACKAGE:-sionna-no-rt==1.2.1}"

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

module --force purge || true
module load "${STDENV_MODULE}"
module load "${GCC_MODULE}"
module load "${PYTHON_MODULE}"
if [[ -n "${CUDA_MODULE}" ]]; then
  module load "${CUDA_MODULE}"
fi

export PYTHONNOUSERSITE=1
export PIP_NO_USER=1
export PIP_USER=0
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
unset PYTHONUSERBASE
unset PYTHONPATH

echo "[INFO] Project root : ${PROJECT_ROOT}"
echo "[INFO] Venv path    : ${VENV_PATH}"
echo "[INFO] StdEnv module: ${STDENV_MODULE}"
echo "[INFO] GCC module   : ${GCC_MODULE}"
echo "[INFO] Python module: ${PYTHON_MODULE}"
echo "[INFO] CUDA module  : ${CUDA_MODULE}"
echo "[INFO] Pip flags    : ${PIP_INDEX_FLAGS:-<default-index>}"
echo "[INFO] Sionna pkg   : ${SIONNA_NO_RT_PACKAGE}"

if [[ ! -d "${PROJECT_ROOT}" ]]; then
  echo "[ERROR] PROJECT_ROOT does not exist: ${PROJECT_ROOT}" >&2
  exit 1
fi
cd "${PROJECT_ROOT}"

python --version

if [[ -d "${VENV_PATH}" && "${UPAIR_RECREATE_VENV:-0}" == "1" ]]; then
  echo "[INFO] Removing existing venv because UPAIR_RECREATE_VENV=1"
  rm -rf "${VENV_PATH:?}"
fi

if [[ ! -d "${VENV_PATH}" ]]; then
  if command -v virtualenv >/dev/null 2>&1; then
    virtualenv --no-download "${VENV_PATH}"
  else
    python -m venv "${VENV_PATH}"
  fi
fi

# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"

if ! python -m pip --isolated --version >/dev/null 2>&1; then
  python -m ensurepip --upgrade || true
fi
PIP_USER=0 python -m pip --isolated install ${PIP_INDEX_FLAGS} --upgrade pip setuptools wheel packaging
wheelhouse_requirements="$(mktemp)"
grep -Eiv '^[[:space:]]*sionna-no-rt([=<>!~ ]|$)' requirements-narval.txt > "${wheelhouse_requirements}"
PIP_USER=0 python -m pip --isolated install ${PIP_INDEX_FLAGS} -r "${wheelhouse_requirements}"
rm -f "${wheelhouse_requirements}"
if ! PIP_USER=0 python -m pip --isolated install --no-deps "${SIONNA_NO_RT_PACKAGE}"; then
  echo "[ERROR] Could not install ${SIONNA_NO_RT_PACKAGE} from PyPI." >&2
  echo "[ERROR] If Narval login-node internet is blocked, download the wheel on another machine and install it manually:" >&2
  echo "[ERROR]   python -m pip --isolated install --no-deps /path/to/sionna_no_rt-1.2.1-py3-none-any.whl" >&2
  exit 3
fi
PIP_USER=0 python -m pip --isolated install -e . --no-deps

mkdir -p logs

python - <<'PY'
import sys
print("[INFO] Python executable:", sys.executable)
print("[INFO] Python version   :", sys.version)

try:
    import tensorflow as tf
    print("[INFO] TensorFlow      :", tf.__version__)
    print("[INFO] GPUs            :", tf.config.list_physical_devices("GPU"))
except Exception as e:
    print("[WARN] TensorFlow import issue:", repr(e))

try:
    import sionna
    print("[INFO] Sionna          :", sionna.__version__)
    import sionna.phy.nr
    print("[INFO] Sionna PHY NR   : available")
except Exception as e:
    print("[WARN] Sionna import issue:", repr(e))
PY

echo "[INFO] Setup complete."
echo "[INFO] Activate later with:"
echo "       source ${VENV_PATH}/bin/activate"
