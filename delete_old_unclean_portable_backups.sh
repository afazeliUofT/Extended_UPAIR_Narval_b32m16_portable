#!/usr/bin/env bash
set -euo pipefail

CURRENT="/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16_portable"
PARENT="/home/rsadve1/scratch"
DELETE="${DELETE:-0}"

echo "============================================================"
echo "SAFE DELETE OF OLD UNCLEAN PORTABLE BACKUPS"
echo "============================================================"
echo "CURRENT=${CURRENT}"
echo "DELETE=${DELETE}"
echo

if [[ ! -d "${CURRENT}" ]]; then
  echo "ERROR: current clean portable folder does not exist:"
  echo "  ${CURRENT}"
  exit 2
fi

echo "============================================================"
echo "1) Validate current clean portable folder"
echo "============================================================"

weight_count="$(
  find "${CURRENT}/TWC_plots_comprehensive/runs_rx16/seed7/1dmrs" \
    -path '*/checkpoints/best.weights.h5' 2>/dev/null | wc -l | tr -d ' '
)"

json_count="$(
  find "${CURRENT}/optuna" -maxdepth 1 -type f \
    -name '*stageB_locked*best_params.json' 2>/dev/null | wc -l | tr -d ' '
)"

u3_csv="${CURRENT}/TWC_plots_comprehensive/comprehensive_plots/u3_evaluation_points.csv"
u2_csv="${CURRENT}/TWC_plots_comprehensive/comprehensive_plots/U2_comprehensive plots/u2_evaluation_points.csv"

echo "best.weights.h5 count: ${weight_count}"
echo "Stage-B locked JSON count: ${json_count}"
echo "U3 CSV exists: $([[ -f "${u3_csv}" ]] && echo yes || echo no)"
echo "U2 CSV exists: $([[ -f "${u2_csv}" ]] && echo yes || echo no)"

if [[ "${weight_count}" != "7" ]]; then
  echo "ERROR: expected 7 best.weights.h5 files."
  exit 3
fi

if [[ "${json_count}" != "7" ]]; then
  echo "ERROR: expected 7 Stage-B locked best JSON files."
  exit 3
fi

if [[ ! -f "${u3_csv}" ]]; then
  echo "ERROR: missing U3 evaluation CSV:"
  echo "  ${u3_csv}"
  exit 3
fi

if [[ ! -f "${u2_csv}" ]]; then
  echo "ERROR: missing U2 evaluation CSV:"
  echo "  ${u2_csv}"
  exit 3
fi

echo "[OK] Current clean portable folder passed validation."

echo
echo "============================================================"
echo "2) Find old unclean backup folders"
echo "============================================================"

mapfile -t backups < <(
  find "${PARENT}" -maxdepth 1 -type d \
    -name 'Extended_UPAIR_Narval_b32m16_portable.unclean_*' \
    | sort
)

if [[ "${#backups[@]}" -eq 0 ]]; then
  echo "No old unclean backup folders found."
  exit 0
fi

echo "Found ${#backups[@]} old unclean backup folder(s):"
for b in "${backups[@]}"; do
  du -sh "$b" 2>/dev/null || true
done

echo
echo "============================================================"
echo "3) Delete behavior"
echo "============================================================"

if [[ "${DELETE}" != "1" ]]; then
  echo "DRY RUN ONLY. Nothing was deleted."
  echo
  echo "To delete the old backup folder(s), run:"
  echo "  DELETE=1 bash delete_old_unclean_portable_backups.sh"
  exit 0
fi

echo "DELETE=1, removing old unclean backup folder(s)..."

for b in "${backups[@]}"; do
  echo "Deleting: ${b}"
  rm -rf -- "${b}"
done

echo
echo "[OK] Old unclean backup folder(s) deleted."

echo
echo "Remaining unclean backups:"
find "${PARENT}" -maxdepth 1 -type d \
  -name 'Extended_UPAIR_Narval_b32m16_portable.unclean_*' \
  -print || true

echo
echo "Current clean portable folder size:"
du -sh "${CURRENT}"
