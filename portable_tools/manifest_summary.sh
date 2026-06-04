#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
echo "============================================================"
echo "Portable UPAIR package summary"
echo "============================================================"
echo "ROOT=${ROOT}"
echo
echo "Total size:"
du -sh "${ROOT}"
echo
echo "Top-level directories:"
du -sh "${ROOT}"/* 2>/dev/null | sort -hr
echo
echo "Best weights:"
find "${ROOT}/TWC_plots_comprehensive/runs_rx16/seed7/1dmrs" -path "*/checkpoints/best.weights.h5" -printf "%s\t%p\n" | awk 'BEGIN{printf "%12s  %s\n","SIZE_MB","PATH"} {printf "%12.2f  %s\n",$1/1024/1024,$2}'
echo
echo "Stage-B locked Optuna best JSON files:"
find "${ROOT}/optuna" -maxdepth 1 -type f -name "*stageB_locked*best_params.json" -printf "%f\n" | sort
echo
echo "Plots/results:"
find "${ROOT}/TWC_plots_comprehensive/comprehensive_plots" -maxdepth 3 -type f \( -name "*.pdf" -o -name "*.png" -o -name "*.csv" -o -name "*.txt" \) -printf "%P\n" | sort
