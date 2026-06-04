# Extended UPAIR Narval b32m16 Portable Package

This is a cleaned portable package for the completed UPAIR experiment.

## Included

- Full code pipeline: `src/`, `scripts/`, `configs/`, `slurm/`
- Stage-B locked Optuna best hyperparameter JSON files in `optuna/`
- Best trained weights for seven UPAIR variants:
  `TWC_plots_comprehensive/runs_rx16/seed7/1dmrs/<variant>/checkpoints/best.weights.h5`
- Minimal training metadata under each variant `metrics/` folder
- Evaluation results and generated figures for U2 and U3 cases under:
  `TWC_plots_comprehensive/comprehensive_plots/`

## Removed

- Optuna trial run folders and intermediate trial checkpoints
- Optuna SQLite DBs
- Slurm logs, debug logs, probe folders, temporary eval configs
- Optimizer training-state checkpoints
- Backup files and Python cache folders

## Quick check

```bash
bash portable_tools/manifest_summary.sh
```
