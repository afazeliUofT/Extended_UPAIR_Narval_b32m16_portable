# Clean staged Optuna package for 1-DMRS UPAIR ablations

This package modifies the original Narval repository for clean-case, batch-64, staged Optuna tuning of the seven 1-DMRS ablation variants.

Key properties:

- Clean case only for Optuna: `impairments.enabled=false`, `legacy_phase_impairments.enabled=false`, `rf_impairments.enabled=false`, and RF train mixture disabled.
- Fixed Optuna training batch size: 64.
- Fixed Optuna validation logical batch size: 64, with validation microbatch size 32.
- Mixed-user training: active users 1--4 are sampled with weights 1:3:6:10.
- Mixed-user Optuna objective: validation samples active users 1--4 with weights 1:2:3:4 and samples the SNR grid `-4,-2,0,2,4`.
- Staged Optuna defaults:
  - Stage A: 30 trials, 6,000 steps
  - Stage B: 8 promoted trials, 10,000 steps
  - Stage C: 3 promoted trials, 20,000 steps
- Final objective excludes validation rows with `step <= 1000`.
- Percentile pruner warmup default is 1,000 steps.
- Learning-rate search range is `5e-5` to `1.2e-3`.
- Resume-safe Slurm behavior: 12-hour walltime, `TERM@600`, resumable training checkpoints, SQLite-backed Optuna studies, and RUNNING-trial recovery.
- Evaluation streaming preserved: final logical evaluation batch can be 96, but receiver evaluation uses microbatches of 32 with scalar error-count extraction and cleanup after each microbatch.
- One fresh TensorFlow worker process is used per Optuna trial to prevent cross-trial CUDA allocator accumulation.

Recommended install target:

```bash
/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16
```

Recommended virtual environment:

```bash
/home/rsadve1/scratch/.venvUPAIR
```
