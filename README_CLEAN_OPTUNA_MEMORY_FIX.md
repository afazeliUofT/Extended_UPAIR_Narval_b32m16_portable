# Clean Optuna restart package: batch-64 + isolated one-trial workers

This package keeps the clean 1-DMRS staged Optuna design, but uses one fresh TensorFlow worker process per Optuna trial to avoid cross-trial CUDA allocator/mempool poisoning.

## What changed

- `scripts/run_optuna_1dmrs_structure_isolated.py` is the default Slurm Optuna driver.
- Each Optuna trial launches `scripts/run_optuna_1dmrs_trial_worker.py` in a fresh Python process.
- The parent Optuna controller does not import TensorFlow.
- Optuna/training batch defaults are now 64 for training and 64 for validation logical batch.
- Optuna validation still uses memory-safe validation microbatches of 32.
- Final receiver evaluation streaming is unchanged: logical evaluation batch defaults to 96, receiver microbatch is 32, scalar error counts are copied back, tensors are deleted, and cleanup runs after each microbatch.
- Training user-count weights default to `[1, 3, 6, 10]` for users 1--4.
- Optuna validation user-count weights default to `[1, 2, 3, 4]` and are explicitly forwarded by Slurm.
- Stage A defaults to 30 trials and 6000 steps.
- Stage B defaults to 8 promoted trials and 10000 steps.
- Stage C defaults to 3 promoted trials and 20000 steps.
- The final Optuna objective excludes validation rows with `step <= 1000`, so the step-1000 validation is ignored in the final recent-mean score.
- The Stage-A percentile pruner warmup default is 1000 steps.
- The learning-rate search lower bound is widened to `5e-5`.
- The Slurm scripts do not inherit `TF_GPU_ALLOCATOR=cuda_malloc_async` by default. Set `OPTUNA_TF_GPU_ALLOCATOR=cuda_malloc_async` or `UPAIR_TF_GPU_ALLOCATOR=cuda_malloc_async` only if you explicitly want to test it.
- `tf.errors.ResourceExhaustedError` is marked as Optuna `FAIL`, not `PRUNED`, so allocator or true OOM failures do not count as valid Stage-A/B/C tuning evidence.
- Interrupted Slurm jobs leave the current trial in `RUNNING` state and resume it from `train_state.json` and TensorFlow checkpoints.

## Recommended restart prefix

Use the new default prefix and do not continue the earlier damaged DBs:

```bash
clean_b32_iso_u34610_1dmrs_stageA
clean_b32_iso_u34610_1dmrs_stageB
clean_b32_iso_u34610_1dmrs_stageC
```
