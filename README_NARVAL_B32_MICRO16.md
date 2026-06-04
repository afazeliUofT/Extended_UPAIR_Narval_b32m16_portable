# Narval clean Optuna package: batch 32, microbatch 16

This package is adapted for Narval jobs that may land on 40 GB GPUs.

Key defaults:

- Optuna/train batch size: 32
- Optuna validation logical batch: 32
- Optuna validation microbatch: 16
- Final training batch size: 32
- Final evaluation logical batch: 64
- Final receiver evaluation microbatch: 16
- Clean case: phase/RF impairments disabled
- DD-CPE baseline removed from clean configs
- Optuna trial process isolation enabled
- TensorFlow default GPU allocator; cuda_malloc_async is not inherited
- ResourceExhausted trials are marked FAIL, not PRUNED
- Study prefix: clean_b32_iso_u34610_1dmrs_stageA/B/C

Use `submit_stageA_all7_narval_b32m16.sh` to submit seven independent one-GPU Stage-A jobs.
