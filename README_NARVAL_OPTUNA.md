# Narval clean batch-64 isolated Optuna package

This package is adapted from the Nibi clean B64 isolated Optuna package for Narval.

Default paths:

```text
PROJECT_ROOT=/home/rsadve1/scratch/Extended_UPAIR_Narval_b32m16
VENV_PATH=/home/rsadve1/scratch/.venvUPAIR
```

Default GPU request in submit scripts:

```text
--account=def-rsadve_gpu
--gres=gpu:1
```

Override if needed:

```bash
export NARVAL_SLURM_ACCOUNT=def-rsadve_gpu
export NARVAL_GRES=gpu:1              # or gpu:a100:1 if Narval requires typed GRES
export NARVAL_SLURM_PARTITION=<name>  # optional, normally leave unset
```

The Optuna execution strategy remains unchanged:

```text
one fresh TensorFlow worker process per Optuna trial
TF_GPU_ALLOCATOR unset/default
ResourceExhausted trials marked FAIL, not PRUNED
Stage-A: 30 trials, 6000 steps
train batch 32, validation logical batch 32, validation microbatch 16
training user weights 1:3:6:10
validation user weights 1:2:3:4
```
