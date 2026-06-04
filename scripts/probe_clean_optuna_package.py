from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from upair5g.config import get_cfg, load_config  # noqa: E402
from optuna_1dmrs_common import STAGE_DEFAULTS, make_trial_config, validation_history_score  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _check_clean_config(path: Path, expected_eval_batch: int) -> None:
    cfg = load_config(path)
    _assert(int(get_cfg(cfg, "system.batch_size_train")) == 32, f"{path}: system.batch_size_train must be 32")
    _assert(int(get_cfg(cfg, "system.batch_size_eval")) == expected_eval_batch, f"{path}: unexpected system.batch_size_eval")
    _assert(get_cfg(cfg, "impairments.enabled") is False, f"{path}: impairments.enabled must be false")
    _assert(get_cfg(cfg, "legacy_phase_impairments.enabled") is False, f"{path}: legacy phase impairments must be false")
    _assert(get_cfg(cfg, "rf_impairments.enabled") is False, f"{path}: rf_impairments.enabled must be false")
    _assert(get_cfg(cfg, "rf_impairments.train_mixture.enabled") is False, f"{path}: RF train mixture must be false")
    _assert(str(get_cfg(cfg, "multiuser.train_user_count_sampler")).lower() == "weighted", f"{path}: train sampler must be weighted")
    _assert([float(x) for x in get_cfg(cfg, "multiuser.train_user_count_weights")] == [1.0, 3.0, 6.0, 10.0], f"{path}: train weights mismatch")
    _assert(str(get_cfg(cfg, "training.val_sampling_mode")).lower() == "sampled", f"{path}: Optuna validation mode should be sampled")
    _assert(str(get_cfg(cfg, "training.val_snr_sampling_mode")).lower() == "sampled_grid", f"{path}: Optuna validation SNR mode should sample the configured grid")
    _assert([int(x) for x in get_cfg(cfg, "training.val_user_counts")] == [1, 2, 3, 4], f"{path}: val user counts mismatch")
    _assert([float(x) for x in get_cfg(cfg, "training.val_user_count_weights")] == [1.0, 2.0, 3.0, 4.0], f"{path}: val user weights mismatch")
    _assert(int(get_cfg(cfg, "training.val_microbatch_size")) == 16, f"{path}: validation microbatch must be 16")
    _assert(bool(get_cfg(cfg, "training.val_memory_cleanup_every_microbatch")) is True, f"{path}: validation microbatch cleanup must be enabled")
    _assert(bool(get_cfg(cfg, "training.val_memory_cleanup_every_batch")) is True, f"{path}: val memory cleanup must be enabled")
    _assert(int(get_cfg(cfg, "training.memory_cleanup_every_steps")) == 100, f"{path}: periodic training memory cleanup should be 100 steps")
    _assert(int(get_cfg(cfg, "evaluation.receiver_microbatch_size")) == 16, f"{path}: receiver microbatch must be 16")
    _assert(bool(get_cfg(cfg, "evaluation.stream_eval_microbatches")) is True, f"{path}: stream eval microbatches must be true")
    _assert(bool(get_cfg(cfg, "evaluation.compiled_receiver_error_counts")) is True, f"{path}: compiled scalar error-count path must be true")
    _assert(bool(get_cfg(cfg, "evaluation.memory_cleanup_every_microbatch")) is True, f"{path}: cleanup every microbatch must be true")
    _assert(int(get_cfg(cfg, "evaluation.memory_cleanup_every_batches")) == 1, f"{path}: cleanup every batch must be 1")
    _assert(int(get_cfg(cfg, "evaluation.logical_batch_size", 64)) == 64, f"{path}: final eval logical batch should be 64")
    for section, key in [("baselines", "enabled_receivers"), ("evaluation", "stopping_receivers"), ("evaluation", "nmse_receivers")]:
        vals = get_cfg(cfg, f"{section}.{key}", [])
        _assert("baseline_ddcpe_ls_lmmse" not in vals, f"{path}: DD-CPE baseline must be disabled/removed from {section}.{key}")


def _check_resolved_optuna_config() -> None:
    args = SimpleNamespace(
        config=str(PROJECT_ROOT / "configs" / "twc_comprehensive_mu32_base.yaml"),
        variant="main_d96_b4_r2",
        study_name="probe_resolved_b32",
        train_batch_size=32,
        validation_batch_size=32,
        validation_microbatch_size=16,
        train_user_count_weights=[1.0, 3.0, 6.0, 10.0],
        train_ebno_min=-6.0,
        train_ebno_max=5.0,
        val_user_counts=[1, 2, 3, 4],
        val_user_count_weights=[1.0, 2.0, 3.0, 4.0],
        val_memory_cleanup_every_batch=True,
        memory_cleanup_every_steps=100,
        seed=7,
        steps=6000,
        eval_every=1000,
        checkpoint_every=1000,
        log_every=100,
        val_steps=96,
        val_ebno_db=[-4.0, -2.0, 0.0, 2.0, 4.0],
    )
    params = {
        "learning_rate_schedule": "cosine_decay",
        "learning_rate": 3e-4,
        "learning_rate_decay_fraction": 1.0,
        "learning_rate_final_fraction": 0.05,
        "learning_rate_polynomial_power": 1.0,
        "weight_decay": 1e-5,
        "nmse_loss_weight": 0.1,
        "grad_clip_norm": 1.0,
        "dropout": 0.05,
        "residual_scale": 0.35,
    }
    cfg = make_trial_config(args, params, trial_number=0)
    _assert(int(get_cfg(cfg, "system.batch_size_train")) == 32, "resolved Optuna config train batch must be 32")
    _assert(int(get_cfg(cfg, "system.batch_size_eval")) == 32, "resolved Optuna validation logical batch must be 32")
    _assert(int(get_cfg(cfg, "training.val_microbatch_size")) == 16, "resolved Optuna validation microbatch must be 16")
    _assert([float(x) for x in get_cfg(cfg, "multiuser.train_user_count_weights")] == [1.0, 3.0, 6.0, 10.0], "resolved train weights mismatch")
    _assert([float(x) for x in get_cfg(cfg, "training.val_user_count_weights")] == [1.0, 2.0, 3.0, 4.0], "resolved val weights mismatch")


def _check_objective_min_step() -> None:
    import json
    import tempfile

    payload = {
        "history": [
            {"step": 1000, "val_nmse_prop": 1.0, "val_nmse_ls": 1.0},
            {"step": 2000, "val_nmse_prop": 0.1, "val_nmse_ls": 1.0},
        ]
    }
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "history.json"
        p.write_text(json.dumps(payload), encoding="utf-8")
        value = validation_history_score(p, objective_metric="prop_nmse", aggregation="last", recent_k=2, min_step=1000)
        _assert(abs(value - (-1.0)) < 1e-9, "objective_min_step=1000 must exclude step-1000 and keep step-2000")


def main() -> None:
    _check_clean_config(PROJECT_ROOT / "configs" / "twc_comprehensive_mu32_base.yaml", expected_eval_batch=32)
    _check_clean_config(PROJECT_ROOT / "configs" / "twc_comprehensive_mu32_eval_safe.yaml", expected_eval_batch=64)
    _check_resolved_optuna_config()
    _check_objective_min_step()

    common_text = (PROJECT_ROOT / "scripts" / "optuna_1dmrs_common.py").read_text(encoding="utf-8")
    isolated_text = (PROJECT_ROOT / "scripts" / "run_optuna_1dmrs_structure_isolated.py").read_text(encoding="utf-8")
    worker_text = (PROJECT_ROOT / "scripts" / "run_optuna_1dmrs_trial_worker.py").read_text(encoding="utf-8")
    merged_optuna_text = common_text + isolated_text + worker_text
    _assert("[32, 48, 64]" not in merged_optuna_text and "[64, 96, 128]" not in merged_optuna_text, "Old unsafe batch-size candidates remain")
    _assert('"constant", "cosine_decay", "polynomial_decay"' in merged_optuna_text, "Expected clean conditional schedule search is missing")
    _assert('trial.suggest_float("learning_rate", 5e-5, 1.2e-3, log=True)' in common_text, "LR lower tail was not widened to 5e-5")
    _assert('set_cfg(cfg, "training.val_sampling_mode", "sampled")' in common_text, "Optuna must sample active-user counts during validation")
    _assert('trial_ref.report' in worker_text and 'trial_ref.should_prune' in worker_text, "Optuna pruning is not connected to validation reports in the worker")
    _assert(STAGE_DEFAULTS["A"]["steps"] == 6000 and STAGE_DEFAULTS["A"]["target_total_trials"] == 30, "Stage-A defaults are wrong")
    _assert('subprocess.Popen' in isolated_text and 'run_optuna_1dmrs_trial_worker.py' in isolated_text, "Isolated one-trial worker mode is missing")
    _assert('resource_exhausted' in isolated_text and 'TrialState.FAIL' in isolated_text, "ResourceExhausted must be FAIL, not PRUNED")

    comprehensive_text = (PROJECT_ROOT / "scripts" / "run_comprehensive_mu32_ablation.py").read_text(encoding="utf-8")
    _assert('"system.batch_size_train": 32' in comprehensive_text, "Built-in Optuna fallback should now use train batch 32")
    _assert('"system.batch_size_eval": 32' in comprehensive_text, "Built-in Optuna fallback should now use validation batch 32")
    _assert("clean_b32_iso_u34610_1dmrs_stageC" in comprehensive_text, "Comprehensive script does not default to the new staged clean Optuna prefix")

    slurm_text = (PROJECT_ROOT / "slurm" / "narval" / "optuna_1dmrs_structures.sbatch").read_text(encoding="utf-8")
    _assert("#SBATCH --time=12:00:00" in slurm_text, "Optuna Slurm walltime must be 12 hours")
    _assert("#SBATCH --array=0-6%7" in slurm_text, "Optuna Slurm array should request 7 one-GPU tasks")
    _assert("--signal=B:TERM@600" in slurm_text, "Optuna Slurm script should send SIGTERM before walltime")
    _assert("run_optuna_1dmrs_structure_isolated.py" in slurm_text, "Slurm script must default to isolated Optuna driver")
    _assert("unset TF_GPU_ALLOCATOR" in slurm_text, "Slurm script must not inherit cuda_malloc_async by default")
    _assert('--train-batch-size "${OPTUNA_TRAIN_BATCH_SIZE:-32}"' in slurm_text, "Slurm must forward Optuna train batch 32")
    _assert('--validation-batch-size "${OPTUNA_VALIDATION_BATCH_SIZE:-32}"' in slurm_text, "Slurm must forward Optuna validation batch 32")
    _assert('--validation-microbatch-size "${OPTUNA_VALIDATION_MICROBATCH_SIZE:-16}"' in slurm_text, "Slurm must forward validation microbatch 16")
    _assert('--train-user-count-weights "${OPTUNA_TRAIN_USER_COUNT_WEIGHTS:-1,3,6,10}"' in slurm_text, "Slurm must forward train user weights")
    _assert('--val-user-count-weights "${OPTUNA_VAL_USER_COUNT_WEIGHTS:-1,2,3,4}"' in slurm_text, "Slurm must forward validation user weights")
    _assert('--objective-min-step "${OPTUNA_OBJECTIVE_MIN_STEP:-1000}"' in slurm_text, "Slurm must forward objective min step")

    print("[PROBE] clean Optuna package checks passed")
    print("[PROBE] stage defaults: A=30 trials/6000 steps, B=8 trials/10000 steps, C=3 trials/20000 steps")
    print("[PROBE] Optuna train batch=32; validation logical batch=32; validation microbatch=16")
    print("[PROBE] training users sampled 1-4 with weights 1:3:6:10")
    print("[PROBE] Optuna validation users sampled 1-4 with weights 1:2:3:4 and SNR grid -4,-2,0,2,4")
    print("[PROBE] final objective excludes validation rows with step <= 1000")
    print("[PROBE] final evaluation remains streamed: logical batch 64, receiver microbatch 16")
    print("[PROBE] Optuna uses one TensorFlow worker process per trial and does not inherit cuda_malloc_async")
    print("[PROBE] ResourceExhausted trials are marked FAIL and do not count as completed/pruned tuning evidence")


if __name__ == "__main__":
    main()
