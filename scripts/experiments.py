"""Experiment registry: each entry is a named set of hyperparameters.

Add an experiment = add a dict entry. Keys:
  iterations   : int, training iterations
  sh_degree    : int (default 3)
  prepare      : dict passed to prepare.prepare_source's `augment` (e.g. {"random_points": N})
  train_args   : list[str] extra flags forwarded to third_party gaussian-splatting train.py
                 (repo stays untouched — we only pass flags)

experiment.py runs these in `holdout` mode (measure on 1/8 withheld views) to compare,
then `submission` mode with the winning config to produce the final zip.
"""

EXPERIMENTS = {
    # Vanilla 3DGS at 30k — the reference we already submitted.
    "baseline": {
        "iterations": 30000,
        "sh_degree": 3,
        "prepare": {},
        "train_args": [],
    },

    # Per-image exposure affine — compensates brightness differences between drone
    # shots during training. Verified in this repo: exposure params are always
    # optimized, but only APPLIED when --train_test_exp is set (default off = baseline
    # has no exposure compensation). Novel test poses still render with identity
    # exposure (our render_test_poses uses use_trained_exp=False), so the benefit is a
    # cleaner base model, not test-time correction.
    "exposure": {
        "iterations": 30000,
        "sh_degree": 3,
        "prepare": {},
        "train_args": ["--train_test_exp"],
    },

    # Denser initialization — inject random points to help sparse regions
    # (thin antennas / low-texture areas COLMAP under-covers).
    "densify": {
        "iterations": 30000,
        "sh_degree": 3,
        "prepare": {"random_points": 100000},
        "train_args": [],
    },
}


def get(name):
    if name not in EXPERIMENTS:
        raise KeyError(f"Unknown experiment '{name}'. Available: {list(EXPERIMENTS)}")
    cfg = EXPERIMENTS[name]
    return {
        "iterations": cfg.get("iterations", 30000),
        "sh_degree": cfg.get("sh_degree", 3),
        "prepare": cfg.get("prepare", {}) or {},
        "train_args": list(cfg.get("train_args", [])),
    }
