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

    # Per-image exposure affine — helps when drone shots vary in brightness.
    # NOTE: exact flag names/defaults MUST be verified against the cloned repo's
    # arguments/__init__.py (grep exposure_lr). If baseline already trains exposure
    # by default, flip this experiment to disable it (--exposure_lr_init 0) instead.
    "exposure": {
        "iterations": 30000,
        "sh_degree": 3,
        "prepare": {},
        "train_args": ["--exposure_lr_init", "0.001", "--exposure_lr_final", "0.0001"],
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
