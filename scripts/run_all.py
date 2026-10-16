"""Hydra-driven experiment entry point (§8: "hydra-core -- do not
hand-roll argparse").

Examples:
    python scripts/run_all.py dataset=adult
    python scripts/run_all.py dataset=german_credit signals=tier_a
    python scripts/run_all.py dataset=adult experiment.n_seeds=10

Appends results to `results/results.parquet` (creating it if absent) --
the single long-format file every table/figure in the paper queries
(§8's non-negotiable engineering rule).
"""
from __future__ import annotations

import os

# --- macOS OpenMP deadlock guard -------------------------------------------
# LightGBM and PyTorch each bundle their own copy of libomp on macOS wheels.
# Importing both in one process (this script does, via src/aggregators/
# a1_stacking.py and a2_coverage_loss.py) makes two OpenMP runtimes try to
# initialize in the same process. On macOS this does not raise -- it hangs
# forever with 0% CPU, every worker thread parked in `__kmp_fork_barrier`,
# and no error message, which is the worst possible failure mode for a
# project whose whole design point is "reproducible from the CLI" (§8).
# Verified by reproducing the hang twice (this script and `pytest`) and
# confirming both resolve immediately once these are set. `n_jobs=-1` in
# `src/signals/tier_c.py`'s NearestNeighbors calls likely compounds it by
# adding a third parallel runtime into the same process, though the two
# reproductions above both import torch + lightgbm together regardless of
# whether Tier C's kNN path runs, so that pairing is the primary suspect.
# Must be set before numpy/sklearn/lightgbm/torch are imported anywhere in
# the process, so this sits above every other import in the entry point.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import sys
import warnings
from pathlib import Path

# Suppress harmless sklearn/LightGBM feature name validation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# --- Python 3.14 / hydra-core 1.3.6 compatibility shim ---------------------
# Python 3.14's argparse added an eager `_check_help` validation at
# `add_argument` time that assumes `help=` is always a real string.
# hydra-core 1.3.6 (the latest release as of writing) registers its
# `--shell-completion` flag with a lazily-stringified `LazyCompletionHelp`
# object, which trips that new check and crashes before any of this
# project's own code runs -- entirely unrelated to anything in this repo.
# There is no newer hydra-core release that fixes this yet (checked via
# `pip index versions hydra-core`). Disabling the eager check is safe: it
# only restores the pre-3.14 behavior of resolving `help=` lazily when
# `--help` is actually invoked, which nothing here relies on anyway.
argparse.ArgumentParser._check_help = lambda self, action: None  # type: ignore[method-assign]

import hydra
import pandas as pd
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment.runner import run_experiment  # noqa: E402

RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "results.parquet"
RAW_DIR = Path(__file__).resolve().parents[1] / "results" / "raw"


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    all_dfs = []
    for seed in range(cfg.experiment.n_seeds):
        print(f"[run_all] dataset={cfg.dataset.name} model={cfg.model.name} "
              f"tiers={cfg.signals.tiers} seed={seed}", flush=True)
        df = run_experiment(
            dataset_name=cfg.dataset.name,
            model_name=cfg.model.name,
            tiers=tuple(cfg.signals.tiers),
            seed=seed,
            n_cv_folds=cfg.experiment.n_cv_folds,
            use_ensemble=cfg.signals.get("use_ensemble", False),
            n_ensemble_members=cfg.signals.get("n_ensemble_members", 5),
            conformal_delta=cfg.experiment.conformal_delta,
            subgroup_col=cfg.dataset.get("subgroup_col"),
            # Per-instance test scores for §7's paired bootstrap. On by
            # default: the files are small (compressed float32, one shared
            # `incorrect` vector per run) and gitignored, and without them
            # the bootstrap test can only be approximated by the coarser
            # per-seed Wilcoxon.
            raw_out_dir=(
                str(RAW_DIR) if cfg.experiment.get("cache_raw_scores", True) else None
            ),
        )
        all_dfs.append(df)

    new_results = pd.concat(all_dfs, ignore_index=True)

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    if RESULTS_PATH.exists():
        existing = pd.read_parquet(RESULTS_PATH)
        # Replace any prior rows for this exact
        # (dataset, model, tiers, seed) combo so re-running a config is
        # idempotent instead of duplicating rows. `tiers` is part of the key
        # because an (A,B,C) run and an (A,C) run are different experiments
        # that must coexist in the file, not overwrite each other.
        key_cols = ["dataset", "base_model", "tiers", "seed"]
        for col in key_cols:
            if col not in existing.columns:
                raise SystemExit(
                    f"results.parquet predates the '{col}' results column and "
                    f"cannot be merged safely. Move it aside (or delete it) "
                    f"and re-run; every number is reproducible from the CLI."
                )
        mask = existing.set_index(key_cols).index.isin(
            new_results.set_index(key_cols).index
        )
        existing = existing[~mask]
        combined = pd.concat([existing, new_results], ignore_index=True)
    else:
        combined = new_results

    combined.to_parquet(RESULTS_PATH)
    print(f"[run_all] wrote {len(combined)} rows -> {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
