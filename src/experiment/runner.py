"""End-to-end experiment runner.

Wires together: dataset load -> four-way split -> honest (cross-fitted)
meta signals -> final deployment model -> aggregators -> conformal wrapper
-> metrics -> one long-format DataFrame row per (method, coverage) per
§8's results-schema rule ("no number in the paper should exist anywhere
except as a query against [the results parquet]").

Design choice, stated explicitly (this is the kind of simplification the
project plan asks you to be honest about, not hide, §6/§7):

  - `model_train`  : fit on D_train only. Used to fit every signal that
                     needs data the base model has not seen -- temperature
                     scaling and the isotonic calibration residual, both fit
                     on D_meta -- and to train the Tier B ensemble. All are
                     honest by construction, since model_train never saw
                     D_meta, so no cross-fitting is needed for them.
  - fold models    : K-fold cross-fit over D_train u D_meta (§6's fix).
                     Used to compute honest out-of-fold Tier A (non-temperature)
                     and Tier C signals, plus correctness labels, for
                     aggregator (meta) training data.
  - `model_final`  : fit on D_train u D_meta. This is the deployed model:
                     it produces every prediction and signal used at
                     evaluation time on D_cal and D_test.

Tier B (ensemble disagreement) is optional (`use_ensemble=True`). It is not
cross-fitted -- an ensemble-of-ensembles over K folds x M members stays out
of scope -- but its ensemble is trained on **D_train only**, which makes a
single ensemble valid for scoring D_meta and D_cal/D_test alike, since both
are disjoint from D_train. Tier B columns are therefore grafted onto every
split in one fixed order, exactly like temperature and the calibration
residual, so the meta and deployment feature spaces agree by construction.

This replaced an earlier arrangement that trained the ensemble on
D_train u D_meta and added Tier B only to the deployment path, leaving the
aggregators trained on (A+C) columns but scored on (A+B+C) -- a mismatch
that crashed on an assertion, so `use_ensemble=True` never ran. See
PROJECT_STATUS.md for the honest accounting of what remains simplified.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..aggregators.a0_rank import RankAverageAggregator
from ..aggregators.a1_stacking import LightGBMStackingAggregator, LogRegStackingAggregator
from ..aggregators.a2_coverage_loss import AdaptiveGatingAggregator, LinearRLBanditAggregator, MLPAggregator, RLBanditAggregator
from ..conformal.risk_control import coverage_at_guaranteed_risk_table
from ..data import splits as split_utils
from ..data.loaders import Dataset, load as load_dataset
from ..metrics.selective import (
    aurc,
    e_aurc,
    ece,
    failure_prediction_auroc,
    risk_at_coverage,
)
from ..metrics.subgroup import worst_group_selective_risk
from ..models.lightgbm_model import LightGBMWrapper
from ..models.logreg_model import LogRegWrapper
from ..signals.base import Signal, SignalBank
from ..signals.ensemble import default_tier_b_bank
from ..signals.tier_a import (
    CalibrationResidualSignal,
    TemperatureScaledMSPSignal,
    default_tier_a_bank,
)
from ..signals.tier_c import default_tier_c_bank
from .seeding import set_seed

MODEL_REGISTRY = {"lightgbm": LightGBMWrapper, "logreg": LogRegWrapper}

REPORT_COVERAGES = (1.0, 0.95, 0.90, 0.80, 0.70, 0.50)
CONFORMAL_ALPHAS = (0.01, 0.02, 0.05, 0.10)


# Signals that must be fit on a split the base model was NOT trained on,
# and so are excluded from the cross-fitting bank and grafted on separately
# by `run_experiment` (fit on D_meta using `model_train`). Both would learn
# the model's in-sample overconfidence if fit inside a training fold.
HELD_OUT_FIT_SIGNALS = (TemperatureScaledMSPSignal, CalibrationResidualSignal)


def _build_signal_bank(
    tiers: tuple[str, ...], n_classes: int = 2, ensemble: Optional[list] = None
) -> SignalBank:
    """Build the cross-fittable part of the signal bank.

    Excludes `HELD_OUT_FIT_SIGNALS`, which the runner grafts on after
    fitting them on D_meta. `n_classes` is forwarded to
    `default_tier_a_bank`, which drops `logitnorm_msp` on binary tasks
    (it is provably constant there -- see its docstring).
    """
    signals: list[Signal] = []
    if "A" in tiers:
        signals += [
            sig
            for sig in default_tier_a_bank(n_classes=n_classes)
            if not isinstance(sig, HELD_OUT_FIT_SIGNALS)
        ]
    if "C" in tiers:
        signals += default_tier_c_bank()
    if "B" in tiers:
        assert ensemble is not None, "Tier B requires an ensemble"
        signals += default_tier_b_bank(ensemble)
    return SignalBank(signals)


def _fit_model(model_name: str, X, y, seed: int, n_classes: Optional[int] = None):
    """`n_classes` is the class count of the *whole dataset*, not of `y`.
    Passing it makes the wrapper emit probability/logit columns for every
    dataset class even when this particular fit never saw one -- without it,
    a rare class absent from a split shifts every column above it and
    label-indexed reads silently return the wrong class. See
    `BaseModelWrapper`'s docstring."""
    return MODEL_REGISTRY[model_name](seed=seed, n_classes=n_classes).fit(X, y)


def _train_ensemble(
    model_name: str, X, y, n_members: int, base_seed: int, n_classes: Optional[int] = None
) -> list:
    return [
        _fit_model(model_name, X, y, seed=base_seed * 1000 + i, n_classes=n_classes)
        for i in range(n_members)
    ]


def run_experiment(
    dataset_name: str,
    model_name: str = "lightgbm",
    tiers: tuple[str, ...] = ("A", "C"),
    seed: int = 0,
    n_cv_folds: int = 5,
    use_ensemble: bool = False,
    n_ensemble_members: int = 5,
    conformal_delta: float = 0.1,
    subgroup_col: Optional[str] = None,
    raw_out_dir: Optional[str] = None,
) -> pd.DataFrame:
    set_seed(seed)
    t_start = time.time()

    ds: Dataset = load_dataset(dataset_name)
    n = len(ds.y)
    sp = split_utils.four_way_split(n, y=ds.y, seed=seed, temporal=ds.is_temporal)

    X_train, y_train = ds.X.iloc[sp.train_idx], ds.y[sp.train_idx]
    X_meta, y_meta = ds.X.iloc[sp.meta_idx], ds.y[sp.meta_idx]
    X_cal, y_cal = ds.X.iloc[sp.cal_idx], ds.y[sp.cal_idx]
    X_test, y_test = ds.X.iloc[sp.test_idx], ds.y[sp.test_idx]

    pool_idx = np.concatenate([sp.train_idx, sp.meta_idx])
    y_pool = ds.y[pool_idx]

    n_classes = int(len(ds.classes))

    # --- model_train: D_train only. Used to fit every signal that needs a
    # split the base model has not seen (temperature scaling, isotonic
    # calibration residual) and, when Tier B is enabled, to train the
    # disagreement ensemble -- see `held_out_signals` / `ensemble_train`.
    model_train = _fit_model(model_name, X_train, y_train, seed=seed, n_classes=n_classes)
    held_out_signals = [
        TemperatureScaledMSPSignal().fit(X_meta, y_meta, model_train),
        CalibrationResidualSignal().fit(X_meta, y_meta, model_train),
    ]
    held_out_names = [sig.name for sig in held_out_signals]

    # --- Tier B (ensemble disagreement), if enabled.
    #
    # Previously this trained on D_train u D_meta and was used only for the
    # deployment/test path, while the cross-fitted meta path silently
    # excluded Tier B. That left the aggregators trained on an (A+C) column
    # set but scored on an (A+B+C) one -- a column-count mismatch that the
    # assertion below turned into a hard crash, so `use_ensemble=True` never
    # ran at all.
    #
    # The fix keeps Tier B out of the cross-fitting loop (an
    # ensemble-of-ensembles over K folds x M members remains out of scope)
    # but trains it on **D_train only**, which makes one ensemble valid for
    # scoring D_meta *and* D_cal/D_test: both are disjoint from D_train, so
    # neither is scored in-sample. Tier B is then grafted on for every split
    # exactly like the held-out-fit signals above, so the meta and
    # deployment column sets agree by construction. The cost is that the
    # ensemble sees 60% of the data rather than 75%; the benefit is that
    # Tier B is honest, consistent across splits, and actually runs.
    ensemble_train = None
    tier_b_bank = None
    tier_b_names: list[str] = []
    if use_ensemble:
        ensemble_train = _train_ensemble(
            model_name, X_train, y_train, n_ensemble_members, seed, n_classes=n_classes
        )
        tier_b_bank = SignalBank(default_tier_b_bank(ensemble_train))
        tier_b_bank.fit(X_train, y_train, model_train)
        tier_b_names = list(tier_b_bank.names)

    def _grafted(X, model) -> np.ndarray:
        """Columns for the signals held out of the cross-fitting loop, in
        the fixed order `held_out_names + tier_b_names`."""
        cols = [sig.score(X, model).reshape(-1, 1) for sig in held_out_signals]
        if tier_b_bank is not None:
            cols.append(tier_b_bank.transform(X, model))
        return np.hstack(cols)

    grafted_names = held_out_names + tier_b_names

    # --- honest cross-fitted Tier A(-held-out)/C signals + correctness
    # labels over the D_train u D_meta pool (§6's fix for the leakage trap).
    # Tier B is grafted on separately (see above), so it is excluded here.
    cf_tiers = tuple(t for t in tiers if t != "B")

    def fit_model_fn(idx_abs: np.ndarray, fold: int = 0):
        Xi, yi = ds.X.iloc[idx_abs], ds.y[idx_abs]
        # Each fold gets its own derived seed (same `base * 1000 + i`
        # pattern as `_train_ensemble`). Previously all K fold models were
        # fit with the outer `seed`, so their subsampling draws followed an
        # identical pattern and the K models were less independent than the
        # cross-fitting design intends -- they differed only by which rows
        # they saw. Distinct data *and* distinct randomness is the point.
        model = _fit_model(model_name, Xi, yi, seed=seed * 1000 + fold, n_classes=n_classes)
        bank = _build_signal_bank(cf_tiers, n_classes=n_classes)
        bank.fit(Xi, yi, model)
        return {"model": model, "bank": bank}

    def compute_signals_fn(bundle: dict, idx_abs: np.ndarray) -> np.ndarray:
        Xi = ds.X.iloc[idx_abs]
        u = bundle["bank"].transform(Xi, bundle["model"])
        pred = bundle["model"].predict_proba(Xi).argmax(axis=1)
        correct = (pred == ds.y[idx_abs]).astype(float).reshape(-1, 1)
        return np.hstack([u, correct])

    oof = split_utils.cross_fitted_signals(
        fit_model_fn, compute_signals_fn, pool_idx, y_pool=y_pool, n_folds=n_cv_folds, seed=seed
    )
    U_pool, correct_pool = oof[:, :-1], oof[:, -1].astype(bool)
    is_meta = np.isin(pool_idx, sp.meta_idx)
    U_meta_cf, correct_meta_cf = U_pool[is_meta], correct_pool[is_meta]

    signal_names = _build_signal_bank(cf_tiers, n_classes=n_classes).names

    # --- naive (leaky) meta signals, for the naive-vs-cross-fit ablation
    naive_bank = _build_signal_bank(cf_tiers, n_classes=n_classes)
    naive_bank.fit(X_train, y_train, model_train)
    U_meta_naive = naive_bank.transform(X_meta, model_train)
    pred_naive = model_train.predict_proba(X_meta).argmax(axis=1)
    correct_meta_naive = pred_naive == y_meta

    # --- model_final: deployed model, D_train u D_meta
    X_pool, y_pool_full = ds.X.iloc[pool_idx], y_pool
    model_final = _fit_model(model_name, X_pool, y_pool_full, seed=seed, n_classes=n_classes)

    final_bank = _build_signal_bank(cf_tiers, n_classes=n_classes)
    # The cross-fittable part of the bank is fit on the full pool (their own
    # geometry-index / no-op fits, as documented in tier_a.py/tier_c.py).
    # Everything in `grafted_names` -- temperature, calibration residual and
    # Tier B -- was fit on a split model_final's training pool does contain,
    # so those are grafted in from objects built off `model_train`/
    # `ensemble_train` above rather than refit here (see module docstring).
    final_bank.fit(X_pool, y_pool_full, model_final)
    final_signal_names = list(final_bank.names) + grafted_names

    def score_final(X) -> np.ndarray:
        u = final_bank.transform(X, model_final)
        return np.hstack([u, _grafted(X, model_final)])

    U_cal = score_final(X_cal)
    U_test = score_final(X_test)
    pred_cal = model_final.predict_proba(X_cal).argmax(axis=1)
    pred_test = model_final.predict_proba(X_test).argmax(axis=1)
    incorrect_cal = (pred_cal != y_cal).astype(int)
    incorrect_test = (pred_test != y_test).astype(int)

    # Align meta-signal columns with final-signal columns. The cross-fit
    # meta path computes only the cross-fittable signals, so the grafted
    # ones (temperature, calibration residual, Tier B) are appended here in
    # the same order `score_final` uses. They are scored with `model_train`,
    # which never saw D_meta, so this stays honest.
    grafted_meta = _grafted(X_meta, model_train)
    U_meta_cf_full = np.hstack([U_meta_cf, grafted_meta])
    U_meta_naive_full = np.hstack([U_meta_naive, grafted_meta])
    # Column sets must agree exactly, or the aggregators train on one
    # feature space and score in another. This assertion is what used to
    # fire when `use_ensemble=True`: Tier B was present in the final bank
    # but absent from the meta path.
    assert final_signal_names == list(signal_names) + grafted_names, (
        f"meta/final signal columns disagree:\n"
        f"  final = {final_signal_names}\n"
        f"  meta  = {list(signal_names) + grafted_names}"
    )
    assert U_meta_cf_full.shape[1] == U_test.shape[1] == len(final_signal_names)

    aggregator_factories = {
        "A0_rank": lambda: RankAverageAggregator(),
        "A1_logreg": lambda: LogRegStackingAggregator(seed=seed),
        "A1_lightgbm": lambda: LightGBMStackingAggregator(seed=seed),
        "A2_mlp_bce": lambda: MLPAggregator(loss="bce", seed=seed),
        "A2_rl_bandit": lambda: RLBanditAggregator(seed=seed, penalty_c=10.0),
        # --- RL penalty sweep (sensitivity analysis) ---
        "A2_rl_c0.5": lambda: RLBanditAggregator(seed=seed, penalty_c=0.5),
        "A2_rl_c1.0": lambda: RLBanditAggregator(seed=seed, penalty_c=1.0),
        "A2_rl_c3.0": lambda: RLBanditAggregator(seed=seed, penalty_c=3.0),
        "A2_rl_c10.0": lambda: RLBanditAggregator(seed=seed, penalty_c=10.0),
        # --- Linear RL ablation (Occam's razor) ---
        "A2_rl_linear": lambda: LinearRLBanditAggregator(seed=seed, penalty_c=10.0),
        "A2_mlp_loss1": lambda: MLPAggregator(loss="loss1", seed=seed),
        "A2_mlp_loss2": lambda: MLPAggregator(loss="loss2", seed=seed),
        "A2_mlp_loss3": lambda: MLPAggregator(loss="loss3", seed=seed),
        "A2_adaptive_loss3": lambda: AdaptiveGatingAggregator(loss="loss3", seed=seed),
    }

    rows = []
    runtime_so_far = time.time() - t_start

    # Per-instance test scores, kept only when `raw_out_dir` is set. The
    # summary rows below are enough for AURC/Wilcoxon, but §7's prescribed
    # test is a *paired bootstrap over test instances*, which needs the raw
    # (uncertainty, incorrect) vectors -- see PROJECT_STATUS.md's
    # simplification about the coarser per-seed Wilcoxon substitute.
    #
    # `incorrect_test` is deliberately stored once per (dataset, tiers,
    # seed) rather than once per method: it depends only on the frozen base
    # model, so duplicating it across ~22 methods would inflate these files
    # ~20x for no information. Scores are float32 (the bootstrap only needs
    # the ordering) and the files are compressed, which keeps a full sweep
    # at a few MB instead of the ~200MB a per-method long-format parquet
    # would cost -- these are intermediate artifacts, so they are
    # gitignored while the summary parquet stays the tracked source.
    raw_scores: dict[str, np.ndarray] = {}

    def _row(cov=np.nan, risk=np.nan, accuracy=np.nan, n_accepted=np.nan, subgroup="overall",
              value=np.nan, method=""):
        return dict(
            dataset=dataset_name,
            base_model=model_name,
            # Which signal tiers produced this row. Part of the results key:
            # without it an (A,B,C) run would silently overwrite the (A,C)
            # rows for the same (dataset, model, seed), and §7's
            # signal-family-only ablation could not hold both side by side.
            tiers="".join(sorted(tiers)),
            method=method,
            seed=seed,
            coverage=cov,
            risk=risk,
            accuracy=accuracy,
            n_accepted=n_accepted,
            subgroup=subgroup,
            value=value,
            runtime=runtime_so_far,
        )

    def add_rows(method: str, uncertainty_test: np.ndarray, extra_conformal: Optional[dict] = None):
        if raw_out_dir is not None:
            raw_scores[method] = np.asarray(uncertainty_test, dtype=np.float32)
        for cov in REPORT_COVERAGES:
            risk = risk_at_coverage(uncertainty_test, incorrect_test, cov)
            k = max(1, min(len(uncertainty_test), int(round(cov * len(uncertainty_test)))))
            rows.append(
                _row(cov=cov, risk=risk, accuracy=1.0 - risk, n_accepted=k,
                     subgroup="overall", method=method)
            )
        rows.append(_row(subgroup="AURC", value=aurc(uncertainty_test, incorrect_test), method=method))
        rows.append(_row(subgroup="E_AURC", value=e_aurc(uncertainty_test, incorrect_test), method=method))
        rows.append(
            _row(
                subgroup="failure_auroc",
                value=failure_prediction_auroc(uncertainty_test, incorrect_test),
                method=method,
            )
        )
        if extra_conformal:
            for alpha, ct in extra_conformal.items():
                rows.append(
                    _row(
                        cov=ct.coverage_at_tau,
                        risk=ct.empirical_risk_at_tau,
                        accuracy=1.0 - ct.empirical_risk_at_tau,
                        subgroup=f"conformal_alpha_{alpha}",
                        method=method,
                    )
                )
        if subgroup_col and subgroup_col in ds.X.columns:
            group_test = ds.X.iloc[sp.test_idx][subgroup_col].values
            for cov in (0.9, 0.8):
                res = worst_group_selective_risk(uncertainty_test, incorrect_test, group_test, cov)
                rows.append(
                    _row(
                        cov=cov,
                        subgroup=f"max_min_gap@{cov}",
                        value=res["max_min_gap"],
                        method=method,
                    )
                )

    # --- Random abstention baseline (lower bound, §7)
    rng = np.random.RandomState(seed)
    add_rows("random", rng.rand(len(y_test)))

    # --- Oracle ordering (upper bound, §7)
    add_rows("oracle", incorrect_test.astype(float))

    # --- Single-signal baselines, evaluated directly off the final bank
    for j, name in enumerate(final_signal_names):
        add_rows(f"signal_{name}", U_test[:, j])

    # --- Aggregators, trained on the honest cross-fitted meta signals
    for agg_name, factory in aggregator_factories.items():
        agg = factory()
        agg.fit(U_meta_cf_full, correct_meta_cf)
        s_test = agg.score(U_test)
        s_cal = agg.score(U_cal)
        conformal = coverage_at_guaranteed_risk_table(
            s_cal, incorrect_cal, alphas=CONFORMAL_ALPHAS, delta=conformal_delta
        )
        add_rows(agg_name, s_test, extra_conformal=conformal)

    # --- Naive-vs-cross-fit ablation (§6, §9 week-3 gate): same aggregator
    # class (A1 logreg) trained on naive vs. honest meta signals.
    agg_naive = LogRegStackingAggregator(seed=seed)
    agg_naive.fit(U_meta_naive_full, correct_meta_naive)
    add_rows("A1_logreg_naive_meta", agg_naive.score(U_test))

    if raw_out_dir is not None:
        _save_raw_scores(
            raw_out_dir,
            dataset_name=dataset_name,
            model_name=model_name,
            tiers="".join(sorted(tiers)),
            seed=seed,
            methods=list(raw_scores),
            scores=np.vstack([raw_scores[m] for m in raw_scores]),
            incorrect=incorrect_test.astype(np.int8),
        )

    return pd.DataFrame(rows)


def raw_scores_path(
    raw_out_dir: str, dataset_name: str, model_name: str, tiers: str, seed: int
) -> Path:
    """Canonical path for one run's per-instance scores. The filename
    carries the full results key `(dataset, base_model, tiers, seed)` so a
    re-run overwrites its own file rather than accumulating duplicates --
    the same idempotency property `run_all.py` gives the summary parquet."""
    return Path(raw_out_dir) / f"{dataset_name}__{model_name}__{tiers}__seed{seed}.npz"


def _save_raw_scores(
    raw_out_dir: str,
    dataset_name: str,
    model_name: str,
    tiers: str,
    seed: int,
    methods: list[str],
    scores: np.ndarray,
    incorrect: np.ndarray,
) -> Path:
    path = raw_scores_path(raw_out_dir, dataset_name, model_name, tiers, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        methods=np.array(methods, dtype=object),
        scores=scores,
        incorrect=incorrect,
        dataset=dataset_name,
        base_model=model_name,
        tiers=tiers,
        seed=seed,
    )
    return path


def load_raw_scores(path: str | Path) -> dict:
    """Read back one `_save_raw_scores` file as
    `{"methods": [...], "scores": (n_methods, n_test), "incorrect": (n_test,), ...}`."""
    with np.load(path, allow_pickle=True) as z:
        return {
            "methods": [str(m) for m in z["methods"]],
            "scores": z["scores"],
            "incorrect": z["incorrect"],
            "dataset": str(z["dataset"]),
            "base_model": str(z["base_model"]),
            "tiers": str(z["tiers"]),
            "seed": int(z["seed"]),
        }
