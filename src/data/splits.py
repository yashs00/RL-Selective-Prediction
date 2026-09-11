"""Splitting and cross-fitting.

This is the part of the project the plan (§6) singles out as the most
common silent failure mode: a leakage bug produces *beautiful* results, so
it goes undetected unless you build the honest version deliberately.

Two things live here:

1. `four_way_split` — the disjoint D_train / D_meta / D_cal / D_test split
   (60/15/10/15) that every experiment must use. Random by default; pass
   `temporal=True` to split by row order instead (required for shift
   datasets like Electricity, per §4 — never shuffle those).

2. `cross_fitted_signals` — implements the fix in §6: computing signal
   vectors for D_meta using a model that was trained on D_meta produces an
   unrepresentative (overconfident) signal distribution. The fix is to
   K-fold cross-fit: train K models, each excluding one fold of
   (D_train ∪ D_meta), and score each fold with the model that never saw it.
   The final deployed model is then retrained on all of D_train ∪ D_meta.

`naive_meta_signals` is also provided — deliberately "wrong" — solely so the
naive-vs-cross-fit ablation in the plan's week-3 gate can be measured
directly instead of asserted.
"""
from __future__ import annotations

import dataclasses
import inspect
from typing import Callable

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold


@dataclasses.dataclass
class Splits:
    train_idx: np.ndarray
    meta_idx: np.ndarray
    cal_idx: np.ndarray
    test_idx: np.ndarray


def four_way_split(
    n: int,
    y: np.ndarray | None = None,
    seed: int = 0,
    temporal: bool = False,
    fracs: tuple[float, float, float, float] = (0.60, 0.15, 0.10, 0.15),
) -> Splits:
    assert abs(sum(fracs) - 1.0) < 1e-8, "split fractions must sum to 1"
    n_train = int(round(fracs[0] * n))
    n_meta = int(round(fracs[1] * n))
    n_cal = int(round(fracs[2] * n))
    # test gets the remainder so rounding never drops/duplicates a row
    n_test = n - n_train - n_meta - n_cal

    if temporal:
        # Split by row order (time), never randomly. §4: Electricity is the
        # active temporal-shift dataset relying on this (Diabetes-130 was
        # also registered for this but later dropped from the project --
        # see PROJECT_STATUS.md).
        idx = np.arange(n)
    else:
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n)

    train_idx = idx[:n_train]
    meta_idx = idx[n_train : n_train + n_meta]
    cal_idx = idx[n_train + n_meta : n_train + n_meta + n_cal]
    test_idx = idx[n_train + n_meta + n_cal :]
    assert len(test_idx) == n_test

    return Splits(train_idx, meta_idx, cal_idx, test_idx)


def naive_meta_signals(
    fit_model_fn: Callable[[np.ndarray], object],
    compute_signals_fn: Callable[[object, np.ndarray], np.ndarray],
    train_idx: np.ndarray,
    meta_idx: np.ndarray,
) -> np.ndarray:
    """The leaky way: train on D_train, score D_meta with that same model.

    Exists only to be compared against `cross_fitted_signals` in the
    naive-vs-cross-fit ablation (§6, §9 week-3 gate). Do not use this path
    for any number that ends up in the paper.
    """
    model = fit_model_fn(train_idx)
    return compute_signals_fn(model, meta_idx)


def cross_fitted_signals(
    fit_model_fn: Callable[[np.ndarray], object],
    compute_signals_fn: Callable[[object, np.ndarray], np.ndarray],
    pool_idx: np.ndarray,
    y_pool: np.ndarray | None = None,
    n_folds: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Honest out-of-fold signal vectors for `pool_idx` (typically
    D_train ∪ D_meta), per the fix in §6:

      1. Split pool into K folds.
      2. For each fold k: fit the base model on the other K-1 folds.
      3. Score fold k with that model (which never saw fold k).
      4. Concatenate the out-of-fold scores.

    `fit_model_fn(idx)` must fit and return a model given absolute row
    indices; `compute_signals_fn(model, idx)` must return an (len(idx), m)
    signal matrix for those same absolute indices. Row order of the
    returned array matches `pool_idx` order (not fold order).

    `fit_model_fn` may optionally accept a second argument, the fold index
    `k`, so a caller can give each fold's model its own derived seed.
    Without that, every fold model shares one `random_state` and is
    differentiated only by which rows it sees -- which leaves the K models
    less independent than intended (their row/column subsampling draws all
    follow the same pattern). Passing the fold index is backward
    compatible: single-argument `fit_model_fn`s still work unchanged.
    """
    n = len(pool_idx)
    if y_pool is not None:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_iter = splitter.split(np.zeros(n), y_pool)
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_iter = splitter.split(np.zeros(n))

    # Does the caller want the fold index? Inspect once rather than
    # try/except per fold, so a genuine TypeError raised *inside* a
    # two-argument fit function can't be silently swallowed and retried.
    try:
        wants_fold = len(inspect.signature(fit_model_fn).parameters) >= 2
    except (TypeError, ValueError):  # builtins / C-callables expose no signature
        wants_fold = False

    out = None
    for k, (train_pos, held_pos) in enumerate(fold_iter):
        train_abs = pool_idx[train_pos]
        held_abs = pool_idx[held_pos]
        model = fit_model_fn(train_abs, k) if wants_fold else fit_model_fn(train_abs)
        sig = compute_signals_fn(model, held_abs)
        if out is None:
            out = np.empty((n,) + sig.shape[1:], dtype=sig.dtype)
        out[held_pos] = sig
    assert out is not None, "pool_idx must be non-empty"
    return out
