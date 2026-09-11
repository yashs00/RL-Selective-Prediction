"""Tests for the disjoint 4-way split and cross-fitting utilities (§6:
the part of the project most likely to silently fail)."""
import numpy as np

from src.data.splits import Splits, cross_fitted_signals, four_way_split


def test_four_way_split_is_disjoint_and_covers_everything():
    n = 1000
    sp = four_way_split(n, seed=0)
    all_idx = np.concatenate([sp.train_idx, sp.meta_idx, sp.cal_idx, sp.test_idx])
    assert len(all_idx) == n
    assert len(set(all_idx.tolist())) == n  # no duplicates -> fully disjoint


def test_four_way_split_fractions_are_approximately_right():
    n = 10000
    sp = four_way_split(n, seed=0)
    assert abs(len(sp.train_idx) / n - 0.60) < 0.01
    assert abs(len(sp.meta_idx) / n - 0.15) < 0.01
    assert abs(len(sp.cal_idx) / n - 0.10) < 0.01
    assert abs(len(sp.test_idx) / n - 0.15) < 0.01


def test_temporal_split_preserves_order():
    n = 100
    sp = four_way_split(n, seed=0, temporal=True)
    assert list(sp.train_idx) == list(range(0, len(sp.train_idx)))
    assert sp.test_idx[-1] == n - 1  # test split is the most recent rows


def test_cross_fitted_signals_never_scores_a_point_with_a_model_that_saw_it():
    # A model that memorizes its training indices and "signals" = whether
    # the scored index was in that model's training set. Cross-fitting must
    # produce all-zeros (no point is ever scored by a model that saw it).
    pool_idx = np.arange(200)

    def fit_model_fn(idx_abs):
        return set(idx_abs.tolist())

    def compute_signals_fn(train_set, idx_abs):
        seen = np.array([[1.0 if i in train_set else 0.0] for i in idx_abs])
        return seen

    out = cross_fitted_signals(fit_model_fn, compute_signals_fn, pool_idx, n_folds=5, seed=0)
    assert (out == 0).all()
