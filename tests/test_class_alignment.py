"""Class-space alignment tests (see `BaseModelWrapper`'s docstring).

sklearn/LightGBM emit one column per class *present in training*, so a
class absent from a split shifts every column above it. The loud failure
(missing last class -> IndexError in temperature scaling) is how this was
found on wine-quality-white, whose rarest quality level has ~5 rows in
4,898. The quiet failure matters more: with a *middle* class missing,
label-indexed reads return a different class's probability with no error
at all. These tests pin the quiet case specifically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.lightgbm_model import LightGBMWrapper
from src.models.logreg_model import LogRegWrapper

WRAPPERS = [LightGBMWrapper, LogRegWrapper]


def _data(n=300, k=5, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = rng.integers(0, k, size=n)
    return X, y


@pytest.mark.parametrize("Wrapper", WRAPPERS)
def test_aligned_output_has_a_column_per_global_class(Wrapper):
    X, y = _data()
    keep = y != 2  # train without the middle class
    m = Wrapper(seed=0, n_classes=5).fit(X[keep], y[keep])

    proba, logits = m.predict_proba(X), m.logits(X)
    assert proba.shape == (len(X), 5)
    assert logits.shape == (len(X), 5)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.isfinite(logits).all(), "padded logits must stay finite for logsumexp"


@pytest.mark.parametrize("Wrapper", WRAPPERS)
def test_unseen_class_gets_zero_probability(Wrapper):
    X, y = _data()
    keep = y != 2
    m = Wrapper(seed=0, n_classes=5).fit(X[keep], y[keep])
    assert m.predict_proba(X)[:, 2].sum() == 0.0


@pytest.mark.parametrize("Wrapper", WRAPPERS)
def test_columns_are_not_silently_shifted(Wrapper):
    """The quiet corruption: without alignment, the model's column j is
    class `classes_[j]`, so a missing middle class makes every later column
    describe a different class than its index suggests. After alignment,
    class c's probability must sit at column c."""
    X, y = _data()
    keep = y != 2
    aligned = Wrapper(seed=0, n_classes=5).fit(X[keep], y[keep])
    unaligned = Wrapper(seed=0).fit(X[keep], y[keep])

    pa, pu = aligned.predict_proba(X), unaligned.predict_proba(X)
    assert pu.shape[1] == 4, "expected the model to expose only the 4 seen classes"
    for local_j, cls in enumerate(unaligned.classes_):
        np.testing.assert_allclose(pa[:, int(cls)], pu[:, local_j], atol=1e-10)


@pytest.mark.parametrize("Wrapper", WRAPPERS)
def test_label_indexed_read_does_not_raise_when_last_class_is_missing(Wrapper):
    """The original crash: temperature scaling does
    `logp[np.arange(n), y]`, which raised IndexError when the *last* class
    was absent from the fit."""
    X, y = _data()
    keep = y != 4  # drop the highest class
    m = Wrapper(seed=0, n_classes=5).fit(X[keep], y[keep])
    logp = np.log(np.clip(m.predict_proba(X), 1e-12, None))
    logp[np.arange(len(y)), y]  # must not raise


@pytest.mark.parametrize("Wrapper", WRAPPERS)
def test_alignment_is_a_no_op_when_every_class_is_present(Wrapper):
    X, y = _data()
    with_n = Wrapper(seed=0, n_classes=5).fit(X, y)
    without = Wrapper(seed=0).fit(X, y)
    np.testing.assert_allclose(with_n.predict_proba(X), without.predict_proba(X), atol=1e-12)
    np.testing.assert_allclose(with_n.logits(X), without.logits(X), atol=1e-12)


@pytest.mark.parametrize("Wrapper", WRAPPERS)
def test_predict_returns_global_labels_either_way(Wrapper):
    """`predict` must return dataset labels, not local column indices --
    aligned (columns already global) and unaligned (needs classes_ lookup)."""
    X, y = _data()
    keep = y != 2
    aligned = Wrapper(seed=0, n_classes=5).fit(X[keep], y[keep])
    unaligned = Wrapper(seed=0).fit(X[keep], y[keep])
    for m in (aligned, unaligned):
        preds = m.predict(X)
        assert set(np.unique(preds)).issubset({0, 1, 3, 4}), preds[:10]
    np.testing.assert_array_equal(aligned.predict(X), unaligned.predict(X))


def test_binary_path_is_unaffected():
    X, y = _data(k=2)
    m = LightGBMWrapper(seed=0, n_classes=2).fit(X, y)
    assert m.predict_proba(X).shape == (len(X), 2)
    assert m.logits(X).shape == (len(X), 2)
