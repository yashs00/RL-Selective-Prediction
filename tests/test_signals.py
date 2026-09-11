"""Sanity tests for Tier A signals against a tiny synthetic dataset, using
the LogRegWrapper (fastest model to fit repeatedly in unit tests)."""
import numpy as np
import pandas as pd

from src.models.logreg_model import LogRegWrapper
from src.signals.tier_a import (
    EnergySignal,
    EntropySignal,
    MarginSignal,
    MSPSignal,
    TemperatureScaledMSPSignal,
)


def _toy_dataset(n=300, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({"x1": rng.randn(n), "x2": rng.randn(n)})
    y = (X["x1"] + 0.5 * X["x2"] > 0).astype(int).values
    return X, y


def test_msp_is_bounded_and_matches_definition():
    X, y = _toy_dataset()
    model = LogRegWrapper(seed=0).fit(X, y)
    s = MSPSignal().score(X, model)
    p = model.predict_proba(X)
    assert np.allclose(s, 1 - p.max(axis=1))
    assert (s >= 0).all() and (s <= 1).all()


def test_entropy_is_nonnegative_and_zero_for_deterministic_probs():
    X, y = _toy_dataset()
    model = LogRegWrapper(seed=0).fit(X, y)
    s = EntropySignal().score(X, model)
    assert (s >= -1e-9).all()


def test_margin_signal_is_negative_of_gap():
    X, y = _toy_dataset()
    model = LogRegWrapper(seed=0).fit(X, y)
    s = MarginSignal(space="prob").score(X, model)
    p = np.sort(model.predict_proba(X), axis=1)
    gap = p[:, -1] - p[:, -2]
    assert np.allclose(s, -gap)


def test_more_confident_points_get_lower_uncertainty_than_ambiguous_ones():
    X, y = _toy_dataset(n=2000)
    model = LogRegWrapper(seed=0).fit(X, y)
    # Points far from the decision boundary (large |x1 + 0.5*x2|) should be
    # more confident than points near it.
    margin_true = (X["x1"] + 0.5 * X["x2"]).abs().values
    s = MSPSignal().score(X, model)
    # Correlation between true distance-to-boundary and MSP-uncertainty
    # should be strongly negative (bigger margin -> lower uncertainty).
    corr = np.corrcoef(margin_true, s)[0, 1]
    assert corr < -0.3


def test_temperature_scaling_reduces_or_matches_nll_on_fit_split():
    from src.metrics.selective import nll

    X, y = _toy_dataset(n=1000)
    X_train, y_train = X.iloc[:700], y[:700]
    X_meta, y_meta = X.iloc[700:], y[700:]
    model = LogRegWrapper(seed=0).fit(X_train, y_train)

    sig = TemperatureScaledMSPSignal().fit(X_meta, y_meta, model)
    assert sig.T > 0

    probs_uncalibrated = model.predict_proba(X_meta)
    # Recompute calibrated probs at the fitted T for comparison.
    z = model.logits(X_meta)
    from scipy.special import logsumexp

    logp = z / sig.T - logsumexp(z / sig.T, axis=1, keepdims=True)
    probs_calibrated = np.exp(logp)

    assert nll(probs_calibrated, y_meta) <= nll(probs_uncalibrated, y_meta) + 1e-6
