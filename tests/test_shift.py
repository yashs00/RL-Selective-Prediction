"""Tests for synthetic covariate shift (RQ2's controlled shift battery).

The whole value of a *synthetic* shift is that its presence and magnitude
are known rather than assumed, so these tests pin exactly that: intensity 0
changes nothing, larger intensity moves the data further, and -- the one
that matters most -- features and labels cannot desynchronise under the
resampling kind.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.shift import apply_covariate_shift


def _toy(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "a": rng.normal(0, 1, n),
            "b": rng.normal(5, 2, n),
            "cat": rng.choice(["x", "y"], n),  # non-numeric: must be left alone
        }
    )


def test_intensity_zero_is_an_exact_no_op():
    """The sweep's baseline is the untouched test set, so intensity 0 must
    be byte-identical -- not merely 'close'."""
    X = _toy()
    for kind in ("gaussian_noise", "feature_scale", "subpopulation"):
        out, idx = apply_covariate_shift(X, kind=kind, intensity=0.0)
        pd.testing.assert_frame_equal(out, X)
        assert idx is None


def test_larger_intensity_moves_the_data_further():
    X = _toy()
    ref = X
    dists = []
    for intensity in (0.1, 0.5, 2.0):
        out, _ = apply_covariate_shift(
            X, kind="gaussian_noise", intensity=intensity,
            rng=np.random.default_rng(0), reference=ref,
        )
        dists.append(float(np.abs(out[["a", "b"]].to_numpy() - X[["a", "b"]].to_numpy()).mean()))
    assert dists == sorted(dists), f"not monotone in intensity: {dists}"


def test_non_numeric_columns_are_untouched():
    X = _toy()
    for kind in ("gaussian_noise", "feature_scale"):
        out, _ = apply_covariate_shift(X, kind=kind, intensity=1.0,
                                       rng=np.random.default_rng(0))
        pd.testing.assert_series_equal(out["cat"], X["cat"])


def test_gaussian_noise_scales_with_the_reference_not_the_input():
    """Perturbation magnitude must be defined by the *training* spread, so
    that a test set with an unusually narrow spread doesn't silently get a
    weaker shift than intended."""
    X = _toy()
    narrow = X.copy()
    narrow[["a", "b"]] = narrow[["a", "b"]] * 0.01  # much tighter test set

    out, _ = apply_covariate_shift(
        narrow, kind="gaussian_noise", intensity=1.0,
        rng=np.random.default_rng(0), reference=X,
    )
    moved = float(np.abs(out[["a", "b"]].to_numpy() - narrow[["a", "b"]].to_numpy()).std())
    # Reference std for 'b' is ~2, so the perturbation should be O(1),
    # far larger than the narrow test set's own ~0.02 spread.
    assert moved > 0.5, f"noise scaled to the input, not the reference (std={moved:.4f})"


def test_feature_scale_is_systematic_not_random():
    X = _toy()
    out, _ = apply_covariate_shift(X, kind="feature_scale", intensity=0.5)
    np.testing.assert_allclose(out["a"].to_numpy(), X["a"].to_numpy() * 1.5)


# --- the desynchronisation footgun -------------------------------------


def test_subpopulation_returns_an_index_so_labels_can_be_realigned():
    """`subpopulation` resamples rows, so it MUST hand back the index --
    an earlier design exposed a separate helper that redrew the resample
    from its own generator, which could mislabel every row."""
    X = _toy()
    out, idx = apply_covariate_shift(
        X, kind="subpopulation", intensity=1.0, rng=np.random.default_rng(0)
    )
    assert idx is not None
    assert len(idx) == len(out) == len(X)
    # The returned frame must be exactly X's rows at `idx`, so that
    # `y[idx]` is the matching label vector.
    expected = X.iloc[idx].reset_index(drop=True)
    pd.testing.assert_frame_equal(out, expected)


def test_subpopulation_actually_shifts_the_feature_distribution():
    X = _toy(n=2000)
    out, _ = apply_covariate_shift(
        X, kind="subpopulation", intensity=3.0, rng=np.random.default_rng(0)
    )
    # Favouring one tail of the leading principal direction must move at
    # least one feature's mean well away from the original.
    shift_a = abs(out["a"].mean() - X["a"].mean()) / X["a"].std()
    shift_b = abs(out["b"].mean() - X["b"].mean()) / X["b"].std()
    assert max(shift_a, shift_b) > 0.3, (
        f"subpopulation shift barely moved the distribution "
        f"(a={shift_a:.3f}, b={shift_b:.3f} std units)"
    )


def test_value_perturbing_kinds_return_no_index():
    """They preserve row order, so there is nothing to realign -- and a
    spurious index would invite a pointless (and possibly wrong) reorder."""
    X = _toy()
    for kind in ("gaussian_noise", "feature_scale"):
        _, idx = apply_covariate_shift(X, kind=kind, intensity=1.0,
                                       rng=np.random.default_rng(0))
        assert idx is None


def test_rejects_bad_arguments():
    X = _toy()
    with pytest.raises(ValueError, match="unknown shift kind"):
        apply_covariate_shift(X, kind="teleport", intensity=0.5)
    with pytest.raises(ValueError, match="intensity must be >= 0"):
        apply_covariate_shift(X, kind="gaussian_noise", intensity=-1.0)
