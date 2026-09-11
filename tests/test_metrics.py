"""Unit tests for AURC/E-AURC with analytically-known cases (§8: "Write a
test where AURC is analytically known (perfect ordering, random ordering).
A metric bug is the most likely source of a wrong headline number, and
it's the cheapest possible bug to prevent.").
"""
import numpy as np

from src.metrics.selective import (
    aurc,
    coverage_at_risk,
    e_aurc,
    ece,
    risk_at_coverage,
    risk_coverage_curve,
)


def test_perfect_ordering_gives_zero_excess_aurc():
    # All correct predictions given lower uncertainty than all incorrect
    # ones -> this *is* the optimal ordering, so E-AURC must be exactly 0.
    n, n_err = 200, 40
    incorrect = np.concatenate([np.zeros(n - n_err), np.ones(n_err)])
    uncertainty = np.arange(n, dtype=float)  # already sorted correct-first
    assert abs(e_aurc(uncertainty, incorrect)) < 1e-9


def test_perfect_ordering_aurc_matches_closed_form():
    # Under the optimal ordering, cumulative risk at prefix k is
    # max(0, k - (n - n_err)) / k. AURC is the mean of that over k=1..n.
    n, n_err = 100, 20
    incorrect = np.concatenate([np.zeros(n - n_err), np.ones(n_err)])
    uncertainty = np.arange(n, dtype=float)
    k = np.arange(1, n + 1)
    expected_risks = np.maximum(0, k - (n - n_err)) / k
    expected_aurc = expected_risks.mean()
    assert abs(aurc(uncertainty, incorrect) - expected_aurc) < 1e-9


def test_worst_case_evenly_interleaved_errors_matches_analytic_subset():
    # Place exactly one error every 5th position (systematic, not random):
    # at every k that is a multiple of 5, cumulative risk is EXACTLY
    # (k/5)/k = 0.2, analytically, regardless of ordering elsewhere.
    n = 200
    incorrect = np.zeros(n)
    incorrect[4::5] = 1.0  # positions 5,10,15,...  (0-indexed: 4,9,14,...)
    uncertainty = np.arange(n, dtype=float)  # ordering == position

    coverages, risks = risk_coverage_curve(uncertainty, incorrect)
    k = np.arange(1, n + 1)
    at_multiples_of_5 = (k % 5 == 0)
    assert np.allclose(risks[at_multiples_of_5], 0.2)

    # And the overall AURC must be close to 0.2 (exact at every 5th point,
    # bounded excursions elsewhere).
    assert abs(aurc(uncertainty, incorrect) - 0.2) < 0.02


def test_aurc_is_worse_than_or_equal_to_optimal_for_random_ordering():
    rng = np.random.RandomState(0)
    n, n_err = 500, 100
    incorrect = np.zeros(n)
    incorrect[rng.choice(n, n_err, replace=False)] = 1.0
    random_uncertainty = rng.rand(n)
    assert e_aurc(random_uncertainty, incorrect) >= -1e-9


def test_risk_and_coverage_at_extremes():
    n = 50
    incorrect = np.zeros(n)
    incorrect[40:] = 1.0  # the 10 errors are the *most uncertain* points
    uncertainty = np.concatenate([np.zeros(40), np.ones(10)])  # errors ranked last (best case)
    assert risk_at_coverage(uncertainty, incorrect, 1.0) == 10 / 50
    assert risk_at_coverage(uncertainty, incorrect, 0.8) == 0.0  # first 40 are all correct
    assert coverage_at_risk(uncertainty, incorrect, target_risk=0.0) == 0.8


def test_ece_perfect_calibration_is_zero():
    rng = np.random.RandomState(0)
    n = 2000
    # Construct probs such that empirical accuracy in every quantile bin
    # exactly matches mean confidence: draw confidence, then draw
    # correctness as a Bernoulli with that exact probability.
    conf = rng.uniform(0.5, 1.0, size=n)
    correct = (rng.rand(n) < conf).astype(int)
    probs = np.zeros((n, 2))
    probs[:, 1] = conf
    probs[:, 0] = 1 - conf
    y_true = np.where(correct == 1, 1, 0)
    # ECE should be small (statistical, not exact-zero, since correctness
    # is sampled) -- a loose bound catches a broken bin-averaging formula
    # without being a flaky statistical test.
    assert ece(probs, y_true, n_bins=15) < 0.05
