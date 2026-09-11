"""Conformal risk-control tests, covering the Hoeffding-Bentkus bound that
replaced the simple Hoeffding one (PROJECT_STATUS.md item R3).

The point of the HB bound is that it is *tighter without being weaker*, so
both halves need pinning: it must dominate the old bound numerically, and
it must still deliver the guarantee it advertises. The second test is also
the "conformal coverage empirical verification" experiment that
`full_research_analysis.md` lists as a P1 missing experiment (its claim C6,
"Need empirical verification that coverage holds across seeds").
"""
from __future__ import annotations

import numpy as np
import pytest

from src.conformal.risk_control import (
    _hb_p_value,
    _hb_ucb,
    _hoeffding_ucb,
    coverage_at_guaranteed_risk_table,
    select_threshold,
)


def _synthetic_cal(n: int, seed: int, risk_slope: float = 0.2):
    """Scores uniform on [0,1] with P(incorrect | s) = risk_slope * s, i.e.
    a well-behaved, genuinely informative risk score."""
    rng = np.random.RandomState(seed)
    s = rng.rand(n)
    incorrect = (rng.rand(n) < risk_slope * s).astype(int)
    return s, incorrect


# --- The bound itself ----------------------------------------------------


@pytest.mark.parametrize(
    "risk_hat,n",
    [(0.0, 1000), (0.01, 2000), (0.02, 5000), (0.05, 1000), (0.10, 500), (0.3, 200)],
)
def test_hb_is_never_looser_than_plain_hoeffding(risk_hat, n):
    """HB must dominate the simple bound everywhere -- that is its whole
    reason for existing. (Both are valid, so if HB were ever *looser* we
    would simply be choosing the worse of two valid bounds.)"""
    delta = 0.1 / 200  # a representative Bonferroni-corrected delta
    assert _hb_ucb(risk_hat, n, delta) <= _hoeffding_ucb(risk_hat, n, delta) + 1e-9


def test_hb_ucb_is_at_least_the_observed_risk():
    """An *upper* bound below the empirical mean would be nonsense."""
    for risk_hat, n in [(0.0, 100), (0.05, 500), (0.2, 1000), (0.5, 50)]:
        assert _hb_ucb(risk_hat, n, 0.05) >= risk_hat - 1e-9


def test_hb_p_value_cannot_certify_below_observed_risk():
    """H0: risk >= alpha must be unrejectable when the *observed* risk is
    already at or above alpha, regardless of n."""
    assert _hb_p_value(0.10, 10_000, 0.10) == 1.0
    assert _hb_p_value(0.20, 10_000, 0.05) == 1.0


def test_hb_ucb_tightens_as_n_grows():
    ucbs = [_hb_ucb(0.02, n, 0.05) for n in (100, 1_000, 10_000, 100_000)]
    assert ucbs == sorted(ucbs, reverse=True), f"not monotone in n: {ucbs}"


# --- The symptom this fix targets ---------------------------------------


def test_hb_admits_small_alpha_where_hoeffding_abstains_entirely():
    """The documented failure of the simple bound: on 20,000 well-behaved
    calibration points it could not satisfy alpha in {0.01, 0.02, 0.05} at
    all -- coverage 0, i.e. abstain on everything, which makes RQ3
    unanswerable in practice rather than merely conservative. HB must
    recover usable coverage at alpha=0.05."""
    s, incorrect = _synthetic_cal(20_000, seed=0)

    hoeff = coverage_at_guaranteed_risk_table(s, incorrect, delta=0.1, bound="hoeffding")
    hb = coverage_at_guaranteed_risk_table(s, incorrect, delta=0.1, bound="hb")

    assert hoeff[0.05].coverage_at_tau == 0.0, (
        "the simple Hoeffding bound was expected to abstain entirely at "
        "alpha=0.05; if this now passes, re-derive the comparison"
    )
    assert hb[0.05].coverage_at_tau > 0.2, (
        f"HB only reached coverage {hb[0.05].coverage_at_tau:.4f} at alpha=0.05; "
        "expected it to recover substantial coverage"
    )
    # And it must not lose ground where the old bound already worked.
    assert hb[0.10].coverage_at_tau >= hoeff[0.10].coverage_at_tau


# --- Validity: the tighter bound must still keep its promise ------------


def test_conformal_guarantee_holds_empirically_across_trials():
    """The guarantee is P(selective risk on fresh data <= alpha) >= 1-delta.

    Calibrate on one sample, evaluate on an independent one, repeat, and
    count violations. With delta=0.1 the violation rate must stay at or
    below 10% (and in practice far below, since HB is still conservative).
    This is the empirical check that a tighter bound did not become an
    invalid one.
    """
    alpha, delta = 0.10, 0.1
    n_trials, n_cal, n_test = 60, 2_000, 2_000
    violations, evaluated = 0, 0

    for trial in range(n_trials):
        s_cal, inc_cal = _synthetic_cal(n_cal, seed=trial)
        s_test, inc_test = _synthetic_cal(n_test, seed=1000 + trial)

        thr = select_threshold(s_cal, inc_cal, alpha=alpha, delta=delta, bound="hb")
        accepted = s_test <= thr.tau
        if accepted.sum() == 0:
            continue  # abstained on everything: vacuously safe, not a violation
        evaluated += 1
        if float(inc_test[accepted].mean()) > alpha:
            violations += 1

    assert evaluated >= n_trials // 2, "too many trials abstained to judge validity"
    rate = violations / evaluated
    assert rate <= delta, (
        f"selective risk exceeded alpha={alpha} in {rate:.1%} of {evaluated} trials, "
        f"above the delta={delta:.0%} the bound promises"
    )


def test_unknown_bound_name_raises():
    s, incorrect = _synthetic_cal(200, seed=0)
    with pytest.raises(ValueError, match="unknown bound"):
        select_threshold(s, incorrect, alpha=0.1, bound="not_a_bound")
