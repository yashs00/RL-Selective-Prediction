"""Conformal risk-control threshold selection (§3.5, §2 Angelopoulos et al.,
"Conformal Risk Control" 2022 and "Learn Then Test" 2021).

Given any heuristic risk score s(x) (from any aggregator in this repo) and
a calibration set D_cal, choose the largest threshold `tau_hat` (i.e. the
highest-coverage threshold) such that a finite-sample upper confidence
bound on selective risk stays below the user's target `alpha`, with
confidence `1 - delta`. Under exchangeability between D_cal and the
deployment distribution, this gives

    P( selective_risk(tau_hat) <= alpha ) >= 1 - delta

distribution-free, with no assumption about how the aggregator was trained
— it wraps A0/A1/A2 identically.

Implementation note: we compute an upper confidence bound (UCB) on the
empirical selective risk at each candidate threshold, search over a grid of
thresholds from `s(x)` on D_cal, and Bonferroni-correct across the grid so
the "search over thresholds" step doesn't itself invalidate the guarantee.

Two bounds are available via `bound=`:

  * `"hb"` (default) -- the **Hoeffding-Bentkus** bound of Bates et al.
    (RCPS, 2021) / Angelopoulos et al. ("Learn Then Test", 2021), i.e. the
    minimum of a Chernoff-Hoeffding (KL-form) tail bound and Bentkus's
    binomial tail bound. The `e` factor on the Bentkus term is what makes
    taking that minimum valid; this is the published form, not an ad-hoc
    combination of two bounds.
  * `"hoeffding"` -- the original simple `R_hat + sqrt(log(1/delta)/2n)`
    bound. Kept so the before/after is *measurable* rather than asserted:
    it was documented as provably loose (it could not admit
    alpha in {0.01, 0.02, 0.05} even on 20,000 well-behaved synthetic
    calibration points, only alpha=0.10).

Both are distribution-free and assumption-identical; HB is simply tighter,
which matters precisely at the small-alpha end where the simple bound
abstained on everything and made RQ3 unanswerable in practice.

Caveat, stated in the paper too (§3.5): exchangeability fails under
distribution shift, so the guarantee is valid in-distribution only and is
*evaluated empirically, not claimed*, under shift.
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy import optimize, stats


def _hoeffding_ucb(risk_hat: float, n: int, delta: float) -> float:
    """Upper confidence bound on the true mean of a [0,1]-bounded random
    variable given its empirical mean over n samples, at confidence 1-delta.

    The simple (sqrt-form) Hoeffding bound. Variance-oblivious, and so
    provably loose when the true risk is small -- which is exactly the
    regime a selective classifier operates in. Kept for comparison against
    `_hb_ucb`; see the module docstring."""
    if n == 0:
        return 1.0
    return float(risk_hat + np.sqrt(np.log(1.0 / delta) / (2.0 * n)))


def _h1(a: float, b: float) -> float:
    """KL divergence between Bernoulli(a) and Bernoulli(b), the exponent in
    the Chernoff-Hoeffding tail bound. Defined for 0 <= a <= b <= 1, with
    the usual `0 * log 0 = 0` convention."""
    if not 0.0 <= a <= 1.0 or not 0.0 < b < 1.0:
        return 0.0
    term_a = a * np.log(a / b) if a > 0 else 0.0
    term_b = (1.0 - a) * np.log((1.0 - a) / (1.0 - b)) if a < 1 else 0.0
    return float(term_a + term_b)


def _hb_p_value(risk_hat: float, n: int, alpha: float) -> float:
    """Hoeffding-Bentkus p-value for H0: true risk >= alpha.

    p_HB = min( exp(-n * h1(min(R_hat, alpha), alpha)),
                e * P(Binom(n, alpha) <= ceil(n * R_hat)) )

    A small p-value licenses rejecting H0, i.e. certifying that the true
    risk is below `alpha`. Per Bates et al. (RCPS) / Angelopoulos et al.
    ("Learn Then Test"), taking the minimum of these two terms is valid
    because of the `e` factor on the Bentkus term.
    """
    if n == 0:
        return 1.0
    if risk_hat >= alpha:
        return 1.0  # cannot certify a risk below what we already observed
    if alpha >= 1.0:
        return 0.0

    hoeffding_term = float(np.exp(-n * _h1(min(risk_hat, alpha), alpha)))
    bentkus_term = float(np.e * stats.binom.cdf(np.ceil(n * risk_hat), n, alpha))
    return min(1.0, hoeffding_term, bentkus_term)


def _hb_ucb(risk_hat: float, n: int, delta: float) -> float:
    """Upper confidence bound obtained by inverting `_hb_p_value`.

    `_hb_p_value` is decreasing in `alpha` (a weaker claim is easier to
    certify), so the certifiable risk levels form an upper interval
    [UCB, 1] and the UCB is its infimum -- found here by bisection. This
    returns a bound directly comparable to `_hoeffding_ucb`'s, so the two
    are interchangeable at the call site.
    """
    if n == 0:
        return 1.0
    if _hb_p_value(risk_hat, n, 1.0 - 1e-12) > delta:
        return 1.0  # not certifiable at any level

    lo, hi = max(risk_hat, 0.0), 1.0
    # `brentq` needs a sign change; frame it as p(alpha) - delta = 0.
    f = lambda a: _hb_p_value(risk_hat, n, a) - delta  # noqa: E731
    if f(hi) > 0:
        return 1.0
    if f(lo) <= 0:
        return float(lo)
    try:
        return float(optimize.brentq(f, lo, hi, xtol=1e-6))
    except ValueError:
        return 1.0


_BOUNDS = {"hb": _hb_ucb, "hoeffding": _hoeffding_ucb}


@dataclasses.dataclass
class ConformalThreshold:
    tau: float
    alpha: float
    delta: float
    empirical_risk_at_tau: float
    ucb_at_tau: float
    coverage_at_tau: float
    n_cal: int


def select_threshold(
    s_cal: np.ndarray,
    incorrect_cal: np.ndarray,
    alpha: float,
    delta: float = 0.1,
    n_grid: int = 200,
    bound: str = "hb",
) -> ConformalThreshold:
    """Select tau_hat = the largest threshold (highest coverage) such that
    the Bonferroni-corrected UCB on selective risk is <= alpha.

    `s_cal`: (n,) risk scores on D_cal (higher = more uncertain; abstain
             when s(x) > tau).
    `incorrect_cal`: (n,) in {0,1}, 1 if the frozen base model was wrong.
    `bound`: "hb" (Hoeffding-Bentkus, default, tighter) or "hoeffding"
             (the original simple bound). See the module docstring.
    """
    n = len(s_cal)
    assert n > 0, "calibration set must be non-empty"
    if bound not in _BOUNDS:
        raise ValueError(f"unknown bound {bound!r}; choose from {sorted(_BOUNDS)}")
    ucb_fn = _BOUNDS[bound]

    # Candidate thresholds: `n_grid` quantiles of the observed scores. This
    # must stay bounded (not grow with n) -- the Bonferroni correction below
    # divides delta by the grid size, so a grid that silently grows to
    # size ~n (e.g. by including every unique observed score) makes the
    # bound needlessly, severely conservative on large calibration sets. A
    # coarser grid can only miss the exact optimal threshold by one
    # quantile step; it never invalidates the guarantee itself.
    grid = np.unique(np.quantile(s_cal, np.linspace(0, 1, n_grid)))
    delta_per_test = delta / len(grid)  # Bonferroni over the threshold search

    best = None
    for tau in grid:
        accepted = s_cal <= tau
        n_acc = int(accepted.sum())
        if n_acc == 0:
            continue
        risk_hat = float(incorrect_cal[accepted].mean())
        ucb = ucb_fn(risk_hat, n_acc, delta_per_test)
        coverage = n_acc / n
        if ucb <= alpha:
            if best is None or coverage > best.coverage_at_tau:
                best = ConformalThreshold(
                    tau=float(tau),
                    alpha=alpha,
                    delta=delta,
                    empirical_risk_at_tau=risk_hat,
                    ucb_at_tau=ucb,
                    coverage_at_tau=coverage,
                    n_cal=n,
                )

    if best is None:
        # No threshold satisfies the risk bound at this alpha/delta/n_cal:
        # abstain on everything (coverage 0) rather than silently violate
        # the guarantee. This is itself an informative result (report it,
        # don't hide it) -- it means either D_cal is too small for this
        # alpha, or the aggregator isn't good enough at any coverage.
        best = ConformalThreshold(
            tau=float(-np.inf),
            alpha=alpha,
            delta=delta,
            empirical_risk_at_tau=0.0,
            ucb_at_tau=0.0,
            coverage_at_tau=0.0,
            n_cal=n,
        )
    return best


def coverage_at_guaranteed_risk_table(
    s_cal: np.ndarray,
    incorrect_cal: np.ndarray,
    alphas: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10),
    delta: float = 0.1,
    bound: str = "hb",
) -> dict[float, ConformalThreshold]:
    """The primary conformal table from §3.5/§7: coverage at guaranteed
    risk <= alpha, for alpha in {1%, 2%, 5%, 10%}."""
    return {
        a: select_threshold(s_cal, incorrect_cal, alpha=a, delta=delta, bound=bound)
        for a in alphas
    }
