"""Tier A signals (§3.1): logit-derived, free, always available. Each one
only reads `model.predict_proba` / `model.logits`, so they work identically
for LightGBM and logistic-regression wrappers.

Two of these — TemperatureScaledMSP and LogitNormMSP — are also listed in
§7's baseline ladder as "the cheap fix[es] that often close the gap"; they
live here and not just in aggregators because the whole point of §7 is to
compare them *as standalone baselines* against the learned aggregator.

Important leakage note (§6): `TemperatureScaledMSP.fit` must be called with
D_meta, never with D_train or D_test — the experiment runner is responsible
for passing the right split. Fitting T on D_train (which the base model has
already memorized) or D_test (which must be touched once, at the end) is
exactly the kind of quiet leakage bug this project is designed to avoid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from .base import Signal


def _softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def _center_logits(z: np.ndarray) -> np.ndarray:
    """Subtract the per-row mean across classes.

    Softmax is invariant to a per-row additive shift of the logits, so this
    is a no-op for every probability-based signal. It is *not* a no-op for
    signals that read the logits' absolute scale -- notably `energy`, which
    is `-logsumexp(z)` and therefore gauge-dependent.

    This matters because the base-model wrappers report binary logits in the
    convention `[0, z]` (see `LightGBMWrapper.logits`), which is an
    arbitrary gauge: it silently places all of the margin in the
    positive-class coordinate. Reading `logsumexp` off that gauge measures
    class identity rather than uncertainty (see `EnergySignal`). Centering
    removes the gauge freedom and makes such signals symmetric in the
    decision margin, which is what they are supposed to measure.
    """
    return z - z.mean(axis=-1, keepdims=True)


class MSPSignal(Signal):
    """Max softmax probability, negated so higher = more uncertain. This is
    Chow's rule (§2) — the baseline everything else in the paper has to
    beat."""

    name = "msp"

    def fit(self, X_train, y_train, model) -> "MSPSignal":
        return self

    def score(self, X: pd.DataFrame, model) -> np.ndarray:
        p = model.predict_proba(X)
        return 1.0 - p.max(axis=1)


class EntropySignal(Signal):
    name = "entropy"

    def fit(self, X_train, y_train, model) -> "EntropySignal":
        return self

    def score(self, X: pd.DataFrame, model) -> np.ndarray:
        p = np.clip(model.predict_proba(X), 1e-12, 1.0)
        return -(p * np.log(p)).sum(axis=1)


class MarginSignal(Signal):
    """Top-1 minus top-2, in probability or logit space (§3.1 item 3).
    Negated so higher = more uncertain (a small margin is uncertain)."""

    name = "margin"

    def __init__(self, space: str = "prob"):
        assert space in ("prob", "logit")
        self.space = space
        self.name = f"margin_{space}"

    def fit(self, X_train, y_train, model) -> "MarginSignal":
        return self

    def score(self, X: pd.DataFrame, model) -> np.ndarray:
        arr = model.predict_proba(X) if self.space == "prob" else model.logits(X)
        sorted_arr = np.sort(arr, axis=1)
        margin = sorted_arr[:, -1] - sorted_arr[:, -2]
        return -margin


class EnergySignal(Signal):
    """Negative energy score (§2 Liu et al. 2020; §3.1 item 4):
    `-logsumexp(z)`, computed on **mean-centered** logits.

    **Bug fix (binary energy was measuring class identity, not
    uncertainty).** The base wrappers emit binary logits in the `[0, s]`
    gauge, where `s` is the positive-class margin. On that gauge
    `logsumexp([0, s]) = ln(1 + e^s)`, so `-logsumexp` is *monotonically
    decreasing in s* -- it is a rank-equivalent copy of `P(class 0)`, not a
    measure of uncertainty at all. Concretely, with `s = -8` (a confident
    class-0 prediction) the old code returned `-0.0003`, while the maximally
    uncertain point `s = 0` returned `-0.6931`: the confident prediction was
    ranked as *more* uncertain than the decision boundary. Every abstention
    decision built on this signal was therefore anti-correlated with
    uncertainty for one of the two classes, which is why `signal_energy`
    scored *worse than random* on Adult (AURC 0.1824 vs. random's 0.1284)
    in the pre-fix results.

    The fix is to center the logits first (see `_center_logits`). Energy
    then becomes `-ln(2 cosh(s/2))`, which is symmetric in `s` and maximal
    at the decision boundary `s = 0`, as intended. This is a genuine sign/
    gauge bug, not a tuning choice.

    Note (documented, not a defect): on a *binary* task the fixed signal is
    still a strictly monotone transform of MSP, so it adds no new *ranking*
    information -- see `default_tier_a_bank`'s note on Tier-A rank
    degeneracy. Fixing it stops it actively harming the aggregators; it
    cannot make it independently informative."""

    name = "energy"

    def fit(self, X_train, y_train, model) -> "EnergySignal":
        return self

    def score(self, X: pd.DataFrame, model) -> np.ndarray:
        z = _center_logits(model.logits(X))
        return -logsumexp(z, axis=1)


class LogitNormMSPSignal(Signal):
    """Logit-norm-normalised MSP (§2 Cattelan & Silva 2023; §3.1 item 5):
    `softmax(z / ||z||_p)`, then negated max prob.

    **This signal is mathematically degenerate for K = 2 and is therefore
    excluded from the binary signal bank** (see `default_tier_a_bank`). The
    reason is structural, not an implementation defect:

    A softmax reads only the *differences* between logits, so for `K = 2`
    the entire signal is carried by the single scalar gap `s = z1 - z0`.
    Dividing by `||z||_p` removes exactly one degree of freedom -- the
    overall scale -- and for `K = 2` that scale *is* `|s|`. What survives is
    only `sign(s)`, so the score takes one value for every confidently-
    classified point regardless of how confident it is. Measured directly on
    a spread of margins `s`, the pre-fix implementation returned the
    constant `0.2689` for every `s != 0`.

    Every candidate repair collapses the same way, which is why this class
    is excluded rather than patched:

      * centering the logits first  -> constant `0.1956`
      * `max_logit / ||z||` on the `[0, s]` gauge -> the binary indicator
        `1[s > 0]` (class identity, strictly worse)
      * `max_logit / ||z||` on centered logits -> constant `0.7071`

    All four variants take at most two distinct values. The degeneracy is a
    property of per-sample logit normalisation at `K = 2`, so no variant of
    it can be informative here. This showed up in the pre-fix results as
    `signal_logitnorm_msp` ranking *below random abstention* (Friedman mean
    rank 20.0 vs. random's 19.7; AURC 0.3814 vs. random's 0.3705 on
    Electricity) -- a constant score induces an arbitrary ordering, so it
    behaves like random abstention plus tie-breaking noise, and it fed that
    noise into every aggregator as a feature.

    For `K >= 3` the normalisation is well-defined and informative (the
    relative spacing of the K logits survives), so the class is kept and is
    used unchanged on multiclass tasks. `score` raises on binary input
    rather than silently returning a constant.

    Worth stating in the paper: Cattelan & Silva's logit normalisation is
    *inapplicable to binary classification* for this reason. That is a
    genuine limitation of the published method, not of this codebase.
    """

    name = "logitnorm_msp"

    def __init__(self, p: int = 2):
        self.p = p

    def fit(self, X_train, y_train, model) -> "LogitNormMSPSignal":
        return self

    def score(self, X: pd.DataFrame, model) -> np.ndarray:
        z = model.logits(X)
        if z.shape[1] < 3:
            raise ValueError(
                "LogitNormMSPSignal is mathematically degenerate for K=2 "
                "(per-sample logit normalisation removes the only degree of "
                "freedom a binary softmax has, leaving a constant score) and "
                "must not be used on a binary task -- it would feed a "
                "constant column into the aggregators. `default_tier_a_bank` "
                "excludes it automatically; see this class's docstring."
            )
        norm = np.linalg.norm(z, ord=self.p, axis=1, keepdims=True)
        norm = np.clip(norm, 1e-12, None)
        p = _softmax(z / norm, axis=1)
        return 1.0 - p.max(axis=1)


class TemperatureScaledMSPSignal(Signal):
    """Temperature-scaled MSP (§2 Guo et al. 2017; §3.1 item 6). T is fit by
    minimizing NLL on whatever split is passed to `.fit` — the caller MUST
    pass D_meta, never D_train or D_test (see module docstring)."""

    name = "temp_msp"

    def __init__(self):
        self.T: float = 1.0

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, model) -> "TemperatureScaledMSPSignal":
        z = model.logits(X_train)
        y = np.asarray(y_train)

        def nll(log_T: float) -> float:
            T = np.exp(log_T)
            logp = z / T - logsumexp(z / T, axis=1, keepdims=True)
            return -logp[np.arange(len(y)), y].mean()

        res = minimize_scalar(nll, bounds=(np.log(1e-2), np.log(1e2)), method="bounded")
        self.T = float(np.exp(res.x))
        return self

    def score(self, X: pd.DataFrame, model) -> np.ndarray:
        z = model.logits(X)
        p = _softmax(z / self.T, axis=1)
        return 1.0 - p.max(axis=1)


class CalibrationResidualSignal(Signal):
    """Local miscalibration of the base model's confidence, as a signal.

    Motivation -- the Tier-A rank-degeneracy problem. On a binary task every
    other Tier-A signal here (MSP, entropy, both margins, temperature-scaled
    MSP, and even the fixed energy) is a strictly monotone transform of the
    predicted probability: measured on 20k samples, all pairwise Spearman
    correlations are exactly 1.0000. Tier A therefore supplies precisely one
    distinct *ranking*, so no aggregator over Tier A alone can order
    abstentions any differently from MSP -- which is the structural reason
    the aggregators could not beat MSP in-distribution, quite apart from any
    bug.

    This signal is deliberately a **non-monotone** function of confidence,
    so it breaks that degeneracy. Fit an isotonic regression
    `iso: confidence -> empirical accuracy` on a held-out split, then score

        r(x) = | conf(x) - iso(conf(x)) |

    i.e. how far the model's stated confidence sits from the accuracy
    actually observed at that confidence level. Because `iso` is monotone
    but crosses the identity line, `r` is V-shaped rather than monotone in
    `conf`.

    Why a non-monotone function of `conf` is worth adding even though it is
    computed from `conf` alone: the ideal abstention ordering is by
    `P(error | x)`, and for a miscalibrated model that is *not* monotone in
    the reported confidence. A linear (or any monotone) aggregator over
    `{conf}` can only ever reproduce MSP's ordering; given `conf` *and* `r`
    it can express non-monotone re-orderings -- "in the confidence band
    where this model is systematically overconfident, treat it as riskier."
    So the gain is in aggregator expressivity, not in new raw information.

    Leakage (§6): `iso` must be fit on a split the base model was **not**
    trained on, or it learns the model's in-sample overconfidence rather
    than its deployment behaviour. This class is therefore handled like
    `TemperatureScaledMSPSignal` -- the runner fits it on D_meta using a
    model trained only on D_train, and it is excluded from the
    cross-fitting bank. Do not add it to a bank that gets fit inside the
    cross-fitting folds.
    """

    name = "calib_residual"

    def __init__(self):
        self._iso = None

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, model) -> "CalibrationResidualSignal":
        from sklearn.isotonic import IsotonicRegression

        p = model.predict_proba(X_train)
        conf = p.max(axis=1)
        correct = (p.argmax(axis=1) == np.asarray(y_train)).astype(float)
        # increasing=True: higher stated confidence should mean higher
        # empirical accuracy. out_of_bounds="clip" so unseen confidences at
        # deployment map to the nearest fitted endpoint instead of NaN.
        self._iso = IsotonicRegression(
            y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
        ).fit(conf, correct)
        return self

    def score(self, X: pd.DataFrame, model) -> np.ndarray:
        if self._iso is None:
            raise RuntimeError(
                "CalibrationResidualSignal.score called before .fit -- the "
                "isotonic calibration map must be fit on a split the base "
                "model did not train on (see class docstring)."
            )
        conf = model.predict_proba(X).max(axis=1)
        return np.abs(conf - self._iso.predict(conf))


def default_tier_a_bank(n_classes: int = 2) -> list[Signal]:
    """Tier-A signal bank.

    `n_classes` gates the signals that are only well-defined for K >= 3.
    `LogitNormMSPSignal` is excluded on binary tasks because per-sample
    logit normalisation is provably constant there (see its docstring); it
    was ranking below random abstention while feeding that noise into every
    aggregator.

    Note on what this bank can and cannot do (measured, see
    `CalibrationResidualSignal`): on a binary task the probability-derived
    members below are all strictly monotone transforms of one another, so
    as *standalone baselines* they are rank-identical to MSP and as
    *aggregator features* they span a single ranking. Tier B (ensemble
    disagreement), Tier C (geometry/density) and `calib_residual` are the
    only sources of genuinely different orderings on binary problems.

    `TemperatureScaledMSPSignal` and `CalibrationResidualSignal` are both
    fit on a held-out split by the runner and are grafted on there, not
    fit inside the cross-fitting folds -- see `src/experiment/runner.py`.
    """
    bank: list[Signal] = [
        MSPSignal(),
        EntropySignal(),
        MarginSignal(space="prob"),
        MarginSignal(space="logit"),
        EnergySignal(),
    ]
    if n_classes >= 3:
        bank.append(LogitNormMSPSignal())
    bank.append(TemperatureScaledMSPSignal())
    return bank
