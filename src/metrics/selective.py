"""Selective-prediction metrics (§7). `uncertainty` is always "higher =
abstain sooner" (the same convention as `Signal.score` and every
aggregator's `.score`), so all functions here share one sort direction.

AURC/E-AURC use the standard discrete definition from the selective-
classification literature (Geifman & El-Yaniv and follow-ups, §2):
sort predictions by ascending uncertainty (most confident first), and at
each prefix length k report the cumulative error rate among the k most
confident predictions. AURC is the mean of that curve over all n prefixes;
E-AURC subtracts the same quantity computed under the *optimal* achievable
ordering (all correct predictions ranked before all incorrect ones), which
the plan calls out as necessary for cross-dataset aggregation since raw
AURC is confounded by base accuracy.
"""
from __future__ import annotations

import numpy as np


def risk_coverage_curve(
    uncertainty: np.ndarray, incorrect: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (coverages, risks), both length n, coverages = 1/n..n/n."""
    uncertainty = np.asarray(uncertainty)
    incorrect = np.asarray(incorrect).astype(float)
    n = len(uncertainty)
    order = np.argsort(uncertainty, kind="stable")  # ascending: most confident first
    incorrect_sorted = incorrect[order]
    cum_errors = np.cumsum(incorrect_sorted)
    k = np.arange(1, n + 1)
    coverages = k / n
    risks = cum_errors / k
    return coverages, risks


def aurc(uncertainty: np.ndarray, incorrect: np.ndarray) -> float:
    _, risks = risk_coverage_curve(uncertainty, incorrect)
    return float(risks.mean())


def _optimal_aurc(incorrect: np.ndarray) -> float:
    """AURC under the best-achievable ordering: every correct prediction
    ranked before every incorrect one."""
    incorrect = np.asarray(incorrect).astype(float)
    n = len(incorrect)
    n_err = int(incorrect.sum())
    optimal_sorted = np.concatenate([np.zeros(n - n_err), np.ones(n_err)])
    cum_errors = np.cumsum(optimal_sorted)
    k = np.arange(1, n + 1)
    risks = cum_errors / k
    return float(risks.mean())


def e_aurc(uncertainty: np.ndarray, incorrect: np.ndarray) -> float:
    return aurc(uncertainty, incorrect) - _optimal_aurc(incorrect)


def risk_at_coverage(uncertainty: np.ndarray, incorrect: np.ndarray, target_coverage: float) -> float:
    coverages, risks = risk_coverage_curve(uncertainty, incorrect)
    n = len(coverages)
    k = int(round(target_coverage * n))
    k = max(1, min(n, k))
    return float(risks[k - 1])


def coverage_at_risk(uncertainty: np.ndarray, incorrect: np.ndarray, target_risk: float) -> float:
    """Max achievable coverage such that cumulative risk stays <=
    target_risk, indexing directly into the actual (possibly
    non-monotonic) risk-coverage curve rather than assuming monotonicity."""
    coverages, risks = risk_coverage_curve(uncertainty, incorrect)
    feasible = coverages[risks <= target_risk]
    return float(feasible.max()) if len(feasible) else 0.0


def failure_prediction_auroc(uncertainty: np.ndarray, incorrect: np.ndarray) -> float:
    """AUROC for discriminating correct (0) from incorrect (1) predictions
    using `uncertainty` as the score (§7 "Failure-prediction AUROC")."""
    from sklearn.metrics import roc_auc_score

    incorrect = np.asarray(incorrect)
    if incorrect.min() == incorrect.max():
        return float("nan")  # undefined: no errors, or all errors
    return float(roc_auc_score(incorrect, uncertainty))


def ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error with equal-*mass* bins (§7), i.e. bins are
    quantiles of the confidence distribution rather than equal-width -- more
    robust when confidence is heavily skewed toward 1, which tabular GBMs
    often are."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)

    order = np.argsort(conf)
    conf_sorted = conf[order]
    correct_sorted = correct[order]
    n = len(conf)
    bin_edges = np.linspace(0, n, n_bins + 1).astype(int)

    total = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if hi <= lo:
            continue
        bin_conf = conf_sorted[lo:hi].mean()
        bin_acc = correct_sorted[lo:hi].mean()
        total += (hi - lo) / n * abs(bin_conf - bin_acc)
    return float(total)


def brier_score(probs: np.ndarray, y_true: np.ndarray) -> float:
    n, k = probs.shape
    onehot = np.zeros((n, k))
    onehot[np.arange(n), y_true] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def nll(probs: np.ndarray, y_true: np.ndarray, eps: float = 1e-12) -> float:
    p_true = np.clip(probs[np.arange(len(y_true)), y_true], eps, 1.0)
    return float(-np.log(p_true).mean())
