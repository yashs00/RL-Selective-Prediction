"""A1 — linear + GBM stacking on correctness (§3.2): fit a classifier to
predict `1[f(x) != y]` from `u(x)` on a disjoint meta-split. This is
ordinary stacking; it is the internal baseline A2 has to beat (§3.3,
§11 risk register: "this is just stacking" is the expected review
objection, and the A1-vs-A2 ablation is the direct answer to it).
"""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LogRegStackingAggregator:
    """L2-regularised logistic regression on the signal vector.

    **Bug fix (unstandardised features under L2 regularisation).** The
    signal vector mixes wildly different scales: Tier-A signals are
    probabilities in [0, 1], while Tier-C signals are raw distances that run
    to 1e3 and beyond (and, before its clamp, `trust_score` could reach
    1e12). L2 regularisation penalises the *coefficient* magnitudes, so to
    exert equal influence on the decision a small-scale feature needs a
    proportionally larger coefficient -- and is therefore penalised far more
    heavily for the same effect. The optimiser's least-cost solution is to
    shrink the coefficients on the informative probability signals towards
    zero and lean on the large-scale distance signals, which is precisely
    backwards: MSP is the strongest single signal in every dataset measured
    here, and `knn_distance`/`mahalanobis` are among the weakest.

    Standardising first makes the penalty scale-free, so regularisation
    strength reflects how *useful* a signal is rather than what units it
    happens to be reported in. The scaler is fit inside `fit` and reused in
    `score`, so no D_test statistics ever reach it (§6).

    Note that A2's torch aggregators already standardise their inputs
    (`_Standardizer` in `a2_coverage_loss.py`); only this estimator was
    missing it, which also made the A1-vs-A2 comparison unfair to A1.
    """

    name = "A1_logreg"

    def __init__(
        self,
        C: float = 1.0,
        penalty: str = "elasticnet",
        l1_ratio: float = 0.5,
        solver: str = "saga",
        seed: int = 0,
    ):
        self.C = C
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.solver = solver
        self.seed = seed
        self.model: Pipeline | None = None

    def fit(self, U_meta: np.ndarray, correct_meta: np.ndarray) -> "LogRegStackingAggregator":
        incorrect = 1 - correct_meta.astype(int)
        kwargs = dict(C=self.C, max_iter=2000, random_state=self.seed)
        if self.penalty == "elasticnet":
            kwargs.update(penalty="elasticnet", solver=self.solver, l1_ratio=self.l1_ratio)
        else:
            kwargs.update(penalty=self.penalty)
        self.model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(**kwargs)),
            ]
        )
        self.model.fit(U_meta, incorrect)
        return self

    def score(self, U: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(U)[:, 1]


class LightGBMStackingAggregator:
    name = "A1_lightgbm"

    def __init__(self, n_estimators: int = 200, seed: int = 0):
        self.n_estimators = n_estimators
        self.seed = seed
        self.model: LGBMClassifier | None = None

    def fit(self, U_meta: np.ndarray, correct_meta: np.ndarray) -> "LightGBMStackingAggregator":
        incorrect = 1 - correct_meta.astype(int)
        self.model = LGBMClassifier(
            n_estimators=self.n_estimators, random_state=self.seed, verbosity=-1
        )
        self.model.fit(U_meta, incorrect)
        return self

    def score(self, U: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(U)[:, 1]
