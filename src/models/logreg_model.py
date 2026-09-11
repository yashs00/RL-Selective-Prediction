"""Logistic regression wrapper — the "interpretable floor" base model (§5).

`logits` uses `decision_function`, sklearn's true pre-sigmoid/softmax score,
expanded to a (n, 2) matrix for binary tasks for the same reason as the
LightGBM wrapper. `features` is the preprocessed input space (there is no
hidden layer to borrow a representation from).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .base import BaseModelWrapper
from .preprocessing import build_preprocessor


class LogRegWrapper(BaseModelWrapper):
    def __init__(
        self, seed: int = 0, C: float = 1.0, max_iter: int = 2000, n_classes: int | None = None
    ):
        self.seed = seed
        # See BaseModelWrapper's docstring on class-space alignment.
        self.n_classes = n_classes
        self.C = C
        self.max_iter = max_iter
        self.preprocessor = None
        self.model: LogisticRegression | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LogRegWrapper":
        self.preprocessor = build_preprocessor(X)
        Xt = self.preprocessor.fit_transform(X)
        self.model = LogisticRegression(
            C=self.C, max_iter=self.max_iter, random_state=self.seed
        )
        self.model.fit(Xt, y)
        self.classes_ = self.model.classes_
        return self

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.preprocessor.transform(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._align_proba(self.model.predict_proba(self._transform(X)))

    def logits(self, X: pd.DataFrame) -> np.ndarray:
        Xt = self._transform(X)
        raw = self.model.decision_function(Xt)
        raw = np.asarray(raw)
        if raw.ndim == 1:
            z = np.zeros((raw.shape[0], 2), dtype=float)
            z[:, 1] = raw
            return self._align_logits(z)
        return self._align_logits(raw)

    def features(self, X: pd.DataFrame) -> np.ndarray:
        return self._transform(X)
