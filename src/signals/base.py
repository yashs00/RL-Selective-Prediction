"""Common signal interface (§8, non-negotiable rule #1): every signal
implements `fit(train_data, model) -> self` and `score(X) -> np.ndarray`.
This is what makes leave-one-signal-out ablations a config change instead
of a rewrite, and lets `SignalBank` concatenate an arbitrary subset of
signals into the vector u(x) in §3.1 without special-casing any one of them.

Higher score = more uncertain / more likely to be wrong, uniformly across
every signal in this codebase (a few raw quantities are naturally
"confidence-like"; those signals negate them internally so the sign
convention never leaks into aggregator or metric code).
"""
from __future__ import annotations

import abc

import numpy as np
import pandas as pd


class Signal(abc.ABC):
    name: str

    @abc.abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, model) -> "Signal": ...

    @abc.abstractmethod
    def score(self, X: pd.DataFrame, model) -> np.ndarray:
        """Return a 1-D array, higher = more uncertain."""
        ...


class SignalBank:
    """Fits a list of signals and concatenates their scores into u(x)."""

    def __init__(self, signals: list[Signal]):
        self.signals = signals

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, model) -> "SignalBank":
        for s in self.signals:
            s.fit(X_train, y_train, model)
        return self

    def transform(self, X: pd.DataFrame, model) -> np.ndarray:
        cols = [np.asarray(s.score(X, model), dtype=float).reshape(-1, 1) for s in self.signals]
        return np.hstack(cols)

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.signals]
