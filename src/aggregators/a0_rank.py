"""A0 — z-score sum / rank average (§3.2). No learning: a legitimate cheap
baseline and a sanity check that the signal bank is even informative before
any aggregator is trained on it.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


class RankAverageAggregator:
    name = "A0_rank_average"

    def fit(self, U_meta: np.ndarray, correct_meta: np.ndarray) -> "RankAverageAggregator":
        # correct_meta unused: A0 has no learned parameters, kept for a
        # uniform aggregator interface with A1/A2.
        return self

    def score(self, U: np.ndarray) -> np.ndarray:
        # Average the per-column rank (higher signal value = more
        # uncertain, by the SignalBank convention), normalised to [0, 1].
        ranks = np.column_stack([rankdata(U[:, j]) for j in range(U.shape[1])])
        ranks /= U.shape[0]
        return ranks.mean(axis=1)


class ZScoreSumAggregator:
    name = "A0_zscore_sum"

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, U_meta: np.ndarray, correct_meta: np.ndarray) -> "ZScoreSumAggregator":
        self.mean_ = U_meta.mean(axis=0)
        self.std_ = U_meta.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def score(self, U: np.ndarray) -> np.ndarray:
        z = (U - self.mean_) / self.std_
        return z.sum(axis=1)
