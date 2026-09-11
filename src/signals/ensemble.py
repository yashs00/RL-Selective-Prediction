"""Tier B signals (§3.1): model-internal, need multiple forward passes or an
ensemble. Implemented here as ensemble-disagreement signals over a list of
independently-trained base-model wrappers (§5: "5 members with different
seeds, for the disagreement signals").

These signals deliberately break the `Signal.fit(X_train, y_train, model)`
contract's use of `model`: the ensemble is trained once by the experiment
runner (each member on its own bootstrap/seed) and injected at construction
time, so `model` in `fit`/`score` is accepted for interface-uniformity with
`SignalBank` but ignored. MC-dropout (items 7-8 in §3.1) is the natural
Tier-B source for neural nets; it's omitted here because both base models in
this repo (LightGBM, logistic regression) have no dropout layers to sample —
add it under `src/signals/` following this same pattern if/when a neural
base model (ResNet/WideResNet for the image benchmarks, §4) is wired in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Signal


class _EnsembleSignalBase(Signal):
    def __init__(self, ensemble: list):
        assert len(ensemble) >= 2, "ensemble disagreement needs >=2 members"
        self.ensemble = ensemble

    def fit(self, X_train, y_train, model) -> "_EnsembleSignalBase":
        return self  # ensemble members are trained externally by the runner

    def _member_probs(self, X: pd.DataFrame) -> np.ndarray:
        # (n_members, n, K)
        return np.stack([m.predict_proba(X) for m in self.ensemble], axis=0)


class EnsembleVoteEntropySignal(_EnsembleSignalBase):
    """Entropy of the ensemble's majority-vote distribution (§3.1 item 9)."""

    name = "ensemble_vote_entropy"

    def score(self, X: pd.DataFrame, model) -> np.ndarray:
        probs = self._member_probs(X)  # (M, n, K)
        votes = probs.argmax(axis=-1)  # (M, n)
        n_classes = probs.shape[-1]
        M = votes.shape[0]
        counts = np.stack(
            [(votes == k).sum(axis=0) for k in range(n_classes)], axis=1
        )  # (n, K)
        freq = counts / M
        freq = np.clip(freq, 1e-12, 1.0)
        return -(freq * np.log(freq)).sum(axis=1)


class EnsembleMeanPairwiseKLSignal(_EnsembleSignalBase):
    """Mean pairwise KL divergence between ensemble members' predictive
    distributions (§3.1 item 10)."""

    name = "ensemble_mean_pairwise_kl"

    def score(self, X: pd.DataFrame, model) -> np.ndarray:
        probs = np.clip(self._member_probs(X), 1e-12, 1.0)  # (M, n, K)
        M = probs.shape[0]
        n = probs.shape[1]
        total = np.zeros(n, dtype=float)
        n_pairs = 0
        for i in range(M):
            for j in range(M):
                if i == j:
                    continue
                kl = (probs[i] * (np.log(probs[i]) - np.log(probs[j]))).sum(axis=1)
                total += kl
                n_pairs += 1
        return total / n_pairs


class EnsembleTopClassVarianceSignal(_EnsembleSignalBase):
    """Variance across members of the predicted top-class probability
    (§3.1 item 11). Uses the class predicted by the mean ensemble
    distribution so the quantity is well-defined per input."""

    name = "ensemble_topclass_variance"

    def score(self, X: pd.DataFrame, model) -> np.ndarray:
        probs = self._member_probs(X)  # (M, n, K)
        mean_probs = probs.mean(axis=0)  # (n, K)
        top_class = mean_probs.argmax(axis=1)  # (n,)
        n = probs.shape[1]
        top_probs = probs[:, np.arange(n), top_class]  # (M, n)
        return top_probs.var(axis=0)


def default_tier_b_bank(ensemble: list) -> list[Signal]:
    return [
        EnsembleVoteEntropySignal(ensemble),
        EnsembleMeanPairwiseKLSignal(ensemble),
        EnsembleTopClassVarianceSignal(ensemble),
    ]
