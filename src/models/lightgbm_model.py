"""LightGBM wrapper — the primary tabular base model (§5: "it's the actual
state of the art here").

`logits`: LightGBM's `predict(..., raw_score=True)` returns the true
pre-sigmoid/pre-softmax margins, so this is an exact logit, not a surrogate.
For binary tasks LightGBM returns a single raw score per row (the positive
class margin); we expand it to a 2-column [0, z] logit matrix so every
downstream signal can assume a (n, K) logits shape uniformly.

`features`: LightGBM has no learned embedding, so we use the preprocessed
input space as the representation for Tier-C distance-based signals. This
is a standard, documented approximation for GBM base models (see §3.1,
Tier C docstring) — not a hidden shortcut.

**Bug fix (seed variance):** LightGBM's own defaults are
`bagging_fraction=1.0`/`feature_fraction=1.0`, i.e. no row/column
subsampling. With those defaults, `random_state` has no source of
randomness left to act on -- on a dataset with a deterministic split
(`is_temporal=True`, e.g. Electricity), every seed then trains a
bit-for-bit identical model, so "10 seeds" only exercises randomness
downstream (cross-fitting folds, aggregator init), not the base model
itself. This was caught by inspecting the Electricity 10-seed run, where
every `signal_msp` AURC was identical to machine precision across seeds.
Fixed by defaulting to `bagging_fraction=0.8`, `bagging_freq=1`,
`feature_fraction=0.8` below, so `random_state` actually varies the
trained model; still overridable via `**lgb_kwargs` (and hence via Hydra's
`model.*` config group) for anyone who wants to turn subsampling back off
deliberately.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from .base import BaseModelWrapper
from .preprocessing import build_preprocessor


class LightGBMWrapper(BaseModelWrapper):
    # Defaults chosen so `random_state` actually produces a different
    # trained model per seed (see the module docstring's "Bug fix" note) --
    # LightGBM's own defaults (bagging_fraction=feature_fraction=1.0) leave
    # random_state with nothing to act on. Any caller can override these
    # via lgb_kwargs (e.g. from a Hydra `model.*` config) to turn
    # subsampling back off deliberately.
    _RANDOMNESS_DEFAULTS = dict(
        bagging_fraction=0.8,
        bagging_freq=1,
        feature_fraction=0.8,
    )

    def __init__(
        self, seed: int = 0, n_estimators: int = 300, n_classes: int | None = None, **lgb_kwargs
    ):
        self.seed = seed
        # Global class count, so predict_proba/logits always return columns
        # for every dataset class even if this fit never saw one -- see
        # BaseModelWrapper's docstring on class-space alignment.
        self.n_classes = n_classes
        self.n_estimators = n_estimators
        self.lgb_kwargs = {**self._RANDOMNESS_DEFAULTS, **lgb_kwargs}
        self.preprocessor = None
        self.model: LGBMClassifier | None = None
        # Classes actually seen by *this* fit -- distinct from `self.n_classes`,
        # the whole dataset's class count used for column alignment.
        self.n_classes_seen_: int = 0

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LightGBMWrapper":
        self.preprocessor = build_preprocessor(X)
        Xt = self.preprocessor.fit_transform(X)
        self.n_classes_seen_ = int(len(np.unique(y)))
        objective = "binary" if self.n_classes_seen_ == 2 else "multiclass"
        self.model = LGBMClassifier(
            random_state=self.seed,
            n_estimators=self.n_estimators,
            objective=objective,
            num_class=self.n_classes_seen_ if objective == "multiclass" else None,
            verbosity=-1,
            **self.lgb_kwargs,
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
        raw = self.model.predict(Xt, raw_score=True)
        raw = np.asarray(raw)
        if raw.ndim == 1:
            # binary: raw is the positive-class margin -> expand to (n, 2)
            z = np.zeros((raw.shape[0], 2), dtype=float)
            z[:, 1] = raw
            return self._align_logits(z)
        return self._align_logits(raw)

    def features(self, X: pd.DataFrame) -> np.ndarray:
        return self._transform(X)
