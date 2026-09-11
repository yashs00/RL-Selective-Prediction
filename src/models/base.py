"""Uniform base-model interface (§8: "every signal implements the same
interface... one file per signal"; models get the same treatment so signals
never need to know which base model produced them).

Every wrapper exposes:
    fit(X, y) -> self
    predict_proba(X) -> (n, K) array
    logits(X) -> (n, K) array of pre-softmax scores (real margins where the
                 model provides them; a monotone surrogate otherwise, noted
                 per-wrapper below)
    features(X) -> (n, d) array, the representation Tier-C signals compute
                   distances in. For tabular models without a learned
                   embedding, this is the preprocessed input space itself —
                   a documented approximation, not a hidden one.
    classes_ : array of class labels seen during fit (0..K-1)

Preprocessing (imputation, scaling, categorical encoding) is fit *inside*
each wrapper's `.fit`, never outside it, so that plugging a wrapper into the
cross-fitting loop in `src/data/splits.py` can never leak fold information
through a globally-fit preprocessor (§6, "fit preprocessing inside the
training fold only").

**Class-space alignment (`n_classes`) -- a silent-correctness fix.**
sklearn and LightGBM return one probability/logit column per class *present
in their training data*, in sorted order. On a multiclass dataset with a
rare class, a training split can miss it entirely -- wine-quality-white has
7 quality levels but only ~5 rows of the rarest, so a 60% train split can
easily contain 6 of the 7. Downstream code indexes these matrices by label
(`logp[np.arange(n), y]` in temperature scaling, for instance), which then
goes wrong in one of two ways:

  * the missing class is the *last* one -> `IndexError`. Loud, and how this
    was actually caught.
  * the missing class is in the *middle* -> every column above it shifts
    left, so `logp[i, y_i]` silently reads a **different class's**
    probability. No error, just wrong numbers everywhere downstream.

The second case is the dangerous one, so the fix belongs here rather than
in each consumer: pass `n_classes` (the class count of the *whole* dataset)
and every wrapper returns matrices with exactly that many columns, in
global label order, padding classes it never saw. Consumers can then index
by label unconditionally. Wrappers still work without `n_classes` (they
fall back to the model's own class space), so this is opt-in but always
supplied by `run_experiment`.
"""
from __future__ import annotations

import abc

import numpy as np
import pandas as pd


class BaseModelWrapper(abc.ABC):
    classes_: np.ndarray
    # Class count of the whole dataset, when the caller supplies it. See the
    # module docstring: this is what lets consumers index a probability or
    # logit matrix by label without worrying about which classes a given
    # training split happened to contain.
    n_classes: int | None = None

    # Padded logit columns get `row_max - LOGIT_PAD_BELOW_MAX`, i.e. a
    # finite value whose softmax weight is ~1e-14 -- effectively zero
    # probability while staying finite through logsumexp (the energy
    # signal) and never entering a top-2 margin.
    LOGIT_PAD_BELOW_MAX: float = 32.0

    @abc.abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "BaseModelWrapper": ...

    @abc.abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...

    @abc.abstractmethod
    def logits(self, X: pd.DataFrame) -> np.ndarray: ...

    @abc.abstractmethod
    def features(self, X: pd.DataFrame) -> np.ndarray: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        idx = np.argmax(self.predict_proba(X), axis=1)
        if self.n_classes is not None:
            # `predict_proba` is already in global label order, so the
            # argmax column *is* the label.
            return idx
        # No global class space given: columns are the model's own classes,
        # so map the local column index back to a label.
        return np.asarray(self.classes_)[idx]

    # --- class-space alignment helpers, shared by every wrapper ---------

    def _needs_alignment(self) -> bool:
        return (
            self.n_classes is not None
            and getattr(self, "classes_", None) is not None
            and len(self.classes_) != self.n_classes
        )

    def _align_proba(self, proba: np.ndarray) -> np.ndarray:
        """Scatter a model-local (n, k_local) probability matrix into
        (n, n_classes) global label order; unseen classes get 0.0."""
        if not self._needs_alignment():
            return proba
        out = np.zeros((proba.shape[0], self.n_classes), dtype=float)
        out[:, np.asarray(self.classes_, dtype=int)] = proba
        return out

    def _align_logits(self, logits: np.ndarray) -> np.ndarray:
        """Scatter a model-local (n, k_local) logit matrix into
        (n, n_classes) global label order; unseen classes are pushed far
        below the row max rather than to -inf, so downstream logsumexp and
        margin computations stay finite."""
        if not self._needs_alignment():
            return logits
        row_max = logits.max(axis=1, keepdims=True)
        out = np.repeat(row_max - self.LOGIT_PAD_BELOW_MAX, self.n_classes, axis=1)
        out[:, np.asarray(self.classes_, dtype=int)] = logits
        return out
