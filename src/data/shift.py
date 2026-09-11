"""Controlled synthetic covariate shift for the RQ2 shift battery (plan §4
lists "synthetic covariate shift" alongside the temporal and image-corruption
experiments).

**Why synthetic shift earns its place rather than being a fallback.** RQ2 --
"is the aggregation advantage regime-dependent, absent in-distribution and
present under shift?" -- is the project's load-bearing question, and it
currently rests on exactly one dataset (Electricity). The obvious way to add
a second is another temporally-ordered dataset, but that requires *proving*
the row order encodes time; Diabetes-130 passed that check and still had to
be dropped for unrelated data-quality reasons, at the cost of hours. A
synthetic shift is verifiable by construction: the shift is applied here, so
its presence, direction and magnitude are known exactly rather than assumed.
It also gives something no single shifted dataset can -- an *intensity
sweep*, so the claim under test becomes the far sharper "the gap over MSP
grows monotonically with shift" instead of "the gap differs on one other
dataset".

Applied to **D_test only**. D_train / D_meta / D_cal stay clean, which is
what makes this a deployment-time shift: the base model, the aggregator and
the conformal threshold were all fit on the pre-shift distribution and are
then asked to cope, exactly as in the temporal setting.

Honest limitation to state alongside any result from this: a synthetic
covariate shift is not a natural one. It tests robustness to a *known,
controlled* perturbation, so it complements Electricity's real (if weak)
temporal drift rather than substituting for it. Report both, and do not
present synthetic-shift results as evidence about naturally occurring shift.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

KINDS = ("gaussian_noise", "feature_scale", "subpopulation")


def _numeric_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]


def apply_covariate_shift(
    X: pd.DataFrame,
    kind: str = "gaussian_noise",
    intensity: float = 0.5,
    rng: Optional[np.random.Generator] = None,
    reference: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, Optional[np.ndarray]]:
    """Return `(X_shifted, row_index)`.

    `row_index` is the array of selected rows for `subpopulation` (which
    resamples) and `None` for the kinds that perturb values in place. The
    caller **must** apply a non-None index to `y` as well --
    `y_shifted = y[row_index]` -- or features and labels desynchronise.

    Returning both from one call is deliberate: an earlier version of this
    module exposed a separate `shifted_index_for_labels` helper that redrew
    the resample from its own generator, so two independently-seeded draws
    could silently disagree and mislabel every row. One call, one draw, no
    way to get it wrong.

    `intensity = 0` is always a no-op, so the sweep's own baseline is the
    untouched test set rather than a separately-computed one.

    `reference` supplies the per-feature scale (use D_train, so the
    perturbation magnitude is defined by the *training* distribution rather
    than by the test set being perturbed); defaults to `X` itself.

    kinds:
      * `gaussian_noise` -- add `N(0, (intensity * sigma_j)^2)` to each
        numeric feature j. The canonical tabular covariate shift: it moves
        mass off the training manifold without changing the marginal
        class balance, so any degradation is attributable to the inputs
        rather than to label drift.
      * `feature_scale` -- multiply numeric features by `1 + intensity`, a
        systematic (rather than random) miscalibration of the input scale,
        e.g. a re-instrumented sensor.
      * `subpopulation` -- resample rows with weights favouring one tail of
        the first principal direction, i.e. shift *which* inputs arrive
        without perturbing any feature value. Complements the other two:
        every row remains a genuine, unmodified observation.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown shift kind {kind!r}; choose from {KINDS}")
    if intensity < 0:
        raise ValueError("intensity must be >= 0")
    if intensity == 0:
        return X.copy(), None

    rng = rng if rng is not None else np.random.default_rng(0)
    ref = reference if reference is not None else X
    num_cols = _numeric_columns(X)
    out = X.copy()

    if not num_cols and kind != "subpopulation":
        return out, None  # nothing numeric to perturb

    if kind == "gaussian_noise":
        sigma = ref[num_cols].std(axis=0).replace(0, 1.0).fillna(1.0).to_numpy()
        noise = rng.normal(0.0, 1.0, size=(len(out), len(num_cols))) * (intensity * sigma)
        out[num_cols] = out[num_cols].to_numpy() + noise
        return out, None

    if kind == "feature_scale":
        out[num_cols] = out[num_cols].to_numpy() * (1.0 + intensity)
        return out, None

    # subpopulation: weight rows by their position along the leading
    # principal direction, then resample with replacement. `intensity`
    # controls how sharply the tail is favoured. Every returned row is a
    # real, unmodified observation -- only *which* rows arrive changes.
    if not num_cols or len(out) < 2:
        return out, None
    Zc = np.nan_to_num(out[num_cols].to_numpy(dtype=float))
    Zc = Zc - Zc.mean(axis=0)
    scale = Zc.std(axis=0)
    scale[scale == 0] = 1.0
    Zc = Zc / scale
    # Leading right singular vector = first principal direction.
    _, _, vt = np.linalg.svd(Zc, full_matrices=False)
    proj = Zc @ vt[0]
    p = np.exp(intensity * (proj - proj.max()))  # max-shift for stability
    total = p.sum()
    if not np.isfinite(total) or total <= 0:
        return out, None
    idx = rng.choice(len(out), size=len(out), replace=True, p=p / total)
    return out.iloc[idx].reset_index(drop=True), idx
