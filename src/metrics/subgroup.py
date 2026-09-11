"""Subgroup / fairness metrics for RQ4 (§7 secondary metrics, §3.5 caveat
about disparities, §2 Jones et al. 2021 "Selective classification can
magnify disparities across groups").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .selective import risk_at_coverage


def worst_group_selective_risk(
    uncertainty: np.ndarray,
    incorrect: np.ndarray,
    group: np.ndarray | pd.Series,
    target_coverage: float,
) -> dict:
    """Selective risk at a fixed *overall* coverage threshold, computed
    per-group, plus the max-min gap across groups (§7: "max-min gap, and
    worst-group selective risk at fixed overall coverage").

    Important: the acceptance threshold is chosen once on the *overall*
    population at `target_coverage`, then applied uniformly -- this is
    what actually happens at deployment (one global threshold), and is why
    a subgroup can end up with a much lower realised coverage than the
    nominal target if its uncertainty scores skew high (the disparity
    Jones et al. describe).
    """
    n = len(uncertainty)
    order = np.argsort(uncertainty, kind="stable")
    k = max(1, min(n, int(round(target_coverage * n))))
    accept_mask = np.zeros(n, dtype=bool)
    accept_mask[order[:k]] = True

    group = np.asarray(group)
    groups = np.unique(group)

    per_group = {}
    for g in groups:
        g_mask = group == g
        accepted_g = g_mask & accept_mask
        n_accepted = int(accepted_g.sum())
        realised_coverage = n_accepted / max(int(g_mask.sum()), 1)
        risk = float(incorrect[accepted_g].mean()) if n_accepted > 0 else float("nan")
        per_group[g] = {
            "n_group": int(g_mask.sum()),
            "n_accepted": n_accepted,
            "realised_coverage": realised_coverage,
            "selective_risk": risk,
        }

    risks = [v["selective_risk"] for v in per_group.values() if not np.isnan(v["selective_risk"])]
    max_min_gap = float(max(risks) - min(risks)) if len(risks) >= 2 else float("nan")

    return {"per_group": per_group, "max_min_gap": max_min_gap}
