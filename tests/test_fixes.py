"""Regression tests for the signal/aggregator defects fixed in the
betterment pass (see PROJECT_STATUS.md, "Bugs found and fixed").

Each test pins one specific defect. They are deliberately written against
the *symptom* rather than the implementation, so that a future refactor
that reintroduces the bug fails here even if the code looks different.
"""
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import spearmanr

from src.aggregators.a1_stacking import LogRegStackingAggregator
from src.aggregators.a2_coverage_loss import AdaptiveGatingAggregator
from src.aggregators.losses import pairwise_ranking_loss
from src.models.logreg_model import LogRegWrapper
from src.signals.tier_a import (
    CalibrationResidualSignal,
    EnergySignal,
    LogitNormMSPSignal,
    MSPSignal,
    default_tier_a_bank,
)
from src.signals.tier_c import (
    LOCAL_LABEL_AGREEMENT_K,
    TRUST_SCORE_CLAMP,
    LocalLabelAgreementSignal,
    TrustScoreSignal,
)


def _toy_dataset(n=400, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({"x1": rng.randn(n), "x2": rng.randn(n)})
    y = (X["x1"] + 0.5 * X["x2"] + 0.3 * rng.randn(n) > 0).astype(int).values
    return X, y


class _FakeBinaryModel:
    """Emits the `[0, s]` binary logit gauge the real wrappers use, for a
    prescribed sweep of margins `s`."""

    def __init__(self, margins):
        self.margins = np.asarray(margins, dtype=float)

    def logits(self, X=None):
        z = np.zeros((len(self.margins), 2))
        z[:, 1] = self.margins
        return z

    def predict_proba(self, X=None):
        z = self.logits()
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


# --- Bug 1: energy measured class identity, not uncertainty -------------


def test_energy_is_symmetric_in_the_margin_not_monotone_in_the_class():
    """Energy must peak at the decision boundary and fall off symmetrically.

    The pre-fix version returned `-logsumexp([0, s])`, monotone decreasing
    in `s`, so a confident class-0 prediction was ranked *more* uncertain
    than the boundary itself.
    """
    margins = np.array([-8.0, -4.0, -1.0, 0.0, 1.0, 4.0, 8.0])
    e = EnergySignal().score(None, _FakeBinaryModel(margins))

    # maximal uncertainty at the boundary
    assert np.argmax(e) == 3, f"energy should peak at s=0, got {e}"
    # symmetric under s -> -s
    assert np.allclose(e, e[::-1], atol=1e-12), f"energy not symmetric: {e}"
    # and strictly NOT monotone (the pre-fix failure mode)
    diffs = np.diff(e)
    assert not (diffs < 0).all(), "energy is monotone decreasing => class identity bug is back"


def test_energy_ranks_a_confident_prediction_below_the_boundary():
    e = EnergySignal().score(None, _FakeBinaryModel([-8.0, 0.0, 8.0]))
    assert e[0] < e[1] and e[2] < e[1], (
        "confident predictions of either class must be scored less uncertain "
        f"than the decision boundary; got {e}"
    )


# --- Bug 2: logit-norm MSP is degenerate for K=2 ------------------------


def test_logitnorm_msp_refuses_binary_input():
    with pytest.raises(ValueError, match="degenerate for K=2"):
        LogitNormMSPSignal().score(None, _FakeBinaryModel([-3.0, 0.0, 3.0]))


def test_logitnorm_msp_excluded_from_binary_bank_but_kept_for_multiclass():
    binary = [s.name for s in default_tier_a_bank(n_classes=2)]
    multi = [s.name for s in default_tier_a_bank(n_classes=3)]
    assert "logitnorm_msp" not in binary
    assert "logitnorm_msp" in multi


# --- The new non-degenerate signal --------------------------------------


def test_calibration_residual_is_not_rank_equivalent_to_msp():
    """Every other binary Tier-A signal is a monotone transform of MSP
    (|rho| == 1). This one must not be, or it adds nothing."""
    X, y = _toy_dataset(n=600)
    n_fit = 300
    model = LogRegWrapper(seed=0).fit(X.iloc[:n_fit], y[:n_fit])
    X_held, y_held = X.iloc[n_fit:], y[n_fit:]

    sig = CalibrationResidualSignal().fit(X_held, y_held, model)
    residual = sig.score(X_held, model)
    msp = MSPSignal().score(X_held, model)

    rho = abs(spearmanr(msp, residual).statistic)
    assert rho < 0.99, f"calib_residual is rank-equivalent to MSP (|rho|={rho:.4f})"
    assert np.isfinite(residual).all()
    assert (residual >= 0).all()


def test_calibration_residual_requires_fit_first():
    with pytest.raises(RuntimeError, match="before .fit"):
        CalibrationResidualSignal().score(None, _FakeBinaryModel([0.0]))


def test_every_other_binary_tier_a_signal_is_still_rank_equivalent_to_msp():
    """Documents the Tier-A degeneracy this pass measured, so that the
    claim in the paper stays tied to a test rather than to a memory."""
    X, y = _toy_dataset(n=500)
    model = LogRegWrapper(seed=0).fit(X, y)
    msp = MSPSignal().score(X, model)
    for sig in default_tier_a_bank(n_classes=2):
        if sig.name in ("msp", "temp_msp"):
            continue
        s = sig.score(X, model)
        rho = abs(spearmanr(msp, s).statistic)
        assert rho > 0.999, f"{sig.name} unexpectedly not monotone in MSP (|rho|={rho:.4f})"


# --- Bug 9 / 10: Tier C robustness --------------------------------------


def test_trust_score_is_clamped_against_zero_distance_blowup():
    """A test point sitting exactly on a training point drives d_pred -> 0;
    the raw ratio then reached ~1e12 and destroyed the aggregators' z-score
    standardisation."""
    X, y = _toy_dataset(n=200)
    model = LogRegWrapper(seed=0).fit(X, y)
    sig = TrustScoreSignal().fit(X, y, model)
    scores = sig.score(X, model)  # scoring the training rows => d_pred == 0
    assert np.isfinite(scores).all()
    assert np.abs(scores).max() <= TRUST_SCORE_CLAMP + 1e-6, (
        f"trust score exceeded the clamp: max |s| = {np.abs(scores).max():.3e}"
    )


def test_local_label_agreement_uses_a_finer_grid_than_the_shared_default():
    X, y = _toy_dataset(n=400)
    model = LogRegWrapper(seed=0).fit(X, y)
    sig = LocalLabelAgreementSignal().fit(X, y, model)
    assert sig.k == LOCAL_LABEL_AGREEMENT_K > 10
    scores = sig.score(X, model)
    # k+1 levels are *available*; assert we beat the old 11-level ceiling.
    assert len(np.unique(scores)) > 11


def test_train_indexed_signal_clamps_k_to_the_training_size():
    """A cross-fitting fold smaller than k must not raise."""
    X, y = _toy_dataset(n=20)
    model = LogRegWrapper(seed=0).fit(X, y)
    sig = LocalLabelAgreementSignal().fit(X, y, model)
    assert sig.k <= 20
    assert np.isfinite(sig.score(X, model)).all()


# --- Bug 4: ranking loss margins were squashed --------------------------


def test_pairwise_ranking_loss_approaches_zero_for_a_perfect_ranking():
    """On sigmoid inputs the loss floored at ~0.313 even for a perfectly
    separated ranking; on logits it must approach 0."""
    incorrect = torch.tensor([0.0] * 64 + [1.0] * 64)
    logits = torch.cat([torch.full((64,), -12.0), torch.full((64,), 12.0)])
    loss = pairwise_ranking_loss(logits, incorrect, n_pairs=512).item()
    assert loss < 1e-6, f"perfect ranking should cost ~0, got {loss:.4f}"


def test_pairwise_ranking_loss_penalises_an_inverted_ranking():
    incorrect = torch.tensor([0.0] * 64 + [1.0] * 64)
    inverted = torch.cat([torch.full((64,), 12.0), torch.full((64,), -12.0)])
    assert pairwise_ranking_loss(inverted, incorrect, n_pairs=512).item() > 10.0


def test_pairwise_ranking_loss_handles_a_degenerate_batch():
    all_correct = torch.zeros(32)
    logits = torch.randn(32)
    assert pairwise_ranking_loss(logits, all_correct, n_pairs=64).item() == 0.0


# --- Bug 3: adaptive gating had no intercept ----------------------------


def test_adaptive_gating_can_represent_a_low_base_error_rate():
    """With a convex-combination score over zero-mean inputs and no bias,
    the mean logit was pinned near 0 (=> P(error) ~ 0.5). With a 10% error
    rate the aggregator must be able to sit near 0.1, not 0.5."""
    rng = np.random.RandomState(0)
    n, m = 500, 6
    U = rng.randn(n, m)
    correct = np.ones(n, dtype=int)
    correct[: int(0.1 * n)] = 0  # 10% incorrect

    agg = AdaptiveGatingAggregator(loss="bce", epochs=60, seed=0).fit(U, correct)
    assert agg.bias is not None, "adaptive gating still has no bias parameter"
    mean_score = float(np.mean(agg.score(U)))
    assert mean_score < 0.35, (
        f"mean predicted risk {mean_score:.3f} is stuck near 0.5 => the "
        "missing-bias bug is back"
    )


# --- Bug 7: A1 logreg was scale-sensitive -------------------------------


def test_a1_logreg_is_invariant_to_feature_rescaling():
    """Standardisation inside the pipeline must make the fit insensitive to
    the units a signal happens to be reported in."""
    rng = np.random.RandomState(0)
    n = 400
    informative = rng.randn(n)
    correct = (informative + 0.3 * rng.randn(n) > 0).astype(int)
    U = np.column_stack([informative, rng.randn(n)])

    base = LogRegStackingAggregator(seed=0).fit(U, correct).score(U)
    # blow up the *uninformative* column by 1000x
    U_scaled = U.copy()
    U_scaled[:, 1] *= 1000.0
    scaled = LogRegStackingAggregator(seed=0).fit(U_scaled, correct).score(U_scaled)

    rho = spearmanr(base, scaled).statistic
    assert rho > 0.99, (
        f"A1 logreg ranking changed (rho={rho:.4f}) when an uninformative "
        "column was rescaled => feature standardisation is missing"
    )


# --- Bugs 5+6 interaction: an unreachable coverage target ---------------


def test_soft_gate_can_reach_the_target_coverage():
    """The gate must be able to attain the requested coverage.

    Applying a soft gate to the *squashed* score confines both `s` and
    `tau` to [0, 1], which caps the reachable `mean(g)` at ~0.717 for
    T=0.5 -- below the default target of 0.8. The coverage penalty is then
    permanently active, and at lambda=10 it dominates the risk term; the
    cheapest way to raise `mean(g)` is to collapse the score to a constant,
    which destroys the ranking (it drove A2_mlp_loss1 on Adult to AURC
    0.1957, worse than random's 0.1284). Gating on the raw logit removes
    the cap. This test pins reachability, not any particular formula.
    """
    from src.aggregators.losses import DEFAULT_GATE_T

    logit = torch.linspace(-8.0, 8.0, 1000)
    best = max(
        torch.sigmoid((tau - logit) / DEFAULT_GATE_T).mean().item()
        for tau in torch.linspace(-8.0, 8.0, 400)
    )
    assert best > 0.9, (
        f"max reachable coverage is only {best:.4f}; the gate cannot attain "
        "a high target, so the coverage penalty is unsatisfiable"
    )


def test_coverage_targeted_losses_do_not_collapse_the_ranking():
    """A2 trained with loss1/loss2 must still rank errors above correct
    predictions -- i.e. beat a coin flip on a cleanly separable problem."""
    from sklearn.metrics import roc_auc_score

    from src.aggregators.a2_coverage_loss import MLPAggregator

    rng = np.random.RandomState(0)
    n = 600
    signal = rng.randn(n)
    incorrect = (signal > 1.0).astype(int)  # ~16% error rate, separable
    correct = 1 - incorrect
    U = np.column_stack([signal, rng.randn(n)])

    for loss in ("loss1", "loss2"):
        agg = MLPAggregator(loss=loss, epochs=200, seed=0).fit(U, correct)
        scores = agg.score(U)
        auc = roc_auc_score(incorrect, scores)
        assert auc > 0.8, (
            f"{loss} produced a near-useless ranking (AUC={auc:.3f}); the "
            "score has probably collapsed toward a constant"
        )
        assert np.std(scores) > 1e-3, f"{loss} scores are constant"


# --- Bug 8: loss1/loss2's freely learned tau could still miss the target
# coverage (fixed by deriving tau as a quantile instead) --------------------


def test_quantile_tau_hits_target_coverage_by_construction():
    """`quantile_tau` must put exactly `kappa` fraction of a batch's logits
    below the returned threshold, for any score distribution -- that is
    the entire point of deriving it rather than learning it. This is what
    replaces the coverage *penalty*, which could only ever nudge a freely
    learned tau toward the target, not guarantee it."""
    from src.aggregators.losses import quantile_tau

    rng = np.random.RandomState(0)
    logits = torch.tensor(rng.randn(2000).astype(np.float32))
    for kappa in (0.5, 0.7, 0.8, 0.95):
        tau = quantile_tau(logits, kappa)
        hard_coverage = (logits <= tau).float().mean().item()
        assert abs(hard_coverage - kappa) < 0.01, (
            f"kappa={kappa}: hard-thresholded coverage was {hard_coverage:.4f}, "
            "should equal kappa by construction"
        )


def test_coverage_targeted_losses_do_not_learn_a_tau_parameter():
    """`MLPAggregator` must not carry a learnable tau/taus after this fix --
    the threshold is derived from the batch each step, not optimised. A
    regression here would mean tau silently became a free parameter again,
    reintroducing the instability quantile_tau was written to remove."""
    from src.aggregators.a2_coverage_loss import MLPAggregator

    rng = np.random.RandomState(0)
    n = 300
    U = np.column_stack([rng.randn(n), rng.randn(n)])
    correct = (rng.rand(n) > 0.15).astype(int)

    for loss in ("loss1", "loss2"):
        agg = MLPAggregator(loss=loss, epochs=20, seed=0).fit(U, correct)
        assert not hasattr(agg, "tau") or agg.tau is None
        assert not hasattr(agg, "taus") or agg.taus is None


def test_electricity_like_shift_does_not_produce_worse_than_random_loss1():
    """Regression for the measured symptom this fix targets: on Electricity,
    `A2_mlp_loss1` had AURC 0.3455 (worst seed 0.4606) against random's
    0.3705 -- i.e. a *learned* tau occasionally did worse than not learning
    anything at all. Simulate a similarly noisy, imbalanced signal bank and
    confirm loss1 still produces a usable (better than chance) ranking
    across several seeds, not just one favorable one."""
    from sklearn.metrics import roc_auc_score

    from src.aggregators.a2_coverage_loss import MLPAggregator

    for seed in range(5):
        rng = np.random.RandomState(seed)
        n = 500
        signal = rng.randn(n)
        noise_signal = rng.randn(n) * 3  # a noisy, uninformative second signal
        incorrect = (signal > 0.8).astype(int)  # ~21% error rate
        correct = 1 - incorrect
        U = np.column_stack([signal, noise_signal])

        agg = MLPAggregator(loss="loss1", epochs=150, seed=seed).fit(U, correct)
        scores = agg.score(U)
        auc = roc_auc_score(incorrect, scores)
        assert auc > 0.7, f"seed={seed}: loss1 AUC={auc:.3f}, worse than expected"
