"""The coverage-targeted losses from §3.3 — the thing that is supposed to
make A2 more than stacking (§11: "this is just stacking" is the expected
review objection; the A1-vs-A2 ablation on these losses is the direct
rebuttal, so it must be run early per §3.3/§9 week-5 gate).

All losses take:
    s          : (n,) tensor, the aggregator's continuous risk score
    incorrect  : (n,) tensor of {0,1}, 1 if the frozen base model was wrong

and are expressed directly in terms of the formulas in §3.3, with a
learnable threshold `tau` standing in for the operating point rather than a
fixed constant, so gradient descent can find the threshold that realizes
the target coverage kappa instead of it being a hand-picked constant.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# Gate temperature for the soft accept/reject decision, in **logit units**.
#
# **Bug fix, in two stages -- read both, because the obvious first fix is
# actively harmful on its own.**
#
# Stage 1, the original defect: T was 0.05 and the gate was applied to the
# squashed score `s = sigmoid(logit)`. That makes `sigmoid((tau - s)/T)` a
# near-step function whose derivative decays as `sigmoid'(20*(tau - s))`,
# below 0.018 once |tau - s| > 0.2. Composed with the already-saturating
# sigmoid producing `s`, the two sigmoids form a double-saturation that
# leaves almost every training row with no usable gradient.
#
# Stage 2, the trap: simply widening T on the *bounded* score makes things
# far worse. With `s` confined to [0, 1] and `tau` to (0, 1), a soft gate
# cannot reach a high coverage at all -- measured, the maximum achievable
# `mean(g)` over all tau is only **0.7170 at T=0.5**, against a target
# kappa of 0.80. The coverage penalty is then permanently unsatisfiable,
# and (especially at the stronger lambda below) it dominates the loss. The
# cheapest way for the optimiser to raise `mean(g)` is to collapse `s`
# toward a constant, which destroys the ranking outright: this drove
# `A2_mlp_loss1` on Adult from AURC 0.0317 to 0.1957, i.e. *worse than
# random abstention* (0.1284), and `A2_mlp_loss2` from 0.0303 to 0.2127.
#
# The actual fix is to gate on the **raw logit**, which is unbounded, so
# any coverage is reachable by moving tau (measured max `mean(g)` = 0.978
# at T=0.5) while the gradient stays healthy across the range the logit
# really occupies. This also removes the double-saturation at its source
# rather than papering over it, since there is now only one sigmoid between
# the network and the gate. `tau` is correspondingly a free parameter in
# logit units, not a probability.
DEFAULT_GATE_T = 0.5

# Weight on the coverage constraint.
#
# **Bug fix (constraint too weak to bind).** This was 1.0, which left the
# penalty numerically irrelevant: missing a target coverage of 0.8 by a full
# 10 points costs `(0.1)^2 * 1.0 = 0.01`, against a selective risk term of
# order 0.1-0.2. The optimiser could therefore ignore the requested coverage
# almost entirely and still report a good loss, which defeats the point of a
# *coverage-targeted* objective (§3.3). At 10.0 the same 10-point miss costs
# 0.1, i.e. comparable to the risk term, so the constraint actually binds.
#
# This is only safe because the gate above now operates on the unbounded
# logit, which makes the target coverage *reachable*. Raising lambda while
# the target was unreachable is what caused the collapse documented under
# DEFAULT_GATE_T -- a strong penalty on an unsatisfiable constraint buys a
# degenerate solution, not a better one. Do not raise lambda without first
# confirming that `mean(g) = kappa` is attainable.
DEFAULT_COVERAGE_LAMBDA = 10.0


def quantile_tau(s_logit: torch.Tensor, kappa: float) -> torch.Tensor:
    """The coverage-exact threshold: the `kappa`-quantile of the current
    batch's risk logits, detached from the autograd graph. The gate accepts
    when `s_logit < tau` (see `soft_selective_risk_loss`), so putting `tau`
    at the `kappa`-quantile means exactly a `kappa` fraction of the batch
    has `s_logit` below it, by the definition of a quantile.

    **Bug fix (loss1/loss2 remained unreliable even after the gate fix,
    §6/PROJECT_STATUS.md).** The gate-on-raw-logit fix made the target
    coverage *reachable*, but `tau` was still a freely learned
    `nn.Parameter`, trained jointly with the network by gradient descent
    against `soft_selective_risk_loss`'s quadratic coverage penalty. That
    is an indirect, iterative way to hit a target that has a direct,
    closed-form answer: for a fixed batch of scores, the threshold that
    puts exactly `kappa` coverage below it *is* that quantile. Measured
    symptom this replaces: `A2_mlp_loss1` on Electricity had AURC 0.3455
    with std 0.0951 and a worst seed of 0.4606 -- worse than random
    abstention (0.3705) -- because a learned `tau` can wander (via the
    optimiser's own trajectory, the lr schedule, or a bad seed) into a
    region where the coverage penalty and the risk term trade off badly,
    especially on small meta-sets (e.g. German Credit's ~150-row D_meta)
    where a handful of points can shift a learned threshold a lot.

    Using the empirical quantile instead makes `mean(hard-thresholded g)
    == kappa` true *by construction* every single step, for any network
    weights -- there is nothing left for a coverage penalty to enforce,
    which is why `soft_selective_risk_loss` below no longer takes `tau` as
    an input at all. The remaining softness (the sigmoid gate width `T`)
    only blurs coverage by a small, symmetric amount around the exact
    quantile cut -- it does not introduce systematic drift the way a
    freely learned threshold could.

    Wrapped in `no_grad` deliberately: `tau` is recomputed fresh from
    whatever the network currently outputs, not optimised itself, so no
    gradient should flow through the quantile operation back into the
    network via this path (gradients still flow into the network normally
    through `s_logit`'s use in the gate and the risk numerator/denominator
    in `soft_selective_risk_loss`).
    """
    with torch.no_grad():
        return torch.quantile(s_logit, kappa)


def soft_selective_risk_loss(
    s_logit: torch.Tensor,
    incorrect: torch.Tensor,
    kappa: float,
    T: float = DEFAULT_GATE_T,
    lam: float = DEFAULT_COVERAGE_LAMBDA,
) -> torch.Tensor:
    """Loss 1 (§3.3): soft selective risk at target coverage kappa.

    g(x) = sigmoid((tau - z(x)) / T)   -- soft "accept" gate (accept when
                                           the risk logit z(x) is
                                           comfortably below tau)
    L = sum g*err / sum g  +  lam * relu(kappa - mean(g))^2

    `s_logit` is the **raw risk logit**, not a probability. `tau` is no
    longer a caller-supplied free parameter -- see `quantile_tau` -- it is
    derived here from `s_logit` itself, so a `kappa`-coverage threshold
    always exists and the coverage penalty term is now a redundant
    backstop (it stays near zero by construction) rather than a live
    constraint fighting the risk term for control of the gradient.
    """
    tau = quantile_tau(s_logit, kappa)
    g = torch.sigmoid((tau - s_logit) / T)
    num = (g * incorrect).sum()
    den = g.sum().clamp_min(1e-6)
    selective_risk = num / den
    coverage_penalty = F.relu(kappa - g.mean()) ** 2
    return selective_risk + lam * coverage_penalty


def aurc_surrogate_loss(
    s_logit: torch.Tensor,
    incorrect: torch.Tensor,
    kappa_grid: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    T: float = DEFAULT_GATE_T,
    lam: float = DEFAULT_COVERAGE_LAMBDA,
) -> torch.Tensor:
    """Loss 2 (§3.3): sum Loss-1 over a grid of target coverages, so a
    single score is trained to be good across the whole risk-coverage
    curve rather than at one operating point. Each `kappa` in the grid
    gets its own quantile-derived threshold (see `soft_selective_risk_loss`
    / `quantile_tau`) computed fresh from `s_logit` every call -- there is
    no longer a persisted per-kappa `taus` parameter to manage."""
    total = s_logit.new_zeros(())
    for kappa in kappa_grid:
        total = total + soft_selective_risk_loss(s_logit, incorrect, kappa, T=T, lam=lam)
    return total / len(kappa_grid)


def pairwise_ranking_loss(
    s_logit: torch.Tensor, incorrect: torch.Tensor, n_pairs: int = 2048, generator=None
) -> torch.Tensor:
    """Loss 3 (§3.3): for (correct, incorrect) pairs, penalise a logistic
    pairwise loss on the score margin — directly optimises the ranking that
    AURC measures, and per §3.3 is "better behaved than Loss 1 on small
    tabular datasets".

    **Bug fix (margin was computed on squashed scores).** The caller used to
    pass `s = sigmoid(logit)`, so the margin `s[incorrect] - s[correct]` was
    confined to (-1, 1). That capped how well the objective could ever be
    satisfied -- `softplus(-margin)` cannot fall below `softplus(-1) = 0.313`
    even for a *perfectly* separated ranking -- so a converged model still
    reported a large loss, and worse, the gradient through the sigmoid
    vanished exactly where the ranking was becoming confident. The parameter
    is now the raw **logit**, giving unbounded margins, a loss that tends to
    0 for a correct ranking, and healthy gradients throughout.

    Note that only the *scale* of the margin changes, not the ranking being
    optimised: sigmoid is strictly monotone, so the sign of every pairwise
    comparison is identical either way.
    """
    correct_idx = torch.nonzero(incorrect == 0, as_tuple=True)[0]
    incorrect_idx = torch.nonzero(incorrect == 1, as_tuple=True)[0]
    if len(correct_idx) == 0 or len(incorrect_idx) == 0:
        # Degenerate batch (e.g. a tiny meta-set or a perfect/failing base
        # model) -- no valid pairs, contribute zero loss rather than NaN.
        return s_logit.new_zeros(())
    kwargs = {"generator": generator} if generator is not None else {}
    ci = correct_idx[torch.randint(0, len(correct_idx), (n_pairs,), **kwargs)]
    ii = incorrect_idx[torch.randint(0, len(incorrect_idx), (n_pairs,), **kwargs)]
    margin = s_logit[ii] - s_logit[ci]  # want > 0 (incorrect = higher risk)
    return F.softplus(-margin).mean()


def bce_loss(s_logit: torch.Tensor, incorrect: torch.Tensor) -> torch.Tensor:
    """Plain correctness-classification loss -- what A1 optimises. Kept
    here so A2 can be run in "BCE mode" for the direct A1-vs-A2 ablation
    on loss choice alone, holding the function class fixed (§3.3, §7
    ablation 4)."""
    return F.binary_cross_entropy_with_logits(s_logit, incorrect)


def expected_bandit_reward_loss(
    s_logit: torch.Tensor, incorrect: torch.Tensor, penalty_c: float = 10.0
) -> torch.Tensor:
    """RL Contextual Bandit Policy Gradient (Expected Reward).
    
    The network outputs `s_logit`, representing the log-odds of abstention.
    Probability of predicting (Action = 1): pi_predict = sigmoid(-s_logit).
    Probability of abstaining (Action = 0): pi_abstain = sigmoid(s_logit).
    
    Reward R:
      - Predict & Correct: +1
      - Predict & Incorrect: -penalty_c
      - Abstain: 0
      
    Since the action space is discrete {0,1}, we can compute the exact Expected Reward
    analytically without sampling variance: E[R|x] = pi_predict * (1 * correct - c * incorrect).
    To optimize with gradient descent, we minimize negative expected reward.
    """
    pi_predict = torch.sigmoid(-s_logit)
    correct = 1.0 - incorrect
    
    # Expected reward for taking action a=1 (predict)
    expected_reward = pi_predict * (correct - penalty_c * incorrect)
    
    # We want to maximize expected_reward, so we minimize its negation.
    return -expected_reward.mean()
