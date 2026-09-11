"""A2 — the coverage-targeted aggregator (§3.2-§3.3), the paper's headline
contribution, plus the instance-adaptive signal-weighting variant (§3.4,
"second novelty axis").

Both classes share the same function-class-vs-A1 comparison: same input
u(x), same general capacity (a small MLP), different training loss. That is
deliberate — §3.3 says the A1-vs-A2 ablation must hold the function class
fixed and vary only the loss, or the comparison proves nothing.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .losses import (
    aurc_surrogate_loss,
    bce_loss,
    pairwise_ranking_loss,
    soft_selective_risk_loss,
    expected_bandit_reward_loss,
)

class _Standardizer:
    def fit(self, U: np.ndarray) -> "_Standardizer":
        self.mean_ = U.mean(axis=0)
        self.std_ = U.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, U: np.ndarray) -> np.ndarray:
        return (U - self.mean_) / self.std_


def _make_mlp(
    m_in: int, hidden: int, out: int, depth: int = 2, dropout: float = 0.0
) -> nn.Sequential:
    """Small MLP. `dropout` (applied after each hidden activation) is a
    meta-overfitting guard: the aggregator trains on D_meta, which is only
    15% of the dataset -- 150 rows on German Credit -- while carrying far
    more capacity than that supports."""
    layers: list[nn.Module] = []
    d = m_in
    for _ in range(depth - 1):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        if dropout > 0:
            layers += [nn.Dropout(dropout)]
        d = hidden
    layers += [nn.Linear(d, out)]
    return nn.Sequential(*layers)


class _BaseTorchAggregator:
    """Shared training loop for MLPAggregator and AdaptiveGatingAggregator.
    Subclasses implement `_build_net` and `_score_logit`."""

    KAPPA_GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

    def __init__(
        self,
        loss: str = "loss1",
        target_coverage: float = 0.8,
        hidden: int = 32,
        depth: int = 2,
        epochs: int = 500,
        lr: float = 1e-2,
        lr_min: float = 1e-5,
        weight_decay: float = 1e-4,
        dropout: float = 0.1,
        grad_clip: float = 1.0,
        seed: int = 0,
    ):
        assert loss in ("bce", "loss1", "loss2", "loss3", "rl_bandit")
        self.loss_name = loss
        self.target_coverage = target_coverage
        self.hidden = hidden
        self.depth = depth
        self.epochs = epochs
        self.lr = lr
        self.lr_min = lr_min
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.grad_clip = grad_clip
        self.seed = seed
        self.scaler = _Standardizer()
        self.net: nn.Module | None = None
        self.bias: nn.Parameter | None = None  # adaptive-gating output bias
        # loss1/loss2's threshold tau used to live here as a learned
        # nn.Parameter. It is now derived fresh each step from the current
        # batch's logits (see losses.quantile_tau) -- a quantile-based
        # threshold hits the target coverage by construction, instead of
        # being trained to approach it via a penalty term that a freely
        # learned tau could still miss. See losses.py's docstring for the
        # bug this fixes (loss1/loss2 remained unreliable even after the
        # earlier gate-on-raw-logit fix).

    def _build_net(self, m_in: int) -> nn.Module:
        raise NotImplementedError

    def _score_logit(self, net_out: torch.Tensor, U_t: torch.Tensor) -> torch.Tensor:
        """Map the network's raw output (and, for the adaptive-gating
        subclass, the standardized input it was computed from) to a scalar
        risk logit per row."""
        raise NotImplementedError

    def _extra_params(self, base_error_rate: float) -> list[nn.Parameter]:
        """Parameters a subclass owns outside `self.net`, given the observed
        fraction of incorrect predictions on the meta set. Default: none."""
        return []

    def fit(self, U_meta: np.ndarray, correct_meta: np.ndarray) -> "_BaseTorchAggregator":
        torch.manual_seed(self.seed)
        gen = torch.Generator().manual_seed(self.seed)

        Un = self.scaler.fit(U_meta).transform(U_meta)
        U_t = torch.tensor(Un, dtype=torch.float32)
        incorrect = torch.tensor(1 - correct_meta.astype(int), dtype=torch.float32)

        self.net = self._build_net(Un.shape[1])
        params = list(self.net.parameters())
        # Subclass-owned extra parameters, initialised from the meta-set
        # base error rate (see AdaptiveGatingAggregator._extra_params).
        params += self._extra_params(float(incorrect.mean()))

        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        # Cosine annealing from `lr` down to `lr_min`. Training was
        # previously a fixed lr=1e-2 for 300 full-batch steps with no
        # schedule and no stopping rule, which is large enough to keep
        # bouncing around a minimum rather than settling into one.
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.epochs, eta_min=self.lr_min
        )

        self.net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            net_out = self.net(U_t)
            logit = self._score_logit(net_out, U_t)

            if self.loss_name == "bce":
                loss = bce_loss(logit, incorrect)
            elif self.loss_name == "rl_bandit":
                loss = expected_bandit_reward_loss(logit, incorrect, penalty_c=10.0)
            elif self.loss_name == "loss1":
                # Losses 1/2 gate on the RAW logit, with tau a quantile of
                # the current batch's logits (see losses.quantile_tau) --
                # not a freely learned parameter. Gating on sigmoid(logit)
                # with tau squashed into (0, 1) caps the reachable coverage
                # at ~0.72, which makes the target unsatisfiable and
                # collapses the score to a constant -- see DEFAULT_GATE_T
                # in losses.py.
                loss = soft_selective_risk_loss(logit, incorrect, kappa=self.target_coverage)
            elif self.loss_name == "loss2":
                loss = aurc_surrogate_loss(logit, incorrect, kappa_grid=self.KAPPA_GRID)
            else:  # loss3
                # Loss 3 is a *ranking* loss and takes the raw logit, so its
                # margins are unbounded -- see `pairwise_ranking_loss`.
                loss = pairwise_ranking_loss(logit, incorrect, generator=gen)

            loss.backward()
            # Loss 1 divides by `sum(g)`, which can get small when the gate
            # closes, so its gradient occasionally spikes; clip before the
            # step rather than letting one batch throw the weights.
            torch.nn.utils.clip_grad_norm_(params, max_norm=self.grad_clip)
            opt.step()
            sched.step()

        self.net.eval()
        return self

    def score(self, U: np.ndarray) -> np.ndarray:
        Un = self.scaler.transform(U)
        U_t = torch.tensor(Un, dtype=torch.float32)
        self.net.eval()  # dropout must be off at scoring time
        with torch.no_grad():
            logit = self._score_logit(self.net(U_t), U_t)
            s = torch.sigmoid(logit)
        return s.numpy()


class MLPAggregator(_BaseTorchAggregator):
    """A2 core: a small MLP over u(x) -> scalar risk score, trained with
    one of the losses in §3.3."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = f"A2_mlp_{self.loss_name}"

    def _build_net(self, m_in: int) -> nn.Module:
        return _make_mlp(m_in, self.hidden, out=1, depth=self.depth, dropout=self.dropout)

    def _score_logit(self, net_out: torch.Tensor, U_t: torch.Tensor) -> torch.Tensor:
        return net_out.squeeze(-1)


class AdaptiveGatingAggregator(_BaseTorchAggregator):
    """§3.4: instance-adaptive signal weighting. A gating network
    w(x) = softmax(h(u(x))) over the m signals, and
    s(x) = sum_j w_j(x) * z_j(x) where z_j is the per-signal z-score.
    Exposes `weights(U)` for the interpretability figure in §3.4 (plotting
    the learned weight distribution for in-distribution vs corrupted
    inputs)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = f"A2_adaptive_{self.loss_name}"

    def _build_net(self, m_in: int) -> nn.Module:
        return _make_mlp(m_in, self.hidden, out=m_in, depth=self.depth, dropout=self.dropout)

    def _extra_params(self, base_error_rate: float) -> list[nn.Parameter]:
        """Create the output bias.

        **Bug fix (the gated score could not express a base rate).** The
        score was `logit(x) = w(x)^T z(x)` with no intercept. `w` is a
        softmax, so it sums to 1, and `z` is the z-scored signal vector, so
        it has zero mean per column -- which pins the *average* logit at
        approximately 0, i.e. a predicted error probability of
        `sigmoid(0) = 0.5`. When the base model is right 90% of the time the
        correct average logit is `log(0.1/0.9) ~= -2.2`, and the
        architecture simply had no parameter capable of representing that.
        The convex-combination constraint also caps the logit's range at
        `max_j z_j(x)`, so it could not be compensated for by scaling.

        Initialising the bias at the meta-set log-odds starts training at
        the right base rate, leaving the network to learn only the
        *deviation* from it -- which is all a gating network should be doing.
        """
        r = float(np.clip(base_error_rate, 1e-4, 1 - 1e-4))
        self.bias = nn.Parameter(torch.tensor(np.log(r / (1.0 - r)), dtype=torch.float32))
        return [self.bias]

    def _score_logit(self, net_out: torch.Tensor, U_t: torch.Tensor) -> torch.Tensor:
        # net_out: (n, m) gating logits over the m standardized signals in
        # U_t; s(x) = sum_j w_j(x) * z_j(x) + bias, per §3.4 (the bias is
        # the fix described in `_extra_params`).
        w = torch.softmax(net_out, dim=1)
        return (w * U_t).sum(dim=1) + self.bias

    def weights(self, U: np.ndarray) -> np.ndarray:
        """Learned per-signal gating weights w(x) for interpretability
        (§3.4's headline figure)."""
        Un = self.scaler.transform(U)
        U_t = torch.tensor(Un, dtype=torch.float32)
        with torch.no_grad():
            logits = self.net(U_t)
            w = torch.softmax(logits, dim=1)
        return w.numpy()


class RLBanditAggregator(MLPAggregator):
    """Contextual Bandit Policy network for Selective Prediction.
    Outputs the log-odds of abstention, trained directly on the expected reward
    of taking a hard discrete action (Abstain or Predict)."""
    
    def __init__(self, **kwargs):
        kwargs["loss"] = "rl_bandit"
        super().__init__(**kwargs)
        self.name = "A2_rl_bandit"
