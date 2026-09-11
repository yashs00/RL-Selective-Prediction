"""Tests for per-instance score caching and the paired bootstrap built on
it (§7's prescribed statistical test; PROJECT_STATUS.md's "no raw
per-instance caching" simplification, `full_research_analysis.md` R7).

The bootstrap's whole advantage over the per-seed Wilcoxon substitute is
that it is *paired over test instances*, so these tests pin the properties
that pairing is supposed to buy -- not merely that the code runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_bootstrap_table import bootstrap_aurc_deltas, summarize_deltas  # noqa: E402

from src.experiment.runner import _save_raw_scores, load_raw_scores, raw_scores_path


def test_raw_scores_round_trip(tmp_path):
    methods = ["signal_msp", "A1_logreg"]
    scores = np.array([[0.1, 0.9, 0.4], [0.2, 0.8, 0.3]], dtype=np.float32)
    incorrect = np.array([0, 1, 0], dtype=np.int8)

    _save_raw_scores(
        str(tmp_path), "toyds", "lightgbm", "AC", 3,
        methods=methods, scores=scores, incorrect=incorrect,
    )
    got = load_raw_scores(raw_scores_path(str(tmp_path), "toyds", "lightgbm", "AC", 3))

    assert got["methods"] == methods
    assert got["dataset"] == "toyds" and got["tiers"] == "AC" and got["seed"] == 3
    np.testing.assert_allclose(got["scores"], scores)
    np.testing.assert_array_equal(got["incorrect"], incorrect)


def test_raw_scores_path_is_keyed_so_a_rerun_overwrites(tmp_path):
    """The filename must carry the full results key, so re-running one
    (dataset, model, tiers, seed) replaces its own file instead of
    accumulating a duplicate that would double-count in the bootstrap."""
    a = raw_scores_path(str(tmp_path), "ds", "lightgbm", "AC", 0)
    b = raw_scores_path(str(tmp_path), "ds", "lightgbm", "AC", 0)
    c = raw_scores_path(str(tmp_path), "ds", "lightgbm", "ABC", 0)
    d = raw_scores_path(str(tmp_path), "ds", "lightgbm", "AC", 1)
    assert a == b
    assert len({a, c, d}) == 3


def test_bootstrap_delta_is_zero_for_a_rank_equivalent_method():
    """Two scores that induce the same *ordering* must give an identically
    zero AURC delta on every resample -- AURC depends only on the ranking.
    This is the property that makes the binary Tier-A degeneracy show up in
    the bootstrap as a zero-width interval rather than as noise."""
    rng = np.random.default_rng(0)
    n = 400
    base = rng.random(n)
    monotone_copy = 3.0 * base + 1.0  # strictly increasing transform
    incorrect = (rng.random(n) < base * 0.4).astype(int)

    deltas = bootstrap_aurc_deltas(
        monotone_copy.astype(np.float32), base.astype(np.float32), incorrect,
        n_boot=50, rng=rng,
    )
    finite = deltas[np.isfinite(deltas)]
    assert finite.size > 0
    np.testing.assert_allclose(finite, 0.0, atol=1e-12)


def test_bootstrap_ci_excludes_zero_for_a_genuinely_better_ranking():
    """An oracle ranking must beat a random one with a CI clear of 0 --
    otherwise the test has no power and cannot support any claim."""
    rng = np.random.default_rng(1)
    n = 800
    incorrect = (rng.random(n) < 0.2).astype(int)
    oracle = incorrect.astype(np.float32)          # perfect ordering
    noise = rng.random(n).astype(np.float32)       # uninformative

    deltas = bootstrap_aurc_deltas(oracle, noise, incorrect, n_boot=200, rng=rng)
    finite = deltas[np.isfinite(deltas)]
    hi = np.percentile(finite, 97.5)
    assert hi < 0, f"oracle-vs-random CI upper bound was {hi:.4f}, expected < 0"


def test_bootstrap_is_paired_not_independently_resampled():
    """Pairing is the point: applying the *same* resample to both methods
    must give a tighter spread of deltas than resampling each
    independently. If this ever fails, the pairing has been lost and every
    CI in the table is wider (and weaker) than it should be."""
    rng = np.random.default_rng(2)
    n = 500
    base = rng.random(n).astype(np.float32)
    other = (base + 0.05 * rng.standard_normal(n)).astype(np.float32)  # highly correlated
    incorrect = (rng.random(n) < base * 0.4).astype(int)

    paired = bootstrap_aurc_deltas(other, base, incorrect, n_boot=300,
                                   rng=np.random.default_rng(3))
    paired = paired[np.isfinite(paired)]

    # Unpaired comparison: resample each method's AURC on independent draws.
    from src.metrics.selective import aurc

    rng_u = np.random.default_rng(3)
    unpaired = []
    for _ in range(300):
        i1 = rng_u.integers(0, n, size=n)
        i2 = rng_u.integers(0, n, size=n)
        if incorrect[i1].sum() in (0, n) or incorrect[i2].sum() in (0, n):
            continue
        unpaired.append(aurc(other[i1], incorrect[i1]) - aurc(base[i2], incorrect[i2]))
    unpaired = np.asarray(unpaired)

    assert paired.std() < unpaired.std(), (
        f"paired spread ({paired.std():.5f}) should be tighter than unpaired "
        f"({unpaired.std():.5f}); the resampling may no longer be paired"
    )


# --- the two intervals that decide reported claims ----------------------


def test_seed_averaged_interval_is_tighter_than_the_pooled_one():
    """The pooled interval carries seed-to-seed spread as well as instance
    noise; the seed-averaged one is the interval for the *mean* effect and
    must therefore be narrower. Both are reported so that a claim cannot be
    quietly based on whichever is more favourable."""
    rng = np.random.default_rng(0)
    # Five seeds whose true effects differ, each with its own noise.
    all_deltas = [rng.normal(mu, 0.01, 500) for mu in (-0.004, -0.002, 0.0, 0.002, 0.004)]

    s = summarize_deltas(all_deltas, alpha=0.05)
    pooled_width = s["ci_hi"] - s["ci_lo"]
    mean_width = s["mean_ci_hi"] - s["mean_ci_lo"]
    assert mean_width < pooled_width, (
        f"seed-averaged width {mean_width:.5f} should be < pooled {pooled_width:.5f}"
    )


def test_summarize_flags_a_clear_win_and_not_a_null():
    """A consistently negative effect must be flagged as beating the
    baseline; a centred-on-zero one must not -- in either interval."""
    rng = np.random.default_rng(1)
    win = [rng.normal(-0.01, 0.001, 400) for _ in range(5)]
    null = [rng.normal(0.0, 0.001, 400) for _ in range(5)]

    w = summarize_deltas(win)
    n = summarize_deltas(null)
    assert w["beats_baseline"] and w["mean_effect_beats_baseline"]
    assert not n["beats_baseline"] and not n["mean_effect_beats_baseline"]
    assert not n["excludes_zero"] and not n["mean_effect_excludes_zero"]


def test_summarize_handles_all_nan_input():
    """Degenerate resamples (no errors in the draw) come back as NaN; an
    all-NaN method must be skipped, not reported as a zero-effect win."""
    assert summarize_deltas([np.full(10, np.nan)]) is None


def test_summarize_ignores_nans_when_averaging_across_seeds():
    """One seed's degenerate iteration must not wipe out that iteration's
    average for every other seed."""
    a = np.array([-0.01, -0.01, -0.01])
    b = np.array([np.nan, -0.01, -0.01])
    s = summarize_deltas([a, b])
    assert s is not None
    assert s["mean_effect_beats_baseline"], s
