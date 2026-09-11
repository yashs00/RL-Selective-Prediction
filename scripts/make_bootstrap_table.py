"""Paired bootstrap on AURC differences over test instances (§7's
prescribed statistical test).

Until now the project substituted a per-seed-paired Wilcoxon test, which is
valid but coarse: 10 paired observations per comparison, versus a bootstrap
over the thousands of test points that actually carry the signal. That
substitution was a documented simplification (PROJECT_STATUS.md, "no raw
per-instance caching"; `full_research_analysis.md` R7) purely because the
runner only persisted aggregated summary rows. It now persists per-instance
test scores to `results/raw/` when `experiment.cache_raw_scores=true`, so
the real test can be run.

What this does, per (dataset, tiers, method):

  1. For each seed, resample test *instances* with replacement (the same
     resampled indices applied to both methods -- that is the "paired"
     part, and it is what makes the comparison far tighter than comparing
     two independently-noisy AURC estimates).
  2. Recompute AURC for the method and for the baseline on each resample,
     and take the difference.
  3. Report *two* percentile intervals, because they answer different
     questions and the distinction decides claims:

     * `ci_lo`/`ci_hi` -- **pooled**: all per-seed bootstrap deltas
       concatenated. Its width therefore includes genuine seed-to-seed
       variation in the effect *plus* instance noise, so it answers "would
       this difference be visible within one typical run?". Conservative.
     * `mean_ci_lo`/`mean_ci_hi` -- **seed-averaged**: on each bootstrap
       iteration, average that iteration's delta across seeds, then take
       percentiles of those averages. This is the interval for the *mean*
       effect, which is what a claim like "method X beats MSP on dataset D"
       actually asserts, so it is the apt estimator for the paper.

     Both are reported deliberately: the seed-averaged interval is
     narrower, and quoting only it (having first seen the pooled one) would
     be indistinguishable from fishing. Quoting both makes the basis of any
     claim auditable.

A CI that excludes 0 is evidence of a real AURC difference. Negative =
the method beats the baseline (lower AURC is better).

Usage:
    python scripts/make_bootstrap_table.py
    python scripts/make_bootstrap_table.py --tiers AC --n-boot 2000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment.runner import load_raw_scores  # noqa: E402
from src.metrics.selective import aurc  # noqa: E402

RAW_DIR = ROOT / "results" / "raw"
TABLE_DIR = ROOT / "paper" / "tables"
TABLE_DIR.mkdir(exist_ok=True, parents=True)


def bootstrap_aurc_deltas(
    scores_method: np.ndarray,
    scores_baseline: np.ndarray,
    incorrect: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """`n_boot` paired AURC differences (method - baseline) under
    resampling of test instances. Both methods see the *same* resampled
    indices on every draw."""
    n = len(incorrect)
    deltas = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        inc_b = incorrect[idx]
        if inc_b.sum() == 0 or inc_b.sum() == n:
            # A resample with no errors (or all errors) has a degenerate
            # risk-coverage curve; skip rather than feed a NaN forward.
            deltas[b] = np.nan
            continue
        deltas[b] = aurc(scores_method[idx], inc_b) - aurc(scores_baseline[idx], inc_b)
    return deltas


def summarize_deltas(all_deltas: list[np.ndarray], alpha: float = 0.05) -> dict | None:
    """Turn per-seed bootstrap delta arrays into the two intervals described
    in the module docstring. Returns None if nothing finite survives.

    Extracted from `main` deliberately: this is the computation that
    decides whether a method is reported as beating the baseline, so it is
    unit-tested rather than trusted (see
    `tests/test_raw_cache_and_bootstrap.py`).
    """
    deltas = np.concatenate(all_deltas)
    deltas = deltas[np.isfinite(deltas)]
    if deltas.size == 0:
        return None

    lo = float(np.percentile(deltas, 100 * alpha / 2))
    hi = float(np.percentile(deltas, 100 * (1 - alpha / 2)))

    # Seed-averaged interval: average across seeds *within* each bootstrap
    # iteration, so the percentiles describe the mean effect rather than a
    # single run's. Seeds are aligned by iteration index, which is valid
    # because every seed draws the same number of iterations.
    stack = np.vstack(all_deltas)
    with np.errstate(invalid="ignore"):
        per_iter = np.nanmean(stack, axis=0)
    per_iter = per_iter[np.isfinite(per_iter)]
    if per_iter.size:
        m_lo = float(np.percentile(per_iter, 100 * alpha / 2))
        m_hi = float(np.percentile(per_iter, 100 * (1 - alpha / 2)))
    else:
        m_lo = m_hi = float("nan")

    return {
        "mean_aurc_delta": float(deltas.mean()),
        "ci_lo": lo,
        "ci_hi": hi,
        "excludes_zero": bool(hi < 0 or lo > 0),
        "beats_baseline": bool(hi < 0),
        "mean_ci_lo": m_lo,
        "mean_ci_hi": m_hi,
        "mean_effect_excludes_zero": bool(m_hi < 0 or m_lo > 0),
        "mean_effect_beats_baseline": bool(m_hi < 0),
        "n_seeds": len(all_deltas),
        "n_boot_total": int(deltas.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiers", default=None, help="Signal-tier config, e.g. 'AC' or 'ABC'")
    parser.add_argument("--baseline", default="signal_msp")
    parser.add_argument("--n-boot", type=int, default=1000)
    # NB: '%%' -- argparse interprets '%' in a help string as a format spec.
    parser.add_argument("--alpha", type=float, default=0.05, help="CI level (0.05 -> 95%% CI)")
    args = parser.parse_args()

    files = sorted(RAW_DIR.glob("*.npz"))
    if not files:
        raise SystemExit(
            f"No raw per-instance scores in {RAW_DIR}. Re-run the sweep with "
            "experiment.cache_raw_scores=true (the default) to generate them."
        )

    runs = [load_raw_scores(f) for f in files]
    available_tiers = sorted({r["tiers"] for r in runs})
    tiers = args.tiers
    if tiers is None:
        if len(available_tiers) > 1:
            raise SystemExit(
                f"raw/ holds multiple tier configs {available_tiers}; these are "
                f"separate experiments and must not be pooled. Pass --tiers <one>."
            )
        tiers = available_tiers[0]
    elif tiers not in available_tiers:
        raise SystemExit(f"--tiers {tiers!r} not found; available: {available_tiers}")
    runs = [r for r in runs if r["tiers"] == tiers]
    print(f"[bootstrap] tiers={tiers}  runs={len(runs)}  n_boot={args.n_boot}")

    rows = []
    by_dataset: dict[str, list[dict]] = {}
    for r in runs:
        by_dataset.setdefault(r["dataset"], []).append(r)

    for dataset, dataset_runs in sorted(by_dataset.items()):
        methods = sorted(set(dataset_runs[0]["methods"]))
        if args.baseline not in methods:
            print(f"[bootstrap] {dataset}: no {args.baseline!r} baseline, skipping")
            continue
        for method in methods:
            if method == args.baseline:
                continue
            all_deltas = []
            for r in dataset_runs:
                names = r["methods"]
                if method not in names:
                    continue
                # A fresh generator per (seed, method) keyed on the run's
                # own seed, so the table is reproducible regardless of the
                # order files happen to be globbed in.
                rng = np.random.default_rng(abs(hash((dataset, method, r["seed"]))) % (2**32))
                all_deltas.append(
                    bootstrap_aurc_deltas(
                        r["scores"][names.index(method)],
                        r["scores"][names.index(args.baseline)],
                        r["incorrect"].astype(int),
                        n_boot=args.n_boot,
                        rng=rng,
                    )
                )
            if not all_deltas:
                continue
            summary = summarize_deltas(all_deltas, alpha=args.alpha)
            if summary is None:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "tiers": tiers,
                    "baseline": args.baseline,
                    "method": method,
                    **summary,
                }
            )

    table = pd.DataFrame(rows)
    out = TABLE_DIR / f"bootstrap_vs_{args.baseline}_{tiers}.csv"
    table.to_csv(out, index=False)
    print(f"\n=== Paired bootstrap AURC delta vs. {args.baseline} "
          f"({100*(1-args.alpha):.0f}% CI; negative = beats baseline) ===")
    if table.empty:
        print("(no comparisons produced)")
    else:
        cols = ["dataset", "method", "mean_aurc_delta", "ci_lo", "ci_hi",
                "mean_ci_lo", "mean_ci_hi"]
        print(table.sort_values(["dataset", "mean_aurc_delta"])[cols].to_string(index=False))

        wins = table[table.beats_baseline]
        print(f"\n[bootstrap] beat {args.baseline}, POOLED CI excludes 0 "
              f"(conservative): {len(wins)}")
        if not wins.empty:
            print(wins[["dataset", "method", "mean_aurc_delta", "ci_lo", "ci_hi"]]
                  .to_string(index=False))

        mwins = table[table.mean_effect_beats_baseline]
        print(f"\n[bootstrap] beat {args.baseline}, SEED-AVERAGED CI excludes 0 "
              f"(mean-effect claim): {len(mwins)}")
        if not mwins.empty:
            print(mwins[["dataset", "method", "mean_aurc_delta", "mean_ci_lo", "mean_ci_hi"]]
                  .to_string(index=False))
    print(f"[bootstrap] wrote {out}")


if __name__ == "__main__":
    main()
