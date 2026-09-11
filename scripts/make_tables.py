"""Generate the paper's primary tables + statistical tests from
results/results.parquet (§7: "10 random seeds... Wilcoxon signed-rank...
Friedman test + Nemenyi... Holm-Bonferroni").

Usage:
    python scripts/make_tables.py

Writes CSVs to paper/tables/.

Honesty note (see PROJECT_STATUS.md): the paired-bootstrap-on-raw-test-
predictions test from §7 needs per-instance test-set uncertainty/
correctness arrays cached to disk, which the current runner does not
persist (it only persists the aggregated long-format summary rows). The
Wilcoxon/Friedman tests below use *per-seed AURC values* as the paired/
blocked unit instead, which is statistically valid given the current
results schema but is a coarser test than the per-instance bootstrap the
plan calls for. Extend `src/experiment/runner.py` to cache raw
(uncertainty, incorrect) arrays per (dataset, method, seed) if you need
the finer-grained bootstrap before submission.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS_PATH = ROOT / "results" / "results.parquet"
TABLE_DIR = ROOT / "paper" / "tables"
TABLE_DIR.mkdir(exist_ok=True, parents=True)


def holm_bonferroni(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction (§7: "Correct for multiple
    comparisons (Holm-Bonferroni) and say so")."""
    order = np.argsort(pvalues)
    n = len(pvalues)
    adjusted = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        corrected = (n - rank) * pvalues[idx]
        running_max = max(running_max, corrected)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()


def aurc_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    aurc_rows = df[df.subgroup == "AURC"]
    summary = (
        aurc_rows.groupby(["dataset", "method"])["value"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "AURC_mean", "std": "AURC_std", "count": "n_seeds"})
        .reset_index()
    )
    if "tiers" in aurc_rows.columns and aurc_rows.tiers.notna().any():
        summary.insert(1, "tiers", aurc_rows.tiers.iloc[0])
    return summary


def pairwise_wilcoxon_vs_baseline(
    df: pd.DataFrame, baseline: str = "signal_msp"
) -> pd.DataFrame:
    """Per dataset: Wilcoxon signed-rank test of each method's per-seed AURC
    against the baseline's per-seed AURC (paired by seed), Holm-corrected
    within each dataset (§7)."""
    aurc_rows = df[df.subgroup == "AURC"]
    results = []
    for dataset, g in aurc_rows.groupby("dataset"):
        pivot = g.pivot(index="seed", columns="method", values="value")
        if baseline not in pivot.columns:
            continue
        methods = [m for m in pivot.columns if m != baseline]
        pvals, deltas, method_names = [], [], []
        for method in methods:
            paired = pivot[[baseline, method]].dropna()
            if len(paired) < 2 or (paired[baseline] == paired[method]).all():
                continue
            try:
                stat, p = stats.wilcoxon(paired[baseline], paired[method])
            except ValueError:
                continue
            pvals.append(p)
            deltas.append(float((paired[method] - paired[baseline]).mean()))
            method_names.append(method)
        if not pvals:
            continue
        adj = holm_bonferroni(pvals)
        for method, p, p_adj, delta in zip(method_names, pvals, adj, deltas):
            results.append(
                {
                    "dataset": dataset,
                    "baseline": baseline,
                    "method": method,
                    "mean_aurc_delta": delta,  # negative = method beats baseline
                    "p_value": p,
                    "p_value_holm": p_adj,
                    "significant_at_0.05": p_adj < 0.05,
                }
            )
    cols = [
        "dataset",
        "baseline",
        "method",
        "mean_aurc_delta",
        "p_value",
        "p_value_holm",
        "significant_at_0.05",
    ]
    return pd.DataFrame(results, columns=cols)


def friedman_and_nemenyi(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Friedman test across datasets (blocks) with methods as treatments,
    using mean AURC per (dataset, method) over seeds (§7). Note: this is
    only well-powered with several datasets -- with the current 2-dataset
    scope it is reported for completeness, not as a headline claim; see
    PROJECT_STATUS.md."""
    aurc_rows = df[df.subgroup == "AURC"]
    block_means = aurc_rows.groupby(["dataset", "method"])["value"].mean().reset_index()
    pivot = block_means.pivot(index="dataset", columns="method", values="value").dropna(axis=1)
    note = ""
    if pivot.shape[0] < 3:
        note = (
            f"WARNING: Friedman test computed over only {pivot.shape[0]} dataset(s)/blocks; "
            "the plan's protocol (§7) assumes >=3, ideally 12, datasets for this test to be "
            "meaningful. Treat this as a placeholder until more datasets are added (§9 week 8)."
        )
    if pivot.shape[0] >= 2 and pivot.shape[1] >= 3:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p = stats.friedmanchisquare(*[pivot[c].values for c in pivot.columns])
    else:
        stat, p = float("nan"), float("nan")
    ranks = pivot.rank(axis=1, method="average")
    mean_ranks = ranks.mean(axis=0).sort_values()
    result = pd.DataFrame({"method": mean_ranks.index, "mean_rank": mean_ranks.values})
    result.attrs["friedman_stat"] = stat
    result.attrs["friedman_p"] = p
    return result, note


def select_tiers(df: pd.DataFrame, requested: str | None) -> pd.DataFrame:
    """Restrict the results to a single signal-tier configuration.

    Results for different tier configurations -- (A,C) vs. (A,B,C) -- are
    different experiments that share the same (dataset, method, seed) keys.
    Aggregating over them would silently average two experiments into one
    number, so a tier configuration must be picked explicitly whenever the
    parquet holds more than one.
    """
    if "tiers" not in df.columns:
        raise SystemExit(
            "results.parquet has no 'tiers' column; it predates the current "
            "results schema. Re-run scripts/run_all.py to regenerate it."
        )
    available = sorted(df.tiers.dropna().unique())
    if requested is None:
        if len(available) > 1:
            raise SystemExit(
                f"results.parquet contains multiple signal-tier configs "
                f"{available}; these are separate experiments and must not be "
                f"pooled. Re-run with --tiers <one of {available}>."
            )
        requested = available[0]
    elif requested not in available:
        raise SystemExit(f"--tiers {requested!r} not in results; available: {available}")
    print(f"[make_tables] signal tiers = {requested}")
    return df[df.tiers == requested].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tiers",
        default=None,
        help="Signal-tier config to report on (e.g. 'AC', 'ABC'). Required "
             "only when results.parquet holds more than one.",
    )
    args = parser.parse_args()

    if not RESULTS_PATH.exists():
        raise SystemExit(f"No results parquet at {RESULTS_PATH}. Run scripts/run_all.py first.")
    df = pd.read_parquet(RESULTS_PATH)
    df = select_tiers(df, args.tiers)

    summary = aurc_summary_table(df)
    summary.to_csv(TABLE_DIR / "aurc_summary.csv", index=False)
    print("=== AURC summary (mean +/- std over seeds) ===")
    print(summary.sort_values(["dataset", "AURC_mean"]).to_string(index=False))

    wilcoxon_tbl = pairwise_wilcoxon_vs_baseline(df, baseline="signal_msp")
    wilcoxon_tbl.to_csv(TABLE_DIR / "wilcoxon_vs_msp.csv", index=False)
    print("\n=== Wilcoxon signed-rank vs. MSP baseline (Holm-corrected) ===")
    if wilcoxon_tbl.empty:
        print("(No pairwise Wilcoxon tests possible with <2 seeds per dataset)")
    else:
        print(wilcoxon_tbl.sort_values(["dataset", "p_value_holm"]).to_string(index=False))

    cd_table, note = friedman_and_nemenyi(df)
    cd_table.to_csv(TABLE_DIR / "friedman_mean_ranks.csv", index=False)
    print("\n=== Friedman mean ranks (lower = better AURC rank) ===")
    print(f"Friedman stat={cd_table.attrs['friedman_stat']:.4f} p={cd_table.attrs['friedman_p']:.4f}")
    print(cd_table.to_string(index=False))
    if note:
        print(f"\n{note}")


if __name__ == "__main__":
    main()
