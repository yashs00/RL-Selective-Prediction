"""RQ4 subgroup-disparity summary (§7's secondary metric: "max-min gap,
and worst-group selective risk at fixed overall coverage"; §2's Jones et
al. 2021 threat/motivation).

`results/results.parquet` already carries `max_min_gap@0.8` /
`max_min_gap@0.9` rows for every dataset with a `subgroup_col` configured
(Adult: sex/race; German Credit: personal_status/age/foreign_worker --
see `configs/dataset/*.yaml`), computed by
`src.metrics.subgroup.worst_group_selective_risk`. No table anywhere in
`paper/tables/` summarized them until this script -- RQ4 was measured but
never reported. Electricity has no subgroup columns and is absent by
construction, not by omission. (Diabetes-130 would have been a third
subgroup-bearing dataset; it was dropped from the project -- too many
incomplete fields -- before contributing any results; see
PROJECT_STATUS.md.)

Reads the exact same per-seed paired structure as
`make_tables.pairwise_wilcoxon_vs_baseline`, applied to the subgroup-gap
value instead of AURC: for each dataset and each coverage threshold,
Wilcoxon-tests every aggregator's max-min gap against MSP's, Holm-corrected
across the methods tested for that (dataset, threshold) pair. A *negative*
`mean_gap_delta` means the method narrows the disparity relative to MSP
(the direction RQ4 asks about); positive means it widens it, which is the
Jones et al. failure mode this whole research question exists to check for.

Usage:
    python scripts/make_subgroup_table.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_tables import holm_bonferroni, select_tiers  # noqa: E402

RESULTS_PATH = ROOT / "results" / "results.parquet"
TABLE_DIR = ROOT / "paper" / "tables"
TABLE_DIR.mkdir(exist_ok=True, parents=True)

GAP_SUBGROUPS = ("max_min_gap@0.8", "max_min_gap@0.9")


def subgroup_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = df[df.subgroup.isin(GAP_SUBGROUPS)]
    if rows.empty:
        return pd.DataFrame(
            columns=["dataset", "coverage_threshold", "method", "gap_mean", "gap_std", "n_seeds"]
        )
    summary = (
        rows.groupby(["dataset", "subgroup", "method"])["value"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "gap_mean", "std": "gap_std", "count": "n_seeds"})
        .reset_index()
        .rename(columns={"subgroup": "coverage_threshold"})
    )
    return summary


def pairwise_wilcoxon_gap_vs_msp(df: pd.DataFrame, baseline: str = "signal_msp") -> pd.DataFrame:
    rows = df[df.subgroup.isin(GAP_SUBGROUPS)]
    results = []
    for (dataset, threshold), g in rows.groupby(["dataset", "subgroup"]):
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
            # negative = method narrows the gap relative to MSP (the
            # direction RQ4 hopes for); positive = widens it (the Jones et
            # al. failure mode).
            deltas.append(float((paired[method] - paired[baseline]).mean()))
            method_names.append(method)
        if not pvals:
            continue
        adj = holm_bonferroni(pvals)
        for method, p, p_adj, delta in zip(method_names, pvals, adj, deltas):
            results.append(
                {
                    "dataset": dataset,
                    "coverage_threshold": threshold,
                    "baseline": baseline,
                    "method": method,
                    "mean_gap_delta": delta,
                    "p_value": p,
                    "p_value_holm": p_adj,
                    "significant_at_0.05": p_adj < 0.05,
                    "narrows_disparity": delta < 0,
                }
            )
    cols = [
        "dataset", "coverage_threshold", "baseline", "method", "mean_gap_delta",
        "p_value", "p_value_holm", "significant_at_0.05", "narrows_disparity",
    ]
    return pd.DataFrame(results, columns=cols)


def main() -> None:
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

    datasets_with_subgroups = sorted(
        df[df.subgroup.isin(GAP_SUBGROUPS)]["dataset"].unique()
    )
    all_datasets = sorted(df["dataset"].unique())
    missing = [d for d in all_datasets if d not in datasets_with_subgroups]
    print(f"[subgroup] datasets with subgroup data: {datasets_with_subgroups}")
    if missing:
        print(
            f"[subgroup] no subgroup columns for: {missing} "
            "(expected -- e.g. Electricity has no subgroup_col configured)"
        )

    summary = subgroup_summary_table(df)
    summary.to_csv(TABLE_DIR / "subgroup_gap_summary.csv", index=False)
    print("\n=== Max-min subgroup selective-risk gap (mean +/- std over seeds) ===")
    print(summary.sort_values(["dataset", "coverage_threshold", "gap_mean"]).to_string(index=False))

    wilcoxon_tbl = pairwise_wilcoxon_gap_vs_msp(df)
    wilcoxon_tbl.to_csv(TABLE_DIR / "subgroup_gap_wilcoxon_vs_msp.csv", index=False)
    print("\n=== Wilcoxon signed-rank on subgroup gap vs. MSP (Holm-corrected) ===")
    if wilcoxon_tbl.empty:
        print("(No pairwise Wilcoxon tests possible -- need subgroup data at >=2 seeds)")
    else:
        print(
            wilcoxon_tbl.sort_values(["dataset", "coverage_threshold", "p_value_holm"])
            .to_string(index=False)
        )
        sig = wilcoxon_tbl["significant_at_0.05"]
        any_sig_narrow = wilcoxon_tbl[sig & wilcoxon_tbl.narrows_disparity]
        any_sig_widen = wilcoxon_tbl[sig & ~wilcoxon_tbl.narrows_disparity]
        print(
            f"\n[subgroup] significantly narrows disparity vs. MSP: "
            f"{len(any_sig_narrow)} (dataset, threshold, method) rows"
        )
        print(
            f"[subgroup] significantly WIDENS disparity vs. MSP: "
            f"{len(any_sig_widen)} (dataset, threshold, method) rows "
            "-- the Jones et al. 2021 failure mode, if any"
        )


if __name__ == "__main__":
    main()
