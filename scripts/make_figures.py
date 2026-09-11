"""Generate the paper's primary figures from results/results.parquet
(§8: "no number in the paper should exist anywhere except as a query
against this file").

Usage:
    python scripts/make_figures.py

Writes PNGs to paper/figures/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_tables import select_tiers  # noqa: E402  (shared tier-selection guard)

RESULTS_PATH = ROOT / "results" / "results.parquet"
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(exist_ok=True)

# A small, legible method subset for the headline figure -- the full
# ladder is in the appendix tables, not the main risk-coverage plot.
HEADLINE_METHODS = [
    "random",
    "signal_msp",
    "signal_temp_msp",
    "A1_logreg",
    "A2_mlp_loss3",
    "A2_adaptive_loss3",
    "oracle",
]

COLORS = {
    "random": "#999999",
    "signal_msp": "#1b9e77",
    "signal_temp_msp": "#66a61e",
    "A1_logreg": "#7570b3",
    "A2_mlp_loss3": "#d95f02",
    "A2_adaptive_loss3": "#e7298a",
    "oracle": "#111111",
}


def plot_risk_coverage(df: pd.DataFrame, dataset: str, ax) -> None:
    sub = df[(df.dataset == dataset) & (df.subgroup == "overall")]
    for method in HEADLINE_METHODS:
        m = sub[sub.method == method]
        if m.empty:
            continue
        agg = m.groupby("coverage")["risk"].agg(["mean", "std"]).reset_index().sort_values("coverage")
        style = "--" if method == "oracle" else "-"
        ax.plot(agg["coverage"], agg["mean"], style, label=method, color=COLORS.get(method))
        if agg["std"].notna().any():
            ax.fill_between(
                agg["coverage"],
                agg["mean"] - agg["std"].fillna(0),
                agg["mean"] + agg["std"].fillna(0),
                alpha=0.15,
                color=COLORS.get(method),
            )
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective risk")
    ax.set_title(dataset)
    ax.invert_xaxis()  # coverage 1.0 -> 0.5, reading left-to-right as "accept fewer"
    ax.grid(alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tiers",
        default=None,
        help="Signal-tier config to plot (e.g. 'AC', 'ABC'). Required only "
             "when results.parquet holds more than one.",
    )
    args = parser.parse_args()

    if not RESULTS_PATH.exists():
        raise SystemExit(f"No results parquet at {RESULTS_PATH}. Run scripts/run_all.py first.")
    df = pd.read_parquet(RESULTS_PATH)
    # Different tier configurations are different experiments sharing the
    # same (dataset, method, seed) keys; averaging over them would silently
    # blend two experiments into one curve.
    df = select_tiers(df, args.tiers)
    datasets = sorted(df["dataset"].unique())

    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        plot_risk_coverage(df, dataset, ax)
    axes[0][0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Risk-coverage curves (mean ± std over seeds)")
    fig.tight_layout()
    out = FIG_DIR / "risk_coverage_curves.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    # Week-2 gate figure: MSP vs random, per dataset, seed-0 only, matching
    # exactly what §12 ("What to do on Monday") asks for.
    fig2, axes2 = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), squeeze=False)
    for ax, dataset in zip(axes2[0], datasets):
        sub = df[(df.dataset == dataset) & (df.subgroup == "overall") & (df.seed == 0)]
        for method, color in [("random", "#999999"), ("signal_msp", "#1b9e77")]:
            m = sub[sub.method == method].sort_values("coverage")
            ax.plot(m["coverage"], m["risk"], "-o", label=method, color=color)
        ax.set_xlabel("Coverage")
        ax.set_ylabel("Selective risk")
        ax.set_title(f"{dataset} (seed 0)")
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
        ax.legend()
    fig2.suptitle("Week-2 gate: MSP thresholding vs. random abstention")
    fig2.tight_layout()
    out2 = FIG_DIR / "week2_gate_msp_vs_random.png"
    fig2.savefig(out2, dpi=150)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
