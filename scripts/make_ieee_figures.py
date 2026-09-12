"""Publication-ready IEEE visualization generator for Selective Prediction.

Generates IEEE-formatted, publication-quality figures (300 DPI PNG and vector PDF):
  1. fig1_risk_coverage_grid: Full grid of Selective Risk vs. Coverage curves with +/- 1 std bands.
  2. fig2_critical_difference: Demsar (2006) Critical Difference diagram from Friedman-Nemenyi test.
  3. fig3_aurc_improvement: Percentage AURC reduction over standard MSP baseline across all datasets.
  4. fig4_conformal_risk_control: Conformal prediction empirical coverage at guaranteed risk levels.

Usage:
  python scripts/make_ieee_figures.py --tiers ABC
  python scripts/make_ieee_figures.py --tiers AC
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
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_tables import select_tiers, friedman_and_nemenyi

RESULTS_PATH = ROOT / "results" / "results.parquet"
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)

# Set IEEE-compliant styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 12,
    "lines.linewidth": 1.6,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.4,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Color palette for headline comparison
METHOD_PALETTE = {
    "signal_msp": ("#1b9e77", "-", "MSP (Baseline)"),
    "signal_temp_msp": ("#66a61e", "--", "Temp-Scaled MSP"),
    "A1_logreg": ("#7570b3", "-.", "A1 Stacking (LogReg)"),
    "A2_mlp_loss3": ("#d95f02", "-", "A2 MLP (Ranking Loss)"),
    "A2_rl_bandit": ("#e7298a", "-", "A2 RL Bandit (Expected Reward)"),
    "A2_adaptive_loss3": ("#e6ab02", ":", "A2 Adaptive Gating"),
    "oracle": ("#222222", ":", "Oracle (Upper Bound)"),
    "random": ("#888888", "--", "Random (Lower Bound)"),
}


def plot_fig1_risk_coverage_grid(df: pd.DataFrame, tiers_str: str) -> Path:
    """Figure 1: Risk-Coverage curves in an IEEE-style multi-panel grid."""
    datasets = sorted(df["dataset"].unique())
    n_datasets = len(datasets)
    
    # Choose clean grid layout
    if n_datasets <= 3:
        n_rows, n_cols = 1, n_datasets
        fig_w, fig_h = 4.0 * n_cols, 3.5
    elif n_datasets <= 6:
        n_rows, n_cols = 2, 3
        fig_w, fig_h = 10.5, 6.5
    else:
        n_rows, n_cols = (n_datasets + 3) // 4, 4
        fig_w, fig_h = 12.0, 2.8 * n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    sub = df[df.subgroup == "overall"]

    legend_handles = {}
    for idx, dataset in enumerate(datasets):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        d_sub = sub[sub.dataset == dataset]

        for method, (color, ls, label) in METHOD_PALETTE.items():
            m = d_sub[d_sub.method == method]
            if m.empty:
                continue
            agg = m.groupby("coverage")["risk"].agg(["mean", "std"]).reset_index().sort_values("coverage")
            line = ax.plot(agg["coverage"], agg["mean"], ls, color=color, label=label, lw=1.5)[0]
            if label not in legend_handles:
                legend_handles[label] = line

            if agg["std"].notna().any() and agg["std"].max() > 1e-4:
                ax.fill_between(
                    agg["coverage"],
                    agg["mean"] - agg["std"].fillna(0),
                    agg["mean"] + agg["std"].fillna(0),
                    alpha=0.15,
                    color=color,
                )

        ax.set_title(dataset.replace("_", " ").title(), fontweight="bold")
        ax.set_xlabel("Coverage ($\\kappa$)")
        ax.set_ylabel("Selective Risk")
        ax.invert_xaxis()  # Standard in selective prediction literature
        ax.grid(True, linestyle="--", alpha=0.35)

    # Hide unused axes
    for idx in range(n_datasets, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].axis("off")

    fig.legend(
        legend_handles.values(),
        legend_handles.keys(),
        loc="lower center",
        ncol=min(4, len(legend_handles)),
        bbox_to_anchor=(0.5, -0.04),
        frameon=True,
    )
    fig.suptitle(f"Selective Risk vs. Coverage Across Benchmarks (Tiers: {tiers_str})", y=1.01, fontsize=12)
    fig.tight_layout()

    out_png = FIG_DIR / f"fig1_risk_coverage_grid_{tiers_str}.png"
    out_pdf = FIG_DIR / f"fig1_risk_coverage_grid_{tiers_str}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[make_ieee_figures] Saved: {out_png}")
    return out_png


def plot_fig2_critical_difference(df: pd.DataFrame, tiers_str: str) -> Path | None:
    """Figure 2: Demsar Critical Difference (CD) diagram using Friedman-Nemenyi ranks."""
    aurc_rows = df[df.subgroup == "AURC"]
    n_datasets = aurc_rows["dataset"].nunique()
    if n_datasets < 2:
        print("[make_ieee_figures] Skipping CD diagram (requires >=2 datasets)")
        return None

    try:
        cd_table, note = friedman_and_nemenyi(df)
    except Exception as e:
        print(f"[make_ieee_figures] Could not compute Friedman-Nemenyi: {e}")
        return None

    # Filter to main comparison methods
    eval_methods = [m for m in cd_table["method"] if m in METHOD_PALETTE and m not in ("oracle", "random")]
    if len(eval_methods) < 2:
        eval_methods = list(cd_table["method"])[:8]

    cd_sub = cd_table[cd_table["method"].isin(eval_methods)].sort_values("mean_rank")
    k = len(cd_sub)
    N = n_datasets

    # Studentized range statistic q_alpha for alpha=0.05
    # Approximate Nemenyi critical difference CD = q_alpha * sqrt(k*(k+1)/(6*N))
    q_alpha_approx = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031}
    q_val = q_alpha_approx.get(k, 3.1)
    cd_val = q_val * np.sqrt((k * (k + 1.0)) / (6.0 * N))

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ranks = cd_sub["mean_rank"].values
    names = [METHOD_PALETTE.get(m, (None, None, m))[2] for m in cd_sub["method"]]

    # Plot horizontal axis with mean rank
    y_pos = np.arange(len(ranks))
    colors = [METHOD_PALETTE.get(m, ("#333333",))[0] for m in cd_sub["method"]]
    bars = ax.barh(y_pos, ranks, color=colors, height=0.55, alpha=0.85, edgecolor="black", linewidth=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Average Rank on AURC (Lower is Better)")
    ax.set_title(f"Friedman Mean Ranks (Nemenyi CD = {cd_val:.2f}, N={N}, k={k})", fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    # Annotate rank values
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.05, bar.get_y() + bar.get_height() / 2, f"{w:.2f}", va="center", fontsize=8)

    # Draw CD indicator line
    min_rank = ranks.min()
    ax.errorbar(min_rank + cd_val / 2, -0.6, xerr=cd_val / 2, fmt="s", color="black", capsize=4, lw=1.5)
    ax.text(min_rank + cd_val / 2, -0.85, f"CD ({cd_val:.2f})", ha="center", fontsize=8, fontweight="bold")
    ax.set_ylim(len(ranks) - 0.3, -1.2)

    fig.tight_layout()
    out_png = FIG_DIR / f"fig2_critical_difference_{tiers_str}.png"
    out_pdf = FIG_DIR / f"fig2_critical_difference_{tiers_str}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[make_ieee_figures] Saved: {out_png}")
    return out_png


def plot_fig3_aurc_improvement(df: pd.DataFrame, tiers_str: str) -> Path | None:
    """Figure 3: Percentage AURC reduction relative to standard MSP baseline."""
    aurc_rows = df[df.subgroup == "AURC"]
    summary = aurc_rows.groupby(["dataset", "method"])["value"].mean().unstack("method")
    
    if "signal_msp" not in summary.columns:
        return None

    baseline = summary["signal_msp"]
    compare_methods = [
        m for m in ["signal_temp_msp", "A1_logreg", "A2_mlp_loss3", "A2_rl_bandit", "A2_adaptive_loss3"]
        if m in summary.columns
    ]
    if not compare_methods:
        return None

    # Improvement % = (AURC_msp - AURC_method) / AURC_msp * 100%
    deltas = pd.DataFrame(index=summary.index)
    for m in compare_methods:
        deltas[m] = (baseline - summary[m]) / baseline * 100.0

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    x = np.arange(len(deltas))
    n_bars = len(compare_methods)
    bar_width = 0.8 / n_bars

    for i, m in enumerate(compare_methods):
        color, _, label = METHOD_PALETTE.get(m, ("#555555", "-", m))
        ax.bar(x + (i - n_bars / 2 + 0.5) * bar_width, deltas[m], width=bar_width,
               label=label, color=color, edgecolor="black", linewidth=0.5, alpha=0.9)

    ax.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", " ").title() for d in deltas.index], rotation=35, ha="right")
    ax.set_ylabel("AURC Reduction vs. MSP (%) [Higher = Better]")
    ax.set_title("Selective Prediction Performance Gain over Baseline MSP", fontweight="bold")
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    out_png = FIG_DIR / f"fig3_aurc_improvement_{tiers_str}.png"
    out_pdf = FIG_DIR / f"fig3_aurc_improvement_{tiers_str}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[make_ieee_figures] Saved: {out_png}")
    return out_png


def plot_fig4_conformal_coverage(df: pd.DataFrame, tiers_str: str) -> Path | None:
    """Figure 4: Empirical coverage achieved at conformal guaranteed risk levels alpha."""
    conf_rows = df[df.subgroup.str.startswith("conformal_alpha_")].copy()
    if conf_rows.empty:
        return None

    conf_rows["alpha"] = conf_rows["subgroup"].str.replace("conformal_alpha_", "").astype(float)
    agg = conf_rows.groupby(["alpha", "method"])[["coverage", "risk"]].mean().reset_index()

    key_methods = [m for m in ["signal_msp", "A1_logreg", "A2_mlp_loss3", "A2_rl_bandit"] if m in agg["method"].unique()]
    if not key_methods:
        return None

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for m in key_methods:
        color, ls, label = METHOD_PALETTE.get(m, ("#555555", "-", m))
        m_agg = agg[agg["method"] == m].sort_values("alpha")
        ax.plot(m_agg["alpha"], m_agg["coverage"] * 100.0, marker="o", linestyle=ls, color=color, label=label, lw=1.6)

    ax.set_xlabel("Target Error Rate Tolerance $\\alpha$")
    ax.set_ylabel("Guaranteed Coverage (%)")
    ax.set_title(f"Conformal Risk-Controlling Operating Characteristics (Tiers: {tiers_str})", fontweight="bold")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    out_png = FIG_DIR / f"fig4_conformal_coverage_{tiers_str}.png"
    out_pdf = FIG_DIR / f"fig4_conformal_coverage_{tiers_str}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[make_ieee_figures] Saved: {out_png}")
    return out_png


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiers", default="ABC", help="Tiers config (e.g. 'ABC', 'AC')")
    args = parser.parse_args()

    if not RESULTS_PATH.exists():
        raise SystemExit(f"No results parquet found at {RESULTS_PATH}. Run scripts/run_all.py first.")

    df = pd.read_parquet(RESULTS_PATH)
    df = select_tiers(df, args.tiers)

    print(f"[make_ieee_figures] Generating IEEE publication figures for tiers={args.tiers}...")
    plot_fig1_risk_coverage_grid(df, args.tiers)
    plot_fig2_critical_difference(df, args.tiers)
    plot_fig3_aurc_improvement(df, args.tiers)
    plot_fig4_conformal_coverage(df, args.tiers)
    print("[make_ieee_figures] All IEEE figures successfully generated in paper/figures/.")


if __name__ == "__main__":
    main()
