"""Signal correlation heatmap -- the mechanism behind the central negative
result (§7's "signal correlation/PCA" ablation).

`tests/test_fixes.py` already pins the specific fact that every Tier-A
signal is a strictly monotone transform of every other one on a binary
task (pairwise Spearman |rho| = 1.0000) -- see PROJECT_STATUS.md, "The
finding that reframes the negative result". This script produces the
figure version of that fact across the *whole* signal bank actually
scored on D_test (Tier A/B/C plus the calibration residual), so the paper
can show, not just assert, which signals are redundant and which are not.

Deliberately does **not** reuse `run_experiment` (which also trains all
eight aggregators -- the expensive part). Computing the signal matrix
alone needs only two base-model fits (model_train, model_final) and the
signal banks' own (cheap) fits, so this is safe to run alongside other
heavy jobs without meaningfully competing for CPU.

Usage:
    python scripts/make_signal_correlation_heatmap.py dataset=adult
    python scripts/make_signal_correlation_heatmap.py dataset=adult tiers=A,B,C use_ensemble=true
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import splits as split_utils  # noqa: E402
from src.data.loaders import load as load_dataset  # noqa: E402
from src.experiment.runner import (  # noqa: E402
    MODEL_REGISTRY,
    _build_signal_bank,
    _fit_model,
    _train_ensemble,
)
from src.signals.ensemble import default_tier_b_bank  # noqa: E402
from src.signals.tier_a import CalibrationResidualSignal, TemperatureScaledMSPSignal  # noqa: E402
from src.signals.base import SignalBank  # noqa: E402

FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(exist_ok=True)


def compute_test_signal_matrix(
    dataset_name: str,
    model_name: str = "lightgbm",
    tiers: tuple[str, ...] = ("A", "C"),
    seed: int = 0,
    use_ensemble: bool = False,
    n_ensemble_members: int = 5,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """The subset of `run_experiment`'s pipeline needed to score every
    signal on D_test: dataset -> split -> model_train/model_final -> signal
    banks -> U_test. No aggregator is trained. Returns
    (U_test, signal_names, incorrect_test)."""
    ds = load_dataset(dataset_name)
    n = len(ds.y)
    sp = split_utils.four_way_split(n, y=ds.y, seed=seed, temporal=ds.is_temporal)

    X_train, y_train = ds.X.iloc[sp.train_idx], ds.y[sp.train_idx]
    X_meta, y_meta = ds.X.iloc[sp.meta_idx], ds.y[sp.meta_idx]
    X_test, y_test = ds.X.iloc[sp.test_idx], ds.y[sp.test_idx]
    pool_idx = np.concatenate([sp.train_idx, sp.meta_idx])
    X_pool, y_pool = ds.X.iloc[pool_idx], ds.y[pool_idx]
    n_classes = int(len(ds.classes))

    model_train = _fit_model(model_name, X_train, y_train, seed=seed)
    held_out_signals = [
        TemperatureScaledMSPSignal().fit(X_meta, y_meta, model_train),
        CalibrationResidualSignal().fit(X_meta, y_meta, model_train),
    ]
    held_out_names = [sig.name for sig in held_out_signals]

    tier_b_bank = None
    tier_b_names: list[str] = []
    if use_ensemble:
        ensemble_train = _train_ensemble(model_name, X_train, y_train, n_ensemble_members, seed)
        tier_b_bank = SignalBank(default_tier_b_bank(ensemble_train))
        tier_b_bank.fit(X_train, y_train, model_train)
        tier_b_names = list(tier_b_bank.names)

    cf_tiers = tuple(t for t in tiers if t != "B")
    model_final = _fit_model(model_name, X_pool, y_pool, seed=seed)
    final_bank = _build_signal_bank(cf_tiers, n_classes=n_classes)
    final_bank.fit(X_pool, y_pool, model_final)

    def _grafted(X, model) -> np.ndarray:
        cols = [sig.score(X, model).reshape(-1, 1) for sig in held_out_signals]
        if tier_b_bank is not None:
            cols.append(tier_b_bank.transform(X, model))
        return np.hstack(cols)

    U_test = np.hstack([final_bank.transform(X_test, model_final), _grafted(X_test, model_final)])
    signal_names = list(final_bank.names) + held_out_names + tier_b_names
    pred_test = model_final.predict_proba(X_test).argmax(axis=1)
    incorrect_test = (pred_test != y_test).astype(int)
    return U_test, signal_names, incorrect_test


def plot_heatmap(U: np.ndarray, names: list[str], dataset_name: str, tiers_label: str) -> Path:
    m = U.shape[1]
    corr = np.eye(m)
    for i in range(m):
        for j in range(i + 1, m):
            rho, _ = spearmanr(U[:, i], U[:, j])
            corr[i, j] = corr[j, i] = rho

    fig, ax = plt.subplots(figsize=(0.55 * m + 2, 0.55 * m + 2))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(m))
    ax.set_yticks(range(m))
    ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    for i in range(m):
        for j in range(m):
            ax.text(
                j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                fontsize=6, color="white" if abs(corr[i, j]) > 0.6 else "black",
            )
    ax.set_title(f"Signal Spearman correlation -- {dataset_name} ({tiers_label})")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman rho")
    fig.tight_layout()

    out_path = FIG_DIR / f"signal_correlation_{dataset_name}_{tiers_label}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    # Minimal `key=value` CLI -- this script is a one-off diagnostic, not
    # part of the Hydra-driven experiment sweep, so it doesn't need Hydra's
    # config composition (§8's rule targets *experiment* runs).
    kwargs = dict(arg.split("=", 1) for arg in sys.argv[1:])
    dataset_name = kwargs.get("dataset", "adult")
    tiers = tuple(kwargs.get("tiers", "A,C").split(","))
    use_ensemble = kwargs.get("use_ensemble", "false").lower() == "true"
    seed = int(kwargs.get("seed", 0))

    print(f"[correlation] dataset={dataset_name} tiers={tiers} use_ensemble={use_ensemble}")
    U, names, incorrect = compute_test_signal_matrix(
        dataset_name, tiers=tiers, seed=seed, use_ensemble=use_ensemble
    )
    tiers_label = "".join(tiers) + ("B" if use_ensemble and "B" not in tiers else "")
    out_path = plot_heatmap(U, names, dataset_name, tiers_label)
    print(f"[correlation] wrote {out_path}")

    # Print the max |rho| between calib_residual (the signal added
    # specifically to break Tier-A degeneracy) and every other Tier-A
    # signal, as a quick numeric sanity check alongside the figure.
    if "calib_residual" in names:
        ci = names.index("calib_residual")
        others = [(n, spearmanr(U[:, ci], U[:, j])[0]) for j, n in enumerate(names) if j != ci]
        print("[correlation] calib_residual vs. others (Spearman rho):")
        for n, rho in sorted(others, key=lambda t: -abs(t[1])):
            print(f"    {n:30s} {rho:+.4f}")


if __name__ == "__main__":
    main()
