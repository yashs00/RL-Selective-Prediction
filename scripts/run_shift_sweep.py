"""RQ2 shift-intensity sweep: does the aggregation advantage over MSP grow
as covariate shift intensifies?

This is the sharpest available form of RQ2. "Does aggregation beat MSP on a
second shifted dataset?" is one noisy comparison; "does the gap over MSP
grow monotonically with a shift we control" is a *trend* with a
pre-specified direction, and a null there is far more informative than a
null on one more dataset.

Protocol (matching the temporal setting's logic): fit everything -- base
model, cross-fitted signals, aggregators, conformal threshold -- on the
**clean** train/meta/cal splits exactly once, then evaluate every method on
D_test perturbed at increasing intensity. Nothing is refit per intensity,
because that is the deployment situation being modelled: the system was
built before the shift arrived.

Writes long-format rows to `results/shift_results.parquet` (kept separate
from `results.parquet`, whose rows are all clean-test-set numbers -- mixing
them would silently pool two different experiments) and a figure of
AURC-delta-vs-MSP against intensity.

Usage:
    python scripts/run_shift_sweep.py --dataset adult --n-seeds 5
    python scripts/run_shift_sweep.py --dataset adult --kind subpopulation
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

from src.aggregators.a0_rank import RankAverageAggregator  # noqa: E402
from src.aggregators.a1_stacking import (  # noqa: E402
    LightGBMStackingAggregator,
    LogRegStackingAggregator,
)
from src.aggregators.a2_coverage_loss import (  # noqa: E402
    AdaptiveGatingAggregator,
    MLPAggregator,
)
from src.data import splits as split_utils  # noqa: E402
from src.data.loaders import load as load_dataset  # noqa: E402
from src.data.shift import apply_covariate_shift  # noqa: E402
from src.experiment.runner import (  # noqa: E402
    _build_signal_bank,
    _fit_model,
)
from src.experiment.seeding import set_seed  # noqa: E402
from src.metrics.selective import aurc  # noqa: E402
from src.signals.tier_a import (  # noqa: E402
    CalibrationResidualSignal,
    TemperatureScaledMSPSignal,
)

RESULTS_PATH = ROOT / "results" / "shift_results.parquet"
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)

DEFAULT_INTENSITIES = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)


def run_one_seed(
    dataset_name: str,
    seed: int,
    kind: str,
    intensities: tuple[float, ...],
    model_name: str = "lightgbm",
    tiers: tuple[str, ...] = ("A", "C"),
    n_cv_folds: int = 5,
) -> list[dict]:
    set_seed(seed)
    ds = load_dataset(dataset_name)
    sp = split_utils.four_way_split(len(ds.y), y=ds.y, seed=seed, temporal=ds.is_temporal)

    X_train, y_train = ds.X.iloc[sp.train_idx], ds.y[sp.train_idx]
    X_meta, y_meta = ds.X.iloc[sp.meta_idx], ds.y[sp.meta_idx]
    X_test, y_test = ds.X.iloc[sp.test_idx], ds.y[sp.test_idx]
    pool_idx = np.concatenate([sp.train_idx, sp.meta_idx])
    X_pool, y_pool = ds.X.iloc[pool_idx], ds.y[pool_idx]
    n_classes = int(len(ds.classes))

    # --- fit on CLEAN data only (see module docstring)
    model_train = _fit_model(model_name, X_train, y_train, seed=seed)
    held_out = [
        TemperatureScaledMSPSignal().fit(X_meta, y_meta, model_train),
        CalibrationResidualSignal().fit(X_meta, y_meta, model_train),
    ]
    held_out_names = [s.name for s in held_out]

    def fit_model_fn(idx_abs: np.ndarray, fold: int = 0):
        Xi, yi = ds.X.iloc[idx_abs], ds.y[idx_abs]
        m = _fit_model(model_name, Xi, yi, seed=seed * 1000 + fold)
        bank = _build_signal_bank(tiers, n_classes=n_classes)
        bank.fit(Xi, yi, m)
        return {"model": m, "bank": bank}

    def compute_signals_fn(bundle: dict, idx_abs: np.ndarray) -> np.ndarray:
        Xi = ds.X.iloc[idx_abs]
        u = bundle["bank"].transform(Xi, bundle["model"])
        pred = bundle["model"].predict_proba(Xi).argmax(axis=1)
        correct = (pred == ds.y[idx_abs]).astype(float).reshape(-1, 1)
        return np.hstack([u, correct])

    oof = split_utils.cross_fitted_signals(
        fit_model_fn, compute_signals_fn, pool_idx,
        y_pool=y_pool, n_folds=n_cv_folds, seed=seed,
    )
    U_pool, correct_pool = oof[:, :-1], oof[:, -1].astype(bool)
    is_meta = np.isin(pool_idx, sp.meta_idx)
    U_meta_cf, correct_meta = U_pool[is_meta], correct_pool[is_meta]

    model_final = _fit_model(model_name, X_pool, y_pool, seed=seed)
    final_bank = _build_signal_bank(tiers, n_classes=n_classes)
    final_bank.fit(X_pool, y_pool, model_final)
    signal_names = list(final_bank.names) + held_out_names

    def score_all_signals(X: pd.DataFrame) -> np.ndarray:
        cols = [final_bank.transform(X, model_final)]
        cols += [s.score(X, model_final).reshape(-1, 1) for s in held_out]
        return np.hstack(cols)

    U_meta_full = np.hstack(
        [U_meta_cf] + [s.score(X_meta, model_train).reshape(-1, 1) for s in held_out]
    )

    aggregators = {
        "A0_rank": RankAverageAggregator(),
        "A1_logreg": LogRegStackingAggregator(seed=seed),
        "A1_lightgbm": LightGBMStackingAggregator(seed=seed),
        "A2_mlp_bce": MLPAggregator(loss="bce", seed=seed),
        "A2_mlp_loss1": MLPAggregator(loss="loss1", seed=seed),
        "A2_mlp_loss2": MLPAggregator(loss="loss2", seed=seed),
        "A2_mlp_loss3": MLPAggregator(loss="loss3", seed=seed),
        "A2_adaptive_loss3": AdaptiveGatingAggregator(loss="loss3", seed=seed),
    }
    for agg in aggregators.values():
        agg.fit(U_meta_full, correct_meta)

    rows = []
    for intensity in intensities:
        # One draw per (seed, intensity); the returned index (subpopulation
        # only) must be applied to y as well -- apply_covariate_shift
        # returns both so they cannot desynchronise.
        rng = np.random.default_rng(seed * 7919 + int(intensity * 1000))
        X_sh, idx = apply_covariate_shift(
            X_test, kind=kind, intensity=intensity, rng=rng, reference=X_train
        )
        y_sh = y_test if idx is None else y_test[idx]

        U_sh = score_all_signals(X_sh)
        pred_sh = model_final.predict_proba(X_sh).argmax(axis=1)
        incorrect_sh = (pred_sh != y_sh).astype(int)
        base_err = float(incorrect_sh.mean())

        def _row(method: str, score: np.ndarray) -> dict:
            return dict(
                dataset=dataset_name, base_model=model_name,
                tiers="".join(sorted(tiers)), shift_kind=kind,
                intensity=intensity, seed=seed, method=method,
                aurc=float(aurc(score, incorrect_sh)),
                base_error_rate=base_err, n_test=int(len(incorrect_sh)),
            )

        for j, nm in enumerate(signal_names):
            rows.append(_row(f"signal_{nm}", U_sh[:, j]))
        for nm, agg in aggregators.items():
            rows.append(_row(nm, agg.score(U_sh)))
        rows.append(_row("oracle", incorrect_sh.astype(float)))
        rows.append(_row("random", np.random.default_rng(seed).random(len(incorrect_sh))))
    return rows


def plot_gap_vs_intensity(df: pd.DataFrame, dataset: str, kind: str) -> Path:
    """AURC(method) - AURC(MSP) against shift intensity. RQ2 predicts this
    trends *downward* (the aggregator's advantage grows) for at least some
    aggregator; a flat or upward line is the null."""
    base = (
        df[df.method == "signal_msp"]
        .groupby("intensity")["aurc"].mean()
        .rename("msp_aurc")
    )
    methods = [m for m in df.method.unique() if m.startswith(("A0", "A1", "A2"))]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for m in sorted(methods):
        g = df[df.method == m].groupby("intensity")["aurc"].mean()
        gap = (g - base).reindex(sorted(df.intensity.unique()))
        ax.plot(gap.index, gap.values, marker="o", label=m, linewidth=1.4)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(f"{kind} shift intensity")
    ax.set_ylabel("AURC(method) - AURC(MSP)   (< 0 = beats MSP)")
    ax.set_title(f"RQ2: aggregation gap vs. shift intensity -- {dataset} ({kind})")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / f"shift_gap_{dataset}_{kind}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="adult")
    p.add_argument("--kind", default="gaussian_noise",
                   help="gaussian_noise | feature_scale | subpopulation")
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--intensities", default=",".join(str(i) for i in DEFAULT_INTENSITIES))
    args = p.parse_args()

    intensities = tuple(float(x) for x in args.intensities.split(","))
    all_rows: list[dict] = []
    for seed in range(args.n_seeds):
        print(f"[shift] {args.dataset} kind={args.kind} seed={seed}", flush=True)
        all_rows += run_one_seed(args.dataset, seed, args.kind, intensities)

    new = pd.DataFrame(all_rows)
    key = ["dataset", "base_model", "tiers", "shift_kind", "seed"]
    if RESULTS_PATH.exists():
        old = pd.read_parquet(RESULTS_PATH)
        mask = old.set_index(key).index.isin(new.set_index(key).index)
        new = pd.concat([old[~mask], new], ignore_index=True)
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    new.to_parquet(RESULTS_PATH)
    print(f"[shift] wrote {len(new)} rows -> {RESULTS_PATH}")

    sub = new[(new.dataset == args.dataset) & (new.shift_kind == args.kind)]
    fig_path = plot_gap_vs_intensity(sub, args.dataset, args.kind)
    print(f"[shift] wrote {fig_path}")

    print("\n=== base error rate and best aggregator gap vs. MSP, by intensity ===")
    msp = sub[sub.method == "signal_msp"].groupby("intensity")["aurc"].mean()
    aggs = sub[sub.method.str.startswith(("A0", "A1", "A2"))]
    for i in sorted(sub.intensity.unique()):
        err = sub[sub.intensity == i]["base_error_rate"].mean()
        gaps = aggs[aggs.intensity == i].groupby("method")["aurc"].mean() - msp.loc[i]
        best = gaps.idxmin()
        print(f"  intensity={i:<5} base_err={err:.4f}  msp_aurc={msp.loc[i]:.4f}  "
              f"best={best} gap={gaps.min():+.5f}")


if __name__ == "__main__":
    main()
