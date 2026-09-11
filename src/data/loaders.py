"""Dataset loaders.

Every loader returns a `Dataset` (features `X`, labels `y`, optional subgroup
columns for fairness analysis, and metadata). Pulled from OpenML per the
project plan §4, which asks that datasets come from curated suites
(OpenML-CC18 / Grinsztajn et al.) rather than being hand-assembled, so that
dataset choice can't be attacked as cherry-picked.

Raw OpenML pulls are cached to disk under `data_cache/` (gitignored) because
repeated fetches are slow and this module is called dozens of times per
experiment sweep.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[2] / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)


@dataclasses.dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: np.ndarray  # integer-encoded class labels, 0..K-1
    classes: list
    subgroup_cols: list[str]  # column names in X usable for RQ4 fairness analysis
    task: str  # "binary" or "multiclass"
    is_temporal: bool = False  # True if row order encodes time (for shift splits, §4)


def _openml_frame_cache_path(dataset_id: int) -> Path:
    return CACHE_DIR / f"openml_{dataset_id}.parquet"


def fetch_openml_dataset(
    dataset_id: int,
    name: str,
    target_column: Optional[str] = None,
    subgroup_cols: Optional[list[str]] = None,
    is_temporal: bool = False,
    drop_cols: Optional[list[str]] = None,
    positive_class: Optional[str] = None,
) -> Dataset:
    """Fetch and cache a dataset from OpenML by numeric dataset id.

    Uses sklearn's fetch_openml under the hood (it talks to the same OpenML
    API and is already a project dependency), which is simpler and more
    robust across OpenML API versions than round-tripping through raw ARFF.

    `drop_cols` removes columns from X *after* caching (so the cache stays a
    faithful copy of the OpenML original). This is not cosmetic: on a
    temporal dataset, any column that encodes row order -- e.g. the
    now-removed Diabetes-130 entry's `encounter_id`, which was monotone in
    time -- lets a tree model read the time index directly, and because a
    temporal split puts every test value outside the training range, the
    model degenerates into a single branch. Such columns must be dropped,
    not merely ignored. No current registry entry needs this, but the
    argument stays generic rather than Diabetes-130-specific.

    `positive_class` binarises a multiclass target to
    `(y_raw == positive_class)`. Was used for Diabetes-130 (dropped from
    the registry, see PROJECT_STATUS.md), whose native target had three
    levels (`NO` / `>30` / `<30`) but whose standard task in the literature
    is the binary "readmitted within 30 days" (`<30`) question. Kept
    generic for any future multiclass-to-binary dataset.
    """
    cache_path = _openml_frame_cache_path(dataset_id)
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
    else:
        from sklearn.datasets import fetch_openml

        bunch = fetch_openml(data_id=dataset_id, as_frame=True, parser="auto")
        df = bunch.frame
        if target_column is None:
            target_column = bunch.target.name if bunch.target is not None else df.columns[-1]
        # Persist target column name in an attrs-preserving way: parquet drops
        # DataFrame.attrs, so we just require the caller to pass the same
        # target_column each time (documented below) or infer it below.
        df.to_parquet(cache_path)
        df.attrs["target_column"] = target_column

    if target_column is None:
        raise ValueError(
            f"target_column must be specified for dataset {dataset_id} on a cache hit "
            f"(parquet caching does not preserve DataFrame.attrs)."
        )

    y_raw = df[target_column]
    X = df.drop(columns=[target_column])

    if drop_cols:
        present = [c for c in drop_cols if c in X.columns]
        missing = sorted(set(drop_cols) - set(present))
        if missing:
            raise ValueError(
                f"drop_cols for dataset {name} names column(s) not present: "
                f"{missing}. Fix the registry rather than silently ignoring "
                f"them -- a typo here would leave a leaking column in X."
            )
        X = X.drop(columns=present)

    if positive_class is not None:
        # Binarise: 1 == positive_class, 0 == everything else.
        as_str = y_raw.astype(str)
        levels = set(as_str.unique())
        if positive_class not in levels:
            raise ValueError(
                f"positive_class={positive_class!r} not found in target "
                f"{target_column!r} of dataset {name}; levels are "
                f"{sorted(levels)}."
            )
        y = (as_str == positive_class).to_numpy().astype(np.int64)
        classes = [f"not_{positive_class}", positive_class]
        task = "binary"
    else:
        # Encode target to 0..K-1 ints, preserving a stable class ordering.
        y_cat = pd.Categorical(y_raw)
        y = y_cat.codes.astype(np.int64)
        classes = list(y_cat.categories)
        task = "binary" if len(classes) == 2 else "multiclass"

    subgroup_cols = [c for c in (subgroup_cols or []) if c in X.columns]

    return Dataset(
        name=name,
        X=X,
        y=y,
        classes=classes,
        subgroup_cols=subgroup_cols,
        task=task,
        is_temporal=is_temporal,
    )


# --- Curated registry, per project plan §4 -------------------------------
# OpenML dataset ids verified against openml.org at the time this file was
# written. Re-check ids if OpenML re-numbers a dataset.
REGISTRY = {
    "adult": dict(
        dataset_id=1590,
        target_column="class",
        subgroup_cols=["sex", "race"],
    ),
    "german_credit": dict(
        dataset_id=31,
        target_column="class",
        subgroup_cols=["personal_status", "age", "foreign_worker"],
    ),
    "electricity": dict(
        dataset_id=151,
        target_column="class",
        subgroup_cols=[],
        is_temporal=True,
    ),
    # --- Multiclass (K >= 3) datasets ------------------------------------
    #
    # Added specifically to test the project's central mechanism claim. On a
    # *binary* task every Tier-A signal is provably a monotone transform of
    # every other one (Spearman |rho| = 1.0000, pinned in
    # tests/test_fixes.py), so Tier A supplies exactly one ranking and no
    # aggregator over it can differ from MSP. At K >= 3 that degeneracy
    # provably breaks: top1-minus-top2 in probability space and in logit
    # space stop being monotone in each other, entropy stops being a
    # function of the top probability alone, and `logitnorm_msp` becomes
    # well-defined (it is excluded on binary input). These datasets are
    # therefore the direct test of "aggregation helps exactly when the
    # signals are not redundant" -- turning the project's negative result
    # into a mechanism rather than an unexplained null.
    #
    # All three come from curated suites (OpenML-CC18 / Grinsztajn et al.),
    # per plan §4's requirement that datasets not be hand-assembled, and all
    # are *smaller* than Adult, so the set runs in a fraction of
    # Diabetes-130's cost. Ids and shapes were verified directly against
    # OpenML rather than recalled.
    #
    # **They were screened on test-set error count before being admitted**,
    # which is the binding constraint for selective prediction and is easy
    # to overlook: a risk-coverage curve is built entirely out of the
    # *errors* in D_test, so an easy dataset yields a curve made of noise no
    # matter how many rows it has. Measured with the standard split and base
    # model: `segment` gave **5** test errors (98.6% accuracy), `pendigits`
    # **11**, `optdigits` **8**, `texture` **9** -- all rejected as
    # unusable. `yeast` (89 errors but only 223 test rows) was left out as
    # too marginal. The three kept:
    #
    # None has a natural demographic subgroup column, so they inform RQ1 and
    # the mechanism question, not RQ4; none is temporal, so they do *not*
    # address RQ2's shift question -- that slot still needs its own dataset.
    "wine_quality_white": dict(
        dataset_id=40498,  # 4898 x 12, K=7, base acc 0.684 -> 232 test errors
        target_column="Class",
        subgroup_cols=[],
    ),
    "letter": dict(
        dataset_id=6,  # 20000 x 17, K=26, base acc 0.966 -> 102 test errors
        target_column="class",
        subgroup_cols=[],
    ),
    "satimage": dict(
        dataset_id=182,  # 6430 x 37, K=6, base acc 0.928 -> 70 test errors
        target_column="class",
        subgroup_cols=[],
    ),
    # --- Extra Datasets for Friedman Test Statistical Power ---
    # Added to guarantee >= 12 datasets for the Friedman test rank analysis
    "phoneme": dict(
        dataset_id=1489,
        target_column="Class",
        subgroup_cols=[],
    ),
    "bank_marketing": dict(
        dataset_id=1461,
        target_column="Class",
        subgroup_cols=[],
    ),
    "nomao": dict(
        dataset_id=1486,
        target_column="Class",
        subgroup_cols=[],
    ),
    "kr_vs_kp": dict(
        dataset_id=3,
        target_column="class",
        subgroup_cols=[],
    ),
    "eeg_eye_state": dict(
        dataset_id=1471,
        target_column="Class",
        subgroup_cols=[],
    ),
    "magic_telescope": dict(
        dataset_id=1120,
        target_column="class",
        subgroup_cols=[],
    ),
    # Diabetes-130 (Strack et al. 2014) was registered and run as the plan's
    # second temporal-shift dataset (§4), but was dropped by explicit
    # decision: too many incomplete fields for the result to be trusted
    # (96.9% of rows carried a '?' sentinel for `weight` alone, on top of
    # the patient-recurrence caveat documented in git history/
    # PROJECT_STATUS.md). Removed rather than left commented out so a
    # stale, half-verified entry doesn't linger in the registry; see git
    # history for the full loader (dataset_id=4541, target="readmitted",
    # drop_cols=["encounter_id","patient_nbr","weight"]) if reviving it.
}


def load(name: str) -> Dataset:
    if name not in REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Known: {list(REGISTRY)}")
    spec = dict(REGISTRY[name])
    dataset_id = spec.pop("dataset_id")
    ds = fetch_openml_dataset(dataset_id=dataset_id, name=name, **spec)
    return ds
