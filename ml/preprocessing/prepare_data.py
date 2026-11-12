"""Phase 3, Parts B/C/H/I — reproducible dataset preparation for H1/H2.

Responsibilities:
  * load the raw UCI #697 frame (read-only, via the Phase-2 loader — no
    duplicated loading logic),
  * select exactly the approved features for a horizon/experiment,
  * stratified 80/20 train/test split with a fixed seed,
  * expose the StratifiedKFold configuration for Phase-4 cross-validation,
  * optionally persist the prepared splits to data/processed/.

WHAT THIS MODULE DELIBERATELY DOES NOT DO: fit any scaler/encoder (that
happens later — inside CV folds only — per Part C), train models, or touch
the test split beyond creating it once.

Split policy (Part B):
  * ratio 80/20 — matches the split recommended by the dataset authors
    (verified from UCI metadata in Phase 2) and leaves ~885 test rows, enough
    for stable per-class metrics on the smallest class (Enrolled ≈ 159 rows).
  * stratify=y — preserves class proportions in both splits (y is mildly
    imbalanced: 49.9 / 32.1 / 17.9 %).
  * random_state=42 — a project-wide constant; every split in this codebase
    uses the same seed so results are comparable and reproducible.
  * the test split is written once and thereafter treated as read-only data;
    no fitted object may ever see it (enforced by leakage_checks).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from ml.eda.run_eda import load_delimited  # reuse Phase-2 loader (no duplication)
from ml.features.feature_sets import (
    H1_EXPERIMENTS,
    H2_FEATURES,
    TARGET_COLUMN,
)
from ml.preprocessing.feature_types import get_feature_types
from ml.preprocessing.leakage_checks import (
    check_split_is_stratified,
    run_static_checks,
)

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
RAW_697 = PROJECT_ROOT / "data" / "raw" / "uci697" / "data.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "uci697"

# Project-wide constants for reproducibility (see module docstring).
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5  # CV folds for Phase 4; 5 folds → each Enrolled sample still
              # appears in validation folds ~once, keeping metrics stable.


@dataclass(frozen=True)
class PreparedData:
    """Container for one horizon's split data + its column configuration.

    Frozen (immutable) on purpose: downstream code must not accidentally
    reassign X_train or append columns, which would invalidate the checks.
    """
    horizon: str            # "H1" or "H2"
    experiment: str         # "A" or "B" (H1 ablation variants; "A" for H2)
    features: list[str]     # approved raw feature columns, in policy order
    numerical: list[str]    # subset treated as numeric by the pipeline
    categorical: list[str]  # subset treated as categorical by the pipeline
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    @property
    def feature_count(self) -> int:
        return len(self.features)


def resolve_features(horizon: str, experiment: str = "A") -> list[str]:
    """Map (horizon, experiment) → the approved feature list.

    Kept separate from loading so policy resolution is testable on its own.
    Experiment B only exists for H1 (it is the uncertain-feature ablation);
    asking for 'B' on H2 is a logic error and raises rather than silently
    returning something ill-defined.
    """
    if horizon == "H1":
        if experiment not in H1_EXPERIMENTS:
            raise ValueError(
                f"Unknown experiment {experiment!r} for H1; "
                f"expected one of {sorted(H1_EXPERIMENTS)}"
            )
        return list(H1_EXPERIMENTS[experiment])
    if horizon == "H2":
        if experiment != "A":
            raise ValueError(
                "The uncertain-feature ablation (Experiment B) is defined for "
                "H1 only; H2 uses experiment='A'."
            )
        return list(H2_FEATURES)
    raise ValueError(f"Unknown horizon {horizon!r}: expected 'H1' or 'H2'")


def load_master_frame() -> pd.DataFrame:
    """Load raw UCI #697 once (read-only) with the verified Phase-2 loader."""
    return load_delimited(RAW_697, "UCI #697")


def prepare_horizon(
    horizon: str,
    experiment: str = "A",
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    df: pd.DataFrame | None = None,
) -> PreparedData:
    """Select approved features, validate them, and create the stratified split.

    The split happens on ROWS only (never on encoded columns): encoding,
    imputation and scaling are fitted later by the ColumnTransformer on the
    training fold — never here, never on the full data.
    """
    features = resolve_features(horizon, experiment)
    numerical, categorical = get_feature_types(horizon)

    # Narrow the transformer column lists to THIS experiment's features.
    # Needed for H1 Experiment B: it drops four *categorical* uncertain
    # features, so the preprocessor must not look for columns that X won't
    # contain (ColumnTransformer raises if a listed column is absent).
    numerical = [f for f in numerical if f in features]
    categorical = [f for f in categorical if f in features]
    if set(numerical) | set(categorical) != set(features):
        # If this fires, a feature was neither classified nor removed — the
        # type tables and the experiment config have drifted apart.
        raise ValueError(
            f"Type classification does not cover experiment {experiment} of "
            f"{horizon}: covered={sorted(set(numerical) | set(categorical))}, "
            f"expected={sorted(features)}"
        )

    if df is None:
        df = load_master_frame()
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Raw data is missing target column {TARGET_COLUMN!r}")

    X = df[features].copy()      # explicit selection → only approved columns enter
    y = df[TARGET_COLUMN].copy()

    # Static policy checks BEFORE splitting: wrong inputs must fail here,
    # not half-way through an experiment (Part J).
    run_static_checks(X, features, y, horizon)

    # Stratified split: y proportions preserved, fixed seed → identical rows
    # every run (verified in Phase 3 verification via double-run hashing).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Post-split integrity: stratification + disjointness (Part J).
    check_split_is_stratified(y_train, y_test, y)

    return PreparedData(
        horizon=horizon,
        experiment=experiment,
        features=features,
        numerical=numerical,
        categorical=categorical,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


def make_stratified_kfold(
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> StratifiedKFold:
    """Cross-validation splitter configured for Phase 4 (NOT run in Phase 3).

    Why shuffled + stratified: folds must each resemble the overall class
    balance, otherwise a fold missing most 'Enrolled' students produces
    meaningless class metrics. shuffle=True with a fixed seed makes fold
    membership reproducible.

    Part C reminder: pass this splitter to cross_validate() together with the
    FULL sklearn Pipeline (preprocessor + model). scikit-learn then refits the
    preprocessor on each training fold independently, so validation-fold
    statistics (means, medians, categories) never influence training.
    NEVER fit a preprocessor on the whole dataset before cross-validation —
    that would leak validation data into every fold.
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def persist_splits(prepared: PreparedData, out_dir: Path = PROCESSED_DIR) -> list[Path]:
    """Write train/test CSVs for a prepared horizon (raw columns, NOT encoded).

    Persisting the *splits* (not transformed matrices) keeps a single source
    of truth: any transformation is redone by the fitted pipeline, so saved
    matrices can never go stale relative to pipeline changes. Files are
    regenerable via verify.py and therefore gitignored with data/processed/.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = prepared.horizon.lower() + (
        "_expB" if prepared.experiment == "B" else ""
    )

    train_path = out_dir / f"{tag}_train.csv"
    test_path = out_dir / f"{tag}_test.csv"

    # Index is preserved: leakage_checks compares train/test indices for
    # overlap, and split provenance remains auditable.
    train_df = prepared.X_train.copy()
    train_df[TARGET_COLUMN] = prepared.y_train
    test_df = prepared.X_test.copy()
    test_df[TARGET_COLUMN] = prepared.y_test

    train_df.to_csv(train_path, index=True, index_label="_row_id")
    test_df.to_csv(test_path, index=True, index_label="_row_id")
    return [train_path, test_path]
