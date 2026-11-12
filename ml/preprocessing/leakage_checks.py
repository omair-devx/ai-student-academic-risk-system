"""Phase 3, Part J — executable data-leakage and data-integrity checks.

Every function here raises AssertionError/ValueError with an explanation of
WHY the failure matters. These run:
  * automatically inside ml.preprocessing.verify (Phase 3 verification), and
  * should be imported by tests in Phase 8 as regression guards.

Design rule: policy (which features are allowed) stays in
ml.features.feature_sets — this module only ENFORCES it. Where Phase 2 already
defined validation logic we call it instead of rewriting it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from ml.features.feature_sets import (
    TARGET_CLASSES,
    TARGET_COLUMN,
    validate as validate_feature_policy,  # reuse Phase-2 logic, no duplication
)


def check_columns_match(df: pd.DataFrame, expected_features: list[str],
                        horizon: str) -> None:
    """Verify the feature frame contains EXACTLY the approved columns.

    Catches three failure modes that would silently corrupt experiments:
      * target accidentally included in X  → direct label leakage
      * unexpected/extra columns           → features outside the audited policy
      * missing required features          → experiments not comparable
    Duplicate column names are also rejected: pandas would silently return
    both copies during selection, double-weighting that feature in encoding.
    """
    actual = list(df.columns)
    if len(actual) != len(set(actual)):
        dupes = sorted({c for c in actual if actual.count(c) > 1})
        raise ValueError(f"{horizon}: duplicate columns in X: {dupes}")

    missing = [f for f in expected_features if f not in actual]
    unexpected = [c for c in actual if c not in expected_features]
    if missing:
        raise ValueError(f"{horizon}: required features missing from data: {missing}")
    if unexpected:
        raise ValueError(
            f"{horizon}: unexpected columns present (not in approved feature "
            f"list): {unexpected}"
        )
    if TARGET_COLUMN in actual:
        raise ValueError(
            f"{horizon}: TARGET COLUMN '{TARGET_COLUMN}' found inside X — this "
            f"is direct label leakage and must never reach the pipeline."
        )
    if set(actual) != set(expected_features):
        raise ValueError(f"{horizon}: column set does not equal approved features")


def check_target_values(y: pd.Series) -> None:
    """The label must contain exactly the three documented classes.

    An unexpected value (e.g. 'Enroleed' typo, or a NaN from a bad merge)
    would silently create a fourth class and change the problem definition.
    """
    observed = set(y.unique())
    expected = set(TARGET_CLASSES)
    if observed != expected:
        raise ValueError(
            f"Unexpected target values. Found {sorted(map(str, observed))}, "
            f"expected exactly {sorted(expected)}."
        )


def check_split_is_stratified(y_train: pd.Series, y_test: pd.Series,
                              y_full: pd.Series, tolerance: float = 0.01) -> None:
    """Class proportions in train/test must mirror the full dataset.

    Without stratification, a 20% random slice can badly under/over-represent
    the minority class ('Enrolled', 17.9%) and make evaluation noisy.
    `tolerance=0.01` allows a 1 percentage-point drift, which is normal
    rounding-level variation for 4,424 rows.
    """
    prop_full = y_full.value_counts(normalize=True)
    for name, part in [("train", y_train), ("test", y_test)]:
        prop = part.value_counts(normalize=True)
        drift = (prop - prop_full).abs().max()
        if drift > tolerance:
            raise ValueError(
                f"{name} class proportions drift from full data by {drift:.3f} "
                f"(> {tolerance}). Split is not properly stratified."
            )
    # Disjointness: the same student must never appear in both splits —
    # that would let the model memorize test rows via the training set.
    overlap = set(y_train.index) & set(y_test.index)
    if overlap:
        raise ValueError(f"Train/test index overlap: {len(overlap)} rows in both")


def check_preprocessor_fit_on_train(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numerical_features: list[str],
) -> dict:
    """Prove the fitted scaler statistics come from TRAIN, not test.

    StandardScaler stores the mean it was fitted on (`scaler.mean_`). We
    compare it to the training-column mean — they must match to float
    precision. As counter-evidence we also record how far the *test* mean is;
    if someone refit on test, `scaler.mean_` would match X_test instead.

    Returns a small dict of numbers for the verification report.
    """
    num_pipe = preprocessor.named_transformers_["num"]
    # Guard: the unscaled variant (for trees/NB) has no 'scaler' step — this
    # provenance check only applies to the scaled variant, and saying so
    # clearly beats a bare KeyError.
    if "scaler" not in num_pipe.named_steps:
        raise AssertionError(
            "Preprocessor has no 'scaler' step — call this check with the "
            "scaled variant (scale_numeric=True)."
        )
    scaler = num_pipe.named_steps["scaler"]
    if not isinstance(scaler, StandardScaler):
        raise TypeError("Expected a StandardScaler inside the 'num' transformer")

    train_means = X_train[numerical_features].astype(float).mean().to_numpy()
    if not np.allclose(scaler.mean_, train_means, rtol=1e-9, atol=1e-9):
        raise AssertionError(
            "Scaler means do NOT match training data — preprocessing appears "
            "to have been fitted on something other than the training split."
        )

    test_means = X_test[numerical_features].astype(float).mean().to_numpy()
    # Report-only: how different are train and test means? Non-zero values
    # demonstrate the scaler could not have been fitted on the test set.
    mean_gap = float(np.abs(scaler.mean_ - test_means).max())
    return {
        "scaler_matches_train_means": True,
        "max_abs_gap_scaler_vs_test_means": round(mean_gap, 6),
    }


def check_no_unseen_category_error(
    preprocessor: ColumnTransformer, sample_unseen_row: pd.DataFrame
) -> None:
    """Unseen-category safety: transforming a row with a brand-new category
    must NOT raise.

    The API (Phase 6) will receive students whose `Course`/occupation codes
    may not exist in training data. OneHotEncoder(handle_unknown='ignore')
    maps unknown categories to an all-zero block instead of crashing — this
    check proves that configuration is actually in effect.
    """
    try:
        out = preprocessor.transform(sample_unseen_row)
    except Exception as exc:  # noqa: BLE001 - any transform error is a failure
        raise AssertionError(
            f"Transforming a row with unseen categories failed: {exc}. "
            f"handle_unknown='ignore' is not working as configured."
        ) from exc
    if np.isnan(np.asarray(out.todense() if hasattr(out, "todense") else out)).any():
        raise AssertionError("Unseen-category transform produced NaN values")


def run_static_checks(df: pd.DataFrame, features: list[str], y: pd.Series,
                      horizon: str) -> None:
    """All checks that only need the prepared frame (no fitted objects).

    Also re-runs the Phase-2 horizon policy (validate_feature_policy) so a
    feature-list edit that violates the leakage audit fails here too.
    """
    validate_feature_policy()
    check_columns_match(df, features, horizon)
    check_target_values(y)
    # Horizon legality, expressed directly on the data we are about to use:
    # an H1 frame must not physically contain semester columns, even if the
    # feature list itself is correct (guards against wrong-column selection).
    if horizon == "H1":
        sem_cols = [c for c in df.columns if "sem" in c.lower()]
        if sem_cols:
            raise ValueError(f"H1 frame contains semester columns: {sem_cols}")
