"""Phase 3, Parts A/E/F — reusable sklearn preprocessing + estimator pipelines.

Architecture (one ColumnTransformer per horizon, reused everywhere):

    raw row
      ├─ numerical columns ──► SimpleImputer(median) ──► [StandardScaler?] ─┐
      └─ categorical columns ► SimpleImputer(most_frequent) ► OneHotEncoder ┴► feature matrix

Key design decisions and WHY:

* ONE fitted object per experiment. The ColumnTransformer is fitted on the
  training split only, then used to transform validation/test/future rows —
  guaranteeing identical feature representation everywhere (Part A rule:
  never encode train and test separately).

* remainder='drop'. The transformer only touches columns it was told about;
  any unexpected column physically cannot enter the model matrix. This turns
  "unexpected feature appeared" from a silent leak into a non-event.

* Median imputation (not mean): robust to the skewed count features seen in
  Phase 2 EDA. Data currently has zero missing values — the imputer is a
  SAFETY NET for future/refreshed data, and its statistics come from train.

* OneHotEncoder(handle_unknown='ignore'): the API will eventually see category
  codes that were absent in training. Unknowns become an all-zero block
  instead of an exception (Part E requirement). Target encoding is deliberately
  NOT used at this stage — it derives feature values from labels and is a
  classic leakage vector unless cross-fitted carefully.

* Scaling only where it matters (Part F):
    - scale_numeric=True  → for Logistic Regression / KNN / SVM, which are
      distance- or margin-based and distorted by mixed scales
      (ages 17-70 vs grades 95-190 vs macro rates ~0-16).
    - scale_numeric=False → for Decision Trees / Random Forest / Naive Bayes:
      trees split on order, which scaling cannot change; GaussianNB estimates
      per-feature mean/variance, making it invariant to affine scaling anyway.
  One-hot columns are intentionally left at 0/1 (scaling them adds nothing
  and keeps output interpretable).

* Feature names: ColumnTransformer.get_feature_names_out() recovers names like
  "num__Age at enrollment" and "cat__Course_9003" for reports and explainability.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numerical_features: list[str],
    categorical_features: list[str],
    *,
    scale_numeric: bool,
) -> ColumnTransformer:
    """Build an UNFITTED ColumnTransformer for the given column groups.

    Parameters
    ----------
    scale_numeric : bool
        True  → pipeline for LR/KNN/SVM (scaled inputs).
        False → pipeline for trees/NB (raw inputs, imputation only).
        Producing both as objects (not separate copied datasets) is Part F's
        rule: algorithms get appropriate preprocessing via pipelines, while
        the underlying data stays identical.
    """
    # Numerical branch: impute → optional scale. Named steps so later phases
    # can introspect the fitted scaler (leakage_checks does exactly that).
    numerical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        *([("scaler", StandardScaler())] if scale_numeric else []),
    ])

    # Categorical branch: impute → one-hot. Most-frequent imputation keeps the
    # original dtype/levels; 'ignore' is the unseen-category safety switch.
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numerical_pipe, numerical_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",  # anything not explicitly listed is refused entry
        verbose_feature_names_out=True,  # "num__"/"cat__" prefixes avoid name
                                          # collisions between the two branches
    )


def build_estimator_pipeline(preprocessor: ColumnTransformer, estimator) -> Pipeline:
    """Glue a fitted-on-train-ready preprocessor to a model.

    Phase 4 will call this as:
        Pipeline([("preprocess", build_preprocessor(...)),
                  ("model", LogisticRegression(...))])
    and cross-validate THE WHOLE PIPELINE — so each fold fits its own
    preprocessing on that fold's training rows only (Part C requirement).

    NOTE (Phase 3 rule): this function only *assembles* the object.
    Nothing here fits or predicts anything.
    """
    return Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", estimator),
    ])
