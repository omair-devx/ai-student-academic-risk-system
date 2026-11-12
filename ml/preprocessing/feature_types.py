"""Phase 3, Part D — type classification for every approved H1/H2 feature.

WHY this module exists: "integer column" does NOT mean "number you can average".
In UCI #697 the value `Course = 9070` is a *label for a degree program*, not a
quantity — averaging or scaling it would be meaningless. Phase 2 identified
these integer-coded categories explicitly (reports/eda_report.md §1.4); this
module turns that knowledge into configuration the pipeline consumes.

Classification summary (documented in docs/PREPROCESSING.md with per-feature
reasoning):

  H1: 6 numerical  + 18 categorical = 24
  H2: 12 numerical + 18 categorical = 30

Type definitions:
  NUMERICAL  — genuine quantities where mean/variance are meaningful; these get
               median imputation + (optionally) StandardScaler.
  CATEGORICAL — labels/codes; these get most-frequent imputation + one-hot
               encoding. Never scaled, never treated as ordered numbers.
"""

from __future__ import annotations

from ml.features.feature_sets import H1_FEATURES, H2_FEATURES, validate

# ---------------------------------------------------------------------------
# Numerical features of H1 (6)
# ---------------------------------------------------------------------------
# Two grade scales (0-200), one age, three national macro indicators — all
# continuous. Verified ranges from Phase 2 EDA (reports/eda_report.md §1.3).
NUMERICAL_FEATURES_H1: tuple[str, ...] = (
    "Previous qualification (grade)",  # 95-190, continuous
    "Admission grade",                 # 95-190, continuous
    "Age at enrollment",               # integer years, but genuinely numeric:
                                       #   averaging ages is meaningful (mean=23.3)
    "Unemployment rate",               # macro %, continuous
    "Inflation rate",                  # macro %, continuous
    "GDP",                             # macro index, continuous
)

# ---------------------------------------------------------------------------
# Numerical features added by H2 (6)
# ---------------------------------------------------------------------------
# Semester-1 curricular columns: counts of units and a 0-20 grade average.
# All are quantities (a mean of counts/grades is meaningful) → numerical.
NUMERICAL_FEATURES_H2_ADDITIONS: tuple[str, ...] = (
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",           # 0-20 average, continuous
    "Curricular units 1st sem (without evaluations)",
)

# ---------------------------------------------------------------------------
# Categorical features (18) — shared by H1 and H2
# ---------------------------------------------------------------------------
# All integer-coded labels, confirmed in Phase 2 with their level counts.
# Key reason they are CATEGORICAL, not numerical: the codes are arbitrary
# identifiers, not magnitudes (Course codes 33 and 9991 are two degrees —
# 9991 is not "300x more" of anything; Mother's occupation 122 vs 4 are
# occupation families, not ranks).
#
# Sub-groups:
#   binary flags (8): Daytime/evening attendance, Displaced,
#     Educational special needs, Debtor, Tuition fees up to date, Gender,
#     Scholarship holder, International
#   multi-level codes (10): Marital status(6), Application mode(18),
#     Application order(8), Course(17), Previous qualification(17),
#     Nacionality(21), Mother's qualification(29), Father's qualification(34),
#     Mother's occupation(32), Father's occupation(46)
#     — numbers in parentheses = unique levels (Phase 2 verified)
#
# NOTE: `Application order` (0=first choice … 9=last choice) IS ordinal in
# meaning, but it is kept categorical for Phase 3: one-hot loses no information
# at 8 levels and avoids assuming a linear effect of rank on risk. Revisit only
# as a documented experiment, never silently.
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
)


def get_feature_types(horizon: str) -> tuple[list[str], list[str]]:
    """Return (numerical_features, categorical_features) for a horizon.

    Acts as the single conversion point between the *policy* layer
    (ml.features.feature_sets: which columns are ALLOWED) and the *plumbing*
    layer (this project: how each column is TREATED).

    Raises ValueError with a clear message if a feature list and the type
    classification ever drift apart — e.g. someone adds a new H1 feature but
    forgets to classify it here. Failing loudly is the point: a silently
    unclassified column would drop out of the pipeline (ColumnTransformer
    remainder='drop') and quietly change model inputs.
    """
    if horizon == "H1":
        allowed = H1_FEATURES
        numerical = list(NUMERICAL_FEATURES_H1)
    elif horizon == "H2":
        allowed = H2_FEATURES
        numerical = list(NUMERICAL_FEATURES_H1) + list(NUMERICAL_FEATURES_H2_ADDITIONS)
    else:
        raise ValueError(f"Unknown horizon {horizon!r}: expected 'H1' or 'H2'")

    categorical = list(CATEGORICAL_FEATURES)
    classified = numerical + categorical

    # Exact-partition check: every allowed feature classified exactly once.
    missing = [f for f in allowed if f not in classified]
    extra = [f for f in classified if f not in allowed]
    if missing or extra:
        raise ValueError(
            f"Type classification out of sync with {horizon} feature list. "
            f"Unclassified features: {missing}. Stale entries: {extra}. "
            f"Update ml/preprocessing/feature_types.py."
        )
    if len(classified) != len(set(classified)):
        raise ValueError("A feature appears in BOTH numerical and categorical lists")
    return numerical, categorical


if __name__ == "__main__":
    # Running as a script re-runs the Phase-2 horizon policy first — if the
    # upstream feature lists are broken, type classification is meaningless.
    validate()
    for h in ("H1", "H2"):
        num, cat = get_feature_types(h)
        print(f"{h}: {len(num)} numerical + {len(cat)} categorical = {len(num) + len(cat)} features")
        print(f"   numerical -> {num}")
