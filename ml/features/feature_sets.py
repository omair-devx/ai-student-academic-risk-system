"""Feature-set definitions for the Student Academic Risk System.

This module is the SINGLE SOURCE OF TRUTH for which columns may be used at
each prediction horizon. Later phases (preprocessing, training, API) must
IMPORT these lists instead of re-typing column names — that way a change to
the leakage policy automatically propagates everywhere.

Prediction horizons (defined in Phase 1, audited in docs/LEAKAGE_AUDIT.md):

  H1 — "at enrollment":   features known the moment the student registers.
                          Earliest warning, hardest prediction.
  H2 — "after semester 1": H1 + what the institution knows once semester 1
                          grades are in. Later warning, easier prediction.
  H3 — "after semester 2": reference ceiling ONLY — semester-2 data arrives so
                          close to the final outcome that it is not a useful
                          warning point. Kept for upper-bound experiments.

Run this file directly to print the sets and run the built-in sanity checks:
    python -m ml.features.feature_sets
"""

# ---------------------------------------------------------------------------
# UCI #697 (primary dataset) — column groups
# ---------------------------------------------------------------------------

# Semester-2 curricular columns. NOT usable in H1 or H2: they are measured
# after semester 2, i.e. almost at the moment the final outcome is decided.
# Using them for an "early warning" model would make the warning meaningless.
SEMESTER_2_COLUMNS = (
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
)

# Semester-1 curricular columns. For H1 they DO NOT exist yet (the semester
# hasn't happened), so they are forbidden there. For H2 they are the core
# signal: after semester 1 the institution genuinely has these numbers.
SEMESTER_1_COLUMNS = (
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
)

# Everything available at enrollment. Grouped by theme so the reasoning for
# each inclusion is visible where the names are (full per-feature audit with
# caveats lives in docs/LEAKAGE_AUDIT.md):
H1_FEATURES = [
    # --- demographics known from the application record ---
    "Marital status",
    "Gender",
    "Age at enrollment",        # name itself anchors the value to enrollment time
    "Nacionality",
    "International",
    # --- application / enrollment choices made before day 1 ---
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    # --- pre-existing academic history (predates this institution) ---
    "Previous qualification",
    "Previous qualification (grade)",
    "Admission grade",          # entry metric used to ADMIT the student
    # --- family background (fixed at enrollment) ---
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    # --- status flags at enrollment.
    #     CAVEAT: UCI does not document the exact measurement date for these
    #     four, so they *could* in principle be updated during the year.
    #     They are kept in H1 (standard practice in papers using this dataset)
    #     but flagged in the leakage audit for a Phase-4 ablation check.
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Scholarship holder",
    # --- macro-economic context (published indicators, no student info) ---
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

# H2 = everything valid at enrollment + what semester 1 reveals.
# Built by concatenation so H2 can never silently DIVERGE from H1: if an H1
# feature is added later, H2 inherits it automatically.
H2_FEATURES = H1_FEATURES + list(SEMESTER_1_COLUMNS)

# H3 = H2 + semester 2 columns. Reference ceiling only — see module docstring.
# Kept explicit so an experiment can quantify "how much do we gain by waiting?"
H3_FEATURES = H2_FEATURES + list(SEMESTER_2_COLUMNS)

# The label. Never a feature — listed here so code can reference it by name
# instead of hardcoding the string "Target" in multiple files.
TARGET_COLUMN = "Target"

# Multi-class label values exactly as they appear in data.csv.
# "Dropout" is also collapsed into the binary product view `at_risk = 1`.
TARGET_CLASSES = ("Dropout", "Enrolled", "Graduate")


# ---------------------------------------------------------------------------
# Ablation configuration for the four timing-ambiguous H1 features (Phase 3)
# ---------------------------------------------------------------------------
# Phase 2 could NOT verify WHEN these four fields were measured (UCI documents
# their values but not their measurement dates — see docs/LEAKAGE_AUDIT.md §3).
# Instead of silently trusting or silently dropping them, we keep BOTH variants
# so Phase 4 can measure the impact:
#
#   Experiment A = H1 as approved (24 features, uncertain four INCLUDED)
#   Experiment B = H1 without the uncertain four (20 features)
#
# If A ≈ B, the timing ambiguity is immaterial and A can be used with a note.
# If A >> B, the four features carry real information — then we must judge
# whether it is legitimately available at enrollment time before using it.
UNCERTAIN_TIMING_FEATURES = (
    "Displaced",
    "Debtor",
    "Tuition fees up to date",
    "Scholarship holder",
)

# Experiment A: the full approved H1 list (uncertain features kept).
EXPERIMENT_A_H1 = tuple(H1_FEATURES)

# Experiment B: H1 minus the uncertain four (order preserved so comparisons
# against A stay readable and deterministic).
EXPERIMENT_B_H1 = tuple(
    f for f in H1_FEATURES if f not in UNCERTAIN_TIMING_FEATURES
)

# Experiment registry resolved by prepare_data.py — a typo like experiment="C"
# fails loudly instead of silently falling back to a default.
H1_EXPERIMENTS = {
    "A": EXPERIMENT_A_H1,
    "B": EXPERIMENT_B_H1,
}


# ---------------------------------------------------------------------------
# UCI #320 (comparison dataset) — feature decisions
# ---------------------------------------------------------------------------

# UCI #320 predicts the final grade G3 of the SAME course the predictors
# describe. Two features would give the model the answer:
#   r(G2, G3) = 0.919 and r(G1, G3) = 0.826  (measured in EDA, Phase 2)
# G1/G2 are period grades of the course whose final grade is G3 — that is
# near-perfect target leakage, so they are excluded from every risk experiment.
UCI320_EXCLUDED_FEATURES = ("G1", "G2")

# Risk view for the comparison dataset: fail if final grade < 10 (Portuguese
# grading scale, 0-20; the dataset's own papers use the same threshold).
UCI320_TARGET_COLUMN = "G3"
UCI320_PASS_THRESHOLD = 10


# ---------------------------------------------------------------------------
# Sanity checks — also runnable as `python -m ml.features.feature_sets`
# ---------------------------------------------------------------------------

def validate() -> None:
    """Assert the horizon and ablation invariants.

    These encode the leakage policy as executable checks: if someone later
    edits a feature list incorrectly, this fails loudly instead of letting a
    leaked column quietly reach training. Called both as a script and by
    ml.preprocessing.leakage_checks (Phase 3) — one definition, no duplication.
    """
    # 1. A horizon's feature list must contain no duplicates (a duplicated
    #    column would silently double its weight during one-hot encoding).
    for name, feats in [("H1", H1_FEATURES), ("H2", H2_FEATURES), ("H3", H3_FEATURES)]:
        assert len(feats) == len(set(feats)), f"{name} contains duplicate columns"

    # 2. Horizon nesting: H1 ⊂ H2 ⊂ H3 — later horizons only ADD information.
    assert set(H1_FEATURES) < set(H2_FEATURES) < set(H3_FEATURES)

    # 3. THE leakage rule: no semester data may appear at enrollment time.
    assert not set(H1_FEATURES) & set(SEMESTER_1_COLUMNS), "H1 must not use semester-1 data"
    assert not set(H1_FEATURES) & set(SEMESTER_2_COLUMNS), "H1 must not use semester-2 data"

    # 4. H2 may use semester 1 but never semester 2.
    assert not set(H2_FEATURES) & set(SEMESTER_2_COLUMNS), "H2 must not use semester-2 data"

    # 5. The label must never be part of a feature list.
    assert TARGET_COLUMN not in set(H1_FEATURES) | set(H2_FEATURES) | set(H3_FEATURES)

    # 6. Comparison-dataset leak columns must be excluded by definition.
    assert "G1" in UCI320_EXCLUDED_FEATURES and "G2" in UCI320_EXCLUDED_FEATURES

    # 7. Ablation invariants (Phase 3): Experiment A must be exactly the full
    #    approved H1, and Experiment B exactly A minus the four uncertain
    #    features — nothing more, nothing less. Catches accidental edits that
    #    would quietly change what an ablation actually compares.
    assert EXPERIMENT_A_H1 == tuple(H1_FEATURES), "Experiment A must equal full H1"
    assert set(EXPERIMENT_B_H1) == set(H1_FEATURES) - set(UNCERTAIN_TIMING_FEATURES), \
        "Experiment B must be H1 minus exactly the uncertain-timing features"
    assert len(UNCERTAIN_TIMING_FEATURES) == 4, "documented ablation covers 4 features"
    # The uncertain features still belong to H2 — the ablation is H1-only.
    assert set(UNCERTAIN_TIMING_FEATURES) <= set(H2_FEATURES)


if __name__ == "__main__":
    validate()  # raises AssertionError if a horizon rule is violated
    print("All horizon sanity checks passed.\n")
    print(f"H1 (at enrollment):        {len(H1_FEATURES)} features")
    for f in H1_FEATURES:
        print(f"  - {f}")
    print(f"\nH2 (after semester 1):     {len(H2_FEATURES)} features "
          f"(H1 + {len(SEMESTER_1_COLUMNS)} semester-1 columns)")
    for f in SEMESTER_1_COLUMNS:
        print(f"  - {f}")
    print(f"\nH3 (after semester 2, reference only): {len(H3_FEATURES)} features")
    print(f"\nAblation: Experiment A={len(EXPERIMENT_A_H1)} features (incl. uncertain), "
          f"Experiment B={len(EXPERIMENT_B_H1)} features "
          f"(excl. {UNCERTAIN_TIMING_FEATURES})")
    print(f"\nTarget: {TARGET_COLUMN} -> {TARGET_CLASSES}")
    print(f"UCI #320 excluded (leakage): {UCI320_EXCLUDED_FEATURES}")
