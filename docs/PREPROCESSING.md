# Preprocessing Architecture (Phase 3)

Covers: preprocessing design, H1/H2 pipelines, feature treatment, encoding,
scaling, split/CV strategy, class imbalance, leakage prevention, ablation plan.

Statements are labeled to keep three categories apart:
- **[VERIFIED]** — measured from the local data or official UCI documentation
- **[DESIGN]** — a decision we made (with reasoning), not a fact about data
- **[FUTURE]** — planned experiment, NOT yet performed

Executable verification: `python -m ml.preprocessing.verify` → `reports/preprocessing_report.md`.

---

## 1. Architecture overview **[DESIGN]**

```
raw CSV (read-only)
   │  ml/eda/run_eda.load_delimited  (Phase-2 loader, reused)
   ▼
feature selection (ml/features/feature_sets.py — the policy: WHICH columns)
   │  ml/preprocessing/prepare_data.prepare_horizon
   ▼
stratified 80/20 split  (rows only — nothing is fitted yet)
   │
   ├── X_train ──► ColumnTransformer.fit_transform(...)   ← the ONLY allowed fit
   └── X_test  ──► ColumnTransformer.transform(...)       ← transform only
                         │
                         ▼
              dense feature matrix  ──► (Phase 4: model)
```

One `ColumnTransformer` per experiment, built by
`ml/preprocessing/build_pipeline.build_preprocessor(...)`:

| Branch | Steps | Why |
|---|---|---|
| numerical | `SimpleImputer(median)` → optional `StandardScaler` | median is robust to the skew seen in Phase-2 EDA; scaler only where algorithms need it |
| categorical | `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown="ignore")` | labels preserved as levels; unknowns safe at inference |

- `remainder="drop"`: any column not explicitly listed physically cannot enter
  the matrix — an unexpected feature becomes impossible rather than silent.
- `get_feature_names_out()` recovers names (`num__Age at enrollment`,
  `cat__Course_9003`) for reports and future explainability.
- Fit happens **inside** `Pipeline([("preprocess", ...), ("model", ...)])`, so
  Phase-4 cross-validation refits preprocessing per fold automatically.

**Never** encode train and test separately — the fitted train transformer is
the only transformer applied to test/future rows.

---

## 2. H1 pipeline **[DESIGN]**

- Input: 24 features (`H1_FEATURES`, Experiment A) or 20 (Experiment B, §9)
- 6 numerical + 18 categorical → **[VERIFIED]** output width **239** columns
  (train-fitted one-hot levels only; rare categories absent from train are
  handled as unknowns later)
- Purpose: earliest possible risk warning at enrollment

## 3. H2 pipeline **[DESIGN]**

- Input: 30 features (`H2_FEATURES` = H1 + 6 semester-1 columns)
- 12 numerical + 18 categorical → **[VERIFIED]** output width **245**
- Same architecture as H1 (only the feature list changes) — comparisons
  between horizons therefore reflect *information*, not pipeline differences

---

## 4. Numerical feature treatment **[VERIFIED classification, DESIGN treatment]**

| Feature | Type | Treatment | Reason |
|---|---|---|---|
| Previous qualification (grade) | numerical | median impute → scale | continuous 0–200 scale |
| Admission grade | numerical | median impute → scale | continuous 0–200 scale |
| Age at enrollment | numerical | median impute → scale | integer years, but a quantity (mean/σ meaningful) |
| Unemployment rate / Inflation rate / GDP | numerical | median impute → scale | macro indicators, continuous |
| 6× `Curricular units 1st sem (...)` (H2 only) | numerical | median impute → scale | counts and 0–20 grade averages |

**[VERIFIED]** no missing values exist today — imputers are a safety net for
refreshed data; their statistics still come from train only.

## 5. Categorical feature treatment **[VERIFIED classification]**

18 integer-coded columns are categorical because the codes are **labels, not
magnitudes** (Course `9070` ≠ "more" than Course `33`; occupation `122` is a
job family, not a rank):

- binary flags (8): attendance, Displaced, special needs, Debtor, Tuition,
  Gender, Scholarship, International
- multi-level codes (10): Marital status(6), Application mode(18),
  Application order(8), Course(17), Previous qualification(17),
  Nacionality(21), M qualification(29), F qualification(34),
  M occupation(32), F occupation(46) — levels **[VERIFIED]** in Phase 2

**[DESIGN]** `Application order` is ordinal in meaning but treated as nominal:
one-hot loses nothing at 8 levels and avoids assuming a linear risk effect of
application rank.

## 6. Encoding strategy **[DESIGN]**

`OneHotEncoder(handle_unknown="ignore", sparse_output=False)`:

- **Safe for unseen categories** — the Phase-6 API will see codes absent from
  training; they map to an all-zero block instead of raising
  (**[VERIFIED]** by the injected `Course=99999` check in the report)
- **No target leakage** — encoding uses only column identity, never labels.
  Target encoding is explicitly banned at this stage (it derives feature
  values from `y`; if ever used later, it must be cross-fitted)
- **Consistent representation** — the same fitted encoder produces train,
  test, and inference matrices
- Note: binary flags expand to two columns (0 and 1). Slightly redundant for
  regularized models; harmless, kept for simplicity and uniformity

## 7. Scaling strategy **[DESIGN]**

| Variant | For | Steps |
|---|---|---|
| `scale_numeric=True` | Logistic Regression, KNN, SVM | impute → `StandardScaler` |
| `scale_numeric=False` | Decision Tree, (Random Forest), Naive Bayes | impute only |

- Mixed scales (age 17–70, grades 95–190, GDP ≈ ±4) distort distance/margin
  algorithms → scaling required there
- Trees split on *order*, which scaling never changes; GaussianNB estimates
  per-feature mean/variance → affine-invariant
- One-hot columns stay 0/1 by design (scaling them adds nothing)
- **No separate scaled datasets are saved** — both variants are pipeline
  configurations over identical rows (Part F rule)

## 8. Train/test & cross-validation strategy

- **[VERIFIED]** split: stratified **80/20**, `random_state=42` →
  3,539 train / 885 test rows; class proportions match full data within
  0.1 pp (report §3). 80/20 matches the dataset authors' recommended split
- **[DESIGN]** test set is created once and never fits anything; proven is by
  the scaler-mean check (`scaler.mean_ == train means`, gap to test means
  ≈ 0.51)
- **[DESIGN]** Phase-4 CV: `StratifiedKFold(n_splits=5, shuffle=True,
  random_state=42)` via `prepare_data.make_stratified_kfold()`
  — configured now, **not run** in Phase 3
- **[DESIGN]** CV must score the *full* pipeline so each fold refits
  preprocessing on its own training rows; pre-fitting a preprocessor on all
  data before CV is banned (validation-fold statistics would leak)

## 9. Class imbalance strategy **[VERIFIED measurement, DESIGN plan]**

**[VERIFIED]** full data: Dropout 1,421 (32.1%) · Enrolled 794 (17.9%) ·
Graduate 2,209 (49.9%); splits preserve this (report §3).

**[VERIFIED]** majority-class baseline: always predicting `Graduate` →
**49.9% accuracy**.

**[DESIGN]** Phase 4:
- primary metrics: macro-F1, balanced accuracy, per-class recall (+ ROC-AUC
  later for the binary view) — accuracy alone is prohibited as a headline
- class-weight support: algorithms that expose `class_weight` will be tested
  with weights (no data alteration)
- **[FUTURE]** resampling (SMOTE/oversampling) only as a controlled,
  isolated experiment — never applied to the full dataset before splitting

## 10. Leakage prevention **[DESIGN, executable]**

Checks live in `ml/preprocessing/leakage_checks.py` and run on every
`prepare_horizon()` / verification call (report §9 lists all). Highlights:

1. target-in-X rejected; exact approved column set enforced; duplicates rejected
2. H1 physically cannot contain semester columns; Phase-2 `validate()` re-run
3. target values must equal exactly {Dropout, Enrolled, Graduate}
4. stratification drift > 1 pp and train/test index overlap rejected
5. scaler fit provenance: `scaler.mean_` must equal training means
6. unseen categories must transform without error/NaN

## 11. Four-feature ablation plan **[FUTURE]**

**[VERIFIED gap]** UCI does not document *when* `Displaced`, `Debtor`,
`Tuition fees up to date`, `Scholarship holder` were measured
(docs/LEAKAGE_AUDIT.md §3).

**[DESIGN]** configuration (no training yet):

| Experiment | Config constant | Features |
|---|---|---|
| A | `EXPERIMENT_A_H1` | 24 — uncertain four **included** |
| B | `EXPERIMENT_B_H1` | 20 — uncertain four excluded |

Access: `prepare_horizon("H1", experiment="A"|"B")`. Both variants verified
end-to-end (report §7). **[FUTURE]** Phase-4 comparison: if A ≈ B the timing
ambiguity is immaterial; if A ≫ B, legitimacy of those features at enrollment
must be argued before adoption.

---

## 12. Reproducing this phase

```powershell
python -m ml.features.feature_sets      # horizon + ablation policy checks
python -m ml.preprocessing.feature_types # type classification checks
python -m ml.preprocessing.verify        # full 18-point verification + report
```

Outputs: `reports/preprocessing_report.md`, `data/processed/uci697/*.csv`
(6 regenerable split files; raw data untouched).
