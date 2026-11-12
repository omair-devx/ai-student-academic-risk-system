# Dataset Card & EDA Findings (Phase 2)

Full automated report: `reports/eda_report.md` · Figures: `reports/figs/` (10 PNGs)
Script: `ml/eda/run_eda.py` (rerunnable: `python -m ml.eda.run_eda`)
Provenance + SHA256 checksums: `data/README.md`
**Leakage policy (per-feature H1/H2 decisions): `docs/LEAKAGE_AUDIT.md`**
**Feature lists (code source of truth): `ml/features/feature_sets.py`**

Raw data is stored read-only under `data/raw/` (zips kept for provenance).

---

## 1. UCI #697 — Predict Students' Dropout and Academic Success (PRIMARY)

| Attribute | Verified value |
|---|---|
| Source | UCI ML Repository, DOI 10.24432/C5MC89, donated 2021 |
| License | CC BY 4.0 (attribution required) |
| File | `data/raw/uci697/data.csv` — **semicolon-separated**, `utf-8-sig`, header contains a stray tab in `Daytime/evening attendance` (cleaned on load) |
| Shape | **4,424 rows × 37 columns** (36 features + `Target`) |
| dtypes | 29 int64, 7 float64, 1 object (Target) |
| Missing values | **0** (confirmed) |
| Duplicate rows | **0** (confirmed) |
| Target | `Graduate` 2,209 (49.9%) · `Dropout` 1,421 (32.1%) · `Enrolled` 794 (17.9%) |
| Imbalance ratio | 2.78 : 1 (moderate — accuracy alone is misleading) |

### Feature structure by prediction horizon
| Horizon | # Features | Contents |
|---|---|---|
| **H1 — at enrollment** | **24** | demographics (age, gender, marital status, nationality), application path, prior qualification + grade, parents' qualification/occupation, admission grade, socioeconomic flags (Debtor, Tuition fees up to date, Displaced, Scholarship holder), macro (Unemployment rate, Inflation rate, GDP) |
| **H2 — end of semester 1** | **30** | H1 + 6 `Curricular units 1st sem (*)` columns |
| **H3 — end of semester 2** (optional/reference) | **36** | H2 + 6 `Curricular units 2nd sem (*)` columns |

### Key EDA signals (correlation with `at_risk = Dropout`)
- Strongest protective: `Tuition fees up to date` (−0.43), `1st sem (approved)` (−0.48), `1st sem (grade)` (−0.48), `2nd sem (approved/grade)` (−0.57)
- Strongest risk: `Age at enrollment` (+0.25), `Debtor` (+0.23), `Gender` (+0.20), `Application mode` (+0.20)
- Interpretation: financial status and first-semester progress dominate → matches the dataset's stated purpose (early identification of at-risk students)

### Additional EDA observations (Phase 2, full run)
- **`Age at enrollment` is heavily right-skewed**: min 17, median 20, max **70**
  (46 unique values; IQR flags 441 rows ≈ 10% as outliers). Older students are
  real, rare cases → **kept, not removed**; tree models are robust to them and
  removing them would bias the model. Revisit in Phase 3 with a documented decision.
- **Rare binary categories**: `Educational special needs` ≈ 1.2% of rows,
  `International` ≈ 2.5% → one-hot encoding is fine, but expect very small
  cell counts for fairness/segment metrics on these groups.
- **Bounded grades flag many IQR "outliers"** (e.g. `1st sem (grade)` 16.4%)
  — an artifact of the IQR fence on a bounded, peaked 0–20 scale, NOT data
  errors. Illustrates why IQR flags ≠ deletion orders.
- **Multicollinearity**: 1st-semester vs 2nd-semester columns are strongly
  correlated (same phenomenon measured twice) — expected; relevant for
  logistic regression / SVM variance, not for trees.
- Integer-coded categoricals confirmed: 18 columns (e.g. `Course` 17 levels,
  `Father's occupation` 46 levels) → must be one-hot encoded, never treated
  as continuous numbers.

### Sensitive attributes (kept for fairness auditing, flagged in model card)
`Nacionality`, `Gender`, `Marital status`, `Age at enrollment`, `Educational special needs`, `International`

### Verified encodings (source: UCI official API variable table, checked 2026-09-29)
- `Gender`: **1 = male, 0 = female** → the +0.20 correlation with at-risk means
  male students show higher dropout rates in this dataset (64.8% female / 35.2% male overall)
- `Daytime/evening attendance`: **1 = daytime, 0 = evening** → 89.1% daytime students
- `Displaced` / `Educational special needs` / `Debtor` / `Tuition fees up to date` /
  `Scholarship holder` / `International`: 1 = yes, 0 = no
- `Target` description (official): outcome "at the end of the normal duration of the course"
- Authors' **recommended split: 80% train / 20% test** — adopted as the project default
- UCI doc quirk found: the description of `Curricular units 2nd sem (without evaluations)`
  says "in the 1st semester" — evidently a copy-paste error in UCI's metadata
  (column name is unambiguous); no impact on usage, recorded for honesty.

### Still unverified
- Measurement *dates* for `Displaced`, `Debtor`, `Tuition fees up to date`,
  `Scholarship holder` — UCI documents values (yes/no) but not when they were
  recorded → the ⚠️ caveats in `docs/LEAKAGE_AUDIT.md` §3 stand, with a planned
  Phase-4 ablation test.

---

## 2. UCI #320 — Student Performance (SECONDARY / baseline)

| Attribute | Verified value |
|---|---|
| Source | UCI ML Repository, DOI 10.24432/C5TG7T, Cortez & Silva 2008 |
| License | CC BY 4.0 |
| Files | `data/raw/uci320/student/student-por.csv`, `student-mat.csv` (semicolon-separated) |
| Shape | **Portuguese: 649 × 33** · **Mathematics: 395 × 33** ← row count now verified (was unverified in Phase 1) |
| Missing values | **0** in both files |
| Duplicate rows | **0** |
| Target | `G3` final grade 0–20; risk view = `fail if G3 < 10` → **100 fail (15.4%) vs 549 pass (84.6%)** — strongly imbalanced |
| Categorical | 17 text columns (school, sex, address, jobs, reason, guardian, yes/no flags…) |

### Leakage check (quantified)
- `r(G2, G3) = +0.919`, `r(G1, G3) = +0.826` → **G1/G2 excluded from all risk features**
- Legitimate early-warning features: `failures`, `absences`, `studytime`, `Medu/Fedu`, family/lifestyle self-reports
- Risk modeling plan: binary `G3 < 10` classification (primary) + optional `G3` regression

---

## 3. Decisions locked in for Phase 3+

1. **Primary modeling target:** UCI #697 multi-class `Target`, with derived binary `at_risk`
2. **Two horizons built as separate models:** H1 (24 feats) and H2 (30 feats); H3 only as a reference ceiling
3. **UCI #320** used to validate the pipeline end-to-end first (small, fast), with G1/G2 excluded
4. **Preprocessing:** `ColumnTransformer` — OneHot for text categoricals; `StandardScaler` for continuous; tree-safe variants (no scaling) for DT/RF. Integer-coded categoricals (e.g. `Course`, `Marital status`) → treated as categorical, not continuous (heuristic flagged in EDA)
5. **Metrics:** macro-F1 primary; balanced accuracy; binary ROC-AUC on `at_risk`; confusion matrices always reported
6. **Attribution:** README must credit Realinho et al. (2021) / UCI / Cortez & Silva (2008), CC BY 4.0

### Known caveats carried forward
- `Unemployment rate` / `Inflation rate` / `GDP` are national macro values (few unique levels) — low signal, kept in H1 but monitored via feature importance
- EDA "categorical-coded" heuristic (≤25 unique values) mislabels macro variables and some count columns; the real categorical list is set explicitly in the Phase 3 feature config, not by heuristic
- Correlation ≠ predictive power (nonlinear models may differ); model comparison in Phase 4 settles it
