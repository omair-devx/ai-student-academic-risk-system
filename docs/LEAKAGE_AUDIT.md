# Leakage Audit — UCI #697 (Phase 2, Part D)

Purpose: decide, **feature by feature**, what information is legitimately
available at each prediction point. A model that trains on information that
did not exist at prediction time produces *leaked* scores that collapse in
production — this document is the policy that prevents that.

Method: every column in `data/raw/uci697/data.csv` was checked against
(a) the variable descriptions published on the official UCI dataset page,
(b) the meaning of the prediction moment (end of normal course duration),
and (c) EDA measurements (correlations) for columns whose timing is unclear.

Machine-readable version of these decisions: `ml/features/feature_sets.py`
(with executable sanity checks). **Edit both together.**

Legend: ✅ allowed · ❌ forbidden · ⚠️ allowed with documented caveat

---

## 1. Full feature audit table (all 36 features + target)

| # | Feature | H1 (at enrollment) | H2 (after sem 1) | Reason |
|---|---|---|---|---|
| 1 | Marital status | ✅ | ✅ | Demographic from the application record; fixed before day 1 |
| 2 | Application mode | ✅ | ✅ | How the student applied — defined at application time |
| 3 | Application order | ✅ | ✅ | Course preference order submitted with the application |
| 4 | Course | ✅ | ✅ | Degree chosen at enrollment |
| 5 | Daytime/evening attendance | ✅ | ✅ | Study regime chosen at enrollment |
| 6 | Previous qualification | ✅ | ✅ | Credential held *before* applying to this institution |
| 7 | Previous qualification (grade) | ✅ | ✅ | Grade of that prior credential — predates enrollment |
| 8 | Nacionality | ✅ | ✅ | Demographic, known at application |
| 9 | Mother's qualification | ✅ | ✅ | Family background, fixed at enrollment |
| 10 | Father's qualification | ✅ | ✅ | Family background, fixed at enrollment |
| 11 | Mother's occupation | ✅ | ✅ | Family background, reported on application |
| 12 | Father's occupation | ✅ | ✅ | Family background, reported on application |
| 13 | Admission grade | ✅ | ✅ | Metric used to **admit** the student — computed before enrollment (distinct from in-course grades) |
| 14 | Displaced | ⚠️ | ✅ | Status from the application record → allowed in H1. *Caveat:* UCI does not document the measurement date; see §3 |
| 15 | Educational special needs | ✅ | ✅ | Declared at registration |
| 16 | Debtor | ⚠️ | ✅ | Financial status flag → allowed in H1. *Caveat:* debt can arise during the year and UCI documents no measurement date; see §3 |
| 17 | Tuition fees up to date | ⚠️ | ✅ | Payment status → allowed in H1. *Caveat:* same timing ambiguity as `Debtor`; see §3 |
| 18 | Gender | ✅ | ✅ | Demographic, known at application |
| 19 | Scholarship holder | ⚠️ | ✅ | Award status → allowed in H1. *Caveat:* scholarships can be granted mid-year; measurement date undocumented |
| 20 | Age at enrollment | ✅ | ✅ | Column name explicitly anchors the value to enrollment time |
| 21 | International | ✅ | ✅ | Residency status known at application |
| 22 | Unemployment rate | ✅ | ✅ | Published macro indicator (country-level, not student-level); available at any time |
| 23 | Inflation rate | ✅ | ✅ | Same as above |
| 24 | GDP | ✅ | ✅ | Same as above |
| 25 | Curricular units 1st sem (credited) | ❌ | ✅ | Does not exist before semester 1 ends |
| 26 | Curricular units 1st sem (enrolled) | ❌ | ✅ | Does not exist before semester 1 ends |
| 27 | Curricular units 1st sem (evaluations) | ❌ | ✅ | Participation data accumulated *during* semester 1 |
| 28 | Curricular units 1st sem (approved) | ❌ | ✅ | Pass counts known only after semester-1 assessment |
| 29 | Curricular units 1st sem (grade) | ❌ | ✅ | Semester-1 grades — the strongest legitimate H2 signal (r with `at_risk` ≈ −0.48) |
| 30 | Curricular units 1st sem (without evaluations) | ❌ | ✅ | Does not exist before semester 1 ends |
| 31 | Curricular units 2nd sem (credited) | ❌ | ❌ | Only known after semester 2 — far too late for an early warning (H3 reference only) |
| 32 | Curricular units 2nd sem (enrolled) | ❌ | ❌ | Same as #31 |
| 33 | Curricular units 2nd sem (evaluations) | ❌ | ❌ | Same as #31 |
| 34 | Curricular units 2nd sem (approved) | ❌ | ❌ | Same as #31 (r with `at_risk` ≈ −0.57 — strong but **too late to be a warning**) |
| 35 | Curricular units 2nd sem (grade) | ❌ | ❌ | Same as #31 (r with `at_risk` ≈ −0.57) |
| 36 | Curricular units 2nd sem (without evaluations) | ❌ | ❌ | Same as #31 |
| — | `Target` (label) | ❌ | ❌ | The outcome itself — never a feature (asserted in `feature_sets.validate()`) |

---

## 2. Feature groups by decision

### 2.1 Semester-related features (rows 25–36) — the core of this audit
- **H1**: all 12 semester columns forbidden. At enrollment, no curricular
  outcome for this enrollment can exist. Any of them in an H1 pipeline is
  leakage by construction, regardless of how predictive it looks.
- **H2**: the 6 semester-1 columns are allowed — this is exactly the
  information an advisor would have at the semester-1 grade release.
- **Never in H1/H2**: the 6 semester-2 columns. They are not "leaked" in the
  statistical sense for an H3 model, but including them in an "early warning"
  system would make the warning pointless — the year is essentially over.
  Kept as H3 reference ceiling to *quantify* the value of predicting early.

### 2.2 Academic performance variables
- `Admission grade` (H1 ✅): entry exam/metric used for admission — measured
  **before** enrollment. Distinct from in-course performance.
- In-course grades appear only as semester columns → governed by §2.1.
- The strongest correlations with the target are all semester-2 grades
  (≈ −0.57) and tuition status (≈ −0.43) — high correlation was treated as a
  **reason to scrutinize timing, not as a reason to include** the feature.

### 2.3 Outcome-revealing fields
- Only `Target` directly reveals the outcome; it is excluded by definition
  and by assertion in code.
- UCI #697 contains **no** "unregistration date" or "final exam score" style
  columns (unlike some other educational datasets), so no additional direct
  outcome proxies exist. *Verified against the full 37-column header.*

---

## 3. ⚠️ Timing-ambiguity features (the honest part)

UCI's variable table gives descriptions like "student's debt situation" but
**does not document when these four fields were measured**:

`Displaced`, `Debtor`, `Tuition fees up to date`, `Scholarship holder`

Risk: if the snapshot was taken at data-extraction time (end of year) rather
than at enrollment, H1 would contain post-enrollment information.
**This could not be verified from the dataset documentation alone.**

Decision (revisit in Phase 4):
1. Keep all four in H1 — they are standard enrollment attributes in published
   work with this dataset, and removing them would weaken the primary model
   on an unconfirmed suspicion.
2. Run an **ablation experiment**: H1 model *with* vs. *without* these four
   features. If metrics barely change, the ambiguity is immaterial.
3. Document the outcome in the model card.

---

## 4. Comparison dataset UCI #320 — permanent exclusions

| Feature | H1 | H2 | Reason |
|---|---|---|---|
| `G1` (1st-period grade) | ❌ | ❌ | Correlation with target `G3` = **+0.826** — same course, earlier period. Near-perfect leakage for a final-grade prediction. |
| `G2` (2nd-period grade) | ❌ | ❌ | Correlation with target `G3` = **+0.919** — same reasoning, even stronger. |
| `G3` (final grade) | ❌ | ❌ | This is the label (risk view: fail if `< 10`). |

`failures`, `absences`, `studytime`, family/lifestyle self-reports remain
allowed: they are legitimate early-warning signals that exist from day 1.
(Relationships are weak-to-moderate, as expected for honest predictors.)

---

## 5. Audit summary

| Horizon | Allowed features | Forbidden |
|---|---|---|
| H1 (at enrollment) | 24 (4 with ⚠️ caveat) | 12 semester columns + `Target` |
| H2 (after semester 1) | 30 (H1 + 6 sem-1) | 6 sem-2 columns + `Target` |
| H3 (reference only) | 36 | `Target` |
| UCI #320 risk model | all except G1/G2/G3 | G1, G2 (measured leakage), G3 (label) |

**Rules going forward:** feature lists come from `ml/features/feature_sets.py`
only; the `validate()` assertions are re-run by
`ml.preprocessing.leakage_checks` on every data load (and will be covered by
tests in Phase 8). Any new feature must be added to this table first.
