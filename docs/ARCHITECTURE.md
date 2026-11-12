# AI Student Academic Risk System — Architecture & Project Plan

> Phase 2 deliverable (Part A). This is a design document, not code.
> Dataset findings: `docs/DATA.md` · Leakage policy: `docs/LEAKAGE_AUDIT.md`

---

## 1. Vision

A full-stack web application where a machine learning model estimates a student's
**academic risk** (at-risk of dropping out / failing vs. on track) from student
profile data, exposed through a REST API and visualized in a web dashboard.

Primary value: demonstrate an end-to-end, honest ML system — data → training →
evaluation → model registry → API → UI — built on public, licensed data.

---

## 2. Scope & phase roadmap

| Phase | Content | Status |
|---|---|---|
| 1 | Problem framing, public dataset research, ML problem definition | ✅ done |
| 2 | Architecture doc (this file), dataset download, EDA, leakage audit, H1/H2 feature sets | ✅ done |
| 3 | Leakage-safe preprocessing pipeline + baseline data preparation | ✅ done |
| 4 | Model training & evaluation (LR, DT, KNN, NB, SVM, RF baseline) | ✅ done |
| 5 | Model registry + serialization + inference layer | ✅ done |
| 6 | Backend API (FastAPI) | ✅ done |
| 7 | Frontend dashboard (React); optional DB deferred | ✅ done |
| 8 | Testing, linting, README, final polish | pending |

Out of scope (explicitly): deep learning, live student data, authentication,
deployment to production infrastructure (local run only).

---

## 3. Tech stack

Justification: everything below pairs naturally with your studied ML topics
(scikit-learn) and is the standard lightweight path for a first serious project.

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.13 (already installed) | sklearn/pandas ecosystem |
| ML | scikit-learn 1.9 | matches studied algorithms: LR, DT, KNN, NB, SVM |
| Data handling | pandas 2.3, numpy 2.2 | already installed |
| API | FastAPI + uvicorn | typed, auto docs, easy validation via pydantic |
| Frontend | React + Vite (JS/TS) | component dashboard, fetch to API |
| Model persistence | joblib (sklearn pipeline) + JSON metadata | single-object preprocessing+model, versionable |
| Env/deps | `requirements.txt` + venv | reproducibility |
| Quality | pytest, ruff (or flake8), black | tested + linted code |

*Frontend alternative if you prefer zero build tooling: server-rendered
templates (Jinja2). Decision can be deferred to Phase 7.*

---

## 4. Repository structure (current state after Phase 2)

```
D:\ML_project
├── README.md                # orientation, reproduce commands, dataset attribution
├── requirements.txt         # Phase 2 deps (scikit-learn etc. added in Phase 3+)
├── .gitignore               # excludes data/raw, .env, model binaries
├── .env.example             # env-var template (consumed by API in Phase 6)
├── docs/
│   ├── ARCHITECTURE.md      # this file
│   ├── DATA.md              # dataset card + verified EDA findings
│   └── LEAKAGE_AUDIT.md     # per-feature H1/H2 allow/forbid table (policy)
├── data/
│   ├── README.md            # provenance: URLs, SHA256, license, immutability rules
│   ├── raw/                 # uci697/ + uci320/ — READ-ONLY, gitignored
│   └── processed/           # cleaned outputs (empty until Phase 3)
├── ml/
│   ├── eda/                 # run_eda.py → reports/eda_report.md + figs
│   ├── features/            # feature_sets.py — H1/H2/H3 lists + sanity checks
│   ├── preprocessing/       # [Phase 3 ✅] feature_types, leakage_checks,
│   │                        #   build_pipeline, prepare_data, verify
│   ├── training/            # [Phase 4 ✅] models.py (config registry),
│   │                        #   metrics.py, train.py → reports/model_report.md
│   ├── registry/            # [Phase 5 ✅] registry.py (versioned artifacts +
│   │                        #   current pointer), export.py → docs/REGISTRY.md
│   └── inference/           # [Phase 5 ✅] predict.py — validated predict_student
│                            #   (Phase 6 /predict calls exactly this)
├── backend/                 # [Phase 6 ✅] app/main.py (routes+CORS+CLI),
│                            #   app/schemas.py (card-driven payload model)
├── frontend/                # [Phase 7 ✅] Vite + React + TS dashboard (see frontend/README.md)
├── database/                # [deferred] optional prediction/history store — placeholder
├── tests/                   # [Phase 8] placeholder README only
└── reports/                 # eda_report.md + figs/*.png (10 figures)
```

Convention: **raw data is read-only.** Every transformation is a script under
`ml/` so results are reproducible from scratch (`python -m ml.eda.run_eda`).

---

## 5. ML design

### 5.1 Problem
- **Type:** multi-class classification: `Target ∈ {Dropout, Enrolled, Graduate}`
- **Product-facing derived view:** binary `at_risk = 1 if Target == Dropout else 0`
- **Primary dataset:** UCI #697 "Predict Students' Dropout and Academic Success"
  (4,424 rows, 36 features, CC BY 4.0)
- **Secondary/baseline dataset:** UCI #320 "Student Performance" (649 rows,
  CC BY 4.0) — small, clean, supports binary risk + regression; used to validate
  the pipeline before scaling up

### 5.2 Prediction horizons (leakage control)
| Horizon | Feature set | Warning time |
|---|---|---|
| **H1 — At enrollment** | demographics, socioeconomic, application path, prior qualification, parents' education, etc. — everything known at registration | earliest, hardest |
| **H2 — End of semester 1** | H1 + 1st-semester curricular-unit metrics (grade, evaluations, approved units) | later, more accurate |

Rules:
- Columns time-stamped after the horizon are dropped for that horizon.
- H1 and H2 are **separate models**, never mixed.
- Explicit leakage audit documented in `docs/DATA.md`.

### 5.3 Preprocessing pipeline (Phase 3)
Single `sklearn.pipeline.Pipeline` / `ColumnTransformer`:
- numeric: `SimpleImputer(median)` → `StandardScaler` (for SVM/KNN/LR)
- categorical: `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown="ignore")`
- class imbalance: `class_weight="balanced"` where supported; optionally compare
  with RandomOverSampler as an experiment
- all steps fitted **inside** CV folds (no pre-fit leakage)

### 5.4 Models (Phase 4)
1. DummyClassifier (majority/stratified) — the bar to beat
2. Logistic Regression (multinomial)
3. Decision Tree (with depth tuning)
4. KNN (with scaling — sensitive to scale)
5. Gaussian NB (baseline; note: assumes feature independence)
6. SVM (RBF/linear; scaled input)
7. RandomForest — stretch baseline for reference

### 5.5 Validation & metrics
- **Split:** stratified train/test (80/20, fixed random_state=42) — test set
  touched once, at the end. 80/20 matches the split recommended by the
  dataset authors (verified via UCI metadata).
- **CV:** StratifiedKFold(5) on train for model selection
- **Multi-class metrics:** macro-F1 (primary), balanced accuracy, per-class recall
- **Binary view metrics:** ROC-AUC, precision/recall for the Dropout class
- **Class imbalance:** confusion matrices reported per model; accuracy never used alone
- **Comparison table** written to `reports/model_comparison.csv`

### 5.6 Model registry (Phase 5)
```
ml/registry/
├── 2026-XX-XX_h1_logreg/       # versioned folders
│   ├── model.joblib            # full preprocessing+model pipeline
│   ├── model_card.json         # horizon, metrics, dataset DOI, feature list, sklearn version
│   └── metrics.json
└── current/                    # pointer (file listing active version) used by API
```

---

## 6. API contract (Phase 6)

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness + loaded model version |
| GET | `/model/info` | model card: metrics, horizons, feature list |
| POST | `/predict` | body = patient-style student dict → risk class + probabilities |

`/predict` response sketch:
```json
{
  "horizon": "H1",
  "prediction": "Dropout",
  "at_risk": true,
  "probabilities": {"Dropout": 0.71, "Enrolled": 0.22, "Graduate": 0.07},
  "model_version": "2026-XX-XX_h1_logreg"
}
```
Validation: pydantic schema mirrors the training feature list; unknown/missing
fields → 422, not silent defaults.

---

## 7. Frontend (Phase 7) — ✅ built
- **Stack:** Vite + React 19 + TypeScript; hand-rolled CSS design system
  (no UI/chart frameworks) — `frontend/` (details: `frontend/README.md`)
- **Input form:** all 24 H1 fields grouped into six fieldsets; the field list
  is driven live by `GET /model/info`, so the UI can never drift from backend
  validation. Coded fields use official UCI codebook labels, cross-checked
  against observed data levels
- **Result card:** risk badge, at-risk flag, probability bars, response time,
  serving model version
- **Model info page:** metric tiles, CV-vs-test + per-class tables, selected
  config & selection rule, comparison chart (snapshot of
  `reports/model_comparison.csv`), feature chips, uncertain-feature
  ablation, protocol, dataset + citation, environment/notes
- **API access:** typed client (`src/api.ts`, `VITE_API_URL` default
  `127.0.0.1:8000`); CORS restricted to the Vite dev origin (backend
  default). FastAPI 422 issue arrays surface as per-field inline errors
- Deferred: optional prediction-history database (still a placeholder)

---

## 8. Quality & conventions
- `git init` at repo root when first committing; `data/raw`, `data/processed`,
  `ml/registry/*/[!current]*` gitignored (data + binaries not committed)
- Every script runnable as `python -m ml.training.train` with CLI args
- Random seeds fixed (`random_state=42`) everywhere
- No notebook-only logic: notebooks (if any) may explore, but scripts are truth
- Tests: preprocessing shape tests, API contract tests, "model loads and
  predicts" smoke test
- License compliance: CC BY 4.0 → attribution to Cortez/Realinho et al. and
  UCI in README

---

## 9. Risks & ethics
- **Sensitive attributes** (nationality, marital status, disability, age) —
  kept for fairness auditing but flagged: model must be evaluated for group-wise
  performance disparities; results presented as *educational support tool*, never
  automated decision-making
- **Class imbalance** — accuracy will mislead; metrics section above mitigates
- **Leakage** — horizons defined up-front; G1/G2 excluded in UCI #320 experiments
- **Small datasets** — conclusions are illustrative, not institution-grade;
  stated honestly in README
- **No real student data** is used anywhere in this project
