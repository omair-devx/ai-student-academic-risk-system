# AI Student Academic Risk System

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-5.9-3178C6?logo=typescript&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-7.1-646CFF?logo=vite&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

Predicts student academic risk (**Dropout / Enrolled / Graduate**) from a public
UCI dataset, using a scikit-learn pipeline served through a FastAPI backend and
an interactive React dashboard.

## Highlights

- **Leakage-safe by construction.** Every candidate feature passes an explicit
  per-horizon allow/deny audit (`docs/LEAKAGE_AUDIT.md`) before it reaches a
  pipeline — target-derived and post-cutoff variables are rejected mechanically,
  not by judgement.
- **Honest evaluation protocol.** 15 configs were pre-declared, selected on
  5-fold CV only, and evaluated on the held-out test split exactly once. Full
  protocol fixed before any result was seen: `reports/model_report.md`.
- **Script-based ML, not notebooks.** Every stage is a reviewable, rerunnable,
  testable Python module (`ml/eda`, `ml/preprocessing`, `ml/training`,
  `ml/registry`, `ml/inference`).
- **Versioned model registry.** Each export produces a self-contained artefact
  folder (model + metrics + model card) with an explicit `current` pointer —
  rollbacks are a one-line change.
- **One pipeline object, trained once, served everywhere.** The API never
  refits or reimplements preprocessing; it calls the exact fitted pipeline from
  the registry.

## Results

Two prediction horizons, both selected by CV and confirmed on a single terminal
test pass:

| Horizon | Features | CV-selected winner | CV macro-F1 | **Test macro-F1** | ROC-AUC (Dropout) |
|---|---|---|---|---|---|
| **H1** — predict from semester 1 | 24 | `svm_rbf_balanced` | 0.580 | **0.546** | 0.813 |
| **H2** — predict from semester 2 | 30 | `random_forest_balanced` | 0.682 | **0.650** | 0.890 |

Baselines for context: a majority-class dummy scores **0.222** macro-F1 on both
horizons. The +0.104 gain from H1 → H2 is the measurable information value of
second-semester data.

> Full tables, per-class recall and the transparency note on CV-vs-test ranking
> are in [`reports/model_report.md`](reports/model_report.md).

## Quick start

```powershell
# 1. clone
git clone https://github.com/omair-devx/ai-student-academic-risk-system.git
cd ai-student-academic-risk-system

# 2. Python deps (Python 3.13)
pip install -r requirements.txt

# 3. RESTORE THE RAW DATASETS — required. data/raw/ is gitignored and there is
#    deliberately no downloader script; fetch the official zips and unpack them.
#    URLs, SHA256 checksums and expected shapes: data/README.md
New-Item -ItemType Directory -Force data\raw\uci697, data\raw\uci320\student | Out-Null
Invoke-WebRequest "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip" -OutFile data\raw\uci697\uci697.zip
Invoke-WebRequest "https://archive.ics.uci.edu/static/public/320/student+performance.zip"                            -OutFile data\raw\uci320\uci320.zip
Expand-Archive data\raw\uci697\uci697.zip -DestinationPath data\raw\uci697 -Force
Expand-Archive data\raw\uci320\uci320.zip -DestinationPath data\raw\uci320 -Force
Expand-Archive data\raw\uci320\student.zip -DestinationPath data\raw\uci320\student -Force

# 4. validate raw data + build data/processed/ (6 split files)
python -m ml.eda.run_eda              # → reports/eda_report.md + reports/figs/
python -m ml.preprocessing.verify     # → reports/preprocessing_report.md + data/processed/uci697/*.csv

# 5. API → http://127.0.0.1:8000/docs
python -m backend.app.main

# 6. dashboard, in a SECOND terminal → http://localhost:5173
cd frontend
npm install
npm run dev
```

Expected raw layout after step 3 (paths are what the code reads —
`ml/eda/run_eda.py:42-44`):

```
data/raw/uci697/data.csv                  533,230 B   UCI #697
data/raw/uci320/student/student-por.csv    93,220 B   UCI #320
data/raw/uci320/student/student-mat.csv    56,993 B   UCI #320
```

If a file is missing or wrong, `ml.eda.run_eda` fails fast with a
`FileNotFoundError` naming the path and pointing at `data/README.md`.

Trained `model.joblib` artefacts are excluded too — regenerate them with
`python -m ml.registry.export` (re-runs the Phase-4 protocol, ~3 min).

**Current status: Phase 7 complete** (architecture, dataset acquisition, EDA,
leakage audit, preprocessing, training, registry + inference, FastAPI backend,
React dashboard). Optional prediction-history DB deferred. Next: Phase 8
(tests, lint, final polish).

## Repository structure

```
.
├── README.md               # this file — orientation + how to reproduce
├── requirements.txt        # Python dependencies (with comments on why each exists)
├── .gitignore              # keeps data, secrets and binaries out of Git
├── .env.example            # template for environment variables (copy to .env later)
│
├── data/                   # ALL datasets live here (see data/README.md)
│   ├── raw/                # ORIGINAL downloads — immutable, never edited
│   │   ├── uci697/         # primary: UCI #697 dropout dataset
│   │   └── uci320/         # comparison: UCI #320 student performance
│   └── processed/          # regenerable prepared splits (Phase 3 outputs)
│
├── ml/                     # all machine-learning code (notebook-free, script-based)
│   ├── eda/                #   exploratory analysis scripts (Phase 2)
│   ├── features/           #   feature-set definitions (H1/H2 horizon + ablation configs)
│   ├── preprocessing/      #   leakage-safe pipeline: feature_types, leakage_checks,
│   │                       #   build_pipeline, prepare_data, verify (Phase 3)
│   ├── training/           #   models.py (config registry), metrics.py, train.py
│   │                       #   (Phase 4) → reports/model_report.md
│   ├── registry/           #   registry.py (versioned artifacts + current
│   │                       #   pointer), export.py (Phase 5) → *.joblib/json
│   └── inference/          #   predict.py — validated student dict → risk
│                           #   prediction (Phase 5, used by API in Phase 6)
│
├── backend/                # [Phase 6 ✅] FastAPI app: /health, /model/info,
│   │                       #   /predict — see backend/README.md
├── frontend/               # [Phase 7 ✅] Vite + React + TS dashboard —
│   │                       #   see frontend/README.md
├── database/               # [deferred] optional prediction/history store — placeholder
│
├── docs/                   # design documentation (read before changing anything)
│   ├── ARCHITECTURE.md     # system design, ML design, API contract, roadmap
│   ├── DATA.md             # dataset card + verified EDA findings
│   ├── LEAKAGE_AUDIT.md    # per-feature H1/H2 allowance table + reasoning
│   ├── PREPROCESSING.md    # Phase 3: pipeline architecture + strategies
│   └── REGISTRY.md         # Phase 5: registry layout, export, inference API
│
├── reports/                # eda_report.md, preprocessing_report.md, model_report.md,
│   │                       # model_comparison.csv, figs/*.png
└── tests/                  # [Phase 8] pytest suite — placeholder README only
```

### Why this layout
- **`data/raw` is read-only**: raw data must stay byte-identical to the official
  source so any result can be reproduced; all transformations write to
  `data/processed` instead.
- **`ml/` is script-based, not notebook-based**: scripts are reviewable, rerunnable
  and testable; notebooks (if ever used) are for exploration only.
- **Placeholders instead of empty scaffolding**: future phases get their own
  directory with a README explaining the plan — no premature infrastructure.

## Datasets

| Role | Dataset | Source | License |
|---|---|---|---|
| Primary | UCI #697 — Predict Students' Dropout and Academic Success | [archive.ics.uci.edu/dataset/697](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) | CC BY 4.0 |
| Comparison | UCI #320 — Student Performance | [archive.ics.uci.edu/dataset/320](https://archive.ics.uci.edu/dataset/320/student+performance) | CC BY 4.0 |

CC BY 4.0 requires **attribution**. Required credit (include in any report/README):

> Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021).
> *Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine
> Learning Repository. https://doi.org/10.24432/C5MC89
>
> Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning
> Repository. https://doi.org/10.24432/C5TG7T

Checksums and file inventory: see `data/README.md`.

## Reproducing Phases 2–4

```powershell
# 1. (first time only) install dependencies
pip install -r requirements.txt

# 2. verify raw data integrity (shapes must match data/README.md)
python -m ml.eda.run_eda          # writes reports/eda_report.md + reports/figs/

# 3. inspect the feature-set definitions used by the leakage audit
python -m ml.features.feature_sets

# 4. Phase 3: type classification checks, then full preprocessing verification
python -m ml.preprocessing.feature_types
python -m ml.preprocessing.verify  # writes reports/preprocessing_report.md +
                                   # data/processed/uci697/*.csv (6 split files)

# 5. Phase 4: train all 15 pre-declared configs with 5-fold CV + terminal
#    test pass (~3-4 min). Selection uses CV only; test is touched once.
python -m ml.training.train        # writes reports/model_report.md +
                                   # reports/model_comparison.csv + 4 figures

# 6. Phase 5: export winners to the versioned registry (re-runs the Phase-4
#    protocol, ~3 min) — ends with a registry → inference smoke test.
python -m ml.registry.export       # writes ml/registry/<version>/* + `current`

# 7. Phase 5: predict with the active model (demo payload = first raw row)
python -m ml.inference.predict     # prints the /predict response JSON

# 8. Phase 6: run the API (OpenAPI docs at http://127.0.0.1:8000/docs)
python -m backend.app.main         # /health, /model/info, /predict

# 9. Phase 7: run the dashboard in a SECOND terminal (API from step 8 up)
cd frontend
npm install                        # first time only
npm run dev                        # http://localhost:5173
#    production build: npm run build  (tsc type-check + vite build)
```

## Project phases

| Phase | Content | Status |
|---|---|---|
| 1 | Problem framing + public dataset research | ✅ done |
| 2 | Architecture, data acquisition, EDA, leakage audit | ✅ done |
| 3 | Preprocessing pipeline + feature engineering | ✅ done |
| 4 | Model training & evaluation (LR/DT/KNN/NB/SVM/RF vs. baseline) | ✅ done |
| 5 | Model registry + inference layer | ✅ done |
| 6 | FastAPI backend | ✅ done |
| 7 | React frontend dashboard (DB deferred) | ✅ done |
| 8 | Tests, lint, final polish | ⬜ |

Design details live in `docs/ARCHITECTURE.md`.
**Read `docs/LEAKAGE_AUDIT.md` before adding or removing any feature.**
Preprocessing architecture and verification results: `docs/PREPROCESSING.md`
and `reports/preprocessing_report.md`. Model comparison and protocol:
`reports/model_report.md` (headline: H1 test macro-F1 0.546, H2 0.650 —
CV-selected winners, honest terminal pass). Registry and inference contract:
`docs/REGISTRY.md`. API contract and endpoints: `backend/README.md`.
Frontend structure, commands and notes: `frontend/README.md`.

## API

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness + loaded model version |
| `GET` | `/model/info` | Full model card (metrics, features, protocol) — served without unpickling the model |
| `POST` | `/predict` | Validated student dict → risk class + class probabilities |

Interactive OpenAPI docs at `http://127.0.0.1:8000/docs` while the backend runs.
Validation contract (unknown / missing / non-numeric fields → `422`) is defined by
Pydantic in `backend/app/schemas.py`.

## Datasets & attribution

Both datasets are **CC BY 4.0** and are redistributed under that licence. The
required citations are listed in the *Datasets* table above — please carry them
into any derived report.

## License

MIT — see [`LICENSE`](LICENSE).

