# Model Registry & Inference Layer (Phase 5)

Source of truth for *what gets served*: every prediction the API (Phase 6)
makes goes through `ml/inference/predict.py` reading exactly one versioned
artifact produced by `ml/registry/export.py`.

## 1. Layout

```
ml/registry/
├── 2026-09-30_h1_svm_rbf_balanced/     # one folder per export
│   ├── model.joblib                    # FULL fitted pipeline (preprocess + model)
│   ├── model_card.json                 # metadata: protocol, metrics, caveats
│   └── metrics.json                    # CV + test metrics (machine-readable)
├── 2026-09-30_h2_random_forest_balanced/
├── current                             # one-line pointer to the ACTIVE version
├── registry.py                         # save/load/list/activate API
└── export.py                           # trains winners, writes versions
```

Version names: `YYYY-MM-DD_<horizon>_<config_id>` (e.g.
`2026-09-30_h1_svm_rbf_balanced`). Re-running export on the same date with the
same config **overwrites** that folder (idempotent, deterministic seeds).

**Committed vs gitignored** (`.gitignore`, ARCHITECTURE §8): version-folder
contents are ignored — they are regenerable from code + data via
`python -m ml.registry.export`. The `current` pointer **is** committed: which
model is live is deliberate, reviewable, one-line state.

## 2. What is saved and why

- **The whole fitted pipeline**, not just the estimator: inference must apply
  the identical imputer medians, one-hot level list and scaler means that were
  evaluated. Re-deriving preprocessing at load time would be slower and
  statistically different.
- **Cards/metrics as JSON**: `/model/info` can answer without unpickling a
  model; humans diff cards in review; tests read metrics.json directly.
- **`model_card.json` contents**: horizon, model config, *how it was selected*
  (incl. whether `--config` forced it), dataset citation/DOI (CC BY 4.0),
  feature lists, split/CV/test protocol, CV + test metrics, library versions,
  ethics notes, and for H1 the **uncertain-feature ablation outcome**
  (required by `docs/LEAKAGE_AUDIT.md` §3: ΔCV −0.052 → NOT immaterial).

## 3. Exporting

```powershell
python -m ml.registry.export                        # both horizons (~3 min)
python -m ml.registry.export --horizons h1          # H1 only
python -m ml.registry.export --horizons h2 --config random_forest_balanced
```

Protocol = Phase 4, re-executed through the same functions
(`ml.training.train.run_cv/select_winner` — one copy of the logic): 5-fold CV
selection over the 15 pre-declared configs → winner fitted on full train →
terminal test metrics **computed from the artifact itself** → saved. Same
seeds ⇒ same winners as `reports/model_report.md`; a report file is never a
data source.

**`current` pointer:** defaults to the H1 version — H1 is the product
(earliest warning; the §6 API sketch is H1). The H2 model stays registered;
switch with `registry.set_current("<version>")`.

**SVM probabilities:** `SVC` is exported with `probability=True` (libsvm Platt
scaling) because `/predict` must return probability bars. Hard predictions
still come from the decision function — identical to how Phase-4 metrics were
computed. `CalibratedClassifierCV` (sklearn's suggested replacement) is
deliberately NOT used: its `predict` is argmax-of-probability and *can differ*
from the calibrated base classifier, which would break metric parity. Hence
`scikit-learn>=1.9,<1.11` in `requirements.txt` (the parameter is removed in
1.11; migrating requires re-evaluation, not a pip bump).

## 4. Inference API

```python
from ml.inference.predict import predict_student
result = predict_student({"Age at enrollment": 19, ...})   # active model
result = predict_student(payload, version="2026-09-30_h2_random_forest_balanced")
```

Response (ARCHITECTURE §6):

```json
{
  "horizon": "H1",
  "prediction": "Dropout",
  "at_risk": true,
  "probabilities": {"Dropout": 0.679, "Enrolled": 0.078, "Graduate": 0.243},
  "model_version": "2026-09-30_h1_svm_rbf_balanced"
}
```

Validation is strict and reports **all** problems at once (unknown field /
missing field / non-numeric / NaN) as `ValueError` — Phase 6 maps these to
HTTP 422. No silent defaults: predicting against a vector the caller never
provided is the worst failure mode for a risk tool.

- `prediction` = `pipeline.predict()` (margin rule, same as evaluation);
  `probabilities` = calibrated estimates for display/thresholding.
- Loaded models are cached per version; `clear_cache()` after an in-place
  re-export in a long-running process.

CLI:

```powershell
python -m ml.inference.predict                          # demo (first raw row)
python -m ml.inference.predict --input student.json     # real payload file
python -m ml.inference.predict --version 2026-09-30_h2_random_forest_balanced --demo
```

## 5. Round-trip verification

`export` ends with a smoke test: registry load → `predict_student` → printed
probabilities, for every version it just wrote (run output shows both
passing). Phase 8 will turn this pattern into pytest cases.
