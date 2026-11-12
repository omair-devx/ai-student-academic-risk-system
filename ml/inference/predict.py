"""Phase 5 — inference layer: validated student dict → risk prediction.

Public entry point (Phase 6's /predict route calls exactly this):

    predict_student(payload, *, version=None) -> dict
        {
          "horizon": "H1",
          "prediction": "Dropout",
          "at_risk": true,
          "probabilities": {"Dropout": 0.71, "Enrolled": 0.22, "Graduate": 0.07},
          "model_version": "2026-09-29_h1_svm_rbf_balanced"
        }

WHY validation is strict — unknown field → error, missing field → error,
non-numeric/NaN → error (every problem reported at once):
silently defaulting a missing feature would score the student against a
vector they never provided — plausible-looking output that is actually wrong
is the worst failure mode for a risk tool. FastAPI (Phase 6) maps these
ValueErrors to HTTP 422 exactly as the API contract requires.

WHY `prediction` comes from pipeline.predict() and not argmax(probabilities):
it must be IDENTICAL to how Phase-4 metrics were computed (for SVM that is
the decision function). The probability bars are Platt-scaled estimates used
for display/thresholding; on rare rows the calibrated maximum can differ
from the margin winner — display stays honest, decision stays comparable.

WHY the payload becomes a one-row DataFrame with the card's feature order:
the ColumnTransformer selects columns BY NAME, so ordering cannot matter and
any structural mismatch has already been rejected by validation.

Model loading is cached per version (joblib unpickle costs ~100ms; the API
predicts per request). clear_cache() exists for long-running processes that
re-export a version in place.

CLI:
    python -m ml.inference.predict                 # demo: first raw dataset row
    python -m ml.inference.predict --input p.json  # predict from a JSON file
    python -m ml.inference.predict --version 2026-09-29_h2_random_forest_balanced --demo
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ml.preprocessing.prepare_data import load_master_frame
from ml.registry.registry import RegisteredModel, get_current, load_model

# version name → loaded model (see module docstring).
_CACHE: dict[str, RegisteredModel] = {}


def clear_cache() -> None:
    """Drop loaded models (call after re-exporting a version in place)."""
    _CACHE.clear()


def _get_model(version: str | None) -> RegisteredModel:
    if version is None:
        version = get_current()   # resolve the pointer first ...
    if version not in _CACHE:     # ... then load/cache by name
        _CACHE[version] = load_model(version)
    return _CACHE[version]


def _validate_payload(payload: dict, features: list[str]) -> None:
    """Collect EVERY problem, then raise once — a form that lists 17 missing
    fields one-by-one wastes the caller's time."""
    if not isinstance(payload, dict):
        raise ValueError(f"Payload must be a JSON object, got {type(payload).__name__}")

    problems: list[str] = []
    unknown = sorted(set(payload) - set(features))
    if unknown:
        problems.append(f"unknown fields: {unknown}")
    missing = sorted(set(features) - set(payload))
    if missing:
        problems.append(f"missing fields: {missing}")

    for field in features:
        if field not in payload:
            continue
        value = payload[field]
        # bool excluded on purpose: in Python bool is an int subclass, and a
        # True/False for e.g. 'GDP' would silently become 1/0.
        if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            problems.append(
                f"{field!r}: expected a number, got {type(value).__name__} ({value!r})"
            )
        elif math.isnan(float(value)) or math.isinf(float(value)):
            problems.append(f"{field!r}: must be finite, got {value!r}")

    if problems:
        raise ValueError("Invalid student payload — " + "; ".join(problems))


def predict_student(payload: dict, *, version: str | None = None) -> dict:
    """Validate + predict with one registry model (default: active/current)."""
    model = _get_model(version)
    features = model.card["data"]["features"]
    _validate_payload(payload, features)

    row = pd.DataFrame([[payload[f] for f in features]], columns=features)
    prediction = str(model.pipeline.predict(row)[0])
    classes = [str(c) for c in model.pipeline.classes_]
    probabilities = {
        cls: round(float(p), 4)
        for cls, p in zip(classes, model.pipeline.predict_proba(row)[0])
    }
    return {
        "horizon": model.card["horizon"],
        "prediction": prediction,
        "at_risk": prediction == "Dropout",
        "probabilities": probabilities,
        "model_version": model.version,
    }


def demo_payload(features: list[str]) -> dict:
    """First row of the raw dataset, trimmed to this model's features —
    a realistic all-fields-present payload for smoke tests/demos only."""
    row = load_master_frame().iloc[0]
    return {
        f: (v.item() if isinstance(v, np.generic) else v)
        for f, v in ((f, row[f]) for f in features)
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Predict academic-risk class for one student payload."
    )
    ap.add_argument("--input", type=Path,
                    help="JSON file containing one student object "
                         "(default: built-in demo payload)")
    ap.add_argument("--demo", action="store_true",
                    help="use the demo payload (first raw dataset row) — "
                         "this is also the default when --input is omitted")
    ap.add_argument("--version", default=None,
                    help="registry version to use (default: active/current)")
    ap.add_argument("--indent", type=int, default=2, help="JSON indent")
    args = ap.parse_args(argv)

    model = _get_model(args.version)
    if args.input and args.demo:
        ap.error("--input and --demo are mutually exclusive")
    if args.input:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
    else:
        payload = demo_payload(model.card["data"]["features"])

    result = predict_student(payload, version=model.version)
    print(json.dumps(result, indent=args.indent, ensure_ascii=False))


if __name__ == "__main__":
    main()
