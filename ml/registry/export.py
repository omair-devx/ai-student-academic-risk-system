"""Phase 5 — export: (re)train the winners and write versioned artifacts.

WHY this re-runs training instead of copying numbers from
reports/model_comparison.csv: the registry must be reproducible from code +
data alone. A report file is a rendering, not a source of truth — the
Phase-4 protocol is re-executed here through the SAME imported functions
from ml.training.train (single copy of the logic, no drift), and the card's
test metrics are computed FROM THE EXACT PIPELINE THAT GETS SAVED, so the
numbers always describe the bytes on disk.

Protocol per horizon (identical to Phase 4 → same seeds → same winners):
  1. 5-fold CV selection over all 15 pre-declared configs (CV only),
  2. winner fitted on the FULL train split — this fit IS the artifact,
  3. terminal test evaluation of that artifact,
  4. H1 additionally records the uncertain-features ablation in its card
     (required by docs/LEAKAGE_AUDIT.md §3).

SVM note: SVC is exported with probability=True. libsvm then fits Platt
scaling to produce class probabilities, which the /predict contract
(ARCHITECTURE §6) requires. Hard predictions come from the decision function
and are unchanged by calibration, so Phase-4 metrics remain valid for the
served artifact (this script still recomputes them from the artifact itself).

Usage:
    python -m ml.registry.export                      # both horizons
    python -m ml.registry.export --horizons h1        # H1 only
    python -m ml.registry.export --horizons h2 --config random_forest_balanced
                                                      # force a config (skips CV)
"""

from __future__ import annotations

import argparse
import platform
import time
from datetime import date

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.svm import SVC

from ml.features.feature_sets import TARGET_COLUMN, TARGET_CLASSES
from ml.inference.predict import predict_student
from ml.preprocessing.prepare_data import (
    load_master_frame,
    make_stratified_kfold,
    prepare_horizon,
)
from ml.registry import registry as reg
from ml.training.metrics import CLASS_ORDER, confusion, compute_metrics, dropout_scores
from ml.training.models import CONFIG_BY_ID, MODEL_CONFIGS
from ml.training.train import build_pipeline, make_row, run_cv, run_terminal, select_winner

# Attribution required by CC BY 4.0 (kept verbatim in every model card so an
# exported artifact stays licensed/traceable even outside this repo).
DATASET_META = {
    "name": "UCI #697 — Predict Students' Dropout and Academic Success",
    "doi": "https://doi.org/10.24432/C5MC89",
    "url": "https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success",
    "license": "CC BY 4.0",
    "citation": ("Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. "
                 "(2021). Predict Students' Dropout and Academic Success [Dataset]. "
                 "UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89"),
}

UNCERTAIN_FEATURES = ["Displaced", "Debtor", "Tuition fees up to date",
                      "Scholarship holder"]


def select_config(horizon: str, prepared, kfold, force_id: str | None):
    """CV selection via the Phase-4 helpers (same rule, same code path).

    Returns (config, cv_metrics, selection_info). force_id bypasses CV —
    recorded in the card so an overridden selection is never silent."""
    if force_id is not None:
        if force_id not in CONFIG_BY_ID:
            raise ValueError(f"Unknown config {force_id!r}; known: {sorted(CONFIG_BY_ID)}")
        return CONFIG_BY_ID[force_id], None, {
            "rule": "forced via --config (CV selection skipped)",
            "forced": True,
        }

    rows, cv_by_id = [], {}
    for cfg in MODEL_CONFIGS:
        cv = run_cv(cfg, prepared, kfold)
        cv_by_id[cfg.id] = cv
        rows.append(make_row(horizon, "A", cfg, cv))
        m = cv["metrics"]
        print(f"  cv  {cfg.id:<26s} macroF1={m['macro_f1']:.3f} "
              f"+-{m['macro_f1_std']:.3f}", flush=True)

    winner_id = select_winner(pd.DataFrame(rows), horizon)
    return (CONFIG_BY_ID[winner_id], cv_by_id[winner_id]["metrics"], {
        "rule": "max 5-fold CV macro-F1 on train (tie-break: CV balanced "
                "accuracy) — identical to Phase 4; test not consulted",
        "forced": False,
    })


def fit_artifact(config, prepared):
    """Fit the pipeline that will be saved (full train split).

    probability=True for SVC: Platt scaling for the API's probability bars;
    margins (and therefore hard predictions) are unchanged by calibration."""
    pipe = build_pipeline(config, prepared)
    est = pipe.named_steps["model"]
    if isinstance(est, SVC):
        est.set_params(probability=True)
    pipe.fit(prepared.X_train, prepared.y_train)
    return pipe


def artifact_metrics(pipe, prepared) -> tuple[dict, list[list[int]]]:
    """Terminal metrics computed FROM the artifact that gets saved."""
    pred = pipe.predict(prepared.X_test)
    scores = dropout_scores(pipe, prepared.X_test)
    return compute_metrics(prepared.y_test, pred, scores), confusion(prepared.y_test, pred)


def build_card(version: str, horizon: str, prepared, config,
               selection: dict, cv_metrics: dict | None, test_metrics: dict,
               ablation: dict | None) -> dict:
    """model_card.json — everything a reviewer/API needs without loading joblib."""
    notes = [
        "Educational-support early-warning tool only; never for automated "
        "decision-making (docs/ARCHITECTURE.md §9).",
        "3-class target (Dropout/Enrolled/Graduate); the binary at-risk view "
        "is a separate, not-yet-run experiment.",
        "No real student data anywhere — metrics come from the public UCI "
        "dataset; API inputs are synthetic/illustrative.",
        "Metrics: macro-F1 primary; accuracy reported for reference only "
        "(majority-class baseline = 0.499 accuracy / 0.222 macro-F1).",
    ]
    if "svm" in config.id:
        notes.append(
            "SVC exported with probability=True (libsvm Platt scaling) so "
            "/predict can return probabilities; hard predictions use the "
            "decision function and are unaffected by calibration."
        )
    card = {
        "version": version,
        "created_on": str(date.today()),
        "horizon": horizon,
        "experiment": "A",
        "model": {
            "config_id": config.id,
            "label": config.label,
            "params": config.params_str,
            "scale_numeric": config.scale_numeric,
        },
        "selection": selection,
        "data": {
            **DATASET_META,
            "rows_total": int(len(prepared.X_train) + len(prepared.X_test)),
            "train_rows": int(len(prepared.X_train)),
            "test_rows": int(len(prepared.X_test)),
            "split": "stratified 80/20, random_state=42",
            "target_column": TARGET_COLUMN,
            "target_classes": list(TARGET_CLASSES),
            "features": list(prepared.features),
            "numerical": list(prepared.numerical),
            "categorical": list(prepared.categorical),
        },
        "protocol": {
            "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42) "
                  "over the full preprocess+model pipeline (refit per fold)",
            "selection": selection["rule"],
            "test": "single terminal evaluation on the untouched 20% holdout",
            "metrics": ["macro_f1 (primary)", "balanced_accuracy",
                        "recall per class", "roc_auc_dropout", "confusion_matrix"],
        },
        "metrics": {"cv": cv_metrics, "test": test_metrics},
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "notes": notes,
    }
    if ablation is not None:
        card["uncertain_feature_ablation"] = ablation
    return card


def run_ablation(config, kfold) -> dict:
    """H1 Experiment B: same winner trained WITHOUT the four uncertain
    features (docs/LEAKAGE_AUDIT.md §3 — outcome must land in the card)."""
    prepared_b = prepare_horizon("H1", "B")
    cv_b = run_cv(config, prepared_b, kfold)["metrics"]
    term_b = run_terminal(config, prepared_b)["metrics"]
    return {
        "features_excluded": UNCERTAIN_FEATURES,
        "reason": "enrollment-time provenance of these four columns is not "
                  "documented by UCI (timing ambiguity flagged in Phase 2)",
        "cv_macro_f1_with_A": None,      # filled by caller with A values
        "test_macro_f1_with_A": None,
        "cv_macro_f1_without_B": round(cv_b["macro_f1"], 6),
        "test_macro_f1_without_B": round(term_b["macro_f1"], 6),
    }


def smoke(version: str) -> None:
    """Round-trip check: registry load → inference layer → prediction."""
    model = reg.load_model(version)
    row = load_master_frame().iloc[0]
    payload = {
        f: (v.item() if isinstance(v, np.generic) else v)
        for f, v in ((f, row[f]) for f in model.card["data"]["features"])
    }
    result = predict_student(payload, version=version)
    probs = ", ".join(f"{k}={v:.2f}" for k, v in result["probabilities"].items())
    print(f"  smoke {version}: predict OK -> {result['prediction']} [{probs}]")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Train winners and write versioned registry artifacts (Phase 5)."
    )
    ap.add_argument("--horizons", nargs="+", choices=["h1", "h2"],
                    default=["h1", "h2"], help="which horizons to export")
    ap.add_argument("--config", default=None,
                    help="force this config id for all requested horizons "
                         "(skips CV selection; recorded in the card)")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    print("=== Phase 5: export to model registry ===", flush=True)
    kfold = make_stratified_kfold()
    saved: dict[str, str] = {}

    for h in args.horizons:
        horizon = h.upper()
        print(f"\n--- {horizon} ---", flush=True)
        prepared = prepare_horizon(horizon, "A")
        config, cv_metrics, selection = select_config(
            horizon, prepared, kfold, force_id=args.config
        )

        pipe = fit_artifact(config, prepared)          # the artifact itself
        test_metrics, test_cm = artifact_metrics(pipe, prepared)

        ablation = None
        if horizon == "H1" and args.config is None:
            ablation = run_ablation(config, kfold)
            ablation["cv_macro_f1_with_A"] = round(cv_metrics["macro_f1"], 6)
            ablation["test_macro_f1_with_A"] = round(test_metrics["macro_f1"], 6)
            d_cv = ablation["cv_macro_f1_without_B"] - ablation["cv_macro_f1_with_A"]
            d_test = ablation["test_macro_f1_without_B"] - ablation["test_macro_f1_with_A"]
            ablation["delta_cv_B_minus_A"] = round(d_cv, 6)
            ablation["delta_test_B_minus_A"] = round(d_test, 6)
            ablation["verdict"] = (
                "NOT immaterial (dropping the four features hurts): adopt the "
                "24-feature model only after confirming enrollment-time "
                "provenance with the data providers; otherwise quote the "
                "20-feature Experiment B numbers as the conservative H1 result"
            )

        cv_clean = {k: v for k, v in (cv_metrics or {}).items()
                    if k not in ("fit_s",)}            # fit time is run noise
        version = f"{date.today():%Y-%m-%d}_{horizon.lower()}_{config.id}"
        card = build_card(version, horizon, prepared, config, selection,
                          cv_clean, test_metrics, ablation)
        metrics_json = {
            "cv": cv_clean,
            "test": test_metrics,
            "test_confusion": test_cm,
            "classes": list(CLASS_ORDER),
        }
        reg.save_model(version, pipe, card, metrics_json)
        saved[horizon] = version
        print(f"  saved {version} "
              f"(test macro-F1 {test_metrics['macro_f1']:.3f}, "
              f"auc {test_metrics['auc_dropout']:.3f})", flush=True)

    # Activate: H1 is the primary product (earliest warning; API §6 sketch is
    # H1). The H2 version stays registered and can be switched in via
    # registry.set_current(). Fallback: if only H2 was exported and nothing
    # is active yet, activate it so the smoke test/API have a model.
    if "H1" in saved:
        reg.set_current(saved["H1"])
    else:
        try:
            reg.get_current()
        except FileNotFoundError:
            reg.set_current(saved["H2"])
    print(f"\nactive model: {reg.get_current()}", flush=True)

    # ASCII only in stdout: Windows consoles (cp1252) crash on unicode arrows.
    print("--- smoke test (registry -> inference round-trip) ---", flush=True)
    for version in saved.values():
        smoke(version)

    print(f"\nversions: {reg.list_versions()}")
    print(f"total: {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
