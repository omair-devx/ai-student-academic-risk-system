"""Phase 5 — model registry: versioned, self-describing model artifacts.

Layout (docs/ARCHITECTURE.md §5.6):

    ml/registry/
    ├── 2026-09-29_h1_svm_rbf_balanced/   # one folder per exported version
    │   ├── model.joblib                   # FULL fitted pipeline (gitignored)
    │   ├── model_card.json                # metadata for humans/API (gitignored)
    │   └── metrics.json                   # CV + test metrics (gitignored)
    ├── current                            # one-line pointer to the ACTIVE
    │                                      # version (committed on purpose)
    ├── registry.py                        # this module (committed)
    └── export.py                          # trainer that writes versions (committed)

WHY the whole pipeline is pickled, not just the estimator:
inference must apply EXACTLY the fitted preprocessor — the same imputer
medians, the same one-hot level list, the same scaler means. Re-fitting or
re-deriving preprocessing at load time would be slower AND statistically
different from what was evaluated. joblib serializes the entire fitted
sklearn graph in one file; loading reconstructs it as-is.

WHY cards/metrics are separate JSON files:
- the API (Phase 6) serves /model/info WITHOUT unpickling the model,
- humans can diff cards in code review,
- metrics.json is machine-readable for tests/dashboards.

WHY a `current` pointer file instead of a hard-coded path:
activating a version is a one-line file write — no code change, no restart
logic, and the active choice is visible (and committed) in the repo.

Machine-specific paths: registry location can be overridden with the
MODEL_REGISTRY_PATH environment variable (declared in .env.example) so no
absolute path is ever hard-coded.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import joblib

# Env override first, else the package directory itself (model files live
# next to this code on purpose — the registry is one self-contained folder).
REGISTRY_DIR = Path(os.environ.get("MODEL_REGISTRY_PATH", Path(__file__).resolve().parent)).resolve()
CURRENT_FILE = REGISTRY_DIR / "current"

MODEL_FILE = "model.joblib"
CARD_FILE = "model_card.json"
METRICS_FILE = "metrics.json"


@dataclass(frozen=True)
class RegisteredModel:
    """Everything inference needs for one version (loaded in one place)."""
    version: str
    pipeline: object          # fitted sklearn Pipeline (preprocess + model)
    card: dict                # model_card.json contents
    metrics: dict             # metrics.json contents


def save_model(version: str, pipeline, card: dict, metrics: dict) -> Path:
    """Write one version folder. Idempotent: re-exporting the same version
    overwrites its files (same date + same config = same logical model)."""
    if not version or "/" in version or "\\" in version:
        raise ValueError(f"Invalid version name {version!r}")
    out = REGISTRY_DIR / version
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, out / MODEL_FILE)
    (out / CARD_FILE).write_text(
        json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out / METRICS_FILE).write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return out


def list_versions() -> list[str]:
    """All exported versions (folders that contain a model file), sorted."""
    if not REGISTRY_DIR.exists():
        return []
    return sorted(
        p.name for p in REGISTRY_DIR.iterdir()
        if p.is_dir() and (p / MODEL_FILE).exists()
    )


def load_model(version: str) -> RegisteredModel:
    """Load one version; fail loudly with actionable messages (no silent
    fallback to another version — predicting with the wrong model is worse
    than refusing to predict)."""
    folder = REGISTRY_DIR / version
    missing = [name for name in (MODEL_FILE, CARD_FILE, METRICS_FILE)
               if not (folder / name).exists()]
    if missing:
        known = ", ".join(list_versions()) or "none"
        raise FileNotFoundError(
            f"Registry version {version!r} is incomplete (missing {missing}) "
            f"in {folder}. Known versions: {known}. "
            "Run 'python -m ml.registry.export' to (re)create models."
        )
    return RegisteredModel(
        version=version,
        pipeline=joblib.load(folder / MODEL_FILE),
        card=json.loads((folder / CARD_FILE).read_text(encoding="utf-8")),
        metrics=json.loads((folder / METRICS_FILE).read_text(encoding="utf-8")),
    )


def set_current(version: str) -> None:
    """Point `current` at an existing version (atomic-ish: write temp + move,
    so a crashed write can't leave a half-written pointer)."""
    if not (REGISTRY_DIR / version / MODEL_FILE).exists():
        raise FileNotFoundError(
            f"Cannot activate {version!r}: no {MODEL_FILE} — known: {list_versions()}"
        )
    tmp = CURRENT_FILE.with_suffix(".tmp")
    tmp.write_text(version + "\n", encoding="utf-8")
    tmp.replace(CURRENT_FILE)


def get_current() -> str:
    """Read the active version name; actionable error if never exported."""
    if not CURRENT_FILE.exists():
        raise FileNotFoundError(
            f"No active model pointer at {CURRENT_FILE}. "
            "Run 'python -m ml.registry.export' to train and register models."
        )
    version = CURRENT_FILE.read_text(encoding="utf-8").strip()
    if not version:
        raise ValueError(f"Active-model pointer {CURRENT_FILE} is empty.")
    return version


def load_current() -> RegisteredModel:
    return load_model(get_current())


def load_card(version: str | None = None) -> dict:
    """Read ONLY model_card.json — no joblib unpickling.

    This is how /model/info serves metadata (docs/REGISTRY.md §2): cheap,
    safe (no pickle execution), and available even while a model load would
    be expensive. `version=None` resolves the `current` pointer first.
    """
    if version is None:
        version = get_current()
    path = REGISTRY_DIR / version / CARD_FILE
    if not path.exists():
        known = ", ".join(list_versions()) or "none"
        raise FileNotFoundError(
            f"No model card for version {version!r} at {path}. "
            f"Known versions: {known}."
        )
    return json.loads(path.read_text(encoding="utf-8"))
