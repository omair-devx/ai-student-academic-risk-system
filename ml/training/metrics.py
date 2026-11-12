"""Phase 4 — metric computation shared by CV and the terminal test pass.

WHY metrics are centralized (docs/ARCHITECTURE.md §5.5):
* macro-F1 is the PRIMARY metric: accuracy alone is prohibited as a headline
  because a constant 'Graduate' predictor already scores 49.9% while never
  flagging a single dropout.
* per-class recall answers the operational question ("what fraction of actual
  dropouts do we catch?") — the metric that matters for an early-warning tool.
* balanced accuracy = mean per-class recall, computed here explicitly so the
  report shows exactly what it averages (no hidden sklearn behavior).
* the binary view (ROC-AUC + precision/recall for 'Dropout') follows the
  architecture's multi-class-first, binary-view-second plan; it is derived
  from the SAME 3-class predictions (Dropout vs rest), never from a separately
  trained binary model — that experiment is reserved for a later phase.

Class order is pinned to CLASS_ORDER everywhere so confusion matrices and
per-class columns never shuffle between runs (sklearn sorts class labels
alphabetically; we make the order explicit anyway to avoid surprises if a
label string ever changes).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Canonical label order for this project (matches alphabetical sklearn order,
# pinned explicitly for stable report/figure layouts).
CLASS_ORDER: list[str] = ["Dropout", "Enrolled", "Graduate"]


def dropout_scores(estimator, X) -> np.ndarray:
    """Per-row score for the 'Dropout' class (higher = more at risk).

    Uses predict_proba when available, else decision_function (SVM has no
    probabilities unless probability=True, which doubles fit time — scores
    are rank-equivalent for ROC-AUC, so probabilities add nothing here).

    Pipeline note: Pipeline.classes_ is delegated to the final estimator; for
    3-class problems decision_function returns an (n, 3) OvR-shaped matrix
    whose columns follow classes_ order.
    """
    classes = list(estimator.classes_)
    idx = classes.index("Dropout")
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        if proba.shape[1] == len(classes):
            return proba[:, idx]
        # Defensive fallback: a fold where some class is absent would break
        # column alignment — map by name instead of position.
        col = classes.index("Dropout") if "Dropout" in classes else -1
        return proba[:, col]
    scores = estimator.decision_function(X)
    if scores.ndim == 1:
        # Only possible for binary problems (not this dataset) — kept so the
        # helper fails loudly-but-wrongly only if the task definition changes.
        raise ValueError("1-D decision_function; expected a 3-class problem")
    return scores[:, idx]


def compute_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Full metric bundle for one set of predictions.

    Parameters
    ----------
    y_true : ground-truth labels
    y_pred : hard predictions
    scores : per-row Dropout score from `dropout_scores` (for ROC-AUC)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    per_class_recall = recall_score(y_true, y_pred, labels=CLASS_ORDER, average=None, zero_division=0)
    per_class_precision = precision_score(y_true, y_pred, labels=CLASS_ORDER, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, labels=CLASS_ORDER, average=None, zero_division=0)

    metrics: dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        # primary metric (architecture §5.5)
        "macro_f1": f1_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0),
        # mean of the three per-class recalls — shown explicitly in the report
        "balanced_accuracy": float(np.mean(per_class_recall)),
        # binary view: Dropout vs (Enrolled | Graduate)
        "auc_dropout": roc_auc_score(y_true == "Dropout", scores),
        # precision/recall of the Dropout decision (hard predictions)
        "dropout_precision": float(per_class_precision[0]),
        "dropout_recall": float(per_class_recall[0]),
    }
    for label, r, p, f in zip(CLASS_ORDER, per_class_recall, per_class_precision, per_class_f1):
        metrics[f"recall_{label}"] = float(r)
        metrics[f"precision_{label}"] = float(p)
        metrics[f"f1_{label}"] = float(f)
    return metrics


def confusion(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    """Confusion matrix with rows=true, cols=predicted, fixed CLASS_ORDER."""
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    return cm.astype(int).tolist()


def majority_baseline_metrics(y_true: pd.Series) -> dict[str, float]:
    """Metrics of the constant 'Graduate' predictor — the bar, computed once
    from the true distribution so every model's numbers sit next to it."""
    y_pred = np.full(len(y_true), "Graduate")
    # Constant score → ROC-AUC is exactly 0.5 (no ranking ability).
    scores = np.zeros(len(y_true))
    return compute_metrics(y_true, y_pred, scores)
