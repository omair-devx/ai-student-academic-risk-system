"""Phase 4 — pre-declared model configuration registry (config-as-data).

WHY this exists instead of instantiating estimators inside the training loop:

* Up-front declaration: the candidate list is fixed BEFORE any result is
  observed. The comparison table can therefore never be quietly padded with
  an extra model after seeing the numbers — a common, hard-to-spot form of
  selection bias.

* Pairing estimator with its required preprocessing variant: KNN/LR/SVM need
  scaled numericals, trees/NB do not care. Encoding that rule next to each
  config makes "KNN on raw scales" a wiring mistake rather than a silent
  quality regression.

* Single source of truth reused by the training script (Phase 4), the model
  registry (Phase 5) and tests (Phase 8).

WHY the searched hyperparameters are minimal (docs/ARCHITECTURE.md §5.4):
- Decision Tree depth grid {3, 5, 10, default}: the architecture explicitly
  asks for depth tuning; the tree's unpruned default overfits badly, so the
  grid is the smallest one that brackets shallow/medium/full depth.
- class_weight="balanced" variants wherever the algorithm supports it
  (promised in docs/PREPROCESSING.md §9): costs nothing extra in data and
  tests whether rebalancing the loss helps the minority classes.
- Everything else stays at sensible defaults. A systematic hyperparameter
  search (grid/Bayesian) is deliberately NOT part of Phase 4 — it needs a
  nested-CV protocol to stay honest and is a later-phase decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class ModelConfig:
    """One candidate in the comparison. `make_estimator` returns a FRESH
    unfitted estimator on every call — sklearn estimators are stateful after
    fit, so reusing one across CV folds would silently carry fitted state
    (means, trees) from fold to fold."""

    id: str                                   # stable key for CSV/report/registry
    label: str                                # human-readable name
    make_estimator: Callable[[], object]      # zero-arg factory → new estimator
    scale_numeric: bool                       # preprocessing variant (Part F)
    params_str: str = "sklearn defaults"      # short description for reports
    notes: str = ""                           # why this candidate exists


def _cfg(id_: str, label: str, cls, scale: bool, *, params: str = "", notes: str = "", **kwargs) -> ModelConfig:
    """Small builder so every entry reads as data, not ceremony.

    kwargs go straight into the estimator constructor; a lambda closes over
    them (late binding is not a risk here — kwargs are captured per call)."""
    return ModelConfig(
        id=id_,
        label=label,
        make_estimator=lambda: cls(**kwargs),
        scale_numeric=scale,
        params_str=params or ", ".join(f"{k}={v!r}" for k, v in kwargs.items()) or "sklearn defaults",
        notes=notes,
    )


MODEL_CONFIGS: tuple[ModelConfig, ...] = (
    # --- the bar to beat (architecture §5.4 item 1) -------------------------
    _cfg(
        "dummy_majority", "Dummy (majority)", DummyClassifier, False,
        strategy="most_frequent", random_state=42,
        notes="always predicts the majority class; defines the accuracy bar "
              "(49.9%) that every real model must beat honestly",
    ),
    # --- linear model -------------------------------------------------------
    _cfg(
        "logreg", "Logistic Regression", LogisticRegression, True,
        max_iter=2000, random_state=42,
        notes="multinomial LR (lbfgs); primary linear baseline, coefficients "
              "are interpretable",
    ),
    _cfg(
        "logreg_balanced", "Logistic Regression (balanced)", LogisticRegression, True,
        max_iter=2000, random_state=42, class_weight="balanced",
        notes="inverse-frequency class weights test whether the minority "
              "classes (Enrolled 17.9%) need a stronger loss term",
    ),
    # --- instance-based -----------------------------------------------------
    _cfg(
        "knn5", "KNN (k=5)", KNeighborsClassifier, True,
        n_neighbors=5,
        notes="distance-based → requires scaled inputs; small k = low bias, "
              "high variance reference point",
    ),
    # --- generative baseline ------------------------------------------------
    _cfg(
        "gauss_nb", "Gaussian NB", GaussianNB, False,
        notes="fast generative baseline; assumes feature independence, which "
              "one-hot + grade features violate — expected to underperform, "
              "kept as the classic 'naive' reference",
    ),
    # --- single tree (pre-declared depth grid, architecture §5.4 item 3) ----
    _cfg(
        "decision_tree", "Decision Tree (default)", DecisionTreeClassifier, False,
        random_state=42,
        notes="fully grown (depth=None) — the overfitting reference",
    ),
    _cfg(
        "decision_tree_balanced", "Decision Tree (balanced)", DecisionTreeClassifier, False,
        random_state=42, class_weight="balanced",
        notes="class-weight variant of the default tree",
    ),
    _cfg(
        "decision_tree_d3", "Decision Tree (depth=3)", DecisionTreeClassifier, False,
        random_state=42, max_depth=3,
        notes="shallowest grid point: high bias, highly interpretable rules",
    ),
    _cfg(
        "decision_tree_d5", "Decision Tree (depth=5)", DecisionTreeClassifier, False,
        random_state=42, max_depth=5,
        notes="middle grid point",
    ),
    _cfg(
        "decision_tree_d10", "Decision Tree (depth=10)", DecisionTreeClassifier, False,
        random_state=42, max_depth=10,
        notes="deep grid point approaching the unpruned default",
    ),
    # --- margin-based -------------------------------------------------------
    _cfg(
        "svm_rbf", "SVM (RBF)", SVC, True,
        kernel="rbf",
        notes="non-linear margin; scaled input mandatory; decision scores "
              "used for the binary view (probability=False keeps fits fast)",
    ),
    _cfg(
        "svm_rbf_balanced", "SVM (RBF, balanced)", SVC, True,
        kernel="rbf", class_weight="balanced",
        notes="class-weight variant of the RBF SVM",
    ),
    _cfg(
        "svm_linear", "SVM (linear)", SVC, True,
        kernel="linear",
        notes="linear margin counterpart of logreg with hinge loss — "
              "agreement between the two is a useful sanity signal",
    ),
    # --- ensemble stretch baseline (architecture §5.4 item 7) ---------------
    _cfg(
        "random_forest", "Random Forest", RandomForestClassifier, False,
        n_estimators=200, random_state=42, n_jobs=-1,
        notes="bagged trees; 200 estimators (> sklearn default 100) to reduce "
              "ensemble variance at negligible cost",
    ),
    _cfg(
        "random_forest_balanced", "Random Forest (balanced)", RandomForestClassifier, False,
        n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced",
        notes="class-weight variant; RF also supports class_weight='balanced_subsample'",
    ),
)

CONFIG_BY_ID: dict[str, ModelConfig] = {c.id: c for c in MODEL_CONFIGS}
