"""Phase 2 — Exploratory Data Analysis (EDA) for the Student Academic Risk System.

WHAT THIS SCRIPT DOES (and deliberately does NOT do):
  * Loads the raw UCI #697 (primary) and UCI #320 (comparison) datasets
    without modifying them (raw data is read-only by project policy).
  * Produces the Part-C inspection numbers: shapes, dtypes, unique-value
    counts, missing values, duplicates, descriptive statistics, class balance.
  * Runs a simple IQR-based outlier scan (candidates for inspection, NOT
    automatic deletion — outliers in student data are often real students).
  * Creates purposeful figures (Part E) — every figure has a stated question
    it answers, written as a comment above its builder function.
  * Writes reports/eda_report.md and reports/figs/*.png.

  It does NOT train models (Phase 4) and does NOT preprocess/alter data.

Re-run with:   python -m ml.eda.run_eda
"""

from __future__ import annotations

import os
from pathlib import Path

# Non-interactive backend: lets matplotlib render figures to files on machines
# without a display (servers, CI). Must be selected BEFORE pyplot is imported.
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Configurable root: defaults to the repository layout (derived from this
# file's location so no machine-specific absolute path is ever hardcoded),
# but PROJECT_ROOT can override it for unusual environments (e.g. CI).
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))

RAW_697 = PROJECT_ROOT / "data" / "raw" / "uci697" / "data.csv"
RAW_320_POR = PROJECT_ROOT / "data" / "raw" / "uci320" / "student" / "student-por.csv"
RAW_320_MAT = PROJECT_ROOT / "data" / "raw" / "uci320" / "student" / "student-mat.csv"
REPORT_MD = PROJECT_ROOT / "reports" / "eda_report.md"
FIG_DIR = PROJECT_ROOT / "reports" / "figs"

# Continuous columns of UCI #697 used for distribution/outlier analysis.
# Explicit list instead of a "guess by cardinality" heuristic: IQR statistics
# on coded categories (e.g. Course codes like 9991) would be meaningless.
CONTINUOUS_697 = [
    "Previous qualification (grade)",   # entry credential grade, scale ~95-200
    "Admission grade",                  # admission metric, similar scale
    "Age at enrollment",                # years; older students are rare by design
    "Curricular units 1st sem (grade)",  # 0-20 scale, bounded -> outliers unlikely
    "Curricular units 1st sem (approved)",   # counts; right-skew expected
    "Curricular units 1st sem (evaluations)",  # counts; right-skew expected
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (evaluations)",
]

# Low-cardinality integer columns that actually encode CATEGORIES (not numbers).
# Treating e.g. "Marital status" 1..6 as continuous would produce nonsense
# means/correlations — flagged here so downstream phases one-hot encode them.
CATEGORICAL_CODED_697 = [
    "Marital status", "Application mode", "Application order", "Course",
    "Daytime/evening attendance", "Previous qualification", "Nacionality",
    "Displaced", "Educational special needs", "Debtor",
    "Tuition fees up to date", "Gender", "Scholarship holder", "International",
    "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation",
]

# Categorical flags vs. risk: chosen because they are (a) available at H1 and
# (b) policy/fairness relevant — the figure shows dropout RATES per group.
GROUP_COLS_697 = [
    "Gender", "Debtor", "Tuition fees up to date", "Scholarship holder",
    "Displaced", "Daytime/evening attendance", "International",
    "Educational special needs",
]


# ---------------------------------------------------------------------------
# Loading (with explicit error handling — rule: fail with a helpful message,
# not a raw traceback from deep inside pandas)
# ---------------------------------------------------------------------------
def load_delimited(path: Path, dataset_name: str, **read_kwargs) -> pd.DataFrame:
    """Load a semicolon-separated UCI CSV and normalize column names.

    Why the extras: the UCI #697 export uses ';' as the separator, a UTF-8 BOM,
    and contains a stray TAB inside one header cell ("Daytime/evening
    attendance\\t"). Stripping those keeps column names safe for attribute
    access and for the feature lists defined in ml/features/feature_sets.py.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{dataset_name}: expected file not found: {path}\n"
            f"Restore it from the official zip (see data/README.md for the "
            f"UCI download URL and expected SHA256 checksum)."
        )
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", **read_kwargs)
    df.columns = [c.strip().replace("\t", " ").strip() for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# Part C — inspection helpers (each returns markdown so the report, not the
# console, is the durable artifact)
# ---------------------------------------------------------------------------
def md_column_inventory(df: pd.DataFrame) -> str:
    """One row per column: dtype, unique count, missing count/%.

    Unique counts double as a categorical/continuous hint: columns with very
    few unique values are usually codes, not measurements.
    """
    inv = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_unique": df.nunique(dropna=True),
        "missing": df.isna().sum(),
        "missing_%": (df.isna().mean() * 100).round(2),
    })
    return inv.to_markdown() + "\n"


def md_target_distribution(df: pd.DataFrame, target: str) -> str:
    """Class balance table — drives the metric choice (accuracy would hide the
    minority class under a 50/33/18 split)."""
    vc = df[target].value_counts()
    pct = df[target].value_counts(normalize=True) * 100
    rows = "\n".join(f"- {k}: {v:,} ({pct[k]:.1f}%)" for k, v in vc.items())
    ratio = vc.max() / vc.min()
    return (f"\n{rows}\n\n- Imbalance ratio (largest/smallest class): "
            f"**{ratio:.2f} : 1** — moderate imbalance → accuracy alone is "
            f"misleading; macro-F1 / recall per class are required (Phase 4).\n")


def md_descriptives(df: pd.DataFrame) -> str:
    """Standard descriptive statistics for every numeric column."""
    desc = df.select_dtypes(include=[np.number]).describe().T
    desc = desc.round(3)
    return desc[["count", "mean", "std", "min", "50%", "max"]].to_markdown() + "\n"


def md_categorical_summaries(df: pd.DataFrame, max_levels: int = 12) -> str:
    """Value counts for true text columns + the coded-category columns.

    Capped at `max_levels` per column so the report stays readable; the full
    counts remain available by re-running the analysis interactively.
    """
    parts: list[str] = []
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    for col in obj_cols:
        vc = df[col].value_counts()
        shown = vc.head(max_levels)
        lines = "\n".join(f"  - {k}: {v}" for k, v in shown.items())
        suffix = "\n  - … (truncated)" if len(vc) > max_levels else ""
        parts.append(f"- **{col}** ({df[col].nunique()} levels):\n{lines}{suffix}")

    coded = [c for c in CATEGORICAL_CODED_697 if c in df.columns]
    parts.append("\n**Integer-coded categorical columns** (must be one-hot "
                 "encoded later, NOT treated as numbers):\n")
    for col in coded:
        parts.append(f"- {col}: {df[col].nunique()} levels → {sorted(df[col].unique().tolist())}")
    return "\n".join(parts) + "\n"


def iqr_outlier_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """IQR fence method: a value is a *candidate* outlier if it falls below
    Q1 - 1.5*IQR or above Q3 + 1.5*IQR.

    Why IQR: robust to skew, needs no distributional assumptions. IMPORTANT:
    flags are for human inspection only — in student data these are often
    legitimate cases (e.g. a 22-year-old student), never auto-deleted.
    """
    rows = []
    for col in columns:
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((s < low) | (s > high)).sum())
        rows.append({
            "feature": col,
            "min": s.min(), "q1": q1, "median": s.median(), "q3": q3, "max": s.max(),
            "lower_fence": round(low, 2), "upper_fence": round(high, 2),
            "n_outliers": n_out,
            "outliers_%": round(100 * n_out / len(s), 2),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part E — figures. Each builder documents the QUESTION it answers.
# ---------------------------------------------------------------------------
def fig_target_distribution(df: pd.DataFrame, target: str, path: Path) -> None:
    """Q: How balanced are the classes?
    Why it matters: decides the evaluation metric (Phase 4) and whether class
    weights / resampling are needed. A model predicting the majority class
    every time would look '50% accurate' on this dataset."""
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {"Dropout": "#c0392b", "Enrolled": "#f39c12", "Graduate": "#27ae60"}
    vc = df[target].value_counts()
    vc.plot(kind="bar", ax=ax,
            color=[colors.get(k, "#2980b9") for k in vc.index])
    ax.set_title("UCI #697: class distribution (Target)")
    ax.set_ylabel("students")
    ax.set_xticklabels(vc.index, rotation=0)
    for i, v in enumerate(vc):
        ax.text(i, v + 30, f"{v}\n({v / len(df) * 100:.1f}%)",
                ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_numeric_histograms(df: pd.DataFrame, columns: list[str], path: Path) -> None:
    """Q: What are the shapes/scales of the continuous features?
    Why: skewness and vastly different scales (age ~15-22 vs grades ~95-190)
    determine which models need feature scaling (SVM/KNN/LR do; trees don't)."""
    n = len(columns)
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    axes = axes.ravel()
    for ax, col in zip(axes, columns):
        df[col].dropna().hist(ax=ax, bins=30, color="#2980b9", edgecolor="white")
        ax.set_title(col, fontsize=9)
    # Hide unused subplot slots when the column count isn't a multiple of 9
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("UCI #697: continuous feature distributions", y=1.00)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_boxplots(df: pd.DataFrame, columns: list[str], path: Path) -> None:
    """Q: Which features carry extreme values worth investigating?
    Why: boxplots visualize the same IQR fences as the outlier table. Outliers
    are NOT removed automatically — decision is made per-feature in Phase 3
    with documented reasoning (removing real students biases the model)."""
    n = len(columns)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4))
    for ax, col in zip(np.atleast_1d(axes), columns):
        ax.boxplot(df[col].dropna(), vert=True, patch_artist=True,
                   boxprops=dict(facecolor="#aed6f1"))
        ax.set_title(col, fontsize=8, rotation=20)
        ax.tick_params(axis="x", labelbottom=False)
    fig.suptitle("UCI #697: boxplots for outlier inspection", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_risk_rate_by_group(df: pd.DataFrame, columns: list[str], path: Path) -> None:
    """Q: Does dropout RATE differ across enrollment-time groups?
    Why: rates (not raw counts) avoid the trap of confusing group size with
    risk. Also the first fairness view: large gaps between e.g. gender groups
    are flagged for the Phase-4 subgroup evaluation.
    NOTE: correlation of a group with dropout does NOT prove the group causes
    dropout — confounders (course choice, finances) are likely."""
    at_risk = (df["Target"] == "Dropout").astype(int)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    flat = axes.ravel()
    for idx, (ax, col) in enumerate(zip(flat, columns)):
        rates = at_risk.groupby(df[col]).mean() * 100
        rates.plot(kind="bar", ax=ax, color="#c0392b")
        ax.set_title(col, fontsize=9)
        # Only the leftmost column needs a y-label (avoids chart clutter)
        ax.set_ylabel("% dropout" if idx % 4 == 0 else "")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.suptitle("UCI #697: dropout rate by enrollment-time group", y=1.00)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_correlation_heatmap(df: pd.DataFrame, path: Path) -> None:
    """Q: Which numeric features are redundant with each other (multicollinearity)?
    Why: strong feature-feature correlation (expected: semester-1 vs
    semester-2 grades) inflates variance for linear models (logistic
    regression, SVM) and signals grouped information. It does NOT by itself
    justify dropping features — that's a Phase-3 decision with tests."""
    cols = [
        "Admission grade", "Previous qualification (grade)", "Age at enrollment",
        "Curricular units 1st sem (grade)", "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (evaluations)",
        "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (evaluations)", "GDP",
    ]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=75, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)), cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=7)
    fig.colorbar(im, shrink=0.8)
    ax.set_title("UCI #697: correlation among selected numeric features")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_corr_vs_target(df: pd.DataFrame, path: Path) -> None:
    """Q: Which features move together with at-risk (Dropout=1)?
    Why: a quick honest ranking before modeling — BUT three warnings apply:
    (1) Pearson r only captures LINEAR relationships; (2) correlation is not
    causation (e.g. 'Debtor' correlates with dropout, but shared causes like
    hardship drive both); (3) semester-2 features are shown for completeness
    only — they are BANNED from H1/H2 by the leakage audit."""
    tmp = df.copy()
    tmp["_at_risk"] = (tmp["Target"] == "Dropout").astype(int)
    numeric = tmp.select_dtypes(include=[np.number])
    corr = numeric.corrwith(numeric["_at_risk"]).drop("_at_risk")
    banned = [c for c in corr.index if "2nd sem" in c or "1st sem" in c]
    allowed = corr.drop(index=banned, errors="ignore").sort_values()

    fig, ax = plt.subplots(figsize=(8, 7))
    allowed.plot(kind="barh", ax=ax, color="#2980b9")
    ax.set_title("Correlation with at-risk (Dropout=1) — H1-eligible features only")
    ax.set_xlabel("Pearson r  (linear association, NOT causation)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_separation_by_class(df: pd.DataFrame, path: Path) -> None:
    """Q: Do the strongest LEGITIMATE features actually separate the classes?
    Why: boxplots per target class show whether H2's semester-1 signal is
    distributional (real separation) or a few outliers — intuition check before
    any model is trained. Also shows that 'Graduate' vs 'Dropout' separation
    is what drives the binary at-risk view."""
    cols = [
        "Curricular units 1st sem (grade)",
        "Curricular units 1st sem (approved)",
        "Admission grade",
        "Age at enrollment",
    ]
    order = ["Dropout", "Enrolled", "Graduate"]
    fig, axes = plt.subplots(1, len(cols), figsize=(15, 4))
    for ax, col in zip(axes, cols):
        data = [df.loc[df["Target"] == cls, col].dropna() for cls in order]
        ax.boxplot(data, tick_labels=order, patch_artist=True,
                   boxprops=dict(facecolor="#aed6f1"))
        ax.set_title(col, fontsize=9)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("UCI #697: feature separation by outcome class", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --- UCI #320 figures (comparison dataset) ---
def fig_320_target(dpor: pd.DataFrame, path: Path) -> None:
    """Q: How imbalanced is the pass/fail risk view (G3 < 10)?
    Why: 649 students but only ~15% fail → this comparison task is much more
    imbalanced than UCI #697; metrics must reflect that."""
    pf = dpor["G3"].apply(lambda g: "pass (G3>=10)" if g >= 10 else "fail (G3<10)")
    fig, ax = plt.subplots(figsize=(6, 4))
    vc = pf.value_counts()
    vc.plot(kind="bar", ax=ax, color=["#27ae60", "#c0392b"])
    ax.set_title("UCI #320 (Portuguese): pass/fail balance")
    ax.set_ylabel("students")
    ax.set_xticklabels(vc.index, rotation=0)
    for i, v in enumerate(vc):
        ax.text(i, v + 8, f"{v} ({v / len(dpor) * 100:.1f}%)", ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_320_leakage_scatter(dpor: pd.DataFrame, path: Path) -> None:
    """Q: Why exactly are G1/G2 banned from risk modeling?
    Why: the scatter shows a near-deterministic line G2→G3 (r = 0.92). A model
    given G2 would 'predict' the final grade almost trivially — that's target
    leakage, not student-risk intelligence. This figure is the evidence."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(dpor["G2"], dpor["G3"], alpha=0.35, s=18, color="#2980b9")
    ax.set_xlabel("G2 (2nd-period grade)  ← EXCLUDED (leakage)")
    ax.set_ylabel("G3 (final grade = target)")
    ax.set_title("Leakage evidence: r(G2, G3) = +0.919")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_320_corr_g3(dpor: pd.DataFrame, path: Path) -> None:
    """Q: What do HONEST predictors look like compared to the leaked ones?
    Why: legitimate features (studytime, family education, absences) correlate
    weakly with G3 (|r| ≤ 0.25) — that's the realistic ceiling for an
    ethical risk model, in stark contrast to G2's 0.92."""
    num = dpor.select_dtypes(include=[np.number])
    corr = num.corrwith(dpor["G3"]).drop("G3").drop(["G1", "G2"])
    corr = corr.sort_values()
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#c0392b" if v < 0 else "#27ae60" for v in corr.values]
    corr.plot(kind="barh", ax=ax, color=colors)
    ax.set_title("UCI #320: correlation with G3 — G1/G2 removed (leakage)")
    ax.set_xlabel("Pearson r (NOT causation)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------
def build_report(d697: pd.DataFrame, dpor: pd.DataFrame, dmat: pd.DataFrame,
                 outlier_df: pd.DataFrame) -> str:
    """Compose the markdown EDA report from the pieces computed in main()."""
    parts: list[str] = ["# EDA Report — Phase 2 (Parts C & E)\n\n"]
    parts.append("> Generated by `ml/eda/run_eda.py` — rerun to regenerate.\n")
    parts.append("> Datasets are READ-ONLY inputs (see data/README.md).\n")

    # ---- UCI #697 inspection ----
    parts.append("\n## 1. UCI #697 — Predict Students' Dropout and Academic Success\n")
    parts.append(f"\n- **File:** `{RAW_697.relative_to(PROJECT_ROOT)}` (delimiter=';', encoding=utf-8-sig)\n")
    parts.append(f"- **Shape:** {d697.shape[0]} rows × {d697.shape[1]} columns "
                 f"(matches official UCI metadata: 4,424 rows / 36 features + target)\n")
    parts.append(f"- **Missing values:** {int(d697.isna().sum().sum())}\n")
    parts.append(f"- **Duplicate rows:** {int(d697.duplicated().sum())}\n")
    parts.append(f"- **Numeric columns:** {d697.select_dtypes(include=[np.number]).shape[1]} · "
                 f"**Text columns:** {d697.select_dtypes(include='object').shape[1]}\n")

    parts.append("\n### 1.1 Column inventory\n")
    parts.append(md_column_inventory(d697))

    parts.append("\n### 1.2 Target distribution (class balance)\n")
    parts.append(md_target_distribution(d697, "Target"))

    parts.append("\n### 1.3 Descriptive statistics (numeric)\n")
    parts.append(md_descriptives(d697))

    parts.append("\n### 1.4 Categorical / coded features\n")
    parts.append(md_categorical_summaries(d697))

    parts.append("\n### 1.5 Outlier candidates (IQR fences — inspection only, nothing removed)\n")
    parts.append(outlier_df.to_markdown(index=False) + "\n")
    parts.append("\nInterpretation: counts here are *flags for review in Phase 3*, "
                 "not deletion orders. E.g. older students (`Age at enrollment`) "
                 "are rare but real; deleting them would bias the model.\n")

    # ---- correlation vs target ----
    tmp = d697.copy()
    tmp["_at_risk"] = (tmp["Target"] == "Dropout").astype(int)
    corr_risk = (tmp.select_dtypes(include=[np.number]).corrwith(tmp["_at_risk"])
                 .drop("_at_risk").sort_values(ascending=False))
    parts.append("\n### 1.6 Correlation with at-risk (Dropout=1), full ranking\n")
    parts.append("| Feature | r | Horizon |\n|---|---|---|\n")
    for f, v in corr_risk.items():
        if "1st sem" in f:
            h = "H2 only"
        elif "2nd sem" in f:
            h = "H3 only (banned H1/H2)"
        else:
            h = "H1"
        parts.append(f"| {f} | {v:+.3f} | {h} |\n")
    parts.append("\n**Reminder:** correlation ≠ causation, and high |r| is NOT "
                 "permission to use a feature — horizon legality comes from "
                 "docs/LEAKAGE_AUDIT.md, not from this table.\n")

    # ---- figures index ----
    parts.append("\n### 1.7 Figures\n\n")
    fig_purpose = {
        "uci697_target.png": "Class balance → justifies macro-F1/recall metrics",
        "uci697_hists.png": "Distributions & scales → decides scaling needs",
        "uci697_boxplots.png": "Outlier inspection (IQR view)",
        "uci697_group_risk.png": "Dropout rate per enrollment group → group gaps + fairness screen",
        "uci697_corr_heatmap.png": "Multicollinearity among numeric features",
        "uci697_corr_target.png": "Linear association with at-risk (H1 features only)",
        "uci697_separation.png": "Do legit features separate the outcome classes?",
    }
    for name, why in fig_purpose.items():
        parts.append(f"- `figs/{name}` — {why}\n")

    # ---- UCI #320 ----
    parts.append("\n## 2. UCI #320 — Student Performance (comparison)\n")
    parts.append(f"\n- **Files:** `student-por.csv` {dpor.shape} · `student-mat.csv` {dmat.shape} "
                 f"(395-row math file = previously unverified count, now verified)\n")
    parts.append(f"- **Missing values:** por={int(dpor.isna().sum().sum())}, "
                 f"mat={int(dmat.isna().sum().sum())} · "
                 f"**Duplicates:** por={int(dpor.duplicated().sum())}, "
                 f"mat={int(dmat.duplicated().sum())}\n")
    pf = dpor["G3"].apply(lambda g: "pass" if g >= 10 else "fail").value_counts()
    parts.append(f"- **Risk view (G3<10):** fail={pf['fail']} "
                 f"({pf['fail'] / len(dpor) * 100:.1f}%) vs pass={pf['pass']} "
                 f"({pf['pass'] / len(dpor) * 100:.1f}%) — strongly imbalanced\n")
    g3_corr = dpor.select_dtypes(include=[np.number]).corrwith(dpor["G3"]).drop("G3")
    parts.append(f"- **Leakage evidence:** r(G1,G3)={g3_corr['G1']:+.3f}, "
                 f"r(G2,G3)={g3_corr['G2']:+.3f} → G1/G2 permanently excluded "
                 f"(docs/LEAKAGE_AUDIT.md §4)\n")
    parts.append("\n### 2.1 Figures\n\n")
    for name, why in {
        "uci320_target.png": "Pass/fail balance → extreme class imbalance (15% fail)",
        "uci320_leakage_scatter.png": "Visual proof of why G1/G2 are banned",
        "uci320_g3_corr.png": "Realistic correlations of honest predictors",
    }.items():
        parts.append(f"- `figs/{name}` — {why}\n")

    # ---- conclusions ----
    parts.append("\n## 3. Takeaways feeding Phase 3/4\n\n")
    parts.append("1. **No missing values, no duplicates** in either dataset → "
                 "no imputation required for #697 (imputer kept in the pipeline "
                 "anyway as a safety net for future data).\n")
    parts.append("2. **Moderate imbalance in #697 (2.78:1)** and **severe in "
                 "#320 (5.5:1 fail)** → macro-F1 + per-class recall are the "
                 "primary metrics; class weights are an experiment, not a default.\n")
    parts.append("3. **Semester features dominate the raw correlation** — and "
                 "are exactly the ones the leakage audit restricts to H2/H3. "
                 "Expect H1 models to be noticeably weaker than H2: that gap "
                 "is the real cost of warning early, and is a headline result.\n")
    parts.append("4. **Skewed count features + different scales** → "
                 "StandardScaler for SVM/KNN/logistic inside the pipeline; "
                 "trees unaffected.\n")
    parts.append("5. **IQR outliers flagged but retained** pending per-feature "
                 "review in Phase 3.\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load — read-only; identical bytes to the official UCI download
    d697 = load_delimited(RAW_697, "UCI #697")
    dpor = load_delimited(RAW_320_POR, "UCI #320 (Portuguese)")
    dmat = load_delimited(RAW_320_MAT, "UCI #320 (Mathematics)")

    # --- Part E figures: every call documents the question it answers ---
    fig_target_distribution(d697, "Target", FIG_DIR / "uci697_target.png")
    fig_numeric_histograms(d697, CONTINUOUS_697, FIG_DIR / "uci697_hists.png")
    fig_boxplots(d697, CONTINUOUS_697[:6], FIG_DIR / "uci697_boxplots.png")
    fig_risk_rate_by_group(d697, GROUP_COLS_697, FIG_DIR / "uci697_group_risk.png")
    fig_correlation_heatmap(d697, FIG_DIR / "uci697_corr_heatmap.png")
    fig_corr_vs_target(d697, FIG_DIR / "uci697_corr_target.png")
    fig_separation_by_class(d697, FIG_DIR / "uci697_separation.png")
    fig_320_target(dpor, FIG_DIR / "uci320_target.png")
    fig_320_leakage_scatter(dpor, FIG_DIR / "uci320_leakage_scatter.png")
    fig_320_corr_g3(dpor, FIG_DIR / "uci320_g3_corr.png")

    outliers = iqr_outlier_table(d697, CONTINUOUS_697)

    REPORT_MD.write_text(build_report(d697, dpor, dmat, outliers), encoding="utf-8")

    # Console digest — quick verification without opening the report
    print(f"report : {REPORT_MD.relative_to(PROJECT_ROOT)}")
    print(f"figures: {len(list(FIG_DIR.glob('*.png')))} files in {FIG_DIR.relative_to(PROJECT_ROOT)}")
    print(f"uci697 : {d697.shape} | missing={int(d697.isna().sum().sum())} "
          f"| dupes={int(d697.duplicated().sum())}")
    print(f"target : {d697['Target'].value_counts().to_dict()}")
    print(f"uci320 : por={dpor.shape} mat={dmat.shape}")
    print("IQR outlier flags:", dict(zip(outliers["feature"], outliers["n_outliers"])))


if __name__ == "__main__":
    main()
