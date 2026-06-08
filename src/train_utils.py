"""
Shared helpers for the per-model training scripts and notebooks.

This module is the single source of truth for three concerns:

    1. **The "42 Guarantee"** — `RANDOM_STATE` is re-exported from here
       for convenience. The literal `42` is defined in exactly one place
       in the project: `src/_constants.py`. Every script and notebook
       imports it (typically as `from src.train_utils import RANDOM_STATE`
       or `from src import RANDOM_STATE`). Reproducibility audit: each
       model's `__init__` is constructed with `random_state=RANDOM_STATE`,
       the train/test split is `random_state=RANDOM_STATE`, the PCA
       stages are `random_state=RANDOM_STATE`, and the MLP's
       validation-split seed is `random_state=RANDOM_STATE`. KNN and the
       binary-tag passthrough have no `random_state` to set (they're
       deterministic).

    2. **The `results/<model_slug>/` directory contract.** Every model
       writes into its own subdirectory under `results/`. The slug is
       passed to `save_metrics(slug, ...)`, `save_predictions(slug, ...)`,
       and `save_figure(slug, ...)`; this module owns the path
       construction so callers never assemble filesystem paths by hand.

    3. **The canonical JSON payload schema** (`build_metrics_payload`).
       This is what the aggregator and the master comparison notebook
       read. Any change to its shape happens here, in one place.

The plotting helpers `confusion_matrix_figure` and `roc_curve_figure`
return matplotlib Figure objects so callers can either display them
inline (notebooks: `plt.show()` after fetching the figure) or persist
them via `save_figure(slug, filename, fig)`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from src._constants import RANDOM_STATE  # the single source of truth
from src.phase1_preprocessing import build_preprocessed_datasets


# Re-export RANDOM_STATE for `from src.train_utils import RANDOM_STATE` callers.
__all__ = ["RANDOM_STATE"]


# ---------------------------------------------------------------------------
# Filesystem contract
# ---------------------------------------------------------------------------
# `src/train_utils.py` is one level deep inside the project; results/ is at
# the project root, so we go up TWO parents from __file__.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RESULTS_DIR: Path = PROJECT_ROOT / "results"

# Conventional ordering of the two feature matrices. Stable for JSON output
# and `.npy` filename ordering.
DATASETS: Tuple[str, ...] = ("Baseline", "Advanced")


def model_results_dir(model_slug: str) -> Path:
    """Return `results/<slug>/`, creating it if needed."""
    if not model_slug or not isinstance(model_slug, str):
        raise ValueError(f"Invalid model_slug: {model_slug!r}")
    path = RESULTS_DIR / model_slug
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Data loading — thin wrapper for consistency across train scripts/notebooks
# ---------------------------------------------------------------------------
def load_preprocessed():
    """Return (datasets_dict, y_train, y_test).

    `datasets_dict` is keyed by "Baseline" / "Advanced" with values
    `(X_train, X_test)`.
    """
    (
        X_train_b, X_test_b,
        X_train_a, X_test_a,
        y_train, y_test,
    ) = build_preprocessed_datasets(verbose=False)

    datasets = {
        "Baseline": (X_train_b, X_test_b),
        "Advanced": (X_train_a, X_test_a),
    }
    return datasets, y_train, y_test


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _confusion_to_dict(cm: np.ndarray) -> Dict[str, int]:
    """Convert a 2x2 sklearn confusion matrix into a labelled dict."""
    return {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def _compute_rates(cm_dict: Dict[str, int]) -> Tuple[float, float]:
    """Return (FP-rate, FN-rate) from a confusion-matrix dict."""
    tn, fp, fn, tp = cm_dict["tn"], cm_dict["fp"], cm_dict["fn"], cm_dict["tp"]
    fp_rate = fp / (tn + fp) if (tn + fp) else 0.0   # P(pred=Hit | true=Miss)
    fn_rate = fn / (fn + tp) if (fn + tp) else 0.0   # P(pred=Miss | true=Hit)
    return fp_rate, fn_rate


def fit_and_score(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Fit a fresh model, predict on test, return metrics + y_pred + model.

    Also captures `proba_hit` (P(class=1)) when the estimator exposes
    `predict_proba` — needed for the ROC plot. For Perceptron and other
    estimators that only expose `decision_function`, the value is left
    as the decision_function output (monotone-equivalent for ROC ranking).
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    proba_hit: Optional[np.ndarray] = None
    if hasattr(model, "predict_proba"):
        proba_hit = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        proba_hit = model.decision_function(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_dict = _confusion_to_dict(cm)
    fp_rate, fn_rate = _compute_rates(cm_dict)

    return {
        "accuracy":         float(accuracy_score(y_test, y_pred)),
        "f1":               float(f1_score(y_test, y_pred)),
        "confusion_matrix": cm_dict,
        "fp_rate":          fp_rate,
        "fn_rate":          fn_rate,
        "y_pred":           y_pred,
        "proba_hit":        proba_hit,
        "model":            model,
    }


# ---------------------------------------------------------------------------
# Payload assembly + persistence
# ---------------------------------------------------------------------------
def build_metrics_payload(
    *,
    model_name: str,
    display_name: str,
    model_config: Dict[str, Any],
    n_train: int,
    n_test: int,
    random_state: int,
    per_dataset_results: Dict[str, Dict[str, Any]],
    extras: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the canonical metrics JSON payload."""
    datasets_block: Dict[str, Dict[str, Any]] = {}
    for ds in DATASETS:
        if ds not in per_dataset_results:
            continue
        r = per_dataset_results[ds]
        datasets_block[ds] = {
            "accuracy":         r["accuracy"],
            "f1":               r["f1"],
            "confusion_matrix": r["confusion_matrix"],
            "fp_rate":          r["fp_rate"],
            "fn_rate":          r["fn_rate"],
        }

    return {
        "model_name":    model_name,
        "display_name":  display_name,
        "model_config":  model_config,
        "n_train":       int(n_train),
        "n_test":        int(n_test),
        "random_state":  int(random_state),
        "datasets":      datasets_block,
        "extras":        extras or {},
    }


def save_metrics(model_slug: str, payload: Dict[str, Any]) -> Path:
    """Write the payload as JSON into `results/<slug>/metrics.json`."""
    path = model_results_dir(model_slug) / "metrics.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False, default=_json_default)
        fh.write("\n")
    return path


def save_predictions(model_slug: str, dataset: str, y_pred: np.ndarray) -> Path:
    """Save per-dataset predictions to `results/<slug>/predictions_<dataset>.npy`."""
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; expected one of {DATASETS}.")
    path = model_results_dir(model_slug) / f"predictions_{dataset.lower()}.npy"
    np.save(path, np.asarray(y_pred, dtype=np.int8))
    return path


def save_figure(model_slug: str, filename: str, fig) -> Path:
    """Save a matplotlib Figure into `results/<slug>/<filename>` at 200 dpi."""
    path = model_results_dir(model_slug) / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return path


def load_metrics(model_slug: str) -> Dict[str, Any]:
    """Read `results/<slug>/metrics.json` and return the parsed payload."""
    path = RESULTS_DIR / model_slug / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"No metrics for slug {model_slug!r} at {path}.")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(obj: Any) -> Any:
    """Cast numpy scalars / arrays so `json.dump` doesn't choke on them."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


# ---------------------------------------------------------------------------
# Plot helpers — return Figure objects; callers display + save as needed
# ---------------------------------------------------------------------------
def confusion_matrix_figure(
    cm: np.ndarray,
    *,
    title: str = "Confusion Matrix",
    figsize: Tuple[float, float] = (5.5, 4.5),
):
    """Render a confusion-matrix heatmap and return the Figure."""
    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Miss (0)", "Hit (1)"],
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    return fig


def roc_curve_figure(
    y_true: Iterable[int],
    y_score: Iterable[float],
    *,
    title: str = "ROC Curve",
    figsize: Tuple[float, float] = (5.5, 4.5),
    model_label: str = "Model",
):
    """Render an ROC curve with AUC and a chance-diagonal. Returns (Figure, AUC)."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = float(roc_auc_score(y_true, y_score))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, lw=2.0, label=f"{model_label} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1.0,
            label="Random (AUC = 0.50)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, auc


# ---------------------------------------------------------------------------
# Pretty-print helpers reused across train scripts
# ---------------------------------------------------------------------------
def print_dataset_block(ds_name: str, X_shape, result: Dict[str, Any]) -> None:
    """Render the per-dataset console output every train_*.py shares."""
    cm = result["confusion_matrix"]
    print(f"\n  --- {ds_name}  (X_train: {X_shape}) ---")
    print(f"     Test Accuracy : {result['accuracy']:.4f}")
    print(f"     Test F1-Score : {result['f1']:.4f}")
    print(f"     Confusion Matrix:")
    print(f"                       Pred:Miss  Pred:Hit")
    print(f"        True:Miss    {cm['tn']:>9}  {cm['fp']:>8}   "
          f"FP rate = {result['fp_rate']:.4f}")
    print(f"        True:Hit     {cm['fn']:>9}  {cm['tp']:>8}   "
          f"FN rate = {result['fn_rate']:.4f}")


def print_delta(per_dataset_results: Dict[str, Dict[str, Any]]) -> None:
    """Print the Δ(Advanced − Baseline) summary line."""
    if "Baseline" not in per_dataset_results or "Advanced" not in per_dataset_results:
        return
    b = per_dataset_results["Baseline"]
    a = per_dataset_results["Advanced"]
    print(
        f"\n  >> Δ (Advanced − Baseline):  "
        f"Acc {a['accuracy'] - b['accuracy']:+.4f}  |  "
        f"F1 {a['f1'] - b['f1']:+.4f}"
    )
