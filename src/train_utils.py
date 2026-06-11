"""
Shared helpers for all the per-model training scripts and notebooks.

This module is the single place that owns 3 things:
    1. The "42 Guarantee" - RANDOM_STATE comes from src/_constants.py and is
       re-exported from here so every model can grab it the same way. Every
       random_state in the project (the splits, PCA, MLP validation split,
       the model __init__) is set to this exact value.
    2. The results/<model_slug>/ directory contract - every model writes into
       its own folder, and save_metrics / save_predictions / save_figure
       build the path so callers don't assemble file paths by hand.
    3. The canonical JSON payload shape (build_metrics_payload) that the
       aggregator and the master comparison notebook read.

The plot helpers (confusion_matrix_figure, roc_curve_figure) return Figure
objects so the caller can either show them inline or persist them through
save_figure.
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
from src.preprocessing import build_preprocessed_datasets


# the full public API of this module (RANDOM_STATE is a re-export from
# src._constants so `from src.train_utils import RANDOM_STATE` keeps working)
__all__ = [
    "DATASETS",
    "PROJECT_ROOT",
    "RANDOM_STATE",
    "RESULTS_DIR",
    "build_metrics_payload",
    "confusion_matrix_figure",
    "fit_and_score",
    "load_metrics",
    "load_preprocessed",
    "model_results_dir",
    "print_dataset_block",
    "print_delta",
    "roc_curve_figure",
    "save_figure",
    "save_metrics",
    "save_predictions",
    "save_test_index",
]


# Filesystem contract

# src/train_utils.py is one level deep inside the project, and results/ lives
# at the project root - so we go up TWO parents from __file__
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RESULTS_DIR: Path = PROJECT_ROOT / "results"

# the order of the 2 feature matrices - kept stable for the JSON output and
# for the .npy filename ordering
DATASETS: Tuple[str, ...] = ("Baseline", "Advanced")


def model_results_dir(model_slug: str) -> Path:
    """
    This function returns the results/<slug>/ path, creating it if needed.
    :param model_slug: the short name of the model (used as the folder name)
    :return: the Path to the folder
    """
    if not model_slug or not isinstance(model_slug, str):
        raise ValueError(f"Invalid model_slug: {model_slug!r}")
    path = RESULTS_DIR / model_slug
    path.mkdir(parents=True, exist_ok=True)
    return path


# Data loading - thin wrapper for consistency across train scripts/notebooks

def load_preprocessed():
    """
    Thin wrapper around build_preprocessed_datasets - we just repackage the
    return value into a dict keyed by "Baseline" / "Advanced" so the train
    scripts can loop over the datasets.
    :return: (datasets_dict, y_train, y_test)
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


# Evaluation

def _confusion_to_dict(cm: np.ndarray) -> Dict[str, int]:
    """
    Convert a 2x2 sklearn confusion matrix into a labelled dict with tn/fp/fn/tp
    so it serializes nicely into the JSON payload.
    :param cm: the 2x2 confusion matrix from sklearn
    :return: a dict with tn, fp, fn, tp keys
    """
    return {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def _compute_rates(cm_dict: Dict[str, int]) -> Tuple[float, float]:
    """
    This function returns the (FP rate, FN rate) pair from a confusion dict.
    :param cm_dict: the dict from _confusion_to_dict
    :return: (fp_rate, fn_rate)
    """
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
    """
    Here we fit a fresh model on the train split, predict on test, and pack
    everything into one dict - the accuracy, F1, confusion matrix, FP/FN rates,
    y_pred, the model itself, and the proba_hit needed for the ROC plot.
    For estimators that don't expose predict_proba (like Perceptron) we fall
    back to decision_function, which is monotone-equivalent for the ROC.
    :param model: an unfitted sklearn estimator
    :param X_train: the train features
    :param y_train: the train labels
    :param X_test: the test features
    :param y_test: the test labels
    :return: a dict with accuracy / f1 / confusion_matrix / fp_rate / fn_rate /
             y_pred / proba_hit / model
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    proba_hit: Optional[np.ndarray] = None
    if hasattr(model, "predict_proba"):
        proba_hit = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        # fall back to decision_function - monotone equivalent for ROC ranking
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


# Payload assembly + persistence

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
    """
    This function packs all the per-dataset results into the canonical JSON
    payload that the aggregator + comparison notebook read.
    :param model_name: the short model name
    :param display_name: the pretty name we show in the reports
    :param model_config: a dict of the actual model hyperparameters
    :param n_train: the train set size
    :param n_test: the test set size
    :param random_state: the seed we used
    :param per_dataset_results: the dict of results, keyed by Baseline/Advanced
    :param extras: any extra fields we want to attach (optional)
    :return: the full payload dict
    """
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
    """
    Write the payload dict as JSON into results/<slug>/metrics.json.
    :param model_slug: the model folder name
    :param payload: the payload dict (usually from build_metrics_payload)
    :return: the Path the file was written to
    """
    path = model_results_dir(model_slug) / "metrics.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False, default=_json_default)
        fh.write("\n")
    return path


def save_predictions(model_slug: str, dataset: str, y_pred: np.ndarray) -> Path:
    """
    Save the per-dataset predictions as a small .npy file inside the model
    results folder. We store them as int8 because they're 0/1 labels.
    :param model_slug: the model folder name
    :param dataset: which dataset the predictions came from (Baseline/Advanced)
    :param y_pred: the prediction array
    :return: the Path the file was written to
    """
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; expected one of {DATASETS}.")
    path = model_results_dir(model_slug) / f"predictions_{dataset.lower()}.npy"
    np.save(path, np.asarray(y_pred, dtype=np.int8))
    return path


def save_test_index(model_slug: str, y_test: pd.Series) -> Path:
    """
    Save the test split's row index next to the predictions, so the .npy
    prediction files are self-describing - a consumer can align predictions
    back to recipes without having to re-derive the split, and a changed
    upstream data file shows up as an index mismatch instead of a silent
    misalignment.
    :param model_slug: the model folder name
    :param y_test: the test labels series (we persist its index)
    :return: the Path the file was written to
    """
    path = model_results_dir(model_slug) / "test_index.npy"
    np.save(path, np.asarray(y_test.index, dtype=np.int64))
    return path


def save_figure(model_slug: str, filename: str, fig) -> Path:
    """
    Save a matplotlib Figure into results/<slug>/<filename> at 200 dpi.
    :param model_slug: the model folder name
    :param filename: the file name (with extension) inside the model folder
    :param fig: the matplotlib Figure to save
    :return: the Path the figure was written to
    """
    path = model_results_dir(model_slug) / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return path


def load_metrics(model_slug: str) -> Dict[str, Any]:
    """
    Read results/<slug>/metrics.json back and return the parsed payload.
    :param model_slug: the model folder name
    :return: the parsed payload dict
    """
    path = RESULTS_DIR / model_slug / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"No metrics for slug {model_slug!r} at {path}.")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(obj: Any) -> Any:
    """
    Small helper for json.dump - numpy scalars and arrays are not JSON
    serializable by default, so here we cast them to native python types.
    :param obj: the object json.dump didn't know how to handle
    :return: a JSON friendly version of obj
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


# Plot helpers - return Figure objects, the caller decides to show + save

def confusion_matrix_figure(
    cm: np.ndarray,
    *,
    title: str = "Confusion Matrix",
    figsize: Tuple[float, float] = (5.5, 4.5),
):
    """
    Here we draw a confusion matrix heatmap and return the Figure so the
    caller can either show or save it.
    :param cm: the 2x2 confusion matrix
    :param title: the title to put on the plot
    :param figsize: the figure size
    :return: the matplotlib Figure
    """
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
    """
    Here we draw the ROC curve with the AUC and a dashed chance-diagonal.
    Returns the (Figure, auc) pair so the caller can show/save the plot and
    also stash the AUC value in the metrics payload.
    :param y_true: the true binary labels
    :param y_score: the model scores (predict_proba or decision_function)
    :param title: the title to put on the plot
    :param figsize: the figure size
    :param model_label: the label of the curve in the legend
    :return: (Figure, auc) - the Figure and the AUC as a float
    """
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


# Pretty-print helpers reused across train scripts

def print_dataset_block(ds_name: str, X_shape, result: Dict[str, Any]) -> None:
    """
    Print the per-dataset console block that every train_*.py script shares.
    :param ds_name: the dataset name (Baseline / Advanced)
    :param X_shape: the X_train shape, just for the header line
    :param result: the result dict that came out of fit_and_score
    """
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
    """
    Print the small Δ(Advanced − Baseline) summary line so we can see at a
    glance if the engineered features actually helped.
    :param per_dataset_results: the dict of results, keyed by Baseline/Advanced
    """
    if "Baseline" not in per_dataset_results or "Advanced" not in per_dataset_results:
        return
    b = per_dataset_results["Baseline"]
    a = per_dataset_results["Advanced"]
    print(
        f"\n  >> Δ (Advanced − Baseline):  "
        f"Acc {a['accuracy'] - b['accuracy']:+.4f}  |  "
        f"F1 {a['f1'] - b['f1']:+.4f}"
    )
