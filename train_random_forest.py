"""
Train and evaluate the Random Forest model on Baseline + Advanced.

Outputs (written to `results/random_forest/`):
    - metrics.json                       (extras: top-20 importances)
    - predictions_baseline.npy
    - predictions_advanced.npy
    - confusion_matrix.png
    - roc_curve.png
    - feature_importance.png             top-20 horizontal bar chart (Advanced)
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.train_utils import (
    DATASETS,
    RANDOM_STATE,
    build_metrics_payload,
    confusion_matrix_figure,
    fit_and_score,
    load_preprocessed,
    print_dataset_block,
    print_delta,
    roc_curve_figure,
    save_figure,
    save_metrics,
    save_predictions,
)


MODEL_SLUG    = "random_forest"
MODEL_NAME    = "RandomForest"
DISPLAY_NAME  = "Random Forest"
TOP_K         = 20

MODEL_CONFIG = {
    "n_estimators": 200,
    "n_jobs":       -1,
    "random_state": RANDOM_STATE,
}


def _build_model() -> RandomForestClassifier:
    return RandomForestClassifier(**MODEL_CONFIG)


def _top_k_importances(model: RandomForestClassifier,
                       feature_names: pd.Index,
                       k: int) -> pd.Series:
    return (
        pd.Series(model.feature_importances_, index=feature_names)
          .sort_values(ascending=False)
          .head(k)
    )


def _feature_importance_figure(top: pd.Series):
    fig, ax = plt.subplots(figsize=(9, 8))
    top.iloc[::-1].plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Feature importance (impurity reduction, Gini)")
    ax.set_ylabel("")
    ax.set_title(f"Random Forest — Top {len(top)} Feature Importances (Advanced)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def main() -> None:
    print("=" * 72)
    print(f"  TRAIN — {DISPLAY_NAME}   (random_state = {RANDOM_STATE})")
    print("=" * 72)

    datasets, y_train, y_test = load_preprocessed()
    per_ds_results = {}
    advanced_model = None
    advanced_feature_names = None

    for ds_name in DATASETS:
        X_train, X_test = datasets[ds_name]
        model = _build_model()
        result = fit_and_score(model, X_train, y_train, X_test, y_test)
        per_ds_results[ds_name] = result

        print_dataset_block(ds_name, X_train.shape, result)
        save_predictions(MODEL_SLUG, ds_name, result["y_pred"])

        if ds_name == "Advanced":
            advanced_model = result["model"]
            advanced_feature_names = X_train.columns

    print_delta(per_ds_results)

    adv = per_ds_results["Advanced"]
    cm_array = np.array([
        [adv["confusion_matrix"]["tn"], adv["confusion_matrix"]["fp"]],
        [adv["confusion_matrix"]["fn"], adv["confusion_matrix"]["tp"]],
    ])
    fig_cm = confusion_matrix_figure(cm_array, title=f"{DISPLAY_NAME} — Confusion Matrix (Advanced)")
    save_figure(MODEL_SLUG, "confusion_matrix.png", fig_cm)
    plt.close(fig_cm)

    fig_roc, auc = roc_curve_figure(
        y_test, adv["proba_hit"],
        title=f"{DISPLAY_NAME} — ROC Curve (Advanced)",
        model_label=DISPLAY_NAME,
    )
    save_figure(MODEL_SLUG, "roc_curve.png", fig_roc)
    plt.close(fig_roc)

    # Feature importance plot — top-K
    top = _top_k_importances(advanced_model, advanced_feature_names, TOP_K)
    print(f"\n  Top {TOP_K} feature importances (Advanced):")
    for rank, (name, val) in enumerate(top.items(), 1):
        print(f"  {rank:>3}  {val:.4f}   {name}")
    fig_imp = _feature_importance_figure(top)
    save_figure(MODEL_SLUG, "feature_importance.png", fig_imp)
    plt.close(fig_imp)

    payload = build_metrics_payload(
        model_name=MODEL_NAME,
        display_name=DISPLAY_NAME,
        model_config=MODEL_CONFIG,
        n_train=len(y_train),
        n_test=len(y_test),
        random_state=RANDOM_STATE,
        per_dataset_results=per_ds_results,
        extras={
            "top_k_feature_importances": [
                {"feature": str(name), "importance": float(val)}
                for name, val in top.items()
            ],
            "feature_importance_plot": "feature_importance.png",
            "roc_auc_advanced": auc,
        },
    )
    metrics_path = save_metrics(MODEL_SLUG, payload)
    print(f"\n  Wrote {metrics_path.relative_to(metrics_path.parent.parent.parent)}")


if __name__ == "__main__":
    main()
