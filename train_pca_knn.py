"""
Train and evaluate the PCA(0.90) + KNN pipeline on Baseline + Advanced.

Outputs (written to `results/pca_knn/`):
    - metrics.json                    (extras: pca_components_retained = 1 / 1)
    - predictions_baseline.npy
    - predictions_advanced.npy
    - confusion_matrix.png
    - roc_curve.png

This is the cautionary-tale model. PCA's `n_components_` after fitting
turns out to be **1** on both feature matrices: the outlier-dominated
nutrition columns hijack 90% of the variance budget, projecting all 678
(or 687) input columns into one dimension before KNN ever sees the data.
The diagnostic-driven fix lives in `train_pca_knn_improved.py`; see
README §4 Lesson #5.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

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


MODEL_SLUG    = "pca_knn"
MODEL_NAME    = "PCA(0.90) + KNN"
DISPLAY_NAME  = "PCA(0.90) + KNN"

PCA_VARIANCE = 0.90
KNN_NEIGHBORS = 5
MODEL_CONFIG = {
    "pca": {
        "n_components": PCA_VARIANCE,
        "random_state": RANDOM_STATE,
    },
    "knn": {
        "n_neighbors": KNN_NEIGHBORS,
        "n_jobs":      -1,
    },
}


def _build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("pca", PCA(**MODEL_CONFIG["pca"])),
            ("knn", KNeighborsClassifier(**MODEL_CONFIG["knn"])),
        ]
    )


def main() -> None:
    print("=" * 72)
    print(f"  TRAIN — {DISPLAY_NAME}   (random_state = {RANDOM_STATE})")
    print("=" * 72)

    datasets, y_train, y_test = load_preprocessed()
    per_ds_results = {}
    pca_components_per_dataset: dict = {}

    for ds_name in DATASETS:
        X_train, X_test = datasets[ds_name]
        model = _build_model()
        result = fit_and_score(model, X_train, y_train, X_test, y_test)
        per_ds_results[ds_name] = result

        pca_stage: PCA = result["model"].named_steps["pca"]
        n_components = int(pca_stage.n_components_)
        pca_components_per_dataset[ds_name] = n_components

        print_dataset_block(ds_name, X_train.shape, result)
        print(f"     PCA components retained for 90% variance : {n_components}")

        save_predictions(MODEL_SLUG, ds_name, result["y_pred"])

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

    payload = build_metrics_payload(
        model_name=MODEL_NAME,
        display_name=DISPLAY_NAME,
        model_config=MODEL_CONFIG,
        n_train=len(y_train),
        n_test=len(y_test),
        random_state=RANDOM_STATE,
        per_dataset_results=per_ds_results,
        extras={
            "pca_components_retained": pca_components_per_dataset,
            "pca_variance_threshold":  PCA_VARIANCE,
            "knn_n_neighbors":         KNN_NEIGHBORS,
            "roc_auc_advanced":        auc,
        },
    )
    metrics_path = save_metrics(MODEL_SLUG, payload)
    print(f"\n  Wrote {metrics_path.relative_to(metrics_path.parent.parent.parent)}")


if __name__ == "__main__":
    main()
