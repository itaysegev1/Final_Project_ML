"""
This script is the improved version of PCA(0.90) + KNN, here we drop the 4 nutrition columns
BEFORE PCA so the projection isn't hijacked by their outlier-dominated variance.

The diagnostic that motivated this script, the original train_pca_knn.py retains exactly 1
PCA component for 90% variance (see results/pca_knn/metrics.json::extras.pca_components_retained).
Dropping the four nutrition columns ('calories', 'protein', 'fat', 'sodium') before PCA expands
the retained search space to about 200 components, and lifts KNN's accuracy by about 4.5-5
percentage points on the test set. See README section 4 Lesson #5 for the full story.

Outputs are written to results/pca_knn_improved/:
    - metrics.json                    (extras: pca_components_retained ~203/182)
    - predictions_baseline.npy
    - predictions_advanced.npy
    - confusion_matrix.png
    - roc_curve.png
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from src.train_utils import (
    DATASETS,
    PROJECT_ROOT,
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
    save_test_index,
)


MODEL_SLUG    = "pca_knn_improved"
MODEL_NAME    = "PCA(0.90) + KNN (Improved)"
DISPLAY_NAME  = "PCA(0.90) + KNN (Improved)"

NUTRITION_COLS = ("calories", "protein", "fat", "sodium")
PCA_VARIANCE = 0.90
KNN_NEIGHBORS = 5
MODEL_CONFIG = {
    "dropped_columns": list(NUTRITION_COLS),
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
    """
    Here we build the improved pipeline, first step is a ColumnTransformer that drops the
    nutrition columns, then PCA, then KNN.
    :return: the pipeline object
    """
    return Pipeline(
        steps=[
            (
                "drop_nutrition",
                ColumnTransformer(
                    transformers=[
                        ("drop_nutrition_cols", "drop", list(NUTRITION_COLS)),
                    ],
                    remainder="passthrough",
                    verbose_feature_names_out=False,
                ),
            ),
            ("pca", PCA(**MODEL_CONFIG["pca"])),
            ("knn", KNeighborsClassifier(**MODEL_CONFIG["knn"])),
        ]
    )


def main() -> None:
    """
    The main function of the improved script, we loop over the datasets, drop the nutrition
    columns inside the pipeline, fit and score, capture the number of PCA components retained,
    save predictions and plots for the Advanced set and then write the metrics.json with all
    the extras (dropped columns, columns before/after the drop, pca components, etc).
    """
    print("=" * 72)
    print(f"  TRAIN — {DISPLAY_NAME}   (random_state = {RANDOM_STATE})")
    print("=" * 72)
    print(f"  Dropping {list(NUTRITION_COLS)} before PCA.")

    # loading the preprocessed Baseline + Advanced matrices and the labels
    datasets, y_train, y_test = load_preprocessed()
    per_ds_results = {}
    pca_components_per_dataset: dict = {}
    cols_before_drop_per_dataset: dict = {}
    cols_after_drop_per_dataset: dict = {}

    for ds_name in DATASETS:
        X_train, X_test = datasets[ds_name]
        n_cols_in = X_train.shape[1]
        cols_before_drop_per_dataset[ds_name] = n_cols_in

        # we want to fail loud if the nutrition columns are not there, otherwise the drop is a no-op
        missing_cols = [c for c in NUTRITION_COLS if c not in X_train.columns]
        if missing_cols:
            raise KeyError(
                f"{ds_name} matrix missing expected nutrition columns: {missing_cols}"
            )

        model = _build_model()
        result = fit_and_score(model, X_train, y_train, X_test, y_test)
        per_ds_results[ds_name] = result

        # grab the fitted PCA step so we can see how many components made the 90% cut this time
        pca_stage: PCA = result["model"].named_steps["pca"]
        n_components_kept = int(pca_stage.n_components_)
        pca_components_per_dataset[ds_name] = n_components_kept
        cols_after_drop_per_dataset[ds_name] = n_cols_in - len(NUTRITION_COLS)

        print_dataset_block(ds_name, X_train.shape, result)
        print(f"     Columns before drop : {n_cols_in}")
        print(f"     Columns after drop  : {n_cols_in - len(NUTRITION_COLS)}")
        print(f"     PCA components retained for 90% variance : {n_components_kept}  "
              f"(was 1 in train_pca_knn.py)")

        # save the predictions to disk for later analysis
        save_predictions(MODEL_SLUG, ds_name, result["y_pred"])

    # printing the delta between Baseline and Advanced
    print_delta(per_ds_results)
    save_test_index(MODEL_SLUG, y_test)

    # now the plots for the Advanced dataset
    adv = per_ds_results["Advanced"]
    cm_array = np.array([
        [adv["confusion_matrix"]["tn"], adv["confusion_matrix"]["fp"]],
        [adv["confusion_matrix"]["fn"], adv["confusion_matrix"]["tp"]],
    ])
    fig_cm = confusion_matrix_figure(cm_array, title=f"{DISPLAY_NAME} — Confusion Matrix (Advanced)")
    save_figure(MODEL_SLUG, "confusion_matrix.png", fig_cm)
    plt.close(fig_cm)

    # ROC curve + AUC for the Advanced set
    fig_roc, auc = roc_curve_figure(
        y_test, adv["proba_hit"],
        title=f"{DISPLAY_NAME} — ROC Curve (Advanced)",
        model_label=DISPLAY_NAME,
    )
    save_figure(MODEL_SLUG, "roc_curve.png", fig_roc)
    plt.close(fig_roc)

    # building the final metrics payload that we will write to metrics.json
    payload = build_metrics_payload(
        model_name=MODEL_NAME,
        display_name=DISPLAY_NAME,
        model_config=MODEL_CONFIG,
        n_train=len(y_train),
        n_test=len(y_test),
        random_state=RANDOM_STATE,
        per_dataset_results=per_ds_results,
        extras={
            "dropped_columns":          list(NUTRITION_COLS),
            "columns_before_drop":      cols_before_drop_per_dataset,
            "columns_after_drop":       cols_after_drop_per_dataset,
            "pca_components_retained":  pca_components_per_dataset,
            "pca_variance_threshold":   PCA_VARIANCE,
            "knn_n_neighbors":          KNN_NEIGHBORS,
            "roc_auc_advanced":         auc,
        },
    )
    metrics_path = save_metrics(MODEL_SLUG, payload)
    print(f"\n  Wrote {metrics_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
