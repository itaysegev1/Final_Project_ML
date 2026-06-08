"""
Improved PCA(0.90) + KNN — drops the 4 nutrition columns BEFORE PCA so the
projection isn't hijacked by their outlier-dominated variance.

Outputs (written to `results/pca_knn_improved/`):
    - metrics.json                    (extras: pca_components_retained ~203/182)
    - predictions_baseline.npy
    - predictions_advanced.npy
    - confusion_matrix.png
    - roc_curve.png

Diagnostic that motivated this script: the original `train_pca_knn.py`
retains exactly 1 PCA component for 90% variance (see
`results/pca_knn/metrics.json::extras.pca_components_retained`). Dropping
the four nutrition columns ('calories', 'protein', 'fat', 'sodium')
before PCA expands the retained search space to ~200 components, lifting
KNN's accuracy by ~4.5–5 percentage points on the test set. See README
§4 Lesson #5 for the full story.
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
    print("=" * 72)
    print(f"  TRAIN — {DISPLAY_NAME}   (random_state = {RANDOM_STATE})")
    print("=" * 72)
    print(f"  Dropping {list(NUTRITION_COLS)} before PCA.")

    datasets, y_train, y_test = load_preprocessed()
    per_ds_results = {}
    pca_components_per_dataset: dict = {}
    cols_before_drop_per_dataset: dict = {}
    cols_after_drop_per_dataset: dict = {}

    for ds_name in DATASETS:
        X_train, X_test = datasets[ds_name]
        n_cols_in = X_train.shape[1]
        cols_before_drop_per_dataset[ds_name] = n_cols_in

        missing = [c for c in NUTRITION_COLS if c not in X_train.columns]
        if missing:
            raise KeyError(
                f"{ds_name} matrix missing expected nutrition columns: {missing}"
            )

        model = _build_model()
        result = fit_and_score(model, X_train, y_train, X_test, y_test)
        per_ds_results[ds_name] = result

        pca_stage: PCA = result["model"].named_steps["pca"]
        n_components = int(pca_stage.n_components_)
        pca_components_per_dataset[ds_name] = n_components
        cols_after_drop_per_dataset[ds_name] = n_cols_in - len(NUTRITION_COLS)

        print_dataset_block(ds_name, X_train.shape, result)
        print(f"     Columns before drop : {n_cols_in}")
        print(f"     Columns after drop  : {n_cols_in - len(NUTRITION_COLS)}")
        print(f"     PCA components retained for 90% variance : {n_components}  "
              f"(was 1 in train_pca_knn.py)")

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
    print(f"\n  Wrote {metrics_path.relative_to(metrics_path.parent.parent.parent)}")


if __name__ == "__main__":
    main()
