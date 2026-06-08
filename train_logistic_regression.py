"""
Train and evaluate the Logistic Regression model on Baseline + Advanced.

Outputs (written to `results/logistic_regression/`):
    - metrics.json
    - predictions_baseline.npy
    - predictions_advanced.npy
    - confusion_matrix.png            annotated heatmap (Advanced fit)
    - roc_curve.png                   ROC + AUC vs predict_proba (Advanced fit)

Hyperparameters are pinned by project convention so numbers reproduce
across the rest of the pipeline (Phase 3 uses the same LR config for
its interpretability work):

    solver='liblinear'   — coordinate descent, converges reliably on the
                           sparse high-dimensional input.
    C=1.0                — default L2 regularisation strength.
    max_iter=5000        — generous head-room; liblinear converges well
                           before this on this dataset.
    random_state=42      — imported from src.train_utils.RANDOM_STATE.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

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


MODEL_SLUG    = "logistic_regression"
MODEL_NAME    = "LogisticRegression"
DISPLAY_NAME  = "Logistic Regression"

MODEL_CONFIG = {
    "solver":       "liblinear",
    "C":            1.0,
    "max_iter":     5000,
    "random_state": RANDOM_STATE,
}


def _build_model() -> LogisticRegression:
    return LogisticRegression(**MODEL_CONFIG)


def main() -> None:
    print("=" * 72)
    print(f"  TRAIN — {DISPLAY_NAME}   (random_state = {RANDOM_STATE})")
    print("=" * 72)

    datasets, y_train, y_test = load_preprocessed()
    per_ds_results = {}

    for ds_name in DATASETS:
        X_train, X_test = datasets[ds_name]
        model = _build_model()
        result = fit_and_score(model, X_train, y_train, X_test, y_test)
        per_ds_results[ds_name] = result

        print_dataset_block(ds_name, X_train.shape, result)
        save_predictions(MODEL_SLUG, ds_name, result["y_pred"])

    print_delta(per_ds_results)

    # --- Plots (Advanced fit) --------------------------------------------
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
    print(f"\n  Test ROC AUC (Advanced) : {auc:.4f}")

    payload = build_metrics_payload(
        model_name=MODEL_NAME,
        display_name=DISPLAY_NAME,
        model_config=MODEL_CONFIG,
        n_train=len(y_train),
        n_test=len(y_test),
        random_state=RANDOM_STATE,
        per_dataset_results=per_ds_results,
        extras={"roc_auc_advanced": auc},
    )
    metrics_path = save_metrics(MODEL_SLUG, payload)
    print(f"  Wrote {metrics_path.relative_to(metrics_path.parent.parent.parent)}")


if __name__ == "__main__":
    main()
