"""
This script trains the Logistic Regression model and checks how it does on the Baseline and Advanced datasets.

Outputs (written to `results/logistic_regression/`):
    - metrics.json
    - predictions_baseline.npy
    - predictions_advanced.npy
    - confusion_matrix.png            annotated heatmap (Advanced fit)
    - roc_curve.png                   ROC + AUC vs predict_proba (Advanced fit)

We pin the hyperparameters here so the numbers stay the same all over the project
(Phase 3 also uses the same LR config when we do the interpretability stuff):

    solver='liblinear'   — coordinate descent, converges nicely on the
                           sparse high dimensional input we have.
    C=1.0                — the default L2 regularisation strength.
    max_iter=5000        — way more than we need, liblinear usually converges
                           long before that on this dataset.
    random_state=42      — we import it from src.train_utils.RANDOM_STATE.
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
    """
    Here we just build a fresh LogisticRegression model using the config dict above
    :return: the Logistic Regression model
    """
    return LogisticRegression(**MODEL_CONFIG)


def main() -> None:
    """
    The main function, this is where we do everything - loading the data, training
    the LR model on both datasets, saving the predictions, and at the end we make
    the plots and write the metrics json.
    :return: nothing, all the results get written to disk
    """
    print("=" * 72)
    print(f"  TRAIN — {DISPLAY_NAME}   (random_state = {RANDOM_STATE})")
    print("=" * 72)

    # loading the preprocessed datasets
    datasets, y_train, y_test = load_preprocessed()
    per_ds_results = {}

    # going over each dataset and training a fresh model
    for ds_name in DATASETS:
        x_train_matrix, x_test_matrix = datasets[ds_name]
        model = _build_model()
        # fit and score the model on this dataset
        result = fit_and_score(model, x_train_matrix, y_train, x_test_matrix, y_test)
        per_ds_results[ds_name] = result

        print_dataset_block(ds_name, x_train_matrix.shape, result)
        save_predictions(MODEL_SLUG, ds_name, result["y_pred"])

    # printing the difference between Baseline and Advanced
    print_delta(per_ds_results)

    # Plots (Advanced fit)
    advanced_result = per_ds_results["Advanced"]
    # building the confusion matrix as a numpy array from the tn/fp/fn/tp
    cm_array = np.array([
        [advanced_result["confusion_matrix"]["tn"], advanced_result["confusion_matrix"]["fp"]],
        [advanced_result["confusion_matrix"]["fn"], advanced_result["confusion_matrix"]["tp"]],
    ])
    fig_cm = confusion_matrix_figure(cm_array, title=f"{DISPLAY_NAME} — Confusion Matrix (Advanced)")
    save_figure(MODEL_SLUG, "confusion_matrix.png", fig_cm)
    plt.close(fig_cm)

    # now we make the ROC curve - LR has predict_proba so we always get an auc back
    fig_roc, auc = roc_curve_figure(
        y_test, advanced_result["proba_hit"],
        title=f"{DISPLAY_NAME} — ROC Curve (Advanced)",
        model_label=DISPLAY_NAME,
    )
    save_figure(MODEL_SLUG, "roc_curve.png", fig_roc)
    plt.close(fig_roc)
    print(f"\n  Test ROC AUC (Advanced) : {auc:.4f}")

    # building the final metrics payload and saving it as json
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
