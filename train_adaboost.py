"""
This script trains the AdaBoost classifier and checks how it does on the Baseline and Advanced datasets.

Outputs (written to `results/adaboost/`):
    - metrics.json
    - predictions_baseline.npy
    - predictions_advanced.npy
    - confusion_matrix.png
    - roc_curve.png

AdaBoost (with boosted decision stumps) is the only kind of non-linear model in
our linear lineup. It shows by far the biggest class asymmetry out of all the models
we tried (~0.575 FP-rate vs 0.257 FN-rate at the default threshold), which makes it
a nice counter example to the symmetric-errors story that Logistic Regression
gives us in the Phase 4 threshold sweep.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import AdaBoostClassifier

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


MODEL_SLUG    = "adaboost"
MODEL_NAME    = "AdaBoost"
DISPLAY_NAME  = "AdaBoost (n=100)"

MODEL_CONFIG = {
    "n_estimators": 100,
    "random_state": RANDOM_STATE,
}


def _build_model() -> AdaBoostClassifier:
    """
    Here we just build a fresh AdaBoost classifier with the config we set above
    :return: the AdaBoost classifier
    """
    return AdaBoostClassifier(**MODEL_CONFIG)


def main() -> None:
    """
    The main function, this is the whole training and evaluation flow for AdaBoost.
    We load the data, run over the datasets, fit a model on each one, save the
    predictions and then save the plots and the metrics json at the end.
    :return: nothing, everything gets written to disk
    """
    print("=" * 72)
    print(f"  TRAIN — {DISPLAY_NAME}   (random_state = {RANDOM_STATE})")
    print("=" * 72)

    # loading the preprocessed datasets and labels
    datasets, y_train, y_test = load_preprocessed()
    per_ds_results = {}

    # looping over Baseline and Advanced and training a fresh model on each
    for ds_name in DATASETS:
        x_train_matrix, x_test_matrix = datasets[ds_name]
        model = _build_model()
        # fit the model and get back all the scores in a dict
        result = fit_and_score(model, x_train_matrix, y_train, x_test_matrix, y_test)
        per_ds_results[ds_name] = result

        print_dataset_block(ds_name, x_train_matrix.shape, result)
        save_predictions(MODEL_SLUG, ds_name, result["y_pred"])

    # printing the delta between the two datasets
    print_delta(per_ds_results)

    # now we make the plots from the Advanced fit
    advanced_result = per_ds_results["Advanced"]
    # building the confusion matrix array from the tn/fp/fn/tp values
    cm_array = np.array([
        [advanced_result["confusion_matrix"]["tn"], advanced_result["confusion_matrix"]["fp"]],
        [advanced_result["confusion_matrix"]["fn"], advanced_result["confusion_matrix"]["tp"]],
    ])
    fig_cm = confusion_matrix_figure(cm_array, title=f"{DISPLAY_NAME} — Confusion Matrix (Advanced)")
    save_figure(MODEL_SLUG, "confusion_matrix.png", fig_cm)
    plt.close(fig_cm)

    # making the ROC curve plot from the predict_proba output
    fig_roc, auc = roc_curve_figure(
        y_test, advanced_result["proba_hit"],
        title=f"{DISPLAY_NAME} — ROC Curve (Advanced)",
        model_label=DISPLAY_NAME,
    )
    save_figure(MODEL_SLUG, "roc_curve.png", fig_roc)
    plt.close(fig_roc)

    # building the metrics payload and saving it as the final json
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
    print(f"\n  Wrote {metrics_path.relative_to(metrics_path.parent.parent.parent)}")


if __name__ == "__main__":
    main()
