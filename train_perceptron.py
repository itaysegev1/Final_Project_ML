"""
This script trains the Perceptron model and checks how it does on the Baseline and Advanced datasets.

Outputs (written to `results/perceptron/`):
    - metrics.json                       canonical metrics payload
    - predictions_baseline.npy           int8 test predictions, Baseline
    - predictions_advanced.npy           int8 test predictions, Advanced
    - confusion_matrix.png               annotated heatmap (Advanced fit)
    - roc_curve.png                      ROC vs decision_function (Advanced fit)

The Perceptron is our weakest linear baseline, we use it kind of like a sanity floor
so we can compare the more serious models against it.

All the randomness comes from `RANDOM_STATE` that we import from `src.train_utils`
(so the results stay reproducible).
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

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
import numpy as np


# Identity
MODEL_SLUG    = "perceptron"
MODEL_NAME    = "Perceptron"
DISPLAY_NAME  = "Perceptron"

MODEL_CONFIG = {
    "max_iter":     1000,
    "tol":          1e-3,
    "random_state": RANDOM_STATE,
}


def _build_model() -> Perceptron:
    """
    Here we just build a new Perceptron model with the config we defined above
    :return: the Perceptron model
    """
    return Perceptron(**MODEL_CONFIG)


def main() -> None:
    """
    The main function, this is where we run the whole training and evaluation pipeline
    for the Perceptron model. We loop over the datasets, fit each one, save the
    predictions, and at the end we save all the plots and the metrics json.
    :return: nothing, we just write the results to disk
    """
    print("=" * 72)
    print(f"  TRAIN — {DISPLAY_NAME}   (random_state = {RANDOM_STATE})")
    print("=" * 72)

    # loading the preprocessed datasets and the labels
    datasets, y_train, y_test = load_preprocessed()
    per_ds_results = {}

    # now we run over each dataset (Baseline and Advanced) and fit a fresh model
    for ds_name in DATASETS:
        x_train_matrix, x_test_matrix = datasets[ds_name]
        model = _build_model()
        # fitting the model and getting all the scores
        result = fit_and_score(model, x_train_matrix, y_train, x_test_matrix, y_test)
        per_ds_results[ds_name] = result

        # printing the block for this dataset and saving the predictions to disk
        print_dataset_block(ds_name, x_train_matrix.shape, result)
        save_predictions(MODEL_SLUG, ds_name, result["y_pred"])

    # printing the delta between Baseline and Advanced
    print_delta(per_ds_results)

    # Plots (from the Advanced fit)
    advanced_result = per_ds_results["Advanced"]
    # building the confusion matrix array from the tn/fp/fn/tp dict
    cm_array = np.array([
        [advanced_result["confusion_matrix"]["tn"], advanced_result["confusion_matrix"]["fp"]],
        [advanced_result["confusion_matrix"]["fn"], advanced_result["confusion_matrix"]["tp"]],
    ])
    fig_cm = confusion_matrix_figure(cm_array, title=f"{DISPLAY_NAME} — Confusion Matrix (Advanced)")
    save_figure(MODEL_SLUG, "confusion_matrix.png", fig_cm)
    plt.close(fig_cm)

    # the Perceptron doesn't always have predict_proba, so we only plot the ROC if we got something
    if advanced_result["proba_hit"] is not None:
        fig_roc, _ = roc_curve_figure(
            y_test, advanced_result["proba_hit"],
            title=f"{DISPLAY_NAME} — ROC Curve (Advanced)",
            model_label=DISPLAY_NAME,
        )
        save_figure(MODEL_SLUG, "roc_curve.png", fig_roc)
        plt.close(fig_roc)

    # Canonical JSON
    # now we build the metrics payload and save it as the final json
    payload = build_metrics_payload(
        model_name=MODEL_NAME,
        display_name=DISPLAY_NAME,
        model_config=MODEL_CONFIG,
        n_train=len(y_train),
        n_test=len(y_test),
        random_state=RANDOM_STATE,
        per_dataset_results=per_ds_results,
    )
    metrics_path = save_metrics(MODEL_SLUG, payload)
    print(f"\n  Wrote {metrics_path.relative_to(metrics_path.parent.parent.parent)}")


if __name__ == "__main__":
    main()
