"""
Train and evaluate the AdaBoost classifier on Baseline + Advanced.

Outputs (written to `results/adaboost/`):
    - metrics.json
    - predictions_baseline.npy
    - predictions_advanced.npy
    - confusion_matrix.png
    - roc_curve.png

AdaBoost (boosted decision stumps) is the mildly-non-linear member of the
linear lineup. It shows the strongest class asymmetry of any model in
the study (~0.575 FP-rate vs 0.257 FN-rate at default threshold) — a
useful counter-example to the symmetric-errors story Logistic Regression
tells in Phase 4's threshold sweep.
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
    return AdaBoostClassifier(**MODEL_CONFIG)


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
        extras={"roc_auc_advanced": auc},
    )
    metrics_path = save_metrics(MODEL_SLUG, payload)
    print(f"\n  Wrote {metrics_path.relative_to(metrics_path.parent.parent.parent)}")


if __name__ == "__main__":
    main()
