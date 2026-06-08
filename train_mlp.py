"""
This script trains and evaluates the MLPClassifier with validation based early stopping.

What we save in `results/mlp/`:
    - metrics.json (the extras has the early stopping diagnostics)
    - predictions_baseline.npy
    - predictions_advanced.npy
    - confusion_matrix.png
    - roc_curve.png
    - loss_curve.png  the training loss with the validation error on top,
                       and a dashed vertical line where we restored the
                       best validation epoch.
"""

from __future__ import annotations

from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier

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


MODEL_SLUG    = "mlp"
MODEL_NAME    = "MLP (128,64)"
DISPLAY_NAME  = "MLP (128, 64) + early stopping"

MODEL_CONFIG = {
    "hidden_layer_sizes":  (128, 64),
    "max_iter":            300,
    "early_stopping":      True,
    "validation_fraction": 0.15,
    "n_iter_no_change":    10,
    "verbose":             True,
    "random_state":        RANDOM_STATE,
}

# the previous run without regularization and without early stopping, we keep it for the before/after comparison
MLP_OVERFITTED_PRIOR = {
    "epochs":       64,
    "final_loss":   0.0087,
    "advanced_acc": 0.5816,
    "advanced_f1":  0.6107,
}


def _build_model() -> MLPClassifier:
    """
    Here we just create the MLP classifier with our config
    :return: the MLP model object
    """
    return MLPClassifier(**MODEL_CONFIG)


def _build_diagnostics(mlp: MLPClassifier) -> Dict[str, Any]:
    """
    This function builds the early stopping diagnostics dict from the trained MLP
    it tells us the total epochs, the best validation epoch and stuff like that
    :param mlp: the trained MLP model
    :return: a dictionary of all the diagnostic values we want to save
    """
    loss = list(mlp.loss_curve_)
    val_scores = list(mlp.validation_scores_)
    # finding the index of the best validation score (0 based)
    best_epoch_0idx = int(np.argmax(val_scores))
    return {
        "epochs_total":             len(loss),
        "best_validation_epoch":    best_epoch_0idx + 1,
        "best_validation_accuracy": float(val_scores[best_epoch_0idx]),
        "training_loss_at_best":    float(loss[best_epoch_0idx]),
        "training_loss_at_final":   float(loss[-1]),
        "validation_fraction":      MODEL_CONFIG["validation_fraction"],
        "n_iter_no_change":         MODEL_CONFIG["n_iter_no_change"],
        "loss_curve_plot":          "loss_curve.png",
    }


def _loss_curve_figure(mlp: MLPClassifier):
    """
    Here we draw the loss curve plot, the training loss together with the validation error.
    we also add a dashed line where the best validation epoch is (the one we restored)
    :param mlp: the trained MLP model
    :return: the matplotlib figure of the plot
    """
    loss = np.asarray(mlp.loss_curve_)
    val_scores = np.asarray(mlp.validation_scores_)
    # the validation error is 1 minus the validation accuracy
    val_err = 1.0 - val_scores
    epochs = np.arange(1, len(loss) + 1)
    # finding the best epoch (1 based for the plot) and the best validation accuracy
    best_epoch = int(np.argmax(val_scores)) + 1
    best_val_acc = float(val_scores.max())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, loss, lw=2.0, color="darkorange",
            label="Training loss (log-loss)")
    ax.plot(epochs, val_err, lw=2.0, color="seagreen",
            label="Validation error (1 − accuracy)")
    # the dashed vertical line for marking the restored best epoch
    ax.axvline(
        x=best_epoch, color="dimgray", linestyle="--", lw=1.5,
        label=f"Restored epoch = {best_epoch} "
              f"(best val acc = {best_val_acc:.4f})",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss / Error")
    ax.set_title(
        f"MLP (128, 64) — Training Loss vs Validation Error "
        f"(Early Stopping; trained for {len(loss)} epochs)"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def main() -> None:
    """
    The main function that runs the whole training pipeline for the MLP model.
    here we load the data, train on each dataset, build the diagnostics, save the plots,
    print the ablation against the previous overfit run and write the final metrics.json
    """
    print("=" * 72)
    print(f"  TRAIN — {DISPLAY_NAME}   (random_state = {RANDOM_STATE})")
    print("=" * 72)

    # loading the preprocessed datasets and the labels
    datasets, y_train, y_test = load_preprocessed()
    per_ds_results: Dict[str, Dict[str, Any]] = {}
    advanced_mlp = None

    # training on each dataset (Baseline and Advanced) one by one
    for ds_name in DATASETS:
        X_train, X_test = datasets[ds_name]
        model = _build_model()
        result = fit_and_score(model, X_train, y_train, X_test, y_test)
        per_ds_results[ds_name] = result

        print_dataset_block(ds_name, X_train.shape, result)
        save_predictions(MODEL_SLUG, ds_name, result["y_pred"])

        # keep the Advanced model since we need it for the diagnostics and the loss curve
        if ds_name == "Advanced":
            advanced_mlp = result["model"]

    print_delta(per_ds_results)

    assert advanced_mlp is not None
    # building the early stopping diagnostics for the Advanced model
    diagnostics = _build_diagnostics(advanced_mlp)

    # the plots for the Advanced fit
    adv = per_ds_results["Advanced"]
    cm_array = np.array([
        [adv["confusion_matrix"]["tn"], adv["confusion_matrix"]["fp"]],
        [adv["confusion_matrix"]["fn"], adv["confusion_matrix"]["tp"]],
    ])
    fig_cm = confusion_matrix_figure(cm_array, title=f"{DISPLAY_NAME} — Confusion Matrix (Advanced)")
    save_figure(MODEL_SLUG, "confusion_matrix.png", fig_cm)
    plt.close(fig_cm)

    # the ROC curve for the Advanced fit
    fig_roc, auc = roc_curve_figure(
        y_test, adv["proba_hit"],
        title=f"{DISPLAY_NAME} — ROC Curve (Advanced)",
        model_label=DISPLAY_NAME,
    )
    save_figure(MODEL_SLUG, "roc_curve.png", fig_roc)
    plt.close(fig_roc)

    # the loss curve plot
    fig_loss = _loss_curve_figure(advanced_mlp)
    save_figure(MODEL_SLUG, "loss_curve.png", fig_loss)
    plt.close(fig_loss)

    # printing the diagnostics nicely
    print("\n" + "=" * 72)
    print("  EARLY-STOPPING DIAGNOSTICS (Advanced fit)")
    print("=" * 72)
    for k, v in diagnostics.items():
        if isinstance(v, float):
            print(f"  {k:<28}: {v:.4f}")
        else:
            print(f"  {k:<28}: {v}")

    # the ablation against the previous overfit MLP run
    prior = MLP_OVERFITTED_PRIOR
    now = per_ds_results["Advanced"]
    # the delta of accuracy and f1 between current run and the previous one
    d_acc = now["accuracy"] - prior["advanced_acc"]
    d_f1 = now["f1"] - prior["advanced_f1"]
    print("\n" + "=" * 72)
    print("  EARLY-STOPPING ABLATION")
    print("=" * 72)
    print(f"\n  Previous MLP (no early stopping, {prior['epochs']} epochs, "
          f"train loss → {prior['final_loss']:.4f}):")
    print(f"     Acc {prior['advanced_acc']:.4f}   F1 {prior['advanced_f1']:.4f}")
    print(f"\n  Current MLP (early stopping, restored at epoch {diagnostics['best_validation_epoch']}):")
    print(f"     Acc {now['accuracy']:.4f} ({d_acc:+.4f})   "
          f"F1 {now['f1']:.4f} ({d_f1:+.4f})")

    # finally we build the metrics payload (canonical json) and save it
    payload = build_metrics_payload(
        model_name=MODEL_NAME,
        display_name=DISPLAY_NAME,
        model_config={
            **MODEL_CONFIG,
            "hidden_layer_sizes": list(MODEL_CONFIG["hidden_layer_sizes"]),
        },
        n_train=len(y_train),
        n_test=len(y_test),
        random_state=RANDOM_STATE,
        per_dataset_results=per_ds_results,
        extras={
            "early_stopping_diagnostics": diagnostics,
            "overfitted_prior_run":       MLP_OVERFITTED_PRIOR,
            "roc_auc_advanced":           auc,
        },
    )
    metrics_path = save_metrics(MODEL_SLUG, payload)
    print(f"\n  Wrote {metrics_path.relative_to(metrics_path.parent.parent.parent)}")


if __name__ == "__main__":
    main()
