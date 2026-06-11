"""
This script trains and evaluates the MLPClassifier in two reproducible
configurations:

    python train_mlp.py                      # the real model, with validation
                                             # based early stopping
                                             # -> results/mlp/
    python train_mlp.py --no-early-stopping  # the deliberately unregularised
                                             # overfit baseline for the
                                             # ablation in research.md §3.6
                                             # -> results/mlp_overfit/

What we save in `results/mlp/` (early-stopping run):
    - metrics.json (the extras has the early stopping diagnostics)
    - predictions_baseline.npy / predictions_advanced.npy / test_index.npy
    - confusion_matrix.png
    - roc_curve.png
    - loss_curve.png  the training loss with the validation error on top,
                       and a dashed vertical line where we restored the
                       best validation epoch.

The overfit run writes the same artifact set to results/mlp_overfit/ (its
loss_curve.png has no validation overlay because sklearn only tracks
validation_scores_ when early_stopping=True). When both runs exist, the
early-stopping run prints a live before/after ablation read from
results/mlp_overfit/metrics.json — no hardcoded numbers.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier

from src.train_utils import (
    DATASETS,
    PROJECT_ROOT,
    RANDOM_STATE,
    build_metrics_payload,
    confusion_matrix_figure,
    fit_and_score,
    load_metrics,
    load_preprocessed,
    print_dataset_block,
    print_delta,
    roc_curve_figure,
    save_figure,
    save_metrics,
    save_predictions,
    save_test_index,
)


# the shared architecture - both runs use exactly this, only the stopping rule differs
BASE_CONFIG = {
    "hidden_layer_sizes":  (128, 64),
    "max_iter":            300,
    "verbose":             True,
    "random_state":        RANDOM_STATE,
}

EARLY_STOPPING_CONFIG = {
    "early_stopping":      True,
    "validation_fraction": 0.15,
    "n_iter_no_change":    10,
}

OVERFIT_SLUG = "mlp_overfit"


def _resolve_identity(early_stopping: bool):
    """
    Here we pick the slug / names / config for the requested variant, so both
    runs share one code path and only the stopping rule differs.
    :param early_stopping: True for the real model, False for the overfit baseline
    :return: (model_slug, model_name, display_name, model_config)
    """
    if early_stopping:
        return (
            "mlp",
            "MLP (128,64)",
            "MLP (128, 64) + early stopping",
            {**BASE_CONFIG, **EARLY_STOPPING_CONFIG},
        )
    return (
        OVERFIT_SLUG,
        "MLP (128,64) (no early stopping)",
        "MLP (128, 64) — no early stopping (overfit baseline)",
        {**BASE_CONFIG, "early_stopping": False},
    )


def _build_diagnostics(mlp: MLPClassifier, model_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    This function builds the early stopping diagnostics dict from the trained MLP
    it tells us the total epochs, the best validation epoch and stuff like that
    :param mlp: the trained MLP model (must have been fit with early_stopping=True)
    :param model_config: the config dict, for the validation fraction / patience
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
        "validation_fraction":      model_config["validation_fraction"],
        "n_iter_no_change":         model_config["n_iter_no_change"],
        "loss_curve_plot":          "loss_curve.png",
    }


def _loss_curve_figure(mlp: MLPClassifier, display_name: str):
    """
    Here we draw the loss curve plot - the training loss, and when the model
    was fit with early stopping also the validation error with a dashed line
    at the restored best epoch. For the overfit run sklearn doesn't track
    validation scores, so we plot the raw training-loss collapse on its own.
    :param mlp: the trained MLP model
    :param display_name: the name to put in the title
    :return: the matplotlib figure of the plot
    """
    loss = np.asarray(mlp.loss_curve_)
    epochs = np.arange(1, len(loss) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, loss, lw=2.0, color="darkorange",
            label="Training loss (log-loss)")

    val_scores = getattr(mlp, "validation_scores_", None)
    if val_scores is not None:
        val_scores = np.asarray(val_scores)
        # the validation error is 1 minus the validation accuracy
        ax.plot(epochs, 1.0 - val_scores, lw=2.0, color="seagreen",
                label="Validation error (1 − accuracy)")
        best_epoch = int(np.argmax(val_scores)) + 1
        best_val_acc = float(val_scores.max())
        # the dashed vertical line for marking the restored best epoch
        ax.axvline(
            x=best_epoch, color="dimgray", linestyle="--", lw=1.5,
            label=f"Restored epoch = {best_epoch} "
                  f"(best val acc = {best_val_acc:.4f})",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss / Error")
    ax.set_title(f"{display_name} — trained for {len(loss)} epochs")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def _load_overfit_prior() -> Optional[Dict[str, Any]]:
    """
    This helper reads the overfit baseline's saved metrics so the ablation
    uses live numbers from results/mlp_overfit/ instead of hardcoded ones.
    :return: the parsed payload, or None when the overfit run hasn't been done
    """
    try:
        return load_metrics(OVERFIT_SLUG)
    except FileNotFoundError:
        return None


def _print_ablation(now: Dict[str, Any], diagnostics: Dict[str, Any]) -> None:
    """
    Here we print the early-stopping ablation against the saved overfit run,
    if that run exists on disk (run `python train_mlp.py --no-early-stopping`
    once to produce it).
    :param now: the current (early stopped) Advanced result dict
    :param diagnostics: the diagnostics dict of the current run
    """
    prior_payload = _load_overfit_prior()
    print("\n" + "=" * 72)
    print("  EARLY-STOPPING ABLATION")
    print("=" * 72)
    if prior_payload is None:
        print(
            "\n  No overfit baseline found at results/mlp_overfit/metrics.json.\n"
            "  Run `python train_mlp.py --no-early-stopping` once to produce it."
        )
        return

    prior_adv = prior_payload["datasets"]["Advanced"]
    prior_epochs = prior_payload["extras"].get("epochs_total", "?")
    prior_loss = prior_payload["extras"].get("training_loss_at_final", float("nan"))
    d_acc = now["accuracy"] - prior_adv["accuracy"]
    d_f1 = now["f1"] - prior_adv["f1"]
    print(f"\n  Overfit baseline (no early stopping, {prior_epochs} epochs, "
          f"train loss → {prior_loss:.4f}):")
    print(f"     Acc {prior_adv['accuracy']:.4f}   F1 {prior_adv['f1']:.4f}")
    print(f"\n  Current MLP (early stopping, restored at epoch "
          f"{diagnostics['best_validation_epoch']}):")
    print(f"     Acc {now['accuracy']:.4f} ({d_acc:+.4f})   "
          f"F1 {now['f1']:.4f} ({d_f1:+.4f})")


def main() -> None:
    """
    The main function that runs the whole training pipeline for the MLP model.
    here we load the data, train on each dataset, build the diagnostics, save the
    plots, print the ablation against the saved overfit run (when it exists) and
    write the final metrics.json
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="train the unregularised overfit baseline (results/mlp_overfit/)",
    )
    args = parser.parse_args()
    early_stopping = not args.no_early_stopping

    model_slug, model_name, display_name, model_config = _resolve_identity(early_stopping)

    print("=" * 72)
    print(f"  TRAIN — {display_name}   (random_state = {RANDOM_STATE})")
    print("=" * 72)

    # loading the preprocessed datasets and the labels
    datasets, y_train, y_test = load_preprocessed()
    per_ds_results: Dict[str, Dict[str, Any]] = {}
    advanced_mlp = None

    # training on each dataset (Baseline and Advanced) one by one
    for ds_name in DATASETS:
        X_train, X_test = datasets[ds_name]
        model = MLPClassifier(**model_config)
        result = fit_and_score(model, X_train, y_train, X_test, y_test)
        per_ds_results[ds_name] = result

        print_dataset_block(ds_name, X_train.shape, result)
        save_predictions(model_slug, ds_name, result["y_pred"])

        # keep the Advanced model since we need it for the diagnostics and the loss curve
        if ds_name == "Advanced":
            advanced_mlp = result["model"]

    print_delta(per_ds_results)
    save_test_index(model_slug, y_test)

    assert advanced_mlp is not None

    # the plots for the Advanced fit
    adv = per_ds_results["Advanced"]
    cm_array = np.array([
        [adv["confusion_matrix"]["tn"], adv["confusion_matrix"]["fp"]],
        [adv["confusion_matrix"]["fn"], adv["confusion_matrix"]["tp"]],
    ])
    fig_cm = confusion_matrix_figure(cm_array, title=f"{display_name} — Confusion Matrix (Advanced)")
    save_figure(model_slug, "confusion_matrix.png", fig_cm)
    plt.close(fig_cm)

    # the ROC curve for the Advanced fit
    fig_roc, auc = roc_curve_figure(
        y_test, adv["proba_hit"],
        title=f"{display_name} — ROC Curve (Advanced)",
        model_label=display_name,
    )
    save_figure(model_slug, "roc_curve.png", fig_roc)
    plt.close(fig_roc)

    # the loss curve plot (with the validation overlay only on the ES run)
    fig_loss = _loss_curve_figure(advanced_mlp, display_name)
    save_figure(model_slug, "loss_curve.png", fig_loss)
    plt.close(fig_loss)

    extras: Dict[str, Any] = {"roc_auc_advanced": auc}

    if early_stopping:
        # building + printing the early stopping diagnostics for the Advanced model
        diagnostics = _build_diagnostics(advanced_mlp, model_config)
        print("\n" + "=" * 72)
        print("  EARLY-STOPPING DIAGNOSTICS (Advanced fit)")
        print("=" * 72)
        for k, v in diagnostics.items():
            if isinstance(v, float):
                print(f"  {k:<28}: {v:.4f}")
            else:
                print(f"  {k:<28}: {v}")
        extras["early_stopping_diagnostics"] = diagnostics
        _print_ablation(adv, diagnostics)
    else:
        # the overfit baseline - record the raw loss collapse so the ES run's
        # ablation (and research.md §3.6) can read live numbers from disk
        loss = list(advanced_mlp.loss_curve_)
        extras["epochs_total"] = len(loss)
        extras["training_loss_at_final"] = float(loss[-1])
        extras["loss_curve_plot"] = "loss_curve.png"
        print(f"\n  Overfit baseline: {len(loss)} epochs, "
              f"final training loss {loss[-1]:.4f}")

    # finally we build the metrics payload (canonical json) and save it
    payload = build_metrics_payload(
        model_name=model_name,
        display_name=display_name,
        model_config={
            **model_config,
            "hidden_layer_sizes": list(model_config["hidden_layer_sizes"]),
        },
        n_train=len(y_train),
        n_test=len(y_test),
        random_state=RANDOM_STATE,
        per_dataset_results=per_ds_results,
        extras=extras,
    )
    metrics_path = save_metrics(model_slug, payload)
    print(f"\n  Wrote {metrics_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
