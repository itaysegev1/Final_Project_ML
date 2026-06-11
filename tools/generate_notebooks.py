"""
This script builds all 8 portfolio notebooks using nbformat.

We keep all the notebook generation in one place so the structure stays the same
across the models, and any change we want to make is in one file. Each model
notebook has the same cells order: markdown intro -> setup -> load data ->
configure -> train -> plot -> save artifacts -> summary.

To run it just do:

    python tools/generate_notebooks.py

It writes all 8 .ipynb files into notebooks/. Each notebook imports from src/
(via sys.path) and saves its artifacts into results/<slug>/.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOKS_DIR.mkdir(exist_ok=True)


# Cell-building helpers
def md(source: str):
    """
    This function wraps a string into a markdown cell, dedenting it first
    so we can keep it nicely indented in the python source.
    :param source: the markdown text we want in the cell
    :return: the new markdown cell
    """
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    """
    Same idea as md but for a code cell.
    :param source: the code we want in the cell
    :return: the new code cell
    """
    return nbf.v4.new_code_cell(dedent(source).strip())


def write_nb(filename: str, cells: List[Any]) -> Path:
    """
    Here we build the notebook object from the cells and write it to disk.
    :param filename: the name of the file we want to write to
    :param cells: the list of cells we want to put inside
    :return: the path of the notebook we just wrote
    """
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    path = NOTEBOOKS_DIR / filename
    with open(path, "w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    return path


# Standard cell blocks we reuse across the model notebooks
def setup_cells() -> List[Any]:
    """
    This function returns the standard setup cells we put at the top of every model notebook.
    :return: a list of the setup cells (markdown + code)
    """
    return [
        md(
            """
            ## 1 - Setup

            Here we set matplotlib to inline mode, add the project root
            to sys.path so we can import from src/, and bring in the
            shared helpers together with the project-wide RANDOM_STATE = 42
            (bound once in src/_constants.py, re-exported via src).
            """
        ),
        code(
            """
            %matplotlib inline
            import sys
            from pathlib import Path

            # notebooks/ sits one level under the project root, so we add
            # the project root to make `from src import ...` work.
            PROJECT_ROOT = Path.cwd().parent
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))

            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt

            from src import (
                RANDOM_STATE,
                DATASETS,
                load_preprocessed,
                fit_and_score,
                build_metrics_payload,
                save_metrics,
                save_predictions,
                save_test_index,
                save_figure,
                confusion_matrix_figure,
                roc_curve_figure,
                print_dataset_block,
                print_delta,
                model_results_dir,
            )

            print(f"RANDOM_STATE = {RANDOM_STATE} (bound once in src/_constants.py)")
            """
        ),
    ]


def data_loading_cells() -> List[Any]:
    """
    This function returns the cells that load the preprocessed matrices.
    :return: the list of cells for data loading
    """
    return [
        md(
            """
            ## 2 - Load the preprocessed feature matrices

            load_preprocessed() gives us back the two matrices that we
            made in Phase 1: the Baseline (just nutrition + tags) and
            the Advanced (the Baseline plus 9 engineered culinary
            features). They share the same train/test split so the A/B
            comparison is fair.
            """
        ),
        code(
            """
            datasets, y_train, y_test = load_preprocessed()

            print(f"Baseline X_train: {datasets['Baseline'][0].shape}")
            print(f"Advanced X_train: {datasets['Advanced'][0].shape}")
            print(f"y_test class balance: Miss={int((y_test == 0).sum())} / "
                  f"Hit={int((y_test == 1).sum())} "
                  f"(majority-class rate: {max(y_test.mean(), 1 - y_test.mean()):.4f})")
            """
        ),
    ]


def training_loop_cells(model_build_code: str) -> List[Any]:
    """
    This function returns the standard training loop cells, where we train the
    model on Baseline and then on Advanced.
    :param model_build_code: the code that defines the _build_model() factory
    :return: the list of cells for the training loop
    """
    return [
        md(
            """
            ## 4 - Train on both matrices

            For each matrix we build a fresh model from scratch (the
            factory below gets called once per dataset). Nothing leaks
            from Baseline to Advanced so the delta we see is really
            only because of the 9 engineered features.
            """
        ),
        code(
            f"""
            {model_build_code.strip()}

            per_ds_results = {{}}
            for ds_name in DATASETS:
                X_train, X_test = datasets[ds_name]
                model = _build_model()
                result = fit_and_score(model, X_train, y_train, X_test, y_test)
                per_ds_results[ds_name] = result

                print_dataset_block(ds_name, X_train.shape, result)
                save_predictions(MODEL_SLUG, ds_name, result["y_pred"])

            print_delta(per_ds_results)
            save_test_index(MODEL_SLUG, y_test)
            """
        ),
    ]


def confusion_matrix_cells() -> List[Any]:
    """
    This function returns the cells for plotting the confusion matrix on the Advanced fit.
    :return: the list of cells
    """
    return [
        md(
            """
            ## 5 - Confusion matrix (Advanced fit)

            An annotated heatmap of the confusion matrix from the
            Advanced fit. We render it inline and also save it to
            results/<slug>/confusion_matrix.png.
            """
        ),
        code(
            """
            adv = per_ds_results["Advanced"]
            cm_array = np.array([
                [adv["confusion_matrix"]["tn"], adv["confusion_matrix"]["fp"]],
                [adv["confusion_matrix"]["fn"], adv["confusion_matrix"]["tp"]],
            ])
            fig_cm = confusion_matrix_figure(
                cm_array,
                title=f"{DISPLAY_NAME} - Confusion Matrix (Advanced)",
            )
            save_figure(MODEL_SLUG, "confusion_matrix.png", fig_cm)
            plt.show()
            """
        ),
    ]


def roc_curve_cells() -> List[Any]:
    """
    This function returns the cells for the ROC curve plot on the Advanced fit.
    :return: the list of cells
    """
    return [
        md(
            """
            ## 6 - ROC curve + AUC (Advanced fit)

            We use predict_proba (or decision_function if the model
            doesn't have probabilities). The AUC tells us how well the
            model ranks the positives over the negatives, regardless of
            the threshold we pick. Phase 4's threshold sweep is built
            on top of this.
            """
        ),
        code(
            """
            fig_roc, auc = roc_curve_figure(
                y_test, adv["proba_hit"],
                title=f"{DISPLAY_NAME} - ROC Curve (Advanced)",
                model_label=DISPLAY_NAME,
            )
            save_figure(MODEL_SLUG, "roc_curve.png", fig_roc)
            print(f"Test ROC AUC (Advanced): {auc:.4f}")
            plt.show()
            """
        ),
    ]


def save_metrics_cells(extras_code: str = "{}") -> List[Any]:
    """
    This function returns the cells that build and write the metrics JSON payload.
    :param extras_code: a string of python code that defines the extras dict
    :return: the list of cells
    """
    return [
        md(
            """
            ## 8 - Persist the canonical metrics JSON

            One JSON per model, written into
            results/<slug>/metrics.json. The schema is defined in
            src.train_utils.build_metrics_payload, and the master
            comparison notebook reads from those files.
            """
        ),
        code(
            f"""
            extras = {extras_code}

            payload = build_metrics_payload(
                model_name=MODEL_NAME,
                display_name=DISPLAY_NAME,
                model_config=MODEL_CONFIG,
                n_train=len(y_train),
                n_test=len(y_test),
                random_state=RANDOM_STATE,
                per_dataset_results=per_ds_results,
                extras=extras,
            )
            metrics_path = save_metrics(MODEL_SLUG, payload)
            print(f"Wrote {{metrics_path.relative_to(metrics_path.parent.parent.parent)}}")
            """
        ),
    ]


def summary_cells(title: str, summary_md: str) -> List[Any]:
    """
    This function returns the closing summary cell for a model notebook.
    :param title: the model name to put on the summary
    :param summary_md: the bullet list with the highlights of the run
    :return: the list of cells
    """
    return [
        md(
            f"""
            ## 9 - Summary

            **Model:** {title}

            {summary_md.strip()}

            To see this model side by side with the other six, run the
            master comparison notebook (`08_Master_Comparison.ipynb`).
            """
        ),
    ]


# 01 - Logistic Regression
def build_logistic_regression_notebook() -> Path:
    """
    This function builds the Logistic Regression notebook.
    :return: the path of the notebook we just wrote
    """
    cells = [
        md(
            """
            # 01 - Logistic Regression

            Logistic Regression (L2 penalty, liblinear solver) is the
            interpretable champion of the project. We get signed
            coefficients we can read directly in IQR units, calibrated
            predict_proba for the Phase 4 threshold sweep, and the
            solver that finally converges (this is also the one that
            caught the lbfgs convergence bug from research.md section 2.4).

            All randomness comes from RANDOM_STATE = 42, bound once in
            src/_constants.py.
            """
        ),
    ]
    cells += setup_cells()
    cells += data_loading_cells()
    cells += [
        md(
            """
            ## 3 - Configure the model

            We pick solver='liblinear' (coordinate descent) because on
            this 678-687 dim sparse-binary input it converges nicely
            while lbfgs didn't. C=1.0 is the default L2 strength,
            and max_iter=5000 is plenty of head-room (liblinear
            actually converges way before that on this dataset).
            """
        ),
        code(
            """
            from sklearn.linear_model import LogisticRegression

            MODEL_SLUG    = "logistic_regression"
            MODEL_NAME    = "LogisticRegression"
            DISPLAY_NAME  = "Logistic Regression"

            MODEL_CONFIG = {
                "solver":       "liblinear",
                "C":            1.0,
                "max_iter":     5000,
                "random_state": RANDOM_STATE,
            }
            MODEL_CONFIG
            """
        ),
    ]
    cells += training_loop_cells(
        """def _build_model():
                return LogisticRegression(**MODEL_CONFIG)"""
    )
    cells += confusion_matrix_cells()
    cells += roc_curve_cells()
    cells += [
        md(
            """
            ## 7 - Top signed coefficients

            LR is the model where we can just read the internals
            directly. The top 10 positive and top 10 negative
            coefficients give us the "Recipe for Success / Disaster"
            story from research.md section 3.4.
            """
        ),
        code(
            """
            adv_model = per_ds_results["Advanced"]["model"]
            X_train_adv = datasets["Advanced"][0]
            coefs = pd.Series(adv_model.coef_[0], index=X_train_adv.columns)

            display(pd.DataFrame({
                "Top 10 Hit indicators": coefs.sort_values(ascending=False).head(10).round(4).to_dict(),
            }))
            display(pd.DataFrame({
                "Top 10 Miss indicators": coefs.sort_values(ascending=True).head(10).round(4).to_dict(),
            }))
            """
        ),
    ]
    cells += save_metrics_cells('{"roc_auc_advanced": auc}')
    cells += summary_cells(
        "Logistic Regression",
        """
        - **Test Accuracy / F1 / AUC:** about 0.6011 / 0.6405 / 0.6491
        - **Interpretable champion** - the top coefficient is what
          drives the "Recipe for Success" story in research.md section 3.4.
        - **Calibrated probabilities** - this is what feeds the Phase 4
          threshold selection (`advanced_tuning.py`, selection on a
          validation split).
        """
    )
    return write_nb("01_Logistic_Regression.ipynb", cells)


# 02 - Random Forest
def build_random_forest_notebook() -> Path:
    """
    This function builds the Random Forest notebook.
    :return: the path of the notebook we just wrote
    """
    cells = [
        md(
            """
            # 02 - Random Forest

            The predictive champion of the project. Random Forest
            breaks past the "linear ceiling" of Logistic Regression
            (+2.28 pp Acc, +3.49 pp F1 on Advanced) because it can pick
            up non-linear interactions between features. What is really
            interesting is that its feature-importance ranking inverts
            the LR coefficient picture - see research.md section 3.5.
            """
        ),
    ]
    cells += setup_cells()
    cells += data_loading_cells()
    cells += [
        md(
            """
            ## 3 - Configure the model

            200 trees gives us a small improvement over the default
            100 for basically no cost, and n_jobs=-1 uses all the
            cores we have. random_state=RANDOM_STATE pushes the seed
            42 through to the bagging RNG.
            """
        ),
        code(
            """
            from sklearn.ensemble import RandomForestClassifier

            MODEL_SLUG    = "random_forest"
            MODEL_NAME    = "RandomForest"
            DISPLAY_NAME  = "Random Forest"
            TOP_K         = 20

            MODEL_CONFIG = {
                "n_estimators": 200,
                "n_jobs":       -1,
                "random_state": RANDOM_STATE,
            }
            MODEL_CONFIG
            """
        ),
    ]
    cells += training_loop_cells(
        """def _build_model():
                return RandomForestClassifier(**MODEL_CONFIG)"""
    )
    cells += confusion_matrix_cells()
    cells += roc_curve_cells()
    cells += [
        md(
            """
            ## 7 - Top-20 feature importances (the inversion story)

            The top 7 features for RF are all continuous numerics -
            three of them are engineered (`avg_words_per_step`,
            `num_ingredients`, `num_steps`) and four come from the
            original nutrition columns - and all six engineered has_*
            keyword features place inside the top 20, above ~670 of
            the editorial tags. This is the opposite of what we saw
            with LR, where the binary tags were on top. The reason: L2
            splits the credit across collinear features, but trees
            just pick the single best splitter at each node and don't
            care about collinearity. See research.md section 3.5.
            """
        ),
        code(
            """
            adv_model = per_ds_results["Advanced"]["model"]
            X_train_adv = datasets["Advanced"][0]

            top = (
                pd.Series(adv_model.feature_importances_, index=X_train_adv.columns)
                  .sort_values(ascending=False)
                  .head(TOP_K)
            )
            display(top.round(4).to_frame("importance"))

            fig_imp, ax = plt.subplots(figsize=(9, 8))
            top.iloc[::-1].plot(kind="barh", ax=ax, color="steelblue")
            ax.set_xlabel("Feature importance (impurity reduction, Gini)")
            ax.set_title(f"Random Forest - Top {TOP_K} Feature Importances (Advanced)")
            ax.grid(axis="x", alpha=0.3)
            fig_imp.tight_layout()
            save_figure(MODEL_SLUG, "feature_importance.png", fig_imp)
            plt.show()
            """
        ),
    ]
    cells += save_metrics_cells(
        '''{
                "top_k_feature_importances": [
                    {"feature": str(name), "importance": float(val)}
                    for name, val in top.items()
                ],
                "feature_importance_plot": "feature_importance.png",
                "roc_auc_advanced": auc,
            }'''
    )
    cells += summary_cells(
        "Random Forest (n=200)",
        """
        - **Test Accuracy / F1:** about 0.6239 / 0.6753 - the top
          scores in the whole lineup.
        - **Breaks the linear ceiling** by +2.28 pp Acc and +3.49 pp F1
          over LR.
        - **Inverts the feature-importance ranking** - the continuous
          numerics dominate (and all six has_* features crack the top
          20) where they sat in the bottom half of the LR ranking.
        """
    )
    return write_nb("02_Random_Forest.ipynb", cells)


# 03 - MLP Neural Network
def build_mlp_notebook() -> Path:
    """
    This function builds the MLP Neural Network notebook (with early stopping).
    :return: the path of the notebook we just wrote
    """
    cells = [
        md(
            """
            # 03 - MLP Neural Network (with early stopping)

            Two-hidden-layer (128, 64) MLP. The non-regularised
            version of this exact model (reproducible with
            `python train_mlp.py --no-early-stopping`) pushed training
            loss down to about 0.024 in 53 epochs while test accuracy
            dropped FAR BELOW Logistic Regression - a textbook
            overfitting signature. In this notebook we train the
            regularised version with early stopping based on
            validation (`early_stopping=True`,
            `validation_fraction=0.15`, `n_iter_no_change=10`), and
            test performance recovers PAST the linear baseline
            (see research.md section 3.6).
            """
        ),
    ]
    cells += setup_cells()
    cells += data_loading_cells()
    cells += [
        md(
            """
            ## 3 - Configure the model

            Validation-based early stopping holds out 15% of X_train
            internally (NOT our test split - so it's leakage-free),
            tracks the validation accuracy each epoch, and restores
            the weights to the best-validation epoch when fit()
            finishes. verbose=True gives us the per-epoch loss /
            validation log.
            """
        ),
        code(
            """
            from sklearn.neural_network import MLPClassifier

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
            MODEL_CONFIG
            """
        ),
    ]
    cells += training_loop_cells(
        """def _build_model():
                return MLPClassifier(**MODEL_CONFIG)"""
    )
    cells += confusion_matrix_cells()
    cells += roc_curve_cells()
    cells += [
        md(
            """
            ## 7 - Training-loss / validation-error overlay

            The training loss (orange) goes smoothly down towards zero,
            and the validation error (green) flattens out and starts
            climbing a bit - the classic overfitting picture. The
            dashed vertical line is the epoch where sklearn restored
            the weights back to (the one with the best validation
            score).
            """
        ),
        code(
            """
            adv_mlp = per_ds_results["Advanced"]["model"]
            loss = np.asarray(adv_mlp.loss_curve_)
            val_scores = np.asarray(adv_mlp.validation_scores_)
            val_err = 1.0 - val_scores
            epochs = np.arange(1, len(loss) + 1)
            best_epoch = int(np.argmax(val_scores)) + 1
            best_val_acc = float(val_scores.max())

            fig_loss, ax = plt.subplots(figsize=(10, 6))
            ax.plot(epochs, loss, lw=2.0, color="darkorange",
                    label="Training loss (log-loss)")
            ax.plot(epochs, val_err, lw=2.0, color="seagreen",
                    label="Validation error (1 - accuracy)")
            ax.axvline(x=best_epoch, color="dimgray", linestyle="--", lw=1.5,
                       label=f"Restored epoch = {best_epoch} (best val acc = {best_val_acc:.4f})")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss / Error")
            ax.set_title(f"MLP - Training Loss vs Validation Error ({len(loss)} epochs)")
            ax.grid(alpha=0.3)
            ax.legend(loc="best")
            fig_loss.tight_layout()
            save_figure(MODEL_SLUG, "loss_curve.png", fig_loss)
            plt.show()

            diagnostics = {
                "epochs_total":             len(loss),
                "best_validation_epoch":    best_epoch,
                "best_validation_accuracy": best_val_acc,
                "training_loss_at_best":    float(loss[best_epoch - 1]),
                "training_loss_at_final":   float(loss[-1]),
                "validation_fraction":      MODEL_CONFIG["validation_fraction"],
                "n_iter_no_change":         MODEL_CONFIG["n_iter_no_change"],
                "loss_curve_plot":          "loss_curve.png",
            }
            display(pd.Series(diagnostics).to_frame("value"))
            """
        ),
        md(
            """
            ## 7.5 - Early-stopping ablation (vs the saved overfit run)

            The overfit baseline lives on disk in
            results/mlp_overfit/metrics.json, written by
            `python train_mlp.py --no-early-stopping` - no hardcoded
            numbers. Without early stopping the same architecture runs
            its training loss to about 0.024 over 53 epochs and lands
            FAR below LR on the test set; with early stopping the
            network restores to the best-validation epoch and clears
            the linear baseline. See research.md section 3.6.
            """
        ),
        code(
            """
            from src import load_metrics

            now = per_ds_results["Advanced"]
            try:
                prior = load_metrics("mlp_overfit")
            except FileNotFoundError:
                prior = None
                print("No overfit baseline found - run "
                      "`python train_mlp.py --no-early-stopping` once to produce it.")

            if prior is not None:
                prior_adv = prior["datasets"]["Advanced"]
                d_acc = now["accuracy"] - prior_adv["accuracy"]
                d_f1 = now["f1"] - prior_adv["f1"]

                display(pd.DataFrame({
                    "Overfit baseline (no early stopping)": [
                        f"{prior['extras'].get('epochs_total', '?')} epochs",
                        f"loss -> {prior['extras'].get('training_loss_at_final', float('nan')):.4f}",
                        f"{prior_adv['accuracy']:.4f}",
                        f"{prior_adv['f1']:.4f}",
                    ],
                    "Current (early stopping)": [
                        f"{diagnostics['epochs_total']} epochs (restored at {best_epoch})",
                        f"loss -> {diagnostics['training_loss_at_final']:.4f}",
                        f"{now['accuracy']:.4f} ({d_acc:+.4f})",
                        f"{now['f1']:.4f} ({d_f1:+.4f})",
                    ],
                }, index=["Training", "Final training loss", "Test Acc", "Test F1"]))
            """
        ),
    ]
    cells += save_metrics_cells(
        '''{
                "early_stopping_diagnostics": diagnostics,
                "roc_auc_advanced":           auc,
            }'''
    )
    cells += summary_cells(
        "MLP (128, 64) + early stopping",
        """
        - **Test Accuracy / F1:** about 0.6126 / 0.6632 - clears the
          linear baseline on both metrics once we turn on early
          stopping (+1.15 pp Acc, +2.27 pp F1 vs LR).
        - **Without early stopping** the exact same architecture
          overfit (Acc 0.5728, far below LR) - see the ablation table
          in section 7.5.
        - **The lesson:** more capacity is actually a liability if
          we don't add the matching regularisation. research.md
          section 4 Lesson #6.
        """
    )
    return write_nb("03_MLP_Neural_Network.ipynb", cells)


# 04 - Perceptron
def build_perceptron_notebook() -> Path:
    """
    This function builds the Perceptron notebook.
    :return: the path of the notebook we just wrote
    """
    cells = [
        md(
            """
            # 04 - Perceptron

            The weakest linear baseline in the lineup. We keep it in
            the project as a sanity-check floor that the more
            sophisticated models can be measured against. It gets
            close to its Bayes-error rate and then just plateaus.
            """
        ),
    ]
    cells += setup_cells()
    cells += data_loading_cells()
    cells += [
        md(
            """
            ## 3 - Configure the model

            Default Perceptron with random_state=RANDOM_STATE for the
            SGD shuffle, max_iter=1000 and tol=1e-3 - these match the
            sklearn defaults.
            """
        ),
        code(
            """
            from sklearn.linear_model import Perceptron

            MODEL_SLUG    = "perceptron"
            MODEL_NAME    = "Perceptron"
            DISPLAY_NAME  = "Perceptron"

            MODEL_CONFIG = {
                "max_iter":     1000,
                "tol":          1e-3,
                "random_state": RANDOM_STATE,
            }
            MODEL_CONFIG
            """
        ),
    ]
    cells += training_loop_cells(
        """def _build_model():
                return Perceptron(**MODEL_CONFIG)"""
    )
    cells += confusion_matrix_cells()
    cells += roc_curve_cells()
    cells += save_metrics_cells('{"roc_auc_advanced": auc}')
    cells += summary_cells(
        "Perceptron",
        """
        - **Test Accuracy / F1:** about 0.5435 / 0.5667 - the weakest
          model in the lineup, and the only one that loses accuracy
          from the engineered features.
        - **Errors are pretty symmetric** (FP is about FN), the
          opposite of AdaBoost which is very asymmetric.
        """
    )
    return write_nb("04_Perceptron.ipynb", cells)


# 05 - AdaBoost
def build_adaboost_notebook() -> Path:
    """
    This function builds the AdaBoost notebook.
    :return: the path of the notebook we just wrote
    """
    cells = [
        md(
            """
            # 05 - AdaBoost

            Boosted decision stumps - the mildly non-linear member of
            the linear lineup. It has the strongest class asymmetry
            of all the models in the study (about 0.575 FP-rate vs
            0.257 FN-rate at the default threshold), so it makes a
            nice counter-example to the symmetric-errors story that
            Logistic Regression tells us in the Phase 4 threshold
            sweep.
            """
        ),
    ]
    cells += setup_cells()
    cells += data_loading_cells()
    cells += [
        md(
            """
            ## 3 - Configure the model

            100 boosting iterations, and random_state=RANDOM_STATE
            seeds the bootstrap sampling.
            """
        ),
        code(
            """
            from sklearn.ensemble import AdaBoostClassifier

            MODEL_SLUG    = "adaboost"
            MODEL_NAME    = "AdaBoost"
            DISPLAY_NAME  = "AdaBoost (n=100)"

            MODEL_CONFIG = {
                "n_estimators": 100,
                "random_state": RANDOM_STATE,
            }
            MODEL_CONFIG
            """
        ),
    ]
    cells += training_loop_cells(
        """def _build_model():
                return AdaBoostClassifier(**MODEL_CONFIG)"""
    )
    cells += confusion_matrix_cells()
    cells += roc_curve_cells()
    cells += save_metrics_cells('{"roc_auc_advanced": auc}')
    cells += summary_cells(
        "AdaBoost (n=100)",
        """
        - **Test Accuracy / F1:** about 0.5967 / 0.6518.
        - **Strongest class asymmetry** of the lineup - FP-rate 0.575
          vs FN-rate 0.257. The engineered features actually re-shape
          the errors instead of reducing them (Delta F1 = -0.0103).
        """
    )
    return write_nb("05_AdaBoost.ipynb", cells)


# 06 - PCA + KNN
def build_pca_knn_notebook() -> Path:
    """
    This function builds the PCA + KNN notebook (the cautionary tale).
    :return: the path of the notebook we just wrote
    """
    cells = [
        md(
            """
            # 06 - PCA(0.90) + KNN  (the cautionary tale)

            This notebook trains the original PCA + KNN pipeline that
            scored only about 0.51 accuracy. The reason is going to
            show up in the cell that prints pca.n_components_: PCA
            keeps exactly 1 component for 90% variance, because the
            nutrition columns with their big outliers hijack the
            projection. After that KNN is doing similarity search on
            a 1-D projection of the data. See research.md section 4
            Lesson #5.

            The fix for this is in 07_PCA_KNN_Improved.ipynb.
            """
        ),
    ]
    cells += setup_cells()
    cells += data_loading_cells()
    cells += [
        md(
            """
            ## 3 - Configure the pipeline

            PCA(n_components=0.90) keeps the smallest set of principal
            components that together cover 90% of the input variance.
            KNN(n_neighbors=5) then works in that reduced space. Both
            stages share the same random_state. KNN by itself is
            deterministic (no random_state for it).
            """
        ),
        code(
            """
            from sklearn.decomposition import PCA
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.pipeline import Pipeline

            MODEL_SLUG    = "pca_knn"
            MODEL_NAME    = "PCA(0.90) + KNN"
            DISPLAY_NAME  = "PCA(0.90) + KNN"

            PCA_VARIANCE = 0.90
            KNN_NEIGHBORS = 5
            MODEL_CONFIG = {
                "pca": {"n_components": PCA_VARIANCE, "random_state": RANDOM_STATE},
                "knn": {"n_neighbors": KNN_NEIGHBORS, "n_jobs": -1},
            }

            def _build_model():
                return Pipeline(steps=[
                    ("pca", PCA(**MODEL_CONFIG["pca"])),
                    ("knn", KNeighborsClassifier(**MODEL_CONFIG["knn"])),
                ])
            MODEL_CONFIG
            """
        ),
        md(
            """
            ## 4 - Train and capture pca.n_components_

            Here we pull out the real number of components that PCA
            actually kept. On this dataset it's exactly 1 - the
            diagnostic is in the next cell.
            """
        ),
        code(
            """
            per_ds_results = {}
            pca_components_per_dataset = {}

            for ds_name in DATASETS:
                X_train, X_test = datasets[ds_name]
                model = _build_model()
                result = fit_and_score(model, X_train, y_train, X_test, y_test)
                per_ds_results[ds_name] = result

                pca_stage = result["model"].named_steps["pca"]
                n_components = int(pca_stage.n_components_)
                pca_components_per_dataset[ds_name] = n_components

                print_dataset_block(ds_name, X_train.shape, result)
                print(f"     PCA components retained for 90% variance : {n_components}")
                save_predictions(MODEL_SLUG, ds_name, result["y_pred"])

            print_delta(per_ds_results)
            save_test_index(MODEL_SLUG, y_test)
            display(pd.Series(pca_components_per_dataset, name="components retained at 90% variance").to_frame())
            """
        ),
    ]
    cells += confusion_matrix_cells()
    cells += roc_curve_cells()
    cells += save_metrics_cells(
        '''{
                "pca_components_retained": pca_components_per_dataset,
                "pca_variance_threshold":  PCA_VARIANCE,
                "knn_n_neighbors":         KNN_NEIGHBORS,
                "roc_auc_advanced":        auc,
            }'''
    )
    cells += summary_cells(
        "PCA(0.90) + KNN  (the cautionary tale)",
        """
        - **Test Accuracy / F1:** about 0.5092 / 0.5508 - below the
          0.534 majority-class rate.
        - **PCA kept just 1 component** - the diagnostic that pushed
          us to write 07_PCA_KNN_Improved.ipynb.
        - **The lesson:** we should grab structural diagnostics
          before reasoning only from theory (research.md section 4
          Lesson #5).
        """
    )
    return write_nb("06_PCA_KNN.ipynb", cells)


# 07 - PCA + KNN (Improved)
def build_pca_knn_improved_notebook() -> Path:
    """
    This function builds the PCA + KNN Improved notebook - the algorithmic fix.
    :return: the path of the notebook we just wrote
    """
    cells = [
        md(
            """
            # 07 - PCA(0.90) + KNN (Improved)  - the algorithmic fix

            This is the diagnostic-driven fix for the 1-component
            collapse from 06_PCA_KNN.ipynb. A ColumnTransformer
            explicitly drops the four nutrition columns
            (calories, protein, fat, sodium) BEFORE PCA, so the
            projection cannot be hijacked by their outlier-dominated
            variance anymore. The recovery is dramatic on the
            dimensionality side (1 -> about 200 components) and
            meaningful on the accuracy side (+4.5 to +6.3 pp). See
            research.md section 4 Lesson #5.
            """
        ),
    ]
    cells += setup_cells()
    cells += data_loading_cells()
    cells += [
        md(
            """
            ## 3 - Configure the improved pipeline

            The new top step of the pipeline is a ColumnTransformer
            whose only job is to DROP the four nutrition columns.
            Everything after it (PCA -> KNN) is identical to
            06_PCA_KNN.ipynb.
            """
        ),
        code(
            """
            from sklearn.compose import ColumnTransformer
            from sklearn.decomposition import PCA
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.pipeline import Pipeline

            MODEL_SLUG    = "pca_knn_improved"
            MODEL_NAME    = "PCA(0.90) + KNN (Improved)"
            DISPLAY_NAME  = "PCA(0.90) + KNN (Improved)"

            NUTRITION_COLS = ("calories", "protein", "fat", "sodium")
            PCA_VARIANCE = 0.90
            KNN_NEIGHBORS = 5

            MODEL_CONFIG = {
                "dropped_columns": list(NUTRITION_COLS),
                "pca": {"n_components": PCA_VARIANCE, "random_state": RANDOM_STATE},
                "knn": {"n_neighbors": KNN_NEIGHBORS, "n_jobs": -1},
            }

            def _build_model():
                return Pipeline(steps=[
                    ("drop_nutrition", ColumnTransformer(
                        transformers=[("drop_nutrition_cols", "drop", list(NUTRITION_COLS))],
                        remainder="passthrough",
                        verbose_feature_names_out=False,
                    )),
                    ("pca", PCA(**MODEL_CONFIG["pca"])),
                    ("knn", KNeighborsClassifier(**MODEL_CONFIG["knn"])),
                ])
            MODEL_CONFIG
            """
        ),
        md(
            """
            ## 4 - Train and capture pca.n_components_ (before/after the fix)

            With the four nutrition columns out of the way, the
            number of components PCA needs for 90% variance jumps
            from 1 (broken) to a few hundred (healthy).
            """
        ),
        code(
            """
            per_ds_results = {}
            pca_components_per_dataset = {}
            cols_before_drop = {}
            cols_after_drop = {}

            for ds_name in DATASETS:
                X_train, X_test = datasets[ds_name]
                cols_before_drop[ds_name] = X_train.shape[1]
                cols_after_drop[ds_name]  = X_train.shape[1] - len(NUTRITION_COLS)

                model = _build_model()
                result = fit_and_score(model, X_train, y_train, X_test, y_test)
                per_ds_results[ds_name] = result

                pca_stage = result["model"].named_steps["pca"]
                n_components = int(pca_stage.n_components_)
                pca_components_per_dataset[ds_name] = n_components

                print_dataset_block(ds_name, X_train.shape, result)
                print(f"     Columns before drop: {cols_before_drop[ds_name]}")
                print(f"     Columns after drop : {cols_after_drop[ds_name]}")
                print(f"     PCA components retained for 90% variance : "
                      f"{n_components}  (was 1 in 06_PCA_KNN.ipynb)")
                save_predictions(MODEL_SLUG, ds_name, result["y_pred"])

            print_delta(per_ds_results)
            save_test_index(MODEL_SLUG, y_test)
            display(pd.DataFrame({
                "components_retained": pca_components_per_dataset,
                "expansion_vs_original": {k: f"{v}x" for k, v in pca_components_per_dataset.items()},
            }))
            """
        ),
    ]
    cells += confusion_matrix_cells()
    cells += roc_curve_cells()
    cells += save_metrics_cells(
        '''{
                "dropped_columns":          list(NUTRITION_COLS),
                "columns_before_drop":      cols_before_drop,
                "columns_after_drop":       cols_after_drop,
                "pca_components_retained":  pca_components_per_dataset,
                "pca_variance_threshold":   PCA_VARIANCE,
                "knn_n_neighbors":          KNN_NEIGHBORS,
                "roc_auc_advanced":         auc,
            }'''
    )
    cells += summary_cells(
        "PCA(0.90) + KNN (Improved)",
        """
        - **Test Accuracy / F1:** about 0.5726 / 0.6023 - a +6.34 pp
          on Acc and +5.15 pp on F1 recovery over the unfixed version
          on Advanced.
        - **Search space grew 178x / 203x** (1 -> 178 components on
          Advanced, 1 -> 203 on Baseline).
        - **Only a partial rescue:** still about 3 pp under LR. The
          rest of the gap is the textbook sparse-binary distance
          penalty (which was the original theoretical objection).
          research.md section 4 Lesson #5.
        """
    )
    return write_nb("07_PCA_KNN_Improved.ipynb", cells)


# 08 - Master Comparison
def build_master_notebook() -> Path:
    """
    This function builds the master comparison notebook that puts all the
    models side by side.
    :return: the path of the notebook we just wrote
    """
    cells = [
        md(
            """
            # 08 - Master Comparison

            The headline table of the portfolio. This notebook does
            not train anything. It reads each model's
            results/<slug>/metrics.json, builds the side-by-side
            pandas summary, and shows the cross-model verdict (does
            the non-linear model break the linear ceiling or not?).

            Run this AFTER the seven model notebooks (or the seven
            train_*.py scripts) finished and filled results/.
            """
        ),
        md(
            """
            ## 1 - Setup
            """
        ),
        code(
            """
            %matplotlib inline
            import sys
            from pathlib import Path

            PROJECT_ROOT = Path.cwd().parent
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))

            import json
            import pandas as pd
            from IPython.display import Image, display

            from src.train_utils import RESULTS_DIR
            from evaluate_all_results import (
                load_metrics_files,
                order_payloads,
                missing_from_preferred,
                row_from_payload,
                build_summary_dataframe,
                PREFERRED_ORDER,
            )

            print(f"RESULTS_DIR = {RESULTS_DIR}")
            """
        ),
        md(
            """
            ## 2 - Discover what's inside results/

            Here we list every model sub-directory under results/ and
            check that the canonical 7-model lineup is all there.
            """
        ),
        code(
            """
            payloads = order_payloads(load_metrics_files(RESULTS_DIR))
            missing = missing_from_preferred(payloads)

            print(f"Found {len(payloads)} model result file(s).")
            if missing:
                print(f"Missing from the expected lineup: {missing}")
            else:
                print("All 7 expected models present.")
            """
        ),
        md(
            """
            ## 3 - The headline summary table

            Each row is one model, and the columns are accuracy and
            F1 on Baseline and on Advanced, plus the Delta between
            them. A positive Delta means the engineered culinary
            features helped that model.
            """
        ),
        code(
            """
            df = build_summary_dataframe(RESULTS_DIR)

            def _style_deltas(v):
                if not isinstance(v, (int, float)) or pd.isna(v):
                    return ""
                if v > 0.001:
                    return "color: green"
                if v < -0.001:
                    return "color: crimson"
                return "color: dimgray"

            (
                df.style
                  .format({
                      "Acc (Baseline)": "{:.4f}",
                      "Acc (Advanced)": "{:.4f}",
                      "Δ Acc": "{:+.4f}",
                      "F1 (Baseline)": "{:.4f}",
                      "F1 (Advanced)": "{:.4f}",
                      "Δ F1": "{:+.4f}",
                  })
                  .map(_style_deltas, subset=["Δ Acc", "Δ F1"])
                  .set_caption("7-model A/B comparison: Baseline vs Advanced (engineered culinary features)")
            )
            """
        ),
        md(
            """
            ## 4 - Cross-model verdict (linear baseline vs non-linear)

            LR is the calibrated linear baseline. Random Forest and
            MLP are the non-linear contenders. The "meaningful gain"
            we set is 1 percentage point on either Acc or F1.
            """
        ),
        code(
            """
            MEANINGFUL = 0.01
            by_name = {p["model_name"]: p for p in payloads}
            lr  = by_name.get("LogisticRegression")
            rf  = by_name.get("RandomForest")
            mlp = by_name.get("MLP (128,64)")
            assert lr is not None, "LogisticRegression metrics missing"

            def _classify(model_name, p):
                lr_a = lr["datasets"]["Advanced"]
                ad = p["datasets"]["Advanced"]
                d_acc = ad["accuracy"] - lr_a["accuracy"]
                d_f1 = ad["f1"] - lr_a["f1"]
                if d_acc >= MEANINGFUL and d_f1 >= MEANINGFUL:
                    verdict = "BREAKS the linear ceiling on both metrics"
                elif d_acc >= MEANINGFUL or d_f1 >= MEANINGFUL:
                    verdict = "PARTIAL gain (one metric only)"
                elif d_acc <= -MEANINGFUL or d_f1 <= -MEANINGFUL:
                    verdict = "UNDERPERFORMS the linear baseline"
                else:
                    verdict = "PLATEAUS within noise of LR"
                return {
                    "Model": model_name,
                    "Acc (Advanced)": ad["accuracy"],
                    "Δ vs LR (Acc)": d_acc,
                    "F1 (Advanced)": ad["f1"],
                    "Δ vs LR (F1)": d_f1,
                    "Verdict": verdict,
                }

            verdict_rows = []
            if rf is not None: verdict_rows.append(_classify("RandomForest", rf))
            if mlp is not None: verdict_rows.append(_classify("MLP (128,64)", mlp))
            verdict_df = pd.DataFrame(verdict_rows).set_index("Model")

            display(verdict_df.style.format({
                "Acc (Advanced)": "{:.4f}",
                "F1 (Advanced)": "{:.4f}",
                "Δ vs LR (Acc)": "{:+.4f}",
                "Δ vs LR (F1)": "{:+.4f}",
            }))
            """
        ),
        md(
            """
            ## 5 - The PCA + KNN before/after story

            This is the most diagnostic-driven result in the project:
            dropping the four nutrition columns before PCA grows the
            retained search space from 1 component to about 200, and
            lifts KNN accuracy by about 5 percentage points. README
            section 4 Lesson #5.
            """
        ),
        code(
            """
            orig = by_name.get("PCA(0.90) + KNN")
            improved = by_name.get("PCA(0.90) + KNN (Improved)")

            if orig is not None and improved is not None:
                pca_orig = orig["extras"]["pca_components_retained"]
                pca_imp  = improved["extras"]["pca_components_retained"]

                cmp = pd.DataFrame({
                    "Original - components": pca_orig,
                    "Improved - components": pca_imp,
                    "Original - Acc": {ds: orig["datasets"][ds]["accuracy"] for ds in pca_orig},
                    "Improved - Acc": {ds: improved["datasets"][ds]["accuracy"] for ds in pca_imp},
                    "Δ Acc": {ds: improved["datasets"][ds]["accuracy"] - orig["datasets"][ds]["accuracy"] for ds in pca_imp},
                })
                display(cmp.style.format({
                    "Original - Acc": "{:.4f}",
                    "Improved - Acc": "{:.4f}",
                    "Δ Acc": "{:+.4f}",
                }))
            """
        ),
        md(
            """
            ## 6 - Embedded plot gallery

            The headline plot of each model, side by side. The images
            come from results/<slug>/ so they show whatever was last
            written by the model's notebook (or by the matching
            train_<model>.py script).
            """
        ),
        code(
            """
            plot_gallery = [
                ("Logistic Regression - Confusion Matrix", "logistic_regression/confusion_matrix.png"),
                ("Logistic Regression - ROC Curve",        "logistic_regression/roc_curve.png"),
                ("Random Forest - Feature Importance",     "random_forest/feature_importance.png"),
                ("MLP - Loss Curve + Validation Overlay",  "mlp/loss_curve.png"),
            ]

            for caption, rel_path in plot_gallery:
                full_path = RESULTS_DIR / rel_path
                if full_path.exists():
                    print(caption)
                    display(Image(str(full_path)))
                else:
                    print(f"[missing] {caption} ({rel_path})")
            """
        ),
        md(
            """
            ## 7 - Final verdict (one-liner per model class)

            - **Linear interpretable champion:** Logistic Regression
              (calibrated probabilities, signed coefficients).
            - **Predictive champion:** Random Forest - it breaks the
              linear ceiling by +2.28 pp Acc / +3.49 pp F1.
            - **MLP:** clears the linear ceiling too once we add early
              stopping (+1.15 pp Acc / +2.27 pp F1 vs LR) - but only
              then; unregularised it lands far below LR.
            - **KNN (cautionary tale -> algorithmic fix):** the
              diagnostic-driven Improved variant lifts accuracy by
              +4.5 to +6.3 pp just by dropping the outlier-dominated
              columns before PCA.

            For the full narrative see results/research.md.
            """
        ),
    ]
    return write_nb("08_Master_Comparison.ipynb", cells)


# Entry point
def main() -> None:
    """
    The main function - here we just call every notebook builder one after the
    other and print the list of files we wrote.
    """
    paths = [
        build_logistic_regression_notebook(),
        build_random_forest_notebook(),
        build_mlp_notebook(),
        build_perceptron_notebook(),
        build_adaboost_notebook(),
        build_pca_knn_notebook(),
        build_pca_knn_improved_notebook(),
        build_master_notebook(),
    ]
    print(f"Wrote {len(paths)} notebooks:")
    for p in sorted(paths):
        print(f"  {p.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
