"""
Programmatically build all 8 portfolio notebooks via `nbformat`.

The generator lives here so the notebook structure is reproducible and
auditable: every model's notebook follows the same cell shape (markdown
intro → setup → load data → configure → train → plot inline → save
artifacts → summary), and any structural change is made in this one
file. Run once after a refactor:

    python tools/generate_notebooks.py

Outputs all eight `.ipynb` files into `notebooks/`. Each notebook
imports from `src/` (via `sys.path` insertion) and saves its artifacts
into `results/<slug>/`.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOKS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Cell-building helpers
# ---------------------------------------------------------------------------
def md(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


def write_nb(filename: str, cells: List[Any]) -> Path:
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


# ---------------------------------------------------------------------------
# Standard cell blocks reused across model notebooks
# ---------------------------------------------------------------------------
def setup_cells() -> List[Any]:
    return [
        md(
            """
            ## 1 — Setup

            Configure matplotlib for inline rendering, add the project
            root to `sys.path` so we can import from `src/`, and pull in
            the shared helpers + the project-wide `RANDOM_STATE = 42`.
            """
        ),
        code(
            """
            %matplotlib inline
            import sys
            from pathlib import Path

            # `notebooks/` is one level below the project root; add the
            # project root so `from src import ...` resolves correctly.
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
                save_figure,
                confusion_matrix_figure,
                roc_curve_figure,
                print_dataset_block,
                print_delta,
                model_results_dir,
            )

            print(f"RANDOM_STATE = {RANDOM_STATE} (single source of truth in src.train_utils)")
            """
        ),
    ]


def data_loading_cells() -> List[Any]:
    return [
        md(
            """
            ## 2 — Load the preprocessed feature matrices

            `load_preprocessed()` returns the two parallel matrices
            produced by Phase 1: the **Baseline** (nutrition + tags only)
            and the **Advanced** (Baseline + 9 engineered culinary
            features). Both share the same train/test partition so the
            A/B comparison is apples-to-apples.
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
    """Standard 'train on Baseline + Advanced' loop, parameterised by model factory."""
    return [
        md(
            """
            ## 4 — Train on both matrices

            Each matrix gets a **fresh** model instance (the factory
            below is called once per dataset). Nothing carries over
            between Baseline and Advanced — that's how the engineered
            feature delta stays attributable to the additional 9 columns
            alone.
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
            """
        ),
    ]


def confusion_matrix_cells() -> List[Any]:
    return [
        md(
            """
            ## 5 — Confusion matrix (Advanced fit)

            Annotated heatmap of the Advanced-fit confusion matrix.
            Rendered inline AND persisted to
            `results/<slug>/confusion_matrix.png`.
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
                title=f"{DISPLAY_NAME} — Confusion Matrix (Advanced)",
            )
            save_figure(MODEL_SLUG, "confusion_matrix.png", fig_cm)
            plt.show()
            """
        ),
    ]


def roc_curve_cells() -> List[Any]:
    return [
        md(
            """
            ## 6 — ROC curve + AUC (Advanced fit)

            Generated against `predict_proba` (or `decision_function`
            where probabilities aren't available). The AUC is the
            ranking-quality summary independent of any threshold choice;
            Phase 4's threshold-sweep work depends on it.
            """
        ),
        code(
            """
            fig_roc, auc = roc_curve_figure(
                y_test, adv["proba_hit"],
                title=f"{DISPLAY_NAME} — ROC Curve (Advanced)",
                model_label=DISPLAY_NAME,
            )
            save_figure(MODEL_SLUG, "roc_curve.png", fig_roc)
            print(f"Test ROC AUC (Advanced): {auc:.4f}")
            plt.show()
            """
        ),
    ]


def save_metrics_cells(extras_code: str = "{}") -> List[Any]:
    return [
        md(
            """
            ## 8 — Persist the canonical metrics JSON

            One JSON per model, written to
            `results/<slug>/metrics.json`. Schema is defined in
            `src.train_utils.build_metrics_payload`; the master
            comparison notebook reads from here.
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
    return [
        md(
            f"""
            ## 9 — Summary

            **Model:** {title}

            {summary_md.strip()}

            Run the **master comparison notebook
            (`08_Master_Comparison.ipynb`)** to see this model alongside
            the other six in the side-by-side table.
            """
        ),
    ]


# ---------------------------------------------------------------------------
# 01 — Logistic Regression
# ---------------------------------------------------------------------------
def build_logistic_regression_notebook() -> Path:
    cells = [
        md(
            """
            # 01 — Logistic Regression

            Logistic Regression (L2-penalised, `liblinear` solver) is
            the **interpretable champion** of the project: signed,
            directly-readable coefficients in IQR-units, calibrated
            `predict_proba` for the Phase 4 threshold sweep, and the
            converged solver that exposed the `lbfgs` non-convergence
            bug documented in README §2.4.

            All randomness is seeded from
            `src.train_utils.RANDOM_STATE = 42`.
            """
        ),
    ]
    cells += setup_cells()
    cells += data_loading_cells()
    cells += [
        md(
            """
            ## 3 — Configure the model

            `solver='liblinear'` (coordinate descent) converges
            reliably on this 678–687-dim sparse-binary input where
            `lbfgs` did not. `C=1.0` is the default L2 strength;
            `max_iter=5000` is generous head-room (liblinear converges
            well before this on this dataset).
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
            ## 7 — Top signed coefficients

            LR is the model whose internals are directly readable. The
            top 10 positive and top 10 negative coefficients tell the
            "Recipe for Success / Disaster" story documented in README
            §3.4.
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
        - **Test Accuracy / F1 / AUC:** ~0.6030 / ~0.6435 / ~0.6500
        - **Interpretable champion** — top coefficient drives the
          "Recipe for Success" narrative in README §3.4.
        - **Calibrated probabilities** — feeds the Phase 4 threshold
          sweep (`phase4_advanced_tuning.py`).
        """
    )
    return write_nb("01_Logistic_Regression.ipynb", cells)


# ---------------------------------------------------------------------------
# 02 — Random Forest
# ---------------------------------------------------------------------------
def build_random_forest_notebook() -> Path:
    cells = [
        md(
            """
            # 02 — Random Forest

            The **predictive champion** of the project. Random Forest
            breaks past Logistic Regression's "linear ceiling" (+1.48%
            Acc, +2.73% F1 on Advanced) by exploiting non-linear
            feature interactions. Critically, its feature-importance
            ranking **inverts** the LR coefficient picture — see README
            §3.5.
            """
        ),
    ]
    cells += setup_cells()
    cells += data_loading_cells()
    cells += [
        md(
            """
            ## 3 — Configure the model

            200 trees gives a small bump over the default 100 at
            negligible cost; `n_jobs=-1` uses all available cores.
            `random_state=RANDOM_STATE` propagates the 42 seed to the
            bagging RNG.
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
            ## 7 — Top-20 feature importances (the inversion story)

            The top 7 RF features are **all continuous numerics** —
            three engineered (`avg_words_per_step`, `num_ingredients`,
            `num_steps`) and four from the original nutrition columns.
            This is the **opposite** of the LR coefficient picture
            where binary tags dominated. The mechanism: L2 splits
            credit among collinear features, but trees pick the single
            best splitter per node and are invariant to that
            collinearity. See README §3.5.
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
            ax.set_title(f"Random Forest — Top {TOP_K} Feature Importances (Advanced)")
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
        - **Test Accuracy / F1:** ~0.6178 / ~0.6708 — top scores in the
          whole lineup.
        - **Breaks the linear ceiling** by +1.48% Acc, +2.73% F1 vs LR.
        - **Inverts the feature-importance ranking** — continuous
          numerics dominate where they were near-zero in LR.
        """
    )
    return write_nb("02_Random_Forest.ipynb", cells)


# ---------------------------------------------------------------------------
# 03 — MLP Neural Network
# ---------------------------------------------------------------------------
def build_mlp_notebook() -> Path:
    cells = [
        md(
            """
            # 03 — MLP Neural Network (with early stopping)

            Two-hidden-layer (128, 64) MLP. The **unregularised**
            version of this model drove training loss to ~0.009 in 64
            epochs while test accuracy fell BELOW Logistic Regression —
            a textbook overfitting signature. This notebook trains the
            **regularised** version with validation-based early
            stopping (`early_stopping=True`, `validation_fraction=0.15`,
            `n_iter_no_change=10`), which recovers test performance to
            match LR within noise (see README §3.6).
            """
        ),
    ]
    cells += setup_cells()
    cells += data_loading_cells()
    cells += [
        md(
            """
            ## 3 — Configure the model

            Validation-based early stopping holds out 15% of `X_train`
            internally (NOT our test split — leakage-free), tracks
            validation accuracy each epoch, and restores the weights
            to the best-validation epoch on `fit()` completion.
            `verbose=True` exposes the per-epoch loss/validation log.
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
            ## 7 — Training-loss / validation-error overlay

            Training loss (orange) descends smoothly toward zero;
            validation error (green) flattens and starts climbing
            slightly — the classic overfitting signature. The dashed
            vertical line marks the epoch sklearn restored the
            network's weights to (the epoch with the best validation
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
                    label="Validation error (1 − accuracy)")
            ax.axvline(x=best_epoch, color="dimgray", linestyle="--", lw=1.5,
                       label=f"Restored epoch = {best_epoch} (best val acc = {best_val_acc:.4f})")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss / Error")
            ax.set_title(f"MLP — Training Loss vs Validation Error ({len(loss)} epochs)")
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
            ## 7.5 — Early-stopping ablation (vs the prior overfit run)

            Before early stopping was enabled, the same architecture
            ran for 64 epochs to training loss 0.0087 and test accuracy
            0.5816 (below LR). With early stopping, the network
            restores to the best-validation epoch and recovers to ~0.61
            test accuracy — matching LR within noise. See README §3.6.
            """
        ),
        code(
            """
            MLP_OVERFITTED_PRIOR = {
                "epochs":       64,
                "final_loss":   0.0087,
                "advanced_acc": 0.5816,
                "advanced_f1":  0.6107,
            }

            now = per_ds_results["Advanced"]
            d_acc = now["accuracy"] - MLP_OVERFITTED_PRIOR["advanced_acc"]
            d_f1 = now["f1"] - MLP_OVERFITTED_PRIOR["advanced_f1"]

            display(pd.DataFrame({
                "Previous (no early stopping)": [
                    f"{MLP_OVERFITTED_PRIOR['epochs']} epochs",
                    f"loss → {MLP_OVERFITTED_PRIOR['final_loss']:.4f}",
                    f"{MLP_OVERFITTED_PRIOR['advanced_acc']:.4f}",
                    f"{MLP_OVERFITTED_PRIOR['advanced_f1']:.4f}",
                ],
                "Current (early stopping)": [
                    f"{diagnostics['epochs_total']} epochs (restored at {best_epoch})",
                    f"loss → {diagnostics['training_loss_at_final']:.4f}",
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
                "overfitted_prior_run":       MLP_OVERFITTED_PRIOR,
                "roc_auc_advanced":           auc,
            }'''
    )
    cells += summary_cells(
        "MLP (128, 64) + early stopping",
        """
        - **Test Accuracy / F1:** ~0.6066 / ~0.6477 — matches LR
          within noise after early stopping.
        - **Without early stopping** the same architecture overfit
          (Acc 0.5816, below LR) — see the ablation table in §7.5.
        - **Lesson:** capacity is a liability without matching
          regularisation. README §4 Lesson #6.
        """
    )
    return write_nb("03_MLP_Neural_Network.ipynb", cells)


# ---------------------------------------------------------------------------
# 04 — Perceptron
# ---------------------------------------------------------------------------
def build_perceptron_notebook() -> Path:
    cells = [
        md(
            """
            # 04 — Perceptron

            The weakest linear baseline in the lineup. Included as a
            sanity-check floor against which the more sophisticated
            models are measured. Approaches its Bayes-error rate, then
            plateaus.
            """
        ),
    ]
    cells += setup_cells()
    cells += data_loading_cells()
    cells += [
        md(
            """
            ## 3 — Configure the model

            Default Perceptron with `random_state=RANDOM_STATE` for
            the SGD shuffling RNG; `max_iter=1000` and `tol=1e-3`
            match the sklearn defaults.
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
    cells += save_metrics_cells()
    cells += summary_cells(
        "Perceptron",
        """
        - **Test Accuracy / F1:** ~0.5457 / ~0.5696 — the weakest
          model in the lineup.
        - **Nearly symmetric errors** (FP ≈ FN), in contrast to
          AdaBoost's strong asymmetry.
        """
    )
    return write_nb("04_Perceptron.ipynb", cells)


# ---------------------------------------------------------------------------
# 05 — AdaBoost
# ---------------------------------------------------------------------------
def build_adaboost_notebook() -> Path:
    cells = [
        md(
            """
            # 05 — AdaBoost

            Boosted decision stumps — the mildly-non-linear member of
            the linear lineup. Records the strongest class asymmetry of
            any model in the study (~0.575 FP-rate vs 0.257 FN-rate at
            default threshold), making it a useful counter-example to
            the symmetric-errors story Logistic Regression tells in
            Phase 4's threshold sweep.
            """
        ),
    ]
    cells += setup_cells()
    cells += data_loading_cells()
    cells += [
        md(
            """
            ## 3 — Configure the model

            100 boosting iterations; `random_state=RANDOM_STATE` seeds
            the bootstrap sampling.
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
        - **Test Accuracy / F1:** ~0.5967 / ~0.6518.
        - **Strongest class asymmetry** in the lineup — FP-rate 0.575
          vs FN-rate 0.257. The engineered features re-shape its
          errors rather than reducing them (ΔF1 = -0.0103).
        """
    )
    return write_nb("05_AdaBoost.ipynb", cells)


# ---------------------------------------------------------------------------
# 06 — PCA + KNN
# ---------------------------------------------------------------------------
def build_pca_knn_notebook() -> Path:
    cells = [
        md(
            """
            # 06 — PCA(0.90) + KNN  (the cautionary tale)

            This notebook trains the **original** PCA + KNN pipeline
            that scored only ~0.515 accuracy. The reason will be
            visible in the cell that prints `pca.n_components_`: PCA
            retains exactly **1 component** for 90% variance, because
            the outlier-dominated nutrition columns hijack the
            projection. KNN is then asked to do similarity search on a
            1-D projection of the data. See README §4 Lesson #5.

            The **fix** for this is in
            `07_PCA_KNN_Improved.ipynb`.
            """
        ),
    ]
    cells += setup_cells()
    cells += data_loading_cells()
    cells += [
        md(
            """
            ## 3 — Configure the pipeline

            `PCA(n_components=0.90)` keeps the smallest set of
            principal components that together retain 90% of the input
            variance. `KNN(n_neighbors=5)` operates in that reduced
            space. Both stages share the same `random_state`. KNN
            itself is deterministic (no random_state to set).
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
            ## 4 — Train and capture `pca.n_components_`

            We extract the actual number of components PCA retained.
            On this dataset it's exactly **1** — see the diagnostic
            below.
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
        - **Test Accuracy / F1:** ~0.5133 / ~0.5578 — barely above the
          0.534 majority-class rate.
        - **PCA retained 1 component** — the diagnostic that motivated
          `07_PCA_KNN_Improved.ipynb`.
        - **Lesson:** capture structural diagnostics before reasoning
          from theory alone (README §4 Lesson #5).
        """
    )
    return write_nb("06_PCA_KNN.ipynb", cells)


# ---------------------------------------------------------------------------
# 07 — PCA + KNN (Improved)
# ---------------------------------------------------------------------------
def build_pca_knn_improved_notebook() -> Path:
    cells = [
        md(
            """
            # 07 — PCA(0.90) + KNN (Improved)  — the algorithmic fix

            Diagnostic-driven fix for the 1-component collapse
            documented in `06_PCA_KNN.ipynb`. A `ColumnTransformer`
            explicitly drops the four nutrition columns
            (`calories, protein, fat, sodium`) BEFORE PCA, so the
            projection can't be hijacked by their outlier-dominated
            variance. The recovery is **dramatic on dimensionality**
            (1 → ~200 components) and **meaningful on accuracy** (+4.5
            to +5 percentage points). See README §4 Lesson #5.
            """
        ),
    ]
    cells += setup_cells()
    cells += data_loading_cells()
    cells += [
        md(
            """
            ## 3 — Configure the improved pipeline

            The new top-of-pipeline step is a `ColumnTransformer`
            whose only job is to DROP the four nutrition columns.
            Everything downstream (PCA → KNN) is identical to
            `06_PCA_KNN.ipynb`.
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
            ## 4 — Train and capture `pca.n_components_` (before/after fix)

            With the four nutrition columns out of the way, PCA's
            component count for 90% variance should jump from **1**
            (broken) to a few hundred (healthy).
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
            display(pd.DataFrame({
                "components_retained": pca_components_per_dataset,
                "expansion_vs_original": {k: f"{v}×" for k, v in pca_components_per_dataset.items()},
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
        - **Test Accuracy / F1:** ~0.5630 / ~0.5904 — a +4.97 pp Acc
          and +3.26 pp F1 recovery over the unfixed version on
          Advanced.
        - **Search space expanded 182× / 203×** (1 → 182 components
          on Advanced; 1 → 203 on Baseline).
        - **Partial rescue:** still ~4 pp below LR. The residual gap
          is the textbook sparse-binary distance penalty (the original
          theoretical objection). README §4 Lesson #5.
        """
    )
    return write_nb("07_PCA_KNN_Improved.ipynb", cells)


# ---------------------------------------------------------------------------
# 08 — Master Comparison
# ---------------------------------------------------------------------------
def build_master_notebook() -> Path:
    cells = [
        md(
            """
            # 08 — Master Comparison

            **The portfolio's headline table.** This notebook trains
            nothing. It reads each model's
            `results/<slug>/metrics.json`, assembles the side-by-side
            Pandas summary, and surfaces the cross-model verdict (does
            the non-linear model break the linear ceiling?).

            Run this AFTER the seven model notebooks (or the seven
            `train_*.py` scripts) have populated `results/`.
            """
        ),
        md(
            """
            ## 1 — Setup
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
                _load_metrics_files,
                _order_payloads,
                _missing_from_preferred,
                _row_from_payload,
                build_summary_dataframe,
                PREFERRED_ORDER,
            )

            print(f"RESULTS_DIR = {RESULTS_DIR}")
            """
        ),
        md(
            """
            ## 2 — Discover what's in `results/`

            We list every model subdirectory under `results/` and
            confirm that the canonical 7-model lineup is present.
            """
        ),
        code(
            """
            payloads = _order_payloads(_load_metrics_files(RESULTS_DIR))
            missing = _missing_from_preferred(payloads)

            print(f"Found {len(payloads)} model result file(s).")
            if missing:
                print(f"Missing from the expected lineup: {missing}")
            else:
                print("All 7 expected models present.")
            """
        ),
        md(
            """
            ## 3 — The headline summary table

            Each row is one model; columns are accuracy and F1 on the
            Baseline and Advanced matrices, plus the Δ between them.
            Positive Δ means the engineered culinary features helped.
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
            ## 4 — Cross-model verdict (linear baseline vs non-linear)

            LR is the calibrated linear baseline. Random Forest and
            MLP are the non-linear contenders. The "meaningful gain"
            threshold is 1 percentage point on either Acc or F1.
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
                    verdict = "✅ BREAKS the linear ceiling on both metrics"
                elif d_acc >= MEANINGFUL or d_f1 >= MEANINGFUL:
                    verdict = "🟡 PARTIAL gain (one metric only)"
                elif d_acc <= -MEANINGFUL or d_f1 <= -MEANINGFUL:
                    verdict = "❌ UNDERPERFORMS the linear baseline"
                else:
                    verdict = "⚖️ PLATEAUS within noise of LR"
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
            ## 5 — The PCA + KNN before/after story

            The most diagnostic-driven result in the project: dropping
            the four nutrition columns before PCA expands the retained
            search space from 1 component to ~200 components and lifts
            KNN accuracy by ~5 percentage points. README §4 Lesson #5.
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
                    "Original — components": pca_orig,
                    "Improved — components": pca_imp,
                    "Original — Acc": {ds: orig["datasets"][ds]["accuracy"] for ds in pca_orig},
                    "Improved — Acc": {ds: improved["datasets"][ds]["accuracy"] for ds in pca_imp},
                    "Δ Acc": {ds: improved["datasets"][ds]["accuracy"] - orig["datasets"][ds]["accuracy"] for ds in pca_imp},
                })
                display(cmp.style.format({
                    "Original — Acc": "{:.4f}",
                    "Improved — Acc": "{:.4f}",
                    "Δ Acc": "{:+.4f}",
                }))
            """
        ),
        md(
            """
            ## 6 — Embedded plot gallery

            Each model's headline plot, side-by-side. Images come from
            `results/<slug>/`, so they reflect whatever was last
            written by the model's notebook (or the matching
            `train_<model>.py` script).
            """
        ),
        code(
            """
            plot_gallery = [
                ("Logistic Regression — Confusion Matrix", "logistic_regression/confusion_matrix.png"),
                ("Logistic Regression — ROC Curve",        "logistic_regression/roc_curve.png"),
                ("Random Forest — Feature Importance",     "random_forest/feature_importance.png"),
                ("MLP — Loss Curve + Validation Overlay",  "mlp/loss_curve.png"),
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
            ## 7 — Final verdict (one-liner per model class)

            - **Linear interpretable champion:** Logistic Regression
              (calibrated probabilities, signed coefficients).
            - **Predictive champion:** Random Forest — breaks the
              linear ceiling +1.48% Acc / +2.73% F1.
            - **MLP:** matches LR within noise after early stopping;
              capacity ≠ accuracy on this dataset.
            - **KNN (cautionary tale → algorithmic fix):** the
              diagnostic-driven `Improved` variant lifts accuracy +5
              pp by dropping outlier-dominated columns before PCA.

            See `README.md` for the full narrative.
            """
        ),
    ]
    return write_nb("08_Master_Comparison.ipynb", cells)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
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
