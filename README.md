# Epicurious Recipe Hit/Miss Classifier

## 📑 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [How to Run](#how-to-run)
- [Where the Outputs Land](#where-the-outputs-land)
- [The Research Story](#the-research-story)
- [About the Author](#about-the-author)


## 🔍 Overview

This project is a binary classification pipeline on the **Epicurious — Recipes
with Rating and Nutrition** dataset. Each recipe has a nutrition profile, a
matrix of ~674 editorial binary tags, and free-text directions/ingredients. The
task is to predict if a recipe is a **"Hit"** (rating ≥ 4.0) or a **"Miss"**
(rating < 4.0).

The full research narrative — methodology, findings, lessons learned, the
"42 Guarantee" audit, threshold-tuning analysis, the PCA / KNN diagnostic story,
etc. — lives in [`results/research.md`](results/research.md). This README is the
short version: how to run the code and how the project is laid out.


## 🏗️ Architecture

The project is split into three surfaces — a foundational `src/` package, a
collection of per-model training scripts, and a `notebooks/` portfolio surface
for interactive review. All three import from the same `src/` package, so a
change to the preprocessing pipeline (for example) shows up everywhere
automatically.

```
.
├── data/                              raw inputs (untouched)
│   ├── epi_r.csv
│   └── full_format_recipes.json
│
├── src/                               foundational package — everything imports from here
│   ├── __init__.py                    re-exports the public API
│   ├── _constants.py                  RANDOM_STATE = 42 (the only place 42 is bound)
│   ├── data_foundation.py             load CSV + JSON, merge, target, train/test split
│   ├── preprocessing.py               imputation, RobustScaler, Baseline + Advanced matrices
│   └── train_utils.py                 shared eval / JSON-payload / I/O / plotting helpers
│
├── notebooks/                         interactive portfolio surface (Jupyter)
│   ├── 01_Logistic_Regression.ipynb
│   ├── 02_Random_Forest.ipynb
│   ├── 03_MLP_Neural_Network.ipynb
│   ├── 04_Perceptron.ipynb
│   ├── 05_AdaBoost.ipynb
│   ├── 06_PCA_KNN.ipynb
│   ├── 07_PCA_KNN_Improved.ipynb
│   └── 08_Master_Comparison.ipynb     reads all metrics.json files, no training
│
├── tools/
│   └── generate_notebooks.py          rebuild the 8 notebooks from one template
│
├── results/                           per-model artifacts + the full research write-up
│   ├── perceptron/
│   ├── logistic_regression/
│   ├── adaboost/
│   ├── pca_knn/
│   ├── pca_knn_improved/
│   ├── random_forest/
│   ├── mlp/
│   ├── mlp_overfit/                   the deliberately unregularised MLP baseline
│   │                                  (train_mlp.py --no-early-stopping) — feeds
│   │                                  the early-stopping ablation
│   └── research.md                    the long-form research report
│
├── legacy/                            the dataset author's original scraper/helpers
│                                      (kept for provenance, imported by nothing)
│
├── train_perceptron.py                headless CLI entry points — one per model.
├── train_logistic_regression.py       Each imports from src/, trains on Baseline +
├── train_adaboost.py                  Advanced, and writes everything to
├── train_pca_knn.py                   results/<slug>/.
├── train_pca_knn_improved.py
├── train_random_forest.py
├── train_mlp.py                       also supports --no-early-stopping for the
│                                      reproducible overfit baseline (see research.md §3.6)
├── evaluate_all_results.py            pure-read aggregator — reads every
│                                      results/<slug>/metrics.json and prints the
│                                      side-by-side 7-model summary table.
├── analysis.py                        LR interpretability — top coefficients,
│                                      confidence buckets, ROC + confusion plots.
├── advanced_tuning.py                 threshold selection (on a validation split)
│                                      + top-20 LR features.
│
├── requirements.txt                   pinned dependency versions — the reproducibility
│                                      guarantee only holds with these exact versions
└── README.md                          this file
```


## 🚀 How to Run

You can drive this project from the command line OR from the Jupyter notebooks.
Both paths read the same data, use the same `src/` helpers, and write to the
same `results/<slug>/` directories.

### Setup

```bash
pip install -r requirements.txt
```

The versions are pinned on purpose — sklearn estimator behaviour changes
between releases, so the "identical metrics from a clean checkout" guarantee
only holds with these exact versions.

### Option A — Headless command line

This is the reproducible path: train every model, aggregate results, run the
post-hoc analysis. All seven train scripts are independent; you can run them in
any order.

```bash
# Train every model (any order, fully independent). 
# Each writes to results/<slug>/metrics.json + predictions + plots.
python train_perceptron.py
python train_logistic_regression.py
python train_adaboost.py
python train_pca_knn.py
python train_pca_knn_improved.py
python train_random_forest.py
python train_mlp.py
python train_mlp.py --no-early-stopping   # optional: the overfit baseline for
                                          # the §3.6 ablation -> results/mlp_overfit/

# Pure-read aggregator — prints the 7-row summary + cross-model verdict.
python evaluate_all_results.py

# Optional post-hoc analyses
python analysis.py          # LR coefficients, confidence buckets, ROC/CM
python advanced_tuning.py   # threshold selection (validation split) + top-20 LR features
```

### Option B — Jupyter notebooks

For the portfolio walk-through. Each notebook trains its model inline, renders
plots inline, and saves the same artifacts to disk as the CLI scripts.

```bash
jupyter lab notebooks/
# then open 01..07 in any order; 08 reads everyone's metrics and trains nothing.
```

### Regenerating the notebooks

The 8 notebooks are programmatically built from a single template in
`tools/generate_notebooks.py`. If you want to change the standard cell shape,
edit that file and re-run:

```bash
python tools/generate_notebooks.py
```


## 📂 Where the Outputs Land

Every model writes to its own subdirectory under `results/`. After a full run
each `results/<slug>/` looks like this:

| File | What it is |
|---|---|
| `metrics.json` | accuracy, F1, confusion matrix, per-class error rates, model config + extras |
| `predictions_baseline.npy` | int8 test predictions on the Baseline matrix |
| `predictions_advanced.npy` | int8 test predictions on the Advanced matrix |
| `test_index.npy` | the test split's row index — makes the prediction files self-describing |
| `confusion_matrix.png` | annotated heatmap (Advanced fit) |
| `roc_curve.png` | ROC + AUC (Advanced fit) |
| `feature_importance.png` *(Random Forest only)* | top-20 impurity-based bar chart |
| `loss_curve.png` *(MLP only)* | training loss + validation error overlay |

The aggregator (`evaluate_all_results.py` and `notebooks/08_Master_Comparison.ipynb`)
both walk these directories — they don't care which models have been trained yet,
and report missing ones nicely.


## 📚 The Research Story

The longer-form report — methodology, the seven-model A/B comparison, the
"Recipe for Success vs Recipe for Disaster" feature analysis, the threshold-vs-
balance trade-off, the MLP overfitting → early-stopping ablation, the PCA + KNN
1-component diagnostic and its algorithmic fix, lessons learned, etc. — is in
[`results/research.md`](results/research.md).


## 👨‍💻 About the Author

**Itay Segev** — Computer Science student.

This project is the final assignment for a Machine Learning course. Built as
a step-by-step exploration: start with a clean data foundation, layer a
preprocessing pipeline on top, compare a fleet of seven classifiers, dig into
the interpretable ones, and end up with a reproducible Jupyter portfolio.
