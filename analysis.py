"""
This script is Phase 3 of the project - the interpretation and confidence analysis.
We fit one LogisticRegression on the Advanced matrix from Phase 1 and then we open
it up in three ways: top-K signed coefficients, predict_proba confidence buckets
on the test set, and two PNGs (confusion matrix + ROC).

We use LogisticRegression because Phase 2 showed it was the best balanced model,
and its coefficients are directly readable on the RobustScaled space - a +1 IQR
move in a feature shifts the log-odds by exactly the coefficient.

About the titles - Phase 0 drops the title column before splitting, so we re-merge
the raw frames here just to recover the title series aligned to the same RangeIndex.
The re-merge is cheap and it keeps the upstream API stable.
"""

from __future__ import annotations

from typing import Iterable

import matplotlib
matplotlib.use("Agg")              # so we can save PNGs without a display backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from src.data_foundation import (
    CulinaryFeatureExtractor,
    clean_and_binarize,
    load_binary_matrix,
    load_recipe_text,
    merge_datasets,
)
from src.preprocessing import build_preprocessed_datasets


from src._constants import RANDOM_STATE  # single source of truth, see src/_constants.py
TOP_K = 15                          # how many top coefficients per direction
TOP_CONFIDENCE = 5                  # how many rows we print per confidence bucket
TITLE_WIDTH = 70                    # truncate titles so the printing stays tidy

# the LR plots go into the same per-model results folder that the LR script uses
from src.train_utils import RESULTS_DIR as _RESULTS_DIR
_LR_RESULTS = _RESULTS_DIR / "logistic_regression"
_LR_RESULTS.mkdir(parents=True, exist_ok=True)
CM_PNG  = str(_LR_RESULTS / "confusion_matrix.png")
ROC_PNG = str(_LR_RESULTS / "roc_curve.png")

# the 9 engineered culinary feature names, built from Phase 0 so they will not
# drift if those constants ever change
CULINARY_FEATURE_NAMES: tuple = (
    tuple(CulinaryFeatureExtractor.BASELINE_FEATURES)
    + tuple(f"has_{g}" for g in CulinaryFeatureExtractor.KEYWORD_GROUPS)
)


# Title recovery
def _recover_titles() -> pd.Series:
    """
    Here we rebuild the title series from Phase 0 aligned to the split's RangeIndex.
    :return: the title series as strings
    """
    bin_df = load_binary_matrix()
    txt_df = load_recipe_text()
    merged = merge_datasets(bin_df, txt_df, verbose=False)
    merged, _ = clean_and_binarize(merged, verbose=False)
    return merged["title"].astype(str)


# Printing helpers
def _truncate(s: str, width: int = TITLE_WIDTH) -> str:
    """
    This helper truncates a title string so the columns stay aligned when we print.
    :param s: the title to truncate
    :param width: the max width we allow
    :return: the truncated title with a trailing ellipsis if it was cut
    """
    s = " ".join(s.split())          # collapse the whitespace
    return s if len(s) <= width else s[: width - 1] + "…"


def _print_top_coefficients(coefs: pd.Series, k: int) -> None:
    """
    This function prints the top-k positive and top-k negative LR coefficients,
    and then spotlights where our engineered culinary features ended up.
    :param coefs: the signed coefficients of the LR model
    :param k: how many to take per direction
    """
    top_pos = coefs.sort_values(ascending=False).head(k)
    top_neg = coefs.sort_values(ascending=True).head(k)

    print(f"\n  Top {k} POSITIVE coefficients (strongest Hit indicators)")
    print(f"  {'coef':>10}   {'feature'}")
    print(f"  {'-' * 10}   {'-' * 50}")
    for name, val in top_pos.items():
        print(f"  {val:+10.4f}   {name}")

    print(f"\n  Top {k} NEGATIVE coefficients (strongest Miss indicators)")
    print(f"  {'coef':>10}   {'feature'}")
    print(f"  {'-' * 10}   {'-' * 50}")
    for name, val in top_neg.items():
        print(f"  {val:+10.4f}   {name}")

    # Spotlight - where did our 9 engineered culinary features end up
    eng_set = set(CULINARY_FEATURE_NAMES)
    eng_coefs = coefs[coefs.index.isin(eng_set)].sort_values(
        key=lambda s: s.abs(), ascending=False
    )
    print("\n  >> ALL 9 engineered culinary features, ranked by |coef|:")
    n_features = len(coefs)
    abs_rank = coefs.abs().rank(ascending=False, method="min").astype(int)
    for name, val in eng_coefs.items():
        rank = int(abs_rank[name])
        direction = "Hit" if val > 0 else "Miss"
        print(
            f"     {val:+.4f}   rank {rank:>4}/{n_features}"
            f"   pushes toward {direction:<4}   {name}"
        )

    # Now we check if any culinary feature actually cracked the top-15 lists
    in_top_pos = [n for n in top_pos.index if n in eng_set]
    in_top_neg = [n for n in top_neg.index if n in eng_set]
    if in_top_pos or in_top_neg:
        print("\n  >> Culinary features that cracked the top-15 lists:")
        for n in in_top_pos:
            print(f"     POSITIVE: {n} ({top_pos[n]:+.4f})")
        for n in in_top_neg:
            print(f"     NEGATIVE: {n} ({top_neg[n]:+.4f})")
    else:
        print(
            "\n  >> None of the engineered culinary features cracked the "
            "top-15 lists — the binary tag matrix dominates."
        )


def _print_confidence_groups(
    proba_hit: np.ndarray,
    y_true: pd.Series,
    titles: pd.Series,
    k: int,
) -> None:
    """
    Here we print the most-confident Hits, the most-confident Misses, and the
    most-borderline test rows so we can look at the actual recipes by hand.
    :param proba_hit: P(class = Hit) from predict_proba
    :param y_true: the true labels for the test set
    :param titles: the titles of the test recipes
    :param k: how many rows we want per bucket
    """
    df = pd.DataFrame({
        "title":    titles.values,
        "true":     y_true.values,
        "p_hit":    proba_hit,
    })
    df["pred"] = (df["p_hit"] >= 0.5).astype(int)
    df["dist_to_05"] = (df["p_hit"] - 0.5).abs()
    df["correct"] = (df["true"] == df["pred"])

    def _emit(label: str, rows: pd.DataFrame) -> None:
        # this little helper prints one bucket of rows
        print(f"\n  {label}")
        print(f"     {'p(Hit)':>7}  {'true':>4}  {'pred':>4}  {'verdict':>7}   title")
        print(f"     {'-'*7}  {'-'*4}  {'-'*4}  {'-'*7}   {'-'*TITLE_WIDTH}")
        for _, r in rows.iterrows():
            verdict = "OK" if r["correct"] else "WRONG"
            true_lbl = "Hit" if r["true"] == 1 else "Miss"
            pred_lbl = "Hit" if r["pred"] == 1 else "Miss"
            print(
                f"     {r['p_hit']:7.4f}  {true_lbl:>4}  {pred_lbl:>4}  "
                f"{verdict:>7}   {_truncate(r['title'])}"
            )

    most_hit = df.sort_values("p_hit", ascending=False).head(k)
    most_miss = df.sort_values("p_hit", ascending=True).head(k)
    borderline = df.sort_values("dist_to_05", ascending=True).head(k)

    _emit(f"Top {k} MOST CONFIDENT HIT predictions  (p closest to 1.0):", most_hit)
    _emit(f"Top {k} MOST CONFIDENT MISS predictions (p closest to 0.0):", most_miss)
    _emit(f"Top {k} MOST BORDERLINE predictions     (p closest to 0.5):", borderline)


# Plots
def _save_confusion_matrix_plot(cm: np.ndarray, path: str) -> None:
    """
    This function saves the confusion matrix as a PNG.
    :param cm: the confusion matrix
    :param path: where we want to save the file
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Miss (0)", "Hit (1)"],
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)
    ax.set_title("Logistic Regression — Confusion Matrix (Advanced)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_roc_plot(y_true: Iterable[int], y_score: Iterable[float], path: str) -> float:
    """
    Here we draw the ROC curve and save it as a PNG, with the AUC in the legend.
    :param y_true: the true labels
    :param y_score: the predicted probability for class Hit
    :param path: where we want to save the PNG
    :return: the AUC we calculated so the caller can sanity check it
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = float(roc_auc_score(y_true, y_score))

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(fpr, tpr, lw=2.0, label=f"LogReg (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1.0,
            label="Random (AUC = 0.50)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("Logistic Regression — ROC Curve (Advanced)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return auc


# Entry point
def main() -> None:
    """
    This is the main function that runs the whole Phase 3 analysis end-to-end:
    train the LR, then run the 3 analyses on it.
    """
    print("=" * 72)
    print("  PHASE 3 — INTERPRETATION & CONFIDENCE ANALYSIS")
    print("=" * 72)

    print("Loading preprocessed datasets (Phase 1)...")
    (
        _, _,
        X_train_adv, X_test_adv,
        y_train, y_test,
    ) = build_preprocessed_datasets(verbose=False)

    print("Recovering recipe titles aligned to the test split...")
    titles_all = _recover_titles()
    test_titles = titles_all.loc[X_test_adv.index]

    # sanity check - the titles should line up 1:1 with the test rows
    assert len(test_titles) == len(X_test_adv), (
        "Title recovery mis-aligned — indices drifted between phases."
    )

    # Now we train the interpretable model
    print(f"\nTraining LogisticRegression on Advanced X_train {X_train_adv.shape}...")
    # The solver choice matters here. With 687 features (mostly sparse 0/1 tags,
    # many of them collinear), lbfgs did not converge clean even with max_iter=5000
    # - the same test recipe's probability swung from 0.9997 to 0.0000 between
    # max_iter=2000 and max_iter=5000, which means the optimum was nowhere near
    # reached. liblinear uses coordinate descent (one feature at a time) and it
    # converges fine on this sparse-binary high-dim input. Without convergence
    # the coefficient importance and confidence numbers below would not be
    # reproducible.
    lr = LogisticRegression(
        solver="liblinear",
        C=1.0,
        max_iter=5000,
        random_state=RANDOM_STATE,
    )
    lr.fit(X_train_adv, y_train)

    y_pred = lr.predict(X_test_adv)
    proba_hit = lr.predict_proba(X_test_adv)[:, 1]    # P(class = 1 = Hit)

    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    test_auc = float(roc_auc_score(y_test, proba_hit))

    print(f"  Test Accuracy : {test_accuracy:.4f}")
    print(f"  Test F1-Score : {test_f1:.4f}")
    print(f"  Test ROC AUC  : {test_auc:.4f}")

    # 1. Feature importance
    print("\n" + "=" * 72)
    print("  1) FEATURE IMPORTANCE — signed LR coefficients")
    print("=" * 72)
    print("  Coefficients are signed log-odds shifts per +1 IQR of the feature.")
    coefs = pd.Series(lr.coef_[0], index=X_train_adv.columns)
    _print_top_coefficients(coefs, k=TOP_K)

    # 2. Confidence analysis
    print("\n" + "=" * 72)
    print("  2) CONFIDENCE ANALYSIS — predict_proba on test set")
    print("=" * 72)
    _print_confidence_groups(proba_hit, y_test, test_titles, k=TOP_CONFIDENCE)

    # 3. Plots
    print("\n" + "=" * 72)
    print("  3) PLOTS")
    print("=" * 72)
    _save_confusion_matrix_plot(cm, CM_PNG)
    auc_check = _save_roc_plot(y_test, proba_hit, ROC_PNG)
    # the AUC from the plot should match exactly what we already computed
    assert abs(auc_check - test_auc) < 1e-9, "AUC mismatch between plot and metrics."
    print(f"  Wrote {CM_PNG}")
    print(f"  Wrote {ROC_PNG}")

    print("\nDone. Inspect the borderline recipes for your qualitative culinary write-up.")


if __name__ == "__main__":
    main()
