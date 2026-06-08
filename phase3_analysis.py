"""
Phase 3 — Interpretation & Confidence Analysis
==============================================

A single interpretable LogisticRegression is fit on the *Advanced* matrix
from Phase 1, then dissected three ways:

    1. Feature importance — top 15 positive (push toward Hit) and top 15
       negative (push toward Miss) signed coefficients, with explicit
       flagging of any engineered culinary feature that survives into the
       top lists.

    2. Confidence analysis — `predict_proba` on the held-out test set, with
       the 5 most-confident Hits, 5 most-confident Misses, and 5 most-
       borderline cases (|p − 0.5| smallest). Each row is annotated with
       the true label and whether the prediction was correct, so the
       user can do qualitative culinary inspection of the failures.

    3. Visualizations — two PNGs at ~200 dpi:
           * `lr_confusion_matrix.png` — annotated confusion-matrix heatmap.
           * `lr_roc_curve.png` — ROC curve with AUC.

Why LogisticRegression?
-----------------------
Phase 2 showed it is the best-balanced classifier on this dataset (best
accuracy AND positive Δ from the engineered features), and its coefficients
are directly interpretable on the RobustScaled feature space (one-standard-
IQR change in the feature shifts the log-odds by exactly the coefficient).
That makes it the natural single model for an interpretability chapter.

A note on aligning predictions back to recipe titles
----------------------------------------------------
Phase 0 strips ``title`` from X before the train/test split, so the model
matrix carries no human-readable labels. We re-derive the merged frame here
solely to recover ``title`` aligned by the same RangeIndex Phase 0 uses
internally (``clean_and_binarize`` resets the index to 0..N-1; the split
preserves that index). Re-running the merge is cheap and keeps the upstream
APIs stable.
"""

from __future__ import annotations

from typing import Iterable

import matplotlib
matplotlib.use("Agg")              # save PNGs without a display backend
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

from src.phase0_data_foundation import (
    CulinaryFeatureExtractor,
    clean_and_binarize,
    load_binary_matrix,
    load_recipe_text,
    merge_datasets,
)
from src.phase1_preprocessing import build_preprocessed_datasets


from src._constants import RANDOM_STATE  # single source of truth, see src/_constants.py
TOP_K = 15                          # top-N coefficients per direction
TOP_CONFIDENCE = 5                  # rows per confidence bucket
TITLE_WIDTH = 70                    # truncate titles for tidy printing

# Phase 3's LR plots now share the per-model results subdirectory with the
# LR notebook / CLI script — see src/train_utils.py::model_results_dir.
from src.train_utils import RESULTS_DIR as _RESULTS_DIR
_LR_RESULTS = _RESULTS_DIR / "logistic_regression"
_LR_RESULTS.mkdir(parents=True, exist_ok=True)
CM_PNG  = str(_LR_RESULTS / "confusion_matrix.png")
ROC_PNG = str(_LR_RESULTS / "roc_curve.png")

# The 9 engineered culinary feature names, derived from Phase 0 so we never
# drift if those constants change.
CULINARY_FEATURE_NAMES: tuple = (
    tuple(CulinaryFeatureExtractor.BASELINE_FEATURES)
    + tuple(f"has_{g}" for g in CulinaryFeatureExtractor.KEYWORD_GROUPS)
)


# ---------------------------------------------------------------------------
# Title recovery
# ---------------------------------------------------------------------------
def _recover_titles() -> pd.Series:
    """Recreate the Phase 0 title series aligned to the split's RangeIndex."""
    bin_df = load_binary_matrix()
    txt_df = load_recipe_text()
    merged = merge_datasets(bin_df, txt_df, verbose=False)
    merged, _ = clean_and_binarize(merged, verbose=False)
    return merged["title"].astype(str)


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------
def _truncate(s: str, width: int = TITLE_WIDTH) -> str:
    s = " ".join(s.split())          # collapse whitespace
    return s if len(s) <= width else s[: width - 1] + "…"


def _print_top_coefficients(coefs: pd.Series, k: int) -> None:
    """Print top-k positive and top-k negative LR coefficients."""
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

    # --- Spotlight: where do the engineered culinary features rank? --------
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
    """Print the most-confident Hits/Misses and the most-borderline rows."""
    df = pd.DataFrame({
        "title":    titles.values,
        "true":     y_true.values,
        "p_hit":    proba_hit,
    })
    df["pred"] = (df["p_hit"] >= 0.5).astype(int)
    df["dist_to_05"] = (df["p_hit"] - 0.5).abs()
    df["correct"] = (df["true"] == df["pred"])

    def _emit(label: str, rows: pd.DataFrame) -> None:
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


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _save_confusion_matrix_plot(cm: np.ndarray, path: str) -> None:
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
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

    # Sanity: titles line up 1:1 with the test rows.
    assert len(test_titles) == len(X_test_adv), (
        "Title recovery mis-aligned — indices drifted between phases."
    )

    # --- Train the interpretable model ------------------------------------
    print(f"\nTraining LogisticRegression on Advanced X_train {X_train_adv.shape}...")
    # Solver choice matters here. With 687 features (mostly sparse 0/1 tags,
    # many collinear), `lbfgs` did not converge cleanly even at max_iter=5000
    # — the same test recipe's probability swung from 0.9997 to 0.0000
    # between max_iter=2000 and max_iter=5000, proving the optimum was far
    # from reached. `liblinear` uses coordinate descent (one feature at a
    # time) and converges reliably on this kind of sparse-binary high-dim
    # input. Without convergence, the coefficient-importance and confidence-
    # analysis findings below would not be reproducible.
    lr = LogisticRegression(
        solver="liblinear",
        C=1.0,
        max_iter=5000,
        random_state=RANDOM_STATE,
    )
    lr.fit(X_train_adv, y_train)

    y_pred = lr.predict(X_test_adv)
    proba_hit = lr.predict_proba(X_test_adv)[:, 1]    # P(class = 1 = Hit)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    auc = float(roc_auc_score(y_test, proba_hit))

    print(f"  Test Accuracy : {acc:.4f}")
    print(f"  Test F1-Score : {f1:.4f}")
    print(f"  Test ROC AUC  : {auc:.4f}")

    # --- 1. Feature importance --------------------------------------------
    print("\n" + "=" * 72)
    print("  1) FEATURE IMPORTANCE — signed LR coefficients")
    print("=" * 72)
    print("  Coefficients are signed log-odds shifts per +1 IQR of the feature.")
    coefs = pd.Series(lr.coef_[0], index=X_train_adv.columns)
    _print_top_coefficients(coefs, k=TOP_K)

    # --- 2. Confidence analysis -------------------------------------------
    print("\n" + "=" * 72)
    print("  2) CONFIDENCE ANALYSIS — predict_proba on test set")
    print("=" * 72)
    _print_confidence_groups(proba_hit, y_test, test_titles, k=TOP_CONFIDENCE)

    # --- 3. Plots ---------------------------------------------------------
    print("\n" + "=" * 72)
    print("  3) PLOTS")
    print("=" * 72)
    _save_confusion_matrix_plot(cm, CM_PNG)
    auc_check = _save_roc_plot(y_test, proba_hit, ROC_PNG)
    assert abs(auc_check - auc) < 1e-9, "AUC mismatch between plot and metrics."
    print(f"  Wrote {CM_PNG}")
    print(f"  Wrote {ROC_PNG}")

    print("\nDone. Inspect the borderline recipes for your qualitative culinary write-up.")


if __name__ == "__main__":
    main()
