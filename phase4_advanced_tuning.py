"""
Phase 4 — Advanced Tuning & Top Feature Extraction
====================================================

Two analyses against the converged LogisticRegression from Phase 3:

    1. Decision-threshold sweep.
       For thresholds t ∈ {0.30, 0.35, …, 0.70} we report F1, accuracy,
       FP-rate, FN-rate, and the Balance Ratio (= FP-rate / FN-rate). The
       default t=0.5 leaves the model over-predicting "Hit" (Phase 2 / 3
       showed FP-rate ≈ 0.48 vs FN-rate ≈ 0.32). Raising t pushes the
       balance ratio toward 1.0 at some F1 cost — this table makes that
       trade-off explicit so a single threshold can be chosen for the
       report.

       Balance Ratio reading:
           >1   →  model favours "Hit" (more false alarms than misses)
           <1   →  model favours "Miss"
           ≈1   →  symmetric errors

       The "most balanced" row is chosen by min |log(ratio)|, so a ratio
       of 2.0 and 0.5 are treated as equally off — symmetry should be
       judged on the log scale, not the raw ratio.

    2. The "Recipe for Success" / "Recipe for Disaster" feature list.
       Top 20 positive (strongest Hit indicators) and top 20 negative
       (strongest Miss indicators) signed LR coefficients across all 687
       features. Any engineered culinary feature that cracks either list
       is called out explicitly.

The model is intentionally NOT retuned (C, regularization, solver are all
unchanged from Phase 3) — Phase 4 is *post-hoc* analysis on a fixed model,
not a new training run. Reusing the same fit lets every coefficient and
probability here line up exactly with what was reported in Phase 3.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.phase0_data_foundation import CulinaryFeatureExtractor
from src.phase1_preprocessing import build_preprocessed_datasets


from src._constants import RANDOM_STATE  # single source of truth, see src/_constants.py
THRESHOLDS = np.linspace(0.30, 0.70, 9)   # 0.30, 0.35, …, 0.70
TOP_K = 20
DEFAULT_THRESHOLD = 0.50                  # baseline for delta comparisons

CULINARY_FEATURE_NAMES: tuple = (
    tuple(CulinaryFeatureExtractor.BASELINE_FEATURES)
    + tuple(f"has_{g}" for g in CulinaryFeatureExtractor.KEYWORD_GROUPS)
)


# ---------------------------------------------------------------------------
# Threshold evaluation
# ---------------------------------------------------------------------------
def _evaluate_at_threshold(
    y_true: pd.Series,
    proba_hit: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Apply a decision threshold and return the metric bundle we report."""
    y_pred = (proba_hit >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])

    n_true_miss = tn + fp
    n_true_hit = fn + tp
    fp_rate = fp / n_true_miss if n_true_miss else 0.0   # P(pred=Hit | true=Miss)
    fn_rate = fn / n_true_hit if n_true_hit else 0.0     # P(pred=Miss | true=Hit)
    balance = (fp_rate / fn_rate) if fn_rate > 0 else float("inf")

    return {
        "threshold":      float(threshold),
        "f1":             f1_score(y_true, y_pred),
        "accuracy":       accuracy_score(y_true, y_pred),
        "fp_rate":        fp_rate,
        "fn_rate":        fn_rate,
        "balance_ratio":  balance,
        "pred_hits":      int(y_pred.sum()),
        "n_total":        len(y_pred),
    }


def _print_threshold_table(rows: List[Dict[str, Any]]) -> None:
    header = (
        f"  {'thresh':>6}  {'F1':>7}  {'Acc':>7}  "
        f"{'FP rate':>8}  {'FN rate':>8}  {'Bal ratio':>10}  "
        f"{'Predicted Hits':>16}"
    )
    print(header)
    print(f"  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*16}")
    for r in rows:
        bal = r["balance_ratio"]
        bal_str = f"{bal:10.3f}" if np.isfinite(bal) else f"{'inf':>10}"
        hits_pct = 100.0 * r["pred_hits"] / r["n_total"]
        hits_str = f"{r['pred_hits']:>5d} ({hits_pct:5.1f}%)"
        print(
            f"  {r['threshold']:>6.2f}  {r['f1']:>7.4f}  {r['accuracy']:>7.4f}  "
            f"{r['fp_rate']:>8.4f}  {r['fn_rate']:>8.4f}  {bal_str}  "
            f"{hits_str:>16}"
        )


def _log_distance_to_balanced(row: Dict[str, Any]) -> float:
    """Distance from balance_ratio=1 on the log scale (so 2.0 and 0.5 are equal)."""
    bal = row["balance_ratio"]
    if not np.isfinite(bal) or bal <= 0:
        return float("inf")
    return abs(np.log(bal))


def _print_threshold_summary(rows: List[Dict[str, Any]]) -> None:
    """Highlight (a) max-F1 row, (b) most-balanced row, and the trade-off vs t=0.5."""
    default_row = next(
        (r for r in rows if abs(r["threshold"] - DEFAULT_THRESHOLD) < 1e-9),
        None,
    )
    best_f1 = max(rows, key=lambda r: r["f1"])
    most_balanced = min(rows, key=_log_distance_to_balanced)

    def _summarize(label: str, r: Dict[str, Any]) -> None:
        bal = r["balance_ratio"]
        bal_str = f"{bal:.3f}" if np.isfinite(bal) else "inf"
        print(
            f"  {label:<28} threshold={r['threshold']:.2f}   "
            f"F1={r['f1']:.4f}   "
            f"FP/FN rates={r['fp_rate']:.4f}/{r['fn_rate']:.4f}   "
            f"balance={bal_str}"
        )

    print()
    if default_row is not None:
        _summarize("Default (t=0.50):", default_row)
    _summarize("Best F1:", best_f1)
    _summarize("Most balanced (|log| min):", most_balanced)

    if default_row is not None:
        d_f1 = most_balanced["f1"] - default_row["f1"]
        print(
            f"\n  Trade-off at the most-balanced threshold vs the default: "
            f"ΔF1 = {d_f1:+.4f}  "
            f"(balance shifts from {default_row['balance_ratio']:.3f} "
            f"to {most_balanced['balance_ratio']:.3f})"
        )


# ---------------------------------------------------------------------------
# Top-coefficient extraction
# ---------------------------------------------------------------------------
def _print_top_coefficients(coefs: pd.Series, k: int) -> None:
    top_pos = coefs.sort_values(ascending=False).head(k)
    top_neg = coefs.sort_values(ascending=True).head(k)

    def _print_block(title: str, series: pd.Series) -> None:
        print(f"\n  {title}")
        print(f"  {'rank':>4}  {'coef':>10}   {'feature'}")
        print(f"  {'-'*4}  {'-'*10}   {'-'*55}")
        for rank, (name, val) in enumerate(series.items(), start=1):
            print(f"  {rank:>4}  {val:+10.4f}   {name}")

    _print_block(
        f"Top {k} POSITIVE — 'Recipe for Success' (strongest Hit indicators)",
        top_pos,
    )
    _print_block(
        f"Top {k} NEGATIVE — 'Recipe for Disaster' (strongest Miss indicators)",
        top_neg,
    )

    # Spotlight: any engineered culinary feature in either top-K list?
    eng_in_pos = [n for n in top_pos.index if n in CULINARY_FEATURE_NAMES]
    eng_in_neg = [n for n in top_neg.index if n in CULINARY_FEATURE_NAMES]
    print()
    if eng_in_pos or eng_in_neg:
        print(f"  >> Engineered culinary features in the top {k}:")
        for n in eng_in_pos:
            print(f"     POSITIVE: {n} ({top_pos[n]:+.4f})")
        for n in eng_in_neg:
            print(f"     NEGATIVE: {n} ({top_neg[n]:+.4f})")
    else:
        print(
            f"  >> No engineered culinary features made the top {k} lists in "
            "either direction. The CSV's binary tag matrix dominates the "
            "interpretable signal."
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 72)
    print("  PHASE 4 — ADVANCED TUNING & TOP FEATURE EXTRACTION")
    print("=" * 72)

    print("\nLoading preprocessed Advanced datasets (Phase 1)...")
    (
        _, _,
        X_train_adv, X_test_adv,
        y_train, y_test,
    ) = build_preprocessed_datasets(verbose=False)
    print(f"  X_train shape: {X_train_adv.shape}")
    print(f"  X_test  shape: {X_test_adv.shape}")

    print("\nTraining LogisticRegression (solver='liblinear', same as Phase 3)...")
    lr = LogisticRegression(
        solver="liblinear",
        C=1.0,
        max_iter=5000,
        random_state=RANDOM_STATE,
    )
    lr.fit(X_train_adv, y_train)
    proba_hit = lr.predict_proba(X_test_adv)[:, 1]

    # --- 1. Threshold tuning ---------------------------------------------
    print("\n" + "=" * 72)
    print("  1) THRESHOLD TUNING — addressing FP/FN asymmetry")
    print("=" * 72)
    print(
        "  Balance Ratio = FP_rate / FN_rate.\n"
        "    >1 → model favours 'Hit' (more false alarms than misses)\n"
        "    <1 → model favours 'Miss'\n"
        "    ≈1 → symmetric errors\n"
    )

    rows = [_evaluate_at_threshold(y_test, proba_hit, t) for t in THRESHOLDS]
    _print_threshold_table(rows)
    _print_threshold_summary(rows)

    # --- 2. Top coefficients ---------------------------------------------
    print("\n" + "=" * 72)
    print(f"  2) ULTIMATE FEATURE RANKING — top {TOP_K} per direction (n_features={X_train_adv.shape[1]})")
    print("=" * 72)
    print("  Coefficients are signed log-odds shifts per +1 IQR (or per +1 for "
          "0/1 binary features) of the feature.")
    coefs = pd.Series(lr.coef_[0], index=X_train_adv.columns)
    _print_top_coefficients(coefs, k=TOP_K)


if __name__ == "__main__":
    main()
