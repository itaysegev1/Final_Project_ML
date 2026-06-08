"""
Phase 1 — Preprocessing Pipeline & Baseline Preparation
========================================================

This module turns the raw feature frame produced by Phase 0 into two
model-ready matrices, fit strictly on the training split:

    * Baseline matrix : robust-scaled nutrition + raw binary tags only.
                        (No JSON-derived text features. This is the control
                        condition that the engineered features must beat.)
    * Advanced matrix : robust-scaled nutrition + raw binary tags + the 9
                        features emitted by `CulinaryFeatureExtractor` (with
                        the 3 numeric culinary features also robust-scaled,
                        the 6 binary ones passed through).

We use `RobustScaler` (median-centered, IQR-scaled) rather than
`StandardScaler` because the nutrition columns contain extreme outliers (a
handful of clearly-bogus calorie/sodium values from the original web scrape)
that inflate the sample standard deviation enough to crush the rest of the
distribution. The Phase-0 sanity panel showed test-set std ≈ 0.03 for `fat`
after standard-scaling — i.e. virtually all of the variance was being
absorbed by a tiny number of outliers, which would in turn destroy the
distance metric for KNN and similar models.

Both matrices share the same train/test row partition produced in Phase 0,
which guarantees that an A/B comparison in Phase 2 is apples-to-apples.

Leakage discipline
------------------
Every transformer (StandardScaler, CulinaryFeatureExtractor) is `fit` on
``X_train`` *only* and then used to ``transform`` ``X_test``. No statistic
derived from the test split ever influences the training pipeline.

Column routing
--------------
The Phase 0 feature frame X has 680 columns of three kinds:
    1. Continuous nutrition (4 cols):   calories, protein, fat, sodium
    2. Raw recipe text (2 cols):        directions, ingredients
    3. Binary recipe tags (674 cols):   everything else from the CSV
``title`` is already removed in Phase 0; ``date``/``desc``/``categories`` are
never pulled in from the JSON in the first place. Note that the column named
``date`` that *does* survive is the CSV's binary "date" tag (the dried
fruit) — a legitimate feature, not the JSON publication date. The defensive
drop below removes any of those names if they ever reappear, but in the
current pipeline it is a no-op (and is reported as such).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.phase0_data_foundation import (
    CulinaryFeatureExtractor,
    build_dataset,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUMERIC_COLUMNS: Tuple[str, ...] = ("calories", "protein", "fat", "sodium")
TEXT_COLUMNS: Tuple[str, ...] = ("directions", "ingredients")

# Names the spec asks us to drop. In the current Phase 0 output none of these
# survive — but we drop defensively in case the upstream changes. The CSV's
# "date" binary tag (the fruit) is kept; see the column-classification logic.
DROP_IF_PRESENT: Tuple[str, ...] = ("title", "desc", "categories")

# The 3 numeric features inside CulinaryFeatureExtractor that need scaling.
# The 6 `has_*` binary features it also emits are passed through untouched.
CULINARY_NUMERIC: Tuple[str, ...] = CulinaryFeatureExtractor.BASELINE_FEATURES


# ---------------------------------------------------------------------------
# Column classification
# ---------------------------------------------------------------------------
def classify_columns(X: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Partition the columns of X into (numeric, text, binary_tags, dropped).

    The partition is mutually exclusive and collectively exhaustive over
    ``X.columns`` — verified at the bottom of this function.
    """
    cols = list(X.columns)

    numeric = [c for c in NUMERIC_COLUMNS if c in cols]
    text = [c for c in TEXT_COLUMNS if c in cols]
    to_drop = [c for c in DROP_IF_PRESENT if c in cols]
    binary_tags = [
        c for c in cols
        if c not in numeric and c not in text and c not in to_drop
    ]

    # Sanity: every input column is accounted for, and only once.
    accounted = numeric + text + binary_tags + to_drop
    assert sorted(accounted) == sorted(cols), (
        "Column classification missed or double-counted some columns."
    )

    # Sanity: nutrition fields really are numeric.
    for col in numeric:
        if not pd.api.types.is_numeric_dtype(X[col]):
            raise TypeError(
                f"Column '{col}' should be numeric but has dtype {X[col].dtype}."
            )

    return numeric, text, binary_tags, to_drop


# ---------------------------------------------------------------------------
# Pipeline factories
# ---------------------------------------------------------------------------
def _numeric_pipeline() -> Pipeline:
    """Median-impute, then robust-scale.

    Median imputation is the natural complement to RobustScaler: the imputer
    fills missing values with the train median, and RobustScaler then centers
    the column on that same train median — so any imputed value lands at
    exactly 0 in the scaled feature space (the neutral "no information"
    position). This pairing avoids the bias mean-impute + StandardScaler
    would inject when ~21% of nutrition values are missing from the CSV.
    """
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
        ]
    )


def build_baseline_preprocessor(
    numeric_cols: List[str],
    binary_tag_cols: List[str],
) -> ColumnTransformer:
    """Baseline preprocessor: scaled nutrition + raw binary tags (no text)."""
    return ColumnTransformer(
        transformers=[
            ("numeric", _numeric_pipeline(), numeric_cols),
            ("tags", "passthrough", binary_tag_cols),
        ],
        remainder="drop",   # explicitly drops directions/ingredients/etc.
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def _build_culinary_pipeline() -> Pipeline:
    """Extract 9 culinary features, then scale only the 3 numeric ones.

    The inner ColumnTransformer relies on the fact that
    ``CulinaryFeatureExtractor.transform`` returns a DataFrame with named
    columns, so we can select ``CULINARY_NUMERIC`` by name and passthrough the
    6 binary ``has_*`` columns.
    """
    return Pipeline(
        steps=[
            ("extract", CulinaryFeatureExtractor()),
            (
                "scale_numeric",
                ColumnTransformer(
                    transformers=[
                        ("scaled", RobustScaler(), list(CULINARY_NUMERIC)),
                    ],
                    remainder="passthrough",
                    verbose_feature_names_out=False,
                ),
            ),
        ]
    )


def build_advanced_preprocessor(
    numeric_cols: List[str],
    text_cols: List[str],
    binary_tag_cols: List[str],
) -> ColumnTransformer:
    """Advanced preprocessor: baseline + culinary features (scaled internally)."""
    return ColumnTransformer(
        transformers=[
            ("numeric", _numeric_pipeline(), numeric_cols),
            ("tags", "passthrough", binary_tag_cols),
            ("culinary", _build_culinary_pipeline(), text_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


# ---------------------------------------------------------------------------
# Fit / transform orchestration
# ---------------------------------------------------------------------------
def fit_transform_pair(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit on train only, then transform both splits — the leakage-free idiom."""
    X_train_out = preprocessor.fit_transform(X_train)
    X_test_out = preprocessor.transform(X_test)
    return X_train_out, X_test_out


def build_preprocessed_datasets(
    verbose: bool = True,
):
    """End-to-end Phase 1 pipeline.

    Returns
    -------
    (X_train_baseline, X_test_baseline,
     X_train_advanced, X_test_advanced,
     y_train,          y_test)
    """
    X_train, X_test, y_train, y_test = build_dataset(verbose=verbose)

    numeric, text, binary_tags, to_drop = classify_columns(X_train)

    if verbose:
        print(
            f"[columns] numeric={len(numeric)} | text={len(text)} | "
            f"binary_tags={len(binary_tags)} | dropped={len(to_drop)} "
            f"{to_drop if to_drop else '(none — defensive drop list is a no-op here)'}"
        )

    # --- Baseline ----------------------------------------------------------
    baseline = build_baseline_preprocessor(numeric, binary_tags)
    X_train_baseline, X_test_baseline = fit_transform_pair(baseline, X_train, X_test)

    # --- Advanced ----------------------------------------------------------
    advanced = build_advanced_preprocessor(numeric, text, binary_tags)
    X_train_advanced, X_test_advanced = fit_transform_pair(advanced, X_train, X_test)

    return (
        X_train_baseline, X_test_baseline,
        X_train_advanced, X_test_advanced,
        y_train, y_test,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    (
        X_train_baseline, X_test_baseline,
        X_train_advanced, X_test_advanced,
        y_train, y_test,
    ) = build_preprocessed_datasets(verbose=True)

    print("\n" + "=" * 60)
    print("PHASE 1 — PREPROCESSED MATRICES READY")
    print("=" * 60)

    print("\nBaseline (no engineered text features):")
    print(f"  X_train_baseline shape : {X_train_baseline.shape}")
    print(f"  X_test_baseline  shape : {X_test_baseline.shape}")

    print("\nAdvanced (baseline + 9 culinary features):")
    print(f"  X_train_advanced shape : {X_train_advanced.shape}")
    print(f"  X_test_advanced  shape : {X_test_advanced.shape}")

    delta = X_train_advanced.shape[1] - X_train_baseline.shape[1]
    print(f"\nDelta: Advanced adds {delta} engineered columns over Baseline.")
    print(f"  (expected = 9 from CulinaryFeatureExtractor: "
          f"{len(CULINARY_NUMERIC)} numeric + "
          f"{len(CulinaryFeatureExtractor.KEYWORD_GROUPS)} binary)")

    # Sanity check: nutrition is properly RobustScaler-scaled in BOTH matrices.
    # RobustScaler centers on the train median and scales by the train IQR,
    # so on X_train: median ≈ 0 and IQR ≈ 1. (Mean/std are NOT guaranteed to
    # be 0/1 — that's a StandardScaler invariant, not a RobustScaler one.)
    def _robust_stats(df: pd.DataFrame) -> pd.DataFrame:
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        return pd.DataFrame({
            "median": df.median().round(3),
            "iqr":    (q3 - q1).round(3),
        })

    nutr_train_baseline = X_train_baseline[list(NUMERIC_COLUMNS)]
    print("\nSanity — RobustScaler on X_train_baseline (median ~0, IQR ~1):")
    print(_robust_stats(nutr_train_baseline).to_string())

    # Sanity check: the test split is NOT median-0 / IQR-1 — it was scaled
    # using TRAIN statistics, so its own median/IQR drift slightly.
    nutr_test_baseline = X_test_baseline[list(NUMERIC_COLUMNS)]
    print("\nSanity — same columns on X_test_baseline "
          "(medians/IQRs need NOT be 0/1; scaler was fit on train only):")
    print(_robust_stats(nutr_test_baseline).to_string())

    # y splits are unchanged from Phase 0 — just report them for completeness.
    print(f"\ny_train: {y_train.shape}  (hit rate {y_train.mean():.3f})")
    print(f"y_test : {y_test.shape}  (hit rate {y_test.mean():.3f})")


if __name__ == "__main__":
    main()
