"""
This module is Phase 1 - it builds the preprocessing pipeline on top of Phase 0
and makes the 2 model ready matrices, Baseline and Advanced.
The Baseline is just the robust scaled nutrition columns + the raw binary tags
(no text features) - this is our control to beat.
The Advanced adds on top of that the 9 culinary features from the
CulinaryFeatureExtractor, with the 3 numeric ones also robust scaled and the
6 binary has_* ones passed through.

We use RobustScaler instead of StandardScaler, because the nutrition columns
have some extreme outliers (a few crazy calorie/sodium values from the
original web scrape) that blow up the std and crush everything else. The
Phase 0 sanity panel showed test std around 0.03 for fat after standard
scaling which is bad for KNN distances.

Both matrices share the same train/test split from Phase 0 so the comparison
in Phase 2 stays apples to apples.

Leakage discipline - every transformer is fit on X_train only and then we
just transform X_test. No test statistic ever leaks into the training pipe.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.data_foundation import (
    CulinaryFeatureExtractor,
    build_dataset,
    make_culinary_extractor,
)


# Configuration
NUMERIC_COLUMNS: Tuple[str, ...] = ("calories", "protein", "fat", "sodium")
TEXT_COLUMNS: Tuple[str, ...] = ("directions", "ingredients")

# these are the names the spec asks us to drop. in the current Phase 0 output
# none of them survive but we drop defensively just in case. note the CSV's
# "date" binary tag (the fruit) is kept - see the column classifier below.
DROP_IF_PRESENT: Tuple[str, ...] = ("title", "desc", "categories")

# the 3 numeric features inside CulinaryFeatureExtractor that we need to scale.
# the 6 has_* binary features it also emits are passed through untouched.
CULINARY_NUMERIC: Tuple[str, ...] = CulinaryFeatureExtractor.BASELINE_FEATURES


# Column classification

def classify_columns(X: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    This function splits the columns of X into 4 groups: numeric, text, binary
    tags and dropped. The partition is mutually exclusive and covers all the
    columns - we verify it at the end.
    :param X: the feature frame coming out of Phase 0
    :return: a tuple (numeric, text, binary_tags, to_drop) of column name lists
    """
    cols = list(X.columns)

    numeric = [c for c in NUMERIC_COLUMNS if c in cols]
    text = [c for c in TEXT_COLUMNS if c in cols]
    to_drop = [c for c in DROP_IF_PRESENT if c in cols]
    binary_tags = [
        c for c in cols
        if c not in numeric and c not in text and c not in to_drop
    ]

    # sanity check - every input column is accounted for and only once
    accounted = numeric + text + binary_tags + to_drop
    assert sorted(accounted) == sorted(cols), (
        "Column classification missed or double-counted some columns."
    )

    # sanity check - the nutrition fields really are numeric
    for col in numeric:
        if not pd.api.types.is_numeric_dtype(X[col]):
            raise TypeError(
                f"Column '{col}' should be numeric but has dtype {X[col].dtype}."
            )

    return numeric, text, binary_tags, to_drop


# Pipeline factories

def _numeric_pipeline() -> Pipeline:
    """
    Here we build the small pipeline for the nutrition columns - first we
    median impute and then robust scale. The pairing is on purpose: the
    imputer fills missing values with the train median, and RobustScaler
    then centers on that same train median, so any imputed value lands
    exactly at 0 in the scaled space (the neutral "no information" spot).
    This avoids the bias that mean impute + StandardScaler would inject
    when about 21% of the nutrition values are missing from the CSV.
    :return: the small numeric Pipeline
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
    """
    This function builds the Baseline preprocessor - scaled nutrition + the
    raw binary tags, no text features at all.
    :param numeric_cols: the names of the numeric (nutrition) columns
    :param binary_tag_cols: the names of the binary tag columns
    :return: the Baseline ColumnTransformer
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", _numeric_pipeline(), numeric_cols),
            ("tags", "passthrough", binary_tag_cols),
        ],
        remainder="drop",   # this explicitly drops directions/ingredients/etc
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def _build_culinary_pipeline() -> Pipeline:
    """
    Here we build the culinary sub pipeline - first we extract the 9 culinary
    features and then we scale only the 3 numeric ones. The inner
    ColumnTransformer counts on the fact that CulinaryFeatureExtractor.transform
    returns a DataFrame with named columns, so we can pick CULINARY_NUMERIC by
    name and passthrough the 6 binary has_* columns.

    IMPORTANT: the extractor MUST come from make_culinary_extractor() so the
    curated CULINARY_KEYWORDS are actually injected. A bare
    CulinaryFeatureExtractor() has empty keyword groups and silently emits
    all-zero has_* columns — that exact bug shipped in the first version of
    this file, and _assert_no_dead_binary_features below now guards against
    any regression of it.
    :return: the culinary Pipeline
    """
    return Pipeline(
        steps=[
            ("extract", make_culinary_extractor()),
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
    """
    This function builds the Advanced preprocessor - same as the Baseline
    plus the culinary features (which are scaled internally).
    :param numeric_cols: the names of the numeric (nutrition) columns
    :param text_cols: the names of the raw text columns
    :param binary_tag_cols: the names of the binary tag columns
    :return: the Advanced ColumnTransformer
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", _numeric_pipeline(), numeric_cols),
            ("tags", "passthrough", binary_tag_cols),
            ("culinary", _build_culinary_pipeline(), text_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


# Fit / transform orchestration

def _assert_no_dead_binary_features(X_train_out: pd.DataFrame) -> None:
    """
    This guard fails loud if any engineered has_* column is all-zeros on the
    TRAIN split. An all-zero binary feature means the keyword group behind it
    never fired on ~14.5k recipes — in practice that means the keywords were
    never injected into the extractor (the bug the first version of this
    pipeline shipped with). Better to crash here than to train and report on
    dead features.
    :param X_train_out: the transformed train matrix to check
    """
    has_cols = [c for c in X_train_out.columns if c.startswith("has_")]
    dead = [c for c in has_cols if int(X_train_out[c].sum()) == 0]
    if dead:
        raise RuntimeError(
            f"Engineered binary features are all-zero on the train split: {dead}. "
            "The CulinaryFeatureExtractor was probably built without its keyword "
            "groups — use make_culinary_extractor() (see src/data_foundation.py)."
        )


def fit_transform_pair(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Here we do the leakage-free idiom: we fit only on the train split and then
    transform both splits.
    :param preprocessor: the ColumnTransformer to fit
    :param X_train: the train features
    :param X_test: the test features
    :return: the (X_train_out, X_test_out) pair after transformation
    """
    X_train_out = preprocessor.fit_transform(X_train)
    X_test_out = preprocessor.transform(X_test)
    return X_train_out, X_test_out


def build_preprocessed_datasets(
    verbose: bool = True,
):
    """
    This is the end-to-end Phase 1 pipeline - it loads the Phase 0 dataset,
    classifies the columns and builds both the Baseline and the Advanced
    matrices in one shot.
    :param verbose: if True we print a small column report
    :return: (X_train_baseline, X_test_baseline,
              X_train_advanced, X_test_advanced,
              y_train, y_test)
    """
    X_train, X_test, y_train, y_test = build_dataset(verbose=verbose)

    numeric, text, binary_tags, to_drop = classify_columns(X_train)

    if verbose:
        print(
            f"[columns] numeric={len(numeric)} | text={len(text)} | "
            f"binary_tags={len(binary_tags)} | dropped={len(to_drop)} "
            f"{to_drop if to_drop else '(none — defensive drop list is a no-op here)'}"
        )

    # Baseline
    baseline = build_baseline_preprocessor(numeric, binary_tags)
    X_train_baseline, X_test_baseline = fit_transform_pair(baseline, X_train, X_test)

    # Advanced
    advanced = build_advanced_preprocessor(numeric, text, binary_tags)
    X_train_advanced, X_test_advanced = fit_transform_pair(advanced, X_train, X_test)

    # guard: the engineered binary features must actually fire on real recipes
    _assert_no_dead_binary_features(X_train_advanced)

    if verbose:
        has_cols = [c for c in X_train_advanced.columns if c.startswith("has_")]
        rates = {c: f"{100 * X_train_advanced[c].mean():.1f}%" for c in has_cols}
        print(f"[culinary] keyword-group activation on train: {rates}")

    return (
        X_train_baseline, X_test_baseline,
        X_train_advanced, X_test_advanced,
        y_train, y_test,
    )


# Entry point

def main() -> None:
    """
    Here we run Phase 1 end to end and print a small sanity panel so we can
    eyeball the shapes and the RobustScaler stats.
    """
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

    # sanity check: nutrition is robust-scaled in BOTH matrices.
    # RobustScaler centers on the train median and scales by the train IQR
    # so on X_train we expect median ~0 and IQR ~1. (mean/std are NOT
    # guaranteed to be 0/1 - that is a StandardScaler invariant, not a
    # RobustScaler one.)
    def _robust_stats(df: pd.DataFrame) -> pd.DataFrame:
        """
        Small helper - returns the median and IQR for each column of df.
        :param df: the frame to summarize
        :return: a small DataFrame with median and iqr columns
        """
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        return pd.DataFrame({
            "median": df.median().round(3),
            "iqr":    (q3 - q1).round(3),
        })

    nutr_train_baseline = X_train_baseline[list(NUMERIC_COLUMNS)]
    print("\nSanity — RobustScaler on X_train_baseline (median ~0, IQR ~1):")
    print(_robust_stats(nutr_train_baseline).to_string())

    # sanity check: the test split is NOT median-0 / IQR-1 because it was
    # scaled with the TRAIN statistics, so its own median/IQR drift a bit
    nutr_test_baseline = X_test_baseline[list(NUMERIC_COLUMNS)]
    print("\nSanity — same columns on X_test_baseline "
          "(medians/IQRs need NOT be 0/1; scaler was fit on train only):")
    print(_robust_stats(nutr_test_baseline).to_string())

    # the y splits are unchanged from Phase 0 - just printing them for completeness
    print(f"\ny_train: {y_train.shape}  (hit rate {y_train.mean():.3f})")
    print(f"y_test : {y_test.shape}  (hit rate {y_test.mean():.3f})")


if __name__ == "__main__":
    main()
