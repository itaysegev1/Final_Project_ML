"""
Phase 0 — Data Foundation & Granular Feature Engineering Setup
==============================================================

Epicurious "Recipes with Rating and Nutrition" — binary classification:
    "Hit"  (1) : rating >= 4.0
    "Miss" (0) : rating <  4.0

This module is intentionally limited to the *data foundation*:

    1. Safe, reproducible loading of `epi_r.csv` (sparse binary tag matrix +
       nutrition + rating) and `full_format_recipes.json` (raw `directions`
       and `ingredients` text).
    2. A robust merge of the two representations.
    3. Target binarization.
    4. A leakage-free 80/20 train/test split (performed *before* any feature
       engineering).
    5. The `CulinaryFeatureExtractor` transformer *skeleton* — six empty
       keyword lists are defined in `__init__` for the analyst to populate
       later with domain knowledge. No models are fitted here.

----------------------------------------------------------------------------
A note on the merge strategy (why we do not merge on row index or title alone)
----------------------------------------------------------------------------
Two facts about these files drive the design:

  * The CSV and JSON are stored in **different row orders** — a positional
    (index-based) merge aligns < 3% of rows correctly, so it is unusable.
  * `title` alone is **not unique**: ~2,300 titles repeat in each file, so a
    title-only join produces a many-to-many cartesian explosion.

Both files independently carry the same five numeric fields
(`rating, calories, protein, fat, sodium`). Combining the (normalized) title
with these five numbers yields a **composite key** that is unique for ~18.2k
recipes and matches the two files cleanly 1:1. Genuine exact-duplicate
recipes (identical title *and* nutrition *and* rating) are collapsed with
`keep="first"`, since they carry no additional signal.

----------------------------------------------------------------------------
A note on the provided helper scripts (`utils.py`, `recipe.py`)
----------------------------------------------------------------------------
Both were reviewed. `recipe.py` is a BeautifulSoup/urllib web-scraper used to
*build* the dataset from live HTML — irrelevant to loading already-saved files
(and importing it would require `bs4`). `utils.py::sublists_to_binaries`
one-hot-encodes a list column; the CSV already ships those binary tag columns,
so it is not needed for the merge. The JSON is already in clean,
record-oriented dict form, so `json.load` is the most efficient parse path and
is used directly.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Resolve data paths from the project root (parent of `src/`), so that the
# same import works whether Python is invoked from the project root or from
# a sub-directory like `notebooks/`.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR: str = os.path.join(_PROJECT_ROOT, "data")
CSV_PATH: str = os.path.join(DATA_DIR, "epi_r.csv")
JSON_PATH: str = os.path.join(DATA_DIR, "full_format_recipes.json")

RATING_THRESHOLD: float = 4.0          # >= 4.0 -> "Hit" (1)
TEST_SIZE: float = 0.20                # 80 / 20 split
from src._constants import RANDOM_STATE  # single source of truth, see src/_constants.py

# Fields present in BOTH files; (normalized title + these) form the merge key.
SHARED_NUMERIC: Tuple[str, ...] = ("rating", "calories", "protein", "fat", "sodium")

# Raw-text columns the CSV lacks and that we pull in from the JSON.
JSON_TEXT_COLUMNS: Tuple[str, ...] = ("directions", "ingredients")

# Columns excluded from the model matrix X. We deliberately keep this minimal:
#   - title  -> identifier
#   - rating -> target leakage (y is derived from it)
# NOTE: we intentionally do NOT list "date" here. The CSV ships a binary tag
# column literally named "date" (the dried fruit), which is a legitimate
# feature. The JSON's unrelated publication "date" is simply never pulled in
# (see `merge_datasets`), avoiding both the name collision and a useless drop.
NON_FEATURE_COLUMNS: Tuple[str, ...] = ("title", "rating")

_MERGE_KEY: str = "_merge_key"


# ===========================================================================
# 1. Data loading
# ===========================================================================
def load_binary_matrix(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """Load the sparse binary tag / nutrition / rating matrix (`epi_r.csv`).

    The CSV header contains some mojibake (e.g. a garbled ``bon appétit``
    duplicate); this affects only a handful of tag *column names* and is
    harmless for our purposes. We try UTF-8 first and fall back to Latin-1 so
    the read never crashes on a stray byte.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find CSV at '{csv_path}'.")
    try:
        return pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, encoding="latin-1")


def load_recipe_text(json_path: str = JSON_PATH) -> pd.DataFrame:
    """Load the raw-text recipe records (`full_format_recipes.json`).

    The file is a JSON array of dicts; `json.load` -> `DataFrame` keeps
    `directions` and `ingredients` as native Python lists, which the
    transformer below consumes directly.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Could not find JSON at '{json_path}'.")
    with open(json_path, "r", encoding="utf-8") as fh:
        records = json.load(fh)
    return pd.DataFrame(records)


# ===========================================================================
# 2. Merge
# ===========================================================================
def _build_merge_key(df: pd.DataFrame) -> pd.Series:
    """Construct the composite merge key: normalized title + shared numerics.

    Numbers are coerced and rounded to 3 decimals so the two files' float
    representations line up exactly.
    """
    def _fmt_num(series: pd.Series) -> pd.Series:
        # Deterministic, fixed-precision string so both files format identically.
        numeric = pd.to_numeric(series, errors="coerce").round(3)
        return numeric.map(lambda v: "nan" if pd.isna(v) else f"{v:.3f}")

    key = df["title"].fillna("").astype(str).str.strip().str.lower()
    for col in SHARED_NUMERIC:
        key = key + "|" + _fmt_num(df[col])
    return key


def merge_datasets(
    binary_df: pd.DataFrame,
    text_df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """Merge the binary/nutrition matrix with the raw recipe text.

    Strategy: de-duplicate each side on the composite key (`keep="first"`),
    then perform a 1:1 inner join. Only the text columns the CSV lacks are
    pulled in from the JSON, so no `_x`/`_y` suffix collisions occur.
    """
    binary_df = binary_df.copy()
    text_df = text_df.copy()

    binary_df[_MERGE_KEY] = _build_merge_key(binary_df)
    text_df[_MERGE_KEY] = _build_merge_key(text_df)

    n_csv_dupes = int(binary_df[_MERGE_KEY].duplicated().sum())
    n_json_dupes = int(text_df[_MERGE_KEY].duplicated().sum())
    binary_df = binary_df.drop_duplicates(subset=_MERGE_KEY, keep="first")
    text_df = text_df.drop_duplicates(subset=_MERGE_KEY, keep="first")

    # Pull ONLY the raw-text columns the CSV lacks. Anything else the JSON
    # offers (categories/date/desc) is either redundant with the CSV's binary
    # tags or unused here; pulling it risks name collisions (e.g. the CSV's
    # "date" fruit tag vs. the JSON's publication "date").
    text_cols_to_pull = [c for c in JSON_TEXT_COLUMNS if c in text_df.columns]
    missing = set(JSON_TEXT_COLUMNS) - set(text_cols_to_pull)
    if missing:
        raise KeyError(f"JSON is missing expected text columns: {sorted(missing)}")
    text_subset = text_df[[_MERGE_KEY, *text_cols_to_pull]]

    merged = binary_df.merge(text_subset, on=_MERGE_KEY, how="inner", validate="1:1")
    merged = merged.drop(columns=_MERGE_KEY)

    if verbose:
        print(
            f"[merge] CSV rows={len(binary_df) + n_csv_dupes} "
            f"(dropped {n_csv_dupes} dup keys) | "
            f"JSON rows={len(text_df) + n_json_dupes} "
            f"(dropped {n_json_dupes} dup keys) -> merged={len(merged)}"
        )
    return merged


# ===========================================================================
# 3. Target + cleaning + split
# ===========================================================================
def clean_and_binarize(df: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """Drop rows missing `rating` or `directions`, then build the target `y`.

    A `directions` value counts as missing if it is null *or* an empty list.
    """
    n_before = len(df)
    has_rating = df["rating"].notna()
    has_directions = df["directions"].apply(
        lambda d: isinstance(d, (list, tuple, np.ndarray)) and len(d) > 0
    )
    df = df.loc[has_rating & has_directions].reset_index(drop=True)

    y = (df["rating"] >= RATING_THRESHOLD).astype(int)
    y.name = "is_hit"

    if verbose:
        print(
            f"[clean] dropped {n_before - len(df)} rows missing rating/directions "
            f"-> {len(df)} usable recipes"
        )
        hits = int(y.sum())
        print(
            f"[target] Hit(1)={hits} ({100 * hits / len(y):.1f}%) | "
            f"Miss(0)={len(y) - hits} ({100 * (len(y) - hits) / len(y):.1f}%)"
        )
    return df, y


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return X: nutrition + binary tags + raw `directions`/`ingredients` text.

    Identifier, target and unused columns are excluded. The two text columns
    are retained so `CulinaryFeatureExtractor` can consume them downstream.
    """
    drop_cols = [c for c in NON_FEATURE_COLUMNS if c in df.columns]
    return df.drop(columns=drop_cols)


def split_data(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 train/test split, performed BEFORE feature engineering."""
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


# ===========================================================================
# 4. Custom transformer skeleton
# ===========================================================================
class CulinaryFeatureExtractor(BaseEstimator, TransformerMixin):
    """Engineer numeric + binary culinary features from recipe text.

    Baseline numeric features (always produced):
        * ``num_steps``           — number of items in ``directions``
        * ``num_ingredients``     — number of items in ``ingredients``
        * ``avg_words_per_step``  — mean word count across direction steps

    Domain keyword features (one binary column per group):
        For each keyword group, the column is 1 if *any* keyword in that group
        appears (as a whole word/phrase) in the recipe's combined
        ``directions`` + ``ingredients`` text, else 0.

    The six keyword groups below are **intentionally empty** — populate them
    with domain knowledge, either at construction time or afterwards, e.g.::

        fx = CulinaryFeatureExtractor()
        fx.high_heat_techniques = ["sear", "broil", "grill", "char"]
        ...
        features = fx.fit_transform(X_train)

    While a group is empty its binary column is emitted as all-zeros, so the
    output schema is stable regardless of which groups have been filled in.
    """

    #: Attribute names of the six keyword groups, in output order.
    KEYWORD_GROUPS: Tuple[str, ...] = (
        "high_heat_techniques",
        "low_and_slow_techniques",
        "technical_execution",
        "prep_and_patience",
        "flavor_development",
        "premium_ingredients",
    )

    BASELINE_FEATURES: Tuple[str, ...] = (
        "num_steps",
        "num_ingredients",
        "avg_words_per_step",
    )

    def __init__(
        self,
        directions_col: str = "directions",
        ingredients_col: str = "ingredients",
        high_heat_techniques: Sequence[str] = [],
        low_and_slow_techniques: Sequence[str] = [],
        technical_execution: Sequence[str] = [],
        prep_and_patience: Sequence[str] = [],
        flavor_development: Sequence[str] = [],
        premium_ingredients: Sequence[str] = [],
    ) -> None:
        self.directions_col = directions_col
        self.ingredients_col = ingredients_col

        # --- Domain keyword groups (EMPTY by default — populate later) -------
        # Per the scikit-learn estimator contract, parameters are stored
        # *verbatim* here: no copying, validation or transformation. That is
        # what lets clone()/GridSearchCV/cross_val_score round-trip these
        # groups correctly. Populate a group by ASSIGNING a fresh list, e.g.
        #     fx.high_heat_techniques = ["sear", "broil", ...]
        # (assign a new list rather than .append() to the shared default).
        self.high_heat_techniques = high_heat_techniques
        self.low_and_slow_techniques = low_and_slow_techniques
        self.technical_execution = technical_execution
        self.prep_and_patience = prep_and_patience
        self.flavor_development = flavor_development
        self.premium_ingredients = premium_ingredients

    # -- sklearn API --------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CulinaryFeatureExtractor":
        """Stateless fit — validates required columns and records output names."""
        for col in (self.directions_col, self.ingredients_col):
            if col not in X.columns:
                raise KeyError(f"CulinaryFeatureExtractor: missing column '{col}'.")
        self.feature_names_out_ = list(self.BASELINE_FEATURES) + [
            f"has_{group}" for group in self.KEYWORD_GROUPS
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Produce the engineered feature matrix (returns a DataFrame)."""
        directions = X[self.directions_col].apply(self._to_token_list)
        ingredients = X[self.ingredients_col].apply(self._to_token_list)

        out = pd.DataFrame(index=X.index)

        # --- Baseline numeric features ---------------------------------------
        out["num_steps"] = directions.apply(len)
        out["num_ingredients"] = ingredients.apply(len)
        out["avg_words_per_step"] = directions.apply(self._avg_words_per_step)

        # --- Combined, lowercased text for keyword lookup --------------------
        combined_text = (
            directions.apply(lambda steps: " ".join(steps))
            + " "
            + ingredients.apply(lambda items: " ".join(items))
        ).str.lower()

        # --- One binary column per keyword group -----------------------------
        for group in self.KEYWORD_GROUPS:
            keywords = getattr(self, group)
            column = f"has_{group}"
            if not keywords:
                out[column] = 0  # group not yet populated -> all zeros
            else:
                pattern = self._build_keyword_pattern(keywords)
                out[column] = combined_text.str.contains(
                    pattern, regex=True, na=False
                ).astype(int)

        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(
            list(self.BASELINE_FEATURES)
            + [f"has_{group}" for group in self.KEYWORD_GROUPS]
        )

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _to_token_list(value) -> List[str]:
        """Normalize a cell into a list of strings (lists, arrays, str, NaN)."""
        if isinstance(value, (list, tuple, np.ndarray)):
            return [str(v) for v in value]
        if isinstance(value, str):
            return [value]
        return []  # None / NaN / other

    @staticmethod
    def _avg_words_per_step(steps: List[str]) -> float:
        """Mean number of words across direction steps (0.0 if no steps)."""
        if not steps:
            return 0.0
        return float(np.mean([len(step.split()) for step in steps]))

    @staticmethod
    def _build_keyword_pattern(keywords: Sequence[str]) -> str:
        """Whole-word/phrase, case-insensitive alternation regex."""
        import re

        escaped = [re.escape(kw.lower()) for kw in keywords if str(kw).strip()]
        return r"\b(?:" + "|".join(escaped) + r")\b"


# ===========================================================================
# Orchestration
# ===========================================================================
def build_dataset(
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Run the full Phase-0 pipeline and return (X_train, X_test, y_train, y_test)."""
    binary_df = load_binary_matrix()
    text_df = load_recipe_text()

    merged = merge_datasets(binary_df, text_df, verbose=verbose)
    merged, y = clean_and_binarize(merged, verbose=verbose)
    X = build_feature_frame(merged)

    return split_data(X, y)


def main() -> None:
    X_train, X_test, y_train, y_test = build_dataset(verbose=True)

    print("\n" + "=" * 60)
    print("PHASE 0 — DATA FOUNDATION READY")
    print("=" * 60)
    print(f"X_train shape : {X_train.shape}")
    print(f"X_test  shape : {X_test.shape}")
    print(f"y_train shape : {y_train.shape}  (hit rate {y_train.mean():.3f})")
    print(f"y_test  shape : {y_test.shape}  (hit rate {y_test.mean():.3f})")

    # -----------------------------------------------------------------------
    # Culinary domain knowledge injection
    # -----------------------------------------------------------------------
    # Keyword groups curated from professional culinary domain knowledge.
    # `CulinaryFeatureExtractor` is stateless w.r.t. these lists, but we still
    # `fit` on X_train only to honor the standard sklearn discipline (and so
    # this slot is the right place to add any training-statistics-based
    # features later without restructuring the pipeline).
    extractor = CulinaryFeatureExtractor(
        high_heat_techniques=[
            "sear", "saute", "sauté", "broil", "stir-fry", "pan-fry", "grill",
            "deep-fry", "blanch", "char", "blacken", "flambe", "flambé",
            "flash-fry", "scald", "scorch", "blister", "wok-fry", "sear-roast",
        ],
        low_and_slow_techniques=[
            "sous vide", "confit", "braise", "slow-roast", "simmer", "stew",
            "poach", "sweat", "render", "coddle", "steep", "infuse",
            "barbecue", "bbq", "smoke", "baste", "slow-cook", "roast",
        ],
        technical_execution=[
            "temper", "emulsify", "deglaze", "clarify", "monter au beurre",
            "puree", "purée", "strain", "knead", "muddle", "macerate", "score",
            "dredge", "whip", "skim", "butterfly", "truss", "chiffonade",
            "supreme", "debone", "fillet", "zest", "bind", "thicken", "mount",
            "crimp", "pipe", "julienne", "brunoise", "batonnet", "oblique",
            "paysanne", "shave", "mandoline", "mortar and pestle",
            "ice-cream maker", "spice mill", "double boiler",
            "candy thermometer", "deep-fat thermometer", "steamer insert",
            "dariole molds", "springform pan", "dutch oven", "cleaver",
            "blowtorch", "piping bag", "immersion circulator",
            "sous vide machine", "vacuum sealer", "proof", "punch down",
            "blind bake", "dock", "flute", "laminate", "bloom", "prove",
        ],
        prep_and_patience=[
            "marinate", "brine", "ferment", "overnight", "rest", "cure",
            "soak", "rise", "proof", "age", "pickle", "steep", "dry-rub",
            "bloom", "activate", "temper",
        ],
        flavor_development=[
            "reduce", "caramelize", "smoke", "jus", "char", "glaze", "infuse",
            "zest", "baste", "sweat", "render", "extract", "curing",
            "smoke-infuse", "dry-roast", "aioli", "hollandaise", "bearnaise",
            "bechamel", "veloute", "espagnole", "demi-glace", "roux", "slurry",
            "chutney", "compote", "pesto", "chimichurri", "gastrique",
            "coulis", "gremolata", "mignonette", "compound butter",
        ],
        premium_ingredients=[
            "truffle", "truffles", "saffron", "bone marrow", "wagyu", "caviar",
            "dry-aged", "foie gras", "kobe", "edible gold", "vanilla bean",
            "chanterelle", "morel", "porcini", "lobster", "langoustine",
            "oyster", "scallops", "prosciutto", "iberico", "quail",
            "duck breast", "sweetbreads", "duck confit", "cognac", "armagnac",
            "pancetta", "guanciale", "uni", "bottarga", "matsutake", "beluga",
            "fish sauce", "oyster sauce", "hoisin", "sriracha", "gochujang",
            "miso", "mirin", "sake", "tahini", "curry paste", "garam masala",
            "harissa", "zaatar", "sumac", "tamarind", "preserved lemon",
            "chipotle", "ancho", "guajillo", "masa", "tomatillo", "dashi",
            "katsuobushi",
        ],
    )

    # Fit on train only, transform both splits (leakage-free).
    culinary_train = extractor.fit_transform(X_train)
    culinary_test = extractor.transform(X_test)

    print("\nCulinaryFeatureExtractor — engineered feature matrix:")
    print(f"  culinary_train shape : {culinary_train.shape}")
    print(f"  culinary_test  shape : {culinary_test.shape}")

    # Hit-rate per binary keyword feature on the training split — a quick
    # sanity check that each populated group actually fires on real recipes.
    print("\nKeyword group activation on X_train (positives / total):")
    n_train = len(culinary_train)
    for group in CulinaryFeatureExtractor.KEYWORD_GROUPS:
        col = f"has_{group}"
        positives = int(culinary_train[col].sum())
        n_keywords = len(getattr(extractor, group))
        print(
            f"  {col:<32} {positives:>5} / {n_train}  "
            f"({100 * positives / n_train:5.1f}%)  [{n_keywords} keywords]"
        )

    # Baseline numerics summary (sanity check, not a model).
    print("\nBaseline numeric features (X_train) — describe():")
    print(culinary_train[list(CulinaryFeatureExtractor.BASELINE_FEATURES)]
          .describe().round(2).to_string())


if __name__ == "__main__":
    main()
