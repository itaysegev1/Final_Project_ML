"""
Phase 0 — Data Foundation and the Culinary Feature Engineering Setup

Epicurious "Recipes with Rating and Nutrition" — binary classification:
    "Hit"  (1) : rating >= 4.0
    "Miss" (0) : rating <  4.0

This module deals only with the data foundation part:

    1. Safe and reproducible loading of epi_r.csv (the sparse binary tag matrix
       with the nutrition and rating) and full_format_recipes.json (the raw
       directions and ingredients text).
    2. A robust merge of the two representations.
    3. Target binarization.
    4. A leakage-free 80/20 train/test split (we do the split BEFORE any
       feature engineering).
    5. The CulinaryFeatureExtractor transformer skeleton — six empty keyword
       lists are defined in __init__ for us to fill later with the domain
       knowledge. No models are fitted here.

A small note on the merge strategy (why we don't merge on row index or only on title)
The two files have two facts that drive the design:

  * The CSV and the JSON are stored in different row orders — a positional
    (index-based) merge aligns less than 3% of the rows correctly, so it is
    not usable.
  * The title alone is NOT unique: around 2,300 titles repeat in each file,
    so a title-only join creates a many-to-many cartesian explosion.

Both files carry the same five numeric fields independently
(rating, calories, protein, fat, sodium). By combining the (normalized) title
with these five numbers we get a composite key that is unique for ~18.2k
recipes and matches the two files cleanly 1:1. Real exact-duplicate recipes
(the same title and the same nutrition and the same rating) are collapsed
with keep="first", because they don't carry any extra signal.

A small note on the helper scripts that were provided (utils.py, recipe.py)
We reviewed both. recipe.py is a BeautifulSoup/urllib web-scraper used to
BUILD the dataset from live HTML — not relevant for loading the already-saved
files (and importing it would also require bs4). utils.py::sublists_to_binaries
one-hot-encodes a list column, but the CSV already comes with those binary
tag columns, so we don't need it for the merge. The JSON is already in clean
record-oriented dict form, so json.load is the most efficient parse path and
that is what we use directly.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split


# Configuration
# We resolve the data paths from the project root (the parent of src/), so the
# same import works whether Python is invoked from the project root or from a
# sub-directory like notebooks/.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR: str = os.path.join(_PROJECT_ROOT, "data")
CSV_PATH: str = os.path.join(DATA_DIR, "epi_r.csv")
JSON_PATH: str = os.path.join(DATA_DIR, "full_format_recipes.json")

RATING_THRESHOLD: float = 4.0          # >= 4.0 -> "Hit" (1)
TEST_SIZE: float = 0.20                # 80 / 20 split
from src._constants import RANDOM_STATE  # the single source of truth, see src/_constants.py

# the fields that exist in BOTH files; (normalized title + these) is the merge key
SHARED_NUMERIC: Tuple[str, ...] = ("rating", "calories", "protein", "fat", "sodium")

# the raw text columns the CSV is missing, that we pull from the JSON
JSON_TEXT_COLUMNS: Tuple[str, ...] = ("directions", "ingredients")

# Columns we don't want in the model matrix X. We keep this list minimal on purpose:
#   - title  -> just an identifier
#   - rating -> target leakage (we derive y from it)
# NOTE: we don't put "date" here on purpose. The CSV ships a binary tag column
# literally named "date" (the dried fruit), and that is a real feature.
# The JSON's unrelated publication "date" we just never pull in
# (see merge_datasets), so we avoid both the name collision and a useless drop.
NON_FEATURE_COLUMNS: Tuple[str, ...] = ("title", "rating")

_MERGE_KEY: str = "_merge_key"


# 1. Data loading
def load_binary_matrix(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """
    This function loads the sparse binary tag / nutrition / rating matrix (epi_r.csv)
    The CSV header contains some mojibake (for example a garbled "bon appetit"
    duplicate); this affects only a few tag column names and is harmless for
    our purposes. We try UTF-8 first and fall back to Latin-1 so that the read
    never crashes on a stray byte.
    :param csv_path: the path of the CSV file we want to load
    :return: the loaded data frame
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find CSV at '{csv_path}'.")
    try:
        return pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, encoding="latin-1")


def load_recipe_text(json_path: str = JSON_PATH) -> pd.DataFrame:
    """
    This function loads the raw text recipe records (full_format_recipes.json)
    The file is a JSON array of dicts; json.load -> DataFrame keeps directions
    and ingredients as native Python lists, which the transformer below consumes
    directly.
    :param json_path: the path of the JSON file we want to load
    :return: the loaded data frame
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Could not find JSON at '{json_path}'.")
    with open(json_path, "r", encoding="utf-8") as fh:
        records = json.load(fh)
    return pd.DataFrame(records)


# 2. Merge
def _build_merge_key(df: pd.DataFrame) -> pd.Series:
    """
    Here we build the composite merge key: normalized title + the shared numerics
    The numbers are coerced and rounded to 3 decimals so the two files' float
    representations line up exactly.
    :param df: the data frame we build the merge key from
    :return: the merge key series
    """
    def _fmt_num(series: pd.Series) -> pd.Series:
        # deterministic, fixed precision string so both files format identically
        numeric_series = pd.to_numeric(series, errors="coerce").round(3)
        return numeric_series.map(lambda v: "nan" if pd.isna(v) else f"{v:.3f}")

    merge_key = df["title"].fillna("").astype(str).str.strip().str.lower()
    for col in SHARED_NUMERIC:
        merge_key = merge_key + "|" + _fmt_num(df[col])
    return merge_key


def merge_datasets(
    binary_df: pd.DataFrame,
    text_df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    In this function we merge the binary/nutrition matrix with the raw recipe text
    The strategy: we de-duplicate each side on the composite key (keep="first")
    and then do a 1:1 inner join. We only pull the text columns the CSV is missing
    from the JSON, so no _x/_y suffix collisions happen.
    :param binary_df: the binary/nutrition data frame (from the CSV)
    :param text_df: the raw text data frame (from the JSON)
    :param verbose: if True we print a small summary of what happened in the merge
    :return: the merged data frame
    """
    binary_df = binary_df.copy()
    text_df = text_df.copy()

    binary_df[_MERGE_KEY] = _build_merge_key(binary_df)
    text_df[_MERGE_KEY] = _build_merge_key(text_df)

    # counting how many duplicate keys are on each side before we drop them
    num_csv_dupes = int(binary_df[_MERGE_KEY].duplicated().sum())
    num_json_dupes = int(text_df[_MERGE_KEY].duplicated().sum())
    binary_df = binary_df.drop_duplicates(subset=_MERGE_KEY, keep="first")
    text_df = text_df.drop_duplicates(subset=_MERGE_KEY, keep="first")

    # We pull ONLY the raw text columns the CSV is missing. Anything else the
    # JSON offers (categories/date/desc) is either redundant with the CSV's
    # binary tags or unused here; pulling it would risk name collisions (for
    # example the CSV's "date" fruit tag vs. the JSON's publication "date").
    text_cols_to_pull = [c for c in JSON_TEXT_COLUMNS if c in text_df.columns]
    missing_text_cols = set(JSON_TEXT_COLUMNS) - set(text_cols_to_pull)
    if missing_text_cols:
        raise KeyError(f"JSON is missing expected text columns: {sorted(missing_text_cols)}")
    text_subset = text_df[[_MERGE_KEY, *text_cols_to_pull]]

    # now we do the 1:1 inner join and drop the merge key column
    merged = binary_df.merge(text_subset, on=_MERGE_KEY, how="inner", validate="1:1")
    merged = merged.drop(columns=_MERGE_KEY)

    if verbose:
        print(
            f"[merge] CSV rows={len(binary_df) + num_csv_dupes} "
            f"(dropped {num_csv_dupes} dup keys) | "
            f"JSON rows={len(text_df) + num_json_dupes} "
            f"(dropped {num_json_dupes} dup keys) -> merged={len(merged)}"
        )
    return merged


# 3. Target + cleaning + split
def clean_and_binarize(df: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """
    In this function we drop the rows that are missing rating or directions, and
    then we build the target y.
    A directions value counts as missing if it is null OR an empty list.
    :param df: the merged data frame
    :param verbose: if True we print how many rows we dropped and the class balance
    :return: the cleaned data frame and the target series y
    """
    num_rows_before = len(df)
    has_rating = df["rating"].notna()
    has_directions = df["directions"].apply(
        lambda d: isinstance(d, (list, tuple, np.ndarray)) and len(d) > 0
    )
    df = df.loc[has_rating & has_directions].reset_index(drop=True)

    # binarize the rating to get the target y
    y = (df["rating"] >= RATING_THRESHOLD).astype(int)
    y.name = "is_hit"

    if verbose:
        print(
            f"[clean] dropped {num_rows_before - len(df)} rows missing rating/directions "
            f"-> {len(df)} usable recipes"
        )
        num_hits = int(y.sum())
        print(
            f"[target] Hit(1)={num_hits} ({100 * num_hits / len(y):.1f}%) | "
            f"Miss(0)={len(y) - num_hits} ({100 * (len(y) - num_hits) / len(y):.1f}%)"
        )
    return df, y


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function returns X: the nutrition + binary tags + raw directions/ingredients text
    We exclude the identifier, the target and the unused columns. The two text
    columns are kept so CulinaryFeatureExtractor can consume them downstream.
    :param df: the cleaned data frame
    :return: the feature data frame X
    """
    cols_to_drop = [c for c in NON_FEATURE_COLUMNS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def split_data(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified 80/20 train/test split, done BEFORE any feature engineering
    :param X: the feature data frame
    :param y: the target series
    :return: X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


# 4. The custom transformer skeleton
class CulinaryFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    This class engineers numeric + binary culinary features from the recipe text

    The baseline numeric features (always produced):
        * num_steps           — number of items in directions
        * num_ingredients     — number of items in ingredients
        * avg_words_per_step  — mean word count across the direction steps

    The domain keyword features (one binary column per group):
        For each keyword group, the column is 1 if ANY keyword in that group
        appears (as a whole word/phrase) in the recipe's combined
        directions + ingredients text, else it is 0.

    The six keyword groups below are EMPTY on purpose — we populate them with
    the domain knowledge either at construction time or later, for example:

        fx = CulinaryFeatureExtractor()
        fx.high_heat_techniques = ["sear", "broil", "grill", "char"]
        ...
        features = fx.fit_transform(X_train)

    While a group is empty its binary column is emitted as all-zeros, so the
    output schema stays stable no matter which groups have been filled.
    """

    #: the attribute names of the six keyword groups, in the output order
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
        """
        In this constructor we just save the parameters as the sklearn estimator
        contract requires (verbatim, no copy or validation), so clone() and
        GridSearchCV can round-trip the groups correctly.
        :param directions_col: name of the directions column in X
        :param ingredients_col: name of the ingredients column in X
        :param high_heat_techniques: keywords for high heat cooking techniques
        :param low_and_slow_techniques: keywords for low and slow cooking techniques
        :param technical_execution: keywords for technical execution (knife/skill work)
        :param prep_and_patience: keywords for prep and time intensive steps
        :param flavor_development: keywords for flavor building methods/sauces
        :param premium_ingredients: keywords for premium / luxury ingredients
        """
        self.directions_col = directions_col
        self.ingredients_col = ingredients_col

        # The domain keyword groups (EMPTY by default — we populate them later).
        # Per the scikit-learn estimator contract, parameters are stored
        # verbatim here: no copy, no validation, no transformation. That is
        # what lets clone()/GridSearchCV/cross_val_score round-trip these
        # groups correctly. We populate a group by ASSIGNING a fresh list, like:
        #     fx.high_heat_techniques = ["sear", "broil", ...]
        # (we assign a new list rather than .append() to the shared default)
        self.high_heat_techniques = high_heat_techniques
        self.low_and_slow_techniques = low_and_slow_techniques
        self.technical_execution = technical_execution
        self.prep_and_patience = prep_and_patience
        self.flavor_development = flavor_development
        self.premium_ingredients = premium_ingredients

    # the sklearn API
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CulinaryFeatureExtractor":
        """
        This is a stateless fit — we just validate that the required columns
        exist and record the output names
        :param X: the input feature data frame
        :param y: not used, kept for the sklearn API compatibility
        :return: self
        """
        for col in (self.directions_col, self.ingredients_col):
            if col not in X.columns:
                raise KeyError(f"CulinaryFeatureExtractor: missing column '{col}'.")
        self.feature_names_out_ = list(self.BASELINE_FEATURES) + [
            f"has_{group}" for group in self.KEYWORD_GROUPS
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Here we produce the engineered feature matrix (returns a DataFrame)
        :param X: the input feature data frame
        :return: the engineered features as a DataFrame with the same index as X
        """
        directions = X[self.directions_col].apply(self._to_token_list)
        ingredients = X[self.ingredients_col].apply(self._to_token_list)

        out = pd.DataFrame(index=X.index)

        # the baseline numeric features
        out["num_steps"] = directions.apply(len)
        out["num_ingredients"] = ingredients.apply(len)
        out["avg_words_per_step"] = directions.apply(self._avg_words_per_step)

        # combined lowercased text for the keyword lookup
        combined_text = (
            directions.apply(lambda steps: " ".join(steps))
            + " "
            + ingredients.apply(lambda items: " ".join(items))
        ).str.lower()

        # one binary column per keyword group
        for group in self.KEYWORD_GROUPS:
            group_keywords = getattr(self, group)
            column = f"has_{group}"
            if not group_keywords:
                out[column] = 0  # the group is not populated yet -> all zeros
            else:
                keyword_pattern = self._build_keyword_pattern(group_keywords)
                out[column] = combined_text.str.contains(
                    keyword_pattern, regex=True, na=False
                ).astype(int)

        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """
        Returns the output feature names in the order transform produces them
        :param input_features: not used, kept for the sklearn API compatibility
        :return: the array of the output feature names
        """
        return np.asarray(
            list(self.BASELINE_FEATURES)
            + [f"has_{group}" for group in self.KEYWORD_GROUPS]
        )

    # helpers
    @staticmethod
    def _to_token_list(value) -> List[str]:
        """
        This helper normalizes a cell value into a list of strings
        (handles lists, arrays, plain strings, NaN, None)
        :param value: the raw cell value
        :return: a list of strings
        """
        if isinstance(value, (list, tuple, np.ndarray)):
            return [str(v) for v in value]
        if isinstance(value, str):
            return [value]
        return []  # None / NaN / other

    @staticmethod
    def _avg_words_per_step(steps: List[str]) -> float:
        """
        This helper computes the mean number of words across the direction steps
        (returns 0.0 if there are no steps so we don't divide by zero)
        :param steps: list of direction steps (strings)
        :return: the average number of words per step
        """
        if not steps:
            return 0.0
        return float(np.mean([len(step.split()) for step in steps]))

    @staticmethod
    def _build_keyword_pattern(keywords: Sequence[str]) -> str:
        """
        This helper builds a whole-word/phrase, case insensitive alternation regex
        :param keywords: the list of keywords for the group
        :return: the regex pattern string
        """
        import re

        escaped_keywords = [re.escape(kw.lower()) for kw in keywords if str(kw).strip()]
        return r"\b(?:" + "|".join(escaped_keywords) + r")\b"


# Orchestration
def build_dataset(
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    This function runs the full Phase 0 pipeline and returns
    (X_train, X_test, y_train, y_test)
    :param verbose: if True we print progress info from each step
    :return: X_train, X_test, y_train, y_test
    """
    binary_df = load_binary_matrix()
    text_df = load_recipe_text()

    merged = merge_datasets(binary_df, text_df, verbose=verbose)
    merged, y = clean_and_binarize(merged, verbose=verbose)
    X = build_feature_frame(merged)

    return split_data(X, y)


def main() -> None:
    """
    The main entry point of phase 0 — builds the dataset, prints the shapes
    and then fits the CulinaryFeatureExtractor on X_train as a sanity check
    """
    X_train, X_test, y_train, y_test = build_dataset(verbose=True)

    print("\n" + "=" * 60)
    print("PHASE 0 — DATA FOUNDATION READY")
    print("=" * 60)
    print(f"X_train shape : {X_train.shape}")
    print(f"X_test  shape : {X_test.shape}")
    print(f"y_train shape : {y_train.shape}  (hit rate {y_train.mean():.3f})")
    print(f"y_test  shape : {y_test.shape}  (hit rate {y_test.mean():.3f})")

    # Culinary domain knowledge injection
    # Keyword groups curated from the professional culinary domain knowledge.
    # CulinaryFeatureExtractor is stateless w.r.t. these lists, but we still
    # fit on X_train only to keep the standard sklearn discipline (and so this
    # slot is the right place to add any training-statistics-based features
    # later without restructuring the pipeline).
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

    # fit on train only, transform both splits (leakage-free)
    culinary_train = extractor.fit_transform(X_train)
    culinary_test = extractor.transform(X_test)

    print("\nCulinaryFeatureExtractor — engineered feature matrix:")
    print(f"  culinary_train shape : {culinary_train.shape}")
    print(f"  culinary_test  shape : {culinary_test.shape}")

    # The hit rate per binary keyword feature on the training split — a quick
    # sanity check that each populated group actually fires on real recipes.
    print("\nKeyword group activation on X_train (positives / total):")
    num_train_rows = len(culinary_train)
    for group in CulinaryFeatureExtractor.KEYWORD_GROUPS:
        col = f"has_{group}"
        num_positives = int(culinary_train[col].sum())
        num_keywords = len(getattr(extractor, group))
        print(
            f"  {col:<32} {num_positives:>5} / {num_train_rows}  "
            f"({100 * num_positives / num_train_rows:5.1f}%)  [{num_keywords} keywords]"
        )

    # baseline numerics summary (just a sanity check, not a model)
    print("\nBaseline numeric features (X_train) — describe():")
    print(culinary_train[list(CulinaryFeatureExtractor.BASELINE_FEATURES)]
          .describe().round(2).to_string())


if __name__ == "__main__":
    main()
