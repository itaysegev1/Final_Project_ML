# Predicting Recipe Success on Epicurious: A Hit/Miss Classification Study

A machine-learning project on the *Epicurious — Recipes with Rating and
Nutrition* dataset. The task is binary classification: predict whether a
recipe will be a **"Hit"** (rating ≥ 4.0) or a **"Miss"** (rating < 4.0)
from its nutrition profile, editorial tags, and raw text instructions.

---

## 1. Executive Summary

The final model is a **Random Forest** ensemble on the Advanced feature
matrix, with a test accuracy of **0.6239**, F1 of **0.6753**, and a
meaningful improvement of **+2.28 pp Acc / +3.49 pp F1** over the
converged Logistic Regression baseline (Acc 0.6011, F1 0.6405). Two
non-linear models were added as an "Academic Deep-Dive" extension after
the linear analysis (Phases 3–4) showed the engineered features barely
registering in the linear model:

- **Random Forest broke the linear ceiling**, validating the hypothesis
  that genuine non-linear interactions exist on this dataset.
- **MLPClassifier (128, 64) initially overfit catastrophically.** In its
  first (unregularised) configuration its training loss collapsed to
  **0.0241** over 53 epochs while its test accuracy *fell far below the
  Logistic Regression baseline* (0.5728 vs 0.6011). After enabling
  validation-based early stopping (`early_stopping=True`,
  `validation_fraction=0.15`, `n_iter_no_change=10`), the network
  stopped after 18 epochs with weights restored to the best-validation
  epoch (epoch 7), recovering to Acc **0.6126** and F1 **0.6632** — a
  **+3.98 pp Acc / +6.96 pp F1** improvement on the test set over the
  overfit run, and clearing the Logistic Regression baseline on both
  metrics (+1.15 pp Acc, +2.27 pp F1). Capacity, on this dataset, is a
  liability without explicit regularisation — and early stopping is the
  cheap regulariser that turns the same architecture from
  below-baseline into above-baseline. The overfit run is fully
  reproducible via `python train_mlp.py --no-early-stopping`.

Before any of these comparisons could be trusted, however, a code audit
caught a **silent dead-feature bug** in the first version of the
pipeline: the six engineered `has_<keyword_group>` binary features were
defined — keyword lists, extractor, the lot — but the keyword lists were
never injected into the extractor instance the pipeline actually used,
so all six columns were identically zero in every model run. The fix
(§2.5.1) and the runtime guard that now prevents a regression are part
of the pipeline; every number in this report comes from the fixed
pipeline. The lesson is recorded as Lesson #7 in §4.

The most informative *modelling* finding is the inversion of the
feature-importance picture between the linear and non-linear analyses:

> **The engineered continuous numerics (`avg_words_per_step`,
> `num_ingredients`, `num_steps`) and the nutrition columns (`sodium`,
> `calories`, `fat`, `protein`) — which sit in the lower half of the
> Logistic Regression |coefficient| ranking (e.g. `avg_words_per_step`
> at rank 597/687) — are the *top seven* features by impurity reduction
> in Random Forest, and all six `has_*` keyword features rank inside
> RF's top 20, above ~670 of the editorial tags.** The linear model
> finds them largely redundant with the binary tag matrix (collinearity
> + L2 shrinkage); a non-linear model, which is invariant to
> collinearity, finds them dominant. The generalisable insight is that
> feature *utility* is model-dependent: features that look useless to
> one model class may carry the deepest signal for another.

The engineered features are therefore vindicated where it matters: the
Advanced matrix produces positive Δ accuracy across every model that can
exploit feature interactions (RF +0.0198, MLP +0.0154, improved
PCA+KNN +0.0129), while the purely linear models stay essentially flat
(LR +0.0005, Perceptron −0.0060). The original "Aha" of Phase 3 (only
`num_ingredients` mattered) was an artifact of the linear lens.

---

## 2. Methodology

### 2.1 Data Alignment & The Cartesian Explosion

Initial attempts to join the CSV and JSON datasets sequentially yielded only a 2.8% match rate, revealing disparate sorting across the files. Furthermore, a naive join on the recipe `title` risked a "Cartesian Explosion" due to approximately 2,300 duplicated titles (e.g., multiple distinct recipes generically named "Chicken Soup"). Merging purely on titles would have created a massive matrix of corrupted, cross-matched recipes. To prevent this data corruption, a composite key was engineered using `title` + `rating` + `calories` + `protein` + `fat` + `sodium`. This precise fingerprinting successfully filtered duplicates and yielded 18,223 clean, 1:1 validated recipe matches.

### 2.2 Pipeline Architecture (Leakage-Free)

The pipeline is split across five explicit phases, each in its own
script:

| Phase | Script | Responsibility |
|-------|--------|----------------|
| 0 | [`src/data_foundation.py`](../src/data_foundation.py) | Load CSV + JSON, merge, target binarization, train/test split |
| 1 | [`src/preprocessing.py`](../src/preprocessing.py) | Imputation, scaling, custom transformer, two output matrices |
| 2 | `train_*.py` + [`evaluate_all_results.py`](../evaluate_all_results.py) | Per-model trainers writing `results/<slug>/metrics.json`; aggregator reads them and prints the summary table — see §2.6 |
| 3 | [`analysis.py`](../analysis.py) | Interpretation, confidence buckets, ROC and confusion plots |
| 4 | [`advanced_tuning.py`](../advanced_tuning.py) | Threshold selection on a validation split, top-20 feature ranking |

Every preprocessing component (`SimpleImputer`, `RobustScaler`,
`CulinaryFeatureExtractor`, `PCA`) is `.fit()` on the training split only
and used to `.transform()` the test split. The 80/20 split is performed
**before any feature engineering or fitted-statistic computation**,
enforced via `train_test_split(stratify=y, random_state=42)`. The same
`random_state` is propagated to every model.

### 2.3 Robust Scaling for Outliers

An initial pass with `StandardScaler` revealed extreme outliers in the
nutrition columns (scraping artifacts in the original Epicurious feed
produced recipes with implausible calorie and sodium values). The effect
was diagnostic: after standard-scaling, the test split's `fat` column had
a standard deviation of **0.034** — virtually all variance was being
absorbed by a small number of outliers, which would in turn destroy any
distance-based or L2-regularized model.

The pipeline switched to `RobustScaler` (centers on the median, scales by
the IQR — both robust to outliers), and the same `fat` IQR on the test
split recovered to **0.962**. This is paired with `SimpleImputer(strategy=
"median")` for the ~21% of nutrition values missing from the CSV; the
median imputation + median centering ensures that any imputed value lands
at exactly 0.0 in the scaled feature space (the neutral "no information"
position).

### 2.4 Solver Convergence: From `lbfgs` to `liblinear`

The default `lbfgs` solver for Logistic Regression did not converge on the
high-dimensional sparse-binary input even at `max_iter = 5000`. The
diagnostic that exposed the issue: one specific test recipe ("Lamb Köfte
with Tarator Sauce") was assigned `p(Hit) = 0.9997` at `max_iter = 2000`
and `p(Hit) = 0.0000` at `max_iter = 5000`, with all other parameters
fixed. A 99-percentage-point swing on a single recipe between two runs of
the same model is unambiguous evidence the optimizer had not located the
optimum.

The solver was therefore switched to `liblinear` (coordinate descent —
sklearn's historical default for binary L2-penalized Logistic Regression
on sparse high-dimensional data). With `liblinear`, the optimizer
converges cleanly, the convergence warning is eliminated, and per-recipe
probabilities are reproducible across runs.

### 2.5 Feature Engineering: The Two Matrices

Phase 1 produces two parallel preprocessed matrices to enable a clean
A/B comparison:

- **Baseline matrix** (`X_*_baseline`, shape *(n, 678)*): 4 robust-scaled
  nutrition columns + 674 raw binary tags. No JSON-derived text features.
- **Advanced matrix** (`X_*_advanced`, shape *(n, 687)*): identical to
  baseline, plus the 9 outputs of `CulinaryFeatureExtractor`:
  3 numeric (`num_steps`, `num_ingredients`, `avg_words_per_step`,
  all robust-scaled) + 6 binary (`has_<keyword_group>`).

Both matrices share the same row partition, so any difference in
downstream metrics is attributable to the additional 9 columns alone.
The six keyword groups fire on a healthy share of the training split —
`has_high_heat_techniques` 29.2%, `has_low_and_slow_techniques` 39.0%,
`has_technical_execution` 36.1%, `has_prep_and_patience` 17.0%,
`has_flavor_development` 34.8%, `has_premium_ingredients` 14.6% — far
from degenerate in either direction.

**Defensive Column Routing:** A rigorous column classification protocol was implemented to prevent namespace collisions during the extraction phase. For example, the `date` feature in the CSV correctly identifies the inclusion of the fruit "date". A naive text-extraction or identifier-drop would have overwritten or deleted this valid culinary feature under the assumption it was a publication date. Identifiers were defensively isolated and dropped without corrupting identically-named domain features.

### 2.5.1 The Dead-Feature Audit: How the Keyword Features Almost Shipped as Zeros

The first version of Phase 1 contained a silent but total bug: the
curated keyword lists existed only inside the Phase-0 sanity-check
entry point, while the pipeline's own
`CulinaryFeatureExtractor()` was constructed **bare** — six empty
keyword groups. The extractor's schema-stability contract ("an empty
group emits an all-zero column") then did exactly what it was told:
every `has_*` column in the Advanced matrix was identically zero, on
both splits, in every model run. No exception, no warning, plausible
shapes — and aggregate metrics that still looked reasonable, because
the three numeric features and 678 baseline columns carried the matrix.

A code audit caught it with a one-line check (column sums on the train
split), which is exactly the kind of structural diagnostic Lesson #5
advocates for. Three changes followed:

1. The keyword lists moved to a single module-level constant,
   `CULINARY_KEYWORDS` in [`src/data_foundation.py`](../src/data_foundation.py),
   and `make_culinary_extractor()` is now the only sanctioned way to
   build a populated extractor.
2. The Phase-1 pipeline uses that factory.
3. `build_preprocessed_datasets` now **fails loud** if any `has_*`
   column is all-zero on the train split
   (`_assert_no_dead_binary_features` in
   [`src/preprocessing.py`](../src/preprocessing.py)), so the bug class
   cannot silently regress.

Every result in this report was produced *after* the fix. The before/
after contrast is itself informative: with dead features, the "Advanced"
deltas measured only the 3 numeric features; with live features, Random
Forest's Advanced accuracy moved from 0.6178 to **0.6239** and all six
keyword features entered RF's top-20 importance ranking (§3.5).

### 2.6 MLOps & Production Architecture

Phase 2 was originally a single monolithic script (`phase2_model_training.py`) that trained all six classifiers, produced both visualisations, and printed the summary table in one run. That structure is convenient for an academic write-up but mirrors none of the patterns used in production ML systems: every change required re-training every model, results lived only in stdout, and there was no way to compose model outputs into downstream tasks (calibration, stacking, threshold tuning) without re-running the whole fleet. The Phase 2 codebase was therefore refactored into a **decoupled per-model architecture** that follows three production-grade conventions:

1. **Per-model training scripts.** Each classifier owns its own entry point: [`train_perceptron.py`](../train_perceptron.py), [`train_logistic_regression.py`](../train_logistic_regression.py), [`train_adaboost.py`](../train_adaboost.py), [`train_pca_knn.py`](../train_pca_knn.py), [`train_pca_knn_improved.py`](../train_pca_knn_improved.py), [`train_random_forest.py`](../train_random_forest.py), [`train_mlp.py`](../train_mlp.py). Each script loads the preprocessed data via Phase 1's public API, trains *only* its model on both the Baseline and Advanced matrices, and writes its results to disk. No script depends on any other train script; a change to the MLP configuration does not invalidate the LR results, and the seven trainers can be run, retried, or replaced independently. The diagnostic-driven addition of `train_pca_knn_improved.py` is itself a concrete demonstration of why the per-model layout matters in practice: it could be authored, run, and aggregated into the existing summary table without touching any other model's code. This is the standard separation in ML platforms — every model is its own pipeline stage.

2. **A canonical JSON contract for results.** Every train script writes a single `results/<slug>/metrics.json` file with the same schema (`model_name`, `display_name`, `model_config`, `n_train`, `n_test`, `random_state`, per-dataset `accuracy`/`f1`/`confusion_matrix`/`fp_rate`/`fn_rate`, plus an optional model-specific `extras` block). Per-recipe test predictions are persisted alongside as `predictions_<dataset>.npy` (int8, ~3.7 KB each) together with `test_index.npy` (the test split's row index, so the prediction files are self-describing), and any model-specific plot lands in the same folder (`random_forest/feature_importance.png`, `mlp/loss_curve.png`). The single source of truth for the schema is `src/train_utils.py::build_metrics_payload`, which is also responsible for the writers. Anything downstream that wants to consume Phase-2 output reads JSON, not Python globals — the same pattern MLflow, Weights & Biases, and homegrown ML platforms all converge on.

3. **A pure-read aggregator.** [`evaluate_all_results.py`](../evaluate_all_results.py) trains nothing. It walks `results/`, loads every `<slug>/metrics.json` it finds, assembles the seven-row summary table, and emits the linear-vs-non-linear cross-model verdict — all by reading the JSON contract above. Because it has no dependency on which trainers have run, it gracefully handles partial fleets (when only some models have been retrained) and reports which expected entries are missing. This decouples *evaluation* from *training* — the textbook structure for CI dashboards, post-hoc analysis notebooks, and any downstream stage that wants to ensemble or rank models.

**Why this matters for this project.** The architecture supports **incremental re-training**: if a Phase-1 change touches only the nutrition imputation, only the models that actually depend on nutrition need to be re-run; the rest still have valid JSON from the previous run. **DRY code** lives in [`src/train_utils.py`](../src/train_utils.py), which owns the shared fit-and-score loop, the JSON payload builder, the prediction writer, and the console formatters — so the seven train scripts each stay around 100 lines, focused entirely on their model's own configuration and any model-specific diagnostics. The two scripts with extra concerns — RF (top-20 feature-importance bar chart) and MLP (training-loss/validation-error overlay + early-stopping diagnostics block in `extras`) — drop their plot code and any extra JSON fields into the per-model script without touching the shared helper, exactly as a production ML platform would expect.

The legacy `phase2_model_training.py` has been retired now that all seven refactored scripts produce the same results through the shared `src/` pipeline.

### 2.6.1 The "42 Guarantee" — single source of truth for randomness

Reproducibility on this project rests on a single named constant.
**The literal `42` is bound in exactly one place** in the entire
codebase: [`src/_constants.py`](../src/_constants.py). Every other file
that needs the value imports it — either directly
(`from src._constants import RANDOM_STATE`) or via the re-export in
`src.train_utils` (`from src.train_utils import RANDOM_STATE`, which is
the form most callers use because they're already importing the other
training helpers from `src.train_utils`).

`src/_constants.py` deliberately has zero other imports. This matters
because `src.train_utils` imports `src.preprocessing`, which imports
`src.data_foundation`. If the canonical constant lived in
`train_utils`, the data-foundation module couldn't import it without
creating an import cycle. Putting it in a tiny isolated module breaks
the cycle cleanly.

The full audit:

| Random process | Where | Source |
|---|---|---|
| `train_test_split(stratify=y, random_state=42)` (80/20 split) | `src/data_foundation.py` | `from src._constants import RANDOM_STATE` |
| `train_test_split(stratify=y_train, random_state=42)` (threshold-selection validation split) | `advanced_tuning.py` | `from src._constants import RANDOM_STATE` |
| `Perceptron(random_state=42, ...)` | `train_perceptron.py` | `from src.train_utils import RANDOM_STATE` |
| `LogisticRegression(random_state=42, ...)` | `train_logistic_regression.py` (config imported from there by `analysis.py` and `advanced_tuning.py`) | `from src.train_utils import RANDOM_STATE` |
| `AdaBoostClassifier(random_state=42, ...)` | `train_adaboost.py` | `from src.train_utils import RANDOM_STATE` |
| `PCA(random_state=42, ...)` | `train_pca_knn.py`, `train_pca_knn_improved.py` | `from src.train_utils import RANDOM_STATE` |
| `RandomForestClassifier(random_state=42, ...)` | `train_random_forest.py` | `from src.train_utils import RANDOM_STATE` |
| `MLPClassifier(random_state=42, ...)` | `train_mlp.py` (weight init + the validation-split RNG; both variants) | `from src.train_utils import RANDOM_STATE` |
| KNeighborsClassifier | `train_pca_knn*.py` | deterministic; no `random_state` parameter exists |

Within a fixed environment, every run from a clean checkout produces
the same `results/<slug>/metrics.json` files (verified by re-running
trainers back-to-back and diffing the JSON). One important honesty
note: the seed guarantee does **not** survive dependency upgrades —
sklearn estimator behaviour changes between releases — which is why
[`requirements.txt`](../requirements.txt) pins exact versions. The
reproducibility claim is "same code + same data + same environment ⇒
same bits", and the environment is part of the contract.

### 2.7 Academic Deep-Dive: Why Random Forest and MLP Were Added

After Phases 3–4, the interpretive picture of the converged Logistic
Regression was clear: the engineered keyword features had been almost
entirely zeroed out by L2 regularization. This had two possible
explanations:

1. **The features carry no signal** — they are genuinely uninformative.
2. **The features carry signal that a linear additive model cannot
   exploit** — for example, signal that lives in *interactions* between
   features ("brisket × oven-baked × high-fat" being predictive in a way
   no single feature is) or in non-linear thresholds on continuous
   features ("recipes with 8–15 ingredients out-perform both shorter and
   longer recipes").

A linear model conflates these two cases. To distinguish them, two
non-linear classifiers were added:

- **`RandomForestClassifier` (n=200)** captures non-linear feature
  interactions through axis-aligned splits across feature pairs. Trees
  also do not share coefficient mass across collinear features — each
  split picks the single best feature locally, so collinearity ceases to
  be a regularization sink.
- **`MLPClassifier` with `hidden_layer_sizes=(128, 64)`** learns
  non-linear representations of the entire feature space through two
  fully-connected layers. `verbose=True` exposes the per-epoch loss
  trajectory; `max_iter=300` gives the optimizer head-room above the
  default 200 to reach its stopping criterion. The first run of this
  configuration overfit decisively (training loss → 0 with worsening
  test accuracy), so the final configuration adds sklearn's
  validation-based early stopping (`early_stopping=True`,
  `validation_fraction=0.15`, `n_iter_no_change=10`): 15% of the
  training split is held out internally as a validation set
  (independent of our test split), the network is trained until the
  validation accuracy stops improving for 10 consecutive epochs, and
  the network's weights are restored to the best-validation epoch on
  `fit()` completion. The `validation_scores_` series exposed by this
  configuration then powers the train-vs-validation diagnostic plot in
  §3.6.

If the non-linear models meaningfully out-perform the linear baseline,
that disconfirms hypothesis (1) — the engineered features *do* carry
signal, just not a kind a linear additive model can use. If they
plateau or under-perform, hypothesis (1) is supported and the predictive
ceiling is structural in the features rather than methodological in the
model.

The results (see §3.5–3.6 and §5.1) support hypothesis (2) twice over:
RandomForest clears the linear ceiling decisively (+2.28 pp Acc,
+3.49 pp F1 over LR), and the *regularised* MLP clears it as well
(+1.15 pp Acc, +2.27 pp F1) — non-linear capacity exploits signal the
linear boundary cannot. The unregularised MLP failure (§3.6), which
lands far *below* the linear baseline, gives a secondary lesson on
capacity-vs-data balance and the value of validation-based early
stopping: the same architecture sits 4 pp below or 1 pp above LR
depending solely on the stopping rule.

---

## 3. Key Findings

### 3.1 Feature Saturation: The Engineered Keywords Are Largely Redundant (for L2)

An honesty note up front: an earlier draft of this section explained
"zero coefficients" on the six `has_<keyword_group>` features as L2
shrinkage of collinear features. The real reason at the time was the
dead-feature bug of §2.5.1 — the columns were identically zero, so
their coefficients were trivially zero and no statistical explanation
applied. With the features actually live, the picture can be measured
instead of assumed, and it turns out to be a *weaker version* of the
original story:

| Engineered feature | LR coef | rank by \|coef\| (of 687) |
|---|---:|---:|
| `num_ingredients` | +0.3978 | 180 |
| `has_premium_ingredients` | +0.0859 | 512 |
| `has_flavor_development` | +0.0609 | 551 |
| `has_prep_and_patience` | +0.0487 | 572 |
| `has_high_heat_techniques` | −0.0459 | 576 |
| `has_technical_execution` | −0.0454 | 578 |
| `avg_words_per_step` | −0.0372 | 597 |
| `has_low_and_slow_techniques` | −0.0309 | 615 |
| `num_steps` | −0.0162 | 642 |

The six binary keyword features receive small but genuinely non-zero
coefficients (|coef| 0.03–0.09), all in the bottom third of the
ranking. This *is* consistent with partial collinearity: the keyword
group `high_heat_techniques` (containing `sear`, `grill`, `broil`,
etc.) overlaps heavily with the CSV's pre-existing `grill`, `broil`,
`fry` and related tags, and L2 splits the coefficient mass among
collinear features, shrinking each toward zero. But the claim is now
backed by the contrast in §3.5 rather than asserted: the same six
features rank in Random Forest's top 20 — trees do not share
coefficient mass across collinear inputs, so the redundancy penalty
disappears.

The qualifier "for L2" matters: §3.5 shows that the same features are
*not* redundant for tree ensembles.

### 3.2 `num_ingredients` Is the One Genuine Linearly-Additive Signal

Within the linear model, the exception that proves the rule is
`num_ingredients`. Coefficient **+0.3978**, **rank 180/687** —
comfortably above the median feature, well into the model's working
set, and *unambiguously* positive. The interpretation is direct:
holding the tag profile constant, a recipe with more ingredients is
more likely to be rated a Hit.

This is consistent with a *recipe complexity* hypothesis — more
ingredients indicate a more ambitious dish, which correlates with higher
rater engagement and more favourable ratings. Critically, this property
is not captured by the binary tag matrix, which encodes *what kind* of
recipe it is but not *how elaborate*.

### 3.3 Threshold Tuning Trade-off (F1 vs. Symmetry) — Selected on Validation, Evaluated on Test

A methodological correction relative to an earlier draft: selecting an
operating threshold by sweeping the *test* set is itself a mild form of
test-set tuning. The selection is therefore done on a **15% stratified
validation split held out of the training data** (n = 2,187; the LR is
fit on the remaining 85%), and only the *selected* thresholds are then
evaluated once on the test set, under an LR refit on the full training
split. `advanced_tuning.py` implements exactly this protocol and also
prints a descriptive test sweep for transparency (it plays no role in
selection).

The default threshold *t* = 0.50 is biased toward predicting Hit
(validation: FP-rate 0.498 vs FN-rate 0.340). The validation sweep from
*t* = 0.30 to 0.70 in 0.05 increments:

| Threshold | F1 | Accuracy | FP-rate | FN-rate | Balance | Predicted Hits |
|-----------|------|----------|---------|---------|---------|----------------|
| 0.30 | 0.7055 | 0.5766 | 0.8508 | 0.0505 | **16.84** | 90.4% |
| 0.40 | 0.6931 | 0.5972 | 0.6948 | 0.1481 | 4.69 | 77.9% |
| 0.45 | 0.6722 | 0.5999 | 0.5927 | 0.2320 | 2.56 | 68.6% |
| 0.50 *(default)* | 0.6304 | 0.5866 | 0.4975 | 0.3399 | 1.46 | 58.4% |
| **0.55** | 0.5850 | 0.5802 | 0.3896 | 0.4461 | **0.87** | 47.7% |
| 0.60 | 0.5169 | 0.5674 | 0.2787 | 0.5668 | 0.49 | 36.1% |
| 0.70 | 0.3544 | 0.5368 | 0.1207 | 0.7620 | 0.16 | 18.3% |

Selected on validation: **best-F1 threshold = 0.30**, **most-balanced
threshold = 0.55** (min |log(balance)|). Evaluated once on the test set
(full-train refit):

| Operating point | Test F1 | Test Acc | FP-rate | FN-rate | Balance | Predicted Hits |
|-----------------|---------|----------|---------|---------|---------|----------------|
| Default *t* = 0.50 | 0.6405 | 0.6011 | 0.4723 | 0.3349 | 1.41 | 57.5% |
| Best-F1 *t* = 0.30 *(val-selected)* | 0.7097 | 0.5822 | 0.8469 | 0.0437 | **19.40** | 90.5% |
| Most-balanced *t* = 0.55 *(val-selected)* | 0.5989 | 0.6005 | 0.3510 | 0.4417 | **0.80** | 46.2% |

Two findings:

1. **The "Best F1" threshold is a methodological trap.** The
   validation-selected F1-maximiser (*t* = 0.30) reaches test
   F1 = 0.7097, but at that threshold the model labels 90.5% of all
   recipes as "Hit" (true Hit rate: 53.4%) and the balance ratio is
   **19.40**. F1 alone is insufficient as a selection criterion;
   balance and predicted-class distribution must be inspected alongside
   it. The selection generalised, though: the threshold chosen on
   validation produced almost identical behaviour on test (val
   F1 0.7055 → test F1 0.7097), which is what a leakage-free protocol
   should look like.
2. **Accuracy is essentially flat across thresholds 0.40–0.55** (test:
   0.5997 → 0.6022 → 0.6011 → 0.6005). A 10-point swing in F1
   (0.6964 → 0.5989) barely moves the total error rate; what changes is
   *which class* absorbs the errors.

The recommended balanced threshold for reporting is **t = 0.55**
(test balance = 0.80, F1 = 0.5989, accuracy within 0.06 pp of the
default).

### 3.4 The "Recipe for Success" vs. "Recipe for Disaster"

Top 10 signed coefficients from the converged Logistic Regression on the
Advanced matrix:

| Rank | Coef | Hit indicator | | Rank | Coef | Miss indicator |
|------|--------|----------------|---|------|--------|------------------|
| 1 | +1.33 | kentucky derby | | 1 | −1.31 | pittsburgh |
| 2 | +1.26 | pasadena | | 2 | −1.30 | aperitif |
| 3 | +1.25 | brisket | | 3 | −1.25 | leftovers |
| 4 | +1.14 | granola | | 4 | −1.23 | lancaster |
| 5 | +1.04 | sangria | | 5 | −1.19 | jícama |
| 6 | +1.01 | 22-minute meals | | 6 | −1.17 | friendsgiving |
| 7 | +1.00 | kahlúa | | 7 | −1.09 | whole wheat |
| 8 | +0.93 | sukkot | | 8 | −1.07 | salsa |
| 9 | +0.91 | snack week | | 9 | −1.01 | scotch |
| 10 | +0.90 | trout | | 10 | −1.00 | #cakeweek |

Patterns:

- **Hit cluster — occasion-driven recipes**: `kentucky derby`, `sangria`,
  `kahlúa`, `pasadena` (Rose Bowl), `sukkot`, `ireland` (St. Patrick's
  Day, rank 12). Users come to occasion recipes with elevated
  expectations and rate generously.
- **Hit cluster — premium / labour-intensive proteins**: `brisket`,
  `trout`, with `rabbit` and `stuffing/dressing` just outside the top
  10. Labour-intensive dishes that reward careful preparation.
- **Miss cluster — restriction- and afterthought-driven labels**:
  `whole wheat`, `wild rice` (rank 14), `leftovers`, `slow cooker`
  (rank 11). Restriction- or convenience-framed recipes appear to
  receive harsher ratings — the "I'd rather have the real thing"
  effect.
- **Editorial-metadata leakage**: `#cakeweek`, `snack week`,
  `22-minute meals` and `friendsgiving` are editorial campaign tags,
  not culinary properties — and they appear on *both* sides of the
  ranking. Geographic tags (`pittsburgh`, `lancaster`,
  `north carolina` at rank 15) similarly dominate the Miss side and
  likely reflect contributor/editorial bias rather than regional
  culinary preference. These are limitations to acknowledge in the
  scope-and-validity discussion.

**Qualitative Error Analysis (The 'Lamb Köfte' Anomaly):** Analysis of test-set outliers revealed a significant prediction error for 'Lamb Köfte with Tarator Sauce'. Despite a true label of 'Hit', the converged model predicted a 'Miss' with near-zero probability. This highlights a clear 'Domain Gap': our engineered culinary dictionary lacked highly specific regional ingredients (like the complex 'Tarator' sauce) and ethnic techniques ('Köfte'). Forced to rely on broader, less favorable tags, the model misclassified the dish. This qualitative finding exposes the limitations of global tag matrices and provides a roadmap for future regional-specific feature engineering.

### 3.5 RandomForest Broke the Linear Ceiling — and Inverted the Feature Ranking

Random Forest (n=200 trees) on the Advanced matrix achieved
**Acc = 0.6239** and **F1 = 0.6753** — meaningful gains of
**+2.28 pp Acc** and **+3.49 pp F1** over the converged Logistic
Regression (0.6011 / 0.6405). This disconfirms the simpler
interpretation that the engineered features "didn't matter": there *is*
signal that the linear boundary could not exploit.

The strongest evidence is the inversion of the feature-importance
ranking. The top 20 RF features by impurity reduction (Gini) are
visualised in
[`random_forest/feature_importance.png`](random_forest/feature_importance.png).
The top 10 are:

| Rank | Importance | Feature | LR coef (Phase 3) | LR rank |
|------|-----------|---------|------------------|---------|
| 1 | 0.0475 | `avg_words_per_step` | −0.04 | 597/687 |
| 2 | 0.0418 | `calories` | small | mid-rank |
| 3 | 0.0416 | `sodium` | small | mid-rank |
| 4 | 0.0396 | `num_ingredients` | +0.40 | 180/687 |
| 5 | 0.0362 | `fat` | small | mid-rank |
| 6 | 0.0342 | `protein` | small | mid-rank |
| 7 | 0.0271 | `num_steps` | −0.02 | 642/687 |
| 8 | 0.0096 | `bon appétit` | — | tag |
| 9 | 0.0093 | `has_technical_execution` | −0.05 | 578/687 |
| 10 | 0.0086 | `has_flavor_development` | +0.06 | 551/687 |

The seven highest-importance features are **all continuous numerics** —
three engineered (`avg_words_per_step`, `num_ingredients`, `num_steps`)
and four from the original nutrition columns. Directly below them, **all
six engineered `has_*` keyword features place inside the top 20**
(ranks 9, 10, 12, 14, 17 and 19) — above roughly 670 of the 674
editorial tags, despite sitting in the bottom third of the LR
|coefficient| ranking (ranks 512–642, §3.1). The remaining binary tags
follow at ~0.005–0.01 importance each. This is the opposite of the
Logistic Regression picture, where the tags dominated the top of the
coefficient ranking and the continuous numerics either flat-lined
(`num_steps`, `avg_words_per_step`) or sat in the mid-rank.

Mechanism: L2 regularization penalises coefficient *magnitude*, which
shrinks correlated features together (each splits a fraction of the
true signal and absorbs half of L2's penalty). Tree ensembles do not
have a coefficient-magnitude penalty — at each split point, the tree
picks the single feature that maximally reduces impurity, then ignores
the rest. Continuous numerics with broad value ranges (`sodium`,
`avg_words_per_step`) provide far more split points than a 0/1 binary
tag does, so they win at the top of the tree where the biggest impurity
reductions live. Binary tags survive in the leaves, where their fine
resolution refines the prediction further.

The generalisable lesson: **feature utility is not a property of the
feature alone, it is a property of the (feature × model class) pair.**
The same engineered features that appeared to fail in the linear
analysis are the structural backbone of the non-linear analysis.

### 3.6 MLPClassifier: From Catastrophic Overfit to Validation-Stopped Recovery

The MLP `(128, 64)` was run in two configurations to isolate the effect
of regularisation. The first configuration (no early stopping) overfit
decisively; the second (validation-based early stopping) recovered past
the Logistic Regression baseline. Both runs used identical
architecture, random seed, learning rate, and feature matrix — only
the stopping rule differed, and both are reproducible from the same
script (`python train_mlp.py --no-early-stopping` writes the overfit
run to `results/mlp_overfit/`, and the early-stopping run reads it from
there for the live ablation below — no hardcoded prior numbers).

**Run 1 — unregularised baseline (no early stopping):**

| Property | Value |
|----------|------:|
| Epochs trained | 53 |
| Final training loss | **0.0241** (essentially zero) |
| Test accuracy (Advanced) | 0.5728 |
| Test F1 (Advanced) | 0.5936 |

Training loss collapsed monotonically toward zero — the 100k-parameter
network near-perfectly memorised the 14,578-recipe training split.
Test accuracy lagged this training near-perfection by ~40 percentage
points and fell *far below* the linear baseline (0.5728 vs 0.6011).
Classic high-variance regime: capacity in excess of dataset
informativeness, with no regularisation absorbing the excess.

**Run 2 — early stopping (`early_stopping=True`, `validation_fraction=
0.15`, `n_iter_no_change=10`):**

| Property | Value |
|----------|------:|
| Total epochs run | 18 |
| Best validation epoch (restored) | **7** |
| Best validation accuracy | 0.6209 |
| Training loss at best epoch | 0.5861 |
| Training loss at final epoch | 0.3357 |
| Test accuracy (Advanced) | **0.6126** |
| Test F1 (Advanced) | **0.6632** |

Sklearn held out 15% of `X_train` as an internal validation set
(independent of our final test split — leakage-free), tracked
validation accuracy per epoch, and stopped after 10 consecutive epochs
without improvement (patience window). The best validation accuracy of
0.6209 occurred at epoch **7**; sklearn restored the network's weights
to that epoch on `fit()` completion. Training continued until epoch 18
purely to confirm the plateau before stopping.

The two curves are plotted on shared axes in
[`mlp/loss_curve.png`](mlp/loss_curve.png) with a vertical dashed line
at the restored epoch. The diagnostic is textbook: training loss
descends smoothly from 0.68 to 0.34 across the 18 epochs, while
validation error flattens around ~0.38–0.40 and stops improving after
epoch 7. The widening gap between the two curves is the overfitting
signature; the dashed line marks the moment the network's
generalisation peaks.

**Did early stopping improve test performance?** Yes, dramatically, on
both metrics:

| Metric | Overfit (Run 1) | Early-stopped (Run 2) | Δ |
|--------|----------------|----------------------|---|
| Test accuracy | 0.5728 | **0.6126** | **+0.0398** |
| Test F1 | 0.5936 | **0.6632** | **+0.0696** |

Run 2 does not merely close the gap with the Logistic Regression
baseline (Acc 0.6011, F1 0.6405) — it clears it on both metrics
(+1.15 pp Acc, +2.27 pp F1, both above the 1 pp meaningfulness
threshold). The same architecture sits 2.8 pp *below* LR or 1.2 pp
*above* it depending solely on the stopping rule. Random Forest
remains the overall champion (0.6239 / 0.6753), but the linear ceiling
is now exceeded by both non-linear models.

One secondary effect deserves comment: the engineered features' *Δ F1*
flips sign between the two regimes. The overfit run gains F1 from the
Advanced matrix (+0.0107), while the early-stopped run loses a little
(−0.0037); on accuracy the early-stopped Δ is clearly positive
(+0.0154). Mechanism: an unregularised model uses the extra 9 features
as additional material to memorise with; a regularised model converts
them into real accuracy instead, while its F1 optimum shifts slightly
toward the tag-only solution. The interpretation is that Δ-comparisons
are only meaningful once the model is properly constrained — the same
features can look F1-helpful purely as overfitting fuel.

---

## 4. Lessons Learned

This project's most transferable lessons are about the obstacles that
mattered, not the models that were tried. In order of impact:

1. **Outliers in the input space were a higher-impact problem than
   model choice.** A small number of scraping artifacts in the
   nutrition columns compressed the standard-scaled feature space to
   the point that the test `fat` column had an effective standard
   deviation of 0.034. No downstream tuning of model hyperparameters
   would have corrected this; the only fix was swapping
   `StandardScaler` for `RobustScaler` and pairing it with median
   imputation. **Distributional pathology in the input space must be
   diagnosed before tuning the model.**

2. **Optimizer convergence is a correctness property, not a performance
   knob.** Logistic Regression's `lbfgs` solver was emitting a warning
   that could easily have been treated as cosmetic. The actual
   consequence was that per-recipe probability estimates swung by up
   to 99 percentage points between runs of the same model on the same
   data, invalidating any per-recipe interpretation. Aggregate metrics
   (accuracy, F1) absorbed the instability and looked fine. **The
   fact that summary metrics are stable does not imply that the model
   is converged.**

3. **Feature utility is model-dependent.** The engineered continuous
   numerics appeared near-useless under L2 Logistic Regression (ranks
   597, 642 / 687 on |coef|) and dominant under Random Forest (top-7
   by impurity reduction). The cause is structural: L2 penalises
   coefficient magnitude and therefore splits credit among collinear
   features, whereas trees pick a single best splitter per node and
   are invariant to that collinearity. **Conclusions about feature
   value drawn from a single model class can be exactly inverted by
   another.** When a feature looks useless, run the diagnostic against
   a different model family before discarding it.

4. **F1 alone is an unsafe model-selection criterion on near-balanced
   binary tasks.** The F1-maximising threshold (*t* = 0.30, selected
   on the validation split) labelled 90.5% of test recipes as Hit,
   with a balance ratio of 19.4 — extreme asymmetry concealed by the
   harmonic mean of precision and recall. The confusion matrix and
   balance ratio must be inspected alongside F1. A second, related
   lesson was applied in the same place: the threshold itself must be
   *selected* on data the final evaluation never sees (§3.3) — an
   earlier draft selected it on the test sweep directly.

5. **K-Nearest-Neighbors failed because PCA collapsed the search
   space to 1 dimension — and the fix is diagnostic-driven, not
   topological.** The original `train_pca_knn.py` scored 0.515 / 0.509
   accuracy on Baseline / Advanced — below the 0.534
   majority-class rate. The instinctive reading is "KNN is poorly
   suited to high-cardinality sparse binary feature spaces because
   Euclidean distance is dominated by noise in the inactive
   dimensions." That intuition is *not wrong* — but it is not what
   actually broke this particular pipeline. Capturing
   `pca.n_components_` after fitting revealed the real cause:

   > **PCA retained exactly *one* component to reach 90% variance,
   > on both feature matrices.** KNN was therefore asked to do
   > similarity search on a 1-D projection of the data — 677 (or 686)
   > of the input columns were discarded before KNN ever saw them.

   **Mechanism.** Even after `RobustScaler` (which centres on the
   median and scales by the IQR), the four nutrition columns
   (`calories`, `protein`, `fat`, `sodium`) retain long-tailed
   residual variance from a handful of clearly-bogus scraping
   artifacts. Their squared deviation from the median dwarfs the
   variance of a typical 0/1 binary tag (~`p·(1-p)` for a typical
   small `p`), so the four nutrition columns together absorbed ≥90%
   of total variance. PCA, by construction, maximises variance per
   component — so it sunk the entire 90% budget into a single
   nutrition-aligned axis and discarded everything else. The 674
   binary tags and 9 engineered culinary features contributed
   essentially nothing to the retained dimensions.

   **Algorithmic fix.** This diagnostic directly motivated
   [`train_pca_knn_improved.py`](train_pca_knn_improved.py). A
   `ColumnTransformer` explicitly drops the four nutrition columns
   *before* PCA, forcing the projection to evaluate the binary tag
   matrix (and, on Advanced, the engineered culinary features) on
   their own variance budget. The recovery is dramatic on
   dimensionality and meaningful on accuracy:

   | Matrix | PCA components retained | Test accuracy | Test F1 |
   |--------|-----------------------:|--------------:|--------:|
   | Baseline (original) | **1** | 0.5147 | 0.5565 |
   | Baseline (improved) | **203** | **0.5597** | **0.5804** |
   | Advanced (original) | **1** | 0.5092 | 0.5508 |
   | Advanced (improved) | **178** | **0.5726** | **0.6023** |

   The search space expanded by **203×** / **178×**, and accuracy
   recovered by **+4.50 pp / +6.34 pp**. The improved KNN now sits
   in the same ballpark as Perceptron (≈0.55 Acc), no longer
   degenerate. The fix is **partial, not total**: KNN is still well
   below the linear baseline (LR Advanced = 0.6011) and far below
   Random Forest (0.6239). The textbook curse-of-dimensionality on
   sparse-binary data is real and explains the *residual* gap; but
   the catastrophic 1-component collapse was the dominant failure
   mode, and identifying it required reading
   `pca.n_components_` from the fitted model rather than reasoning
   about KNN's distance metric in the abstract.

   A subtler secondary observation falls out of the same table: the
   improved PCA retains **fewer** components on Advanced (178) than
   on Baseline (203), even though Advanced has 9 more input columns.
   That is because the three engineered culinary numerics
   (`num_steps`, `num_ingredients`, `avg_words_per_step`) are
   continuous robust-scaled features with wider variance than any
   0/1 tag, so PCA reallocates some component budget back toward
   them — recreating the original problem in miniature. A *fully*
   fixed pipeline would drop those three numerics too (or, better,
   scale all continuous features into the [0, 1] band of the
   binaries before PCA). This is left as a future-work
   observation — the current improved model is enough to establish
   that the original failure was algorithmic, not topological.

   **The generalisable lesson.** When a distance- or projection-based
   model performs poorly, capture the fit's structural diagnostics
   (`n_components_`, explained variance ratios, condition numbers)
   *before* reasoning about the model from theory alone. The
   theoretical objection ("Euclidean distance is dominated by noise
   in sparse binary spaces") was true but secondary; the real cause
   was a one-line diagnostic that the original pipeline never
   printed.

6. **Model capacity is a liability without matching regularisation,
   and validation-based early stopping is the cheapest fix.** The
   MLPClassifier `(128, 64)` initially drove its training loss to
   0.0241 (effectively zero) while its test accuracy fell *far below*
   the Logistic Regression baseline — approximately 100,000 parameters
   fit on 14,578 samples, with no dropout, no weight decay, and only
   loss-plateau stopping. Adding sklearn's validation-based early
   stopping (`early_stopping=True`, `validation_fraction=0.15`,
   `n_iter_no_change=10`) cut the epoch count from 53 to 18, restored
   the weights to the best validation epoch (epoch 7 of 18), and
   improved the held-out test set by **+3.98 pp Acc and +6.96 pp F1**
   over the overfit run — enough to clear the linear baseline — at
   zero additional code complexity. The lesson generalises: when an
   over-parameterised model shows the train-vs-test divergence
   signature, the first remedy to reach for is validation-based early
   stopping. It is essentially free and directly addresses the failure
   mode.

7. **Audit that engineered features actually reach the model — an
   all-zero column fails silently and poisons every conclusion built
   on it.** The first version of the pipeline defined six curated
   keyword groups but never injected them into the extractor instance
   the pipeline used; every `has_*` column was identically zero in
   every model run (§2.5.1). Nothing crashed: shapes were right,
   metrics were plausible, and an entire interpretive story ("L2
   zeroed the redundant keyword features") was written about features
   that did not exist. The one-line diagnostic that caught it —
   summing the engineered columns on the train split — is the same
   class of structural check as Lesson #5's `pca.n_components_`. Two
   structural fixes followed: the domain knowledge moved to a single
   constant consumed by a factory (`make_culinary_extractor`), and the
   pipeline now raises at build time if any engineered binary column
   is all-zero. **A feature pipeline should fail loud when a feature
   is degenerate, because models will happily train around it and
   report nothing.**

---

## 5. Final Results

### 5.1 Comparative Model Performance (Phase 2)

Seven classifiers, each trained independently on the Baseline matrix
(678 cols: nutrition + tags only) and the Advanced matrix (687 cols:
Baseline + 9 engineered culinary features), evaluated on a held-out
3,645-recipe test set (53.4% Hit, 46.6% Miss).

| Model | Acc (Baseline) | Acc (Advanced) | Δ Acc | F1 (Baseline) | F1 (Advanced) | Δ F1 |
|-------|----------------|----------------|-------|---------------|---------------|------|
| Perceptron | 0.5495 | 0.5435 | −0.0060 | 0.5726 | 0.5667 | −0.0060 |
| Logistic Regression | 0.6005 | 0.6011 | +0.0005 | 0.6452 | 0.6405 | −0.0048 |
| AdaBoost (n=100) | 0.5948 | 0.5967 | +0.0019 | 0.6621 | 0.6518 | −0.0103 |
| PCA(0.90) + KNN | 0.5147 | 0.5092 | −0.0055 | 0.5565 | 0.5508 | −0.0057 |
| PCA(0.90) + KNN (Improved — drop nutrition) | 0.5597 | 0.5726 | +0.0129 | 0.5804 | 0.6023 | +0.0220 |
| **Random Forest (n=200)** | **0.6041** | **0.6239** | **+0.0198** | **0.6565** | **0.6753** | **+0.0188** |
| MLP (128, 64) + early stopping | 0.5973 | 0.6126 | +0.0154 | 0.6668 | 0.6632 | −0.0037 |

(An eighth diagnostic row exists on disk — the deliberately overfit
MLP baseline in `results/mlp_overfit/` — but it is an ablation
artifact, not a candidate model; see §3.6.)

The **Improved PCA + KNN** row is the diagnostic-driven algorithmic fix
described in §4 Lesson #5: dropping the four nutrition columns before
PCA expanded the retained search space from **1** to **203 / 178**
principal components, recovering **+4.50 pp / +6.34 pp** of accuracy
and **+2.39 pp / +5.15 pp** of F1 versus the unfixed version. The fix
also flips the engineered-feature Δ from negative (−0.0057 F1 on the
original) to clearly positive (+0.0220 F1 on the improved) — KNN can
now actually benefit from the culinary features once it has a
non-degenerate distance metric. The improvement is real but partial:
even fixed, KNN sits ~3 pp below Logistic Regression on Acc, confirming
the residual sparse-binary distance penalty cited in §4 Lesson #5.

**Random Forest** is the champion — highest accuracy, highest F1 in
the Advanced column, and the largest beneficiary of the engineered
features (+1.98 pp Acc). It does not produce probability estimates as
well-calibrated as Logistic Regression (an important caveat for any
downstream threshold-tuning work), but for raw predictive accuracy it
dominates.

**Logistic Regression** remains the model of choice for *interpretation*
(Phases 3–4): its coefficients are signed and directly readable, its
probabilities are calibrated for threshold sweeps, and it is the only
model in this study against which the asymmetry diagnostics in §3.3
have a clean meaning. RF's predictive lead does not displace LR from
this role. Note that for LR the engineered features are essentially
neutral (+0.0005 Acc, −0.0048 F1) — the linear lens cannot exploit
them, which is precisely the §2.7 hypothesis test.

**MLPClassifier (with early stopping)** now clears the linear ceiling
(+0.0115 Acc, +0.0227 F1 vs LR — both above the 1 pp meaningfulness
threshold). This is a dramatic recovery from the unregularised
configuration, which fell below LR by 2.8 pp Acc — see §3.6 for the
full ablation. Together with RF, this places *both* non-linear models
above the linear baseline: the predictive limit of the linear models
is a property of the model class, not of the features. The negative
*Δ F1* (−0.0037) is small and discussed in §3.6.

**AdaBoost** records a competitive F1 (0.6621 Baseline) but exhibits
the strongest class asymmetry (FP-rate 0.575, FN-rate 0.257). Its Δ F1
on the Advanced matrix is negative — the engineered features re-shape
its errors rather than reducing them.

**Perceptron** approaches its Bayes-error rate, then plateaus; it is
the only model that loses accuracy from the extra features
(−0.0060), consistent with a non-converging linear separator on a
slightly harder problem.

**PCA + KNN** is the cautionary tale. The result corroborates the
theoretical expectation that distance-based classifiers degrade in
high-cardinality sparse-binary feature spaces.

### 5.2 Best Operating Points

| Role | Model | Acc | F1 | Notes |
|------|-------|------|------|-------|
| **Best predictive accuracy** | Random Forest | **0.6239** | **0.6753** | Default threshold; probabilities less calibrated than LR |
| **Best interpretable model** | Logistic Regression | 0.6011 | 0.6405 | Signed coefficients, ROC-AUC 0.6491, calibrated `predict_proba` |
| **Balanced LR threshold** | LR @ *t* = 0.55 *(selected on validation)* | 0.6005 | 0.5989 | FP/FN balance = 0.80 on test |
| **Max-F1 LR threshold** | LR @ *t* = 0.30 *(selected on validation)* | 0.5822 | 0.7097 | Balance = 19.4 — the §3.3 "F1 trap"; reported for completeness, not recommended |

### 5.3 Dataset Summary

| Property | Value |
|----------|-------|
| Source files | `epi_r.csv` (CSV, 20,052 rows) + `full_format_recipes.json` (20,130 records) |
| Merge strategy | Composite key (title + nutrition); inner join, deduplicated, 1:1 validated |
| Final usable rows | 18,223 |
| Class balance | Hit 53.4% / Miss 46.6% (no resampling required) |
| Train / Test | 14,578 / 3,645 (80/20, stratified, `random_state=42`) |
| Feature matrix (Baseline) | 678 columns (4 nutrition robust-scaled + 674 binary tags) |
| Feature matrix (Advanced) | 687 columns (Baseline + 3 culinary numerics + 6 culinary binaries) |

**Known label limitation — `rating == 0`.** About 9% of the CSV's
recipes carry a rating of exactly 0.000. In the Epicurious feed this
value conflates "rated terribly" with "not yet rated", and the
`rating ≥ 4.0` binarization folds all of them into the Miss class. Some
fraction of the negative class is therefore recipes nobody disliked —
recipes nobody *rated*. This dilutes the Miss class with label noise
and likely depresses every model's ceiling; a follow-up ablation that
excludes rating-0 rows (or models "unrated" as its own class) is the
single most promising data-side improvement this study did not run.

### 5.4 Generated Visualizations

Every `results/<slug>/` directory contains an annotated
`confusion_matrix.png` and a `roc_curve.png` for the Advanced fit. The
model-specific headline plots (paths relative to `results/`, where this
report lives):

| File | Phase | Content |
|------|-------|---------|
| [`logistic_regression/confusion_matrix.png`](logistic_regression/confusion_matrix.png) | 2–3 | Logistic Regression confusion matrix heatmap (annotated; `analysis.py` and the train script write the same slot) |
| [`logistic_regression/roc_curve.png`](logistic_regression/roc_curve.png) | 2–3 | Logistic Regression ROC curve with AUC = 0.6491 |
| [`random_forest/feature_importance.png`](random_forest/feature_importance.png) | 2 | Random Forest top-20 feature importances (horizontal bar) — emitted by `train_random_forest.py` |
| [`mlp/loss_curve.png`](mlp/loss_curve.png) | 2 | MLPClassifier training loss + validation error across 18 epochs, with dashed vertical line marking the restored best-validation epoch (7) — emitted by `train_mlp.py` |
| [`mlp_overfit/loss_curve.png`](mlp_overfit/loss_curve.png) | 2 | The unregularised baseline's raw training-loss collapse across 53 epochs — emitted by `train_mlp.py --no-early-stopping` |

---

## 6. Reproducing the Pipeline

The project now offers **two interchangeable execution paths** — both
write to the same canonical `results/<slug>/` directories, so a user
can run the headless CLI for CI/automation OR open the notebooks for
interactive exploration. Both surfaces share the same source of truth
in `src/` (see §2.6 and the randomness audit in §2.6.1).

```bash
# === Setup ======================================================================
pip install -r requirements.txt      # pinned versions — part of the repro contract

# === A) Headless CLI path (for CI / reproducibility) ============================
# All seven models can be trained in any order; nothing depends on the others.

python train_perceptron.py
python train_logistic_regression.py
python train_adaboost.py
python train_pca_knn.py
python train_pca_knn_improved.py     # diagnostic-driven fix; see §4 Lesson #5
python train_random_forest.py
python train_mlp.py
python train_mlp.py --no-early-stopping   # optional: the §3.6 overfit baseline
python evaluate_all_results.py        # 7-row summary table

# --- Phase 3 + 4: interpretation, confidence, threshold selection ---
python analysis.py
python advanced_tuning.py


# === B) Jupyter notebook path (for interactive portfolio review) ================
# Open these in JupyterLab / VS Code / Colab. They import from `src/`,
# train inline, render plots inline, and save the same artifacts to
# `results/<slug>/` as the CLI scripts do.

jupyter lab  notebooks/
# Then run:
#   01_Logistic_Regression.ipynb
#   02_Random_Forest.ipynb
#   03_MLP_Neural_Network.ipynb
#   04_Perceptron.ipynb
#   05_AdaBoost.ipynb
#   06_PCA_KNN.ipynb
#   07_PCA_KNN_Improved.ipynb
#   08_Master_Comparison.ipynb       # reads metrics from all 7; trains nothing
```

All `random_state` parameters in the pipeline are seeded from the
single source of truth `src/_constants.py::RANDOM_STATE = 42`
(re-exported through `src.train_utils` for convenience). Every
estimator that accepts `random_state` reads it from there — see §2.6.1.

End-to-end runtime is dominated by `train_mlp.py` (or
`03_MLP_Neural_Network.ipynb`) at ~1–2 minutes; the other six models
and both downstream phases together complete in well under a minute.

Each `results/<slug>/` directory after a full run contains:
- `metrics.json` — canonical metrics payload (schema: `src.train_utils.build_metrics_payload`)
- `predictions_baseline.npy` / `predictions_advanced.npy` — int8 test predictions
- `test_index.npy` — the test split's row index (makes the predictions self-describing)
- `confusion_matrix.png` — annotated heatmap (Advanced fit)
- `roc_curve.png` — ROC curve + AUC (Advanced fit)
- `feature_importance.png` *(Random Forest only)* — top-20 importances bar chart
- `loss_curve.png` *(MLP / MLP-overfit only)* — training loss (+ validation error overlay on the early-stopping run)

Phase 3 also writes to `results/logistic_regression/` (same
`confusion_matrix.png` and `roc_curve.png` slots), so the LR notebook's
plots and the Phase 3 plots stay in lockstep.

### 6.1 Notebook regeneration

The eight notebooks are programmatically built by
[`tools/generate_notebooks.py`](tools/generate_notebooks.py) — running
that script regenerates all of them from the same template. This keeps
the notebook structure DRY and editable in one place; if you want to
change the standard cell shape (e.g., add a new plot to every model's
section), edit the template helpers in that file and re-run.

---

## Appendix: Project File Layout

```
.
├── data/                                  Raw inputs (untouched)
│   ├── epi_r.csv                          (binary tags + nutrition + rating)
│   └── full_format_recipes.json           (raw directions + ingredients)
│
├── src/                                   Foundational python package — imported by
│   │                                      every CLI script and notebook in the repo.
│   ├── __init__.py                        Re-exports the public API.
│   ├── _constants.py                      RANDOM_STATE = 42 — the ONLY binding of the
│   │                                      literal 42 in the codebase (see §2.6.1).
│   ├── data_foundation.py                 Load CSV + JSON, merge, target binarisation,
│   │                                      train/test split, CulinaryFeatureExtractor +
│   │                                      CULINARY_KEYWORDS + make_culinary_extractor().
│   │                                      DATA_DIR is anchored to PROJECT_ROOT so
│   │                                      notebooks/ can import safely.
│   ├── preprocessing.py                   Imputation, RobustScaler, two output matrices,
│   │                                      and the dead-feature guard (§2.5.1).
│   └── train_utils.py                     Shared training surface:
│                                           • RANDOM_STATE re-export (bound in _constants)
│                                           • RESULTS_DIR + model_results_dir(slug)
│                                           • build_metrics_payload() (JSON schema)
│                                           • save_metrics / save_predictions /
│                                             save_test_index / save_figure
│                                           • confusion_matrix_figure / roc_curve_figure
│                                           • fit_and_score / load_preprocessed
│
├── notebooks/                             Portfolio surface — one .ipynb per model
│   ├── 01_Logistic_Regression.ipynb       + a master comparison notebook. Each model
│   ├── 02_Random_Forest.ipynb             notebook imports from src/, trains inline,
│   ├── 03_MLP_Neural_Network.ipynb        renders its plots inline via %matplotlib
│   ├── 04_Perceptron.ipynb                inline, and saves the same artifacts to
│   ├── 05_AdaBoost.ipynb                  results/<slug>/ as the CLI scripts.
│   ├── 06_PCA_KNN.ipynb
│   ├── 07_PCA_KNN_Improved.ipynb
│   └── 08_Master_Comparison.ipynb         Trains nothing. Reads results/<slug>/metrics.json
│                                          across all 7 models and renders a styled
│                                          Pandas summary + the cross-model verdict.
│
├── tools/
│   └── generate_notebooks.py              Regenerates the 8 notebooks from one template.
│
├── results/                               Per-model artifacts — see §2.6
│   ├── perceptron/
│   │   ├── metrics.json                    canonical schema (see src/train_utils.py)
│   │   ├── predictions_baseline.npy        int8 test predictions
│   │   ├── predictions_advanced.npy
│   │   ├── test_index.npy                  the test split's row index
│   │   ├── confusion_matrix.png
│   │   └── roc_curve.png
│   ├── logistic_regression/                (same six files; analysis.py refreshes the
│   │                                        two plots in place)
│   ├── adaboost/                           (same six files)
│   ├── pca_knn/                            (same six files; extras: pca_components_retained = 1)
│   ├── pca_knn_improved/                   (same six files; extras: pca_components_retained = 203/178)
│   ├── random_forest/                      (same six files + feature_importance.png)
│   ├── mlp/                                (same six files + loss_curve.png; extras:
│   │                                        early-stopping diagnostics)
│   └── mlp_overfit/                        (the §3.6 ablation baseline, written by
│                                            train_mlp.py --no-early-stopping)
│
├── legacy/                                The dataset author's original scraper and
│   ├── recipe.py                          one-hot helpers. Kept for provenance only —
│   └── utils.py                           imported by nothing (see legacy/README.md).
│
├── train_perceptron.py                    Phase 2 per-model CLI entry points
├── train_logistic_regression.py           (seven independent scripts; each imports
├── train_adaboost.py                      from src, writes to results/<slug>/, can be
├── train_pca_knn.py                       run in any order — see §2.6)
├── train_pca_knn_improved.py              (diagnostic-driven fix — drops nutrition
│                                          before PCA; see §4 Lesson #5)
├── train_random_forest.py
├── train_mlp.py                           (--no-early-stopping trains the §3.6 baseline)
├── evaluate_all_results.py                Pure-read aggregator (reads results/<slug>/
│                                          metrics.json, prints the 7-row summary +
│                                          LR-vs-non-linear cross-model verdict)
│
├── analysis.py                            LR interpretation, confidence, plots (writes
│                                          to results/logistic_regression/)
├── advanced_tuning.py                     Threshold selection on a validation split +
│                                          top-20 features
│
├── requirements.txt                       Pinned dependency versions (repro contract)
└── README.md                              The repo-level quick-start (this report is
                                           results/research.md)
```
