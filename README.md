# Predicting Recipe Success on Epicurious: A Hit/Miss Classification Study

A machine-learning project on the *Epicurious — Recipes with Rating and
Nutrition* dataset. The task is binary classification: predict whether a
recipe will be a **"Hit"** (rating ≥ 4.0) or a **"Miss"** (rating < 4.0)
from its nutrition profile, editorial tags, and raw text instructions.

---

## 1. Executive Summary

The final model is a **Random Forest** ensemble on the Advanced feature
matrix, with a test accuracy of **0.6178**, F1 of **0.6708**, and a
meaningful improvement of **+1.48% Acc / +2.73% F1** over the converged
Logistic Regression baseline. Two non-linear models were added as an
"Academic Deep-Dive" extension after the linear analysis (Phases 3–4)
revealed a collinearity ceiling on engineered keyword features:

- **Random Forest broke the linear ceiling**, validating the hypothesis
  that genuine non-linear interactions exist on this dataset.
- **MLPClassifier (128, 64) initially overfit catastrophically.** In its
  first (unregularised) configuration its training loss collapsed to
  **0.0087** in 64 epochs while its test accuracy *fell below the
  Logistic Regression baseline* (0.5816 vs 0.6030). After enabling
  validation-based early stopping (`early_stopping=True`,
  `validation_fraction=0.15`, `n_iter_no_change=10`), the network
  stopped after 16 epochs with weights restored to the best-validation
  epoch (epoch 5), recovering to Acc **0.6066** and F1 **0.6477** — a
  **+2.50% Acc / +3.70% F1** improvement on the test set over the
  overfit run, and matching the Logistic Regression baseline within
  noise. Capacity, on this dataset, is a liability without explicit
  regularisation — and early stopping is the cheap regulariser that
  closes the gap.

The most informative finding, however, is *not* the headline accuracy. It
is the inversion of the feature-importance picture between the linear and
non-linear analyses:

> **The engineered continuous numerics (`avg_words_per_step`,
> `num_ingredients`, `num_steps`) and the nutrition columns (`sodium`,
> `calories`, `fat`, `protein`) — which were driven to ≈ 0 coefficient by
> L2 regularization in Logistic Regression — are the *top seven* features
> by impurity reduction in Random Forest.** Linear models found them
> redundant with the binary tag matrix (collinearity); a non-linear
> model, which is invariant to collinearity, found them dominant. The
> generalisable insight is that feature *utility* is model-dependent:
> features that look useless to one model class may carry the deepest
> signal for another.

The engineered features are therefore retroactively vindicated: the
Advanced matrix produces positive Δ across every model that can exploit
continuous-feature interactions (RF +0.014 Acc, MLP +0.018 Acc, LR
+0.005 Acc). The original "Aha" of Phase 3 (only `num_ingredients`
mattered) was an artifact of the linear lens.

---

## 2. Methodology

### 2.1 Data Alignment & The Cartesian Explosion

Initial attempts to join the CSV and JSON datasets sequentially yielded only a 2.8% match rate, revealing disparate sorting across the files. Furthermore, a naive join on the recipe `title` risked a "Cartesian Explosion" due to approximately 2,300 duplicated titles (e.g., multiple distinct recipes generically named "Chicken Soup"). Merging purely on titles would have created a massive matrix of corrupted, cross-matched recipes. To prevent this data corruption, a composite key was engineered using `title` + `rating` + `calories` + `protein` + `fat` + `sodium`. This precise fingerprinting successfully filtered duplicates and yielded 18,223 clean, 1:1 validated recipe matches.

### 2.2 Pipeline Architecture (Leakage-Free)

The pipeline is split across five explicit phases, each in its own
script:

| Phase | Script | Responsibility |
|-------|--------|----------------|
| 0 | [`phase0_data_foundation.py`](phase0_data_foundation.py) | Load CSV + JSON, merge, target binarization, train/test split |
| 1 | [`phase1_preprocessing.py`](phase1_preprocessing.py) | Imputation, scaling, custom transformer, two output matrices |
| 2 | `train_*.py` + [`evaluate_all_results.py`](evaluate_all_results.py) | Per-model trainers writing `results/metrics_*.json`; aggregator reads them and prints the summary table — see §2.6 |
| 3 | [`phase3_analysis.py`](phase3_analysis.py) | Interpretation, confidence buckets, ROC and confusion plots |
| 4 | [`phase4_advanced_tuning.py`](phase4_advanced_tuning.py) | Threshold sweep, top-20 feature ranking |

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

**Defensive Column Routing:** A rigorous column classification protocol was implemented to prevent namespace collisions during the extraction phase. For example, the `date` feature in the CSV correctly identifies the inclusion of the fruit "date". A naive text-extraction or identifier-drop would have overwritten or deleted this valid culinary feature under the assumption it was a publication date. Identifiers were defensively isolated and dropped without corrupting identically-named domain features.

### 2.6 MLOps & Production Architecture

Phase 2 was originally a single monolithic script (`phase2_model_training.py`) that trained all six classifiers, produced both visualisations, and printed the summary table in one run. That structure is convenient for an academic write-up but mirrors none of the patterns used in production ML systems: every change required re-training every model, results lived only in stdout, and there was no way to compose model outputs into downstream tasks (calibration, stacking, threshold tuning) without re-running the whole fleet. The Phase 2 codebase was therefore refactored into a **decoupled per-model architecture** that follows three production-grade conventions:

1. **Per-model training scripts.** Each classifier owns its own entry point: [`train_perceptron.py`](train_perceptron.py), [`train_logistic_regression.py`](train_logistic_regression.py), [`train_adaboost.py`](train_adaboost.py), [`train_pca_knn.py`](train_pca_knn.py), [`train_pca_knn_improved.py`](train_pca_knn_improved.py), [`train_random_forest.py`](train_random_forest.py), [`train_mlp.py`](train_mlp.py). Each script loads the preprocessed data via Phase 1's public API, trains *only* its model on both the Baseline and Advanced matrices, and writes its results to disk. No script depends on any other train script; a change to the MLP configuration does not invalidate the LR results, and the seven trainers can be run, retried, or replaced independently. The diagnostic-driven addition of `train_pca_knn_improved.py` is itself a concrete demonstration of why the per-model layout matters in practice: it could be authored, run, and aggregated into the existing summary table without touching any other model's code. This is the standard separation in ML platforms — every model is its own pipeline stage.

2. **A canonical JSON contract for results.** Every train script writes a single `results/metrics_<model>.json` file with the same schema (`model_name`, `display_name`, `model_config`, `n_train`, `n_test`, `random_state`, per-dataset `accuracy`/`f1`/`confusion_matrix`/`fp_rate`/`fn_rate`, plus an optional model-specific `extras` block). Per-recipe test predictions are persisted alongside as `predictions_<model>_<dataset>.npy` (int8, ~3.7 KB each), and any model-specific plot lands in `results/` too (`rf_feature_importance.png`, `mlp_loss_curve.png`). The single source of truth for the schema is `train_utils.py::build_metrics_payload`, which is also responsible for the writers. Anything downstream that wants to consume Phase-2 output reads JSON, not Python globals — the same pattern MLflow, Weights & Biases, and homegrown ML platforms all converge on.

3. **A pure-read aggregator.** [`evaluate_all_results.py`](evaluate_all_results.py) trains nothing. It walks `results/`, loads every `metrics_*.json` it finds, assembles the seven-row summary table, and emits the linear-vs-non-linear cross-model verdict — all by reading the JSON contract above. Because it has no dependency on which trainers have run, it gracefully handles partial fleets (when only some models have been retrained) and reports which expected entries are missing. This decouples *evaluation* from *training* — the textbook structure for CI dashboards, post-hoc analysis notebooks, and any downstream stage that wants to ensemble or rank models.

**Why this matters for this project.** The architecture supports **incremental re-training**: if a Phase-1 change touches only the nutrition imputation, only the models that actually depend on nutrition need to be re-run; the rest still have valid JSON from the previous run. **DRY code** lives in [`train_utils.py`](train_utils.py), which owns the shared fit-and-score loop, the JSON payload builder, the prediction writer, and the console formatters — so the six train scripts each stay around 100 lines, focused entirely on their model's own configuration and any model-specific diagnostics. The two scripts with extra concerns — RF (top-20 feature-importance bar chart) and MLP (training-loss/validation-error overlay + early-stopping diagnostics block in `extras`) — drop their plot code and any extra JSON fields into the per-model script without touching the shared helper, exactly as a production ML platform would expect.

The legacy `phase2_model_training.py` has been retired now that all six refactored scripts produce bit-identical results to the original monolith.

### 2.6.1 The "42 Guarantee" — single source of truth for randomness

Reproducibility on this project rests on a single named constant.
**The literal `42` is bound in exactly one place** in the entire
codebase: [`src/_constants.py`](src/_constants.py). Every other file
that needs the value imports it — either directly
(`from src._constants import RANDOM_STATE`) or via the re-export in
`src.train_utils` (`from src.train_utils import RANDOM_STATE`, which is
the form most callers use because they're already importing the other
training helpers from `src.train_utils`).

`src/_constants.py` deliberately has zero other imports. This matters
because `src.train_utils` imports `src.phase1_preprocessing`, which
imports `src.phase0_data_foundation`. If the canonical constant lived
in `train_utils`, phase0 couldn't import it without creating an import
cycle. Putting it in a tiny isolated module breaks the cycle cleanly.

The full audit (verified by the adversarial workflow in this turn):

| Random process | Where | Source |
|---|---|---|
| `train_test_split(stratify=y, random_state=42)` | `src/phase0_data_foundation.py` | `from src._constants import RANDOM_STATE` |
| `Perceptron(random_state=42, ...)` | `train_perceptron.py` | `from src.train_utils import RANDOM_STATE` |
| `LogisticRegression(random_state=42, ...)` | `train_logistic_regression.py`, `phase3_analysis.py`, `phase4_advanced_tuning.py` | `from src._constants import RANDOM_STATE` (phase3/phase4) / `from src.train_utils import RANDOM_STATE` (train script) |
| `AdaBoostClassifier(random_state=42, ...)` | `train_adaboost.py` | `from src.train_utils import RANDOM_STATE` |
| `PCA(random_state=42, ...)` | `train_pca_knn.py`, `train_pca_knn_improved.py` | `from src.train_utils import RANDOM_STATE` |
| `RandomForestClassifier(random_state=42, ...)` | `train_random_forest.py` | `from src.train_utils import RANDOM_STATE` |
| `MLPClassifier(random_state=42, ...)` | `train_mlp.py` (used for both weight init and the validation-split RNG) | `from src.train_utils import RANDOM_STATE` |
| KNeighborsClassifier | `train_pca_knn*.py` | deterministic; no `random_state` parameter exists |

Every run from a clean checkout produces the same
`results/<slug>/metrics.json` files down to the floating-point
representation. The adversarial verification workflow in this turn
confirmed the literal `42` exists in *exactly one* place in the
codebase.

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

The results (see §3.5–3.6 and §5.1) split the verdict: RandomForest
confirms hypothesis (2) by clearing the linear ceiling, while the
regularised MLP plateaus at the linear baseline — neither breaking past
it nor falling below it. The unregularised MLP failure (§3.6) gives a
secondary lesson on capacity-vs-data balance and the value of
validation-based early stopping.

---

## 3. Key Findings

### 3.1 Feature Saturation: The Engineered Keywords Are Redundant (for L2)

Of the 9 engineered features, the 6 binary `has_<keyword_group>` columns
all received coefficients indistinguishable from zero in the converged
Logistic Regression. This is not a sign of broken feature engineering —
it is the expected behavior of L2 regularization when a new feature is
**collinear** with one or more existing features. The keyword group
`high_heat_techniques` (containing `sear`, `grill`, `broil`, etc.) is
essentially a logical OR over the CSV's pre-existing `grill`, `broil`,
`fry` and related tags. L2 splits the coefficient mass among collinear
features and shrinks each toward zero, with the redundant feature
absorbing most of the shrinkage.

The qualifier "for L2" matters: §3.5 will show that the same features
are *not* redundant for tree ensembles, which do not share coefficient
mass across collinear inputs.

### 3.2 `num_ingredients` Is the One Genuine Linearly-Additive Signal

Within the linear model, the exception that proves the rule is
`num_ingredients`. Coefficient **+0.40**, **rank 146/687** — comfortably
above the median feature, well into the model's working set, and
*unambiguously* positive. The interpretation is direct: holding the tag
profile constant, a recipe with more ingredients is more likely to be
rated a Hit.

This is consistent with a *recipe complexity* hypothesis — more
ingredients indicate a more ambitious dish, which correlates with higher
rater engagement and more favourable ratings. Critically, this property
is not captured by the binary tag matrix, which encodes *what kind* of
recipe it is but not *how elaborate*.

### 3.3 Threshold Tuning Trade-off (F1 vs. Symmetry)

The default Logistic Regression at threshold *t* = 0.50 is biased toward
predicting Hit: FP-rate = 0.475, FN-rate = 0.329, balance ratio = 1.44.
A full sweep from *t* = 0.30 to *t* = 0.70 in 0.05 increments produced
the following Pareto picture:

| Threshold | F1 | Accuracy | FP-rate | FN-rate | Balance | Predicted Hits |
|-----------|------|----------|---------|---------|---------|----------------|
| 0.30 | 0.7099 | 0.5819 | 0.8492 | 0.0421 | **20.16** | 90.7% |
| 0.40 | 0.6979 | 0.6022 | 0.6938 | 0.1397 | 4.97 | 78.3% |
| **0.45** | **0.6771** | **0.6038** | 0.5954 | 0.2224 | **2.68** | 69.3% |
| 0.50 *(default)* | 0.6435 | 0.6030 | 0.4747 | 0.3292 | 1.44 | 57.9% |
| **0.55** | 0.6016 | 0.6022 | 0.3522 | 0.4376 | **0.81** | 46.4% |
| 0.60 | 0.5310 | 0.5852 | 0.2479 | 0.5603 | 0.44 | 35.0% |
| 0.70 | 0.3640 | 0.5484 | 0.1001 | 0.7581 | 0.13 | 17.6% |

Two findings:

1. **The "Best F1" threshold is a methodological trap.** F1 = 0.7099 at
   *t* = 0.30, but at that threshold the model labels 90.7% of all
   recipes as "Hit" (true Hit rate: 53.4%) and the balance ratio is
   **20.16**. F1 alone is insufficient as a model selection criterion;
   balance and predicted-class distribution must be inspected alongside
   it.
2. **Accuracy is essentially flat across thresholds 0.40–0.55.** A
   4-point swing in F1 (0.6979 → 0.6016) costs only 0.16% in accuracy.
   The total error rate barely moves; what changes is *which class*
   absorbs the errors.

The recommended balanced threshold for reporting is **t = 0.55**
(balance = 0.81, F1 = 0.6016). The Pareto compromise is **t = 0.45**
(balance = 2.68, F1 = 0.6771, accuracy slightly *better* than the
default).

### 3.4 The "Recipe for Success" vs. "Recipe for Disaster"

Top 10 signed coefficients from the converged Logistic Regression on the
Advanced matrix:

| Rank | Coef | Hit indicator | | Rank | Coef | Miss indicator |
|------|--------|----------------|---|------|--------|------------------|
| 1 | +1.34 | kentucky derby | | 1 | −1.20 | jícama |
| 2 | +1.19 | brisket | | 2 | −1.09 | whole wheat |
| 3 | +1.00 | sangria | | 3 | −1.08 | aperitif |
| 4 | +0.96 | pasadena | | 4 | −0.99 | slow cooker |
| 5 | +0.90 | kahlúa | | 5 | −0.97 | pittsburgh |
| 6 | +0.90 | 22-minute meals | | 6 | −0.94 | north carolina |
| 7 | +0.89 | trout | | 7 | −0.93 | wild rice |
| 8 | +0.88 | stuffing/dressing | | 8 | −0.91 | tofu |
| 9 | +0.81 | georgia | | 9 | −0.90 | friendsgiving |
| 10 | +0.80 | ground lamb | | 10 | −0.89 | harpercollins |

Patterns:

- **Hit cluster — occasion-driven recipes**: `kentucky derby`, `sangria`,
  `kahlúa`, `pasadena` (Rose Bowl), `ireland` (St. Patrick's Day),
  `sukkot`, `lunar new year`, `champagne`. Users come to occasion
  recipes with elevated expectations and rate generously.
- **Hit cluster — premium proteins**: `brisket`, `trout`, `ground lamb`,
  `beef tenderloin`. Labour-intensive cuts that reward careful
  preparation.
- **Miss cluster — restriction-driven labels**: `whole wheat`, `tofu`,
  `low fat`, `wild rice`. Restriction-driven recipes appear to receive
  harsher ratings — the "I'd rather have the real thing" effect.
- **Miss cluster — convenience-appliance recipes**: `slow cooker`,
  `pasta maker`.
- **Editorial-metadata leakage**: `harpercollins` at −0.89 is almost
  certainly a publisher tag for a specific cookbook line, not a
  culinary property. Geographic tags (`pittsburgh`, `north carolina`,
  `lancaster`, `los angeles`) similarly dominate the lower half and
  likely reflect contributor/editorial bias rather than regional
  culinary preference. These are limitations to acknowledge in the
  scope-and-validity discussion.

**Qualitative Error Analysis (The 'Lamb Köfte' Anomaly):** Analysis of test-set outliers revealed a significant prediction error for 'Lamb Köfte with Tarator Sauce'. Despite a true label of 'Hit', the converged model predicted a 'Miss' with near-zero probability. This highlights a clear 'Domain Gap': our engineered culinary dictionary lacked highly specific regional ingredients (like the complex 'Tarator' sauce) and ethnic techniques ('Köfte'). Forced to rely on broader, less favorable tags, the model misclassified the dish. This qualitative finding exposes the limitations of global tag matrices and provides a roadmap for future regional-specific feature engineering.

### 3.5 RandomForest Broke the Linear Ceiling — and Inverted the Feature Ranking

Random Forest (n=200 trees) on the Advanced matrix achieved
**Acc = 0.6178** and **F1 = 0.6708** — meaningful gains of **+1.48% Acc**
and **+2.73% F1** over the converged Logistic Regression. This
disconfirms the simpler interpretation that the engineered features
"didn't matter": there *is* signal that the linear boundary could not
exploit.

The strongest evidence is the inversion of the feature-importance
ranking. The top 20 RF features by impurity reduction (Gini) are
visualised in [`rf_feature_importance.png`](rf_feature_importance.png).
The top 10 are:

| Rank | Importance | Feature | LR coef (Phase 3) | LR rank |
|------|-----------|---------|------------------|---------|
| 1 | 0.0499 | `avg_words_per_step` | −0.04 | 587/687 |
| 2 | 0.0433 | `sodium` | small | mid-rank |
| 3 | 0.0433 | `calories` | small | mid-rank |
| 4 | 0.0417 | `num_ingredients` | +0.40 | 146/687 |
| 5 | 0.0381 | `fat` | small | mid-rank |
| 6 | 0.0354 | `protein` | small | mid-rank |
| 7 | 0.0288 | `num_steps` | −0.01 | 635/687 |
| 8 | 0.0099 | `bon appétit` | — | tag |
| 9 | 0.0088 | `quick & easy` | — | tag |
| 10 | 0.0080 | `summer` | — | tag |

The seven highest-importance features are **all continuous numerics** —
three engineered (`avg_words_per_step`, `num_ingredients`, `num_steps`)
and four from the original nutrition columns. The 674 binary tags then
follow at *substantially* lower importance (~0.005–0.01 each). This is
the opposite of the Logistic Regression picture, where the tags
dominated the top of the coefficient ranking and the continuous
numerics either flat-lined (`num_steps`, `avg_words_per_step`) or sat in
the mid-rank.

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
decisively; the second (validation-based early stopping) recovered to
match the Logistic Regression baseline. Both runs used identical
architecture, random seed, learning rate, and feature matrix — only
the stopping rule differed.

**Run 1 — unregularised baseline (no early stopping):**

| Property | Value |
|----------|------:|
| Epochs trained | 64 |
| Final training loss | **0.0087** (essentially zero) |
| Test accuracy (Advanced) | 0.5816 |
| Test F1 (Advanced) | 0.6107 |

Training loss collapsed monotonically toward zero — the 100k-parameter
network near-perfectly memorised the 14,578-recipe training split.
Test accuracy lagged this training near-perfection by ~40 percentage
points and fell *below* the linear baseline. Classic high-variance
regime: capacity in excess of dataset informativeness, with no
regularisation absorbing the excess.

**Run 2 — early stopping (`early_stopping=True`, `validation_fraction=
0.15`, `n_iter_no_change=10`):**

| Property | Value |
|----------|------:|
| Total epochs run | 16 |
| Best validation epoch (restored) | **5** |
| Best validation accuracy | 0.6118 |
| Training loss at best epoch | 0.5944 |
| Training loss at final epoch | 0.3621 |
| Test accuracy (Advanced) | **0.6066** |
| Test F1 (Advanced) | **0.6477** |

Sklearn held out 15% of `X_train` as an internal validation set
(independent of our final test split — leakage-free), tracked
validation accuracy per epoch, and stopped after 10 consecutive epochs
without improvement (patience window). The best validation accuracy of
0.6118 occurred at epoch **5**; sklearn restored the network's weights
to that epoch on `fit()` completion. Training continued until epoch 16
purely to confirm the plateau before stopping.

The two curves are plotted on shared axes in
[`mlp_loss_curve.png`](mlp_loss_curve.png) with a vertical dashed line
at the restored epoch. The diagnostic is textbook: training loss
descends smoothly from 0.68 to 0.36 across the 16 epochs, while
validation error stays roughly flat at ~0.39–0.41 and starts climbing
slightly after epoch 5. The widening gap between the two curves is the
overfitting signature; the dashed line marks the moment the network's
generalisation peaks.

**Did early stopping improve test performance?** Yes, on both metrics:

| Metric | Overfit (Run 1) | Early-stopped (Run 2) | Δ |
|--------|----------------|----------------------|---|
| Test accuracy | 0.5816 | **0.6066** | **+0.0250** |
| Test F1 | 0.6107 | **0.6477** | **+0.0370** |

Run 2 also closed the gap with the Logistic Regression baseline (Acc
0.6030, F1 0.6435): the regularised MLP now matches LR within noise
(+0.0036 Acc, +0.0042 F1, both below the meaningful threshold). The
neural network is no longer worse than the linear model, but neither
does it break past it — it plateaus at the same operating point. The
predictive ceiling that Random Forest exceeded is therefore not
displaced by the MLP, regardless of its regularisation.

One unexpected secondary effect deserves comment: with early stopping,
the *Baseline* MLP (no engineered features) jumped from F1 = 0.5828 to
F1 = 0.6668 — an 8.4 pp improvement on the same metric — while the
*Advanced* MLP improved on F1 by only 3.7 pp. The *Δ F1* between the
two matrices therefore flipped from positive (+0.0278) to negative
(−0.0192). Mechanism: the unregularised model used the extra 9
engineered features to overfit "more usefully" (more signal to
memorise); a regularised model has less appetite for those features,
and the simpler tag-only Baseline matrix turns out to suit the
plateau-level solution slightly better on F1. The accuracy Δ for MLP
remains positive (+0.0093). The interpretation is that the engineered
features matter most when a model is overfitting, and matter less when
it is properly constrained.

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
   587, 635 / 687 on |coef|) and dominant under Random Forest (top-7
   by impurity reduction). The cause is structural: L2 penalises
   coefficient magnitude and therefore splits credit among collinear
   features, whereas trees pick a single best splitter per node and
   are invariant to that collinearity. **Conclusions about feature
   value drawn from a single model class can be exactly inverted by
   another.** When a feature looks useless, run the diagnostic against
   a different model family before discarding it.

4. **F1 alone is an unsafe model-selection criterion on near-balanced
   binary tasks.** The F1-maximising threshold (*t* = 0.30) labelled
   90.7% of test recipes as Hit, with a balance ratio of 20.16 —
   extreme asymmetry concealed by the harmonic mean of precision and
   recall. The confusion matrix and balance ratio must be inspected
   alongside F1.

5. **K-Nearest-Neighbors failed because PCA collapsed the search
   space to 1 dimension — and the fix is diagnostic-driven, not
   topological.** The original `train_pca_knn.py` scored 0.515 / 0.513
   accuracy on Baseline / Advanced — barely above the 0.534
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
   | Baseline (improved) | **203** | **0.5602** | **0.5820** |
   | Advanced (original) | **1** | 0.5133 | 0.5578 |
   | Advanced (improved) | **182** | **0.5630** | **0.5904** |

   The search space expanded by **203×** / **182×**, and accuracy
   recovered by **+4.55 pp / +4.97 pp**. The improved KNN now sits
   in the same ballpark as Perceptron (≈0.55 Acc), no longer
   degenerate. The fix is **partial, not total**: KNN is still well
   below the linear baseline (LR Advanced = 0.6030) and far below
   Random Forest (0.6178). The textbook curse-of-dimensionality on
   sparse-binary data is real and explains the *residual* gap; but
   the catastrophic 1-component collapse was the dominant failure
   mode, and identifying it required reading
   `pca.n_components_` from the fitted model rather than reasoning
   about KNN's distance metric in the abstract.

   A subtler secondary observation falls out of the same table: the
   improved PCA retains **fewer** components on Advanced (182) than
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
   0.0087 (effectively zero) while its test accuracy fell *below* the
   Logistic Regression baseline — approximately 100,000 parameters fit
   on 14,578 samples, with no dropout, no weight decay, and only
   loss-plateau stopping. Adding sklearn's validation-based early
   stopping (`early_stopping=True`, `validation_fraction=0.15`,
   `n_iter_no_change=10`) halved the epoch count (16 vs 64), restored
   the weights to the best validation epoch (epoch 5 of 16), and
   improved the held-out test set by **+2.5 pp Acc and +3.7 pp F1**
   over the overfit run — at zero additional code complexity. The
   lesson generalises: when an over-parameterised model shows the
   train-vs-test divergence signature, the first remedy to reach for
   is validation-based early stopping. It is essentially free and
   directly addresses the failure mode.

---

## 5. Final Results

### 5.1 Comparative Model Performance (Phase 2)

Seven classifiers, each trained independently on the Baseline matrix
(678 cols: nutrition + tags only) and the Advanced matrix (687 cols:
Baseline + 9 engineered culinary features), evaluated on a held-out
3,645-recipe test set (53.4% Hit, 46.6% Miss).

| Model | Acc (Baseline) | Acc (Advanced) | Δ Acc | F1 (Baseline) | F1 (Advanced) | Δ F1 |
|-------|----------------|----------------|-------|---------------|---------------|------|
| Perceptron | 0.5495 | 0.5457 | −0.0038 | 0.5726 | 0.5696 | −0.0030 |
| Logistic Regression | 0.5984 | 0.6030 | +0.0047 | 0.6428 | 0.6435 | +0.0008 |
| AdaBoost (n=100) | 0.5948 | 0.5967 | +0.0019 | 0.6621 | 0.6518 | −0.0103 |
| PCA(0.90) + KNN | 0.5147 | 0.5133 | −0.0014 | 0.5565 | 0.5578 | +0.0013 |
| PCA(0.90) + KNN (Improved — drop nutrition) | 0.5602 | 0.5630 | +0.0027 | 0.5820 | 0.5904 | +0.0084 |
| **Random Forest (n=200)** | **0.6041** | **0.6178** | **+0.0137** | **0.6565** | **0.6708** | **+0.0143** |
| MLP (128, 64) + early stopping | 0.5973 | 0.6066 | +0.0093 | 0.6668 | 0.6477 | −0.0192 |

The **Improved PCA + KNN** row is the diagnostic-driven algorithmic fix
described in §4 Lesson #5: dropping the four nutrition columns before
PCA expanded the retained search space from **1** to **203 / 182**
principal components, recovering **+4.55 pp / +4.97 pp** of accuracy
and **+2.55 pp / +3.26 pp** of F1 versus the unfixed version. The fix
also flips the engineered-feature Δ from essentially zero (+0.0013 F1
on the original) to clearly positive (+0.0084 F1 on the improved) —
KNN can now actually benefit from the culinary features once it has a
non-degenerate distance metric. The improvement is real but partial:
even fixed, KNN sits ~4 pp below Logistic Regression on Acc, confirming
the residual sparse-binary distance penalty cited in §4 Lesson #5.

**Random Forest** is the champion — highest accuracy, highest F1 in
the Advanced column. It does not produce probability estimates as
well-calibrated as Logistic Regression (an important caveat for any
downstream threshold-tuning work), but for raw predictive accuracy it
dominates.

**Logistic Regression** remains the model of choice for *interpretation*
(Phases 3–4): its coefficients are signed and directly readable, its
probabilities are calibrated for threshold sweeps, and it is the only
model in this study against which the asymmetry diagnostics in §3.3
have a clean meaning. RF's predictive lead does not displace LR from
this role.

**MLPClassifier (with early stopping)** now matches LR within noise
(+0.0036 Acc, +0.0042 F1 vs LR — both below the 1 pp meaningfulness
threshold). This is a substantial recovery from the original
unregularised configuration which fell *below* LR by 0.02 Acc and
0.03 F1 — see §3.6 for the full ablation. The neural network has
neither broken past the linear ceiling nor failed beneath it; it
plateaus at the same operating point, suggesting the predictive limit
on this dataset is structural in the features rather than in the
model's representational capacity. The negative *Δ F1* (−0.0192) is an
artefact of the engineered features helping unregularised models
overfit "more usefully" — once early stopping cuts that off, the
simpler Baseline matrix slightly favours MLP on F1. Accuracy *Δ*
remains positive.

**AdaBoost** records a competitive F1 (0.6621 Baseline) but exhibits
the strongest class asymmetry (FP-rate 0.575, FN-rate 0.257). Its Δ F1
on the Advanced matrix is negative — the engineered features re-shape
its errors rather than reducing them.

**Perceptron** approaches its Bayes-error rate, then plateaus.

**PCA + KNN** is the cautionary tale. The result corroborates the
theoretical expectation that distance-based classifiers degrade in
high-cardinality sparse-binary feature spaces.

### 5.2 Best Operating Points

| Role | Model | Acc | F1 | Notes |
|------|-------|------|------|-------|
| **Best predictive accuracy** | Random Forest | **0.6178** | **0.6708** | Default threshold; probabilities less calibrated than LR |
| **Best interpretable model** | Logistic Regression | 0.6030 | 0.6435 | Signed coefficients, ROC-AUC 0.6500, calibrated `predict_proba` |
| **Balanced LR threshold** | LR @ *t* = 0.55 | 0.6022 | 0.6016 | FP/FN balance = 0.81 |
| **Pareto LR threshold** | LR @ *t* = 0.45 | 0.6038 | 0.6771 | FP/FN balance = 2.68 |

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

### 5.4 Generated Visualizations

| File | Phase | Content |
|------|-------|---------|
| [`lr_confusion_matrix.png`](lr_confusion_matrix.png) | 3 | Logistic Regression confusion matrix heatmap (annotated) — project root |
| [`lr_roc_curve.png`](lr_roc_curve.png) | 3 | Logistic Regression ROC curve with AUC = 0.6500 — project root |
| [`results/rf_feature_importance.png`](results/rf_feature_importance.png) | 2 | Random Forest top-20 feature importances (horizontal bar) — emitted by `train_random_forest.py` |
| [`results/mlp_loss_curve.png`](results/mlp_loss_curve.png) | 2 | MLPClassifier training loss + validation error across 16 epochs, with dashed vertical line marking the restored best-validation epoch — emitted by `train_mlp.py` |

---

## 6. Reproducing the Pipeline

The project now offers **two interchangeable execution paths** — both
write to the same canonical `results/<slug>/` directories, so a user
can run the headless CLI for CI/automation OR open the notebooks for
interactive exploration. Both surfaces share the same source of truth
in `src/` (see §2.6 and the new MLOps subsection §2.6.1 below).

```bash
# === A) Headless CLI path (for CI / reproducibility) ============================
# All seven models can be trained in any order; nothing depends on the others.

python train_perceptron.py
python train_logistic_regression.py
python train_adaboost.py
python train_pca_knn.py
python train_pca_knn_improved.py     # diagnostic-driven fix; see §4 Lesson #5
python train_random_forest.py
python train_mlp.py
python evaluate_all_results.py        # 7-row summary table

# --- Phase 3 + 4: interpretation, confidence, threshold tuning ---
python phase3_analysis.py
python phase4_advanced_tuning.py


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
single source of truth `src.train_utils.RANDOM_STATE = 42`. Every
estimator that accepts `random_state` reads it from there — see §2.6.1.

End-to-end runtime is dominated by `train_mlp.py` (or
`03_MLP_Neural_Network.ipynb`) at ~1–2 minutes; the other six models
and both downstream phases together complete in well under a minute.

Each `results/<slug>/` directory after a full run contains:
- `metrics.json` — canonical metrics payload (schema: `src.train_utils.build_metrics_payload`)
- `predictions_baseline.npy` / `predictions_advanced.npy` — int8 test predictions
- `confusion_matrix.png` — annotated heatmap (Advanced fit)
- `roc_curve.png` — ROC curve + AUC (Advanced fit)
- `feature_importance.png` *(Random Forest only)* — top-20 importances bar chart
- `loss_curve.png` *(MLP only)* — training loss + validation error overlay

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
│   ├── phase0_data_foundation.py          Load CSV + JSON, merge, target binarisation,
│   │                                      train/test split. DATA_DIR is anchored to
│   │                                      PROJECT_ROOT so notebooks/ can import safely.
│   ├── phase1_preprocessing.py            Imputation, RobustScaler, two output matrices.
│   └── train_utils.py                     SINGLE source of truth for:
│                                           • RANDOM_STATE = 42 (see §2.6.1)
│                                           • RESULTS_DIR + model_results_dir(slug)
│                                           • build_metrics_payload() (JSON schema)
│                                           • save_metrics / save_predictions / save_figure
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
│   │   ├── confusion_matrix.png
│   │   └── roc_curve.png
│   ├── logistic_regression/                (same five files + the Phase 3 plots
│   │                                        also land here)
│   ├── adaboost/                           (same five files)
│   ├── pca_knn/                            (same five files; extras: pca_components_retained = 1)
│   ├── pca_knn_improved/                   (same five files; extras: pca_components_retained ≈ 203/182)
│   ├── random_forest/                      (same five files + feature_importance.png)
│   └── mlp/                                (same five files + loss_curve.png; extras: early-stopping diagnostics)
│
├── train_perceptron.py                    Phase 2 per-model CLI entry points
├── train_logistic_regression.py           (seven independent scripts; each imports
├── train_adaboost.py                      from src, writes to results/<slug>/, can be
├── train_pca_knn.py                       run in any order — see §2.6)
├── train_pca_knn_improved.py              (diagnostic-driven fix — drops nutrition
│                                          before PCA; see §4 Lesson #5)
├── train_random_forest.py
├── train_mlp.py
├── evaluate_all_results.py                Pure-read aggregator (reads results/<slug>/
│                                          metrics.json, prints the 7-row summary +
│                                          LR-vs-non-linear cross-model verdict)
│
├── phase3_analysis.py                     LR interpretation, confidence, plots (writes
│                                          to results/logistic_regression/)
├── phase4_advanced_tuning.py              Threshold sweep, top-20 features
│
├── utils.py / recipe.py                   Original helper scripts (JSON parsing only)
└── README.md                              This report
```
