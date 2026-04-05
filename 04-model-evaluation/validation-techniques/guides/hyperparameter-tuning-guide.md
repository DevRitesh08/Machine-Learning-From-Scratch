# Hyperparameter Tuning

---

## Parameters vs Hyperparameters

| | Parameters | Hyperparameters |
|--|-----------|----------------|
| **What** | Values learned by the model during training | Values set by the developer **before** training begins |
| **Examples** | Weights in linear regression, coefficients in SVM | `n_neighbors` in KNN, `max_depth` in Decision Tree, learning rate |
| **How set** | Optimized automatically by the learning algorithm | Must be tuned manually or via a search strategy |
| **Stored in** | `model.coef_`, `model.intercept_` | `model.get_params()` |

The model cannot learn its own hyperparameters — they must be decided externally. Choosing the wrong values leads to underfitting or overfitting.

---

## Why Hyperparameter Tuning Matters

Without tuning, you are using the algorithm's **default** hyperparameters, which are rarely optimal for your specific dataset. Tuning directly improves:

- **Generalization** — the model performs better on unseen data
- **Model selection** — lets you compare the true best version of each algorithm fairly

The standard approach is to search over a range of hyperparameter combinations, evaluate each using **cross-validation**, and select the combination that achieves the best CV score.

> Training a baseline model first (with default params) gives you a reference point. Any tuned model that doesn't beat the baseline is not worth the complexity.

---

## GridSearchCV — Exhaustive Search

GridSearchCV tries **every possible combination** from a defined parameter grid. For each combination it runs full k-fold cross-validation and records the score.

### How it Works

Given a grid:
```
n_neighbors : [1, 3, 5]
weights     : ['uniform', 'distance']
p           : [1, 2]
```
Total combinations = 3 × 2 × 2 = **12 fits × k folds each**

With `cv=5`, GridSearchCV trains and evaluates the model **60 times** (12 × 5).

### Key Parameters

| Parameter | Purpose |
|-----------|---------|
| `estimator` | The model to tune |
| `param_grid` | Dictionary of hyperparameter names → lists of values to try |
| `scoring` | Metric to optimize (e.g. `'r2'`, `'accuracy'`, `'neg_mean_squared_error'`) |
| `cv` | Cross-validation strategy (pass a `KFold` object or integer) |
| `refit` | If `True`, retrains the best model on the **entire** training set after search — ready for `.predict()` |
| `verbose` | Controls logging; `verbose=2` prints progress for every combination |

### Important Attributes After Fitting

| Attribute | What it Returns |
|-----------|----------------|
| `.best_params_` | Dictionary of the best hyperparameter combination found |
| `.best_score_` | Mean CV score of the best combination |
| `.cv_results_` | Full results for every combination — convertible to a DataFrame for analysis |
| `.best_estimator_` | The refitted model using best params (only if `refit=True`) |

### Reading `cv_results_`

Converting to a DataFrame is essential for a thorough analysis:

```python
pd.DataFrame(gcv.cv_results_)[
    ['param_algorithm', 'param_n_neighbors', 'param_p', 'param_weights', 'mean_test_score']
].sort_values('mean_test_score', ascending=False)
```

> **Interview point:** The best parameters are not always the right choice in production. If a second-best combination is significantly cheaper to compute and scores only marginally lower, it is often the smarter engineering decision. Always inspect `cv_results_`, not just `best_params_`.

### `refit=True` — What it Actually Does

After the search completes, GridSearchCV uses the best hyperparameters to retrain a **fresh model on the entire training dataset** (not just one fold). This final model is then exposed via `.best_estimator_` and is what gets called when you do `gcv.predict(X_new)`.

Without `refit=True`, `.predict()` is not available on the GridSearchCV object.

### Pros and Cons

| Pros | Cons |
|------|------|
| Guaranteed to find the best combination within the defined grid | Computationally expensive — scales multiplicatively with grid size |
| Fully reproducible | Does not explore outside the predefined grid |
| Straightforward to interpret | Large grids become infeasible quickly |

---

## RandomizedSearchCV — Random Sampling

Instead of trying every combination, RandomizedSearchCV **randomly samples** `n_iter` combinations from the parameter space and evaluates each using cross-validation.

### Key Difference from GridSearchCV

```
GridSearchCV     → tries ALL combinations (deterministic)
RandomizedSearchCV → tries n_iter RANDOM combinations (stochastic)
```

### Key Parameters

| Parameter | Purpose |
|-----------|---------|
| `estimator` | Model to tune |
| `param_distributions` | Same dict structure as `param_grid`; can also pass scipy distributions |
| `n_iter` | Number of random combinations to sample (default: 10) |
| `scoring` | Same as GridSearchCV |
| `cv` | Same as GridSearchCV |
| `refit` | Same as GridSearchCV |
| `random_state` | Seed for reproducibility |

### Same Result Attributes

`.best_params_`, `.best_score_`, `.cv_results_`, `.best_estimator_` — all work identically to GridSearchCV.

### When to Use RandomizedSearchCV

- The parameter space is large (many hyperparameters or wide ranges)
- Compute budget is limited
- Some hyperparameters have **continuous ranges** (e.g., learning rate from 0.001 to 1.0) — GridSearchCV cannot handle continuous ranges but RandomizedSearchCV can sample from them using scipy distributions

### Pros and Cons

| Pros | Cons |
|------|------|
| Much faster than GridSearchCV for large grids | Not guaranteed to find the globally best combination |
| Can explore continuous distributions | Results vary with `random_state` |
| Scales well to high-dimensional search spaces | May miss good regions if `n_iter` is too small |

---

## GridSearchCV vs RandomizedSearchCV — Summary

| | GridSearchCV | RandomizedSearchCV |
|-|-------------|-------------------|
| Search strategy | Exhaustive | Random sampling |
| Combinations tried | All (M₁ × M₂ × … × Mₙ) | `n_iter` |
| Continuous ranges | No | Yes (via distributions) |
| Guaranteed optimal | Yes (within the grid) | No |
| Compute cost | High for large grids | Controlled via `n_iter` |
| Reproducibility | Always | With fixed `random_state` |
| Best for | Small, well-defined grids | Large or continuous search spaces |

---

## Bayesian Search — Introduction

Both GridSearchCV and RandomizedSearchCV are **uninformed** — they do not use results from previous evaluations to decide where to search next. Every combination is tried independently.

**Bayesian Optimization** is a smarter strategy:

1. It builds a **probabilistic surrogate model** of the objective function (CV score as a function of hyperparameters)
2. After each evaluation, it **updates** this model based on the new result
3. It uses this model to **decide which combination to try next** — balancing exploration (trying uncertain regions) and exploitation (refining known good regions)

The result: it finds good hyperparameter combinations in **far fewer evaluations** than either grid or random search, because it actively learns from past trials.

> In scikit-learn, Bayesian optimization is available through third-party libraries such as `scikit-optimize` (`BayesSearchCV`) and `optuna`. It is particularly valuable when each model training run is expensive (e.g., deep learning or large datasets).

---

## Interview Quick Reference

| Question | Answer |
|----------|--------|
| What is a hyperparameter? | A configuration value set before training; not learned by the model |
| What does `refit=True` do? | Retrains the best model on the full training set after the search, enabling `.predict()` |
| Why use `cv_results_` instead of just `best_params_`? | To inspect the full performance landscape — a slightly worse combination may be far cheaper to compute |
| When does RandomizedSearchCV outperform GridSearchCV? | Large grids, continuous parameter ranges, or limited compute budget |
| What is the risk of not tuning hyperparameters? | Using defaults that may cause underfitting or overfitting on your specific data |
| What is the key limitation of both grid and random search? | They are uninformed — they don't learn from prior evaluations |
| What does Bayesian optimization do differently? | Builds a surrogate model of the objective and uses it to direct the search intelligently |
| What scoring should you use for regression? | `'r2'` or `'neg_mean_squared_error'` (note the negative sign for MSE) |

---

## Related

- [03-hyperparameter-tuning.ipynb](./03-hyperparameter-tuning.ipynb) — Practical implementation with KNN on Boston Housing dataset
- [01-cross-validation.md](./01-cross-validation.md) — Cross-validation is the backbone of every tuning strategy
