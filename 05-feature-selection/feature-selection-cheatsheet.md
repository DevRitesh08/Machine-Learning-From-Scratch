# Feature Selection — Interview Cheatsheet

> **Why Feature Selection?** Reduces overfitting, improves accuracy, and cuts training time by keeping only the most informative features.

---

## At a Glance

| Category | Method | Best For | Speed |
|----------|--------|----------|-------|
| **Filter** | Variance Threshold | Removing constants / near-constants | ⚡ Fast |
| **Filter** | Correlation Coefficient | Dropping redundant numeric features | ⚡ Fast |
| **Filter** | Chi-Square Test | Categorical → Categorical dependency | ⚡ Fast |
| **Filter** | Mutual Information | Non-linear dependencies | ⚡ Fast |
| **Filter** | ANOVA F-Test | Categorical input → Continuous target | ⚡ Fast |
| **Wrapper** | RFE | Model-guided feature ranking | 🐢 Slow |
| **Wrapper** | Sequential (SFS/SBS) | Finding optimal subset iteratively | 🐢 Slow |
| **Wrapper** | Exhaustive Search | Guaranteed best subset (few features) | 🐌 Very Slow |
| **Embedded** | Lasso (L1) | Sparse, interpretable models | ⚡ Fast |
| **Embedded** | Ridge (L2) | Handling multicollinearity | ⚡ Fast |
| **Embedded** | Elastic Net (L1+L2) | Correlated feature groups | ⚡ Fast |
| **Embedded** | Random Forest Importance | Non-linear importance ranking | 🔶 Medium |

---

## 1 · Filter Methods

> **Key idea:** Rank features using statistical tests *before* any model is trained. Fast, model-agnostic, but ignores feature interactions.

### Variance Threshold
- Drops features with variance below a set value.
- **Use when:** many features are constant or nearly constant.
- `sklearn`: `VarianceThreshold(threshold=0.01)`

### Correlation Coefficient (Pearson)
- Measures linear relationship between feature pairs.
- Remove one of any pair with |r| > 0.8–0.9.
- **Use when:** you suspect redundant numeric features.

### Chi-Square (χ²) Test
- Tests association between two categorical variables.
- **Use when:** categorical features + categorical/binary target.
- `sklearn`: `SelectKBest(chi2, k=10)`

### Mutual Information
- Captures **both linear and non-linear** dependencies.
- Generalisation of correlation; information-theoretic measure.
- **Use when:** relationships may be non-linear.
- `sklearn`: `mutual_info_classif` / `mutual_info_regression`

### ANOVA F-Test
- Compares group means across categories.
- **Use when:** categorical inputs → continuous target.
- `sklearn`: `f_classif` with `SelectKBest`

---

## 2 · Wrapper Methods

> **Key idea:** Use a model's performance to evaluate feature subsets. Accurate but computationally expensive.

### Recursive Feature Elimination (RFE)
- Trains model → removes weakest feature → repeats.
- **Use when:** you want model-driven ranking and can afford the compute.
- `sklearn`: `RFE(estimator, n_features_to_select=k)`

### Sequential Feature Selection (SFS / SBS)
- **Forward (SFS):** starts empty, adds best feature each step.
- **Backward (SBS):** starts full, removes worst feature each step.
- **Use when:** you need an optimal subset and compute budget allows.
- `sklearn`: `SequentialFeatureSelector(estimator, direction='forward')`

### Exhaustive Feature Selection
- Evaluates **every** possible subset (brute-force).
- **Use when:** ≤ 15–20 features *only*; guarantees the global best subset.
- Complexity: $O(2^n)$ — impractical for large feature sets.

---

## 3 · Embedded Methods

> **Key idea:** Feature selection happens *during* model training (built into the algorithm). Balances speed and accuracy.

### Lasso Regression (L1)
- Adds $\lambda \sum |w_i|$ penalty → drives irrelevant weights to **exactly zero**.
- **Use when:** you want automatic feature elimination + a simple model.
- `sklearn`: `Lasso(alpha=0.1)`

### Ridge Regression (L2)
- Adds $\lambda \sum w_i^2$ penalty → shrinks weights but **never zeros** them.
- **Does not** perform feature selection; reduces multicollinearity.
- `sklearn`: `Ridge(alpha=1.0)`

### Elastic Net (L1 + L2)
- Combines Lasso + Ridge: $\lambda_1 \sum |w_i| + \lambda_2 \sum w_i^2$.
- **Use when:** correlated feature groups exist — Lasso alone might arbitrarily pick one; Elastic Net keeps the group.
- `sklearn`: `ElasticNet(alpha=0.1, l1_ratio=0.5)`

### Random Forest Feature Importance
- Mean Decrease in Impurity (MDI) or permutation importance.
- **Use when:** you want a non-linear importance ranking without assumptions.
- `sklearn`: `model.feature_importances_`

---

## Quick Decision Flowchart

```
Start
  │
  ├─ Need fast, model-free filtering? ──► FILTER METHODS
  │     ├─ All numeric? ──► Correlation / Variance Threshold
  │     ├─ Categorical target? ──► Chi-Square / Mutual Information
  │     └─ Continuous target? ──► ANOVA / Mutual Information
  │
  ├─ Want model-driven selection? ──► WRAPPER METHODS
  │     ├─ Few features (≤20)? ──► Exhaustive Search
  │     └─ Many features? ──► RFE or SFS
  │
  └─ Want selection during training? ──► EMBEDDED METHODS
        ├─ Linear model + sparsity? ──► Lasso / Elastic Net
        ├─ Multicollinearity issue? ──► Ridge
        └─ Non-linear model? ──► Random Forest Importance
```

---

## Common Interview Questions

| Question | Key Answer |
|----------|-----------|
| Filter vs Wrapper vs Embedded? | Filter = stats, no model; Wrapper = uses model perf; Embedded = built into training |
| Lasso vs Ridge? | Lasso (L1) zeros out features; Ridge (L2) only shrinks them |
| When does Elastic Net beat Lasso? | When features are correlated — Lasso drops some arbitrarily, Elastic Net keeps the group |
| Chi-Square vs ANOVA? | Chi-Square: categorical→categorical; ANOVA: categorical→continuous |
| RFE vs SFS? | RFE removes worst iteratively; SFS adds/removes one at a time by performance |
| Why not always use Exhaustive? | $O(2^n)$ complexity — infeasible beyond ~20 features |
| Mutual Info vs Correlation? | MI captures non-linear relationships; correlation is linear only |

---

*Last updated: Feb 2026*
