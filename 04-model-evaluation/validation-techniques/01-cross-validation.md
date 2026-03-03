# Cross-Validation: Evaluating Model Performance

Cross-validation is a resampling technique used to evaluate how well a model generalizes to unseen data. It divides the dataset into multiple subsets (folds), trains the model on some folds, and tests it on the remaining fold. By rotating through all folds, we can obtain a more reliable estimate of the model's performance than a single hold-out validation set.

---

## The Hold-Out Approach

The simplest method to evaluate a model is the hold-out approach: split the dataset once into a training set (e.g., 70–80% of data) and a validation set (20–30%). Train the model on the training set and evaluate it on the validation set. This provides an estimate of how well the model will perform on new, unseen data.

### Problems with the Hold-Out Approach

The hold-out method splits data once into a training set and a validation set , and evaluates the model on the validation set. This approach has several significant drawbacks:

**1. High variance in the estimate**

The validation error depends heavily on which specific observations fall into the training set versus the validation set. A single random split can produce an optimistic or pessimistic estimate purely by chance. If the random seed changes, the reported error can shift substantially. This makes the hold-out estimate an unreliable measure of true model performance.

**2. Pessimistic bias (underestimation of model quality)**

Statistical learning methods generally perform worse when trained on fewer observations. When 20–30% of the data is withheld for validation, the model is trained on only 70–80% of available data. The resulting error estimate is systematically higher than what the model would achieve if trained on the full dataset. The hold-out error therefore tends to *overestimate* the true test error of a model trained on all the data.

**3. Wasteful use of data**

A significant portion of the data is removed from training entirely for the duration of model development. For datasets with limited samples — common in medical, scientific, or niche domains — this is a serious constraint. Every observation withheld for validation is an observation the model cannot learn from.

**4. Arbitrariness of the split**

There is no principled basis for which observations go into training versus validation. The split is determined by a random state, not by any property of the data. There is no guarantee the validation set is representative of the full distribution, particularly for small datasets or when class imbalance exists. Two analysts using different random seeds will report different validation errors for the same model.

Cross-validation directly addresses all four problems: it rotates the validation set across the full dataset, uses all observations for both training and evaluation across iterations, and averages over multiple splits to reduce variance and bias in the estimate.

### Why is it used ?

- **Simplicity**: Easy to implement and understand. Just one split and one evaluation.
- **Speed**: Only requires training the model once, so it's computationally efficient.
- **Sufficient for large datasets**: When the dataset is very large, a single hold-out set can provide a stable estimate of performance without needing multiple splits since the variance of the estimate is low.

---

## The Core Idea: K-Fold Cross-Validation

The dataset is divided into **k** equally-sized folds. The model is trained on **k-1** folds and evaluated on the remaining fold. This process repeats **k** times, each fold serving as the test set exactly once. The final reported metric is the mean (and optionally the standard deviation) across all **k** scores.

![K-Fold Cross-Validation](https://scikit-learn.org/stable/_images/grid_search_cross_validation.png)

The CV score is:

$$\text{CV Score} = \frac{1}{k} \sum_{i=1}^{k} \text{Score}_i$$

Reporting standard deviation alongside the mean is important:

$$\sigma = \sqrt{\frac{1}{k} \sum_{i=1}^{k} (\text{Score}_i - \overline{\text{Score}})^2}$$

A high standard deviation indicates instability — the model is sensitive to the specific data in each fold.

### Choosing k

| k Value | Data Usage | Variance | Bias | Compute |
|---------|-----------|----------|------|---------|
| 5       | 80% train, 20% test per fold | Moderate | Moderate | Low |
| 10      | 90% train, 10% test per fold | Lower | Lower | Medium |
| n (LOO) | All but one sample | Very high | Very low | Very high |

**Practical recommendation:** use **5 or 10 folds** in most settings. Empirical research consistently shows 10-fold provides a good bias-variance tradeoff for the estimate of generalization error.

---

## Hold-Out vs Cross-Validation: Summary

| Problem | Hold-Out | Cross-Validation |
|---------|----------|------------------|
| Variance of error estimate | High — depends on which samples are split | Low — averaged over k splits |
| Pessimistic bias | Yes — model trains on subset only | Reduced — each model trains on (k-1)/k of data |
| Data wastage | Yes — validation set never trains the model | No — every sample trains in k-1 folds |
| Arbitrary split | Yes — random state determines result | No — all possible partitions are covered |

---

## Cross-Validation Strategies

### 1. K-Fold

The baseline approach. The dataset is split into k folds of equal size. Each fold serves as the test set once.

In k-fold cross-validation, the dataset is divided into k equal parts, and the model is trained on k−1 folds and tested on the remaining fold. This process is repeated k times so that each fold serves as the test set once. The average performance across all folds is used to evaluate the model. Cross-validation is mainly used for model selection and hyperparameter tuning. After selecting the best configuration, a single final model is trained on the entire training dataset and evaluated on a separate test set.

```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(LogisticRegression(), X, y, cv=kf, scoring='accuracy')

print(f"Mean: {scores.mean():.4f}  Std: {scores.std():.4f}")
```

**Characteristics:**

- Computationally efficient for moderate k (5 or 10)
- Widely used and supported in libraries
- **Reduction in variance**: Averaging over multiple splits reduces the variance of the performance estimate
- **Potential high bias**: If k is too low, the training sets are small, leading to higher bias in the estimate. Conversely, if k is too high (e.g., LOO), the training sets are almost as large as the full dataset, which can lead to low bias but very high variance in the estimate.
- **May not work well with imbalanced datasets**: K-Fold does not guarantee that each fold will have a similar distribution of classes, which can be problematic for imbalanced datasets.

**When to use:** When the dataset is large enough to support multiple splits and the classes are relatively balanced

`Default choice for regression and balanced classification problems.`

---

### 2. Stratified K-Fold

A variation of K-Fold that preserves the class distribution in each fold. For a dataset where 10% of samples are class 1, each fold also contains approximately 10% class-1 samples.

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
```

**When to use:** Classification tasks, especially with class imbalance. `cross_val_score` uses `StratifiedKFold` by default when the estimator is a classifier.

---

### 3. Leave-One-Out (LOO)

A special case of K-Fold where k = n (number of samples). Each sample is the test set once; the model trains on all remaining n-1 samples.

So, if you have 100 samples, LOO will train 100 models, each time leaving out one sample for testing.

```python
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo)
```

**Characteristics:**

- Minimal bias — each model sees almost all the data
- Very high variance in the estimate due to near-identical training sets
- Low bias but high variance makes it a poor choice for general use.
- Computationally expensive for large datasets
- r2 can not be used with LOO for regression because the test set has only one sample, making the denominator zero.

**When to use:** Only for very small datasets (< 50 samples) where data cannot be afforded to be wasted.

---

### 4. Leave-P-Out (LPO)

Generalizes LOO to leave out p samples at each iteration. Generates $\binom{n}{p}$ train-test splits, which grows combinatorially.

```python
from sklearn.model_selection import LeavePOut

lpo = LeavePOut(p=2)
```

**Practical note:** Rarely used in practice due to the explosive number of iterations even for small p.

---

### 5. Repeated K-Fold

Runs K-Fold multiple times with different random splits each time. Reduces the variance in the CV estimate at the cost of more computation.

```python
from sklearn.model_selection import RepeatedKFold

rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
scores = cross_val_score(model, X, y, cv=rkf)
```

**When to use:** When a more stable estimate is needed and compute is available. Particularly useful for small datasets.

---

### 6. Shuffle Split

Generates independent train/test splits by randomly shuffling the data. The number of splits is configurable independently of the test size.

```python
from sklearn.model_selection import ShuffleSplit

ss = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
scores = cross_val_score(model, X, y, cv=ss)
```

**When to use:** When you want fine control over train/test proportions without the constraint of exhaustive fold coverage.

---

### 7. Group K-Fold

Prevents data leakage when samples within the same group are correlated (e.g., multiple readings from the same patient). No group appears in both the training set and test set of a given split.

```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=3)
scores = cross_val_score(model, X, y, cv=gkf, groups=patient_ids)
```

**When to use:** Medical data, time-series collected from multiple subjects, sensor data per device.

---

### 8. Time Series Split

Designed for temporal data. Training always precedes testing in time — future data is never used to predict the past.

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X, y, cv=tscv)
```

Each successive training set is a superset of the previous one. Standard K-Fold would mix past and future data, producing optimistically biased estimates for time-series.

---

## Strategy Selection Reference

| Scenario | Recommended Strategy |
|----------|---------------------|
| General regression | `KFold(n_splits=10)` |
| Classification (balanced) | `StratifiedKFold(n_splits=5)` |
| Classification (imbalanced) | `StratifiedKFold` |
| Very small dataset | `LeaveOneOut` or `RepeatedKFold` |
| Grouped / dependent samples | `GroupKFold` |
| Time-dependent data | `TimeSeriesSplit` |
| Custom proportion control | `ShuffleSplit` |
| Stable estimate needed | `RepeatedStratifiedKFold` |

---

## Bias-Variance Tradeoff in Cross-Validation

Cross-validation itself has a bias-variance tradeoff in how well it estimates the true generalization error:

- **High k (e.g., LOO):** Low bias (training set size close to n) but high variance in the estimate. Computationally expensive.
- **Low k (e.g., 2-fold):** Higher bias (smaller training sets) but more stable estimates.
- **k = 5 or 10:** Standard balance. Most practical recommendations land here.

This mirrors the bias-variance tradeoff in model selection — using more data reduces bias; having more independent partitions reduces variance of the estimate.

---

## Using `cross_validate` for Multiple Metrics

`cross_val_score` returns scores for one metric. `cross_validate` returns scores for multiple metrics plus timing information.

```python
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, f1_score

scoring = {
    'accuracy': 'accuracy',
    'f1': make_scorer(f1_score, average='weighted'),
    'roc_auc': 'roc_auc'
}

results = cross_validate(
    model, X, y,
    cv=StratifiedKFold(n_splits=5),
    scoring=scoring,
    return_train_score=True
)

print(f"Test accuracy:  {results['test_accuracy'].mean():.4f}")
print(f"Train accuracy: {results['train_accuracy'].mean():.4f}")
```

Comparing train and test scores across folds is a direct diagnostic for overfitting:

- Train >> Test → model is overfit
- Train ≈ Test but both low → model is underfit

---

## A Note on Shuffling

If samples with the same class label are grouped contiguously in the dataset (a common artifact of how data is collected or loaded), splitting without shuffling will cause each fold to see only certain classes in training.

Always verify:

```python
# Inspect class distribution before CV
import numpy as np
print(np.bincount(y))  # Are classes contiguous or mixed?
```

`KFold(shuffle=True, random_state=42)` handles this. `StratifiedKFold` resolves it by construction.

**Exception:** Time series data should never be shuffled. `TimeSeriesSplit` enforces the correct temporal order.

---

## Common Mistakes

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| Fitting scaler on the full dataset before CV | Test statistics leak into training folds, inflating metrics | Fit scaler only on training fold in each iteration |
| Using K-Fold on imbalanced classification | Some folds may contain no minority class samples | Use `StratifiedKFold` |
| Shuffling time-series data | Future data contaminates training, optimistic estimates | Use `TimeSeriesSplit` |
| Using K-Fold when samples within groups are correlated | Model is tested on data similar to what it trained on | Use `GroupKFold` |
| Reporting only mean CV score | Hides instability across folds | Report mean ± std |

---

## Common Questions

**Q: After 5-fold CV, do we deploy one of the 5 trained models — or average them?**  
Neither. The 5 fold models exist only for evaluation. Once the best hyperparameters are selected, a single final model is trained on the entire training dataset and that model is deployed.

**Q: Why not average the k fold models in standard cross-validation?**  
Cross-validation is an *evaluation* technique, not an ensembling method. Its purpose is to estimate generalization performance and guide model selection — not to combine the fold models into a final predictor.

**Q: When do we actually average (combine) models trained on different folds?**  
In ensemble techniques: bagging, stacking, or CV-based blending. In those methods, combining multiple models is a deliberate strategy to *improve* performance, not just measure it.

> **Key Distinction:** Cross-validation = model *selection* strategy. Ensembling = model *combination* strategy.

---

## Practical Workflow Summary

1. Hold out a final test set before anything else
2. Define a CV strategy appropriate for the data structure (grouped, temporal, imbalanced, etc.)
3. Report mean and standard deviation of the CV metric — never just the mean
4. Train the final model on the full training set after selecting hyperparameters
5. Evaluate once on the held-out test set — do not retune after this step

---

## Key Concepts Quick Reference

| Term | Definition |
|------|-----------|
| Fold | One partition of the dataset used as the test set in one iteration |
| k | Number of folds; typically 5 or 10 |
| Stratification | Preserving class proportion across folds |
| Pessimistic bias | Hold-out trains on less data, so its error estimate is systematically too high |
| Data leakage | Test information contaminating training, inflating metrics |
| Generalization error | True error on unseen data; what CV estimates |

---

## References

- Scikit-learn: [Cross-validation documentation](https://scikit-learn.org/stable/modules/cross_validation.html)
- Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning*, Chapter 7
- James, Witten, Hastie, Tibshirani — *An Introduction to Statistical Learning*, Chapter 5
- Kohavi, R. (1995). "A study of cross-validation and bootstrap for accuracy estimation and model selection." *IJCAI*
