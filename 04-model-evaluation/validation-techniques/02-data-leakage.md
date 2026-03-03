# Data Leakage

Data leakage is one of the most subtle and damaging problems in machine learning. It produces models that appear to work brilliantly during development but fail catastrophically in production. This document covers the full landscape — what it is, how it occurs, how to detect it, and how to systematically prevent it.

---

## What is Data Leakage?

> **Data leakage occurs when information from outside the training dataset is used to build or evaluate a model — information that would not be available at prediction time in the real world.**

When a model is trained on data that "knows" the answer (directly or indirectly), it learns patterns that will not exist when the model is deployed. The model memorizes rather than generalizes.

The result: **unrealistically high performance during evaluation that completely collapses in production.**

---

## Why It's a Serious Problem

| Scenario | Impact |
|----------|--------|
| Kaggle competitions | Leaky features inflate leaderboard ranks; models don't generalize |
| Company data pipelines | Anonymization can be reversed; privacy breach |
| Production ML systems | Models are useless in the real world despite looking perfect in evaluation |
| Research publications | Overly optimistic results that cannot be reproduced |

The hardest part about data leakage is that **it doesn't always look wrong**. The model trains cleanly, CV scores are high, and everything passes review — but the model is fundamentally invalid.

---

## The Core Mental Model

Before building any model, establish a strict **time boundary**:

> At the moment of making a prediction, what information is **actually available**? Only that information should be used in training and evaluation.

Anything that crosses this boundary — information that "travels back in time" from the future or from the test set — is leakage.

---

## Types of Data Leakage

### 1. Target Leakage

**Definition:** A feature in the training data is directly or indirectly derived from the target variable, and that feature would not be available at prediction time.

**Classic Example — Fraud Detection:**

Suppose you're predicting whether a transaction is fraudulent. If your dataset contains a feature like `account_flagged_for_review = True/False` that was only populated *after* the fraud investigation completed, training on it is target leakage. At prediction time (when the transaction just happened), this field would be empty.

| Feature | Status |
|---------|--------|
| `transaction_amount` | Safe — available at prediction time |
| `merchant_category` | Safe — available at prediction time |
| `account_flagged_for_review` | **LEAKAGE** — populated after fraud confirmed |
| `refund_issued` | **LEAKAGE** — only exists after fraudulent charge confirmed |

**Another Example — Disease Prediction:**

Predicting whether a patient has a disease. If the dataset includes `treatment_administered` (which is only assigned after diagnosis), this is leakage. The model doesn't learn to predict disease — it learns to detect whether treatment has started.

**How to detect it:**
- Unusually high model accuracy (>90–95%) on a hard problem
- A single feature that dominates importance (often the leaky one)
- Features with names that imply outcomes (`_result`, `_approved`, `_confirmed`, `_after`)

---

### 2. Train-Test Contamination

**Definition:** Information from the test/validation set leaks into the training process, most commonly through data preprocessing applied on the full dataset before splitting.

This is the most frequently committed form of leakage — often by beginners and sometimes by experienced practitioners.

#### The Preprocessing Contamination Pattern

**Wrong approach (leaky):**
```
1. Load full dataset (train + test combined)
2. Fit scaler on full dataset → scaler learns mean/std from test data too
3. Transform full dataset
4. Split into train / test
5. Train model → evaluate on test
```

The scaler has already "seen" the test set. The normalization coefficients (mean, std, min, max) encode test distribution into every scaled value. The model indirectly has knowledge of the test set.

**Correct approach (no leakage):**
```
1. Load full dataset
2. Split into train / test FIRST
3. Fit scaler on train set ONLY
4. Transform train set using that fitted scaler
5. Transform test set using the SAME fitted scaler (no re-fitting)
6. Train model → evaluate on test
```

**Operations that require fit → transform pattern:**

| Operation | Leaky if done on full data | Safe approach |
|-----------|---------------------------|---------------|
| StandardScaler / MinMaxScaler | Yes | Fit on train, transform both |
| Missing value imputation (mean/median) | Yes | Compute mean from train, fill both |
| Encoding (target encoding, frequency encoding) | Yes | Compute encodings from train only |
| PCA / dimensionality reduction | Yes | Fit PCA on train, project both |
| Feature selection (statistical tests) | Yes | Run selection on train fold only |
| Outlier removal | Yes | Detect/fit on train only |
| SMOTE / oversampling | Yes | Apply only inside train fold |

---

### 3. Temporal Leakage (Future Leakage)

**Definition:** In time-series or time-dependent data, the model uses future information to predict the past.

This is an extremely common and devastating form of leakage in financial, demand forecasting, and predictive maintenance problems.

#### How it Happens

**Scenario:** Predicting daily stock price movement for Day T.
- Safe features: price history up to Day T-1, volume on Day T-1
- Leaky features: stock closing price on Day T, news headlines from Day T after market close

**Random split on time-series data is wrong:**
```
Random split: train = [Day 1, 3, 5, 7, 9] | test = [Day 2, 4, 6, 8, 10]
The model trains on Day 9 and predicts Day 2. It has seen the "future".
```

**Correct split for time-series:**
```
Temporal split: train = [Day 1 → Day 7] | test = [Day 8 → Day 10]
The model only ever sees past data to predict future data.
```

#### Temporal CV — Walk-Forward Validation

Each fold expands the training window forward in time — test data is always strictly after all training data.

---

### 4. Preprocessing Leakage (Scaling / Imputation)

This deserves its own category because it is so widespread and so counterintuitive.

**The mechanism:**

Scaling requires statistics (mean, standard deviation, min, max). If you compute those statistics from the full dataset (including the test set), you are encoding test-set information into the scaling transformation. Every training sample now carries the "fingerprint" of the test distribution.

**Concrete example:**

Say your training set has ages from 20–60 with mean 40.  
Your test set has ages 70–90 (older population).

If you standardize the full dataset together:
- Scaler computes mean = 50 (influenced by test data)  
- Training points get scaled with test-influenced mean  
- Model trains on "contaminated" scaled values

If you standardize on training only:
- Scaler computes mean = 40 from training  
- Training points are scaled correctly  
- Test points are scaled using train statistics (some may be "out of range" — this is **correct and expected**)

**Missing value imputation:**

```python
# WRONG — leaks test information
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)  # X = full dataset (train + test)
X_train, X_test = train_test_split(X_imputed)

# CORRECT — no leakage
X_train, X_test = train_test_split(X)
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)   # learn mean from train only
X_test  = imputer.transform(X_test)        # apply train's mean to test
```

---

### 5. Group/Subject Leakage

**Definition:** When multiple observations belong to the same subject/entity, a random split may put different observations from the same subject in both train and test sets. The model then "knows" the subject already.

**Examples:**
- Medical data: 5 patient visits from the same patient split across train and test
- Audio/image recognition: Multiple recordings/images from the same person
- NLP: Multiple sentences from the same document

The model memorizes the subject rather than learning general patterns.

**Fix:** Use `GroupKFold` — ensure all observations from the same group stay in the same fold.

```python
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
for train_idx, test_idx in gkf.split(X, y, groups=patient_ids):
    ...
```

---

### 6. Duplicate / Near-Duplicate Leakage

**Definition:** Near-identical records (due to data collection errors, augmentation, or data re-entry) end up on both sides of a train/test split.

The model literally sees the test record during training — or a near-identical copy.

**Fix:** Deduplicate before splitting. Check for near-duplicates in time-series or image augmentation pipelines.

---

## Summary: Leakage Taxonomy

```
Data Leakage
├── Target Leakage
│   ├── Future outcome features used as predictors
│   └── Post-event features included in training
│
├── Train-Test Contamination
│   ├── Preprocessing on full dataset before split
│   ├── Feature selection on full dataset
│   └── Hyperparameter tuning without holdout
│
├── Temporal Leakage
│   ├── Random split on time-ordered data
│   └── Future timestamps as features
│
├── Preprocessing Leakage
│   ├── Scaling with full-dataset statistics
│   └── Imputation using test-set values
│
├── Group / Subject Leakage
│   └── Same entity in both train and test
│
└── Duplicate Leakage
    └── Near-identical records across splits
```

---

## Detecting Data Leakage

### Signal 1: Suspiciously High Accuracy

| Domain | Suspicious Accuracy |
|--------|-------------------|
| Fraud detection | > 99% |
| Medical diagnosis | > 95% |
| Stock price prediction | > 70% (directional) |
| NLP sentiment | > 98% |

**"Too good to be true" performance is a dead giveaway of leakage.**

### Signal 2: Feature Importance Dominated by One Feature

If a single feature accounts for 90%+ of model importance, especially a feature with a name that implies an outcome, investigate immediately.

```python
import pandas as pd
import matplotlib.pyplot as plt

# For tree-based models
feat_imp = pd.Series(model.feature_importances_, index=feature_names)
feat_imp.sort_values(ascending=False).head(10).plot(kind='barh')
plt.title('Feature Importances — Check for Leaky Features')
plt.show()
```

### Signal 3: Train/Test Performance Gap

If your model has 98% accuracy on the test set but only 70% in production, this is a strong sign of leakage. The model is not learning generalizable patterns — it's memorizing the test set.

### Signal 4: Domain Knowledge Check

if a feature like `account_flagged_for_review` perfectly separates classes, it's likely a leaky feature. The model isn't learning to predict fraud — it's learning to read the "answer key" from the dataset.

---

## The Correct ML Pipeline (Leak-Free)

```
Raw Dataset
    │
    ▼
┌─────────────────────┐
│    Train / Test      │  ← SPLIT FIRST, before any preprocessing
│   Split (80 / 20)    │
└────────┬────────────┘
         │
    ┌────┴─────┐
    │          │
  TRAIN      TEST (held out, untouched until final evaluation)
    │
    ▼
┌─────────────────────────────────────┐
│   Cross-Validation (K-Fold on TRAIN) │
│                                     │
│  For each fold:                     │
│    1. Fit scaler on train fold      │
│    2. Transform train fold          │
│    3. Transform validation fold     │
│    4. Train model on train fold     │
│    5. Evaluate on validation fold   │
└─────────────────────────────────────┘
    │
    ▼
Best model trained on full TRAIN set
    │
    ▼
Final evaluation on held-out TEST set  ← only evaluated ONCE
```

---

## Prevention: Using scikit-learn Pipelines

**Pipelines are the most reliable defense against preprocessing leakage.** They enforce that all fitting happens only on training data.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Define the pipeline — all steps are fit only on training folds
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

# cross_val_score fits the entire pipeline fresh on each training fold
# No leakage possible — the imputer and scaler never see the test fold
scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
```

**Why this works:**

- For each fold, the entire pipeline is re-fit on the training portion
- The scaler computes mean/std from training data only
- The imputer computes fill values from training data only
- The validation fold is only transformed, never fit

---

## Prevention: Correct Feature Selection

Feature selection performed on the full dataset is leakage — the selected features are influenced by test data.

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# CORRECT: feature selection inside the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(f_classif, k=10)),  # re-fitted each fold
    ('model', SVC())
])

scores = cross_val_score(pipeline, X, y, cv=5)

# WRONG: feature selection outside the pipeline
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)  # uses full X including test data
scores = cross_val_score(SVC(), X_selected, y, cv=5)  # LEAKY
```

---

## Prevention: Holdout Validation Set

Even with a proper pipeline and cross-validation, maintain a **completely untouched holdout set** that is evaluated **exactly once** at the very end.

```
Full Dataset (100%)
├── Holdout Set (20%) → never touched until final evaluation
└── Development Set (80%)
    ├── Used for all cross-validation, tuning, and model selection
    └── Final model trained on all 80%
        └── Evaluated once on Holdout Set → final production estimate
```

This provides a sanity check that your cross-validation estimate hasn't leaked through repeated model comparisons and hyperparameter tuning.

---

## Quick Reference: 5 Rules to Prevent Leakage

| Rule | What to Do |
|------|-----------|
| **Split first** | Always split train/test before any data transformation |
| **Fit on train only** | All preprocessing (scaling, imputation, encoding) must be fit on training data only |
| **Use pipelines** | Let scikit-learn pipelines enforce this automatically during cross-validation |
| **Temporal awareness** | For time-series, never let any test timestamp precede any training timestamp |
| **Domain knowledge check** | For each feature, ask: "Would this be available at prediction time in production?" |

---

## Common Beginner Mistakes

| Mistake | Why It's Leakage | Fix |
|---------|-----------------|-----|
| `scaler.fit_transform(X_full)` then split | Scaler sees test distribution | Split first, fit on train only |
| Select features on full dataset | Selected features influenced by test | Use Pipeline with feature selector |
| Fill NaN with `X.mean()` (full dataset) | Mean includes test values | Use `SimpleImputer` inside Pipeline |
| Random split on time-series | Model trains on "future" data | Use `TimeSeriesSplit` |
| Include post-event features (e.g., `refund_issued`) | Feature exists because of the target | Remove features unavailable at prediction time |
| Use same person's data in train and test | Model memorizes subject | Use `GroupKFold` |
| Compute SMOTE on full dataset | Synthetic points influenced by test | Apply SMOTE inside CV fold, after split |

---

## Real-World Case Studies

### Case 1: Healthcare — Predicting Patient Readmission

**Dataset contains:** patient demographics, diagnoses, `discharge_disposition` (where patient went after discharge).

`discharge_disposition = "Skilled Nursing Facility"` is highly predictive of readmission — but this field is determined at the time of discharge, at the same moment you'd want to make the prediction. In retrospective data it's available, but in real deployment you'd need to predict before discharge is finalized.

**Result without fix:** 92% accuracy in evaluation, 51% in production.

---

### Case 2: Finance — Credit Default Prediction

**Dataset contains:** `total_payments_made` during the loan period.

This feature can only be computed after the loan period ends — which is after the default event. An approved loan with many payments is clearly not defaulted.

**Result without fix:** Model "predicts" default by checking if payments are low — trivially true because defaulted loans have fewer payments, not a causal relationship.

---

### Case 3: Scaling Before Split

**Standard beginner code:**
```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # WRONG: uses full dataset

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
```

**Correct code:**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # fit on train only
X_test  = scaler.transform(X_test)       # transform test using train stats
```

---

## Key Takeaways

1. **Data leakage makes models look exceptional in development and fail in production.** It is the single most common reason ML projects disappoint in deployment.

2. **The golden rule:** at every preprocessing step, ask — *does this step use information the model wouldn't have at prediction time?*

3. **Pipelines solve 80% of leakage problems** by enforcing that fit operations only see training data during cross-validation.

4. **Temporal data requires temporal splits.** A random split on time-ordered data is almost always leaky.

5. **A holdout set is your insurance policy.** It catches leakage that slips through cross-validation due to repeated model comparisons.

---

## Related Topics

- [Cross-Validation](./01-cross-validation.md) — The evaluation framework that, when used correctly, prevents train-test contamination
- [Handling Imbalanced Datasets](../../01-data-preparation/preprocessing/02-handling-imbalanced-datasets.ipynb) — SMOTE must be applied inside CV folds to avoid leakage
- [Feature Selection Methods](../../01-data-preparation/feature-engineering/) — Feature selection done on the full dataset is a form of leakage

---

## References

- Kaufman, S., et al. (2012). *Leakage in Data Mining: Formulation, Detection, and Avoidance.* ACM TKDD.
- Brownlee, J. (2020). *Data Leakage in Machine Learning.* Machine Learning Mastery.
- Kaggle. *Data Leakage* — Learn Track, Intermediate ML.
- Cawley, G. & Talbot, N. (2010). *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation.* JMLR.
