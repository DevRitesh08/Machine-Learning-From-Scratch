# 📊 Supervised Learning

The largest section — covers regression, classification, regularization, and diagnostics.

## Subfolders

### `/regression`
Predicting continuous values.

| # | Topic | Type |
|---|-------|------|
| 01 | Simple Linear Regression | Notebook |
| 02 | Comprehensive Linear Regression | Notebook |
| 03 | Linear Regression from Scratch (Normal Eq.) | Notebook |
| 04 | Multiple Linear Regression | Notebook |
| 05 | Polynomial Regression | Notebook |

**`/regression/regularization`** — Ridge, Lasso, Elastic Net

**`/regression/guides`** — Complete linear regression guide

---

### `/classification`
Predicting categorical labels. Each algorithm has its own subfolder.

> **Note:** Some algorithms (KNN, Decision Trees, SVM) work for both classification AND regression. They're organized here by their classification implementation. Regression variants can be added under `/regression` in the future.

| Subfolder | Algorithm | Notebooks |
|-----------|-----------|-----------|
| `knn/` | K-Nearest Neighbors | from-scratch implementation |
| `logistic-regression/` | Logistic Regression | interview guide, multiclass, OvR, softmax, polynomial |
| `naive-bayes/` | Naive Bayes | from-scratch, sentiment analysis, out-of-core |
| `decision-tree/` | Decision Trees | theory + dtreeviz visualization |
| `svm/` | Support Vector Machines | SVM demo |

**`/classification/references`** — PDF lecture slides for reference

---

### `/diagnostics`
Validating model assumptions.

| # | Topic | Type |
|---|-------|------|
| 01 | Assumptions of Linear Regression | Notebook |

---

### `/projects`
End-to-end projects applying supervised learning.

| # | Topic | Type |
|---|-------|------|
| 01 | Classification Project | Notebook |

## Prerequisites

- [02-optimization](../02-optimization/) — understand gradient descent
- [01-data-preparation](../01-data-preparation/) — know how to prep data

## Next Steps

→ [04-model-evaluation](../04-model-evaluation/) to measure model performance
