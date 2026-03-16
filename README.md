# Machine Learning From Scratch

A structured, hands-on journey through core Machine Learning concepts — from theory to implementation. Every algorithm is explored with detailed explanations, mathematical foundations, and Python code (both from scratch and using scikit-learn).

---

## Repository Structure

```
Machine-Learning-From-Scratch/
├── datasets/                              # Shared datasets used across all modules
├── 00-foundations/                        # Core ML theory & foundational concepts
├── 01-data-preparation/                   # Feature engineering & preprocessing
│   ├── feature-engineering/              # Filter, Wrapper & Embedded selection methods
│   └── preprocessing/                    # Missing values, imbalanced data handling
├── 02-optimization/                       # Gradient descent variants & optimization
├── 03-supervised-learning/               # All supervised learning algorithms
│   ├── regression/                       # Linear, Multiple, Polynomial regression
│   │   └── regularization/              # Ridge, Lasso, Elastic Net
│   ├── classification/                   # KNN and classification algorithms
│   └── diagnostics/                      # Regression assumptions & model validation
├── 04-model-evaluation/                   # Metrics, validation & evaluation techniques
│   ├── classification-metrics/           # Accuracy, Precision, Recall, F1, ROC-AUC
│   ├── regression-metrics/               # MAE, MSE, RMSE, R² (coming soon)
│   └── validation-techniques/            # Cross-validation strategies (coming soon)
├── 05-unsupervised-learning/             # Clustering & dimensionality reduction
│   ├── clustering/                       # K-Means and other clustering (coming soon)
│   └── dimensionality-reduction/         # PCA and related techniques (coming soon)
├── 06-ensemble-methods/                   # Bagging, Boosting, Stacking (coming soon)
│   ├── bagging/
│   ├── boosting/
│   └── stacking/
├── 07-neural-networks/                    # Neural network fundamentals & architectures
│   ├── fundamentals/                     # (coming soon)
│   └── architectures/                    # (coming soon)
└── _archive/                              # Superseded early drafts (for reference)
```

---

## Architecture Principles

1. **Sequential Learning** — Each module builds on previous knowledge
2. **Theory → Practice** — Concepts before implementation before application
3. **Domain Grouping** — Algorithms grouped by purpose (supervised, unsupervised, etc.)
4. **Modular Design** — Self-contained modules with clear dependency boundaries
5. **Scalable Structure** — Subdirectory organization supports growth without renumbering

---

## Learning Path

### 0. Foundations (`00-foundations/`)
Start here. Covers what Machine Learning is, types of ML, an introduction to scikit-learn, and the supervised learning paradigm.

| # | Topic | Type |
|---|-------|------|
| 01 | What is Machine Learning? | Markdown |
| 02 | Types of Machine Learning | Markdown |
| 03 | Introduction to Scikit-Learn | Markdown |
| 04 | Supervised Learning | Markdown |
| 05 | Bias-Variance Tradeoff | Notebook |
| 06 | Regression Analysis | Notebook |

---

### 1. Data Preparation (`01-data-preparation/`)

#### Feature Engineering (`feature-engineering/`)
Techniques to identify and select the most relevant features for your model.

| # | Topic | Type |
|---|-------|------|
| 01 | Filter Methods (Variance, Correlation, Chi-Square, MI, ANOVA) | Notebook |
| 02 | Wrapper Methods (Exhaustive, RFE, Sequential) | Notebook |
| 03 | Embedded Methods (Lasso, Ridge for selection) | Notebook |
| — | Feature Selection Cheatsheet | Markdown |

#### Preprocessing (`preprocessing/`)
Practical techniques for handling real-world data issues.

| # | Topic | Type |
|---|-------|------|
| 01 | Handling Missing Values (MCAR / MAR / MNAR) | Notebook |
| 02 | Handling Imbalanced Datasets (Upsampling / Downsampling) | Notebook |

---

### 2. Optimization (`02-optimization/`)
Deep dive into gradient descent — from intuition to animated visualizations to full implementations.

| # | Topic | Type |
|---|-------|------|
| 01 | Gradient Descent Step by Step | Notebook |
| 02 | Gradient Descent from Scratch | Notebook |
| 03 | Gradient Descent Animation (slope + intercept) | Notebook |
| 04 | Batch Gradient Descent | Notebook |
| 05 | Stochastic Gradient Descent | Notebook |
| 06 | Mini-Batch Gradient Descent | Notebook |
| — | Batch GD Guide | Markdown |
| — | Stochastic GD Guide | Markdown |
| — | Mini-Batch GD Guide | Markdown |

---

### 3. Supervised Learning (`03-supervised-learning/`)

#### Regression (`regression/`)
Complete coverage of linear regression — simple, multiple, polynomial, and from-scratch implementation.

| # | Topic | Type |
|---|-------|------|
| 01 | Simple Linear Regression | Notebook |
| 02 | Comprehensive Linear Regression (encoding, scaling, overfitting) | Notebook |
| 03 | Linear Regression from Scratch (Normal Equation) | Notebook |
| 04 | Multiple Linear Regression | Notebook |
| 05 | Polynomial Regression | Notebook |

#### Regularization (`regression/regularization/`)
Ridge, Lasso, and Elastic Net — theory, math, and practical experiments.

| # | Topic | Type |
|---|-------|------|
| 01 | Ridge Regression (Theory + Code) | Notebook |
| 02 | Ridge Regression — Key Insights | Notebook |
| 03 | Lasso Regression (Theory + Code) | Notebook |
| 04 | Lasso Regression — Key Insights | Notebook |
| 05 | Elastic Net Regression | Notebook |
| — | Why Lasso Creates Sparsity | Markdown |

#### Classification (`classification/`)

Classification algorithms built from scratch.

| #  | Topic                                         | Type     |
| -- | --------------------------------------------- | -------- |
| 01 | K-Nearest Neighbors from Scratch              | Notebook |
| 06 | Maximum Likelihood Estimation Interview Guide | Markdown |

#### Diagnostics (`diagnostics/`)

Test and validate regression model assumptions.

| # | Topic | Type |
|---|-------|------|
| 01 | Assumptions of Linear Regression (linearity, normality, homoscedasticity, multicollinearity) | Notebook |

---

### 4. Model Evaluation (`04-model-evaluation/`)

#### Classification Metrics (`classification-metrics/`)

| # | Topic | Type |
|---|-------|------|
| 01 | Classification Metrics (Accuracy, Precision, Recall, F1, ROC-AUC) | Notebook |

#### Regression Metrics (`regression-metrics/`)
> Coming soon — MAE, MSE, RMSE, R²

#### Validation Techniques (`validation-techniques/`)
> Coming soon — K-Fold CV, Stratified CV, TimeSeriesSplit

---

### 5. Unsupervised Learning (`05-unsupervised-learning/`)
> Coming soon — K-Means clustering, DBSCAN, PCA, t-SNE

---

### 6. Ensemble Methods (`06-ensemble-methods/`)
> Coming soon — Random Forest (Bagging), XGBoost (Boosting), Stacking

---

### 7. Neural Networks (`07-neural-networks/`)
> Coming soon — Perceptron, MLP, backpropagation from scratch

---

## Datasets

All datasets live in `datasets/` and are shared across modules:

| File | Used In |
|------|---------|
| `breast-cancer-wisconsin.csv` | Classification (KNN) |
| `diabetes.csv` | Feature Engineering (Embedded Methods) |
| `heart_disease_uci.csv` | Classification Metrics |
| `height-weight-simple.csv` | Regression (Simple Linear) |
| `housing-data.csv` | Feature Engineering (Wrapper Methods) |
| `human-activity-recognition.csv` | Feature Engineering (Filter Methods) |
| `insurance.csv` | Regression (Comprehensive Linear) |
| `modified-synthetic-economic-data.csv` | Regression (Multiple Linear) |
| `studentscores.csv` | (Archived) Simple LR draft |

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # macOS / Linux
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels mlxtend plotly
```

---

## Naming Conventions

- **Folders**: `kebab-case` with zero-padded numbering (`03-supervised-learning/`)
- **Files**: `kebab-case` with numbering for learning sequence (`01-simple-linear-regression.ipynb`)
- **Assets**: Descriptive names in `assets/` subfolders
- **References**: PDF reference materials in `references/` subfolders
- **Guides**: Markdown companion documents co-located with related notebooks
