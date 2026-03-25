# Machine Learning From Scratch

A structured, hands-on journey through core Machine Learning concepts — from theory to implementation. Every algorithm is explored with detailed explanations, mathematical foundations, and Python code (both from scratch and using scikit-learn).

---

## Repository Structure

```
Machine-Learning-From-Scratch/
├── datasets/                              # Shared datasets used across all modules
├── 00-foundations/                        # Core ML theory & foundational concepts
├── 01-data-preparation/                   # Feature engineering & preprocessing
│   ├── feature-engineering/               # Filter, Wrapper & Embedded selection methods
│   └── preprocessing/                     # Missing values, imbalanced data handling
├── 02-optimization/                       # Gradient descent variants & optimization
├── 03-supervised-learning/
│   ├── regression/                        # Linear, Multiple, Polynomial regression
│   │   └── regularization/               # Ridge, Lasso, Elastic Net
│   ├── classification/                    # One subfolder per algorithm
│   │   ├── knn/
│   │   ├── logistic-regression/
│   │   ├── decision-tree/
│   │   ├── naive-bayes/
│   │   ├── svm/
│   │   └── guides/                        # Shared interview guides & cheat sheets
│   └── diagnostics/                       # Regression assumptions & model validation
├── 04-model-evaluation/
│   ├── classification-metrics/
│   ├── regression-metrics/               # coming soon
│   └── validation-techniques/             # coming soon
├── 05-unsupervised-learning/             # coming soon
│   ├── clustering/
│   └── dimensionality-reduction/
├── 06-ensemble-methods/                   # coming soon
│   ├── bagging/
│   ├── boosting/
│   └── stacking/
├── 07-neural-networks/                    # coming soon
│   ├── fundamentals/
│   └── architectures/
└── _archive/
```

---

## Learning Path

### 0. Foundations (`00-foundations/`)

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

#### Feature Engineering

| # | Topic | Type |
|---|-------|------|
| 01 | Filter Methods (Variance, Correlation, Chi-Square, MI, ANOVA) | Notebook |
| 02 | Wrapper Methods (Exhaustive, RFE, Sequential) | Notebook |
| 03 | Embedded Methods (Lasso, Ridge for selection) | Notebook |
| — | Feature Selection Cheatsheet | Markdown |

#### Preprocessing

| # | Topic | Type |
|---|-------|------|
| 01 | Handling Missing Values (MCAR / MAR / MNAR) | Notebook |
| 02 | Handling Imbalanced Datasets (Upsampling / Downsampling) | Notebook |

---

### 2. Optimization (`02-optimization/`)

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

#### Regression

| # | Topic | Type |
|---|-------|------|
| 01 | Simple Linear Regression | Notebook |
| 02 | Comprehensive Linear Regression (encoding, scaling, overfitting) | Notebook |
| 03 | Linear Regression from Scratch (Normal Equation) | Notebook |
| 04 | Multiple Linear Regression | Notebook |
| 05 | Polynomial Regression | Notebook |

#### Regularization

| # | Topic | Type |
|---|-------|------|
| 01 | Ridge Regression (Theory + Code) | Notebook |
| 02 | Ridge Regression — Key Insights | Notebook |
| 03 | Lasso Regression (Theory + Code) | Notebook |
| 04 | Lasso Regression — Key Insights | Notebook |
| 05 | Elastic Net Regression | Notebook |
| — | Why Lasso Creates Sparsity | Markdown |

#### Classification

##### KNN (`knn/`)

| # | Topic | Type |
|---|-------|------|
| 01 | KNN from Scratch | Notebook |

##### Logistic Regression (`logistic-regression/`) — *coming soon*

| # | Topic | Type |
|---|-------|------|
| 01 | Binary Logistic Regression | Notebook |
| 02 | Logistic Regression from Scratch (Gradient Descent) | Notebook |
| 03 | Multiclass — OvR & Softmax | Notebook |

##### Decision Trees (`decision-tree/`) — *coming soon*

| # | Topic | Type |
|---|-------|------|
| 01 | Decision Tree (Entropy + Gini) | Notebook |
| 02 | Decision Tree from Scratch | Notebook |

##### Naive Bayes (`naive-bayes/`) — *coming soon*

| # | Topic | Type |
|---|-------|------|
| 01 | Gaussian Naive Bayes | Notebook |
| 02 | Multinomial & Bernoulli NB | Notebook |

##### SVM (`svm/`) — *coming soon*

| # | Topic | Type |
|---|-------|------|
| 01 | SVM (Hard + Soft Margin) | Notebook |
| 02 | Kernel SVM | Notebook |

##### Guides (`guides/`)

| File | Topic |
|------|-------|
| `maximum-likelihood-estimation.md` | MLE Interview Guide |

#### Diagnostics

| # | Topic | Type |
|---|-------|------|
| 01 | Assumptions of Linear Regression (linearity, normality, homoscedasticity, multicollinearity) | Notebook |

---

### 4. Model Evaluation (`04-model-evaluation/`)

#### Classification Metrics

| # | Topic | Type |
|---|-------|------|
| 01 | Classification Metrics (Accuracy, Precision, Recall, F1, ROC-AUC) | Notebook |

#### Regression Metrics — *coming soon*

MAE, MSE, RMSE, R²

#### Validation Techniques — *coming soon*

K-Fold CV, Stratified CV, TimeSeriesSplit

---

### 5. Unsupervised Learning (`05-unsupervised-learning/`) — *coming soon*

K-Means, DBSCAN, PCA, t-SNE

---

### 6. Ensemble Methods (`06-ensemble-methods/`) — *coming soon*

Random Forest (Bagging), AdaBoost, XGBoost (Boosting), Stacking

---

### 7. Neural Networks (`07-neural-networks/`) — *coming soon*

Perceptron, MLP, backpropagation from scratch

---

## Datasets

| File | Used In |
|------|---------|
| `breast-cancer-wisconsin.csv` | Classification → KNN |
| `diabetes.csv` | Feature Engineering (Embedded Methods) |
| `heart_disease_uci.csv` | Classification Metrics |
| `height-weight-simple.csv` | Regression (Simple Linear) |
| `housing-data.csv` | Feature Engineering (Wrapper Methods) |
| `human-activity-recognition.csv` | Feature Engineering (Filter Methods) |
| `insurance.csv` | Regression (Comprehensive Linear) |
| `modified-synthetic-economic-data.csv` | Regression (Multiple Linear) |

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

pip install numpy pandas matplotlib seaborn scikit-learn statsmodels mlxtend plotly
```
