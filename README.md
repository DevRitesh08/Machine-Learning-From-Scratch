# Machine Learning From Scratch

A structured, hands-on journey through core Machine Learning concepts — from theory to implementation. Every algorithm is explored with detailed explanations, mathematical foundations, and Python code (both from scratch and using scikit-learn).

---

## Repository Structure

```
├── datasets/                        # Shared datasets used across all modules
├── 00-ml-foundations/               # Core ML theory & concepts
├── 01-linear-regression/           # Simple, Multiple & Polynomial regression
├── 02-regression-diagnostics/      # Assumptions testing & model validation
├── 03-gradient-descent/            # Batch, Stochastic & Mini-batch GD
├── 04-regularization/              # Ridge, Lasso & Elastic Net
├── 05-feature-selection/           # Filter, Wrapper & Embedded methods
├── 06-data-preprocessing/          # Missing values, imbalanced data handling
└── _archive/                       # Superseded early drafts (for reference)
```

---

## Project Architecture & Guidelines

### Current State Analysis
**Strengths:**
- Well-organized regression-focused learning path
- Consistent naming conventions and numbering
- Clear separation of concerns (theory, implementation, diagnostics)
- Comprehensive gradient descent and regularization coverage

**Architectural Gaps Identified:**
- Missing classification algorithms (KNN, Logistic Regression, Decision Trees)
- No model evaluation modules (metrics, cross-validation, confusion matrices) 
- Unused datasets (breast-cancer.csv has no corresponding notebooks)
- Incomplete supervised learning coverage (regression only)

### Recommended Expansion Structure
```
├── 07-classification-algorithms/          # ← NEW: Add classification here
│   ├── 01-knn-from-scratch.ipynb         # K-Nearest Neighbors theory & implementation
│   ├── 02-knn-breast-cancer.ipynb        # KNN applied to breast cancer dataset
│   ├── 03-logistic-regression.ipynb      # Logistic regression fundamentals
│   ├── 04-decision-trees.ipynb           # Tree-based classification
│   ├── classification-guide.md           # Classification overview
│   └── references/                       # Supporting materials
├── 08-model-evaluation/                   # ← NEW: Evaluation metrics & techniques
│   ├── 01-classification-metrics.ipynb   # Accuracy, Precision, Recall, F1, ROC-AUC
│   ├── 02-cross-validation.ipynb         # K-Fold, Stratified CV, TimeSeriesCV
│   ├── 03-confusion-matrix-analysis.ipynb# Deep dive into confusion matrices
│   └── evaluation-guide.md               # Metrics selection guide
├── 09-ensemble-methods/                   # ← FUTURE: Advanced techniques
├── 10-unsupervised-learning/             # ← FUTURE: Clustering, PCA
└── 11-advanced-topics/                   # ← FUTURE: Neural networks, etc.
```

### Architecture Principles
1. **Sequential Learning:** Each module builds on previous knowledge
2. **Theory → Practice:** Concepts before implementation before application
3. **Modular Design:** Self-contained modules with clear dependencies
4. **Consistent Structure:** Each module follows same organization pattern
5. **Scalable Numbering:** Room for insertion without renumbering existing content

### Adding New Content Guidelines
**For Classification Algorithms:** Place in `07-classification-algorithms/`
**For New Datasets:** 
- Add to `datasets/` folder
- Update dataset table in README
- Create corresponding application notebooks

**Naming Convention:**
- `01-algorithm-theory.ipynb` - Mathematical foundation & intuition
- `02-algorithm-implementation.ipynb` - From-scratch coding
- `03-algorithm-application.ipynb` - Real-world dataset application
- `algorithm-guide.md` - Comprehensive reference document

### Future Architecture Roadmap
**Phase 1 (Immediate):** Complete supervised learning with classification
**Phase 2 (Short-term):** Model evaluation and validation techniques  
**Phase 3 (Medium-term):** Ensemble methods and advanced algorithms
**Phase 4 (Long-term):** Unsupervised learning and specialized topics

This architecture ensures systematic progression from foundations → regression → classification → evaluation → advanced topics, creating a complete machine learning curriculum.

---

## Learning Path

### 0. ML Foundations (`00-ml-foundations/`)
Start here. Covers what Machine Learning is, types of ML, an introduction to scikit-learn, and the supervised learning paradigm.

| # | Topic | Type |
|---|-------|------|
| 01 | What is Machine Learning? | Markdown |
| 02 | Types of Machine Learning | Markdown |
| 03 | Introduction to Scikit-Learn | Markdown |
| 04 | Supervised Learning | Markdown |
| 05 | Bias-Variance Tradeoff | Notebook |
| 06 | Regression Analysis | Notebook |

### 1. Linear Regression (`01-linear-regression/`)
Complete coverage of linear regression — simple, multiple, polynomial, from-scratch implementation, and a comprehensive guide.

| # | Topic | Type |
|---|-------|------|
| 01 | Simple Linear Regression | Notebook |
| 02 | Comprehensive Linear Regression (encoding, scaling, overfitting) | Notebook |
| 03 | Linear Regression from Scratch (Normal Equation) | Notebook |
| 04 | Multiple Linear Regression | Notebook |
| 05 | Polynomial Regression | Notebook |
| — | Linear Regression Complete Guide | Markdown |

### 2. Regression Diagnostics (`02-regression-diagnostics/`)
Test and validate regression model assumptions.

| # | Topic | Type |
|---|-------|------|
| 01 | Assumptions of Linear Regression (linearity, normality, homoscedasticity, multicollinearity) | Notebook |

### 3. Gradient Descent (`03-gradient-descent/`)
Deep dive into optimization — from intuition to animated visualizations to full implementations.

| # | Topic | Type |
|---|-------|------|
| 01 | Gradient Descent Step by Step | Notebook |
| 02 | Gradient Descent from Scratch | Notebook |
| 03 | Gradient Descent Animation (slope + intercept) | Notebook |
| 04 | Batch Gradient Descent | Notebook |
| 05 | Stochastic Gradient Descent | Notebook |
| 06 | Mini-Batch Gradient Descent | Notebook |
| — | Batch / SGD / Mini-Batch Guides | Markdown |

### 4. Regularization (`04-regularization/`)
Ridge, Lasso, and Elastic Net — theory, math, and practical experiments.

| # | Topic | Type |
|---|-------|------|
| 01 | Ridge Regression (Theory + Code) | Notebook |
| 02 | Ridge Regression — Key Insights | Notebook |
| 03 | Lasso Regression (Theory + Code) | Notebook |
| 04 | Lasso Regression — Key Insights | Notebook |
| 05 | Elastic Net Regression | Notebook |
| — | Why Lasso Creates Sparsity | Markdown |

### 5. Feature Selection (`05-feature-selection/`)
Techniques to identify the most relevant features for your model.

| # | Topic | Type |
|---|-------|------|
| 01 | Filter Methods (Variance, Correlation, Chi-Square, MI, ANOVA) | Notebook |
| 02 | Wrapper Methods (Exhaustive, RFE, Sequential) | Notebook |
| 03 | Embedded Methods (Lasso, Ridge for selection) | Notebook |
| — | Feature Selection Cheatsheet | Markdown |

### 6. Data Preprocessing (`06-data-preprocessing/`)
Practical techniques for handling real-world data issues.

| # | Topic | Type |
|---|-------|------|
| 01 | Handling Missing Values (MCAR / MAR / MNAR) | Notebook |
| 02 | Handling Imbalanced Datasets (Upsampling / Downsampling) | Notebook |

---

## Datasets

All datasets live in `datasets/` and are shared across modules:

| File | Used In |
|------|---------|
| `breast-cancer.csv` | — |
| `diabetes.csv` | Feature Selection (Embedded Methods) |
| `height-weight-simple.csv` | Simple Linear Regression |
| `housing-data.csv` | Feature Selection (Wrapper Methods) |
| `human-activity-recognition.csv` | Feature Selection (Filter Methods) |
| `insurance.csv` | Comprehensive Linear Regression |
| `modified-synthetic-economic-data.csv` | Multiple Linear Regression |
| `studentscores.csv` | (Archived) Simple LR draft |

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels mlxtend plotly
```

---

## Naming Conventions

- **Folders**: `kebab-case` with zero-padded numbering (`01-linear-regression/`)
- **Files**: `kebab-case` with numbering for learning sequence (`01-simple-linear-regression.ipynb`)
- **Assets**: Descriptive names in `assets/` subfolders (`gd-slope-intercept-animation-1.gif`)
- **References**: PDF reference materials in `references/` subfolders
