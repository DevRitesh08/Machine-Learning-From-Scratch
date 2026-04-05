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
│   └── guides/
├── 03-supervised-learning/
│   ├── regression/                        # Linear, Multiple, Polynomial regression
│   │   ├── guides/
│   │   └── regularization/                # Ridge, Lasso, Elastic Net
│   ├── classification/                    # One subfolder per algorithm
│   │   ├── knn/
│   │   ├── logistic-regression/
│   │   ├── naive-bayes/
│   │   ├── decision-tree/
│   │   ├── svm/
│   │   └── references/                    # Shared lecture PDFs
│   ├── diagnostics/                       # Regression assumptions & model validation
│   └── projects/                          # End-to-end projects
├── 04-model-evaluation/
│   ├── classification-metrics/
│   ├── regression-metrics/                # coming soon
│   └── validation-techniques/
├── 05-unsupervised-learning/
│   ├── clustering/                        # coming soon
│   └── dimensionality-reduction/
├── 06-ensemble-methods/
│   ├── bagging/
│   ├── boosting/
│   └── stacking/                          # coming soon
├── 07-neural-networks/                    # coming soon
│   ├── fundamentals/
│   └── architectures/
└── _archive/                              # Old/superseded files
```

> **Note:** Each numbered folder contains its own `README.md` with detailed topic listings and learning paths.

---

## Setup & Requirements

```bash
# Clone the repository
git clone https://github.com/DevRitesh08/Machine-Learning-From-Scratch.git
cd Machine-Learning-From-Scratch

# Set up virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
