# Introduction to Scikit-Learn

## What is Scikit-Learn?

**Scikit-Learn** (sklearn) is an open-source machine learning library for Python that provides simple and efficient tools for data analysis and modeling. It is built on top of NumPy, SciPy, and Matplotlib, making it a powerful yet accessible library for both beginners and professionals.

**Official Website:** https://scikit-learn.org

**Key Highlight:** Scikit-learn supports both **supervised** and **unsupervised learning** algorithms, making it a one-stop solution for most machine learning tasks.

---

## Why Use Scikit-Learn?

### 1. Well-Documented

Scikit-learn has excellent documentation with:

- Clear API references for every function
- Detailed user guides with theory explanations
- Extensive examples for each algorithm
- Tutorials for beginners to advanced users

**Documentation:** https://scikit-learn.org/stable/documentation.html

This makes it easy to understand what each function does and how to use it correctly.

---

### 2. Easy to Learn

Scikit-learn follows a consistent and simple API design:

**The Universal Pattern:**

```python
# Step 1: Import the algorithm
from sklearn.algorithm_module import AlgorithmName

# Step 2: Create model instance
model = AlgorithmName()

# Step 3: Train the model
model.fit(X_train, y_train)

# Step 4: Make predictions
predictions = model.predict(X_test)
```

**Every algorithm follows this same pattern**, making it easy to switch between different models without learning new syntax.

---

### 3. Works Well with NumPy and Pandas

Scikit-learn is designed to integrate seamlessly with the Python data science ecosystem:

- **NumPy arrays:** Primary data structure for scikit-learn
- **Pandas DataFrames:** Can be directly used as input
- **Matplotlib/Seaborn:** For visualizing results

**Example:**

```python
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load data with Pandas
df = pd.read_csv('housing_data.csv')

# Use DataFrame columns directly
X = df[['area', 'bedrooms', 'age']]
y = df['price']

# Train model
model = LinearRegression()
model.fit(X, y)
```

---

### 4. Professionally Built

Scikit-learn is:

- **Maintained by experts:** Developed and maintained by data scientists and engineers worldwide
- **Production-ready:** Used in industry by companies like Spotify, Booking.com, Evernote
- **Optimized:** Efficient implementations of algorithms with C/Cython backends
- **Tested:** Rigorous testing ensures reliability
- **Open-source:** Free to use, with a large community

---

## Installation

### Using pip:

```bash
pip install scikit-learn
```

### Using conda:

```bash
conda install scikit-learn
```

### Verify installation:

```python
import sklearn
print(sklearn.__version__)
```

---

## Core Concepts

### 1. Estimators

**Definition:** Any object that learns from data. All machine learning algorithms in scikit-learn are implemented as estimators.

**Common methods:**

- `fit(X, y)`: Train the model on data
- `predict(X)`: Make predictions on new data
- `score(X, y)`: Evaluate model performance

**Examples:** LinearRegression, LogisticRegression, KNeighborsClassifier

---

### 2. Transformers

**Definition:** Objects that transform data (e.g., scaling, encoding, dimensionality reduction).

**Common methods:**

- `fit(X)`: Learn transformation parameters from data
- `transform(X)`: Apply the transformation
- `fit_transform(X)`: Fit and transform in one step

**Examples:**

- `StandardScaler`: Standardize features (mean=0, variance=1)
- `MinMaxScaler`: Scale features to a range (e.g., 0-1)
- `LabelEncoder`: Encode categorical labels as numbers
- `PCA`: Reduce dimensionality

---

### 3. Predictors

**Definition:** Estimators that can make predictions. All supervised learning algorithms are predictors.

**Methods:**

- `predict(X)`: Returns predicted values
- `predict_proba(X)`: Returns class probabilities (for classifiers)

---

### 4. Pipelines

**Definition:** A way to chain multiple steps (transformers and estimators) into a single workflow.

**Benefits:**

- Cleaner code
- Prevents data leakage
- Easy to deploy

**Example:**

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# Fit and predict in one go
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

---

## Scikit-Learn Algorithms

Scikit-learn provides implementations of most popular machine learning algorithms. Here's a categorized overview:

### 1. Linear Regression

**Type:** Regression (Supervised Learning)

**Purpose:** Predict continuous numerical values by fitting a linear relationship between features and target.

**Use Cases:**

- House price prediction
- Sales forecasting
- Salary estimation

**Import:**

```python
from sklearn.linear_model import LinearRegression
```

**Simple Example:**

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Sample data
X = np.array([[1], [2], [3], [4], [5]])  # Area (in 100 sq ft)
y = np.array([100, 150, 200, 250, 300])  # Price (in $1000s)

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Predict
new_area = np.array([[6]])
predicted_price = model.predict(new_area)
print(f"Predicted price: ${predicted_price[0]}k")
# Output: Predicted price: $350k
```

**Mathematical Form:**

```
y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

**When to use:** When you expect a linear relationship between features and target.

---

### 2. Logistic Regression

**Type:** Classification (Supervised Learning)

**Purpose:** Predict categorical outcomes (binary or multi-class) using a logistic function.

**Use Cases:**

- Email spam detection (spam/not spam)
- Disease diagnosis (positive/negative)
- Customer churn prediction (will churn/won't churn)

**Import:**

```python
from sklearn.linear_model import LogisticRegression
```

**Simple Example:**

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

# Sample data: Study hours vs Pass/Fail
X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([0, 0, 0, 1, 1, 1])  # 0=Fail, 1=Pass

# Create and train model
model = LogisticRegression()
model.fit(X, y)

# Predict
new_student = np.array([[3.5]])
prediction = model.predict(new_student)
probability = model.predict_proba(new_student)

print(f"Prediction: {'Pass' if prediction[0] == 1 else 'Fail'}")
print(f"Probability of passing: {probability[0][1]:.2f}")
```

**Mathematical Form:**

```
P(y=1) = 1 / (1 + e^-(wx + b))
```

**Key Point:** Despite the name, Logistic Regression is used for **classification**, not regression.

---

### 3. K-Nearest Neighbors (kNN)

**Type:** Classification or Regression (Supervised Learning)

**Full Form:** K-Nearest Neighbors

**Purpose:** Classify or predict based on the majority vote or average of the K nearest data points.

**Intuition:** "Tell me who your neighbors are, and I'll tell you who you are."

**How it works:**

1. Store all training data
2. For a new data point, find K closest training examples
3. **For classification:** Majority vote of those K neighbors
4. **For regression:** Average value of those K neighbors

**Use Cases:**

- Image recognition
- Recommendation systems
- Pattern recognition

**Import:**

```python
from sklearn.neighbors import KNeighborsClassifier  # For classification
from sklearn.neighbors import KNeighborsRegressor   # For regression
```

**Simple Example:**
```python
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Sample data: Height, Weight vs Gender
X = np.array([
    [170, 60],  # Female
    [165, 55],  # Female
    [180, 75],  # Male
    [175, 70],  # Male
    [160, 50]   # Female
])
y = np.array(['F', 'F', 'M', 'M', 'F'])

# Create and train model with k=3
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# Predict
new_person = np.array([[172, 65]])
prediction = model.predict(new_person)
print(f"Predicted gender: {prediction[0]}")
```

**Key Parameter:**

- `n_neighbors (k)`: Number of neighbors to consider
  - Small k: More sensitive to noise, complex boundaries
  - Large k: Smoother boundaries, may miss patterns

**Pros:**

- Simple to understand and implement
- No training phase (lazy learning)
- Works well for non-linear data

**Cons:**

- Slow prediction for large datasets
- Sensitive to feature scaling
- Requires choosing optimal k

---

### 4. Decision Trees

**Type:** Classification (C) and Regression (R) (Supervised Learning)

**Purpose:** Make decisions by learning simple if-then-else rules from data, creating a tree-like structure.

**Intuition:** Like a flowchart of questions leading to a decision.

**How it works:**

1. Start with all data at the root
2. Find the best feature to split data
3. Create branches for each value
4. Repeat recursively for each branch
5. Stop when a stopping criterion is met

**Use Cases:**

- Credit approval decisions
- Medical diagnosis
- Customer segmentation
- Any problem requiring interpretability

**Import:**

```python
from sklearn.tree import DecisionTreeClassifier  # For classification
from sklearn.tree import DecisionTreeRegressor   # For regression
```

**Simple Example:**

```python
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Sample data: Weather conditions vs Play Tennis?
# [Temperature, Humidity, Windy]
X = np.array([
    [85, 85, 0],  # Hot, High Humidity, Not Windy
    [80, 90, 1],  # Hot, High Humidity, Windy
    [75, 70, 0],  # Mild, Normal, Not Windy
    [70, 65, 1],  # Cool, Normal, Windy
    [68, 75, 0]   # Cool, Normal, Not Windy
])
y = np.array([0, 0, 1, 1, 1])  # 0=No, 1=Yes

# Create and train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Predict
new_day = np.array([[73, 70, 0]])
prediction = model.predict(new_day)
print(f"Play tennis? {'Yes' if prediction[0] == 1 else 'No'}")
```

**Visualization Example:**
```
           [Temperature > 72?]
               /          \
             Yes           No
             /              \
    [Humidity > 80?]     Play=Yes
        /        \
      Yes        No
      /           \
  Play=No      Play=Yes
```

**Pros:**

- Easy to understand and interpret
- Handles both numerical and categorical data
- No need for feature scaling
- Can capture non-linear relationships

**Cons:**

- Prone to overfitting
- Sensitive to small changes in data
- Can create biased trees for imbalanced data

**Key Parameters:**

- `max_depth`: Maximum depth of tree (controls overfitting)
- `min_samples_split`: Minimum samples required to split a node
- `min_samples_leaf`: Minimum samples required at a leaf node

---

### 5. Naive Bayes

**Type:** Classification (C) (Supervised Learning)

**Purpose:** Probabilistic classifier based on Bayes' Theorem with the "naive" assumption that features are independent.

**Intuition:** Calculate the probability of each class given the features, then pick the most likely class.

**Bayes' Theorem:**
```
P(Class | Features) = [P(Features | Class) × P(Class)] / P(Features)
```

**"Naive" Assumption:** All features are independent of each other (which is often not true in reality, but the algorithm still works well).

**Use Cases:**

- Text classification (spam detection, sentiment analysis)
- Document categorization
- Real-time prediction (very fast)

**Variants in sklearn:**

- `GaussianNB`: For continuous features (assumes Gaussian distribution)
- `MultinomialNB`: For discrete counts (word frequencies in text)
- `BernoulliNB`: For binary features

**Import:**

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
```

**Simple Example (Email Spam Detection):**

```python
from sklearn.naive_bayes import MultinomialNB
import numpy as np

# Sample data: Word frequencies [free, win, money, meeting]
X = np.array([
    [5, 3, 4, 0],  # Spam email
    [6, 4, 5, 0],  # Spam email
    [0, 0, 0, 3],  # Not spam
    [0, 1, 0, 4],  # Not spam
    [1, 0, 0, 5]   # Not spam
])
y = np.array([1, 1, 0, 0, 0])  # 1=Spam, 0=Not Spam

# Create and train model
model = MultinomialNB()
model.fit(X, y)

# Predict
new_email = np.array([[4, 2, 3, 0]])  # High spam words
prediction = model.predict(new_email)
print(f"Email is: {'Spam' if prediction[0] == 1 else 'Not Spam'}")
```

**Pros:**
- Very fast training and prediction
- Works well with high-dimensional data (e.g., text)
- Requires small amount of training data
- Performs well even when independence assumption is violated

**Cons:**
- Assumes feature independence (rarely true)
- Can't learn interactions between features
- Zero frequency problem (solved with smoothing)

**When to use:** Text classification, real-time prediction, when you need a fast baseline model.

---

### 6. Support Vector Machines (SVM)

**Type:** Classification (C) and Regression (R) (Supervised Learning)

**Purpose:** Find the optimal hyperplane that best separates different classes with maximum margin.

**Intuition:** Draw a line (or hyperplane in higher dimensions) that separates classes with the widest possible gap.

**Key Concepts:**

1. **Hyperplane:** Decision boundary that separates classes
2. **Support Vectors:** Data points closest to the hyperplane (these define the boundary)
3. **Margin:** Distance between hyperplane and nearest data points (we want to maximize this)

**Kernel Trick:** SVM can handle non-linear data by transforming it into higher dimensions using kernel functions:

- **Linear kernel:** For linearly separable data
- **RBF (Radial Basis Function):** For non-linear data (most common)
- **Polynomial kernel:** For polynomial relationships

**Use Cases:**

- Image classification
- Text categorization
- Bioinformatics (protein classification)
- Handwriting recognition

**Import:**

```python
from sklearn.svm import SVC  # For classification
from sklearn.svm import SVR  # For regression
```

**Simple Example:**

```python
from sklearn.svm import SVC
import numpy as np

# Sample data: Study hours, Sleep hours vs Pass/Fail
X = np.array([
    [2, 4],   # Fail
    [3, 3],   # Fail
    [5, 6],   # Pass
    [6, 7],   # Pass
    [7, 8]    # Pass
])
y = np.array([0, 0, 1, 1, 1])  # 0=Fail, 1=Pass

# Create and train model with RBF kernel
model = SVC(kernel='rbf', gamma='auto')
model.fit(X, y)

# Predict
new_student = np.array([[4, 5]])
prediction = model.predict(new_student)
print(f"Result: {'Pass' if prediction[0] == 1 else 'Fail'}")
```

**Pros:**

- Effective in high-dimensional spaces
- Works well when number of features > number of samples
- Memory efficient (uses only support vectors)
- Versatile (different kernels for different data)

**Cons:**

- Slow for large datasets
- Requires feature scaling
- Difficult to interpret
- Choosing the right kernel and parameters is tricky

**Key Parameters:**

- `C`: Regularization parameter (controls overfitting)
- `kernel`: Type of kernel function ('linear', 'rbf', 'poly')
- `gamma`: Kernel coefficient (for 'rbf', 'poly')

---

## Algorithm Selection Guide

### For Regression Problems:

| Algorithm | When to Use | Pros | Cons |
|-----------|-------------|------|------|
| **Linear Regression** | Linear relationships, simple baseline | Fast, interpretable | Only linear relationships |
| **Decision Tree Regressor** | Non-linear data, need interpretability | Handles non-linearity, no scaling needed | Prone to overfitting |
| **kNN Regressor** | Small datasets, non-linear patterns | Simple, no training | Slow prediction, needs scaling |
| **SVR** | High-dimensional data, complex patterns | Handles non-linearity well | Slow, requires scaling |

### For Classification Problems:

| Algorithm | When to Use | Pros | Cons |
|-----------|-------------|------|------|
| **Logistic Regression** | Binary classification, baseline | Fast, interpretable, probabilistic | Only linear boundaries |
| **kNN** | Small datasets, multi-class | Simple, no training, handles multi-class | Slow prediction, needs scaling |
| **Decision Tree** | Need interpretability, categorical features | Easy to understand, no scaling | Overfits easily |
| **Naive Bayes** | Text classification, real-time | Very fast, works with small data | Assumes independence |
| **SVM** | High-dimensional, complex boundaries | Powerful, memory efficient | Slow, hard to interpret |

---

## Basic Workflow Example

Here's a complete end-to-end example using scikit-learn:

```python
# Step 1: Import libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Step 2: Load data
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)  # Features
y = df['target']                # Target

# Step 3: Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Scale features (important for many algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Create and train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test_scaled)

# Step 7: Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

---

## Essential Scikit-Learn Modules

### 1. Model Selection (`sklearn.model_selection`)

**Purpose:** Tools for splitting data, cross-validation, and hyperparameter tuning.

**Key Functions:**

```python
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Cross-validation
scores = cross_val_score(model, X, y, cv=5)
print(f"Average accuracy: {scores.mean():.2f}")

# Hyperparameter tuning
param_grid = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

---

### 2. Preprocessing (`sklearn.preprocessing`)

**Purpose:** Tools for data transformation and feature engineering.

**Common Transformers:**

```python
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Standardization (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Normalization (scale to 0-1)
minmax = MinMaxScaler()
X_normalized = minmax.fit_transform(X)

# Encode categorical labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
```

---

### 3. Metrics (`sklearn.metrics`)

**Purpose:** Evaluation metrics for model performance.

**For Classification:**

```python
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Precision, Recall, F1
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Complete report
report = classification_report(y_test, y_pred)
```

**For Regression:**

```python
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)

# Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# Root Mean Squared Error
rmse = np.sqrt(mse)

# R² Score
r2 = r2_score(y_test, y_pred)
```

---

### 4. Datasets (`sklearn.datasets`)

**Purpose:** Built-in sample datasets for practice and testing.

```python
from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from sklearn.datasets import load_digits
from sklearn.datasets import make_classification

# Load famous Iris dataset (classification)
iris = load_iris()
X, y = iris.data, iris.target

# Load Boston housing dataset (regression)
boston = load_boston()
X, y = boston.data, boston.target

# Create synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2)
```

---

## Interview Questions

### Q1: What is scikit-learn and why is it popular?

**Answer:**

Scikit-learn is an open-source machine learning library for Python that provides simple and efficient tools for data analysis and modeling.

**Why it's popular:**

1. **Easy to use:** Consistent API across all algorithms
2. **Well-documented:** Excellent documentation with examples
3. **Comprehensive:** Covers most ML algorithms for supervised and unsupervised learning
4. **Integration:** Works seamlessly with NumPy, Pandas, Matplotlib
5. **Production-ready:** Used in industry, professionally maintained
6. **Free and open-source:** No licensing costs

**Key strength:** You can learn one pattern and apply it to all algorithms.

---

### Q2: Explain the basic scikit-learn workflow for a supervised learning problem.

**Answer:**

The standard workflow has 7 steps:

```python
# 1. Import libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 2. Load and prepare data
X, y = load_data()

# 3. Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

# 4. Preprocess (scale features if needed)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Create and train model
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. Make predictions
y_pred = model.predict(X_test)

# 7. Evaluate
accuracy = accuracy_score(y_test, y_pred)
```

**Key points:**
- Always fit on training data only
- Transform both train and test with same scaler
- Evaluate on test data to check generalization

---

### Q3: What's the difference between fit(), transform(), and fit_transform()?

**Answer:**

These are methods used primarily with **transformers** (preprocessing objects):

**fit(X):**
- Learns parameters from the data
- Example: StandardScaler learns mean and std
- Does NOT transform the data
- Use on training data

**transform(X):**
- Applies learned transformation
- Uses parameters learned during fit()
- Returns transformed data
- Use on both train and test

**fit_transform(X):**
- Convenience method: fit() + transform() in one step
- Equivalent to: `scaler.fit(X); scaler.transform(X)`
- Use ONLY on training data

**Example:**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Training data: learn AND apply
X_train_scaled = scaler.fit_transform(X_train)
# Equivalent to:
# scaler.fit(X_train)
# X_train_scaled = scaler.transform(X_train)

# Test data: ONLY apply (use training parameters)
X_test_scaled = scaler.transform(X_test)
```

**Critical Rule:** Never use `fit_transform()` on test data. This would leak information from test set into training.

---

### Q4: When would you use Linear Regression vs Logistic Regression in scikit-learn?

**Answer:**

**Linear Regression:**
- **Type:** Regression
- **Output:** Continuous numerical values
- **Use when:** Predicting "how much" or "what value"
- **Examples:** House prices, temperature, sales amount

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
price = model.predict([[1500, 3]])  # Returns: 350000.0
```

**Logistic Regression:**
- **Type:** Classification (despite the name!)
- **Output:** Categories/classes
- **Use when:** Predicting "which class" or "yes/no"
- **Examples:** Spam/not spam, pass/fail, disease/no disease

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
result = model.predict([[5, 6]])  # Returns: 1 or 0
```

**Key difference:** Look at your target variable:
- Continuous numbers → Linear Regression
- Categories/classes → Logistic Regression

---

### Q5: What is the purpose of train_test_split() and why is it important?

**Answer:**

`train_test_split()` divides your dataset into two parts: training set and test set.

**Purpose:**

1. **Training set:** Used to teach the model (fit())
2. **Test set:** Used to evaluate the model (predict() and score())

**Why important:**

If you test on the same data you trained on, the model might have just memorized the answers (overfitting). Testing on separate, unseen data ensures the model has truly learned patterns and can generalize.

**Usage:**

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% for testing, 80% for training
    random_state=42     # For reproducibility
)
```

**Best practices:**
- Common split: 70-80% train, 20-30% test
- Use `random_state` for reproducible results
- For small datasets, consider cross-validation instead

---

### Q6: Explain the difference between predict() and predict_proba().

**Answer:**

Both are methods for classification models, but return different outputs:

**predict():**
- Returns the predicted class label
- Output: Array of class labels
- Example: [0, 1, 1, 0]

**predict_proba():**
- Returns probability of each class
- Output: Array of probabilities for each class
- Each row sums to 1.0
- Example: [[0.8, 0.2], [0.3, 0.7], ...]

**Example:**

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

# Hard prediction (class label)
prediction = model.predict([[5, 6]])
print(prediction)  # Output: [1]

# Soft prediction (probabilities)
probabilities = model.predict_proba([[5, 6]])
print(probabilities)  # Output: [[0.25, 0.75]]
#                             [P(class=0), P(class=1)]
```

**When to use:**
- `predict()`: When you need the final decision
- `predict_proba()`: When you need confidence scores or want to set custom thresholds

**Note:** Not all classifiers support `predict_proba()`. Some only provide `decision_function()`.

---

### Q7: What is cross-validation and how do you implement it in scikit-learn?

**Answer:**

**Cross-validation** is a technique to evaluate model performance more reliably by testing on multiple different train/test splits.

**Why use it:**
- Single train/test split might be lucky or unlucky
- Cross-validation gives average performance across multiple splits
- Better estimate of how model will perform on unseen data

**K-Fold Cross-Validation:**

1. Split data into K equal parts (folds)
2. Train on K-1 folds, test on remaining fold
3. Repeat K times, each time with different test fold
4. Average the K results

**Implementation:**

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

# 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5)

print(f"Scores: {scores}")
print(f"Average: {scores.mean():.2f}")
print(f"Std Dev: {scores.std():.2f}")

# Output example:
# Scores: [0.82, 0.85, 0.81, 0.87, 0.83]
# Average: 0.84
# Std Dev: 0.02
```

**Common values for K:**
- K=5 or K=10 (most common)
- Larger K = more computation, less bias
- Smaller K = less computation, more variance

---

### Q8: How do you handle categorical features in scikit-learn?

**Answer:**

Categorical features (non-numerical) need to be encoded into numbers before using scikit-learn algorithms.

**Two main approaches:**

**1. Label Encoding (Ordinal)**

For categorical features with inherent order:

```python
from sklearn.preprocessing import LabelEncoder

# Example: Size = ['Small', 'Medium', 'Large']
encoder = LabelEncoder()
encoded = encoder.fit_transform(['Small', 'Medium', 'Large', 'Small'])
print(encoded)  # Output: [2, 1, 0, 2]
```

**Use when:** Categories have natural ordering (Small < Medium < Large)

**2. One-Hot Encoding (Nominal)**

For categorical features without order:

```python
from sklearn.preprocessing import OneHotEncoder

# Example: Color = ['Red', 'Blue', 'Green']
encoder = OneHotEncoder(sparse=False)
colors = [['Red'], ['Blue'], ['Green'], ['Red']]
encoded = encoder.fit_transform(colors)
print(encoded)
# Output:
# [[0, 1, 0],  # Red   -> [0, 1, 0]
#  [1, 0, 0],  # Blue  -> [1, 0, 0]
#  [0, 0, 1],  # Green -> [0, 0, 1]
#  [0, 1, 0]]  # Red   -> [0, 1, 0]
```

**Use when:** No natural ordering (Red is not > Blue)

**Using Pandas:**

```python
import pandas as pd

df = pd.DataFrame({'Color': ['Red', 'Blue', 'Green', 'Red']})
encoded_df = pd.get_dummies(df, columns=['Color'])
```

**Warning:** One-hot encoding can create many features if category has many unique values (high cardinality).

---

### Q9: What is the difference between StandardScaler and MinMaxScaler?

**Answer:**

Both scale features, but use different methods:

**StandardScaler (Standardization):**

- Transforms data to have mean=0 and standard deviation=1
- Formula: `z = (x - mean) / std`
- Range: Unbounded (can be negative)
- Assumes data is normally distributed

```python
from sklearn.preprocessing import StandardScaler

# Example: [1, 2, 3, 4, 5]
scaler = StandardScaler()
scaled = scaler.fit_transform([[1], [2], [3], [4], [5]])
# Result (approximately): [-1.41, -0.71, 0, 0.71, 1.41]
```

**MinMaxScaler (Normalization):**

- Scales data to a fixed range (default: 0 to 1)
- Formula: `x_scaled = (x - min) / (max - min)`
- Range: [0, 1] (or custom range)
- Preserves the shape of original distribution

```python
from sklearn.preprocessing import MinMaxScaler

# Example: [1, 2, 3, 4, 5]
scaler = MinMaxScaler()
scaled = scaler.fit_transform([[1], [2], [3], [4], [5]])
# Result: [0, 0.25, 0.5, 0.75, 1.0]
```

**When to use:**

| Use StandardScaler when: | Use MinMaxScaler when: |
|--------------------------|------------------------|
| Data is normally distributed | Data is NOT normally distributed |
| Algorithms assume normal distribution (LR, SVM) | You need bounded range |
| Outliers are important | Sensitive to outliers |
| Most common choice | Neural networks, image processing |

**Important:** Always fit on training data only, then transform both train and test!

---

### Q10: Name 3 advantages of using scikit-learn over implementing algorithms from scratch.

**Answer:**

**1. Time-Saving:**
- Pre-built, optimized implementations
- No need to code complex math and optimization
- Can focus on problem-solving rather than implementation
- Example: Implementing SVM from scratch could take weeks; using sklearn takes 3 lines

**2. Performance and Reliability:**
- Algorithms are optimized (C/Cython backends)
- Thoroughly tested and debugged by community
- Handle edge cases automatically
- Production-ready code
- Example: sklearn's Random Forest is much faster than a naive Python implementation

**3. Ecosystem Integration:**
- Works seamlessly with NumPy, Pandas, Matplotlib
- Consistent API across all algorithms
- Rich set of preprocessing, evaluation, and selection tools
- Extensive documentation and community support
- Example: Can easily switch from Decision Tree to Random Forest with just one line change

**Bonus advantages:**
- Regular updates and improvements
- Standardized best practices
- Easy to deploy models
- Compatible with other tools (scikit-learn pipelines can be deployed to production)

---

## Common Scikit-Learn Pitfalls to Avoid

### 1. Data Leakage

**Problem:** Using information from test set during training

**Common mistakes:**

```python
# WRONG: Fitting scaler on entire dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Learns from test data too!
X_train, X_test = train_test_split(X_scaled)

# CORRECT: Fit scaler only on training data
X_train, X_test = train_test_split(X)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

### 2. Not Scaling Features

**Problem:** Many algorithms (SVM, kNN, Logistic Regression) are sensitive to feature scales

```python
# Features with different scales
X = [[1000, 2],      # [income, years_of_education]
     [5000, 4],
     [3000, 6]]

# Income dominates because it's much larger
# Always scale before using these algorithms!
```

---

### 3. Forgetting random_state

**Problem:** Results are different each time you run the code

```python
# WRONG: Results change every run
X_train, X_test = train_test_split(X, y, test_size=0.2)

# CORRECT: Reproducible results
X_train, X_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## Summary

### Key Takeaways

1. **Scikit-learn** is the most popular ML library for Python - easy to use, well-documented, production-ready

2. **Consistent API:** All algorithms follow the same pattern:
   - `fit()` to train
   - `predict()` to make predictions
   - `score()` to evaluate

3. **Main algorithms covered:**
   - **Linear Regression:** Predict continuous values (regression)
   - **Logistic Regression:** Classify into categories (classification)
   - **kNN:** Classify based on nearest neighbors
   - **Decision Trees:** Rule-based decisions (R/C)
   - **Naive Bayes:** Probabilistic classification
   - **SVM:** Find optimal separation boundary (R/C)

4. **Essential modules:**
   - `model_selection`: Split data, cross-validation
   - `preprocessing`: Scale and transform data
   - `metrics`: Evaluate model performance
   - `datasets`: Sample datasets for practice

5. **Best practices:**
   - Always split data before preprocessing
   - Scale features when needed
   - Use cross-validation for better evaluation
   - Set random_state for reproducibility

---

## Next Steps

1. **Practice with built-in datasets:**
   ```python
   from sklearn.datasets import load_iris, load_boston
   ```

2. **Try different algorithms on the same problem:**
   - Compare performance
   - Understand strengths and weaknesses

3. **Learn feature engineering:**
   - Create better features
   - Handle missing data
   - Encode categorical variables

4. **Explore advanced topics:**
   - Ensemble methods (Random Forest, Gradient Boosting)
   - Hyperparameter tuning (GridSearchCV)
   - Pipeline creation
   - Model deployment

---

## Resources

**Official Documentation:** https://scikit-learn.org/stable/

**Tutorials:**
- Scikit-learn official tutorials: https://scikit-learn.org/stable/tutorial/index.html
- User Guide: https://scikit-learn.org/stable/user_guide.html

**Practice Datasets:**
- Kaggle: https://www.kaggle.com/datasets
- UCI ML Repository: https://archive.ics.uci.edu/ml/index.php

**Cheat Sheet:**
- Scikit-learn Algorithm Cheat Sheet: https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

---

## Key Takeaway

Scikit-learn makes machine learning accessible. With just a few lines of code, you can build, train, and evaluate sophisticated models. The consistent API means once you learn one algorithm, you've learned the pattern for all of them. Focus on understanding the problem, choosing the right algorithm, and preparing your data well - scikit-learn handles the complex implementation details for you.
