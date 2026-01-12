# Linear Regression - Complete Guide

## Table of Contents

### Part 1: Fundamentals
1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [Mathematical Foundation](#mathematical-foundation)
4. [Cost Function](#cost-function)
5. [Key Intuitions](#key-intuitions)
6. [Real-World Applications](#real-world-applications)

### Part 2: Optimization
7. [Gradient Descent Algorithm](#gradient-descent-algorithm)
8. [Mathematical Derivation](#mathematical-derivation)
9. [Learning Rate](#learning-rate)
10. [Convergence and Optimization](#convergence-and-optimization)

### Part 3: Implementation and Evaluation
11. [Evaluation Metrics](#evaluation-metrics)
12. [Implementation from Scratch](#implementation-from-scratch)
13. [Implementation with Scikit-Learn](#implementation-with-scikit-learn)
14. [Complete End-to-End Examples](#complete-end-to-end-examples)
15. [Model Evaluation and Interpretation](#model-evaluation-and-interpretation)

### Resources
16. [Interview Questions - Fundamentals](#interview-questions-fundamentals)
17. [Interview Questions - Gradient Descent](#interview-questions-gradient-descent)
18. [Interview Questions - Implementation](#interview-questions-implementation)
19. [Summary and Best Practices](#summary-and-best-practices)

---

# Part 1: Fundamentals

## Introduction

Linear Regression is a supervised learning algorithm used to model the relationship between a dependent variable (target) and one or more independent variables (features). It assumes a linear relationship between input and output.

**Use Case**: Predicting continuous numerical values based on input features.

**Example**: Predicting exam scores based on study hours.

---

## Core Concepts

### 1. Dependent and Independent Variables

- **Independent Variable (x)**: The input feature(s) used to make predictions
  - Example: Study hours
  
- **Dependent Variable (y)**: The output or target we want to predict
  - Example: Exam score

### 2. Best Fit Line

The goal of linear regression is to find the **best fit line** that minimizes the difference between predicted and actual values.

**Simple Linear Regression Equation**:

$$y = mx + c$$

Where:
- $y$ = Predicted value (dependent variable)
- $x$ = Input feature (independent variable)
- $m$ = Slope (coefficient) - rate of change
- $c$ = Intercept (bias) - value when x = 0

### 3. Hypothesis Function

In machine learning notation, the hypothesis function represents our prediction model:

**For Simple Linear Regression (1 feature)**:

$$h_b(x) = b_0 + b_1 x_1$$

**For Multiple Linear Regression (n features)**:

$$h_b(x) = b_0 + b_1 x_1 + b_2 x_2 + ... + b_n x_n$$

Where:
- $h_b(x)$ = Hypothesis function (predicted value)
- $b_0$ = Bias/Intercept (constant term)
- $b_1, b_2, ..., b_n$ = Coefficients/Parameters for each feature
- $x_1, x_2, ..., x_n$ = Input features

---

## Mathematical Foundation

### Dimensionality Perspective

1. **1D (One Feature)**: Line in 2D space
   - Example: Score vs Hours studied
   
2. **2D (Two Features)**: Plane in 3D space
   - Example: House price vs (area, bedrooms)
   
3. **3D+ (Multiple Features)**: Hyperplane in higher dimensional space
   - Example: Salary prediction based on multiple factors

---

## Cost Function

The cost function measures how well the model's predictions match the actual data. It quantifies the error between predicted and actual values.

### Error (Residual)

$$\text{Error} = \text{Actual Value} - \text{Predicted Value}$$

$$\text{Error} = y^{(i)} - h_b(x^{(i)})$$

### Mean Squared Error (MSE)

The most common cost function for linear regression:

$$J(b) = \frac{1}{2m} \sum_{i=1}^{m} (h_b(x^{(i)}) - y^{(i)})^2$$

Where:
- $J(b)$ = Cost function
- $m$ = Number of data points
- $h_b(x^{(i)})$ = Predicted value for i-th data point
- $y^{(i)}$ = Actual value for i-th data point

**Objective**: Minimize the cost function to find optimal parameters $b_0, b_1, ..., b_n$

---

## Key Intuitions

### 1. Why Linear Regression Works

Linear regression finds the line (or hyperplane) that best represents the relationship between features and target by minimizing the total squared error. This ensures predictions are as close as possible to actual values.

### 2. Why Square the Errors?

- **Positive and negative errors don't cancel out**: Squaring makes all errors positive
- **Penalizes large errors more**: Large deviations are penalized quadratically
- **Mathematical convenience**: Differentiable and convex, making optimization easier

### 3. The Learning Process

```
Step 1: Initialize parameters (b₀, b₁, ..., bₙ) randomly
Step 2: Calculate predictions using hypothesis function
Step 3: Compute cost function (error)
Step 4: Update parameters to reduce cost
Step 5: Repeat steps 2-4 until cost is minimized
```

### 4. When to Use Linear Regression

**Use When**:
- Relationship between features and target is approximately linear
- You need interpretable results
- You want to understand feature importance
- Predicting continuous values

**Don't Use When**:
- Relationship is highly non-linear
- Data has complex patterns
- Classification problems (use logistic regression instead)

---

## Real-World Applications

### 1. Finance
- Stock price prediction
- Risk assessment
- Credit scoring
- Revenue forecasting

### 2. Real Estate
- House price prediction based on area, location, bedrooms
- Rental price estimation
- Property valuation

### 3. Healthcare
- Predicting patient recovery time
- Medical cost estimation
- Disease progression modeling

### 4. Marketing
- Sales forecasting
- Customer lifetime value prediction
- Advertising spend optimization
- Demand prediction

### 5. Business Analytics
- Employee salary prediction
- Churn prediction (with modifications)
- Inventory management
- Performance metrics forecasting

### 6. Agriculture
- Crop yield prediction
- Weather-based harvest estimation

### 7. Education
- Student performance prediction
- Grade forecasting based on study time

---

# Part 2: Optimization

## Gradient Descent Algorithm

### What is Gradient Descent?

Gradient Descent is an **iterative optimization algorithm** used to minimize the cost function by adjusting model parameters in the direction of the steepest descent of the function's gradient.

**Key Idea**: Move downhill on the cost function surface until we reach the minimum.

### Intuition

Imagine you are standing on a mountain (cost function surface) and want to reach the valley (minimum cost). Gradient descent helps you:

1. Look around to find the steepest downward slope (gradient)
2. Take a step in that direction
3. Repeat until you reach the bottom (convergence)

### Visual Understanding

```
Cost Function J(b) is like a bowl-shaped curve:

    J(b)
     |
     |    *               Starting point (high cost)
     |     \
     |      \    *        Moving down
     |       \  /
     |        \/          Minimum (lowest cost)
     |________|__________  b (parameters)
              b*
```

---

## Mathematical Derivation

### Cost Function (Recap)

$$J(b) = \frac{1}{2m} \sum_{i=1}^{m} (h_b(x^{(i)}) - y^{(i)})^2$$

Where:
- $h_b(x) = b_0 + b_1 x_1$ (for simple linear regression)

### The Update Rule

The core equation of gradient descent:

$$b_k = b_k - \alpha \cdot \frac{\partial J(b)}{\partial b_k}$$

Where:
- $b_k$ = Parameter being updated (can be $b_0$, $b_1$, ..., $b_n$)
- $\alpha$ = Learning rate (controls step size)
- $\frac{\partial J(b)}{\partial b_k}$ = Partial derivative (gradient/slope) of cost function with respect to $b_k$

### Breaking Down the Components

**Gradient**: $\frac{\partial J(b)}{\partial b}$
- Represents the **slope** of the cost function
- Indicates the direction and magnitude of steepest ascent
- **Positive gradient**: Cost increases → move parameters left (decrease)
- **Negative gradient**: Cost decreases → move parameters right (increase)

**Descent**:
- Moving **downwards** on the cost function surface
- Subtracting the gradient moves us toward minimum

### Partial Derivatives for Linear Regression

For the hypothesis function $h_b(x) = b_0 + b_1 x$:

**Update for $b_0$ (intercept/bias)**:

$$b_0 = b_0 - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m} (h_b(x^{(i)}) - y^{(i)})$$

**Update for $b_1$ (slope/weight)**:

$$b_1 = b_1 - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m} (h_b(x^{(i)}) - y^{(i)}) \cdot x^{(i)}$$

### Example Walkthrough

**Given Data**: (1, 2), (2, 4), (3, 6)

**Case 1**: $b_0 = 0$, $b_1 = 2$
- Hypothesis: $h_b(x) = 2x$
- Cost: $J(b) = \frac{1}{6}(0^2 + 0^2 + 0^2) = 0$ ✓ Perfect fit!

**Case 2**: $b_0 = 0$, $b_1 = 1$
- Hypothesis: $h_b(x) = x$
- Predictions: (1, 1), (2, 2), (3, 3)
- Errors: (1), (2), (3)
- Cost: $J(b) = 2.34$

**Case 3**: $b_0 = 0$, $b_1 = 3$
- Hypothesis: $h_b(x) = 3x$
- Predictions: (1, 3), (2, 6), (3, 9)
- Errors: (-1), (-2), (-3)
- Cost: $J(b) = 2.34$

The algorithm finds the **global minimum** at $b_1 = 2$ where cost is minimized.

---

## Learning Rate

### What is Learning Rate (α)?

The learning rate is a **hyperparameter** that controls how big a step we take during each iteration of gradient descent.

**Symbol**: $\alpha$ (alpha)

### Effect of Learning Rate

**1. Too Small ($\alpha$ → very small)**:
- **Advantage**: Stable, precise convergence
- **Disadvantage**: 
  - Extremely slow training
  - Takes many iterations to converge
  - Computationally expensive

**2. Too Large ($\alpha$ → very large)**:
- **Advantage**: Faster initial progress
- **Disadvantage**:
  - May overshoot the minimum
  - Can diverge (cost increases)
  - Never converges
  - Oscillates around minimum

**3. Optimal ($\alpha$ → appropriate)**:
- Reaches minimum efficiently
- Smooth convergence
- Reasonable number of iterations

### Choosing Learning Rate

**Common values**: 0.001, 0.003, 0.01, 0.03, 0.1, 0.3

**Best practice**: 
- Start with a moderate value (e.g., 0.01)
- Plot cost vs iterations
- If cost increases: reduce $\alpha$
- If cost decreases very slowly: increase $\alpha$

### Adaptive Learning Rate

Some advanced algorithms automatically adjust learning rate:
- **AdaGrad**: Adapts learning rate per parameter
- **RMSprop**: Uses moving average of squared gradients
- **Adam**: Combines momentum and adaptive learning rates

---

## Convergence and Optimization

### Convergence Theorem

The algorithm **converges** when parameters stop changing significantly, meaning we've reached the minimum.

**Convergence Condition**:
- Cost function change becomes very small: $|J(b)_{new} - J(b)_{old}| < \epsilon$
- Parameter change becomes negligible: $|b_{new} - b_{old}| < \epsilon$
- Where $\epsilon$ is a small threshold (e.g., 0.0001)

### The Complete Algorithm

```
Step 1: Initialize b₀, b₁ with random values (or zeros)

Step 2: Calculate predictions for all data points
        h_b(x^(i)) = b₀ + b₁x^(i)

Step 3: Calculate cost function J(b)
        J(b) = 1/(2m) Σ(h_b(x^(i)) - y^(i))²

Step 4: Update parameters using gradient descent
        b₀ = b₀ - α · (1/m) Σ(h_b(x^(i)) - y^(i))
        b₁ = b₁ - α · (1/m) Σ(h_b(x^(i)) - y^(i)) · x^(i)

Step 5: Check convergence
        If converged → Stop
        Else → Go to Step 2

Step 6: Return final parameters b₀, b₁
```

### Batch vs Stochastic Gradient Descent

**Batch Gradient Descent**:
- Uses **all** training examples in each iteration
- More stable, smoother convergence
- Slower for large datasets
- Guaranteed to converge to global minimum (for convex functions)

**Stochastic Gradient Descent (SGD)**:
- Uses **one** training example at a time
- Faster iterations
- Noisier updates, may oscillate
- Good for large datasets

**Mini-Batch Gradient Descent**:
- Uses **small batches** of training examples
- Balance between batch and stochastic
- Most commonly used in practice

### Feature Scaling Importance

When features have different scales, gradient descent can be slow and inefficient.

**Example**:
- Feature 1: House area (500-5000 sq ft)
- Feature 2: Number of bedrooms (1-5)

**Solution**:
- **Standardization**: Mean = 0, Std = 1
  $$x_{scaled} = \frac{x - \mu}{\sigma}$$

- **Normalization**: Scale to [0, 1]
  $$x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

---

# Part 3: Implementation and Evaluation

## Evaluation Metrics

Evaluation metrics measure how well our model's predictions match actual values. They help us quantify model performance and compare different models.

### 1. Mean Absolute Error (MAE)

**Definition**: Average of absolute differences between predicted and actual values.

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

Where:
- $y_i$ = Actual value
- $\hat{y}_i$ = Predicted value
- $n$ = Number of data points

**Characteristics**:
- **Units**: Same as the target variable
- **Range**: [0, ∞), lower is better
- **Interpretation**: Average error magnitude
- **Robust**: Less sensitive to outliers than MSE

**When to use**:
- When all errors should be weighted equally
- When outliers should not dominate the metric
- When you need intuitive interpretation in original units

**Example**: If MAE = 5 for house price prediction, on average, predictions are off by $5,000.

---

### 2. Mean Squared Error (MSE)

**Definition**: Average of squared differences between predicted and actual values.

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**Characteristics**:
- **Units**: Squared units of target variable
- **Range**: [0, ∞), lower is better
- **Interpretation**: Average squared error
- **Sensitive**: Heavily penalizes large errors (outliers)

**When to use**:
- When large errors are particularly undesirable
- Standard metric for gradient descent optimization
- When you want to penalize outliers more

---

### 3. Root Mean Squared Error (RMSE)

**Definition**: Square root of MSE, bringing error back to original units.

$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

**Characteristics**:
- **Units**: Same as target variable (like MAE)
- **Range**: [0, ∞), lower is better
- **Interpretation**: Standard deviation of prediction errors
- **Sensitive**: Still penalizes large errors like MSE

**MAE vs RMSE**:
- RMSE ≥ MAE always
- If RMSE >> MAE, there are large outliers
- If RMSE ≈ MAE, errors are uniform

---

### 4. R-Squared (R² Score)

**Definition**: Proportion of variance in the dependent variable explained by the model.

$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

Where:
- $\sum(y_i - \hat{y}_i)^2$ = Sum of squared residuals (SSR)
- $\sum(y_i - \bar{y})^2$ = Total sum of squares (SST)
- $\bar{y}$ = Mean of actual values

**Characteristics**:
- **Range**: (-∞, 1], typically [0, 1]
  - 1 = Perfect predictions
  - 0 = Model no better than predicting mean
  - Negative = Model worse than predicting mean
- **Unit-free**: No units, easy comparison across datasets
- **Interpretation**: Percentage of variance explained

**Example**: R² = 0.85 means model explains 85% of variance in the target variable.

---

### 5. Adjusted R-Squared

**Definition**: R² adjusted for the number of features, penalizing unnecessary complexity.

$$R^2_{adj} = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}$$

Where:
- $n$ = Number of samples
- $p$ = Number of features
- $R^2$ = Regular R-squared

**When to use**:
- Feature selection (which features to include)
- Comparing models with different feature counts
- Preventing overfitting

**Interpretation**:
- If Adjusted R² decreases when adding a feature, that feature doesn't improve the model
- Always ≤ R²

---

### Comparison of Metrics

| Metric | Units | Range | Outlier Sensitivity | Best For |
|--------|-------|-------|-------------------|----------|
| **MAE** | Original | [0, ∞) | Low | Interpretability, robustness |
| **MSE** | Squared | [0, ∞) | High | Optimization, penalizing large errors |
| **RMSE** | Original | [0, ∞) | High | Standard reporting, interpretable |
| **R²** | None | (-∞, 1] | Medium | Model comparison, variance explained |
| **Adj R²** | None | (-∞, 1] | Medium | Feature selection, model complexity |

---

## Implementation from Scratch

### Complete Linear Regression Class

```python
import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionScratch:
    """
    Linear Regression implementation from scratch using Gradient Descent.
    
    Parameters:
    -----------
    learning_rate : float, default=0.01
        Step size for gradient descent
    n_iterations : int, default=1000
        Number of training iterations
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.b0 = None  # Intercept
        self.b1 = None  # Coefficients
        self.cost_history = []
        
    def fit(self, X, y):
        """
        Train the model using gradient descent.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        """
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Handle 1D array (single feature)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Get dimensions
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.b0 = 0  # Intercept
        self.b1 = np.zeros(n_features)  # Coefficients
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Predictions
            y_pred = self.predict(X)
            
            # Calculate cost
            cost = (1/(2*n_samples)) * np.sum((y_pred - y)**2)
            self.cost_history.append(cost)
            
            # Calculate gradients
            error = y_pred - y
            gradient_b0 = (1/n_samples) * np.sum(error)
            gradient_b1 = (1/n_samples) * np.dot(X.T, error)
            
            # Update parameters
            self.b0 -= self.learning_rate * gradient_b0
            self.b1 -= self.learning_rate * gradient_b1
            
            # Print progress every 100 iterations
            if (i+1) % 100 == 0:
                print(f"Iteration {i+1}/{self.n_iterations}, Cost: {cost:.4f}")
        
        return self
    
    def predict(self, X):
        """Make predictions using the trained model."""
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        return self.b0 + np.dot(X, self.b1)
    
    def score(self, X, y):
        """Calculate R² score."""
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y))**2)
        ss_residual = np.sum((y - y_pred)**2)
        r2 = 1 - (ss_residual / ss_total)
        return r2
    
    def plot_cost_history(self):
        """Plot cost function over iterations."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.cost_history)+1), self.cost_history)
        plt.xlabel('Iteration')
        plt.ylabel('Cost J(b)')
        plt.title('Cost Function vs Iterations')
        plt.grid(True)
        plt.show()
```

### Evaluation Metrics Implementation

```python
def calculate_metrics(y_true, y_pred):
    """Calculate all regression metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Mean Squared Error
    mse = np.mean((y_true - y_pred)**2)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # R-Squared
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_residual = np.sum((y_true - y_pred)**2)
    r2 = 1 - (ss_residual / ss_total)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2
    }

def adjusted_r2(r2, n_samples, n_features):
    """Calculate Adjusted R²."""
    adj_r2 = 1 - ((1 - r2) * (n_samples - 1) / (n_samples - n_features - 1))
    return adj_r2
```

---

## Implementation with Scikit-Learn

### Basic Usage

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Get parameters
intercept = model.intercept_
coefficients = model.coef_

print(f"Intercept (b0): {intercept}")
print(f"Coefficients (b1): {coefficients}")

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
```

---

## Complete End-to-End Examples

### Example 1: Student Score Prediction (Simple Linear Regression)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Create sample data
np.random.seed(42)
hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
scores = np.array([50, 55, 65, 75, 90, 88, 92, 95, 98, 100])
scores = scores + np.random.normal(0, 3, size=len(scores))

# Step 2: Prepare data
X = hours.reshape(-1, 1)
y = scores

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Train model
model_scratch = LinearRegressionScratch(learning_rate=0.01, n_iterations=1000)
model_scratch.fit(X_train, y_train)

print(f"\nLearned Parameters:")
print(f"Intercept (b0): {model_scratch.b0:.4f}")
print(f"Coefficient (b1): {model_scratch.b1[0]:.4f}")

# Step 4: Evaluate
y_pred_test = model_scratch.predict(X_test)
test_metrics = calculate_metrics(y_test, y_pred_test)

print(f"\nTest Set Performance:")
for metric, value in test_metrics.items():
    print(f"{metric}: {value:.4f}")

# Step 5: Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', s=100, alpha=0.7, label='Training Data')
plt.scatter(X_test, y_test, color='green', s=100, alpha=0.7, label='Test Data')
plt.plot(X, model_scratch.predict(X), color='red', linewidth=2, label='Best Fit Line')
plt.xlabel('Study Hours')
plt.ylabel('Score')
plt.title('Linear Regression: Study Hours vs Score')
plt.legend()
plt.grid(True)
plt.show()
```

### Example 2: House Price Prediction (Multiple Linear Regression)

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Create dataset
np.random.seed(42)
n_samples = 200

# Features: Area (sqft), Bedrooms, Age (years)
area = np.random.uniform(500, 3000, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
age = np.random.uniform(0, 30, n_samples)

# Target: Price
price = (300 * area + 50000 * bedrooms - 2000 * age + 100000 + 
         np.random.normal(0, 50000, n_samples))

# Create DataFrame
df = pd.DataFrame({
    'Area': area,
    'Bedrooms': bedrooms,
    'Age': age,
    'Price': price
})

# Step 2: Prepare data
X = df[['Area', 'Bedrooms', 'Age']].values
y = df['Price'].values

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Step 4: Evaluate
y_pred_test = model.predict(X_test_scaled)

print(f"\nModel Parameters:")
print(f"Intercept (b0): ${model.intercept_:,.2f}")
print(f"\nCoefficients:")
features = ['Area', 'Bedrooms', 'Age']
for feature, coef in zip(features, model.coef_):
    print(f"  {feature}: ${coef:,.2f}")

print(f"\nTest Set Metrics:")
print(f"MAE: ${mean_absolute_error(y_test, y_pred_test):,.2f}")
print(f"RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred_test)):,.2f}")
print(f"R²: {r2_score(y_test, y_pred_test):.4f}")
```

---

## Model Evaluation and Interpretation

### 1. Checking Model Assumptions

```python
import scipy.stats as stats

def check_assumptions(y_true, y_pred):
    """Check linear regression assumptions."""
    residuals = y_true - y_pred
    
    print("Assumption Checks")
    print("=" * 50)
    
    # 1. Linearity
    print(f"\n1. Linearity: R² = {r2_score(y_true, y_pred):.4f}")
    
    # 2. Normality of residuals
    _, p_value = stats.shapiro(residuals)
    print(f"2. Normality: Shapiro-Wilk p-value = {p_value:.4f}")
    
    # 3. Homoscedasticity
    print(f"3. Homoscedasticity: Check residual plot")
    
    # 4. Independence
    dw = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
    print(f"4. Independence: Durbin-Watson = {dw:.4f}")
    
    # Residual plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True)
    plt.show()
```

### 2. Cross-Validation

```python
from sklearn.model_selection import cross_val_score, KFold

def cross_validate_model(X, y, n_folds=5):
    """Perform k-fold cross-validation."""
    model = LinearRegression()
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # R² scores
    r2_scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')
    
    # MSE scores
    mse_scores = -cross_val_score(model, X, y, cv=kfold, 
                                   scoring='neg_mean_squared_error')
    
    print(f"\n{n_folds}-Fold Cross-Validation Results:")
    print(f"Mean R²: {r2_scores.mean():.4f} (+/- {r2_scores.std() * 2:.4f})")
    print(f"Mean RMSE: {np.sqrt(mse_scores).mean():.4f}")
    
    return r2_scores, mse_scores
```

---

# Interview Questions

## Interview Questions: Fundamentals

**Q1: What is Linear Regression and when would you use it?**

**Answer**: Linear regression is a supervised learning algorithm that models the relationship between dependent and independent variables by fitting a linear equation. Use it when:
- The relationship is approximately linear
- You need to predict continuous values
- You want interpretable results
- Example: Predicting house prices, sales forecasting

---

**Q2: Explain the difference between simple and multiple linear regression.**

**Answer**: 
- **Simple Linear Regression**: Uses one independent variable. Equation: $y = mx + c$
- **Multiple Linear Regression**: Uses two or more independent variables. Equation: $y = b_0 + b_1 x_1 + b_2 x_2 + ... + b_n x_n$

---

**Q3: What is the cost function in linear regression? Why do we use it?**

**Answer**: The cost function (Mean Squared Error) measures how well our model's predictions match actual values:

$$J(b) = \frac{1}{2m} \sum_{i=1}^{m} (h_b(x^{(i)}) - y^{(i)})^2$$

We use it to:
- Quantify prediction error
- Guide parameter optimization
- Evaluate model performance
- Find the best fit line by minimizing this function

---

**Q4: What does the slope (m or b₁) represent in linear regression?**

**Answer**: The slope represents the rate of change of the dependent variable with respect to the independent variable. For every unit increase in x, y changes by m units.

Example: If predicting salary based on years of experience, slope = 5000 means each additional year increases salary by $5000.

---

**Q5: What is the significance of the intercept (c or b₀)?**

**Answer**: The intercept is the predicted value when all independent variables are zero. It represents the baseline value of the dependent variable.

Example: In salary prediction, if intercept = 30000, it means starting salary (with 0 years experience) is $30,000.

---

**Q6: Why do we square the errors in the cost function instead of using absolute values?**

**Answer**: 
1. **Eliminates sign issues**: Positive and negative errors don't cancel out
2. **Penalizes outliers more**: Large errors are punished quadratically
3. **Mathematical properties**: Differentiable everywhere, convex function
4. **Unique solution**: Guarantees a single global minimum
5. **Computational efficiency**: Easier to optimize using gradient descent

---

**Q7: What are the assumptions of linear regression?**

**Answer**:
1. **Linearity**: Linear relationship between features and target
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant variance of residuals
4. **Normality**: Residuals are normally distributed
5. **No multicollinearity**: Features are not highly correlated

---

**Q8: How do you evaluate a linear regression model?**

**Answer**: Common metrics include:
- **R² Score**: Proportion of variance explained (0 to 1, higher is better)
- **Mean Squared Error (MSE)**: Average squared difference
- **Root Mean Squared Error (RMSE)**: Square root of MSE (same units as target)
- **Mean Absolute Error (MAE)**: Average absolute difference
- **Residual plots**: Visual check for patterns

---

**Q9: Can linear regression be used for classification problems?**

**Answer**: Not recommended. Linear regression outputs continuous values, while classification requires discrete classes. Issues:
- No bounded output (can predict values < 0 or > 1)
- Treats classes as ordered numbers
- Better alternatives: Logistic Regression, Decision Trees, SVM

---

**Q10: What happens if features are on different scales?**

**Answer**: Features with larger scales dominate the model. Solutions:
- **Standardization**: Mean = 0, Standard Deviation = 1
- **Normalization**: Scale to [0, 1] range
- **Important for**: Gradient descent convergence, regularization, interpretation

---

## Interview Questions: Gradient Descent

**Q11: What is Gradient Descent and why do we use it?**

**Answer**: Gradient Descent is an iterative optimization algorithm that minimizes the cost function by updating parameters in the direction opposite to the gradient. We use it because:
- Finding analytical solutions is computationally expensive for large datasets
- Works for any differentiable cost function
- Scalable to millions of parameters
- Foundation for training neural networks

---

**Q12: Explain the update rule for Gradient Descent.**

**Answer**: The update rule is:

$$b_k = b_k - \alpha \cdot \frac{\partial J(b)}{\partial b_k}$$

This means:
- Calculate the gradient (slope) of cost function with respect to parameter
- Multiply by learning rate to control step size
- Subtract from current parameter value (move opposite to gradient)
- Repeat until convergence

---

**Q13: What is the learning rate and why is it important?**

**Answer**: Learning rate ($\alpha$) controls the step size in gradient descent. It's important because:
- **Too small**: Slow convergence, many iterations needed
- **Too large**: Overshooting, divergence, never converges
- **Optimal**: Efficient convergence in reasonable time
- Common values: 0.001, 0.01, 0.1

---

**Q14: How do you know when Gradient Descent has converged?**

**Answer**: Convergence indicators:
1. Cost function change is very small: $|J(b)_{new} - J(b)_{old}| < \epsilon$
2. Parameters stop changing significantly
3. Gradient approaches zero
4. Reached maximum iterations
5. Cost vs iteration plot flattens out

Typical threshold $\epsilon$ = 0.0001 or 0.00001

---

**Q15: What is the difference between Batch, Stochastic, and Mini-Batch Gradient Descent?**

**Answer**:

| Type | Examples Used | Speed | Convergence | Use Case |
|------|--------------|-------|-------------|----------|
| **Batch GD** | All (m) | Slow | Smooth, stable | Small datasets |
| **Stochastic GD** | 1 | Fast | Noisy, oscillates | Large datasets |
| **Mini-Batch GD** | 32-256 | Moderate | Balance | Most common |

---

**Q16: Why is feature scaling important for Gradient Descent?**

**Answer**: Without scaling, features with larger ranges dominate the gradient, causing:
- Elongated cost function contours
- Slow, inefficient convergence
- Oscillating path to minimum

**Solution**: Standardize features to similar scales (mean=0, std=1) for faster, more direct convergence.

---

**Q17: Can Gradient Descent get stuck in local minima?**

**Answer**: 
- For **linear regression**: No. The cost function (MSE) is **convex** (bowl-shaped), so there's only one global minimum
- For **non-convex functions** (e.g., neural networks): Yes, can get stuck in local minima or saddle points
- Solutions: Random initialization, momentum, adaptive learning rates

---

**Q18: What happens if the learning rate is too large?**

**Answer**: 
- Parameters overshoot the minimum
- Cost function may **increase** instead of decrease
- Algorithm **diverges** (never converges)
- Oscillates around the minimum
- Solution: Reduce learning rate, monitor cost function

---

**Q19: How many iterations does Gradient Descent typically need?**

**Answer**: Depends on:
- Dataset size
- Learning rate
- Feature scaling
- Convergence criteria

**Typical range**: 100 - 10,000 iterations
- With proper learning rate and scaling: 100-1000
- Monitor cost function to determine when to stop

---

**Q20: How does Gradient Descent differ from Normal Equation?**

**Answer**:

| Aspect | Gradient Descent | Normal Equation |
|--------|-----------------|-----------------|
| **Type** | Iterative | Analytical |
| **Speed** | Fast for large n | Slow for large n |
| **Complexity** | O(iterations × m × n) | O(n³) |
| **Scaling** | Needed | Not needed |
| **Works when** | Always | n is small (<10,000) |

---

## Interview Questions: Implementation

**Q21: What is the difference between MAE and RMSE? When would you use each?**

**Answer**: 
- **MAE**: Average of absolute errors. Less sensitive to outliers. Use when outliers should not dominate.
- **RMSE**: Square root of average squared errors. More sensitive to outliers. Use when large errors are particularly costly.

Key difference: RMSE penalizes large errors more heavily than MAE.

---

**Q22: Explain what R² score represents and how to interpret it.**

**Answer**: R² measures the proportion of variance in the dependent variable explained by the model.

**Interpretation**:
- R² = 1.0: Perfect predictions (100% variance explained)
- R² = 0.8: Model explains 80% of variance
- R² = 0.0: Model no better than predicting the mean
- R² < 0: Model worse than predicting the mean

---

**Q23: Why do we need Adjusted R²? How is it different from R²?**

**Answer**: Adjusted R² penalizes model complexity by accounting for the number of features.

**Problem with R²**: Always increases when adding features, even if they're irrelevant.

**Adjusted R²**:
- Only increases if new feature improves the model more than expected by chance
- Decreases if new feature doesn't contribute enough
- Better for comparing models with different numbers of features

---

**Q24: What does it mean if RMSE is much larger than MAE?**

**Answer**: 
- RMSE ≥ MAE always (due to squaring)
- If RMSE >> MAE: Large outliers or high variance in errors
- If RMSE ≈ MAE: Errors are relatively uniform

This indicates you should investigate outliers.

---

**Q25: Can R² be negative? What does it mean?**

**Answer**: Yes, R² can be negative.

**Meaning**: Model predictions are worse than simply predicting the mean of y.

**Causes**:
- Model is very poor
- Wrong model for the data
- Evaluated on different dataset than trained on
- Severe overfitting

---

**Q26: How do you handle outliers in linear regression?**

**Answer**:
1. **Identify**: Use residual plots, box plots, z-scores
2. **Investigate**: Determine if outliers are errors or valid extreme values
3. **Options**:
   - Remove if data errors
   - Keep if valid extreme cases
   - Use robust regression techniques
   - Transform features (log, sqrt)
   - Use MAE instead of MSE

---

**Q27: What is a residual plot and why is it important?**

**Answer**: A residual plot shows residuals (errors) vs predicted values.

**What to look for**:
- **Random scatter around 0**: Good, assumptions met ✓
- **Curved pattern**: Non-linear relationship
- **Funnel shape**: Heteroscedasticity (non-constant variance)
- **Outliers**: Points far from zero line

---

**Q28: Walk me through the complete workflow of building a linear regression model.**

**Answer**:

```
1. Data Collection and Exploration
   - Load data, check shape, missing values
   - Exploratory data analysis (EDA)

2. Data Preprocessing
   - Handle missing values
   - Encode categorical variables
   - Feature scaling
   - Split into train/test sets

3. Model Training
   - Choose implementation
   - Fit model on training data
   - Learn parameters

4. Model Evaluation
   - Make predictions on test set
   - Calculate metrics (MAE, RMSE, R²)
   - Check assumptions

5. Model Interpretation
   - Analyze coefficients
   - Feature importance
   - Validate assumptions

6. Deployment
   - Save model
   - Monitor performance
```

---

**Q29: How do you interpret the coefficients in a linear regression model?**

**Answer**:

For model: $y = b_0 + b_1 x_1 + b_2 x_2$

- **Intercept ($b_0$)**: Predicted value when all features are zero
- **Coefficient ($b_i$)**: Change in y for one unit increase in $x_i$, holding other features constant

**Example**: House price = 100,000 + 300(Area) + 50,000(Bedrooms)
- b₀ = $100,000: Base price
- b₁ = $300: Each additional sq ft increases price by $300
- b₂ = $50,000: Each additional bedroom increases price by $50,000

---

**Q30: How do you know if your model is overfitting or underfitting?**

**Answer**:

**Overfitting**:
- Training R² high, Test R² low
- Training error << Test error
- **Solutions**: Reduce features, regularization, more data

**Underfitting**:
- Both training and test R² low
- Both errors high
- **Solutions**: Add features, polynomial terms, try different model

**Good Fit**:
- Training R² ≈ Test R²
- Both errors acceptably low

---

## Summary and Best Practices

### Complete Linear Regression Workflow

**1. Fundamentals**
- Understand hypothesis function: $h_b(x) = b_0 + b_1 x_1 + ... + b_n x_n$
- Know cost function: $J(b) = \frac{1}{2m} \sum(h_b(x^{(i)}) - y^{(i)})^2$
- Recognize when to use linear regression

**2. Optimization**
- Use gradient descent to minimize cost
- Choose appropriate learning rate
- Monitor convergence
- Apply feature scaling

**3. Implementation**
- Can implement from scratch or use Scikit-Learn
- Always split data (train/test)
- Calculate multiple evaluation metrics
- Visualize results

**4. Evaluation**
- Use MAE, RMSE, R² to assess performance
- Check model assumptions with residual plots
- Compare training vs test error
- Apply cross-validation

### Key Takeaways

1. **Linear Regression** models linear relationships using $y = b_0 + b_1 x_1 + ...$
2. **Cost Function** (MSE) quantifies prediction error
3. **Gradient Descent** optimizes parameters iteratively
4. **Learning Rate** controls convergence speed and stability
5. **Feature Scaling** improves gradient descent efficiency
6. **Evaluation Metrics** (MAE, RMSE, R²) measure model quality
7. **Assumptions** must be validated for reliable results
8. **Regularization** prevents overfitting (Ridge, Lasso)

### Best Practices

1. **Data Preparation**
   - Handle missing values
   - Encode categorical variables
   - Scale features
   - Split train/test

2. **Training**
   - Start with simple model
   - Monitor cost function
   - Use appropriate learning rate
   - Check convergence

3. **Evaluation**
   - Use multiple metrics
   - Check assumptions
   - Validate with cross-validation
   - Compare train/test performance

4. **Interpretation**
   - Analyze coefficients
   - Understand feature importance
   - Visualize results
   - Check residuals

### Common Pitfalls to Avoid

1. ❌ Not scaling features
2. ❌ Using linear regression for non-linear data
3. ❌ Ignoring outliers
4. ❌ Not checking assumptions
5. ❌ Overfitting with too many features
6. ❌ Using only one evaluation metric
7. ❌ Not validating on test set

### Next Steps

- Practice with real datasets
- Learn regularization techniques (Ridge, Lasso, ElasticNet)
- Explore polynomial regression
- Study advanced optimization algorithms (Adam, RMSprop)
- Understand multicollinearity and VIF
- Learn feature selection techniques

---

**Document Version**: 2.0  
**Last Updated**: January 12, 2026  
**Topic**: Linear Regression - Complete Guide  
**Coverage**: Fundamentals, Optimization, Implementation, and Evaluation
