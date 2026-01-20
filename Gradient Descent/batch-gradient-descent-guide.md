# Batch Gradient Descent - Complete Guide

## 📚 Table of Contents

1. [What is Batch Gradient Descent?](#what-is-batch-gradient-descent)
2. [Mathematical Intuition](#mathematical-intuition)
3. [How It Works - Step by Step](#how-it-works---step-by-step)
4. [Advantages and Disadvantages](#advantages-and-disadvantages)
5. [When to Use Batch GD](#when-to-use-batch-gd)
6. [Comparison with Other Methods](#comparison-with-other-methods)

---

## What is Batch Gradient Descent?

**Batch Gradient Descent** is an optimization algorithm used to minimize the cost function in machine learning models by updating model parameters using the **entire training dataset** in each iteration.

### 🔑 Key Characteristics:
- Uses **all training samples** to compute gradients
- Updates parameters **once per epoch**
- Provides **stable and consistent** convergence
- Computationally **expensive** for large datasets

### 📊 Visual Intuition:

Imagine you're standing on a mountain (error surface) blindfolded and want to reach the valley (minimum error). Batch Gradient Descent:
- Takes **one careful step** after surveying the **entire landscape**
- Each step considers information from **all data points**
- Moves in the **most accurate direction** toward the minimum
- Takes **fewer but more informed steps**

---

## Mathematical Intuition

### Cost Function (Mean Squared Error)

For linear regression, we minimize:

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

Where:
- $J(\theta)$ = Cost function (what we want to minimize)
- $m$ = Number of training samples
- $h_\theta(x^{(i)})$ = Predicted value for $i$-th sample
- $y^{(i)}$ = Actual value for $i$-th sample
- $\theta$ = Model parameters (weights and bias)

### Prediction Formula

$$\hat{y} = X\theta + b$$

Or in expanded form:

$$\hat{y}^{(i)} = \theta_0 + \theta_1 x_1^{(i)} + \theta_2 x_2^{(i)} + ... + \theta_n x_n^{(i)}$$

### Gradient Calculation

The gradient (partial derivative) tells us the direction and magnitude of the steepest ascent:

**For coefficients (weights):**

$$\frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}$$

**In vectorized form:**

$$\nabla_\theta J = \frac{1}{m} X^T (X\theta - y)$$

**For intercept (bias):**

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})$$

### Parameter Update Rule

After computing gradients using **ALL** training data:

$$\theta_j := \theta_j - \alpha \frac{\partial J}{\partial \theta_j}$$

$$b := b - \alpha \frac{\partial J}{\partial b}$$

Where:
- $\alpha$ = Learning rate (step size)
- $:=$ means "update to"
- The minus sign because we want to go **downhill** (minimize)

### Why the Factor of 2?

In implementation, you'll often see:

$$\frac{\partial J}{\partial \theta} = -2 \cdot \frac{X^T(y - \hat{y})}{m}$$

The factor of **2** comes from the derivative of the squared term. However, it's often absorbed into the learning rate, so some implementations use:

$$\frac{\partial J}{\partial \theta} = -\frac{X^T(y - \hat{y})}{m}$$

And simply adjust $\alpha$ accordingly.

---

## How It Works - Step by Step

### Algorithm Pseudocode

```
1. Initialize parameters randomly or to zero:
   θ = [0, 0, ..., 0]  or  θ = small random values
   b = 0

2. For epoch = 1 to max_epochs:
   
   a) Compute predictions for ALL samples:
      ŷ = Xθ + b
   
   b) Compute error for ALL samples:
      error = y - ŷ
   
   c) Compute gradients using ALL samples:
      ∂J/∂θ = -2 * (X^T · error) / m
      ∂J/∂b = -2 * mean(error)
   
   d) Update parameters:
      θ = θ - α * (∂J/∂θ)
      b = b - α * (∂J/∂b)
   
   e) Optionally: compute and store cost
      J = (1/2m) * sum(error^2)

3. Return learned parameters θ and b
```

### 🔍 Detailed Walkthrough with Example

Let's say we have a simple dataset:

| Sample | $x_1$ | $x_2$ | $y$ (actual) |
|--------|-------|-------|--------------|
| 1      | 1     | 2     | 5            |
| 2      | 2     | 3     | 8            |
| 3      | 3     | 4     | 11           |

**Step 0: Initialize**
```
θ₁ = 1.0
θ₂ = 1.0
b = 0.0
α = 0.1
```

**Step 1: Compute Predictions (using ALL 3 samples)**
```
ŷ₁ = 1×1 + 2×1 + 0 = 3
ŷ₂ = 2×1 + 3×1 + 0 = 5
ŷ₃ = 3×1 + 4×1 + 0 = 7
```

**Step 2: Compute Errors (for ALL samples)**
```
e₁ = 5 - 3 = 2
e₂ = 8 - 5 = 3
e₃ = 11 - 7 = 4
```

**Step 3: Compute Gradients (using ALL errors)**
```
∂J/∂θ₁ = -2/3 × (2×1 + 3×2 + 4×3) = -2/3 × 20 = -13.33
∂J/∂θ₂ = -2/3 × (2×2 + 3×3 + 4×4) = -2/3 × 29 = -19.33
∂J/∂b = -2/3 × (2 + 3 + 4) = -6.0
```

**Step 4: Update Parameters**
```
θ₁ = 1.0 - 0.1 × (-13.33) = 2.333
θ₂ = 1.0 - 0.1 × (-19.33) = 2.933
b = 0.0 - 0.1 × (-6.0) = 0.6
```

**Key Observation:** 
- We used **ALL 3 samples** to compute the gradient
- We made **ONE update** after processing all data
- This is one complete **epoch**

---

## Implementation Details

### Code Structure

```python
class GDRegressor:
    
    def __init__(self, learning_rate=0.01, epochs=100):
        self.coef_ = None
        self.intercept_ = None
        self.lr = learning_rate
        self.epochs = epochs
        
    def fit(self, X_train, y_train):
        # Initialize parameters
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1])
        
        # Training loop
        for i in range(self.epochs):
            # 1. Compute predictions for ALL samples
            y_hat = np.dot(X_train, self.coef_) + self.intercept_
            
            # 2. Compute gradient for intercept using ALL samples
            intercept_der = -2 * np.mean(y_train - y_hat)
            
            # 3. Update intercept
            self.intercept_ = self.intercept_ - (self.lr * intercept_der)
            
            # 4. Compute gradient for coefficients using ALL samples
            coef_der = -2 * np.dot((y_train - y_hat), X_train) / X_train.shape[0]
            
            # 5. Update coefficients
            self.coef_ = self.coef_ - (self.lr * coef_der)
    
    def predict(self, X_test):
        return np.dot(X_test, self.coef_) + self.intercept_
```

### 🧮 Matrix Operations Breakdown

**1. Prediction:**
```python
y_hat = np.dot(X_train, self.coef_) + self.intercept_
```
- `X_train`: shape (m, n) where m=samples, n=features
- `self.coef_`: shape (n,)
- `y_hat`: shape (m,) - predictions for all samples

**2. Intercept Gradient:**
```python
intercept_der = -2 * np.mean(y_train - y_hat)
```
- `y_train - y_hat`: shape (m,) - errors for all samples
- `np.mean()`: averages over ALL samples
- Result: single scalar value

**3. Coefficient Gradient:**
```python
coef_der = -2 * np.dot((y_train - y_hat), X_train) / X_train.shape[0]
```
- `(y_train - y_hat)`: shape (m,)
- `X_train`: shape (m, n)
- `np.dot()`: results in shape (n,)
- Division by m (number of samples) gives average gradient

---

## Advantages and Disadvantages

### ✅ Advantages

1. **Stable Convergence**
    - Accurate gradients using entire dataset
    - Smooth descent path with guaranteed convergence for convex functions

2. **Optimal for Small Datasets**
    - Best accuracy when data fits in memory (< 10,000 samples)
    - No sampling bias or variance

3. **Easy Parallelization**
    - Efficient matrix operations using BLAS/LAPACK
    - Straightforward GPU acceleration

4. **Predictable Behavior**
    - Well-studied theoretical guarantees
    - Minimal hyperparameters (only learning rate and epochs)
    - Reproducible results

### ❌ Disadvantages

1. **Resource Intensive**
    - Must load entire dataset into memory
    - Not feasible for large datasets (> 1GB)
    - Slow scaling with dataset size

2. **Computational Inefficiency**
    - One update per full epoch
    - Redundant computations on similar samples
    - Can take hours for big data

3. **Limited Flexibility**
    - Cannot handle streaming or online learning
    - Requires full retraining for new data
    - Slow adaptation to changing patterns

---

## When to Use Batch GD

### ✅ Best Use Cases

- **Small datasets** (< 10,000 samples) that fit in RAM
- **High accuracy requirements** with deterministic results
- **Convex problems** (linear/logistic regression, linear SVMs)
- **Distributed computing** environments with parallelization
- **Debugging and prototyping** for learning purposes

### ❌ Avoid When

- **Large datasets** (> 100,000 samples) or limited memory
- **Online/streaming learning** with continuous data updates
- **Deep learning** applications (use mini-batch instead)
- **Resource-constrained** environments (mobile/edge devices)

---

## Comparison with Other Methods

### Batch GD vs Stochastic GD vs Mini-Batch GD

| Aspect | Batch GD | Stochastic GD | Mini-Batch GD |
|--------|----------|---------------|---------------|
| **Samples per update** | All (m) | 1 | k (e.g., 32-512) |
| **Updates per epoch** | 1 | m | m/k |
| **Convergence** | Smooth | Noisy | Moderate |
| **Speed** | Slow | Fast | Balanced |
| **Memory** | High | Low | Medium |
| **Accuracy** | Best | Lowest | Good |
| **Use case** | Small data | Large data | Most common |

### Performance Comparison

**Dataset: 100,000 samples, 10 features**

| Method | Time/Epoch | Epochs to Converge | Total Time | Final Accuracy |
|--------|------------|-------------------|------------|----------------|
| Batch GD | 2.5s | 500 | 1250s | 0.95 |
| Stochastic GD | 0.1s | 50 | 5s | 0.92 |
| Mini-Batch (64) | 0.3s | 100 | 30s | 0.94 |

**Winner:** Mini-Batch GD (best balance) ⭐