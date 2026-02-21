# Stochastic Gradient Descent - Complete Guide

## 📚 Table of Contents

1. [What is Stochastic Gradient Descent?](#what-is-stochastic-gradient-descent)
2. [Mathematical Intuition](#mathematical-intuition)
3. [How It Works - Step by Step](#how-it-works---step-by-step)
4. [Implementation Details](#implementation-details)
5. [Advantages and Disadvantages](#advantages-and-disadvantages)
6. [When to Use Stochastic GD](#when-to-use-stochastic-gd)
7. [Comparison with Other Methods](#comparison-with-other-methods)

---

## What is Stochastic Gradient Descent?

**Stochastic Gradient Descent (SGD)** is an optimization algorithm used to minimize the cost function in machine learning models by updating model parameters using **one random training sample** at a time in each iteration.

### Note:

It never really converges to the minimum but keeps oscillating around it due to the noisy updates from single samples.

To solve this we use learning rate Schedule (commonly used technique in Deep Learning).

Learning Rate Schedule : It is a technique where the learning rate is adjusted during training, typically decreasing it over time to allow the model to make finer adjustments as it approaches the minimum.

**Example:** Start with learning rate 0.1, then reduce it to 0.01 after 10 epochs, then 0.001 after 20 epochs, and so on.

```python
# Simple learning rate schedule
initial_lr = 0.1
epoch = 15

# After every 10 epochs, multiply learning rate by 0.1
current_lr = initial_lr * (0.1 ** (epoch // 10))
# epoch 0-9: lr = 0.1
# epoch 10-19: lr = 0.01
# epoch 20-29: lr = 0.001
```

### 🔑 Key Characteristics:

- Uses **one training sample** to compute gradients
- Updates parameters **m times per epoch** (where m = number of samples)
- Provides **fast but noisy** convergence
- **Memory efficient** for large datasets
- **Fast updates** with immediate learning

### Note:

If number of epoch is fixed ( let 100 ) then SGD will do total of 100*m updates where m is number of samples in dataset.
Whereas Batch GD does only 100 updates in this case.
and mini-batch GD does (100 * m)/k updates where k is the mini-batch size.

- **So for same number of epochs, SGD makes significantly more updates than Batch GD and Mini-Batch GD, hence it takes more time overall.**

- **Here we can't guarantee that the next update will be better than previous one because we are taking only one sample at a time so it may lead to increase in cost function sometimes.**

### 📊 Visual Intuition:

Imagine you're standing on a mountain (error surface) blindfolded and want to reach the valley (minimum error). Stochastic Gradient Descent:
- Takes **many quick steps** based on **one data point at a time**
- Each step uses information from **only one sample**
- Moves in a **noisy, zigzag direction** but generally toward the minimum
- Takes **many fast but imprecise steps**
- Can **escape local minima** due to randomness

---

## Mathematical Intuition

### Cost Function (Mean Squared Error)

For linear regression, we minimize:

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

**But in SGD**, we approximate this using **one sample at a time**:

$$J^{(i)}(\theta) = \frac{1}{2} (h_\theta(x^{(i)}) - y^{(i)})^2$$

Where:
- $J^{(i)}(\theta)$ = Cost for the $i$-th sample
- $h_\theta(x^{(i)})$ = Predicted value for $i$-th sample
- $y^{(i)}$ = Actual value for $i$-th sample
- $\theta$ = Model parameters (weights and bias)

### Prediction Formula

For a single sample:

$$\hat{y}^{(i)} = \theta_0 + \theta_1 x_1^{(i)} + \theta_2 x_2^{(i)} + ... + \theta_n x_n^{(i)}$$

Or in vector form:

$$\hat{y}^{(i)} = x^{(i)}\theta + b$$

### Gradient Calculation

The gradient is computed using **only one sample** at a time:

**For coefficients (weights):**

$$\frac{\partial J^{(i)}}{\partial \theta_j} = (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}$$

**In vectorized form for one sample:**

$$\nabla_\theta J^{(i)} = (x^{(i)}\theta - y^{(i)}) \cdot x^{(i)}$$

**For intercept (bias):**

$$\frac{\partial J^{(i)}}{\partial b} = h_\theta(x^{(i)}) - y^{(i)}$$

### Parameter Update Rule

After computing gradients using **ONE** training sample:

$$\theta_j := \theta_j - \alpha \frac{\partial J^{(i)}}{\partial \theta_j}$$

$$b := b - \alpha \frac{\partial J^{(i)}}{\partial b}$$

Where:
- $\alpha$ = Learning rate (step size)
- Updates happen **immediately** after each sample
- Parameters get updated **m times per epoch**

### Key Difference from Batch GD

| Aspect | Batch GD | Stochastic GD |
|--------|----------|---------------|
| Gradient calculation | Uses **all m samples** | Uses **1 sample** |
| Updates per epoch | **1** | **m** |
| Formula | $\nabla J = \frac{1}{m}\sum_{i=1}^{m}$ | $\nabla J^{(i)}$ for one sample |

---

## How It Works - Step by Step

### Algorithm Pseudocode

```
1. Initialize parameters randomly or to zero:
   θ = [0, 0, ..., 0]  or  θ = small random values
   b = 0

2. For epoch = 1 to max_epochs:
   
   a) Shuffle the training data (important!)
   
   b) For each sample i = 1 to m:
      
      i.   Pick one sample: (x⁽ⁱ⁾, y⁽ⁱ⁾)
      
      ii.  Compute prediction for THIS sample:
           ŷ⁽ⁱ⁾ = x⁽ⁱ⁾θ + b
      
      iii. Compute error for THIS sample:
           error⁽ⁱ⁾ = y⁽ⁱ⁾ - ŷ⁽ⁱ⁾
      
      iv.  Compute gradients using THIS sample:
           ∂J/∂θ = -2 * error⁽ⁱ⁾ * x⁽ⁱ⁾
           ∂J/∂b = -2 * error⁽ⁱ⁾
      
      v.   Update parameters IMMEDIATELY:
           θ = θ - α * (∂J/∂θ)
           b = b - α * (∂J/∂b)

3. Return learned parameters θ and b
```

### 🔍 Detailed Walkthrough with Example

Let's use the same dataset:

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

#### Epoch 1, Sample 1 (x₁=1, x₂=2, y=5)

**Step 1: Compute Prediction (for THIS sample only)**
```
ŷ₁ = 1×1 + 2×1 + 0 = 3
```

**Step 2: Compute Error (for THIS sample only)**
```
e₁ = 5 - 3 = 2
```

**Step 3: Compute Gradients (using THIS sample only)**
```
∂J/∂θ₁ = -2 × 2 × 1 = -4
∂J/∂θ₂ = -2 × 2 × 2 = -8
∂J/∂b = -2 × 2 = -4
```

**Step 4: Update Parameters IMMEDIATELY**
```
θ₁ = 1.0 - 0.1 × (-4) = 1.4
θ₂ = 1.0 - 0.1 × (-8) = 1.8
b = 0.0 - 0.1 × (-4) = 0.4
```

#### Epoch 1, Sample 2 (x₁=2, x₂=3, y=8)

**Step 1: Compute Prediction (with updated parameters)**
```
ŷ₂ = 2×1.4 + 3×1.8 + 0.4 = 8.6
```

**Step 2: Compute Error**
```
e₂ = 8 - 8.6 = -0.6
```

**Step 3: Compute Gradients**
```
∂J/∂θ₁ = -2 × (-0.6) × 2 = 2.4
∂J/∂θ₂ = -2 × (-0.6) × 3 = 3.6
∂J/∂b = -2 × (-0.6) = 1.2
```

**Step 4: Update Parameters IMMEDIATELY**
```
θ₁ = 1.4 - 0.1 × 2.4 = 1.16
θ₂ = 1.8 - 0.1 × 3.6 = 1.44
b = 0.4 - 0.1 × 1.2 = 0.28
```

#### Epoch 1, Sample 3 (x₁=3, x₂=4, y=11)

**...and so on**

**Key Observations:** 
- We process **ONE sample at a time**
- We make **ONE update per sample**
- Parameters change **3 times** in this epoch (once per sample)
- Next epoch: **shuffle data** and repeat

---

## Implementation Details

### Code Structure

```python
class SGDRegressor:
    
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
            
            # Shuffle the data for each epoch
            indices = np.random.permutation(X_train.shape[0])
            
            # Process ONE sample at a time
            for idx in indices:
                # Get single sample
                x_i = X_train[idx]
                y_i = y_train[idx]
                
                # 1. Compute prediction for THIS sample
                y_hat = np.dot(x_i, self.coef_) + self.intercept_
                
                # 2. Compute error for THIS sample
                error = y_i - y_hat
                
                # 3. Compute gradient for intercept
                intercept_der = -2 * error
                
                # 4. Update intercept IMMEDIATELY
                self.intercept_ = self.intercept_ - (self.lr * intercept_der)
                
                # 5. Compute gradient for coefficients
                coef_der = -2 * error * x_i
                
                # 6. Update coefficients IMMEDIATELY
                self.coef_ = self.coef_ - (self.lr * coef_der)
    
    def predict(self, X_test):
        return np.dot(X_test, self.coef_) + self.intercept_
```

### 🧮 Operations Breakdown

**1. Data Shuffling:**
```python
indices = np.random.permutation(X_train.shape[0])
```
- Creates random permutation of indices [0, 1, 2, ..., m-1]
- Ensures each epoch sees data in different order
- **Critical** for preventing cycles and improving convergence

**2. Single Sample Selection:**
```python
x_i = X_train[idx]  # shape: (n,) - one row
y_i = y_train[idx]  # scalar - one value
```

**3. Prediction (scalar):**
```python
y_hat = np.dot(x_i, self.coef_) + self.intercept_
```
- `x_i`: shape (n,) - single sample
- `self.coef_`: shape (n,)
- `y_hat`: scalar - single prediction

**4. Gradients (vectors):**
```python
intercept_der = -2 * error  # scalar
coef_der = -2 * error * x_i  # shape (n,)
```

**5. Immediate Updates:**
```python
self.intercept_ -= self.lr * intercept_der
self.coef_ -= self.lr * coef_der
```
- Happens **m times per epoch**
- No averaging needed

---

## Advantages and Disadvantages

### ✅ Advantages

1. **Fast Updates**
    - Learns immediately from each sample
    - Parameters updated m times per epoch
    - Quick convergence for large datasets

2. **Memory Efficient**
    - Processes one sample at a time
    - No need to load entire dataset
    - Suitable for streaming/online learning

3. **Escapes Local Minima**
    - Noisy updates help escape saddle points
    - Better for non-convex problems
    - Randomness aids exploration

4. **Scalable**
    - Works with millions of samples
    - Can handle data that doesn't fit in memory
    - Suitable for online/incremental learning

### ❌ Disadvantages

1. **Noisy Convergence**
    - Erratic path to minimum
    - Never truly converges (oscillates around minimum)
    - Requires learning rate decay

2. **Slower Computation**
    - Cannot vectorize efficiently
    - Many sequential operations
    - Harder to parallelize

3. **Hyperparameter Sensitivity**
    - Learning rate critical (too high = diverge, too low = slow)
    - Requires careful tuning
    - May need learning rate schedules

4. **Variance in Results**
    - Different runs give different results
    - Depends on shuffling order
    - Less reproducible

---

## When to Use Stochastic GD

### ✅ Best Use Cases

- **Large datasets** (> 100,000 samples) that don't fit in memory
- **Online learning** with streaming data
- **Non-convex problems** (neural networks)
- **When fast initial progress** is needed
- **Memory-constrained** environments

### ❌ Avoid When

- **Small datasets** (< 10,000 samples) - use Batch GD instead
- **Require stable convergence** - use Mini-Batch GD
- **Need reproducibility** - Mini-Batch gives better consistency
- **GPU acceleration** available - Mini-Batch is more efficient

---

## Comparison with Other Methods

### SGD vs Batch GD vs Mini-Batch GD

| Aspect | Batch GD | Stochastic GD | Mini-Batch GD |
|--------|----------|---------------|---------------|
| **Samples per update** | All (m) | 1 | k (32-512) |
| **Updates per epoch** | 1 | m | m/k |
| **Convergence** | Smooth | Very noisy | Moderately noisy |
| **Speed per update** | Slow | Very fast | Fast |
| **Memory** | High | Very low | Medium |
| **Accuracy** | Best | Lowest | Good |
| **Vectorization** | Excellent | None | Good |
| **Parallelization** | Easy | Hard | Easy |
| **Use case** | Small data | Very large data | Most common |

### ⚙️ Key Parameters for `SGDRegressor` in Scikit-Learn

```python
from sklearn.linear_model import SGDRegressor

# Basic usage
reg = SGDRegressor(max_iter=100, learning_rate='constant', eta0=0.01)
```

**Important Parameters:**

| Parameter | Options | Description |
|-----------|---------|-------------|
| `loss` | `'squared_error'` (default), `'huber'`, `'epsilon_insensitive'` | Loss function to minimize |
| `penalty` | `'l2'` (default), `'l1'`, `'elasticnet'`, `None` | Regularization type |
| `alpha` | Default: `0.0001` | Regularization strength |
| `max_iter` | Default: `1000` | Number of epochs |
| `learning_rate` | `'constant'`, `'optimal'`, `'invscaling'`, `'adaptive'` | Learning rate schedule |
| `eta0` | Default: `0.01` | Initial learning rate |

**Loss Functions:**
- `squared_error`: Standard MSE (L2 loss)
- `huber`: Robust to outliers (combines L1 and L2)
- `epsilon_insensitive`: Ignores errors within epsilon (SVR-like)

**Penalty (Regularization):**
- `l2`: Ridge regression (default)
- `l1`: Lasso regression (feature selection)
- `elasticnet`: Combination of L1 and L2
- `None`: No regularization

### 📝 Quick Example

```python
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# Always scale your data!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# L2 regularization (Ridge)
reg = SGDRegressor(loss='squared_error', penalty='l2', alpha=0.001, 
                   max_iter=1000, learning_rate='constant', eta0=0.01)
reg.fit(X_train_scaled, y_train)

# L1 regularization (Lasso)
reg = SGDRegressor(loss='squared_error', penalty='l1', alpha=0.001)

# Elastic Net
reg = SGDRegressor(penalty='elasticnet', alpha=0.001, l1_ratio=0.5)
```

⚠️ **Critical:** Always use `StandardScaler()` before SGD! because SGD is sensitive to feature scaling.

**Why ?**

- Features on different scales can lead to inefficient updates , meaning some features dominate the gradient calculations
- Scaling ensures uniform contribution to gradients , so that each feature influences the model equally
- Improves convergence speed and stability .
