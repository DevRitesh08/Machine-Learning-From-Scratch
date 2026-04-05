# Mini-Batch Gradient Descent - Complete Guide

## 📚 Table of Contents

1. [What is Mini-Batch Gradient Descent?](#what-is-mini-batch-gradient-descent)
2. [Mathematical Intuition](#mathematical-intuition)
3. [How It Works - Step by Step](#how-it-works---step-by-step)
4. [Implementation Details](#implementation-details)
5. [Advantages and Disadvantages](#advantages-and-disadvantages)
6. [When to Use Mini-Batch GD](#when-to-use-mini-batch-gd)
7. [Comparison with Other Methods](#comparison-with-other-methods)

---

## What is Mini-Batch Gradient Descent?

**Mini-Batch Gradient Descent** is an optimization algorithm used to minimize the cost function in machine learning models by updating model parameters using **a small batch of training samples** in each iteration. It combines the best aspects of both Batch GD and Stochastic GD.

### 🔑 Key Characteristics:
- Uses **a small batch** of training samples (typically 32, 64, 128, 256, or 512)
- Updates parameters **m/k times per epoch** (where m = total samples, k = batch size)
- Provides **balanced convergence** (less noisy than SGD, faster than Batch GD)
- **Vectorization efficient** - utilizes GPU/CPU parallelization
- **Most commonly used** in practice

### 📊 Visual Intuition:

Imagine you're standing on a mountain (error surface) blindfolded and want to reach the valley (minimum error). Mini-Batch Gradient Descent:
- Takes **moderate-sized steps** based on **small groups of data points**
- Each step uses information from **k samples** (not 1, not all)
- Moves in a **slightly noisy but generally accurate direction**
- **Best trade-off** between speed and stability
- Can leverage **hardware acceleration** (GPUs, vectorization)

---

## Mathematical Intuition

### Cost Function (Mean Squared Error)

For linear regression, we minimize:

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

**In Mini-Batch GD**, we approximate this using **a small batch at a time**:

$$J^{(B)}(\theta) = \frac{1}{2k} \sum_{i \in B} (h_\theta(x^{(i)}) - y^{(i)})^2$$

Where:
- $J^{(B)}(\theta)$ = Cost for batch $B$
- $k$ = Batch size (e.g., 32, 64, 128)
- $B$ = Set of indices in the current batch
- $h_\theta(x^{(i)})$ = Predicted value for $i$-th sample in batch
- $y^{(i)}$ = Actual value for $i$-th sample in batch

### Prediction Formula

For a mini-batch:

$$\hat{Y}^{(B)} = X^{(B)}\theta + b$$

Where:
- $X^{(B)}$ = Feature matrix for batch (shape: k × n)
- $\hat{Y}^{(B)}$ = Predictions for batch (shape: k × 1)

### Gradient Calculation

The gradient is computed using **one mini-batch** at a time:

**For coefficients (weights):**

$$\frac{\partial J^{(B)}}{\partial \theta_j} = \frac{1}{k} \sum_{i \in B} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}$$

**In vectorized form for a batch:**

$$\nabla_\theta J^{(B)} = \frac{1}{k} X^{(B)T} (X^{(B)}\theta - Y^{(B)})$$

**For intercept (bias):**

$$\frac{\partial J^{(B)}}{\partial b} = \frac{1}{k} \sum_{i \in B} (h_\theta(x^{(i)}) - y^{(i)})$$

### Parameter Update Rule

After computing gradients using **one mini-batch**:

$$\theta_j := \theta_j - \alpha \frac{\partial J^{(B)}}{\partial \theta_j}$$

$$b := b - \alpha \frac{\partial J^{(B)}}{\partial b}$$

Where:
- $\alpha$ = Learning rate (step size)
- Updates happen after processing each batch
- Parameters get updated **m/k times per epoch**

### Comparison Across All Methods

| Aspect | Batch GD | Mini-Batch GD | Stochastic GD |
|--------|----------|---------------|---------------|
| Samples used | **All m** | **k samples** | **1 sample** |
| Gradient formula | $\frac{1}{m}\sum_{i=1}^{m}$ | $\frac{1}{k}\sum_{i \in B}$ | Single sample |
| Updates per epoch | **1** | **m/k** | **m** |

---

## How It Works - Step by Step

### Algorithm Pseudocode

```
1. Initialize parameters randomly or to zero:
   θ = [0, 0, ..., 0]  or  θ = small random values
   b = 0
   
2. Set batch_size k (e.g., 32, 64, 128)

3. For epoch = 1 to max_epochs:
   
   a) Shuffle the training data (important!)
   
   b) Split data into mini-batches of size k:
      batches = [batch_1, batch_2, ..., batch_(m/k)]
   
   c) For each mini-batch B in batches:
      
      i.   Get batch samples: (X^(B), Y^(B))
           - X^(B): shape (k, n) - k samples, n features
           - Y^(B): shape (k,) - k target values
      
      ii.  Compute predictions for THIS batch:
           Ŷ^(B) = X^(B)θ + b
      
      iii. Compute errors for THIS batch:
           errors^(B) = Y^(B) - Ŷ^(B)
      
      iv.  Compute gradients using THIS batch:
           ∂J/∂θ = -2 * (X^(B)^T · errors^(B)) / k
           ∂J/∂b = -2 * mean(errors^(B))
      
      v.   Update parameters:
           θ = θ - α * (∂J/∂θ)
           b = b - α * (∂J/∂b)

4. Return learned parameters θ and b
```

### 🔍 Detailed Walkthrough with Example

Let's use a small dataset with 8 samples and batch size 3:

| Sample | $x_1$ | $x_2$ | $y$ (actual) |
|--------|-------|-------|--------------|
| 1      | 1     | 2     | 5            |
| 2      | 2     | 3     | 8            |
| 3      | 3     | 4     | 11           |
| 4      | 4     | 5     | 14           |
| 5      | 5     | 6     | 17           |
| 6      | 6     | 7     | 20           |
| 7      | 7     | 8     | 23           |
| 8      | 8     | 9     | 26           |

**Step 0: Initialize**
```
θ₁ = 1.0
θ₂ = 1.0
b = 0.0
α = 0.1
batch_size = 3
```

**Step 1: Shuffle Data**
```
After shuffling: [3, 7, 1, 5, 8, 2, 4, 6]
```

**Step 2: Create Batches**
```
Batch 1: samples [3, 7, 1]  (size 3)
Batch 2: samples [5, 8, 2]  (size 3)
Batch 3: samples [4, 6]     (size 2) - last batch can be smaller
```

#### Epoch 1, Batch 1 (samples 3, 7, 1)

**Step 1: Extract Batch Data**
```
X^(B) = [[3, 4],   (sample 3)
         [7, 8],   (sample 7)
         [1, 2]]   (sample 1)

Y^(B) = [11, 23, 5]
```

**Step 2: Compute Predictions**
```
Ŷ^(B) = X^(B) · θ + b
      = [[3, 4],     [[1],      [0]
         [7, 8],  ·   [1]]  +   [0]
         [1, 2]]              [0]
      
      = [3×1 + 4×1,
         7×1 + 8×1,
         1×1 + 2×1]
      
      = [7, 15, 3]
```

**Step 3: Compute Errors**
```
errors = Y^(B) - Ŷ^(B) = [11, 23, 5] - [7, 15, 3] = [4, 8, 2]
```

**Step 4: Compute Gradients**
```
∂J/∂θ₁ = -2 × mean([4×3, 8×7, 2×1]) = -2 × mean([12, 56, 2]) = -2 × 23.33 = -46.67
∂J/∂θ₂ = -2 × mean([4×4, 8×8, 2×2]) = -2 × mean([16, 64, 4]) = -2 × 28 = -56
∂J/∂b = -2 × mean([4, 8, 2]) = -2 × 4.67 = -9.33
```

**Step 5: Update Parameters**
```
θ₁ = 1.0 - 0.1 × (-46.67) = 5.67
θ₂ = 1.0 - 0.1 × (-56) = 6.60
b = 0.0 - 0.1 × (-9.33) = 0.93
```

#### Epoch 1, Batch 2 (samples 5, 8, 2)

**...process next batch with updated parameters...**

**Key Observations:**
- We process **k samples at a time** (batch of 3)
- We make **one update per batch**
- Parameters change **3 times** in this epoch (8 samples / batch_size 3 ≈ 3 batches)
- Next epoch: **shuffle again** and create new batches

---

## Implementation Details

### Code Structure

```python
class MiniBatchGDRegressor:
    
    def __init__(self, batch_size=32, learning_rate=0.01, epochs=100):
        self.coef_ = None
        self.intercept_ = None
        self.batch_size = batch_size
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
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Process mini-batches
            for j in range(0, X_train.shape[0], self.batch_size):
                
                # Get mini-batch
                X_batch = X_shuffled[j:j+self.batch_size]
                y_batch = y_shuffled[j:j+self.batch_size]
                
                # 1. Compute predictions for THIS batch
                y_hat = np.dot(X_batch, self.coef_) + self.intercept_
                
                # 2. Compute errors for THIS batch
                errors = y_batch - y_hat
                
                # 3. Compute gradient for intercept
                intercept_der = -2 * np.mean(errors)
                
                # 4. Update intercept
                self.intercept_ = self.intercept_ - (self.lr * intercept_der)
                
                # 5. Compute gradient for coefficients
                coef_der = -2 * np.dot(errors, X_batch) / X_batch.shape[0]
                
                # 6. Update coefficients
                self.coef_ = self.coef_ - (self.lr * coef_der)
    
    def predict(self, X_test):
        return np.dot(X_test, self.coef_) + self.intercept_
```

### 🧮 Operations Breakdown

**1. Data Shuffling:**
```python
indices = np.random.permutation(X_train.shape[0])
X_shuffled = X_train[indices]
y_shuffled = y_train[indices]
```
- Shuffles both X and y together
- Maintains correspondence between features and targets
- **Critical** for breaking correlations in sequential data

**2. Batch Extraction:**
```python
for j in range(0, X_train.shape[0], self.batch_size):
    X_batch = X_shuffled[j:j+self.batch_size]  # shape: (batch_size, n_features)
    y_batch = y_shuffled[j:j+self.batch_size]  # shape: (batch_size,)
```
- `range(start, stop, step)` creates batch boundaries
- Example with 100 samples, batch_size=32:
  - j=0: samples [0:32]
  - j=32: samples [32:64]
  - j=64: samples [64:96]
  - j=96: samples [96:100] (last batch smaller)

**3. Batch Prediction (Vectorized):**
```python
y_hat = np.dot(X_batch, self.coef_) + self.intercept_
```
- `X_batch`: shape (k, n)
- `self.coef_`: shape (n,)
- `y_hat`: shape (k,) - predictions for all k samples in batch
- **Much faster than looping** through individual samples

**4. Batch Gradients:**
```python
intercept_der = -2 * np.mean(errors)  # scalar
coef_der = -2 * np.dot(errors, X_batch) / X_batch.shape[0]  # shape (n,)
```
- Averages gradients over the batch
- More stable than single sample (SGD)
- Less computation than full dataset (Batch GD)

---

## Advantages and Disadvantages

### ✅ Advantages

1. **Balanced Speed and Stability**
    - Faster than Batch GD (fewer computations per update)
    - More stable than SGD (averages over multiple samples)
    - **Sweet spot** for most applications

2. **Hardware Efficiency**
    - **Vectorization** - fully utilizes CPU/GPU
    - **Parallelization** - modern hardware optimized for this
    - Batch operations much faster than sequential

3. **Memory Efficient**
    - Doesn't need to load entire dataset
    - Processes manageable chunks
    - Scales to large datasets

4. **Better Generalization**
    - Noise helps escape local minima
    - Regularization effect from batch sampling
    - Often achieves better final performance

5. **Flexible Batch Sizes**
    - Can tune batch size for hardware
    - 32, 64, 128, 256, 512 are common choices
    - Smaller = more noise, larger = more stability

### ❌ Disadvantages

1. **Hyperparameter Tuning**
    - Need to choose batch size
    - Still need to tune learning rate
    - Batch size affects convergence

2. **Not as Stable as Batch GD**
    - Still has some noise (less than SGD)
    - May not converge to exact minimum
    - Oscillates slightly around minimum

3. **Implementation Complexity**
    - More complex than Batch or Stochastic GD
    - Need to handle batch creation
    - Edge cases (last incomplete batch)

---

## When to Use Mini-Batch GD

### ✅ Best Use Cases (Most Common Choice!)

- **Medium to large datasets** (10,000+ samples)
- **Deep learning** applications (almost always)
- **GPU acceleration** available
- **Production systems** requiring balance of speed and accuracy
- **When you want best practices** - this is the industry standard

### ❌ Avoid When

- **Tiny datasets** (< 1,000 samples) - use Batch GD
- **Theoretical analysis** needed - Batch GD is more predictable
- **Extreme memory constraints** - use SGD
- **Online learning** required - use SGD

### 🎯 Choosing Batch Size

**Common batch sizes:** 32, 64, 128, 256, 512

**Guidelines:**

| Dataset Size | Recommended Batch Size |
|--------------|------------------------|
| < 10,000 | 32 or 64 |
| 10,000 - 100,000 | 64 or 128 |
| 100,000 - 1,000,000 | 128 or 256 |
| > 1,000,000 | 256 or 512 |

**Considerations:**
- **Smaller batches (32):** More noise, better generalization, slower per epoch
- **Larger batches (512):** Less noise, faster per epoch, may generalize worse
- **Powers of 2:** Optimal for GPU memory alignment (32, 64, 128, 256, 512)

---

## Comparison with Other Methods

### Detailed Comparison Table

| Aspect | Batch GD | Mini-Batch GD | Stochastic GD |
|--------|----------|---------------|---------------|
| **Samples per update** | All (m) | k (32-512) | 1 |
| **Updates per epoch** | 1 | m/k | m |
| **Convergence** | Very smooth | Moderately smooth | Very noisy |
| **Speed per update** | Slowest | Fast | Fastest |
| **Total time (small data)** | Fast | Medium | Slow |
| **Total time (large data)** | Slow | **Fast** ⭐ | Medium |
| **Memory usage** | High | Medium | Low |
| **Accuracy** | Best | **Very Good** ⭐ | Good |
| **Vectorization** | Excellent | **Excellent** ⭐ | None |
| **GPU efficiency** | Good | **Excellent** ⭐ | Poor |
| **Noise level** | None | Low-Medium | High |
| **Generalization** | Good | **Best** ⭐ | Good |
| **Industry usage** | Rare | **Very Common** ⭐ | Rare |

### Convergence Visualization

**Batch GD:**
```
Cost
  │    ╲
  │     ╲___
  │         ╲___
  │             ╲___
  └──────────────────► Iterations
  Smooth, slow descent
```

**Mini-Batch GD:**
```
Cost
  │   ╲╱╲
  │    ╲ ╲╱
  │     ╲  ╲
  │      ╲__╲
  └──────────────────► Iterations
  Slight noise, fast descent ⭐
```

**Stochastic GD:**
```
Cost
  │  ╲  ╱╲ ╱
  │   ╲╱  ╲  ╱╲
  │   ╱    ╲╱  ╲ ╱
  │  ╱          ╲╱
  └──────────────────► Iterations
  Very noisy descent
```

### Real-World Example (100,000 samples, 10 features)

| Method | Batch Size | Updates/Epoch | Time/Epoch | Total Time (100 epochs) | Final R² |
|--------|------------|---------------|------------|------------------------|----------|
| Batch GD | 100,000 | 1 | 2.5s | 250s | 0.950 |
| Mini-Batch (32) | 32 | 3,125 | 0.8s | 80s | **0.952** |
| Mini-Batch (64) | 64 | 1,563 | 0.4s | **40s** | **0.952** |
| Mini-Batch (128) | 128 | 781 | 0.3s | **30s** | 0.951 |
| Mini-Batch (256) | 256 | 391 | 0.3s | **30s** | 0.950 |
| SGD | 1 | 100,000 | 8.0s | 800s | 0.945 |

**Winner:** Mini-Batch GD with batch_size=64 or 128 ⭐

### Updates Calculation

For **m = 1000 samples**, **epochs = 10**:

| Method | Batch Size | Updates per Epoch | Total Updates |
|--------|------------|-------------------|---------------|
| Batch GD | 1000 | 1000/1000 = **1** | 1 × 10 = **10** |
| Mini-Batch | 32 | 1000/32 = **31.25 ≈ 32** | 32 × 10 = **320** |
| Mini-Batch | 64 | 1000/64 = **15.6 ≈ 16** | 16 × 10 = **160** |
| Mini-Batch | 128 | 1000/128 = **7.8 ≈ 8** | 8 × 10 = **80** |
| SGD | 1 | 1000/1 = **1000** | 1000 × 10 = **10,000** |

---

## Summary

Mini-Batch Gradient Descent is the **gold standard** for training machine learning models, especially in deep learning. It provides:

✅ **Best balance** between speed and stability  
✅ **Efficient use** of modern hardware (GPUs)  
✅ **Better generalization** than pure Batch GD  
✅ **Industry standard** for most applications

### Quick Decision Guide

```
Dataset < 1,000 samples?
├─ YES → Use Batch GD
└─ NO → Use Mini-Batch GD ⭐

Need online learning?
├─ YES → Use SGD
└─ NO → Use Mini-Batch GD ⭐

Have GPU available?
├─ YES → Use Mini-Batch GD ⭐
└─ NO → Still use Mini-Batch GD ⭐

When in doubt → Use Mini-Batch GD! ⭐
```

**Recommended starting point:** `batch_size=64`, `learning_rate=0.01`, `epochs=100`
