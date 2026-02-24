# Curse of Dimensionality

> **Core Idea:** As the number of features (dimensions) in a dataset increases, the data becomes exponentially sparse — making learning harder, slower, and less reliable.

---

## What Are Dimensions?

In machine learning, **features = dimensions**.

| Data Type        | Features (Dimensions)                          |
|------------------|-----------------------------------------------|
| Tabular data     | Columns (age, salary, height…)                |
| Image (28×28)    | 784 pixels → 784 features                    |
| Text document    | Vocabulary size (~3000+ unique words)         |

```
Image (28×28)
     ↓ flatten
[f₁, f₂, f₃, … f₇₈₄]   ← 784 dimensions
```

---

## Why Is High Dimensionality a Problem?

### 1. Sparsity — The Core Issue

As dimensions grow, the **volume of space increases exponentially**, but the number of data points stays the same. The data becomes extremely sparse — points are spread so far apart that patterns become meaningless.

**Intuition with a fixed grid (5 units per axis):**

| Dimensions | Cells Needed | Data Points Required |
|:----------:|:------------:|:--------------------:|
| 1D         | 5            | 5                    |
| 2D         | 5² = 25      | 25                   |
| 3D         | 5³ = 125     | 125                  |
| n-D        | 5ⁿ           | 5ⁿ                   |

```
1D:  • • • • •              (5 points fill space)

2D:  □□□□□
     □□□□□
     □•□•□                  (25 points needed)
     □□□□□

3D:  [cube with sparse points scattered]  (125 points needed)
```

> Adding just one dimension multiplies the data requirement by 5×. With 784 dimensions (MNIST), you'd need an astronomically large dataset to densely populate the space.

---

### 2. Performance Decreases

Models trained on high-dimensional data tend to:
- **Overfit** — they memorize noise instead of learning patterns
- **Lose distance meaning** — in high dimensions, all points become roughly equidistant, breaking distance-based algorithms like KNN

$$\lim_{d \to \infty} \frac{dist_{max} - dist_{min}}{dist_{min}} \to 0$$

As dimensions ($d$) grow, the contrast between nearest and farthest neighbors vanishes.

---

### 3. Computation Increases

More features = more parameters = more training time and memory:

$$E_1 \to E_2 \to E_3 \quad \text{(error grows with irrelevant dimensions)}$$

- Training time scales **polynomially or exponentially** with dimension count
- Irrelevant features add noise, increasing generalization error

---

## Real-World Examples of High Dimensionality

| Domain       | Example                                     | Dimensions     |
|--------------|---------------------------------------------|----------------|
| **Images**   | MNIST digit (28×28 grayscale)               | 784            |
| **Images**   | Color photo (256×256 RGB)                   | 196,608        |
| **Text**     | Bag-of-words representation                 | ~3,000–50,000  |
| **Genomics** | Gene expression data                        | ~20,000+       |

---

## Solution: Dimensionality Reduction

Reduce from $f_m$ features down to $f_n$ features where $f_n < f_m$, while retaining maximum information.

```
[f₁, f₂, f₃, … fₘ]   →   [f₁, f₁₀, f₁₅₀, … fₙ]
  (m features)                  (n features, n < m)
```

### Two Approaches

```
Dimensionality Reduction
        │
        ├── Feature Selection          ← Keep original features, discard irrelevant ones
        │       ├── Forward Selection
        │       └── Backward Elimination
        │
        └── Feature Extraction         ← Transform features into a new compressed space
                ├── PCA  (Principal Component Analysis)
                ├── LDA  (Linear Discriminant Analysis)
                └── t-SNE
```

---

### Feature Selection

Select a **subset of original features** without transforming them.

| Method                  | How It Works                                                  |
|-------------------------|---------------------------------------------------------------|
| **Forward Selection**   | Start with 0 features, add the most useful one at each step  |
| **Backward Elimination**| Start with all features, remove the least useful one at each step |

> See: [`01-data-preparation/feature-engineering/`](../../01-data-preparation/feature-engineering/) for detailed notebooks on filter, wrapper, and embedded methods.

---

### Feature Extraction

**Create new features** that are mathematical combinations of originals, capturing the most variance in fewer dimensions.

| Method   | Type        | Key Idea                                              |
|----------|-------------|-------------------------------------------------------|
| **PCA**  | Unsupervised| Projects data onto directions of maximum variance     |
| **LDA**  | Supervised  | Maximizes class separability while reducing dimensions|
| **t-SNE**| Unsupervised| Non-linear; preserves local structure for visualization|

---

## Summary

```
High Dimensions
      │
      ├── Sparsity (data points spread far apart)
      ├── Performance drops (overfitting, distance collapse)
      └── Computation explodes
            │
            ▼
    Dimensionality Reduction
            │
            ├── Feature Selection  → Keep best original features
            └── Feature Extraction → Create new compact features (PCA, LDA, t-SNE)
```

| Problem               | Symptom                              | Fix                        |
|-----------------------|--------------------------------------|----------------------------|
| Too many features     | Slow training, overfitting           | Feature selection          |
| Correlated features   | Redundant information                | PCA / LDA                  |
| Visualization needed  | Can't plot 784-D data                | t-SNE / PCA to 2D          |
| Sparse data           | Model can't find patterns            | Reduce to meaningful dims  |

---

## Key Takeaways

1. **Features = Dimensions** — every column in your dataset is a dimension.
2. **More is not always better** — beyond a point, adding features hurts more than it helps.
3. **Sparsity is the enemy** — high dimensions cause data to thin out exponentially.
4. **Dimensionality reduction is not data loss** — done right, it removes noise and retains signal.
5. **Choose your method wisely** — use Feature Selection when interpretability matters; use Feature Extraction (PCA/LDA) when you need compression.

---

## References & Next Steps

- `02-pca.ipynb` — Principal Component Analysis from scratch
- `03-lda.ipynb` — Linear Discriminant Analysis
- `04-tsne-visualization.ipynb` — t-SNE for high-dimensional visualization
- [`01-data-preparation/feature-engineering/`](../../01-data-preparation/feature-engineering/feature-selection-cheatsheet.md) — Feature Selection methods
