# Kernel PCA — When Linear PCA Fails

> **Core Idea:** Standard PCA assumes data varies along **straight lines**. When the structure is non-linear (like concentric circles), PCA fails completely. Kernel PCA uses the **kernel trick** to find non-linear principal components without explicitly working in high-dimensional space.

---

## Table of Contents

1. [The Problem — Why PCA Fails](#1-the-problem--why-pca-fails)
2. [The Intuition Behind Kernel PCA](#2-the-intuition-behind-kernel-pca)
3. [The Kernel Trick](#3-the-kernel-trick)
4. [Common Kernels](#4-common-kernels)
5. [Kernel PCA Algorithm](#5-kernel-pca-algorithm)
6. [PCA vs Kernel PCA — When to Use Which](#6-pca-vs-kernel-pca--when-to-use-which)

---

## 1. The Problem — Why PCA Fails

PCA assumes principal components are **linear combinations** of original features. It finds straight axes of maximum variance.

**This works when data has linear structure:**

```
  ↗ PC1 (captures most variance along this straight line)
 ·  ·  · 
·  ·  ·  ·
 ·  ·  ·
                 ✓ PCA works great here
```

**But what about concentric circles?**

```
         · · · · ·
       ·           ·
     ·    · · · ·    ·
     ·   ·       ·   ·
     ·   · (inner) ·  ·
     ·   ·       ·   ·
     ·    · · · ·    ·
       ·    (outer)·
         · · · · ·

  Two classes arranged in concentric rings
  → NO straight line can separate them
  → PCA projects everything onto a line → overlap!
```

When you apply standard PCA to separate concentric circles:
1. PC1 finds the direction of maximum spread → a **diameter** line
2. Projecting onto it → **inner and outer rings overlap** completely
3. The class structure is **destroyed**

> **Root cause:** PCA can only capture **linear** relationships. Concentric circles have a **radial** (non-linear) relationship.

---

## 2. The Intuition Behind Kernel PCA

### The Big Idea: Lift to Higher Dimensions

If data isn't linearly separable in its current space, **map it to a higher-dimensional space** where it becomes separable.

```
  2D (not separable)                    3D (separable!)
                                          ↑ z = x² + y²
     · · · · ·                           │
   ·  (outer)  ·                  outer →│ · · · · · (high z)
  ·   · · · ·   ·                        │
  ·  · inner ·   ·                       │
  ·   · · · ·   ·                inner →│ · · · (low z)
   ·           ·                         │
     · · · · ·                    ───────┼──────────▶ x
                                         │
                              A flat plane can now
                              separate the two groups!
```

**Step by step:**

1. **Original 2D data:** Points at $(x_1, x_2)$
2. **Map to 3D:** Add feature $z = x_1^2 + x_2^2$ (distance from origin)
   - Inner circle → small $z$
   - Outer circle → large $z$
3. **In 3D:** A simple horizontal plane separates them
4. **Apply PCA in this new space** → First principal component separates classes

> This is exactly the same idea as in **SVM with kernels** — the kernel trick was borrowed from SVM into PCA.

---

## 3. The Kernel Trick

### The Problem with Explicit Mapping

Mapping to higher dimensions sounds great, but:
- What if you need to map to 1,000,000 dimensions?
- Computing $\Phi(x)$ for every data point is **extremely expensive**
- Storing the high-dimensional data is impractical

### The Solution: Never Compute $\Phi(x)$ Explicitly!

The kernel trick says: **you don't need the actual coordinates in high-dimensional space — you only need the dot products between points.**

$$
k(x, y) = \Phi(x)^T \Phi(y)
$$

A kernel function computes what the dot product **would be** in high-dimensional space, without ever going there.

```
  Without kernel trick:
  x → Φ(x) → compute Φ(x)ᵀΦ(y) in high-dim space     EXPENSIVE!

  With kernel trick:
  x, y → k(x, y) directly                               CHEAP!
         (one function call, stays in original dim)
```

### Mathematical Kernel Function

Instead of:
1. Map $x \to \Phi(x)$ (expensive)
2. Map $y \to \Phi(y)$ (expensive)
3. Compute $\Phi(x)^T \Phi(y)$ (expensive)

Just compute:

$$
k(x, y) = \text{some function of } x \text{ and } y
$$

This gives the **same answer** as steps 1-3, in a fraction of the time.

---

## 4. Common Kernels

| Kernel | Formula | Maps to | Best for |
|---|---|---|---|
| **Linear** | $k(x,y) = x^T y$ | Same space (no mapping) | Already linear data (= regular PCA) |
| **Polynomial** | $k(x,y) = (x^T y + c)^d$ | Finite higher-dim space | Polynomial-shaped boundaries |
| **RBF / Gaussian** | $k(x,y) = e^{-\frac{\lVert x-y \rVert^2}{2\sigma^2}}$ | **Infinite**-dim space | Most common, works on many shapes |
| **Sigmoid** | $k(x,y) = \tanh(\alpha \cdot x^T y + c)$ | Non-linear space | Neural network-like boundaries |

### RBF Kernel — The Most Important One

$$
k(x, y) = e^{-\frac{\lVert x - y \rVert^2}{2\sigma^2}}
$$

- Returns **1** when $x = y$ (identical points)
- Returns **~0** when points are far apart
- $\sigma$ controls the "width" — how far the influence reaches

> **Why RBF is powerful:** It implicitly maps to an **infinite-dimensional** space. It can capture virtually any non-linear pattern. The parameter $\sigma$ is the key hyperparameter to tune.

---

## 5. Kernel PCA Algorithm

### Standard PCA (for comparison)

1. Center data: $X \leftarrow X - \bar{X}$
2. Compute covariance: $\Sigma = \frac{1}{n}X^TX$
3. Eigendecompose: $\Sigma = V\Lambda V^T$
4. Project: $X_{\text{reduced}} = X V_k$

### Kernel PCA

1. Compute the **kernel matrix** $K$ where $K_{ij} = k(x_i, x_j)$ — this is an $N \times N$ matrix
2. **Center** the kernel matrix: $K' = K - \mathbf{1}_N K - K \mathbf{1}_N + \mathbf{1}_N K \mathbf{1}_N$
   - Where $\mathbf{1}_N$ is an $N \times N$ matrix with all entries $\frac{1}{N}$
3. **Eigendecompose** $K'$: solve $N\lambda \alpha = K' \alpha$
4. **Normalize** eigenvectors $\alpha^k$
5. **Project** new point: $\text{projection}_k = \sum_{i=1}^{N} \alpha_i^k \cdot k(x_i, x_{\text{new}})$

```
  Standard PCA:                     Kernel PCA:
  
  Data X (n×d)                      Data X (n×d)
     │                                  │
     ▼                                  ▼
  Covariance Σ (d×d)               Kernel Matrix K (n×n)
     │                                  │
     ▼                                  ▼
  Eigendecomp of Σ                 Eigendecomp of K
     │                                  │
     ▼                                  ▼
  Eigenvectors (in feature space)  Eigenvectors (in kernel space)
     │                                  │
     ▼                                  ▼
  Project via V                    Project via kernel evaluations
```

> **Key difference:** Regular PCA decomposes a $d \times d$ covariance matrix. Kernel PCA decomposes an $N \times N$ kernel matrix. This means Kernel PCA complexity scales with **number of samples**, not dimensions.

---

## 6. PCA vs Kernel PCA — When to Use Which

| Criterion | Standard PCA | Kernel PCA |
|---|---|---|
| **Data structure** | Linear | **Non-linear** |
| **Computes** | Covariance matrix eigen decomposition | Kernel matrix eigen decomposition |
| **Matrix size** | $d \times d$ (features) | $N \times N$ (samples) |
| **Scalability** | Scales with features | Scales with **samples** (expensive for large $N$) |
| **Inverse transform** | Exact | Approximate (pre-image problem) |
| **Hyperparameters** | None | Kernel choice + kernel parameters ($\sigma$, degree, etc.) |
| **Interpretability** | High (components are linear combos) | Low (components in implicit high-dim space) |

### Decision Flowchart

```
  Is your data linearly structured?
         │
    ┌────┴────┐
    │ YES     │ NO / UNCLEAR
    ▼         ▼
  Use PCA    Is N (sample count) manageable? (< ~10,000)
              │
         ┌────┴────┐
         │ YES     │ NO
         ▼         ▼
     Use Kernel   Consider:
     PCA          • Sparse approximations
                  • Random Fourier Features
                  • t-SNE / UMAP instead
```

### Related Kernel Methods

The kernel trick isn't unique to PCA — it's a general technique:

| Method | Linear Version | Kernel Version |
|---|---|---|
| PCA | Standard PCA | Kernel PCA |
| SVM | Linear SVM | Kernel SVM (RBF, polynomial, etc.) |
| Ridge Regression | Linear Ridge | Kernel Ridge Regression |
| Fisher Discriminant | LDA | Kernel FDA |

---

> **Prerequisites:** [Linear Algebra for PCA](references/linear-algebra-for-pca.md) (special matrices, eigen decomposition) → [PCA Theory](02-pca.ipynb) → this note.
>
> **Next steps:** Implement Kernel PCA from scratch using scikit-learn's `KernelPCA` class, or dive into [SVD-based PCA](03-pca-implementation.ipynb) for the implementation perspective.
