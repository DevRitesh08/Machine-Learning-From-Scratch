# Kernel PCA — When Linear PCA Fails

> **Core Idea:** Standard PCA assumes data varies along **straight lines**. When the structure is non-linear (like concentric circles), PCA fails completely. Kernel PCA uses the **kernel trick** to find non-linear principal components without explicitly working in high-dimensional space.

---

## Table of Contents

1. [The Problem — Why PCA Fails](#1-the-problem--why-pca-fails)
2. [The Intuition Behind Kernel PCA](#2-the-intuition-behind-kernel-pca)
3. [The Kernel Trick](#3-the-kernel-trick)
4. [Kernel PCA Algorithm](#4-kernel-pca-algorithm)
5. [PCA vs Kernel PCA — When to Use Which](#5-pca-vs-kernel-pca--when-to-use-which)

---

## 1. The Problem — Why PCA Fails

PCA assumes principal components are **linear combinations** of original features. It finds straight axes of maximum variance.

This works when data has linear structure, but **not for non-linear patterns** like concentric circles — no straight line can separate them, and PCA's projection destroys the class structure.

![PCA Fails on Concentric Circles](assets/kpca_01_pca_fails.png)

> **Root cause:** PCA can only capture **linear** relationships. Concentric circles have a **radial** (non-linear) relationship.

---

## 2. The Intuition Behind Kernel PCA

It is an extension of PCA that can capture non-linear relationships by implicitly mapping data to a higher-dimensional space where linear PCA can be applied.

### The Big Idea: Lift to Higher Dimensions

If data isn't linearly separable in its current space, **map it to a higher-dimensional space** where it becomes separable.

![Lifting to 3D](assets/kpca_02_lift_to_3d.png)

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

### The Problem

Mapping data to higher dimensions works, but it's **expensive**:

- Need millions of dimensions to capture complex patterns
- Computing and storing $\Phi(x)$ for every point is impractical

### The Solution

> **Key insight:** PCA only needs **dot products** between points — not the actual coordinates.

So instead of:

$$
\underbrace{x \xrightarrow{\Phi} \Phi(x)}_{\text{expensive map}} \quad \text{then} \quad \underbrace{\Phi(x)^T \Phi(y)}_{\text{expensive dot product in high-dim}}
$$

We use a **kernel function** that gives the same result directly:

$$
k(x, y) = \Phi(x)^T \Phi(y) \quad \leftarrow \text{computed cheaply, without ever computing } \Phi(x)
$$

### RBF Kernel — The Most Common Choice

$$
k(x, y) = e^{-\gamma \lVert x - y \rVert^2}
$$

- Takes two points $x, y$ in **original** space
- Returns a number = how similar they are
- Implicitly corresponds to a dot product in **infinite-dimensional** space
- Cost: a single formula evaluation — no mapping needed
- **High γ** → only very close points matter (complex boundaries)
- **Low γ** → distant points also matter (smooth boundaries)

| Approach | Steps | Cost |
|---|---|---|
| **Explicit mapping** | Map → Map → Dot product | $O(\infty)$ |
| **Kernel trick** | One function call | $O(d)$ |

---

## 4. Kernel PCA Algorithm

### Standard PCA (for comparison)

1. Center data: $X \leftarrow X - \bar{X}$
2. Compute covariance: $\Sigma = \frac{1}{n}X^TX$
3. Eigendecompose: $\Sigma = V\Lambda V^T$
4. Project: $X_{\text{reduced}} = X V_k$

### Kernel PCA

1. Compute the **kernel matrix** $K$ where $K_{ij} = k(x_i, x_j)$ — this is an $N \times N$ matrix
2. **Center** the kernel matrix: $K' = K - \mathbf{1}_N K - K \mathbf{1}_N + \mathbf{1}_N K \mathbf{1}_N$
   - Where $\mathbf{1}_N$ is an $N \times N$ matrix with all entries $\frac{1}{N}$
   - We can't subtract the mean in kernel space since we never have explicit $\Phi(x)$ coordinates
3. **Eigendecompose** $K'$: solve $N\lambda \alpha = K' \alpha$
4. **Normalize** eigenvectors $\alpha^k$
5. **Project** new point: $\text{projection}_k = \sum_{i=1}^{N} \alpha_i^k \cdot k(x_i, x_{\text{new}})$

> **Key difference:** Regular PCA decomposes a $d \times d$ covariance matrix. Kernel PCA decomposes an $N \times N$ kernel matrix. This means Kernel PCA complexity scales with **number of samples**, not dimensions.

| | Standard PCA | Kernel PCA |
|---|---|---|
| **Key matrix** | Covariance $\Sigma$ $(d \times d)$ | Kernel Matrix $K$ $(n \times n)$ |
| **Eigendecompose** | $\Sigma$ | Centered $K$ |
| **Result** | Eigenvectors in feature space | Eigenvectors in kernel space |
| **Projection** | Via $V$ | Via kernel evaluations |

---

## 5. PCA vs Kernel PCA — When to Use Which

| Criterion | Standard PCA | Kernel PCA |
|---|---|---|
| **Data structure** | Linear | **Non-linear** |
| **Computes** | Covariance matrix eigen decomposition (or SVD) | Kernel matrix eigen decomposition |
| **Matrix size** | $d \times d$ (features) | $N \times N$ (samples) |
| **Scalability** | Scales with features | Scales with **samples** (expensive for large $N$) |
| **Inverse transform** | Exact | Approximate (pre-image problem) |
| **Hyperparameters** | None | Kernel choice + kernel parameters ($\gamma$, degree, etc.) |
| **Interpretability** | High (components are linear combos) | Low (components in implicit high-dim space) |
| **Memory** | $O(d^2)$ or $O(nd)$ | $O(N^2)$ — stores full kernel matrix |
| **Time complexity** | $O(nd^2)$ or $O(nd \cdot k)$ randomized | $O(N^2 d + N^3)$ |

### Decision Guide

> 1. **Data is linearly structured?** → Use standard PCA
> 2. **Non-linear + manageable N (< ~10,000)?** → Use Kernel PCA
> 3. **Non-linear + large N?** → Consider Nystroem approximation, t-SNE/UMAP, or Autoencoders

---

> **Prerequisites:** [Linear Algebra for PCA](references/linear-algebra-for-pca.md) (special matrices, eigen decomposition) → [PCA Theory](02-pca.ipynb) → this note.
>
> **Related:** [SVD](05-svd.md) — the matrix decomposition that powers standard PCA under the hood.
