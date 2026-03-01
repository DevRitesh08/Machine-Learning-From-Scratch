# Singular Value Decomposition (SVD) — The Universal Matrix Factorization

> **Core Idea:** SVD decomposes **any** matrix (any shape, any rank) into three simple matrices: a rotation, a scaling, and another rotation. It's the most general and powerful matrix decomposition, and the mathematical backbone of PCA, recommender systems, image compression, and NLP.

---

## Table of Contents

1. [Why SVD? — Motivation](#1-why-svd--motivation)
2. [The SVD Formula](#2-the-svd-formula)
3. [Geometric Intuition — Rotate, Scale, Rotate](#3-geometric-intuition--rotate-scale-rotate)
4. [Full SVD vs Truncated SVD](#4-full-svd-vs-truncated-svd)
5. [How to Compute SVD](#5-how-to-compute-svd)
6. [SVD and PCA — The Deep Connection](#6-svd-and-pca--the-deep-connection)
7. [Applications of SVD](#7-applications-of-svd)
8. [SVD vs Eigen Decomposition — When to Use Which](#8-svd-vs-eigen-decomposition--when-to-use-which)
9. [Key Properties & Theorems](#9-key-properties--theorems)

---

## 1. Why SVD? — Motivation

### The Limitation of Eigen Decomposition

Eigen decomposition requires a **square** matrix and only works when the matrix is **diagonalizable**:

$$
A = V\Lambda V^{-1} \quad \text{(only for square, diagonalizable matrices)}
$$

But in ML, our data matrix $X$ is almost never square — it's $n \times d$ (samples × features) — **can't do eigen decomposition on $X$ directly!**

### SVD to the Rescue

SVD works on **any** matrix — square or rectangular, any rank, any size:

$$
\boxed{A_{m \times n} = U_{m \times m} \cdot \Sigma_{m \times n} \cdot V^T_{n \times n}}
$$

> **SVD always exists.** There is no matrix for which SVD cannot be computed. This is the **Existence Theorem of SVD**.

---

## 2. The SVD Formula

Any real matrix $A$ of size $m \times n$ can be factorized as:

$$
\boxed{A = U \Sigma V^T}
$$

### Components Explained

| Component | Name | Size | Properties |
|---|---|---|---|
| $U$ | **Left singular vectors** | $m \times m$ | Orthogonal ($U^TU = I$). Columns = orthonormal basis for column space |
| $\Sigma$ | **Singular value matrix** | $m \times n$ | Diagonal-like with $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$ on the diagonal |
| $V^T$ | **Right singular vectors** (transposed) | $n \times n$ | Orthogonal ($V^TV = I$). Rows = orthonormal basis for row space |

### Singular Values

The diagonal entries $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r$ where $r = \text{rank}(A)$ are the **singular values**.

- Always **real and non-negative** ($\sigma_i \geq 0$)
- Ordered from largest to smallest
- Number of non-zero singular values = **rank** of the matrix
- They represent the "importance" or "strength" of each component

---

## 3. Geometric Intuition — Rotate, Scale, Rotate

SVD says: **every linear transformation is a rotation, followed by a scaling, followed by another rotation.**

$$
A\vec{x} = U \cdot \Sigma \cdot V^T \cdot \vec{x}
$$

Reading **right to left**:

| Step | Matrix | Operation | Geometric Effect |
|---|---|---|---|
| 1 | $V^T$ | Rotation in **input space** | Aligns data with canonical axes |
| 2 | $\Sigma$ | Scaling | Stretches/compresses along each axis by $\sigma_i$ |
| 3 | $U$ | Rotation in **output space** | Rotates the scaled result to final orientation |

> **Key insight:** Any matrix maps the unit circle to an ellipse. The right singular vectors $V$ define the **input directions**, the left singular vectors $U$ define the **output directions**, and the singular values $\sigma_i$ define the **stretching factors** along each direction.

---

## 4. Full SVD vs Truncated SVD

### Full SVD

$$
A_{m \times n} = U_{m \times m} \cdot \Sigma_{m \times n} \cdot V^T_{n \times n}
$$

### Truncated SVD (the key for dimensionality reduction!)

Keep only the top $k$ singular values (where $k < r = \text{rank}(A)$):

$$
\boxed{A \approx A_k = U_k \cdot \Sigma_k \cdot V_k^T}
$$

As a sum of rank-1 matrices:

**Full:** $A = \sigma_1 u_1 v_1^T + \sigma_2 u_2 v_2^T + \cdots + \sigma_r u_r v_r^T$

**Truncated:** $A \approx \sigma_1 u_1 v_1^T + \sigma_2 u_2 v_2^T + \cdots + \sigma_k u_k v_k^T \quad (k < r)$

> **Eckart–Young–Mirsky Theorem:** The truncated SVD $A_k$ is the **best rank-k approximation** of $A$, minimizing $\lVert A - A_k \rVert$ in both Frobenius and spectral norms. No other rank-$k$ matrix is closer to $A$.

| Type | Size of $U$ | Size of $\Sigma$ | Size of $V$ | Use Case |
|---|---|---|---|---|
| **Full** | $m \times m$ | $m \times n$ | $n \times n$ | Theoretical / exact |
| **Compact** | $m \times r$ | $r \times r$ | $n \times r$ | Exact, less memory |
| **Truncated** | $m \times k$ | $k \times k$ | $n \times k$ | **Dimensionality reduction, compression** |

### How Much Information Is Retained?

$$
\text{Explained Variance Ratio} = \frac{\sum_{i=1}^{k} \sigma_i^2}{\sum_{i=1}^{r} \sigma_i^2}
$$

---

## 5. How to Compute SVD

### Relationship to Eigendecomposition

$$
A^T A = V (\Sigma^T \Sigma) V^T \qquad A A^T = U (\Sigma \Sigma^T) U^T
$$

- $A^TA$ is $n\times n$ symmetric → eigendecompose → gives $V$ and $\sigma_i^2$
- $AA^T$ is $m\times m$ symmetric → eigendecompose → gives $U$ and $\sigma_i^2$
- $\sigma_i = \sqrt{\text{eigenvalue of } A^TA} = \sqrt{\text{eigenvalue of } AA^T}$

### Algorithm

Given $A$ ($m \times n$):

1. **Eigendecompose** $A^T A$ → eigenvalues $\lambda_i$, eigenvectors $v_i$ → form $V$
2. **Singular values**: $\sigma_i = \sqrt{\lambda_i}$ → form $\Sigma$
3. **Left singular vectors**: $u_i = \frac{A v_i}{\sigma_i}$ for each $\sigma_i > 0$ → form $U$

---

## 6. SVD and PCA — The Deep Connection

### The Two Paths to PCA

**Path 1: Eigen Decomposition** — Center X → Covariance $C = X^TX/(n-1)$ → Eigendecompose $C = V\Lambda V^T$ → $V$ = PC directions, $\Lambda$ = variances

**Path 2: SVD** — Center X → $X = U\Sigma V^T$ (SVD directly on X) → $V$ = right singular vectors (same V!), $\Lambda = \Sigma^2/(n-1)$

### Why They Give the Same Result

If $X = U\Sigma V^T$, then:

$$
C = \frac{X^T X}{n-1} = \frac{V\Sigma^T U^T U\Sigma V^T}{n-1} = V \cdot \frac{\Sigma^2}{n-1} \cdot V^T
$$

Comparing with $C = V\Lambda V^T$:

$$
\boxed{\lambda_i = \frac{\sigma_i^2}{n-1}}
$$

> **The principal component directions $V$ from PCA are exactly the right singular vectors from SVD!** The eigenvalues (variances) are the squared singular values divided by $(n-1)$.

### Why sklearn Uses SVD Instead of Eigen Decomposition

| | Eigen Decomposition Path | SVD Path (what sklearn does) |
|---|---|---|
| **Steps** | Center X → Compute $C = X^TX/(n-1)$ → Eigendecompose $C$ | Center X → SVD of X directly |
| **Stability** | $X^TX$ loses precision (squaring amplifies rounding errors) | More numerically stable, no need to form $X^TX$ |
| **Efficiency** | Unstable for ill-conditioned data | Efficient randomized algorithms exist |

> **sklearn's `PCA` class internally uses SVD, NOT eigen decomposition.** This is a deliberate design choice for numerical stability.

---

## 7. Applications of SVD

| Application | How SVD is Used |
|---|---|
| **Image Compression** | Image = pixel matrix. Truncated SVD keeps top-$k$ components → rank-50 of 1000×1000 image ≈ 10% storage, visually similar |
| **Recommender Systems** | User-Movie rating matrix → $U$ = user preferences, $V^T$ = movie characteristics, missing ratings $\approx U_k \Sigma_k V_k^T$ |
| **Latent Semantic Analysis (NLP)** | Term-Document matrix → SVD discovers latent topics, maps documents & terms to same "topic space" |
| **Noise Reduction** | Large singular values = real signal, small = noise. Truncation removes noise, preserves structure |
| **Pseudoinverse (Moore-Penrose)** | $A^+ = V \Sigma^+ U^T$ (reciprocal of non-zero $\sigma_i$). Used in least squares, sklearn's `LinearRegression` |

---

## 8. SVD vs Eigen Decomposition — When to Use Which

| Criterion | Eigen Decomposition | SVD |
|---|---|---|
| **Input shape** | Square only ($n \times n$) | **Any** ($m \times n$) |
| **Existence** | Not always (needs diagonalizability) | **Always exists** |
| **Values** | Eigenvalues (can be negative, complex) | Singular values (**always $\geq 0$, real**) |
| **Numerical stability** | Less stable (forming $A^TA$ squares condition number) | **More stable** (avoids squaring) |
| **PCA: operates on** | Covariance matrix $C$ ($d \times d$) | Data matrix $X$ ($n \times d$) directly |
| **Vectors** | Eigenvectors (may not be orthogonal) | Left & right singular vectors (**always orthogonal**) |

### Connection Between Singular Values and Eigenvalues

| Matrix | Relationship |
|---|---|
| $A^TA$ or $AA^T$ | eigenvalues $\lambda_i$ = singular values² $\sigma_i^2$ |
| Covariance $C = X^TX/(n-1)$ | $\lambda_i = \sigma_i^2 / (n-1)$ |

### Decision Guide

- **Rectangular matrix?** → SVD (only option)
- **Square, symmetric?** → Either works
- **Square, non-symmetric?** → SVD (more stable)
- **General rule:** When in doubt, use SVD — it always works and is more numerically stable

---

## 9. Key Properties & Theorems

### Fundamental Properties

| Property | Formula / Statement |
|---|---|
| **Rank** | $\text{rank}(A) = $ number of non-zero singular values |
| **Frobenius norm** | $\lVert A \rVert_F = \sqrt{\sum \sigma_i^2}$ |
| **Spectral norm (2-norm)** | $\lVert A \rVert_2 = \sigma_1$ (largest singular value) |
| **Condition number** | $\kappa(A) = \sigma_1 / \sigma_r$ (ratio of largest to smallest non-zero) |
| **Determinant** (square $A$) | $\lvert\det(A)\rvert = \prod \sigma_i$ |

### Eckart–Young–Mirsky Theorem

$$
A_k = \arg\min_{\text{rank}(B)=k} \lVert A - B \rVert
$$

- Frobenius norm: $\lVert A - A_k \rVert_F^2 = \sigma_{k+1}^2 + \cdots + \sigma_r^2$
- Spectral norm: $\lVert A - A_k \rVert_2 = \sigma_{k+1}$

### The Four Fundamental Subspaces (via SVD)

- **Column space** of $A$ = span of first $r$ columns of $U$
- **Left null space** of $A$ = span of remaining $(m-r)$ columns of $U$
- **Row space** of $A$ = span of first $r$ columns of $V$
- **Null space** of $A$ = span of remaining $(n-r)$ columns of $V$

### Quick Reference

| Item | Details |
|---|---|
| **Formula** | $A = U\Sigma V^T$ |
| **U** $(m \times m)$ | Left singular vectors, orthogonal. Columns = eigenvectors of $AA^T$ |
| **$\Sigma$** $(m \times n)$ | Singular values on diagonal, $\sigma_i = \sqrt{\text{eigenvalue of } A^TA}$ |
| **$V^T$** $(n \times n)$ | Right singular vectors, orthogonal. Rows = eigenvectors of $A^TA$ |
| **PCA connection** | PC directions = columns of $V$. Variance per PC = $\sigma_i^2 / (n-1)$ |
| **Best rank-k** | $A_k = U_k \Sigma_k V_k^T$ (Eckart-Young theorem) |

---

> **Prerequisites:** [Linear Algebra for PCA](references/linear-algebra-for-pca.md) (eigen decomposition, special matrices) → [PCA Theory](02-pca.ipynb) → this note.
>
> **Related:** [Kernel PCA](04-kernel-pca.md) — when linear PCA (and SVD) can't capture non-linear structure.
