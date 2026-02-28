# Special Matrices for Eigen Decomposition & SVD

> **Why this matters for PCA:** The covariance matrix in PCA is always **real and symmetric**, its eigen decomposition relies on properties of **diagonal**, **orthogonal**, and **symmetric** matrices. Understanding these three matrix types is essential before diving into eigen decomposition or singular value decomposition.

---

## 1. Diagonal Matrix

A diagonal matrix is a **square matrix** where all entries outside the main diagonal are zero. The main diagonal runs from the top-left to the bottom-right.

$$
D = \begin{bmatrix} a & 0 & 0 \\ 0 & b & 0 \\ 0 & 0 & c \end{bmatrix}
$$

### Key Properties

| Property | Description | Example |
|---|---|---|
| **Powers** | $D^n$ is obtained by raising each diagonal element to the $n$-th power | $\begin{bmatrix} 5 & 0 \\ 0 & 6 \end{bmatrix}^{100} = \begin{bmatrix} 5^{100} & 0 \\ 0 & 6^{100} \end{bmatrix}$ |
| **Eigenvalues** | The eigenvalues are simply the values on the diagonal | $D = \begin{bmatrix} a & 0 \\ 0 & b \end{bmatrix} \Rightarrow \lambda_1 = a,\ \lambda_2 = b$ |
| **Eigenvectors** | The eigenvectors are the **standard basis vectors** | $e_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix},\ e_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$ |
| **Vector Multiplication** | Scales each component of the vector by the corresponding diagonal element | $\begin{bmatrix} a & 0 \\ 0 & b \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} ax \\ by \end{bmatrix}$ |
| **Matrix Multiplication** | Product of two diagonal matrices = diagonal matrix with elements multiplied | $\begin{bmatrix} a & 0 \\ 0 & b \end{bmatrix} \begin{bmatrix} c & 0 \\ 0 & d \end{bmatrix} = \begin{bmatrix} ac & 0 \\ 0 & bd \end{bmatrix}$ |

### Why It Matters for PCA

In eigen decomposition $A = PDP^{-1}$, the matrix $D$ is a **diagonal matrix of eigenvalues**. The diagonal structure makes computing $A^{100}$ trivial — just raise each eigenvalue to the 100th power instead of multiplying $A$ by itself 100 times.

$$
A^{100} = P \cdot D^{100} \cdot P^{-1} = P \begin{bmatrix} \lambda_1^{100} & 0 \\ 0 & \lambda_2^{100} \end{bmatrix} P^{-1}
$$

---

## 2. Orthogonal Matrix

An orthogonal matrix is a **square matrix** whose columns and rows are **orthonormal vectors** — meaning they are all of **unit length** and are at **right angles** to each other.

$$
Q^T = Q^{-1} \quad \Longleftrightarrow \quad Q^T Q = I
$$

> **Geometric interpretation:** An orthogonal matrix represents a **pure rotation** (or reflection). No scaling, no shearing — just rotation.

### Column Conditions

For a $2 \times 2$ orthogonal matrix $Q = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$:

| Condition | Meaning |
|---|---|
| $ab + cd = 0$ | Columns are **orthogonal** (dot product = 0, i.e., 90° apart) |
| $\sqrt{a^2 + c^2} = 1$ | First column has **unit length** |
| $\sqrt{b^2 + d^2} = 1$ | Second column has **unit length** |

### Key Properties

| Property | Description |
|---|---|
| **Inverse = Transpose** | $Q^{-1} = Q^T$ — makes computation very efficient |
| **Preserves lengths** | $\|Qx\| = \|x\|$ for any vector $x$ |
| **Preserves angles** | The angle between any two vectors is unchanged after transformation |
| **Determinant** | $\det(Q) = \pm 1$ (+1 for rotation, −1 for reflection) |

### Example: 2D Rotation Matrix

The standard rotation matrix for angle $\theta$ is orthogonal:

$$
R(\theta) = \begin{bmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix}
$$

**Verification for $\theta = 30°$:**

$$
R(30°) = \begin{bmatrix} \frac{\sqrt{3}}{2} & \frac{1}{2} \\[6pt] -\frac{1}{2} & \frac{\sqrt{3}}{2} \end{bmatrix}
$$

- **Orthogonality check:** $\frac{\sqrt{3}}{2} \cdot \frac{1}{2} + \left(-\frac{1}{2}\right) \cdot \frac{\sqrt{3}}{2} = \frac{\sqrt{3}}{4} - \frac{\sqrt{3}}{4} = 0$ ✓
- **Unit length check:** $\sqrt{\left(\frac{\sqrt{3}}{2}\right)^2 + \left(\frac{1}{2}\right)^2} = \sqrt{\frac{3}{4} + \frac{1}{4}} = 1$ ✓

> **Special case:** The **identity matrix** $I = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$ is an orthogonal matrix (rotation by $\theta = 0°$).

### Why It Matters for PCA

In eigen decomposition of a symmetric matrix, the eigenvector matrix $P$ is **orthogonal**. This means principal components are orthogonal directions — they capture independent, uncorrelated axes of variance.

---

## 3. Symmetric Matrix

A symmetric matrix is a **square matrix** that is equal to its own transpose:

$$
A = A^T
$$

If you swap rows with columns, you get the **same matrix**.

$$
\begin{bmatrix} a & c \\ c & b \end{bmatrix}^T = \begin{bmatrix} a & c \\ c & b \end{bmatrix}
$$

### Key Properties

| Property | Description |
|---|---|
| **Real Eigenvalues** | The eigenvalues of a real symmetric matrix are **always real** (never complex) |
| **Orthogonal Eigenvectors** | Eigenvectors corresponding to **different eigenvalues** are always orthogonal to each other |
| **Orthonormal Basis** | If eigenvalues are distinct, you can choose an orthonormal basis of eigenvectors |
| **Diagonalizable** | Every real symmetric matrix can be decomposed as $A = Q \Lambda Q^T$ |

### Spectral Theorem (Eigen Decomposition of Symmetric Matrices)

For any real symmetric matrix $A$:

$$
A = Q \Lambda Q^T
$$

Where:
- $Q$ — orthogonal matrix of eigenvectors (columns)
- $\Lambda$ — diagonal matrix of eigenvalues
- $Q^T = Q^{-1}$ (because $Q$ is orthogonal)

> This connects all three matrix types: a **symmetric** matrix decomposes into an **orthogonal** matrix and a **diagonal** matrix.

### Why It Matters for PCA

The **covariance matrix** is always symmetric:

$$
\Sigma = \begin{bmatrix} \text{Var}(x_1) & \text{Cov}(x_1, x_2) \\ \text{Cov}(x_2, x_1) & \text{Var}(x_2) \end{bmatrix}
$$

Since $\text{Cov}(x_1, x_2) = \text{Cov}(x_2, x_1)$, the matrix is symmetric by definition. This guarantees:

1. **Real eigenvalues** → variance explained is always a real positive number
2. **Orthogonal eigenvectors** → principal components are perpendicular/uncorrelated
3. **Clean decomposition** → $\Sigma = Q \Lambda Q^T$ gives us the exact PCA solution

---

## How All Three Connect in PCA

```
Covariance Matrix (Symmetric)
        │
        ▼
  Eigen Decomposition
  Σ = Q · Λ · Qᵀ
       │     │
       ▼     ▼
   Orthogonal  Diagonal
   Matrix Q    Matrix Λ
       │          │
       ▼          ▼
  Eigenvectors  Eigenvalues
  (PC directions)  (Variance captured)
```

| Matrix Type | Role in PCA |
|---|---|
| **Symmetric** | The covariance matrix $\Sigma$ is always symmetric |
| **Diagonal** | Eigenvalue matrix $\Lambda$ — holds the variance explained by each PC |
| **Orthogonal** | Eigenvector matrix $Q$ — holds the principal component directions |

---

## Quick Reference

| Property | Diagonal | Orthogonal | Symmetric |
|---|---|---|---|
| **Definition** | Off-diagonal entries = 0 | $Q^T = Q^{-1}$ | $A = A^T$ |
| **Eigenvalues** | Diagonal entries | \|λ\| = 1 | Always real |
| **Eigenvectors** | Standard basis | — | Always orthogonal |
| **Geometric effect** | Scaling only | Rotation/reflection only | Scaling along orthogonal axes |
| **Key computation** | $D^n$ = raise each entry | Cheap inverse via transpose | Spectral decomposition |

---

> **Further Reading:** These concepts are prerequisites for the [Eigen Decomposition in PCA](../02-pca.ipynb) and [PCA Implementation](../03-pca-implementation.ipynb) notebooks.
