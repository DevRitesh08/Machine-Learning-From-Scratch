# Why Lasso Creates Sparsity

---

## Core Intuition

> **Ridge divides** — $\lambda$ in the denominator scales a coefficient down, but division can never produce zero.  
> **Lasso subtracts** — $\lambda$ is deducted directly, so once the feature's signal is weaker than $\lambda$, the coefficient hits exactly zero.

Everything below unpacks *why* that is true, from three complementary angles: the algebra, the geometry, and the optimization update.

---

## 1. The Penalty Functions and What They Cost

Both Ridge and Lasso append a penalty term to the OLS loss, but the shape of that penalty is fundamentally different.

$$\text{Ridge: } J = \underbrace{\sum(y_i - \hat{y}_i)^2}_{\text{fit}} + \lambda \underbrace{\sum b_j^2}_{\text{L2 penalty}}$$

$$\text{Lasso: } J = \underbrace{\sum(y_i - \hat{y}_i)^2}_{\text{fit}} + \lambda \underbrace{\sum |b_j|}_{\text{L1 penalty}}$$

The penalty landscapes look like this — the key difference is the **kink at zero** in L1:

![L1 vs L2 penalty shape and constraint contours](https://explained.ai/regularization/images/L1L2contour.png)

*The L1 penalty (left) has a sharp corner at the origin. The L2 penalty (right) is smooth — no corner, no hard stop.*

---

## 2. The Update Formulas — λ in the Denominator vs. Subtracted

When solving one coefficient at a time (coordinate descent), both models have an exact closed-form update per step.

Let $\rho_j$ = the partial correlation of feature $j$ with the current residuals — how much the data *wants* this coefficient to be non-zero.

### Ridge — Division

$$b_j = \frac{\rho_j}{1 + \lambda}$$

$\lambda$ is in the **denominator**. No matter how large $\lambda$ grows, you are just dividing by a bigger number. The result asymptotically approaches zero but **never reaches it**. You would need $\lambda = \infty$ to zero a coefficient.

### Lasso — Subtraction (Soft Thresholding)

$$b_j = \text{sign}(\rho_j) \cdot \max\!\Big(|\rho_j| - \lambda,\ 0\Big)$$

$\lambda$ is **subtracted from the absolute signal**. The moment $\lambda \geq |\rho_j|$, you are deducting more than what's available — the `max(..., 0)` clamps the result to exactly **zero**. Features with weak signal get fully eliminated.

### Concrete Example: $\rho_j = 0.3$, $\lambda = 0.5$

| Model | Calculation | Result |
|-------|-------------|--------|
| Ridge | $0.3 \,/\, (1 + 0.5)$ | **0.20** — survived, shrunk |
| Lasso | $\max(0.3 - 0.5,\ 0)$ | **0.00** — eliminated |

---

## 3. Why Does the Coefficient Stop at Zero and Not Go Negative?

Lasso subtracts $\lambda$ — so why doesn't it push the coefficient past zero into negative territory?

**Because zero is a two-sided barrier, not a cliff.**

Consider a positive coefficient being driven toward zero:

- **From the data side:** the feature has positive correlation with the target, so the data gradient opposes any move past zero. Going negative would mean predicting in the wrong direction — data loss increases.
- **From the penalty side:** $\lambda|b_j|$ is symmetric around zero. If the coefficient crossed to the negative side, the penalty still pulls it back toward zero from the other side.

Both forces oppose crossing zero. The `max(..., 0)` in the formula is not an artificial clamp — it is the exact solution of the sub-differential condition at the kink.

Formally, at $b_j = 0$, the sub-gradient of $|b_j|$ is the interval $[-1, +1]$. The optimality condition is satisfied for any data gradient with magnitude $\leq \lambda$ — so zero is a valid (and stable) solution whenever the signal is weak.

---

## 4. The Geometry — Why the Diamond Produces Zeros

The classic picture from *Elements of Statistical Learning* shows this most directly:

![Lasso (diamond) vs Ridge (circle) constraint regions with OLS error ellipses](https://towardsdatascience.com/wp-content/uploads/2018/09/1Jd03Hyt2bpEv1r7UijLlpg-768x576.png)

*The cyan diamond (Lasso) has corners on the coordinate axes. The green circle (Ridge) is smooth. The OLS error ellipses expand outward from the unconstrained minimum; the first contact point is the regularized solution.*

**Why the diamond corner matters:**

- Ridge (circle): The ellipse touches the smooth boundary at a point where both $\beta_1$ and $\beta_2$ are non-zero — the gradient of the circle is never parallel to a coordinate axis except at the poles.
- Lasso (diamond): The ellipse almost always hits a corner first. At the corners, one coefficient is exactly **on an axis** → the other is zero.

In higher dimensions ($p$ features), the L1 ball is a cross-polytope with an exponentially large number of corners and edges aligned with coordinate axes — making sparsity the rule, not the exception.

---

## 5. The Simulation View — L1 Encourages Zeros, L2 Merely Tolerates Them

The following simulation runs many random loss functions (different shapes and minima) and marks whether the regularized solution lands at zero (green), non-zero (blue), or near-miss (orange):

| L1 (Lasso) | L2 (Ridge) |
|:---:|:---:|
| ![L1 zero regions](https://explained.ai/regularization/images/l1-cloud.png) | ![L2 zero regions](https://explained.ai/regularization/images/l2-cloud.png) |

*Left: L1 produces zeros across a wide range of loss function positions. Right: L2 only produces zeros when the loss minimum is extremely close to the axis — essentially never in practice.*

L1 **actively encourages** zeros. L2 only permits zeros accidentally.

---

## 6. The Loss Function Shift

Adding the penalty physically moves the optimal coefficient of the combined loss toward zero. The soft constraint view makes this concrete:

![Soft constraint: loss bowl shifting toward origin as λ increases](https://explained.ai/regularization/images/ESL_reg.png)

*Left: The classic hard-constraint picture (ESL p.71). Right: The equivalent soft-constraint view — the penalty term (orange) adds to the OLS loss (blue-red bowl) and pulls the combined minimum toward the origin.*

For Ridge, the combined bowl shifts smoothly — minimum moves toward zero but stays off it.  
For Lasso, the V-shaped penalty creates a **kink at zero** in the combined loss — below a threshold, zero *is* the minimum of the combined function.

---

## 7. When Does a Feature Survive?

The threshold condition from the soft-thresholding update is explicit:

$$\boxed{|\rho_j| > \lambda \implies b_j \neq 0 \qquad |\rho_j| \leq \lambda \implies b_j = 0}$$

- $\rho_j$ is the feature's partial correlation with the target after accounting for all other features
- Strong, relevant features have large $|\rho_j|$ → survive
- Weak or redundant features have small $|\rho_j|$ → eliminated first
- Raising $\lambda$ raises the threshold → more features zeroed → sparser model

---

## Summary

| Mechanism | Ridge | Lasso |
|-----------|-------|-------|
| Penalty shape | Smooth bowl ($b^2$) | V-shape with kink ($\|b\|$) |
| Update formula | $\rho / (1+\lambda)$ — division | $\|\rho\| - \lambda$ — subtraction |
| λ position | Denominator → asymptotic shrinkage | Numerator (subtracted) → hard threshold |
| Constraint region | Circle — no corners | Diamond — corners on axes |
| Can reach zero? | No (would need $\lambda = \infty$) | Yes, when $\lambda \geq \|\rho_j\|$ |
| Why stops at zero | N/A | Both data gradient and penalty oppose crossing zero |
| Sparsity | Never (all features retained) | Yes — automatic feature selection |

> **The sparsity in Lasso is not a side effect — it is an algebraic certainty when a feature's partial correlation with the target falls below $\lambda$.**
