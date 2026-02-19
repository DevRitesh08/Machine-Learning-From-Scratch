# Why Lasso Creates Sparsity

> **Short answer:** The L1 penalty has a *constant* gradient everywhere except zero. That constant pull is strong enough to drag small coefficients all the way to exactly zero — something Ridge's shrinking gradient can never do.

---

## 1. The Penalty Functions Side by Side

| | Ridge (L2) | Lasso (L1) |
|---|---|---|
| Penalty term | $\lambda \sum b_j^2$ | $\lambda \sum \|b_j\|$ |
| Gradient of penalty | $2\lambda b_j$ → shrinks as $b_j \to 0$ | $\lambda \cdot \text{sign}(b_j)$ → **constant** |
| Can reach zero? | No | **Yes** |

---

## 2. The Gradient Argument (Why It Matters)

At optimum, the total gradient must equal zero:

$$\frac{\partial \text{Loss}}{\partial b_j} + \frac{\partial \text{Penalty}}{\partial b_j} = 0$$

### Ridge
The penalty gradient is $2\lambda b_j$.  
As $b_j$ gets smaller, the penalty gradient also gets smaller — the penalty "gives up" just before pushing the coefficient to zero.

$$\text{penalty gradient} \xrightarrow{b_j \to 0} 0$$

The coefficient is attracted toward zero but **never reaches it**.

### Lasso
The penalty gradient is $\lambda \cdot \text{sign}(b_j)$ — it is **±λ regardless of how small $b_j$ is**.

$$\text{penalty gradient} = \pm\lambda \quad \text{(constant, no matter how small } b_j \text{ is)}$$

If the data gradient at $b_j = 0$ is smaller than $\lambda$ in magnitude, the only solution that satisfies the zero-gradient condition is $b_j = 0$ exactly.  
The coefficient is not just attracted — it is **forced** to zero.

---

## 3. The Geometry Argument (Most Intuitive)

In 2D coefficient space, the optimization looks like this:

```
               β₂
               |
               |    ← OLS minimum (unconstrained)
          *    |
        /   \  |
       |  ●  \ |          ● = OLS minimum
       |       \|
───────┼────────┼──────────  β₁
       |       /|
       |      / |
        \   /   |
          \/    |
          ▲
     Diamond (Lasso)    Circle (Ridge would be here)
```

- **Ridge constraint**: $\beta_1^2 + \beta_2^2 \leq t$ → a **circle** (smooth, no corners)
- **Lasso constraint**: $|\beta_1| + |\beta_2| \leq t$ → a **diamond** (sharp corners on the axes)

When the OLS error ellipses expand outward looking for the constrained minimum:
- They almost always **hit the diamond at a corner first**
- Corners sit exactly on the axes → one (or more) coefficients = **0**
- A circle has no corners → the ellipse touches it on the smooth edge → both coefficients non-zero

**Sparsity is a direct consequence of the diamond shape.**

---

## 4. The Soft Thresholding Argument (Mathematical)

Lasso is solved via **coordinate descent**. For each coordinate $j$, the update has a closed form:

$$b_j = \text{sign}(\rho_j) \cdot \max(|\rho_j| - \lambda, 0)$$

where $\rho_j$ is the partial residual correlation for feature $j$.

This is called **soft thresholding**:

```
        output b_j
            |          /  (slope = 1)
            |         /
            |        /
   ─────────┼───────•────────── ρⱼ (input)
            |   •───
            |  / (zero zone: |ρⱼ| ≤ λ)
            | /
```

- If $|\rho_j| \leq \lambda$ → $b_j = 0$ exactly (the feature is completely removed)
- If $|\rho_j| > \lambda$ → $b_j$ is shrunk by $\lambda$ but remains non-zero

**The zero zone is what creates sparsity.** Ridge uses a different update that scales but never hard-zeros.

---

## 5. Intuitive Summary

Think of it like this:

> Ridge is like applying **friction** to the coefficients — they slow down but keep moving forever.  
> Lasso is like applying **a wall at zero** — once the coefficient's evidence from the data is weaker than $\lambda$, it hits the wall and stays there.

| Analogy | Ridge | Lasso |
|---------|-------|-------|
| Physical | Ball on a slope with air resistance (slows, never stops) | Ball on a slope with a wall at zero (stops hard) |
| Signal processing | Shrinkage | Soft thresholding |
| Effect | All features kept, weights reduced | Some features eliminated entirely |

---

## 6. When Does a Feature Get Zeroed Out?

A feature $j$ gets zeroed out when its **correlation with the residual** is weaker than the regularization threshold $\lambda$:

$$|\rho_j| \leq \lambda \implies b_j = 0$$

In plain English:
- **Relevant features** have high correlation with the target → survive
- **Irrelevant/redundant features** have low correlation → eliminated first
- Higher $\lambda$ → higher threshold → more features zeroed out

---

## Key Takeaways

1. **L1's constant gradient** doesn't decrease near zero → it can overpower the data gradient → coefficient set to zero
2. **L2's shrinking gradient** decreases as coefficient shrinks → data gradient always wins at small values → never reaches zero
3. **Geometrically**, the L1 diamond has corners on the axes; constrained solutions tend to land there
4. **Algorithmically**, Lasso uses soft thresholding — features below the threshold are hard-set to zero
5. **Sparsity ∝ λ** — larger λ = more features zeroed out = simpler, more interpretable model
