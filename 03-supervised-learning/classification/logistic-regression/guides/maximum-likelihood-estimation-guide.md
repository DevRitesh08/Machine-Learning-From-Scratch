# Maximum Likelihood Estimation

## Definition

Maximum Likelihood Estimation (MLE) is a method for choosing model parameters that make the observed data most probable under the assumed model.

> It is only used for parameteric models where we have a likelihood function defined, examples include linear regression, logistic regression, Naive Bayes, and many probabilistic models.

### Quick Intuitive Example

Suppose a coin is tossed 10 times, and you observe 8 heads.

Now test three candidate values for the probability of heads $p$:

- $p = 0.3$: seeing 8 heads would be surprising
- $p = 0.5$: seeing 8 heads is possible but not very typical
- $p = 0.8$: seeing 8 heads is very plausible

MLE says: choose the value of $p$ that makes the observed result (8 heads) most plausible.

So here, the estimate is near $p = 0.8$.

---

## Intuition Before the Math

Suppose you see a coin produce 8 heads in 10 tosses.

Now imagine testing different candidate values of $p$, the probability of heads:

- if $p = 0.2$, then 8 heads looks very unlikely
- if $p = 0.5$, then 8 heads is possible but not especially typical
- if $p = 0.8$, then 8 heads looks very reasonable

MLE simply formalizes this intuition. It asks:

Which parameter value makes the data we actually saw look most believable?

That is the parameter estimate we keep.

---

## Intuition of Likelihood

Likelihood is a way of scoring how well different parameter values explain the data you have already observed.

Think of it like this:

- the data is already on the table
- the model form is fixed
- only the parameter value is being tested

Suppose again that we observed 8 heads in 10 tosses.

Now test three possible values of $p$:

- $p = 0.2$: this says heads is rare, so 8 heads looks poorly explained
- $p = 0.5$: this says heads and tails are equally likely, so 8 heads is possible but not especially convincing
- $p = 0.8$: this says heads is common, so 8 heads looks well explained

Likelihood is not asking, "What can happen next?"

It is asking, "Which parameter value makes what already happened look most reasonable?"

That is the intuition behind likelihood: it is a goodness-of-explanation score for parameter values.

---

## Intuition of Maximum Likelihood

Once we understand likelihood as a score over parameter values, maximum likelihood becomes very natural.

Maximum likelihood means:

Out of all possible parameter values, choose the one with the highest likelihood.

So the word "maximum" is doing something simple but important. We are not just evaluating likelihood. We are searching for the best-scoring parameter.

For the coin example:

- many values of $p$ are possible
- each value gives a different likelihood for the observed 8 heads and 2 tails
- the value that gives the highest likelihood becomes our estimate

That best value is the maximum likelihood estimate.

So before thinking about formulas, it is enough to remember this:

Likelihood tells us how well a parameter explains the observed data.

Maximum likelihood chooses the parameter that explains it best.

---

## Probability vs Likelihood

Probability and likelihood use the same underlying formula, but they answer different questions.

### Probability

Probability asks:

Given the parameter, how likely is the data?

Example:

If a coin has probability of heads $p = 0.7$, what is the probability of observing 8 heads in 10 tosses?

The parameter is fixed. The data is treated as random.

This is the standard forward view used in probability theory.

### Likelihood

Likelihood asks:

Given the observed data, which parameter value is most plausible?

Example:

If we observed 8 heads in 10 tosses, which value of $p$ makes that observation most believable?

The data is fixed. The parameter is treated as variable.

This is the reverse view used for parameter estimation.

### Clean Distinction

Probability is a function of data given parameters.

Likelihood is the same mathematical expression viewed as a function of parameters given fixed observed data.

### Mental Model

- probability predicts possible outcomes before seeing data
- likelihood scores parameter values after seeing data

If you remember that probability is about outcomes and likelihood is about parameters, the distinction becomes much easier.

---

## Core MLE Idea

Assume we have data points $x_1, x_2, \dots, x_n$ and a model with parameter $\theta$.

The likelihood is:

$$
L(\theta) = P(x_1, x_2, \dots, x_n \mid \theta)
$$

If the observations are independent and identically distributed (i.i.d.), then:

$$
L(\theta) = \prod_{i=1}^{n} P(x_i \mid \theta)
$$

MLE chooses:

$$
\hat{\theta}_{MLE} = \arg\max_{\theta} L(\theta)
$$

In practice, we maximize the log-likelihood instead:

$$
\ell(\theta) = \log L(\theta) = \sum_{i=1}^{n} \log P(x_i \mid \theta)
$$

This gives the same optimum because log is a strictly increasing function.

### What MLE Is Doing Conceptually

MLE does not ask whether the observed data was absolutely likely in a universal sense. It compares parameter values relative to each other.

For example, a dataset may be rare under every parameter setting, but one setting can still be less bad than the others. MLE chooses the best-fitting parameter among the candidates allowed by the model.

---

## Why We Usually Use Log-Likelihood

- products become sums, which are easier to differentiate
- numerical stability improves because tiny probabilities do not underflow as easily
- optimization becomes cleaner and better aligned with gradient-based methods

In machine learning, we usually minimize:

$$
-\ell(\theta)
$$

This is the negative log-likelihood (NLL).

### Why Negative Log-Likelihood Becomes a Loss

Machine learning frameworks are usually built around minimization rather than maximization. So instead of maximizing log-likelihood, we minimize its negative.

That is why many model-training objectives are written as losses even when their statistical meaning comes from likelihood.

---

## A General Recipe for MLE

When deriving an MLE, the usual workflow is:

1. Choose a probability model for the data.
2. Write the likelihood of the observed sample.
3. Take the logarithm to get the log-likelihood.
4. Differentiate with respect to the parameter or parameters.
5. Set the derivative to zero and solve.
6. Check that the solution corresponds to a maximum.

This pattern appears again and again in Bernoulli, Gaussian, Poisson, exponential-family, and many machine learning models.

---

## Intuition Through Examples

### Example 1: Coin Toss

Suppose a coin is tossed 10 times and we observe 8 heads and 2 tails.

Let $p$ be the probability of heads.

The likelihood is:

$$
L(p) = p^8 (1-p)^2
$$

The log-likelihood is:

$$
\ell(p) = 8 \log p + 2 \log(1-p)
$$

Differentiate and set to zero:

$$
\frac{8}{p} - \frac{2}{1-p} = 0
$$

Solving gives:

$$
\hat{p} = 0.8
$$

### Key Takeaway

For Bernoulli data, the MLE of the success probability is the sample mean.

If $x_i \in \{0,1\}$, then:

$$
\hat{p}_{MLE} = \frac{1}{n} \sum_{i=1}^{n} x_i
$$

This is a useful result because it shows that MLE often produces estimators that are intuitive. If 80 percent of the observed outcomes are successes, then the estimated success probability becomes 0.8.

---

## MLE in Machine Learning

### Linear Regression

If we assume:

$$
y_i = w^T x_i + b + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2)
$$

then maximizing the likelihood of the observed targets is equivalent to minimizing:

$$
\sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

So under Gaussian noise assumptions:

- MLE leads to least squares
- mean squared error is a negative log-likelihood up to constants

Linear regression uses MSE because assuming Gaussian noise makes MSE equivalent to negative log-likelihood minimization.

This connection is one of the main reasons MLE matters in machine learning. A loss function is often not arbitrary. It usually reflects an assumption about the data-generating process.

---

### Logistic Regression

For binary classification, assume:

$$
P(y_i = 1 \mid x_i) = \sigma(w^T x_i + b)
$$

where $\sigma(z)$ is the sigmoid function.

Each label follows a Bernoulli distribution, so the likelihood is:

$$
L(w,b) = \prod_{i=1}^{n} p_i^{y_i}(1-p_i)^{1-y_i}
$$

Taking logs gives the log-likelihood:

$$
\ell(w,b) = \sum_{i=1}^{n} \left[y_i \log p_i + (1-y_i)\log(1-p_i)\right]
$$

Maximizing this is equivalent to minimizing:

$$
-\sum_{i=1}^{n} \left[y_i \log p_i + (1-y_i)\log(1-p_i)\right]
$$

which is binary cross-entropy or log loss.

Logistic regression is trained by MLE under a Bernoulli model for the labels.

So logistic regression is not just a classification rule with a sigmoid. It is a probabilistic model trained by maximizing the likelihood of observed class labels.

---

### Naive Bayes

Naive Bayes also uses MLE, but in a different way.

Instead of optimizing weights with gradient descent, it estimates probabilities directly from counts.

Examples:

- class prior: $P(y=c)$ is estimated from class frequency
- conditional probability: $P(x_j \mid y=c)$ is estimated from feature counts or distribution-specific formulas

Naive Bayes is a generative model, and many of its parameter estimates come directly from MLE applied to counts or distribution parameters.

This is why Naive Bayes training often looks simple. In many cases, the MLEs can be written down directly from counts or sample statistics without iterative optimization.

---

## MLE and Loss Functions

One of the most useful ideas in machine learning is this:

Many popular loss functions are just negative log-likelihoods under specific probabilistic assumptions.

Examples:

- squared error corresponds to Gaussian noise
- cross-entropy corresponds to Bernoulli or categorical likelihoods
- Poisson loss corresponds to Poisson-distributed count targets

### Practical Interpretation

When you minimize a loss, you are often implicitly making a distributional assumption.

That is why MLE matters.

For example:

- choosing squared error means you are treating residuals as Gaussian
- choosing binary cross-entropy means you are treating labels as Bernoulli
- choosing categorical cross-entropy means you are treating class labels as categorical outcomes

---

## Why Minimize Loss Instead of Maximize Likelihood

These are usually the same optimization problem, just written differently.

We often prefer loss functions because:

- minimization is the more common optimization convention in ML libraries
- negative log-likelihood is easier to compute than raw likelihood
- customized losses are sometimes more convenient than strict probabilistic modeling
- not every algorithm is naturally expressed through a likelihood

We usually minimize loss because negative log-likelihood is easier to optimize, more numerically stable, and consistent with standard optimization tooling. For many models, minimizing loss is exactly equivalent to maximizing likelihood.

---

## Assumptions Behind MLE

MLE is powerful, but it is only as good as the model assumptions.

Common assumptions include:

- the chosen probability model is appropriate for the data
- observations are often assumed i.i.d.
- the model is identifiable, meaning different parameter values produce different distributions
- there is enough data for stable estimation

If these assumptions fail, MLE may produce poor or misleading estimates.

This is an important point for understanding model quality. MLE can be mathematically correct for the chosen model and still perform poorly if the model itself is a bad description of reality.

---

## Advantages of MLE

- principled and widely applicable
- often gives estimators with good asymptotic properties
- connects directly to optimization objectives used in machine learning
- makes model comparison possible through likelihood-based criteria such as AIC and BIC
- forms the foundation for advanced methods like MAP estimation, EM, and many probabilistic models

---

## Limitations of MLE

- sensitive to model misspecification
- can overfit when the model is too flexible
- can be sensitive to outliers
- may not have a closed-form solution
- can become numerically difficult in high-dimensional or non-convex problems

For this reason, MLE is often combined with regularization, priors, or robust alternatives when the raw maximum-likelihood solution is unstable.

---

## MLE vs MAP vs Bayesian Inference

### MLE

Chooses the parameter that maximizes:

$$
P(D \mid \theta)
$$

It uses only the observed data.

### MAP

Chooses the parameter that maximizes:

$$
P(\theta \mid D) \propto P(D \mid \theta)P(\theta)
$$

It combines data with a prior belief about parameters.

### Bayesian Inference

Does not output just one best parameter value. It keeps the full posterior distribution over parameters.

MLE ignores priors, MAP adds priors, Bayesian inference keeps uncertainty instead of collapsing everything to a single point estimate.

This comparison is useful because it shows where MLE sits in the broader family of estimation methods. MLE gives a single best parameter value based only on the observed data. MAP modifies that estimate with prior knowledge. Bayesian inference goes further by modeling uncertainty over parameters explicitly.

---

## Common Mistakes and Misunderstandings

- confusing likelihood with probability
- forgetting that likelihood is a function of parameters, not data
- saying MLE always works without mentioning assumptions
- missing the connection between MLE and standard loss functions
- claiming the Gaussian variance MLE uses $1/(n-1)$ instead of $1/n$
- saying all machine learning algorithms use MLE

Another common mistake is to think that MLE always gives a closed-form answer. In simple models like Bernoulli and Gaussian distributions, it often does. In more complex models, we usually need iterative optimization such as gradient descent or second-order methods.

---

## Does Every ML Algorithm Use MLE

No.

MLE is a general statistical idea, but not every ML algorithm is built around it.

Examples that often use MLE:

- linear regression under Gaussian noise assumptions
- logistic regression
- Naive Bayes
- many neural-network classification models through cross-entropy

Examples that are not naturally MLE-first:

- k-nearest neighbors
- decision trees in their standard form
- k-means clustering
- many reinforcement learning methods

So MLE is broad and foundational, but it is not the only way to define a learning objective.

---

## 5 Most Important Questions

### 1. Is Maximum Likelihood Estimation a general concept applicable to all machine learning algorithms? 

Maximum Likelihood Estimation is a general statistical concept and is widely used in machine learning, but it is not used by every algorithm.

It works naturally when:

- the model is parametric
- a likelihood function can be defined
- training can be written as fitting parameters to observed data

Common ML models where MLE is central:

- linear regression (under Gaussian error assumptions)
- logistic regression (Bernoulli likelihood)
- Naive Bayes (probabilities estimated from counts/distributions)
- many neural network classifiers (via cross-entropy as NLL)

Examples where MLE is not the primary training principle:

- k-nearest neighbors
- standard decision trees
- k-means clustering
- many reinforcement learning methods

Example:

In logistic regression, we maximize the likelihood of class labels, so MLE is direct. In k-nearest neighbors, there are no global parameters learned via likelihood, so MLE is not the core framework.

### 2. How is MLE related to the concept of loss functions?

MLE and loss minimization are closely related. In many models, minimizing a specific loss is exactly equivalent to maximizing likelihood (or log-likelihood).

The key identity is:

$$
\text{Minimize loss} \quad \Longleftrightarrow \quad \text{Minimize } -\log L(\theta)
$$

Examples:

- linear regression with Gaussian noise: minimizing MSE corresponds to maximizing Gaussian likelihood
- logistic regression: minimizing binary cross-entropy corresponds to maximizing Bernoulli likelihood
- multiclass classification: minimizing categorical cross-entropy corresponds to maximizing categorical likelihood

Interpretation:

When you choose a loss function, you are often choosing an implicit probability model for your data.

### 3. If MLE is so useful, why do we usually talk about minimizing loss instead of maximizing likelihood?

These two views are usually the same optimization problem written in different language.

We often use loss language because:

- optimization toolchains are built around minimization
- negative log-likelihood is numerically more stable than raw likelihood products
- loss functions can be adapted to business goals (class weighting, robustness, margin-based behavior)
- not every practical model is written from a strict likelihood perspective

Example:

In deep learning code, we call `BCEWithLogitsLoss` or `CrossEntropyLoss`. Mathematically, these are negative log-likelihood objectives, but operationally we optimize them as losses.

### 4. Why should we study MLE if we can already train models using losses and optimizers?

Studying MLE gives a stronger foundation for understanding why a training objective works and when it might fail.

What MLE gives you:

- statistical meaning behind common losses
- clarity on model assumptions (for example, Gaussian residuals)
- better model diagnostics when assumptions break
- access to likelihood-based model comparison ideas (AIC/BIC)
- a stepping stone to MAP estimation, EM, and Bayesian inference

Example:

If residuals are heavy-tailed but you still use MSE, MLE thinking helps you realize your Gaussian assumption may be poor. You can then switch to a more robust objective and justify that choice statistically.

### 5. What are the most important assumptions and pitfalls of MLE in practice?

MLE is powerful, but it depends on assumptions.

Core assumptions:

- the model family is appropriate for the data
- observations are often treated as i.i.d.
- parameters are identifiable
- enough data is available for stable estimation

Common pitfalls:

- model misspecification leads to misleading estimates
- outliers can distort estimates significantly
- non-convex likelihood surfaces can cause optimization difficulty
- small sample sizes can produce unstable parameter estimates

Example:

In a small dataset with a few extreme points, Gaussian MLE can overreact to outliers. A robust alternative or regularized approach can produce better generalization.
