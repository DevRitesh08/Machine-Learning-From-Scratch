# Maximum Likelihood Estimation Guide

## What MLE Is in One Line

Maximum Likelihood Estimation (MLE) is a method for choosing model parameters that make the observed data most probable under the assumed model.

---

## Why Interviewers Ask About It

MLE is one of the core bridges between statistics and machine learning.

If you understand MLE well, you can explain:

- why mean squared error appears in linear regression
- why log loss appears in logistic regression
- how Naive Bayes estimates probabilities from data
- why training often becomes minimizing negative log-likelihood

In interviews, MLE is less about memorizing formulas and more about showing that you understand how a model, its assumptions, and its objective function connect.

---

## Probability vs Likelihood

This is the first thing interviewers usually test.

### Probability

Probability asks:

Given the parameter, how likely is the data?

Example:

If a coin has probability of heads $p = 0.7$, what is the probability of observing 8 heads in 10 tosses?

The parameter is fixed. The data is treated as random.

### Likelihood

Likelihood asks:

Given the observed data, which parameter value is most plausible?

Example:

If we observed 8 heads in 10 tosses, which value of $p$ makes that observation most believable?

The data is fixed. The parameter is treated as variable.

### Interview-Ready Distinction

Probability is a function of data given parameters.

Likelihood is the same mathematical expression viewed as a function of parameters given fixed observed data.

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

### Interview Insight

For Bernoulli data, the MLE of the success probability is the sample mean.

If $x_i \in \{0,1\}$, then:

$$
\hat{p}_{MLE} = \frac{1}{n} \sum_{i=1}^{n} x_i
$$

That is a very common interview result.

---

### Example 2: Drawing Balls From a Bag

Suppose a bag contains red and blue balls, but the red-ball proportion $p$ is unknown.

You draw 12 balls with replacement and observe 9 red and 3 blue.

The likelihood is:

$$
L(p) = p^9 (1-p)^3
$$

The MLE is:

$$
\hat{p} = \frac{9}{12} = 0.75
$$

### Why This Example Matters

It is the same structure as coin toss, but it helps you explain that MLE is not about coins. It is about estimating parameters of a probability model from observed data.

---

### Example 3: Normal Distribution

Assume:

$$
x_i \sim \mathcal{N}(\mu, \sigma^2)
$$

with unknown $\mu$ and $\sigma^2$.

The likelihood is:

$$
L(\mu, \sigma^2) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i-\mu)^2}{2\sigma^2}\right)
$$

The log-likelihood simplifies to:

$$
\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i-\mu)^2
$$

Maximizing this gives:

$$
\hat{\mu}_{MLE} = \frac{1}{n} \sum_{i=1}^{n} x_i
$$

and

$$
\hat{\sigma}^2_{MLE} = \frac{1}{n} \sum_{i=1}^{n} (x_i - \hat{\mu})^2
$$

### Important Interview Detail

The MLE for variance uses $1/n$, not $1/(n-1)$.

Why?

- $1/n$ comes from maximizing likelihood
- $1/(n-1)$ appears in the unbiased sample variance estimator

This distinction is asked often.

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

### Interview Sound Bite

Linear regression uses MSE because assuming Gaussian noise makes MSE equivalent to negative log-likelihood minimization.

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

### Interview Sound Bite

Logistic regression is trained by MLE under a Bernoulli model for the labels.

---

### Naive Bayes

Naive Bayes also uses MLE, but in a different way.

Instead of optimizing weights with gradient descent, it estimates probabilities directly from counts.

Examples:

- class prior: $P(y=c)$ is estimated from class frequency
- conditional probability: $P(x_j \mid y=c)$ is estimated from feature counts or distribution-specific formulas

### Interview Sound Bite

Naive Bayes is a generative model, and many of its parameter estimates come directly from MLE applied to counts or distribution parameters.

---

## MLE and Loss Functions

One of the most useful interview ideas is this:

Many popular loss functions are just negative log-likelihoods under specific probabilistic assumptions.

Examples:

- squared error corresponds to Gaussian noise
- cross-entropy corresponds to Bernoulli or categorical likelihoods
- Poisson loss corresponds to Poisson-distributed count targets

### Practical Interpretation

When you minimize a loss, you are often implicitly making a distributional assumption.

That is why MLE matters.

---

## Why Minimize Loss Instead of Maximize Likelihood

These are usually the same optimization problem, just written differently.

We often prefer loss functions because:

- minimization is the more common optimization convention in ML libraries
- negative log-likelihood is easier to compute than raw likelihood
- customized losses are sometimes more convenient than strict probabilistic modeling
- not every algorithm is naturally expressed through a likelihood

### Clean Interview Answer

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

### Interview Shortcut

MLE ignores priors, MAP adds priors, Bayesian inference keeps uncertainty instead of collapsing everything to a single point estimate.

---

## Common Interview Mistakes

- confusing likelihood with probability
- forgetting that likelihood is a function of parameters, not data
- saying MLE always works without mentioning assumptions
- missing the connection between MLE and standard loss functions
- claiming the Gaussian variance MLE uses $1/(n-1)$ instead of $1/n$
- saying all machine learning algorithms use MLE

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

---

## Most Important Interview Questions

### 1. What is maximum likelihood estimation?

MLE is a method for estimating parameters by choosing the values that maximize the probability of the observed data under the assumed model.

### 2. What is the difference between probability and likelihood?

Probability treats parameters as fixed and data as random. Likelihood treats observed data as fixed and varies the parameters to see which values best explain the data.

### 3. Why do we use log-likelihood instead of likelihood?

Because it converts products into sums, improves numerical stability, and gives the same optimum since log is monotonic.

### 4. Why is MLE important in machine learning?

Because many training objectives are derived from MLE. It explains where losses like MSE and cross-entropy come from.

### 5. How is MLE related to linear regression?

If residuals are Gaussian, maximizing likelihood is equivalent to minimizing squared error.

### 6. How is MLE related to logistic regression?

Logistic regression assumes Bernoulli labels, and maximizing the Bernoulli likelihood is equivalent to minimizing binary cross-entropy.

### 7. What is the MLE of the Bernoulli parameter?

The sample mean:

$$
\hat{p} = \frac{1}{n}\sum_{i=1}^{n} x_i
$$

### 8. What is the MLE of the mean of a normal distribution?

The sample mean.

### 9. What is the MLE of the variance of a normal distribution?

$$
\hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i-\hat{\mu})^2
$$

not the unbiased estimator with $1/(n-1)$.

### 10. Is minimizing loss always the same as maximizing likelihood?

Not always, but for many probabilistic models the loss is exactly the negative log-likelihood, possibly up to constants.

### 11. What are the assumptions behind MLE?

Correct model specification, often i.i.d. samples, identifiability, and enough data.

### 12. What are the weaknesses of MLE?

It can be sensitive to wrong assumptions, outliers, small sample sizes, and complex non-convex optimization landscapes.

### 13. What is the difference between MLE and MAP?

MLE uses only data. MAP uses data plus a prior over parameters.

### 14. Do all machine learning algorithms use MLE?

No. Many important algorithms do, but many non-parametric, clustering, and reinforcement-learning methods do not.

### 15. Why should we study MLE if we already use loss functions?

Because MLE gives the statistical meaning behind the loss, clarifies model assumptions, and improves your ability to reason about model design.

---

## Short Revision Box

- MLE finds parameters that make observed data most likely
- optimize log-likelihood, not raw likelihood
- negative log-likelihood becomes a loss function
- linear regression plus Gaussian noise leads to MSE
- logistic regression plus Bernoulli labels leads to cross-entropy
- Bernoulli MLE equals sample mean
- Gaussian variance MLE uses $1/n$
- MLE is powerful, but only under the model assumptions

---

## Final Interview Answer Template

If asked to explain MLE in under a minute, say:

Maximum Likelihood Estimation is a parameter estimation method where we choose the parameter values that make the observed data most probable under an assumed statistical model. In practice, we maximize the log-likelihood because it is easier to optimize. In machine learning, this idea explains many common loss functions. For example, linear regression with Gaussian noise gives mean squared error, and logistic regression with Bernoulli labels gives cross-entropy loss. So MLE is not just a statistics concept. It is one of the main reasons common ML training objectives look the way they do.

---

## Source Context Covered

This guide consolidates and sharpens the core themes from the reference session:

- probability vs likelihood
- coin toss and bag-draw intuition
- MLE for normal distributions
- MLE in machine learning
- MLE in logistic regression
- most important interview questions