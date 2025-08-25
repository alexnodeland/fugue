# Bayesian Coin Flip Tutorial

**Level: Beginner** | **Time: 30 minutes**

Welcome to your first complete Bayesian analysis with Fugue! In this tutorial, we'll estimate the bias of a coin using Bayesian inference. You'll learn fundamental concepts while building a complete, working probabilistic program.

## Learning Objectives

By the end of this tutorial, you'll understand:

- How to formulate a Bayesian model
- The relationship between priors, likelihoods, and posteriors
- How to use conjugate priors for exact solutions
- How to run MCMC inference in Fugue
- How to analyze and interpret results

## The Problem

You have a coin that might be biased. You flip it 10 times and observe 7 heads. Questions:

1. What's the most likely bias of the coin?
2. How confident can we be in our estimate?
3. What's the probability the coin is fair (50% heads)?

We'll answer these questions using Bayesian inference.

## Mathematical Setup

**Prior**: We start with a uniform belief about the coin's bias

- Bias ~ Beta(1, 1) [uniform on [0, 1]]

**Likelihood**: Given the bias, heads follow a binomial distribution

- Heads | Bias ~ Binomial(n=10, p=bias)

**Posterior**: After observing 7 heads out of 10 flips

- Bias | Heads=7 ~ Beta(1+7, 1+10-7) = Beta(8, 4)

This is a **conjugate analysis** - the Beta prior combined with a Binomial likelihood gives a Beta posterior.

## Step 1: Basic Model Implementation

Let's start with the simplest possible model:

```rust
{{#include ../../../examples/bayesian_coin_flip/basic_model_implementation.rs}}
```

**Try it**: Save this as `src/main.rs` and run `cargo run`.

## Step 2: MCMC Inference

Prior sampling doesn't give us proper posterior samples. Let's use MCMC to get the true posterior:

```rust
{{#include ../../../examples/bayesian_coin_flip/mcmc_inference.rs}}
```

## Step 3: Analytical Comparison

Since we're using conjugate priors, we can compute the exact posterior analytically. Let's compare:

```rust
{{#include ../../../examples/bayesian_coin_flip/analytical_comparison.rs}}
```

## Step 4: Exploring Different Scenarios

Let's explore how different data affects our conclusions:

```rust
{{#include ../../../examples/bayesian_coin_flip/exploring_different_scenarios.rs}}
```

## Step 5: Advanced Analysis with Multiple Questions

Let's extend our model to answer more sophisticated questions:

```rust
{{#include ../../../examples/bayesian_coin_flip/advanced_analysis.rs}}
```

## Key Concepts Review

Let's solidify the key concepts from this tutorial:

### 1. Bayesian Framework

- **Prior**: What we believe before seeing data
- **Likelihood**: How probable the data is given our hypothesis
- **Posterior**: What we believe after seeing data (prior × likelihood)

### 2. Conjugate Analysis

- Beta + Binomial = Beta (convenient mathematical property)
- Allows exact analytical solutions
- MCMC confirms these analytical results

### 3. Practical Insights

- **More data = more precision**: 7/10 vs 28/40 heads
- **Extreme observations**: 0/10 or 10/10 heads strongly suggest bias
- **Fair coin hypothesis**: Can be tested probabilistically

### 4. Fugue Features Used

- **Type-safe distributions**: `Bernoulli` returns `bool`, `Binomial` returns `u64`
- **`prob!` macro**: Clean, readable model specification
- **MCMC inference**: `adaptive_mcmc_chain` for automatic tuning
- **Trace analysis**: Extract and analyze parameter samples

## Exercise: Extend the Analysis

Try these extensions to deepen your understanding:

1. **Different priors**: Use `Beta(2, 2)` (weakly favors fair) or `Beta(0.5, 0.5)` (Jeffrey's prior)
2. **Model comparison**: Compare fair coin vs biased coin models using model evidence
3. **Sequential updating**: Update beliefs as you observe more flips one by one
4. **Multiple coins**: Analyze several coins simultaneously with hierarchical modeling

## Next Steps

Now that you understand basic Bayesian inference:

1. **[Linear Regression Tutorial](linear-regression.md)** - Continuous parameters and multiple variables
2. **[Understanding Models](../getting-started/understanding-models.md)** - Deepen your model composition skills
3. **[Working with Distributions](../how-to/working-with-distributions.md)** - Master all distribution types

Congratulations! You've completed your first Bayesian analysis with Fugue. You now understand the core concepts of probabilistic programming and can apply them to real problems.

---

**Ready for more complex models?** → **[Linear Regression Tutorial](linear-regression.md)**
