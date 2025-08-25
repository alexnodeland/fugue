# Mixture Models Tutorial

**Level: Intermediate** | **Time: 50 minutes**

Welcome to mixture modeling with Fugue! In this tutorial, you'll learn how to build models with discrete latent variables, handle multi-modal data, and perform clustering with uncertainty quantification. We'll build from simple 2-component mixtures to sophisticated model selection frameworks.

## Learning Objectives

By the end of this tutorial, you'll understand:

- Mixture model formulation with latent categorical variables
- Component assignment and parameter inference
- Model selection for unknown number of components
- Clustering vs classification in a Bayesian framework
- Initialization strategies and convergence diagnostics

## The Problem

You're analyzing customer spending patterns at an e-commerce site. The data shows clear multi-modal behavior - different types of customers with distinct spending habits. Questions:

1. How many distinct customer segments exist?
2. What characterizes each segment?
3. Which segment does a new customer belong to?
4. How confident can we be in our segment assignments?

## Mathematical Setup

**Model**: Gaussian mixture with K components

- Customer type: z_i ~ Categorical(π₁, π₂, ..., πₖ)
- Spending given type: x_i | z_i=k ~ Normal(μₖ, σₖ²)

**Priors**:

- Mixing weights: π ~ Dirichlet(α, α, ..., α)
- Component means: μₖ ~ Normal(μ₀, τ²)
- Component scales: σₖ ~ Exponential(λ)

**Likelihood**: Marginalized over latent assignments

- x_i ~ Σₖ πₖ × Normal(μₖ, σₖ²)

## Step 1: Generate Mixture Data

Let's create realistic customer spending data with multiple segments:

```rust
{{#include ../../../examples/mixture_models_01_data.rs}}
```

## Step 2: Two-Component Mixture Model

Let's start with a simple 2-component mixture:

```rust
{{#include ../../../examples/mixture_models_02_two_component.rs}}
```

## Step 3: Three-Component Mixture with Full Bayesian Treatment

Now let's build a proper 3-component mixture model with explicit latent variables:

```rust
{{#include ../../../examples/mixture_models_03_bayesian.rs}}
```

## Step 4: Model Selection and Information Criteria

Let's implement model selection to automatically determine the optimal number of components:

```rust
{{#include ../../../examples/mixture_models_04_model_selection.rs}}
```

## Step 5: Clustering New Customers

Finally, let's use our fitted model to classify new customers:

```rust
{{#include ../../../examples/mixture_models_05_clustering.rs}}
```

## Key Concepts Review

### 1. Mixture Model Framework

- **Latent variables**: Unobserved component assignments
- **Hierarchical structure**: Components → assignments → observations
- **Identifiability**: Label switching and parameter ordering
- **Model complexity**: Balance fit vs overfitting

### 2. Bayesian Clustering vs K-means

- **Uncertainty quantification**: Probabilistic assignments vs hard clusters
- **Automatic complexity selection**: Information criteria vs fixed K
- **Robustness**: Handles overlapping clusters naturally
- **Interpretability**: Component-specific parameters have meaning

### 3. Model Selection

- **Information criteria**: AIC vs BIC trade-offs
- **Cross-validation**: Out-of-sample predictive performance
- **Posterior model probabilities**: Full Bayesian model comparison
- **Practical considerations**: Computational cost vs accuracy

### 4. Fugue Features Used

- **Categorical variables**: Component assignments via uniform sampling
- **Complex model structure**: Nested loops and conditional observations
- **MCMC robustness**: Handles multi-modal posteriors effectively
- **Type safety**: Natural handling of discrete and continuous variables

## Exercise: Extend the Analysis

Try these extensions to deepen your understanding:

1. **Multivariate mixtures**: Add customer age and purchase frequency
2. **Non-Gaussian components**: Use Student-t or skewed distributions
3. **Infinite mixtures**: Implement Dirichlet Process mixtures
4. **Time-varying mixtures**: Handle evolving customer segments

## Next Steps

Now that you understand mixture modeling:

1. **[Hierarchical Models Tutorial](hierarchical-models.md)** - Multi-level modeling with shared parameters
2. **[Custom Handlers](../how-to/custom-handlers.md)** - Build specialized mixture model interpreters
3. **[Debugging Models](../how-to/debugging-models.md)** - Diagnose convergence issues in complex models

Congratulations! You can now build sophisticated clustering models that handle uncertainty and automatically select model complexity.

---

**Ready for hierarchical modeling?** → **[Hierarchical Models Tutorial](hierarchical-models.md)**
