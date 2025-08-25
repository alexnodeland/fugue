# Bayesian Linear Regression Tutorial

**Level: Intermediate** | **Time: 45 minutes**

Welcome to Bayesian linear regression with Fugue! In this tutorial, you'll learn how to build flexible regression models that quantify uncertainty in both parameters and predictions. We'll start simple and build up to sophisticated models with multiple predictors and polynomial features.

## Learning Objectives

By the end of this tutorial, you'll understand:

- How to formulate Bayesian regression models
- Prior specification for regression parameters
- Posterior inference for slope, intercept, and noise parameters
- Uncertainty quantification in predictions
- Model comparison and diagnostics
- Polynomial and multiple regression

## The Problem

You're analyzing the relationship between a company's advertising spend and revenue. You have data from 20 quarters and want to:

1. Estimate the relationship between advertising and revenue
2. Quantify uncertainty in your estimates
3. Predict revenue for future advertising budgets
4. Assess if a linear relationship is adequate

## Mathematical Setup

**Model**: Linear relationship with Gaussian noise

- Revenue = α + β × Advertising + ε
- ε ~ Normal(0, σ²)

**Priors**:

- Intercept: α ~ Normal(0, 100) [weakly informative]
- Slope: β ~ Normal(0, 10) [positive relationship expected]
- Noise: σ ~ Exponential(1) [positive, moderately informative]

**Likelihood**: Given parameters, revenue follows Normal distribution

- Revenue | α, β, σ ~ Normal(α + β × Advertising, σ²)

## Step 1: Generate Synthetic Data

Let's start by creating some realistic data:

```rust
{{#include ../../../examples/linear_regression_01_data.rs}}
```

## Step 2: Simple Linear Regression Model

Now let's build our Bayesian linear regression model:

```rust
{{#include ../../../examples/linear_regression_02_model.rs}}
```

## Step 3: Posterior Predictions with Uncertainty

Let's extend our model to make predictions with uncertainty bands:

```rust
{{#include ../../../examples/linear_regression_03_predictions.rs}}
```

## Step 4: Model Diagnostics and Validation

Let's add comprehensive model checking:

```rust
{{#include ../../../examples/linear_regression_04_diagnostics.rs}}
```

## Step 5: Polynomial Regression

Now let's extend to polynomial features to capture non-linear relationships:

```rust
{{#include ../../../examples/linear_regression_05_polynomial.rs}}
```

## Key Concepts Review

### 1. Bayesian Regression Framework

- **Parameters as random variables**: α, β, σ all have uncertainty
- **Prior specification**: Encodes domain knowledge
- **Posterior inference**: Combines prior beliefs with data
- **Prediction uncertainty**: Natural consequence of parameter uncertainty

### 2. Model Building Process

- **Start simple**: Linear relationship first
- **Add complexity gradually**: Polynomial terms, interactions
- **Validate assumptions**: Residual analysis, posterior predictive checks
- **Compare models**: Log-likelihood, cross-validation

### 3. Practical Insights

- **Uncertainty quantification**: Prediction intervals vs point estimates
- **Economic interpretation**: ROI analysis, optimal spending
- **Diminishing returns**: Quadratic terms capture non-linearity
- **Model diagnostics**: Essential for reliable inference

### 4. Fugue Features Used

- **Type-safe continuous distributions**: `Normal`, `Exponential`
- **Vector operations**: Efficient handling of multiple observations
- **MCMC inference**: `adaptive_mcmc_chain` with automatic tuning
- **Trace analysis**: Parameter extraction and correlation analysis

## Exercise: Extend the Analysis

Try these extensions to deepen your understanding:

1. **Multiple predictors**: Add seasonality or competitor spending
2. **Robust regression**: Use Student-t likelihood for outlier resistance
3. **Hierarchical priors**: Different slopes for different market segments
4. **Model selection**: Implement formal Bayesian model comparison

## Next Steps

Now that you understand Bayesian regression:

1. **[Mixture Models Tutorial](mixture-models.md)** - Handle discrete latent variables
2. **[Working with Distributions](../how-to/working-with-distributions.md)** - Master advanced distribution features
3. **[Trace Manipulation](../how-to/trace-manipulation.md)** - Advanced posterior analysis

Congratulations! You can now build sophisticated regression models that properly quantify uncertainty and make reliable predictions.

---

**Ready for discrete latent variables?** → **[Mixture Models Tutorial](mixture-models.md)**
