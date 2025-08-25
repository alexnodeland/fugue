# Hierarchical Models Tutorial

**Level: Advanced** | **Time: 60 minutes**

Welcome to hierarchical modeling with Fugue! In this tutorial, you'll learn how to build multi-level models with shared parameters, understand partial pooling and shrinkage effects, and handle complex grouped data structures. This is the most sophisticated modeling approach in the tutorial series.

## Learning Objectives

By the end of this tutorial, you'll understand:

- Hierarchical model structure and shared hyperparameters
- Partial pooling vs complete pooling vs no pooling
- Shrinkage effects and when they occur
- Group-level and population-level inference
- Hierarchical model diagnostics and interpretation

## The Problem

You're analyzing student test scores across multiple schools in a district. Each school has different numbers of students, teacher quality, and resources. You want to:

1. Estimate the true performance of each school
2. Account for varying sample sizes across schools
3. Borrow strength between similar schools
4. Predict performance for new schools
5. Identify schools that are truly exceptional vs lucky

This is a classic hierarchical modeling problem where schools are exchangeable units with shared population-level parameters.

## Mathematical Setup

**Hierarchical Structure**:

- Population level: μ, τ (overall mean and between-school variance)
- School level: θⱼ ~ Normal(μ, τ) for j = 1, ..., J
- Student level: yᵢⱼ ~ Normal(θⱼ, σ) for student i in school j

**Priors**:

- Population mean: μ ~ Normal(75, 15) [reasonable test score range]
- Between-school std: τ ~ Exponential(0.1) [allows substantial variation]
- Within-school std: σ ~ Exponential(0.1) [student-level noise]

**Key Insight**: Schools with small sample sizes will shrink toward the population mean more than schools with large samples.

## Step 1: Generate Hierarchical Data

Let's create realistic school test score data:

```rust
{{#include ../../../examples/hierarchical_models_01_data.rs}}
```

## Step 2: No Pooling vs Complete Pooling Models

Let's start by comparing naive approaches to see why hierarchical modeling is needed:

```rust
{{#include ../../../examples/hierarchical_models_02_pooling.rs}}
```

## Step 3: Hierarchical Model Implementation

Now let's build the full hierarchical model that balances between these extremes:

```rust
{{#include ../../../examples/hierarchical_models_03_implementation.rs}}
```

## Step 4: Model Diagnostics and Validation

Let's add comprehensive diagnostics for hierarchical models:

```rust
{{#include ../../../examples/hierarchical_models_04_diagnostics.rs}}
```

## Step 5: Predictions for New Schools

Finally, let's use our hierarchical model to make predictions for new schools:

```rust
{{#include ../../../examples/hierarchical_models_05_predictions.rs}}
```

## Key Concepts Review

### 1. Hierarchical Model Structure

- **Multiple levels**: Population → Groups → Individuals
- **Partial pooling**: Optimal balance between complete and no pooling
- **Exchangeability**: Groups are similar but not identical
- **Shrinkage**: Automatic regularization based on data quality

### 2. Bayesian Advantages

- **Uncertainty propagation**: From hyperparameters to group estimates
- **Automatic complexity control**: No manual tuning of shrinkage
- **Natural handling of unbalanced data**: Different group sizes
- **Principled inference**: Full posterior distributions at all levels

### 3. Practical Applications

- **Education**: School effects, teacher effectiveness
- **Medicine**: Hospital quality, treatment effects by clinic
- **Marketing**: Customer segments, regional preferences
- **A/B Testing**: Treatment effects across user segments

### 4. Model Diagnostics

- **Convergence**: Split-chain R-hat for population parameters
- **Model fit**: Posterior predictive checking of group-level variance
- **Outlier detection**: Schools far from population distribution
- **ICC**: Quantifies clustering and model necessity

## Exercise: Extend the Analysis

Try these extensions to deepen your understanding:

1. **Three-level hierarchy**: Students → Teachers → Schools → District
2. **Regression coefficients**: School-specific slopes for covariates
3. **Non-normal data**: Hierarchical logistic regression for test pass rates
4. **Time series**: Longitudinal hierarchical models for school improvement

## Next Steps

Congratulations! You've mastered the most sophisticated modeling approach in this tutorial series. You now understand:

- **Complete PPL workflow**: From simple models to complex hierarchies
- **Bayesian inference**: Principled uncertainty quantification
- **Model comparison**: Information criteria and cross-validation
- **Real-world applications**: Practical probabilistic programming

### Continue Your Journey

1. **Advanced topics**: Explore non-parametric Bayesian methods
2. **Computational methods**: Learn about advanced MCMC techniques
3. **Model checking**: Develop sophisticated diagnostic workflows
4. **Production deployment**: Scale Bayesian models to real applications

### Fugue Mastery Checklist

- ✅ **Basic models**: Coin flips, simple inference
- ✅ **Regression**: Linear, polynomial, multiple predictors
- ✅ **Clustering**: Mixture models, model selection
- ✅ **Hierarchical**: Multi-level, partial pooling, shrinkage
- ✅ **Diagnostics**: Convergence, model fit, outlier detection
- ✅ **Predictions**: New observations, uncertainty quantification

You're now equipped to tackle sophisticated probabilistic modeling challenges with Fugue!

---

**Ready to build production systems?** → **[Custom Handlers Guide](../how-to/custom-handlers.md)**
