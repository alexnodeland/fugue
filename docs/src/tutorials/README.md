# Tutorials

Complete, hands-on tutorials for mastering probabilistic programming with Fugue. Each tutorial builds on previous concepts while introducing new techniques.

## Learning Path

### **Beginner Level**

Start here if you're new to Bayesian inference or probabilistic programming.

#### 1. [Basic Inference](basic-inference.md) ⭐ **Start Here**
**Time: 25 minutes** | **Examples: `gaussian_mean`, `conjugate_beta_binomial`**

Learn fundamental Bayesian concepts with simple, well-understood models:
- Prior specification and interpretation
- Conjugate priors and analytical solutions  
- MCMC vs analytical comparison
- Parameter estimation with uncertainty

#### 2. [Bayesian Coin Flip](bayesian-coin-flip.md)
**Time: 30 minutes** | **Examples: `coin_flip_01_basic` through `coin_flip_05_advanced`**

Your first complete Bayesian analysis:
- Formulating a Bayesian model
- Prior, likelihood, and posterior relationships
- MCMC inference and trace analysis
- Answering probabilistic questions

### **Intermediate Level**

Build sophisticated models with multiple parameters and complex relationships.

#### 3. [Type Safety Features](type-safety-features.md)  
**Time: 35 minutes** | **Examples: `fully_type_safe`, `type_safe_improvements`**

Master Fugue's advanced type system:
- Natural return types (`bool`, `u64`, `usize`)
- Safe trace accessors and handlers
- Type-specific diagnostics  
- Production-ready error handling

#### 4. [Linear Regression](linear-regression.md)
**Time: 45 minutes** | **Examples: `linear_regression_01_data` through `linear_regression_05_polynomial`**  

Bayesian regression from basic to advanced:
- Continuous parameter relationships
- Uncertainty quantification in predictions
- Model diagnostics and validation
- Polynomial and multiple regression

#### 5. [Simple Mixtures](simple-mixtures.md)
**Time: 20 minutes** | **Examples: `gaussian_mixture`, `simple_mixture`**

Introduction to mixture models and latent variables:
- Multi-modal data and component identification
- Probabilistic vs hard clustering
- When to use mixture models
- Component analysis and interpretation

#### 6. [Mixture Models](mixture-models.md)
**Time: 50 minutes** | **Examples: `mixture_models_01_data` through `mixture_models_05_clustering`**

Comprehensive treatment of mixture modeling:
- Model selection for unknown number of components
- Information criteria (AIC, BIC)
- Customer segmentation and clustering
- Production classification pipelines

### **Advanced Level**

Master sophisticated modeling techniques and production-ready inference.

#### 7. [Hierarchical Models](hierarchical-models.md)
**Time: 60 minutes** | **Examples: `hierarchical_models_01_data` through `hierarchical_models_05_predictions`**

Multi-level modeling with shared parameters:
- Partial pooling and shrinkage effects
- Group-level and population-level inference  
- Handling unbalanced data
- School performance analysis case study

#### 8. [Trace Manipulation](trace-manipulation.md)
**Time: 40 minutes** | **Examples: `trace_manipulation`**

Advanced runtime operations and debugging:
- Deterministic replay and counterfactual analysis
- Manual trace modification
- MCMC algorithm internals
- Model debugging techniques

#### 9. [Advanced Inference](advanced-inference.md)
**Time: 50 minutes** | **Examples: `improved_gaussian_mean`, `exponential_hazard`**

Production-ready inference techniques:
- Comprehensive convergence diagnostics
- Multi-chain analysis and validation
- Specialized distributions and domains
- Numerical stability and robust pipelines

## Tutorial Features

Each tutorial includes:

- **🎯 Clear learning objectives** - Know what you'll master
- **💻 Runnable examples** - Every code block works with `cargo run --example`
- **📊 Real applications** - Practical problems, not toy examples
- **🔍 Key concepts** - Deep understanding, not just recipes
- **🎓 Exercises** - Extend your learning
- **➡️ Next steps** - Clear progression path

## Quick Reference

| Tutorial | Level | Time | Key Concepts | Examples |
|----------|-------|------|--------------|----------|
| [Basic Inference](basic-inference.md) | Beginner | 25m | Priors, conjugacy, MCMC basics | 2 |
| [Bayesian Coin Flip](bayesian-coin-flip.md) | Beginner | 30m | Complete analysis workflow | 5 |
| [Type Safety Features](type-safety-features.md) | Intermediate | 35m | Type system, safe operations | 2 |
| [Linear Regression](linear-regression.md) | Intermediate | 45m | Continuous relationships, diagnostics | 5 |
| [Simple Mixtures](simple-mixtures.md) | Intermediate | 20m | Latent variables, components | 2 |
| [Mixture Models](mixture-models.md) | Intermediate | 50m | Model selection, clustering | 5 |
| [Hierarchical Models](hierarchical-models.md) | Advanced | 60m | Multi-level, partial pooling | 5 |
| [Trace Manipulation](trace-manipulation.md) | Advanced | 40m | Runtime operations, debugging | 1 |
| [Advanced Inference](advanced-inference.md) | Advanced | 50m | Production techniques, diagnostics | 2 |

## Getting Help

- **Examples not working?** Check that you have the latest Fugue version
- **Concepts unclear?** Start with [Basic Inference](basic-inference.md) fundamentals
- **Ready for more?** Check the [How-To Guides](../how-to/README.md) for specific techniques
- **Production deployment?** See [Advanced Inference](advanced-inference.md) for robust pipelines

## Example Command Reference

All tutorial examples can be run directly:

```bash
# Basic concepts
cargo run --example gaussian_mean
cargo run --example conjugate_beta_binomial

# Complete workflows  
cargo run --example coin_flip_01_basic
cargo run --example linear_regression_02_model

# Advanced techniques
cargo run --example mixture_models_04_model_selection
cargo run --example hierarchical_models_03_implementation
cargo run --example trace_manipulation
```

**Happy learning!** Start with [Basic Inference](basic-inference.md) and work your way up to become a Fugue expert.
