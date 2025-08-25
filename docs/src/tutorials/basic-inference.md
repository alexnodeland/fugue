# Basic Inference Tutorial

**Level: Beginner** | **Time: 25 minutes**

Welcome to basic Bayesian inference with Fugue! This tutorial covers fundamental concepts using simple, well-understood models. You'll learn core principles that apply to all probabilistic programming.

## Learning Objectives

By the end of this tutorial, you'll understand:

- Basic Bayesian inference workflow
- Prior specification and interpretation
- Conjugate priors and analytical solutions
- MCMC vs analytical comparison
- Parameter estimation with uncertainty

## Concepts Covered

We'll explore two foundational examples:
1. **Gaussian mean estimation** - The "hello world" of Bayesian inference
2. **Beta-Binomial analysis** - Conjugate priors in action

## Part 1: Gaussian Mean Estimation

### The Problem

You measure some quantity (temperature, response time, etc.) and get a single observation. What can you infer about the true mean?

**Try it**: Run with `cargo run --example gaussian_mean`

```rust
{{#include ../../../examples/gaussian_mean.rs}}
```

### Key Concepts

- **Prior**: `Normal(0, 5)` - our initial belief about the mean
- **Likelihood**: `Normal(μ, 1)` - how observations relate to the parameter  
- **Posterior**: Updated belief after seeing data

**Mathematical insight**: With a Normal prior and Normal likelihood, the posterior is also Normal (conjugate relationship).

## Part 2: Beta-Binomial Conjugate Analysis

### The Problem  

Classic coin flip scenario: estimate probability of success from binary outcomes.

**Try it**: Run with `cargo run --example conjugate_beta_binomial`

```rust
{{#include ../../../examples/conjugate_beta_binomial.rs}}
```

### Key Concepts

- **Beta prior**: `Beta(2, 2)` - slightly favors balanced probabilities
- **Binomial likelihood**: Number of successes out of n trials
- **Beta posterior**: `Beta(2+k, 2+n-k)` - exact analytical update

**Why conjugacy matters**:
- Closed-form posterior updates
- No MCMC needed for simple cases
- Clear mathematical interpretation
- Foundation for more complex models

### Understanding the Results

The example shows:
```
Data: 6/10 successes (60.0%)
Estimated success probability: 0.625
Posterior sample suggests 62.5% success rate
```

The Bayesian estimate (62.5%) is pulled toward the prior mean (50%) compared to the raw percentage (60%). This is **shrinkage** - a key Bayesian principle.

## Comparing Approaches

Let's understand when to use different inference methods:

| Method | When to Use | Advantages | Disadvantages |
|--------|-------------|------------|---------------|
| **Analytical** | Simple, conjugate models | Exact, fast | Limited cases |
| **MCMC** | Complex models | General purpose | Computational cost |
| **Prior sampling** | Model testing | Quick prototyping | Not posterior samples |

## Model Building Principles

### 1. Start Simple
```rust
// Begin with basic models
let prior = Normal::new(0.0, 1.0).unwrap();
observe(addr!("y"), Normal::new(mu, 1.0).unwrap(), observation);
```

### 2. Validate Against Known Solutions
```rust
// For conjugate cases, compare MCMC vs analytical
let analytical_posterior_mean = /* calculate */;
let mcmc_posterior_mean = /* estimate from samples */;
assert!((analytical_posterior_mean - mcmc_posterior_mean).abs() < 0.01);
```

### 3. Interpret Results in Context
```rust
println!("Estimated parameter: {:.3} ± {:.3}", mean, std);
println!("95% credible interval: [{:.3}, {:.3}]", ci_lower, ci_upper);
```

## Common Patterns

### Parameter Estimation Template
```rust
fn parameter_estimation_model(data: Vec<f64>) -> Model<f64> {
    prob! {
        // 1. Specify prior
        let parameter <- sample(addr!("param"), /* prior distribution */);
        
        // 2. Define likelihood for each observation
        for (i, &observation) in data.iter().enumerate() {
            observe(addr!("obs", i), /* likelihood */, observation);
        }
        
        // 3. Return parameter of interest
        pure(parameter)
    }
}
```

### MCMC Analysis Template
```rust
fn analyze_posterior(samples: &[(f64, Trace)]) {
    let param_samples: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("param")))
        .collect();
    
    let mean = param_samples.iter().sum::<f64>() / param_samples.len() as f64;
    let variance = param_samples.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / param_samples.len() as f64;
    
    println!("Posterior mean: {:.3}", mean);
    println!("Posterior std: {:.3}", variance.sqrt());
}
```

## Exercises

Try these variations to deepen your understanding:

### Exercise 1: Different Priors
Modify the Gaussian mean example:
```rust
// Try different priors
let informative_prior = Normal::new(2.0, 0.5).unwrap();  // Strong belief
let vague_prior = Normal::new(0.0, 100.0).unwrap();     // Weak belief
```

### Exercise 2: Multiple Observations
Extend to multiple data points:
```rust
let observations = vec![2.1, 1.8, 2.5, 2.0, 2.3];
for (i, &obs) in observations.iter().enumerate() {
    observe(addr!("y", i), Normal::new(mu, 1.0).unwrap(), obs);
}
```

### Exercise 3: Different Success Rates
Try the Beta-Binomial model with different scenarios:
- Rare events: 1 success in 100 trials
- Common events: 90 successes in 100 trials  
- No data: 0 trials (prior only)

## Understanding Uncertainty

Bayesian inference naturally quantifies uncertainty:

```rust
// Point estimate (classical)
let point_estimate = successes as f64 / trials as f64;

// Bayesian estimate with uncertainty
let (estimate, trace) = run_mcmc(model);
let posterior_samples = extract_parameter_samples(trace);
let credible_interval = compute_credible_interval(&posterior_samples, 0.95);

println!("Classical: {:.2}", point_estimate);
println!("Bayesian: {:.2} ± {:.2}", estimate, posterior_std);
println!("95% CI: [{:.2}, {:.2}]", credible_interval.0, credible_interval.1);
```

The Bayesian approach tells you not just the estimate, but **how certain you should be** about it.

## Next Steps

Now that you understand basic inference:

1. **[Bayesian Coin Flip Tutorial](bayesian-coin-flip.md)** - Apply these concepts to a complete analysis
2. **[Type Safety Features](type-safety-features.md)** - Learn about Fugue's type system
3. **[Linear Regression Tutorial](linear-regression.md)** - Scale to continuous relationships

## Key Takeaways

- **Bayesian inference** combines prior knowledge with data
- **Conjugate priors** allow exact analytical solutions
- **MCMC** handles cases where analytical solutions aren't available
- **Uncertainty quantification** is automatic and principled
- **Start simple** and validate against known solutions

Congratulations! You now understand the fundamental building blocks of Bayesian inference.

---

**Ready for a complete analysis?** → **[Bayesian Coin Flip Tutorial](bayesian-coin-flip.md)**
