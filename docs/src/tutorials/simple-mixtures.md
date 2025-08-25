# Simple Mixtures Introduction

**Level: Beginner** | **Time: 20 minutes**

Get started with mixture models using simple examples! This tutorial introduces the concepts before diving into the comprehensive [Mixture Models Tutorial](mixture-models.md).

## Learning Objectives

By the end of this tutorial, you'll understand:

- What mixture models represent
- How to implement basic 2-component mixtures
- Component assignment and interpretation
- When to use mixture models vs single distributions

## What Are Mixture Models?

A **mixture model** represents data that comes from multiple subpopulations (components):

```
Overall Data = Component 1 × Weight 1 + Component 2 × Weight 2 + ...
```

**Examples**:
- Customer segments (budget, premium)
- Medical conditions (healthy, sick)  
- Market regimes (bull, bear markets)

## Part 1: Simple Gaussian Mixture

**Try it**: Run with `cargo run --example gaussian_mixture`

```rust
{{#include ../../../examples/gaussian_mixture.rs}}
```

### Understanding the Code

```rust
let choose_first = u < 0.5;  // 50/50 mixture
let mu = if choose_first { -2.0 } else { 2.0 };  // Two components at -2 and +2
```

This creates two Normal distributions:
- Component 1: Normal(-2, 1) with 50% weight
- Component 2: Normal(+2, 1) with 50% weight  

The model samples which component generates each observation, then generates the observation from that component.

### Key Concepts

- **Latent variable**: Which component generated each observation (hidden)
- **Mixture weights**: Probability of each component (50/50 here)
- **Component parameters**: Each component has its own parameters

## Part 2: Type-Safe Mixture Model

**Try it**: Run with `cargo run --example simple_mixture`

```rust
{{#include ../../../examples/simple_mixture.rs}}
```

### Type Safety in Action

Notice how this example uses natural types:

```rust
let comp <- sample(addr!("component"), Bernoulli::new(weight).unwrap());
let mu = if comp { mu2 } else { mu1 };  // comp is naturally bool!
```

**Benefits**:
- `comp` is a `bool`, not a `f64` that needs conversion
- More readable and less error-prone
- Compiler catches type mismatches

### Understanding the Results

The output shows:
```
Component 1 mean: -1.845
Component 2 mean: 2.123  
Observation: 0.0
Total log weight: -2.456
```

The model inferred two components with means around -2 and +2, which matches our prior expectations.

## When to Use Mixture Models

### ✅ Good Use Cases

**Heterogeneous populations**:
```rust
// Customer types with different spending patterns
let customer_segments = vec![
    ("Budget", 25.0, 5.0),      // mean=$25, std=$5
    ("Premium", 200.0, 50.0),   // mean=$200, std=$50
];
```

**Multi-modal data**:
```rust
// Reaction times: fast automatic vs slow deliberate responses
let reaction_modes = vec![
    ("Automatic", 0.3, 0.1),   // 300ms ± 100ms
    ("Deliberate", 1.2, 0.3),  // 1200ms ± 300ms  
];
```

**Regime switching**:
```rust
// Market volatility: calm vs turbulent periods
let market_regimes = vec![
    ("Calm", 0.05, 0.02),      // 5% ± 2% daily returns
    ("Turbulent", 0.0, 0.08),  // 0% ± 8% daily returns
];
```

### ❌ When NOT to Use

**Single homogeneous population**:
```rust
// Just use Normal distribution directly
let heights = Normal::new(170.0, 10.0).unwrap();  // Adult heights
```

**Continuous gradation** (no distinct groups):
```rust
// Use regression instead
let income_by_age = LinearRegression::new(age_data, income_data);
```

## Building Intuition

### Component Identification

Mixture models automatically **separate** mixed populations:

```
Input:  [1.8, -2.1, 2.3, -1.9, 2.1, -2.0, 1.9, -2.2]
         Mixed data from two components

Output: Component 1: [-2.1, -1.9, -2.0, -2.2]  (mean ≈ -2.0)
        Component 2: [1.8, 2.3, 2.1, 1.9]      (mean ≈ +2.0)
         Separated components
```

### Uncertainty Quantification

Unlike clustering algorithms (K-means), mixture models provide **probabilistic assignments**:

```rust
// Hard assignment (K-means style)
let cluster = if value > 0.0 { 1 } else { 0 };

// Soft assignment (Bayesian mixture)
let prob_component_1 = weight * normal1.likelihood(value);
let prob_component_2 = (1.0 - weight) * normal2.likelihood(value);
let assignment_probability = prob_component_1 / (prob_component_1 + prob_component_2);
```

The Bayesian approach tells you **how confident** the assignment is.

## Common Patterns

### 2-Component Template

```rust
fn two_component_mixture(data: Vec<f64>) -> Model<(f64, f64, f64)> {
    prob! {
        // Mixture weight
        let weight <- sample(addr!("weight"), Beta::new(1.0, 1.0).unwrap());
        
        // Component parameters
        let mu1 <- sample(addr!("mu1"), Normal::new(0.0, 5.0).unwrap());
        let mu2 <- sample(addr!("mu2"), Normal::new(0.0, 5.0).unwrap());
        let sigma <- sample(addr!("sigma"), Exponential::new(1.0).unwrap());
        
        // Mixture likelihood
        for (i, &x) in data.iter().enumerate() {
            let component <- sample(addr!("z", i), Bernoulli::new(weight).unwrap());
            let chosen_mu = if component { mu2 } else { mu1 };
            observe(addr!("x", i), Normal::new(chosen_mu, sigma).unwrap(), x);
        }
        
        pure((weight, mu1, mu2))
    }
}
```

### Component Analysis

```rust
fn analyze_components(samples: &[(f64, f64, f64)]) {
    let (weights, mu1s, mu2s): (Vec<f64>, Vec<f64>, Vec<f64>) = 
        samples.iter().cloned().collect();
    
    let avg_weight = weights.iter().sum::<f64>() / weights.len() as f64;
    let avg_mu1 = mu1s.iter().sum::<f64>() / mu1s.len() as f64;
    let avg_mu2 = mu2s.iter().sum::<f64>() / mu2s.len() as f64;
    
    println!("Component 1: {:.1}% weight, mean={:.2}", (1.0 - avg_weight) * 100.0, avg_mu1);
    println!("Component 2: {:.1}% weight, mean={:.2}", avg_weight * 100.0, avg_mu2);
}
```

## Limitations of Simple Mixtures

These basic examples are great for learning, but have limitations:

- **Fixed number of components** (2 components hard-coded)
- **Simple component shapes** (just Normal distributions)
- **No model comparison** (can't choose optimal number of components)
- **Basic inference** (prior sampling, not proper MCMC)

For production use, see the comprehensive [Mixture Models Tutorial](mixture-models.md).

## Exercises

Try these variations:

### Exercise 1: Different Weights
```rust
// Instead of 50/50, try unbalanced mixtures
let choose_first = u < 0.8;  // 80% component 1, 20% component 2
```

### Exercise 2: Overlapping Components  
```rust  
// Components closer together
let mu = if choose_first { 0.0 } else { 1.0 };  // Harder to separate
```

### Exercise 3: Different Variances
```rust
// Components with different spreads
let sigma = if choose_first { 0.5 } else { 2.0 };
observe(addr!("y"), Normal::new(mu, sigma).unwrap(), obs);
```

## Interpreting Results

### Good Separation
```
Component 1 mean: -2.01 ± 0.1
Component 2 mean: +1.98 ± 0.1
Weight: 0.52 ± 0.05
```
**Interpretation**: Clear separation, well-identified components.

### Poor Separation  
```
Component 1 mean: -0.23 ± 1.2
Component 2 mean: +0.31 ± 1.5  
Weight: 0.51 ± 0.25
```
**Interpretation**: Components overlap heavily, uncertain identification.

## Next Steps

Ready for more sophisticated mixture modeling:

1. **[Mixture Models Tutorial](mixture-models.md)** - Comprehensive treatment with model selection
2. **[Type Safety Features](type-safety-features.md)** - Learn more about Fugue's type system
3. **[Hierarchical Models](hierarchical-models.md)** - Multi-level mixture models

## Key Takeaways

- **Mixture models** separate heterogeneous populations
- **Latent variables** represent component assignments  
- **Bayesian inference** provides probabilistic assignments
- **Type safety** makes models clearer and more reliable
- **Start simple** then move to comprehensive approaches

Simple mixtures are a great entry point to understanding probabilistic clustering and latent variable models!

---

**Ready for comprehensive mixture modeling?** → **[Mixture Models Tutorial](mixture-models.md)**
