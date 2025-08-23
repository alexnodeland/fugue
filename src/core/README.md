# Core Module

## Overview

The core module provides the fundamental building blocks for probabilistic programming in Fugue. It defines the core abstractions that enable type-safe, composable probabilistic models through monadic composition, unique addressing, and a rich set of probability distributions.

## Quick Start

```rust
use fugue::*;

// Create a simple Bayesian model
let model = prob! {
    let mu <- sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap());
    let is_outlier <- sample(addr!("outlier"), Bernoulli::new(0.1).unwrap());  // Returns bool!
    let sigma = if is_outlier { 5.0 } else { 1.0 };  // Natural boolean usage
    observe(addr!("y"), Normal::new(mu, sigma).unwrap(), 2.5);
    pure(mu)
};
```

## Components

### `address.rs` - Site Addressing

- `Address`: Unique identifiers for random choice sites
- `addr!` macro: Creates addresses from names and optional indices
- `scoped_addr!` macro: Creates hierarchical addresses with scoping

```rust
let a1 = addr!("mu");           // Address("mu")
let a2 = addr!("x", 5);         // Address("x#5")
let a3 = scoped_addr!("layer1", "weight"); // Address("layer1::weight")
```

### `distribution.rs` - Type-Safe Probability Distributions

- `Distribution<T>` trait: Generic interface for distributions over any type `T`
- **Type-safe sampling**: Each distribution returns its natural type
  - `Bernoulli` → `bool` (no more f64 comparisons!)
  - `Poisson`, `Binomial` → `u64` (natural counting)
  - `Categorical` → `usize` (safe array indexing)
  - `Normal`, `Beta`, etc. → `f64` (continuous values)
- Built-in distributions: Normal, Uniform, LogNormal, Exponential, Beta, Gamma, Bernoulli, Categorical, Binomial, Poisson
- All distributions support sampling and log-density evaluation

```rust
// Continuous distribution
let normal = Normal{mu: 0.0, sigma: 1.0};
let x: f64 = normal.sample(&mut rng);
let logp = normal.log_prob(&x);

// Discrete distributions now type-safe!
let coin = Bernoulli{p: 0.5};
let flip: bool = coin.sample(&mut rng);  // Returns bool directly!
let prob = coin.log_prob(&flip);

let counter = Poisson{lambda: 3.0};
let count: u64 = counter.sample(&mut rng);  // Returns u64 directly!

let choice = Categorical{probs: vec![0.3, 0.5, 0.2]};
let idx: usize = choice.sample(&mut rng);  // Returns usize for safe indexing!
```

### `model.rs` - Monadic Model Representation

**Key Types/Functions:**

- `Model<A>`: The core probabilistic program type
- `pure`, `bind`, `map`, `and_then`: Monadic operations
- `sample`, `observe`, `factor`: Primitive operations
- `zip`, `sequence_vec`, `traverse_vec`, `guard`: Combinators

**Example:**

```rust
let model = prob! {
    let mu <- sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap());  // Returns f64
    let is_outlier <- sample(addr!("outlier"), Bernoulli::new(0.1).unwrap());  // Returns bool!
    let sigma = if is_outlier { 5.0 } else { 1.0 };  // Natural boolean usage
    observe(addr!("y"), Normal::new(mu, sigma).unwrap(), 2.5);
    pure(mu)
};
```

### `numerical.rs` - Numerical Stability

**Key Functions:**

- `log_sum_exp`: Numerically stable log-sum-exp computation
- `safe_ln`: Safe logarithm with validation
- `log1p_exp`: Stable computation of log(1 + exp(x))
- `normalize_log_probs`: Normalize log probabilities

**Example:**

```rust
let log_probs = vec![-1.0, -2.0, -0.5];
let normalized = normalize_log_probs(&log_probs);
```

## Common Patterns

### Sequential Model Building

Build complex models step-by-step using monadic composition.

```rust
let hierarchical_model = prob! {
    let global_mu <- sample(addr!("global_mu"), Normal::new(0.0, 1.0).unwrap());
    let local_effects <- plate!(i in 0..10 => {
        sample(addr!("local", i), Normal::new(global_mu, 0.1).unwrap())
    });
    pure((global_mu, local_effects))
};
```

### Conditional Logic with Type-Safe Distributions

Leverage type safety for cleaner conditional models.

```rust
let mixture_model = prob! {
    let component <- sample(addr!("component"), Categorical::new(vec![0.3, 0.7]).unwrap()); // Returns usize!
    let mu = match component {
        0 => -2.0,
        1 => 2.0,
        _ => 0.0,
    };
    let x <- sample(addr!("x"), Normal::new(mu, 1.0).unwrap());
    observe(addr!("y"), Normal::new(x, 0.1).unwrap(), observed_value);
    pure((component, x))
};
```

### Macro-Driven Development

**`prob!` - Do-notation Style Composition:**

```rust
let model = prob! {
    let x <- sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
    let y = x * 2.0;  // Regular let binding
    observe(addr!("obs"), Normal::new(y, 0.1).unwrap(), 1.5);
    pure(x)
};
```

**`plate!` - Vectorized Operations:**

```rust
let model = plate!(i in 0..10 => {
    sample(addr!("x", i), Normal::new(0.0, 1.0).unwrap())
});
```

## Performance Considerations

- **Memory**: Models are zero-cost abstractions that compile to efficient code
- **Computation**: Distributions use numerically stable algorithms
- **Best Practices**:
  - Use `plate!` for vectorized operations instead of manual loops
  - Prefer specific imports over wildcard imports in performance-critical code
  - Use the appropriate distribution type for your data (e.g., `u64` for counts)

## Integration

**Related Modules:**

- [`inference`](../inference/README.md): Use core models with MCMC, SMC, VI, and ABC algorithms
- [`runtime`](../runtime/README.md): Execute models with different handlers and trace management
- [`error`](../error.rs): Comprehensive error handling for distribution validation

**See Also:**

- Main documentation: [API docs](https://docs.rs/fugue)
- Examples: [`examples/gaussian_mean.rs`](../../examples/gaussian_mean.rs), [`examples/fully_type_safe.rs`](../../examples/fully_type_safe.rs)

## Extension Points

How to extend the core module:

1. **Custom Distributions**: Implement the `Distribution<T>` trait for new probability distributions
2. **Model Combinators**: Add new higher-order functions that operate on `Model<A>`
3. **Addressing Schemes**: Extend the address system for complex hierarchical models
4. **Numerical Functions**: Add domain-specific numerical stability functions

## Design Principles

- **Monadic**: Models compose via bind/map following monad laws
- **Pure**: No side effects; interpretation happens at runtime
- **Typed**: Strong typing prevents many modeling errors
- **Addressable**: Every random choice has a unique, stable address
- **Extensible**: Easy to add new distributions and combinators
