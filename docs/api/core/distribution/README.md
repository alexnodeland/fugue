# Distributions

Type-safe probability distributions with natural return types. This module provides a unified, type-safe interface for probability distributions used in Fugue models. All distributions implement the `Distribution<T>` trait, which provides sampling and log-probability density computation. The trait is generic over the sample type `T`, enabling natural return types for each distribution.

## Key Innovation: Type Safety

Unlike traditional probabilistic programming libraries that force all distributions to return `f64`, Fugue's distributions return their natural types:

- **Continuous distributions** → `f64` (as expected)
- **Bernoulli** → `bool` (not 0.0/1.0!)
- **Poisson/Binomial** → `u64` (natural counting)
- **Categorical** → `usize` (safe array indexing)

## Available Distributions

### Continuous Distributions (return `f64`)

- [`Normal`]: Normal/Gaussian distribution
- [`LogNormal`]: Log-normal distribution
- [`Uniform`]: Uniform distribution over an interval
- [`Exponential`]: Exponential distribution
- [`Beta`]: Beta distribution on \[0,1\]
- [`Gamma`]: Gamma distribution

### Discrete Distributions (return natural types!)

- [`Bernoulli`]: Bernoulli distribution → **`bool`**
- [`Binomial`]: Binomial distribution → **`u64`**
- [`Categorical`]: Categorical distribution → **`usize`**
- [`Poisson`]: Poisson distribution → **`u64`**

All distributions can be used both within the Model system (with `sample()` and `observe()`) and for direct statistical computation outside the Model system by calling `sample()` and `log_prob()` directly on distribution instances.

## Usage Examples

### Type-Safe Sampling

```rust
use fugue::*;
// Continuous distribution returns f64
let normal_model: Model<f64> = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
// Bernoulli returns bool - no more awkward comparisons!
let coin_model: Model<bool> = sample(addr!("coin"), Bernoulli::new(0.5).unwrap());
let decision = coin_model.bind(|heads| {
    if heads {
        pure("Take action!".to_string())
    } else {
        pure("Wait...".to_string())
    }
});
// Poisson returns u64 - perfect for counting!
let count_model: Model<u64> = sample(addr!("events"), Poisson::new(3.0).unwrap());
let analysis = count_model.bind(|count| {
    let status = match count {
        0 => "No events",
        1 => "Single event",
        n if n > 10 => "Many events!",
        n => &format!("{} events", n),
    };
    pure(status.to_string())
});
// Categorical returns usize - safe array indexing!
let choice_model: Model<usize> = sample(addr!("color"),
    Categorical::new(vec![0.5, 0.3, 0.2]).unwrap());
let colors = vec!["red", "green", "blue"];
let result = choice_model.bind(move |color_idx| {
    let chosen_color = colors[color_idx]; // Direct indexing - no casting!
    pure(chosen_color.to_string())
});
```

### Type-Safe Observations

```rust
use fugue::*;
// Observe with natural types
let model = observe(addr!("coin_result"), Bernoulli::new(0.6).unwrap(), true)      // bool
    .bind(|_| observe(addr!("count"), Poisson::new(4.0).unwrap(), 7u64))      // u64
    .bind(|_| observe(addr!("choice"), Categorical::new(
        vec![0.3, 0.5, 0.2]
    ).unwrap(), 1usize))  // usize
    .bind(|_| observe(addr!("temp"), Normal::new(20.0, 2.0).unwrap(), 18.5)); // f64
```
