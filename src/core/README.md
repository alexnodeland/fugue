# Core Module

The core module provides the fundamental building blocks for probabilistic programming:

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

- `Model<A>`: The core probabilistic program type
- Monadic operations: `pure`, `bind`, `map`, `and_then`
- Primitive operations: `sample`, `observe`, `factor`
- Combinators: `zip`, `sequence_vec`, `traverse_vec`, `guard`

```rust
let model = prob! {
    let mu <- sample(addr!("mu"), Normal{mu: 0.0, sigma: 1.0});  // Returns f64
    let is_outlier <- sample(addr!("outlier"), Bernoulli{p: 0.1});  // Returns bool!
    let sigma = if is_outlier { 5.0 } else { 1.0 };  // Natural boolean usage
    observe(addr!("y"), Normal{mu, sigma}, 2.5);
    pure(mu)
};
```

## Macros

### `prob!` - Do-notation Style Composition

Provides imperative-style syntax for monadic composition:

```rust
let model = prob! {
    let x <- sample(addr!("x"), Normal{mu: 0.0, sigma: 1.0});
    let y = x * 2.0;  // Regular let binding
    observe(addr!("obs"), Normal{mu: y, sigma: 0.1}, 1.5);
    pure(x)
};
```

### `plate!` - Vectorized Operations

Replicates models over ranges:

```rust
let model = plate!(i in 0..10 => {
    sample(addr!("x", i), Normal{mu: 0.0, sigma: 1.0})
});
```

## Design Principles

- **Monadic**: Models compose via bind/map following monad laws
- **Pure**: No side effects; interpretation happens at runtime
- **Typed**: Strong typing prevents many modeling errors
- **Addressable**: Every random choice has a unique, stable address
- **Extensible**: Easy to add new distributions and combinators
