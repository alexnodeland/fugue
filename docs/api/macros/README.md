# `macros` module

## Overview

The macros module provides convenient syntactic sugar for writing probabilistic programs in Fugue. These macros make it easier to express complex probabilistic computations using familiar programming constructs.

## Available Macros

### `prob!` - Probabilistic Programming Notation

The `prob!` macro provides Haskell-style do-notation for probabilistic programming, making it easier to chain probabilistic computations.

**Syntax:**

- `let var <- expr` - Sample from a probabilistic computation
- `let var = expr` - Regular variable assignment
- `expr` - Final return value

**Example:**

```rust
use fugue::*;

let model = prob! {
    let x <- sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
    let y <- sample(addr!("y"), Normal::new(x, 1.0).unwrap());
    let z = x + y;  // Regular assignment
    pure(z)
};
```

### `plate!` - Vectorized Operations

The `plate!` macro implements plate notation for replicating probabilistic computations over ranges or collections.

**Syntax:**

- `plate!(var in range => body)` - Execute `body` for each element in `range`

**Example:**

```rust
use fugue::*;

// Sample 10 independent normal variables
let model = plate!(i in 0..10 => {
    sample(addr!("x", i), Normal::new(0.0, 1.0).unwrap())
});

// With observations - using move to capture i
let model = plate!(i in 0..3 => {
    sample(addr!("mu", i), Normal::new(0.0, 1.0).unwrap())
        .bind(move |mu| observe(addr!("obs", i), Normal::new(mu, 0.5).unwrap(), 1.0 + i as f64))
});
```

### `scoped_addr!` - Hierarchical Addresses

The `scoped_addr!` macro creates hierarchical addresses for organizing model parameters.

**Syntax:**

- `scoped_addr!(scope, name)` - Creates "scope::name"
- `scoped_addr!(scope, name, format, args...)` - Creates "scope::name#formatted"

**Example:**

```rust
use fugue::*;

// Simple scoped address
let addr1 = scoped_addr!("layer1", "weight");  // "layer1::weight"

// With indices
let addr2 = scoped_addr!("layer1", "weight", "{}", 0);  // "layer1::weight#0"
let addr3 = scoped_addr!("layer1", "bias", "{}_{}", 2, 3);  // "layer1::bias#2_3"
```

## Common Patterns

### Hierarchical Models with Plate Notation

```rust
use fugue::*;

// Simple hierarchical model example
let n_groups = 3;
let model = prob! {
    // Global hyperparameters
    let global_mu <- sample(addr!("global_mu"), Normal::new(0.0, 10.0).unwrap());

    // Group-level parameters
    let group_means <- plate!(g in 0..n_groups => {
        sample(scoped_addr!("group", "mu", "{}", g),
               Normal::new(global_mu, 1.0).unwrap())
    });

    pure(group_means)
};
```

### Sequential Models

```rust
use fugue::*;

// Simple sequential model example
let model = prob! {
    // Sample parameters
    let states <- plate!(t in 0..3 => {
        sample(addr!("x", t), Normal::new(0.0, 1.0).unwrap())
            .bind(move |x_t| {
                observe(addr!("y", t), Normal::new(x_t, 0.5).unwrap(), 1.0 + t as f64)
                    .map(move |_| x_t)
            })
    });

    pure(states)
};
```

## Best Practices

1. **Use descriptive addresses**: Make your model structure clear through well-named addresses
2. **Scope your addresses**: Use `scoped_addr!` for complex hierarchical models
3. **Leverage plate notation**: Use `plate!` for vectorized operations rather than manual loops
4. **Mix syntax styles**: Combine `prob!` with regular function calls for complex models

## Integration

**Related Modules:**

- [`core`](../core/README.md): Core model and distribution types used in macros
- [`runtime`](../runtime/README.md): Execution of models created with macros

**See Also:**

- Main documentation: [API docs](https://docs.rs/fugue)
