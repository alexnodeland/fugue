# Model

**Core model representation: a tiny monadic PPL in direct style.**

This module provides the core `Model<A>` type and operations for building probabilistic programs. `Model<A>` represents a program that, when interpreted by a runtime handler, yields a value of type `A`. The monadic interface (`pure`, `bind`, `map`) enables compositional probabilistic programs.

## Model Types

A `Model<A>` can be one of four types:

- **Pure**: Contains a deterministic value
- **SampleF**: Samples from a probability distribution
- **ObserveF**: Conditions on observed data
- **FactorF**: Adds log-weight factors for soft constraints

## Basic Operations

### Creating Models

```rust
use fugue::*;
// Pure deterministic value
let model = pure(42.0);
// Sample from a distribution
let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
// Observe/condition on data
let model = observe(addr!("y"), Normal::new(0.0, 1.0).unwrap(), 2.5);
// Add log-weight factor
let model = factor(0.5); // log(exp(0.5)) weight
```

### Composing Models

Models can be composed using monadic operations:

```rust
use fugue::*;
let composed_model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
    .bind(|x| {
        sample(addr!("y"), Normal::new(x, 0.5).unwrap())
            .map(move |y| x + y)
    });
```

### Working with Collections

```rust
use fugue::*;
// Create multiple independent samples
let models = vec![
    sample(addr!("x", 0), Normal::new(0.0, 1.0).unwrap()),
    sample(addr!("x", 1), Normal::new(0.0, 1.0).unwrap()),
];
let combined = sequence_vec(models); // Model<Vec<f64>>
// Apply a function to each item
let data = vec![1.0, 2.0, 3.0];
let model = traverse_vec(data, |x| {
    sample(addr!("noise", x as usize), Normal::new(0.0, 0.1).unwrap())
        .map(move |noise| x + noise)
});
```
