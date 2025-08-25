# `sequence_vec` function

Execute a vector of models and collect their results into a single model of a vector. This function takes a collection of independent models and runs them all, collecting their results into a vector. This is useful for running multiple similar probabilistic computations.

## Arguments

- `models` - Vector of models to execute

## Returns

A `Model<Vec<A>>` containing all the results in order.

## Examples

```rust
use fugue::*;
// Create multiple independent samples
let models = vec![
    sample(addr!("x", 0), Normal::new(0.0, 1.0).unwrap()),
    sample(addr!("x", 1), Normal::new(1.0, 1.0).unwrap()),
    sample(addr!("x", 2), Normal::new(2.0, 1.0).unwrap()),
];
let all_samples = sequence_vec(models); // Model<Vec<f64>>
// Mix deterministic and probabilistic models
let mixed_models = vec![
    pure(1.0),
    sample(addr!("random"), Uniform::new(0.0, 1.0).unwrap()),
    pure(3.0),
];
let results = sequence_vec(mixed_models);
```
