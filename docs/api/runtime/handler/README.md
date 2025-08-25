# Handlers

Generic handler interface and model execution engine.

This module provides the core abstraction for interpreting probabilistic models.

The `Handler` trait defines how to process the three fundamental effects in probabilistic programming: sampling, observation, and factoring. Different handler implementations enable different execution modes (prior sampling, conditioning, scoring, etc.).

## Handler Pattern

Handlers implement the algebraic effects pattern with **full type safety**. Each effect is handled by type-specific methods that match the natural return types of distributions:

- `on_sample_f64` for continuous distributions (Normal, Beta, etc.)
- `on_sample_bool` for Bernoulli (returns bool directly!)
- `on_sample_u64` for count distributions (Poisson, Binomial)
- `on_sample_usize` for categorical distributions (safe indexing!)

This design enables:

- **Type Safety**: Handlers work with natural types, not just f64
- **Modularity**: Different handlers for different purposes
- **Composability**: Handlers can be combined and extended
- **Testability**: Effects can be mocked and controlled
- **Performance**: Zero overhead from unnecessary type conversions

## Execution Model

The `run` function acts as the interpreter, walking through a `Model` and dispatching effects to the handler. It returns both the model's final value and the accumulated execution trace.

## Examples

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
// Run type-safe models with prior sampling
let normal_model: Model<f64> = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
let coin_model: Model<bool> = sample(addr!("coin"), Bernoulli::new(0.5).unwrap());
let count_model: Model<u64> = sample(addr!("events"), Poisson::new(3.0).unwrap());
let mut rng = StdRng::seed_from_u64(42);
// Handler automatically dispatches to correct type-specific method
let (value, trace) = runtime::handler::run(
    PriorHandler {
        rng: &mut rng,
        trace: Trace::default(),
    },
    coin_model, // Handler calls on_sample_bool, returns bool
);
println!("Coin flip: {}, log-weight: {}", value, trace.total_log_weight());
```
