# `handler` trait

Trait for handling probabilistic effects during model execution.

Handlers define the interpretation of the three fundamental effects in probabilistic programming. Different handler implementations enable different execution modes:

- Prior sampling (draw fresh random values)
- Replay (use values from an existing trace)
- Scoring (compute log-probability of a fixed trace)

## Required Methods

### Sampling Methods (Type-Specific)

- [`on_sample_f64`](Self::on_sample_f64): Handle f64 sampling (continuous distributions)
- [`on_sample_bool`](Self::on_sample_bool): Handle bool sampling (Bernoulli)
- [`on_sample_u64`](Self::on_sample_u64): Handle u64 sampling (Poisson, Binomial)
- [`on_sample_usize`](Self::on_sample_usize): Handle usize sampling (Categorical)

### Observation Methods (Type-Specific)

- [`on_observe_f64`](Self::on_observe_f64): Handle f64 observations
- [`on_observe_bool`](Self::on_observe_bool): Handle bool observations
- [`on_observe_u64`](Self::on_observe_u64): Handle u64 observations
- [`on_observe_usize`](Self::on_observe_usize): Handle usize observations

### Other Methods

- [`on_factor`](Self::on_factor): Handle arbitrary log-weight contributions
- [`finish`](Self::finish): Finalize and return the accumulated trace

## Examples

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
// Use a built-in handler
let mut rng = StdRng::seed_from_u64(42);
let handler = PriorHandler {
    rng: &mut rng,
    trace: Trace::default(),
};
let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
let (result, trace) = runtime::handler::run(handler, model);
```
