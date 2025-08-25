# `PriorHandler` struct

Handler for prior sampling - generates fresh random values from distributions.

This handler implements the standard "forward sampling" interpretation of probabilistic models. When encountering sampling sites, it draws fresh random values from the specified distributions. Observations contribute their log-probabilities to the likelihood, and factors are accumulated directly. This is the most basic handler and is often used as a building block for more sophisticated inference algorithms.

## Fields

- `rng` - Random number generator for sampling
- `trace` - Trace to accumulate choices and log-weights

## Examples

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
    .bind(|x| observe(addr!("y"), Normal::new(x, 0.5).unwrap(), 1.2).map(move |_| x));
let mut rng = StdRng::seed_from_u64(123);
let (result, trace) = runtime::handler::run(
    PriorHandler {
        rng: &mut rng,
        trace: Trace::default(),
    },
    model,
);
println!("Sampled x: {}", result);
println!("Log-likelihood: {}", trace.log_likelihood);
assert!(result.is_finite());
```
