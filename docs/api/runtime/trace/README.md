# `trace` module

Execution traces capturing choices and accumulated log-weights.

This module provides data structures for recording the execution history of probabilistic models. Traces capture:

- **Choices**: Named random variable assignments with their log-probabilities
- **Log-weights**: Accumulated prior, likelihood, and factor contributions

Traces enable key capabilities in probabilistic programming:

- **Replay**: Re-executing models with the same random choices
- **Conditioning**: Computing model probabilities given fixed data
- **Inference**: Tracking and updating random variable assignments
- **Debugging**: Understanding model execution flow and weights

## Structure

A trace consists of:

- A map of choices keyed by address
- Separate accumulators for prior, likelihood, and factor log-weights

The total log-weight combines all three components and represents the unnormalized log-probability of the execution.

## Examples

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
// Execute a model and examine its trace
let model = sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap())
    .bind(|mu| observe(addr!("y"), Normal::new(mu, 0.5).unwrap(), 2.0));
let mut rng = StdRng::seed_from_u64(42);
let (_, trace) = runtime::handler::run(
    PriorHandler { rng: &mut rng, trace: Trace::default() },
    model,
);
println!("Prior log-weight: {}", trace.log_prior);
println!("Likelihood log-weight: {}", trace.log_likelihood);
println!("Total log-weight: {}", trace.total_log_weight());
// Access specific choices
if let Some(choice) = trace.choices.get(&addr!("mu")) {
    println!("Sampled mu: {:?}", choice.value);
}
```
