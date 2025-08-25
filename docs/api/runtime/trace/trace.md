# `Trace` struct

Complete execution trace of a probabilistic model.

A trace records the full execution history of a probabilistic model, including all random choices made and the accumulated log-weights from different sources. Traces are essential for:

- **Replay**: Re-executing models with the same random choices
- **Scoring**: Computing log-probabilities of specific executions
- **Inference**: Updating random variables while keeping others fixed
- **Debugging**: Understanding model behavior and weight contributions

## Log-weight Components

The total log-weight is decomposed into three components:

- **Prior**: Log-probabilities of sampled values under their prior distributions
- **Likelihood**: Log-probabilities of observed data given the model
- **Factors**: Additional log-weight contributions from factor statements

## Fields

- `choices` - Map from addresses to the choices made at those sites
- `log_prior` - Accumulated log-prior probability
- `log_likelihood` - Accumulated log-likelihood of observations
- `log_factors` - Accumulated log-weight from factor statements

## Examples

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
// Create a model with different weight sources
let model = sample(addr!("theta"), Normal::new(0.0, 1.0).unwrap())
    .bind(|theta| {
        observe(addr!("y"), Normal::new(theta, 0.5).unwrap(), 1.5)
            .bind(move |_| factor(-0.1).bind(move |_| pure(theta)))
    });
let mut rng = StdRng::seed_from_u64(42);
let (theta, trace) = runtime::handler::run(
    PriorHandler { rng: &mut rng, trace: Trace::default() },
    model,
);
println!("Sampled theta: {}", theta);
println!("Prior contribution: {}", trace.log_prior);
println!("Likelihood contribution: {}", trace.log_likelihood);
println!("Factor contribution: {}", trace.log_factors);
println!("Total log-weight: {}", trace.total_log_weight());
// Access individual choices
if let Some(choice) = trace.choices.get(&addr!("theta")) {
    println!("Theta choice: {:?}", choice.value);
}
```
