# `ScoreGivenTrace` struct

Handler for scoring a model given a complete trace of choices.

This handler computes the log-probability of a model execution where all random choices are fixed by an existing trace. It does not perform any sampling - instead, it looks up values from the base trace and computes their log-probabilities under the current model's distributions. This is essential for:

- Computing proposal densities in MCMC
- Importance weighting in particle filters
- Model comparison and Bayes factors

## Fields

- `base` - Trace containing the fixed choices to score
- `trace` - New trace to accumulate log-probabilities

## Panics

Panics if the base trace is missing a value for any sampling site encountered during execution. The base trace must be complete for the model being scored.

## Examples

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
// Create a trace with some choices
let model_fn = || sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
let mut rng = StdRng::seed_from_u64(123);
let (_, complete_trace) = runtime::handler::run(
    PriorHandler { rng: &mut rng, trace: Trace::default() },
    model_fn()
);
// Score the model under different parameters
let different_model_fn = || sample(addr!("x"), Normal::new(1.0, 2.0).unwrap());
let (value, score_trace) = runtime::handler::run(
    ScoreGivenTrace {
        base: complete_trace,
        trace: Trace::default(),
    },
    different_model_fn(),
);
assert!(score_trace.total_log_weight().is_finite());
```
