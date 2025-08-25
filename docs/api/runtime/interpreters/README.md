# `interpreters` module

Built-in interpreters for different model execution modes.

This module provides three fundamental handlers that form the building blocks for inference algorithms:

- [`PriorHandler`]: Samples fresh values from prior distributions
- [`ReplayHandler`]: Replays from an existing trace with fallback sampling
- [`ScoreGivenTrace`]: Computes log-probability of a fixed trace

These handlers accumulate execution traces while interpreting models and are the foundation for more complex inference algorithms like MCMC, SMC, and ABC.

## Usage Patterns

### Prior Sampling

Use `PriorHandler` to generate samples from the model's prior distribution:

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
let mut rng = StdRng::seed_from_u64(42);
let (value, trace) = runtime::handler::run(
    PriorHandler { rng: &mut rng, trace: Trace::default() },
    model
);
```

### Trace Replay

Use `ReplayHandler` to replay a model with values from an existing trace:

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
# let existing_trace = Trace::default(); // From previous execution
let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
let mut rng = StdRng::seed_from_u64(42);
let (value, new_trace) = runtime::handler::run(
    ReplayHandler {
        rng: &mut rng,
        base: existing_trace,
        trace: Trace::default()
    },
    model
);
```

### Scoring

Use `ScoreGivenTrace` to compute the log-probability of a model given fixed choices:

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
// Create a trace with choices first
let model_fn = || sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
let mut rng = StdRng::seed_from_u64(42);
let (_, existing_trace) = runtime::handler::run(
    PriorHandler { rng: &mut rng, trace: Trace::default() },
    model_fn()
);
// Now score the same model with the trace
let (value, score_trace) = runtime::handler::run(
    ScoreGivenTrace {
        base: existing_trace,
        trace: Trace::default()
    },
    model_fn()
);
assert!(score_trace.total_log_weight().is_finite());
```
