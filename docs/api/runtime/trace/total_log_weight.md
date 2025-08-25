# `Trace::total_log_weight` method

Compute the total unnormalized log-probability of this execution.

The total log-weight combines all three components (prior, likelihood, factors) and represents the unnormalized log-probability of this particular execution path through the model.

## Returns

The sum of log_prior + log_likelihood + log_factors.

## Examples

```rust
use fugue::*;

let trace = Trace {
    log_prior: -1.5,
    log_likelihood: -2.3,
    log_factors: 0.8,
    ..Default::default()
};

assert_eq!(trace.total_log_weight(), -3.0);
```
