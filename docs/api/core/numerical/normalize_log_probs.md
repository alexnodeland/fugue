# `normalize_log_probs` function

Normalize log-probabilities to linear probabilities stably.

Converts a vector of log-probabilities to normalized linear probabilities without underflow or overflow issues.

## Arguments

- `log_probs` - Log-probabilities to normalize

## Returns

Vector of normalized linear probabilities that sum to 1.0

## Examples

```rust
use fugue::core::numerical::normalize_log_probs;

let log_probs = vec![-1.0, -2.0, -3.0];
```
