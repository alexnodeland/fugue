# `weighted_log_sum_exp` function

Compute log(sum(w_i \* exp(x_i))) stably for weighted log-sum-exp.

This generalizes log_sum_exp to handle weighted sums, commonly needed in importance sampling and particle filtering.

## Arguments

- `log_values` - Log-values to sum
- `weights` - Linear weights (not log-weights)

## Returns

log(Σᵢ wᵢ exp(xᵢ)) computed stably

## Examples

```rust
use fugue::core::numerical::weighted_log_sum_exp;

let log_values = vec![-1.0, -2.0, -3.0];
```
