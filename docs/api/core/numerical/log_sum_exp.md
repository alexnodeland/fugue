# `log_sum_exp` function

Compute log(sum(exp(x_i))) in a numerically stable way.

This is essential for normalizing log-probabilities without underflow. The standard trick is to factor out the maximum value to prevent overflow.

## Arguments

- `log_values` - Slice of log-values to sum

## Returns

log(Σᵢ exp(xᵢ)) computed stably, or -∞ if all inputs are -∞

## Examples

```rust
use fugue::core::numerical::log_sum_exp;

let log_vals = vec![-1.0, -2.0, -3.0];
let result = log_sum_exp(&log_vals);
// log_sum_exp([-1, -2, -3]) ≈ log(e^(-1) + e^(-2) + e^(-3)) ≈ -0.591
assert!((result - (-0.5914)).abs() < 0.01);
```
