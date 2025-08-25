# `factor` function

Add an unnormalized log-weight factor to the model.

Factors allow encoding soft constraints or arbitrary log-probability contributions to the model. They are particularly useful for:

- Encoding constraints that should be "mostly satisfied"
- Adding custom log-likelihood terms
- Implementing rejection sampling (using negative infinity)

## Arguments

- `logw` - Log-weight to add to the model's total weight

## Returns

A `Model<()>` that contributes the given log-weight.

## Examples

```rust
use fugue::*;
// Add positive log-weight (increases probability)
let model = factor(1.0); // Adds log(e) = 1.0 to weight
// Add negative log-weight (decreases probability)
let model = factor(-2.0); // Subtracts 2.0 from log-weight
// Reject/fail (zero probability)
let model = factor(f64::NEG_INFINITY);
// Soft constraint: prefer values near zero
let x = 5.0;
let soft_constraint = factor(-0.5 * x * x); // Gaussian-like penalty
```
