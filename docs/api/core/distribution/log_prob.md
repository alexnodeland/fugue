# `log_prob` function

Compute the log-probability density/mass of a value under this distribution.

Accepts a reference to a value of type `T` to avoid unnecessary copying and to maintain consistency across all distribution types.

## Arguments

- `x` - Reference to the value to compute log-probability for

## Returns

The natural logarithm of the probability density/mass at `x`. Returns negative infinity for values outside the distribution's support.

## Examples

```rust
use fugue::core::distribution::Normal;
```
