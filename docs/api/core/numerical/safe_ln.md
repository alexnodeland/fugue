# `safe_ln` function

Safe logarithm that handles edge cases gracefully.

Returns -∞ for non-positive inputs instead of NaN or panicking.

## Arguments

- `x` - Input value

## Returns

- `ln(x)` if `x` is positive, otherwise `-∞`
