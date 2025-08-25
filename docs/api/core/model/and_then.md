# `and_then` function

Alias for `bind` - chains dependent probabilistic computations.

This method provides a more familiar interface for Rust developers used to `Option::and_then` and `Result::and_then`.

## Arguments

- `k` - Function that takes the result of this model and returns a new model
