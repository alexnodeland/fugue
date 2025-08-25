# `sample` function

Generate a random sample from this distribution.

Returns a value of type `T` drawn from this distribution. The return type is naturally suited to the distribution:

- `f64` for continuous distributions (Normal, Beta, etc.)
- `bool` for Bernoulli (true/false outcomes)
- `u64` for count distributions (Poisson, Binomial)
- `usize` for categorical indices (safe array indexing)

## Arguments

- `rng` - Random number generator to use for sampling

## Returns

A sample from the distribution of type `T`.

## Examples

```rust
use fugue::core::distribution::Normal;
```
