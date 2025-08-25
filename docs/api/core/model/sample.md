# `sample` function

Sample from a distribution (generic version - chooses the right variant automatically). This is the main sampling function that works with any distribution type. The return type is inferred from the distribution type.

## Type-specific variants

For explicit type control, you can also use the type-specific sampling functions:

- `sample_f64` - Sample from f64 distributions (continuous distributions)
- `sample_bool` - Sample from bool distributions (Bernoulli)
- `sample_u64` - Sample from u64 distributions (Poisson, Binomial)
- `sample_usize` - Sample from usize distributions (Categorical)

These functions provide the same functionality but with explicit type annotations, which can be useful when the compiler cannot infer the type or when you want to be explicit about the expected return type.

## Examples

```rust
use fugue::*;
// Automatically returns f64 for continuous distributions
let normal_sample: Model<f64> = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
// Automatically returns bool for Bernoulli
let coin_flip: Model<bool> = sample(addr!("coin"), Bernoulli::new(0.5).unwrap());
// Automatically returns u64 for Poisson
let count: Model<u64> = sample(addr!("count"), Poisson::new(3.0).unwrap());
// Automatically returns usize for Categorical
let choice: Model<usize> = sample(addr!("choice"),
    Categorical::new(vec![0.3, 0.5, 0.2]).unwrap());
```
