# `bind` function

Monadic bind operation (>>=).

Chains two probabilistic computations where the second depends on the result of the first. This is the fundamental operation for building complex probabilistic models from simpler parts.

## Arguments

* `k` - Function that takes the result of this model and returns a new model

## Examples

```rust
use fugue::*;
// Dependent sampling: y depends on x
let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
    .bind(|x| sample(addr!("y"), Normal::new(x, 0.1).unwrap()));
```
