# `Model` type

Core model type representing probabilistic computations.

A `Model<A>` represents a probabilistic program that yields a value of type `A` when executed.
Models are built using four fundamental operations:

- `Pure(a)`: Deterministic computation returning value `a`
- `SampleF`: Sample from a probability distribution at a named address
- `ObserveF`: Condition on observed data at a named address
- `FactorF`: Add a log-weight factor (for soft constraints)
  Models form a monad, allowing compositional construction using `bind`, `map`, and related operations.

## Examples

```rust
use fugue::*;
// Simple deterministic model
let model = pure(42.0);
// Probabilistic model with sampling
let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
// Composed model
let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
    .bind(|x| pure(x * 2.0));
```
