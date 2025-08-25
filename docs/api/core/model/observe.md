# `observe` function

Observe a value from a distribution (generic version). This function automatically chooses the right observation variant based on the distribution type and observed value type.

## Examples

```rust
use fugue::*;
// Observe f64 value from continuous distribution
let model = observe(addr!("y"), Normal::new(1.0, 0.5).unwrap(), 2.5);
// Observe bool value from Bernoulli
let model = observe(addr!("coin"), Bernoulli::new(0.6).unwrap(), true);
// Observe u64 count from Poisson
let model = observe(addr!("count"), Poisson::new(3.0).unwrap(), 5u64);
// Observe usize choice from Categorical
let model = observe(addr!("choice"),
    Categorical::new(vec![0.3, 0.5, 0.2]).unwrap(), 1usize);
```
