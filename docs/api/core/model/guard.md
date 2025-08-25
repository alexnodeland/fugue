# `guard` function

Conditional execution: fail with zero probability when predicate is false. Guards provide a way to enforce hard constraints in probabilistic models. When the predicate is true, the model continues normally. When false, the model receives negative infinite log-weight, effectively ruling out that execution path.

## Arguments

- `pred` - Boolean predicate to check

## Returns

A `Model<()>` that either succeeds (pred=true) or fails with zero probability (pred=false).

## Examples

```rust
use fugue::*;
// Ensure a sampled value is positive
let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
    .bind(|x| {
        guard(x > 0.0).bind(move |_| pure(x))
    });
// Multiple constraints
let model = sample(addr!("x"), Uniform::new(-2.0, 2.0).unwrap())
    .bind(|x| {
        guard(x > -1.0).bind(move |_|
            guard(x < 1.0).bind(move |_| pure(x * x))
        )
    });
```
