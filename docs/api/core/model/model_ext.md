# `ModelExt` trait

Monadic operations for composing and transforming models.

This trait provides the fundamental monadic operations that enable compositional probabilistic programming. All models implement this trait, allowing them to be chained and transformed in a principled way.

## Core Operations

- [`bind`](Self::bind): Monadic bind (>>=) - chains dependent computations
- [`map`](Self::map): Functor map - transforms the result without adding probabilistic behavior
- [`and_then`](Self::and_then): Alias for `bind` for those familiar with Rust's `Option`/`Result`

## Examples

```rust
use fugue::*;
// Using bind for dependent sampling
let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
    .bind(|x| sample(addr!("y"), Normal::new(x, 0.5).unwrap()));
// Using map for transformations
let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
    .map(|x| x * 2.0 + 1.0);
// Chaining multiple operations
let model = sample(addr!("x"), Uniform::new(0.0, 1.0).unwrap())
    .bind(|x| {
        if x > 0.5 {
            sample(addr!("high"), Normal::new(10.0, 1.0).unwrap())
        } else {
            sample(addr!("low"), Normal::new(-10.0, 1.0).unwrap())
        }
    })
    .map(|result| result.abs());
```
