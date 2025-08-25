# `prob` macro

Probabilistic programming macro.

This macro is used to define probabilistic programs with do-notation.

## Examples

```rust
use fugue::*;

let model = prob! {
    let x <- Normal::new(0.0, 1.0).unwrap();
    let y <- Normal::new(x, 1.0).unwrap();
    y
};
```
