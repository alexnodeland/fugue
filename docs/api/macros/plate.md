# `plate` macro

Plate notation for replicating models over ranges.

## Examples

```rust
use fugue::*;

let model = plate!(i in 0..10 => {
    sample(addr!("x", i), Normal::new(0.0, 1.0).unwrap())
});
```
