# `addr!` macro

Create an address for naming random variables and observation sites. This macro provides a convenient way to create `Address` instances with human-readable names and optional indices. The macro supports two forms:

- `addr!("name")` - Simple named address
- `addr!("name", index)` - Indexed address using "name#index" format

## Examples

```rust
use fugue::*;
// Simple addresses
let mu = addr!("mu");
let sigma = addr!("sigma");
// Indexed addresses for collections
let data_0 = addr!("data", 0);
let data_1 = addr!("data", 1);
// Use in models
let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
    .bind(|x| {
        // Index can be dynamic
        let i = 42;
        sample(addr!("y", i), Normal::new(x, 0.1).unwrap())
    });
```
