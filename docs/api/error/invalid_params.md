# `invalid_params` macro

Create an InvalidParameters error with optional context.

## Examples

```rust
# use fugue::*;
let err = invalid_params!("Normal", "sigma must be positive", InvalidVariance);
let err_with_ctx = invalid_params!("Normal", "sigma must be positive", InvalidVariance, 
    "sigma" => "-1.0", "expected" => "> 0.0");
```
