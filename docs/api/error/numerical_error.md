# `numerical_error` macro

Create a NumericalError with optional context.

## Examples

```rust
# use fugue::*;
let err = numerical_error!("log", "input was negative", NumericalInstability);
let err_with_ctx = numerical_error!("log", "input was negative", NumericalInstability,
    "input" => "-1.5");
```
