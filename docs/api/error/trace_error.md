# `trace_error` macro

Create a TraceError with optional context.

## Examples

```rust
# use fugue::*;
let err = trace_error!("get_f64", Some(addr!("mu")), "address not found", TraceAddressNotFound);
```
