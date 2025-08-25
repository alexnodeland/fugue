# `map` function

Apply a function to transform the result of this model.

This is the functor map operation - it transforms the output of a model without adding any additional probabilistic behavior.

## Arguments

* `f` - Function to apply to the model's result

## Examples

```rust
use fugue::*;
// Transform the sampled value
let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
    .map(|x| x.exp()); // Apply exponential function
```
