# `zip` function

Combine two independent models into a model of their paired results.

This operation runs both models and combines their results into a tuple. The models are executed independently (neither depends on the other's result).

## Arguments

- `ma` - First model to execute
- `mb` - Second model to execute

## Returns

A `Model<(A, B)>` containing the paired results.

## Examples

```rust
use fugue::*;
// Sample two independent random variables
let x_model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
let y_model = sample(addr!("y"), Uniform::new(0.0, 1.0).unwrap());
let paired = zip(x_model, y_model); // Model<(f64, f64)>
// Can be used with any model types
let mixed = zip(pure(42.0), sample(addr!("z"), Exponential::new(1.0).unwrap()));
```
