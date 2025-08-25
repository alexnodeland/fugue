# `traverse_vec` function

Apply a function that produces models to each item in a vector, collecting the results. This is a higher-order function that maps each item in the input vector through a function that produces a model, then sequences all the resulting models into a single model of a vector. This is equivalent to `sequence_vec(items.map(f))` but more convenient.

## Arguments

- `items` - Vector of input items to process
- `f` - Function that takes an item and produces a model

## Returns

A `Model<Vec<A>>` containing all the results in order.

## Examples

```rust
use fugue::*;
// Add noise to each data point
let data = vec![1.0, 2.0, 3.0];
let noisy_data = traverse_vec(data, |x| {
    sample(addr!("noise", x as usize), Normal::new(0.0, 0.1).unwrap())
        .map(move |noise| x + noise)
});
// Create observations for each data point
let observations = vec![1.2, 2.1, 2.9];
let model = traverse_vec(observations, |obs| {
    observe(addr!("y", obs as usize), Normal::new(2.0, 0.5).unwrap(), obs)
});
```
