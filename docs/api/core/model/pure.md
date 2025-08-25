# `pure` function

Lift a deterministic value into the model monad.

Creates a `Model` that always returns the given value without any probabilistic behavior.
This is the unit operation for the model monad.

## Arguments

* `a` - The value to lift into a model

## Examples

```rust
use fugue::*;

let model = pure(42.0);
// When executed, this model will always return 42.0
```
