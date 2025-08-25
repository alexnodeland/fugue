# `ChoiceValue` enum

Value stored at a choice site in an execution trace.

Different types of random variables can be stored in traces, though currently only `f64` values are used by the built-in distributions. Additional variants support future extensions to other value types.

## Variants

- `F64` - Floating-point values (most common)
- `I64` - Integer values
- `Bool` - Boolean values

## Examples

```rust
use fugue::*;
// Most distributions use F64 values
let normal_value = ChoiceValue::F64(1.23);
let discrete_value = ChoiceValue::F64(3.0); // Categorical/Poisson as f64
// Future extensions might use other types
let integer_value = ChoiceValue::I64(42);
let boolean_value = ChoiceValue::Bool(true);
```
