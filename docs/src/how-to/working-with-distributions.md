# Working with Distributions

Fugue's type-safe distribution system is designed to eliminate common runtime errors while maintaining statistical expressiveness. This guide shows you practical techniques for working with distributions effectively.

## Type Safety in Practice

Traditional probabilistic programming libraries return `f64` for everything, leading to casting overhead and runtime errors. Fugue distributions return their natural types:

```rust,ignore
{{#include ../../../examples/working_with_distributions.rs:type_safety_demo}}
```

No casting, no comparisons with floating-point values—just natural boolean logic.

## Continuous Distributions

For modeling continuous phenomena like measurements, temperatures, or errors:

```rust,ignore
{{#include ../../../examples/working_with_distributions.rs:continuous_distributions}}
```

**Key Points:**

- `sample()` returns `f64` for direct arithmetic
- `log_prob()` computes log-density (avoids numerical underflow)
- Parameter validation happens at construction time

```admonish tip
Always work with log-probabilities for numerical stability. Only convert to regular probabilities when necessary for interpretation.
```

## Discrete Distributions

Count data and categorical outcomes use natural integer types:

```rust,ignore
{{#include ../../../examples/working_with_distributions.rs:discrete_distributions}}
```

**Benefits:**

- `u64` counts support direct arithmetic without casting
- No precision loss from floating-point representations
- Natural integration with Rust's type system

## Safe Categorical Sampling

Categorical distributions return `usize` for safe array indexing:

```rust,ignore
{{#include ../../../examples/working_with_distributions.rs:categorical_usage}}
```

```admonish note
The `usize` return type eliminates bounds checking errors—the sampled index is guaranteed to be valid for the probability vector length.
```

## Parameter Validation

All distributions validate parameters at construction:

```rust,ignore
{{#include ../../../examples/working_with_distributions.rs:parameter_validation}}
```

**Validation Rules:**

- Normal: σ > 0
- Beta: α > 0, β > 0  
- Poisson: λ ≥ 0
- Categorical: probabilities sum to 1, all non-negative

## Storing Mixed Distributions

Use trait objects for collections of distributions with the same return type:

```rust,ignore
{{#include ../../../examples/working_with_distributions.rs:distribution_composition}}
```

This enables dynamic distribution selection and model composition patterns.

## Practical Modeling Patterns

Common modeling scenarios demonstrate natural type usage:

```rust,ignore
{{#include ../../../examples/working_with_distributions.rs:practical_modeling}}
```

Each distribution serves its natural domain without artificial conversions.

## Working with Log-Probabilities

For numerical stability, always accumulate log-probabilities:

```rust,ignore
{{#include ../../../examples/working_with_distributions.rs:probability_calculations}}
```

```admonish warning
Converting large negative log-probabilities back to regular probabilities can underflow to zero. Keep computations in log-space when possible.
```

## Advanced Patterns

For complex modeling scenarios, see these patterns:

### Hierarchical Models

```rust,ignore
{{#include ../../../examples/advanced_distribution_patterns.rs:hierarchical_priors}}
```

### Mixture Components

```rust,ignore
{{#include ../../../examples/advanced_distribution_patterns.rs:mixture_components}}
```

### Conjugate Priors

```rust,ignore
{{#include ../../../examples/advanced_distribution_patterns.rs:conjugate_pairs}}
```

## Testing Your Distributions

Always test distribution properties and parameter validation:

```rust,ignore
{{#include ../../../examples/working_with_distributions.rs:distribution_testing}}
```

## Common Pitfalls

1. **Underflow in probability space**: Use log-probabilities for accumulation
2. **Parameter validation**: Check constructor errors, don't assume success
3. **Precision with counts**: Use `u64` return types directly, avoid `f64` conversion
4. **Categorical indexing**: Trust the `usize` return—it's guaranteed valid

## Next Steps

- **Complex Models**: See [Building Complex Models](./building-complex-models.md) for compositional patterns
- **Debugging**: Check out [Debugging Models](./debugging-models.md) for troubleshooting
- **Custom Logic**: Learn [Custom Handlers](./custom-handlers.md) for specialized inference

The type-safe distribution system eliminates entire classes of runtime errors while making statistical code more readable and maintainable.
