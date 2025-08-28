# Building Complex Models

Fugue's macro system and compositional design enables building sophisticated probabilistic models with clean, readable syntax. This guide demonstrates practical techniques for model composition, from basic macros to complex hierarchical structures.

## Do-Notation with `prob!`

The `prob!` macro provides Haskell-style do-notation for chaining probabilistic computations:

```rust,ignore
{{#include ../../../examples/building_complex_models.rs:basic_prob_macro}}
```

**Key Features:**
- `<-` for probabilistic binding (monadic bind)
- `=` for regular variable assignment  
- `pure()` to lift deterministic values
- Natural control flow without callback nesting

```admonish tip
Use `prob!` when you need to chain multiple probabilistic operations. It's especially powerful for dependent sampling where later variables depend on earlier ones.
```

## Vectorized Operations with `plate!`

Plate notation handles independent replications efficiently:

```rust,ignore
{{#include ../../../examples/building_complex_models.rs:plate_notation_basic}}
```

**Benefits:**
- Automatic address indexing prevents conflicts
- Natural iteration over data structures
- Vectorized likelihood computations
- Clear intent for independent operations

```admonish note
The `plate!` macro automatically appends indices to addresses, so `addr!("sample", i)` becomes unique for each iteration without manual address management.
```

## Hierarchical Address Management

Complex models need systematic parameter organization:

```rust,ignore
{{#include ../../../examples/building_complex_models.rs:hierarchical_scoping}}
```

**Address Strategy:**
- `scoped_addr!` prevents parameter name collisions
- Hierarchical structure mirrors model dependencies
- Systematic naming aids debugging and introspection
- Indices enable parameter arrays

## Sequential Dependencies

For time series and state-dependent models:

```rust,ignore
{{#include ../../../examples/building_complex_models.rs:sequential_dependencies}}
```

**Patterns:**
- Explicit state threading through computations
- Observation conditioning at each time step
- Autoregressive dependencies
- Mixed probabilistic and deterministic updates

```admonish warning
Sequential models can create large traces. Consider using memory-efficient handlers for long sequences.
```

## Composable Model Functions

Build reusable model components:

```rust,ignore
{{#include ../../../examples/building_complex_models.rs:model_composition}}
```

**Design Principles:**
- Functions return `Model<T>` for composability
- Pattern matching enables model selection
- Pure functions for deterministic transformations
- Higher-order functions for model templates

## Advanced Address Patterns

For large-scale models like neural networks:

```rust,ignore
{{#include ../../../examples/building_complex_models.rs:address_management}}
```

**Scaling Strategies:**
- Systematic parameter naming conventions
- Multi-level scoping for complex architectures
- Consistent indexing schemes
- Hierarchical parameter organization

## Mixing Styles for Flexibility

Combine macros with traditional function composition:

```rust,ignore
{{#include ../../../examples/building_complex_models.rs:mixture_models}}
```

**Best Practices:**
- Use functions for reusable components
- Use macros for readable composition
- Separate concerns (priors, likelihood, observations)
- Document parameter dependencies

## Real-World Applications

### Bayesian Linear Regression

Complete end-to-end modeling workflow:

```rust,ignore
{{#include ../../../examples/building_complex_models.rs:bayesian_regression}}
```

### Hierarchical Clustering

Multi-level parameter structures:

```rust,ignore
{{#include ../../../examples/building_complex_models.rs:multilevel_hierarchy}}
```

### State Space Models

Sequential latent variable models:

```rust,ignore
{{#include ../../../examples/building_complex_models.rs:sequential_dependencies}}
```

## Multi-Level Hierarchies

Population → Groups → Individuals structure:

```rust,ignore
{{#include ../../../examples/building_complex_models.rs:multilevel_hierarchy}}
```

**Key Features:**
- Partial pooling across hierarchy levels
- Systematic parameter organization
- Natural shrinkage properties
- Scalable to large group structures

## Configurable Model Factories

Dynamic model construction:

```rust,ignore
{{#include ../../../examples/building_complex_models.rs:model_composition}}
```

**Flexibility Benefits:**
- Runtime model configuration
- Conditional model components
- A/B testing different model structures
- Experiment management

## Testing Complex Models

Always validate model construction:

```rust,ignore
{{#include ../../../examples/building_complex_models.rs:composition_testing}}
```

## Common Pitfalls

1. **Address Conflicts**: Use `scoped_addr!` for complex models
2. **Memory Usage**: Large plate operations can create big traces
3. **Sequential Dependencies**: Explicit state management required
4. **Type Inference**: Sometimes need explicit type annotations

## Performance Considerations

- **Plate Size**: Very large plates may exceed memory limits
- **Nesting Depth**: Deep hierarchies increase trace size
- **Address Complexity**: Simple addresses are more efficient
- **Function Composition**: Pure functions are optimized away

## Next Steps

- **Optimization**: See [Optimizing Performance](./optimizing-performance.md) for efficiency techniques
- **Debugging**: Check [Debugging Models](./debugging-models.md) for troubleshooting complex models
- **Production**: Learn [Production Deployment](./production-deployment.md) for scaling

Complex models become manageable through systematic composition, clear addressing, and thoughtful abstraction. The macro system provides the syntactic sugar while maintaining the underlying monadic structure that makes inference algorithms possible.