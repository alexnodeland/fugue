# Debugging Models

Debugging probabilistic models requires specialized techniques since traditional debugging tools don't reveal the stochastic behavior. Fugue provides comprehensive debugging capabilities through trace inspection, diagnostics, and validation tools.

## Trace Inspection and Analysis

Every model execution produces a complete trace that records all choices and log-weights. This is your primary debugging tool:

```rust,ignore
{{#include ../../../examples/debugging_models.rs:trace_inspection}}
```

**Key Debugging Insights:**

- **Choice count** reveals model complexity and structure
- **Log-weight decomposition** identifies prior vs. likelihood vs. factor issues
- **Per-choice analysis** shows individual parameter contributions
- **Finite log-weights** indicate valid model execution

## Type-Safe Value Access

Fugue provides robust access patterns that handle type mismatches gracefully:

```rust,ignore
{{#include ../../../examples/debugging_models.rs:type_safe_access}}
```

**Error Handling Strategies:**

- Use `get_*_result()` for detailed error information
- Use `get_*()` for simple None-handling
- Always check for missing addresses before assuming success
- Iterate through all choices to understand model structure

## Model Validation and Testing

Systematic validation ensures your model behaves as expected:

```rust,ignore
{{#include ../../../examples/debugging_models.rs:model_validation}}
```

**Validation Best Practices:**

- Test against known analytical solutions
- Verify all traces have finite log-weights
- Check basic statistical properties (means, variances)
- Test edge cases and boundary conditions

## Safe vs Strict Error Handling

Fugue provides both strict (fail-fast) and safe (error-resilient) execution modes:

```rust,ignore
{{#include ../../../examples/debugging_models.rs:safe_handlers}}
```

**When to Use Each:**

- **Strict handlers** (`ReplayHandler`, `ScoreGivenTrace`): Development and testing
- **Safe handlers** (`SafeReplayHandler`, `SafeScoreGivenTrace`): Production systems
- Safe handlers log warnings instead of panicking on mismatches

## MCMC Diagnostics

For inference algorithms, convergence diagnostics are essential:

```rust,ignore
{{#include ../../../examples/debugging_models.rs:mcmc_diagnostics}}
```

**Convergence Indicators:**

- **R-hat < 1.1**: Chains have converged
- **High ESS**: Efficient sampling without excessive correlation
- **Multiple chains**: Essential for reliable convergence assessment
- **Visual inspection**: Always examine trace plots when possible

## Model Structure Analysis

Complex models benefit from systematic structure analysis:

```rust,ignore
{{#include ../../../examples/debugging_models.rs:model_structure_debugging}}
```

**Structure Analysis Benefits:**

- Understand parameter organization and hierarchies
- Detect unexpected address patterns
- Verify choice counts match model expectations
- Identify bottlenecks in complex models

## Performance Diagnostics

Monitor computational efficiency and identify bottlenecks:

```rust,ignore
{{#include ../../../examples/debugging_models.rs:performance_diagnostics}}
```

**Performance Warning Signs:**

- Zero choices recorded (model execution failure)
- Infinite log-weights (constraint violations)
- Excessive execution time (optimization needed)
- Large memory footprint (consider streaming approaches)

## Common Debugging Patterns

Systematic debugging approaches for robust model development:

```rust,ignore
{{#include ../../../examples/debugging_models.rs:debugging_patterns}}
```

**Debugging Workflow:**

1. **Start Simple**: Test individual components before complex composition
2. **Validate Incrementally**: Add complexity one piece at a time
3. **Check Address Uniqueness**: Prevent parameter collision bugs
4. **Monitor Log-Weights**: Track prior, likelihood, and factor contributions
5. **Use Systematic Testing**: Automated validation for all model components

## Testing Framework Integration

Embed debugging checks in your test suite:

```rust,ignore
{{#include ../../../examples/debugging_models.rs:debugging_tests}}
```

**Testing Strategy:**

- Unit tests for individual model components
- Integration tests for complete workflows
- Performance regression tests
- Statistical validation against known results

## Common Issues and Solutions

### Issue: Infinite Log-Weights

**Symptoms:** `trace.total_log_weight().is_infinite()`

**Causes:**

- Factor statements with impossible constraints
- Parameters outside valid ranges
- Numerical overflow in likelihood computations

**Solutions:**

- Check factor conditions carefully
- Validate parameter ranges in constructors
- Use log-space computations for numerical stability

### Issue: Missing or Wrong Parameter Values

**Symptoms:** `get_*()` returns `None` or wrong types

**Causes:**

- Address typos or inconsistencies
- Model structure doesn't match expectations
- Type mismatches in trace replay

**Solutions:**

- Use consistent address naming conventions
- Print all addresses for verification
- Use safe handlers for production resilience

### Issue: Poor MCMC Convergence

**Symptoms:** High R-hat values, low ESS

**Causes:**

- Inappropriate step sizes
- Poor model parameterization
- Insufficient warm-up periods

**Solutions:**

- Increase warm-up iterations
- Reparameterize for better geometry
- Use adaptive algorithms with proper tuning

### Issue: Slow Model Execution

**Symptoms:** High execution times, memory usage

**Causes:**

- Inefficient model structure
- Excessive address creation
- Large trace construction overhead

**Solutions:**

- Use `plate!` for vectorized operations
- Pre-allocate data structures when possible
- Profile with performance diagnostics

## Best Practices Summary

1. **Debug Incrementally**: Start simple and add complexity systematically
2. **Use All Tools**: Combine trace inspection, validation, and diagnostics
3. **Test Edge Cases**: Verify behavior at parameter boundaries
4. **Monitor Performance**: Track execution time and memory usage
5. **Validate Statistically**: Compare against known theoretical results
6. **Handle Errors Gracefully**: Use safe handlers in production
7. **Document Assumptions**: Clear model specifications aid debugging

Effective debugging transforms probabilistic programming from guesswork into systematic model development. Fugue's comprehensive debugging toolkit enables confident deployment of complex probabilistic systems.
